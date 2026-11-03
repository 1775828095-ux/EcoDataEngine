"""Freeze supplementation predictions, then score against the unchanged cohort."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,sys,shutil
import numpy as np
import pandas as pd
import xarray as xr
from run_robit_era5_mutation import ROOT,OLD,OUT,load,freeze
from evaluate_robit_era5_fusion import bench,labels,extra_metrics,raw_weather,VAR
from ecodata.algorithms.autonomous_field import measured
from ecodata.algorithms.numerical_search import save
from ecodata.models.adapters.swat_plus import SWATPlusAdapter


def predict():
    freeze()
    target=OUT/'predictions_frozen.json'
    if target.exists():raise ValueError('Predictions already frozen; do not overwrite')
    manifest=[]
    for kind in ('random','short','long'):
        for control,prefix in [('alns','alns'),('random','random_mutation'),('none','portrait_only')]:
            for seed in (11,29,47):
                run=OUT/'runs'/(kind+'_'+str(seed)+'_'+control);s=load(run/'summary.json')
                if s['status']!='PASS':raise ValueError('Failed seed retained, not replaced: '+str(run))
                name=prefix+'_'+str(seed);path=OUT/'predictions'/kind/(name+'.nc');path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(s['product'],path)
                manifest.append(dict(case=kind,method=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for row in load(OLD/'predictions_frozen.json'):
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
        manifest.append(row)
    save(target,manifest)
    print('PREDICTIONS FROZEN BEFORE SUPPLEMENT TEST SCORING',len(manifest),flush=True)


def score():
    freeze()
    manifest=load(OUT/'predictions_frozen.json');weather=raw_weather()
    training=weather.loc[weather.time.dt.year<=1995,VAR].to_numpy();threshold=float(np.quantile(training[training>=1],.9))
    contexts={kind:bench(kind)[0] for kind in ('random','short','long')};rows=[]
    for entry in manifest:
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        ref=labels(entry['case'],'test')
        with xr.open_dataset(path) as f:ds=f.load()
        metric=measured(ds,ref,contexts[entry['case']])
        rows.append(dict(case=entry['case'],method=entry['method'],**metric,
                         **extra_metrics(ds[VAR].sel(time=ref.time).values.ravel(),ref.values.ravel(),threshold)))
    groups={name:lambda r,n=name:r['method'].startswith(n+'_') for name in ['alns','random_mutation','portrait_only','contract_guided']}
    groups['old_random']=lambda r:r['method'] in {'random_11','random_29','random_47'}
    for name in ['station_only','era5_raw','era5_calibrated','fixed_global','fixed_per_regime']:
        groups[name]=lambda r,n=name:r['method']==n
    results={};year_values={}
    for name,select in groups.items():
        subset=[r for r in rows if select(r)]
        annual=np.array([np.mean([r['window_RMSE'][str(y)] for r in subset]) for y in range(2008,2014)])
        year_values[name]=annual
        results[name]=dict(RMSE=float(annual.mean()),regimes={k:float(np.mean([r['objective'] for r in subset if r['case']==k])) for k in contexts},
             MAE=float(np.mean([r['MAE'] for r in subset])),Bias=float(np.mean([r['Bias'] for r in subset])))
    rng=np.random.default_rng(20261028);comparisons={}
    for name,base in year_values.items():
        if name=='alns':continue
        diff=base-year_values['alns'];indices=rng.integers(0,6,(10000,6))
        comparisons[name]=dict(improvement_percent=float(100*(1-year_values['alns'].mean()/base.mean())),
             annual_differences=diff.tolist(),year_wins=int((diff>1e-10).sum()),
             descriptive_bootstrap_95_interval=np.quantile(diff[indices].mean(axis=1),[.025,.975]).tolist())
    costs=[]
    for p in sorted((OUT/'runs').iterdir()):
        s=load(p/'summary.json');trace=load(p/'mutation_trace.json');attempts=[json.loads(x) for x in (p/'attempts.jsonl').read_text().splitlines()]
        xsdba=sum(t.get('operation','').startswith('bias.xsdba_') and not t.get('cache_hit',False) for a in attempts for t in a.get('trace',[]))
        costs.append(dict(run=p.name,executions=s['executed_candidates'],adapter_calls=s['adapter_invocations'],
             deployment_calls=s['deployment_adapter_calls'],seconds=s['elapsed_seconds'],mutations=s['mutation_new_executions'],
             failed=s['failed_candidates'],accepted=sum(t['accepted'] for t in trace),
             validation_initial=s['initial_best']['metrics']['objective'],validation_final=s['best']['metrics']['objective'],
             xsdba_train_adjust_calls=xsdba,empty_neighbourhoods=s['empty_neighbourhoods']))
    save(OUT/'test_metrics.json',rows);save(OUT/'effect_summary.json',dict(results=results,comparisons=comparisons,
        analysis_unit='Six test years; seeds and mask regimes are repeated measures',test_previously_inspected=True,
        global_optimum=False,source_and_masks_unchanged=True))
    save(OUT/'execution_costs.json',costs)
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


def weather():
    manifest=load(OUT/'predictions_frozen.json');base=raw_weather();base=base.loc[base.time.dt.year>=2008].copy();index=[]
    for row in manifest:
        if row['method'] not in {'alns_11','random_mutation_11','portrait_only_11'}:continue
        with xr.open_dataset(row['path']) as f:values=f[VAR].sel(time=pd.DatetimeIndex(base.time)).values.ravel()
        frame=base.copy();frame[VAR]=values;folder=OUT/'weather_handoff'/row['case']/row['method'];folder.mkdir(parents=True,exist_ok=True)
        # Reuse the official format writer; retain weather only, no model controls.
        temp=OUT/'weather_writer'/row['case']/row['method']
        SWATPlusAdapter().save_package(temp,dict(project_name='Robit_ERA5_mutation_masked_recovery',start='2008-01-01',end='2013-12-31',
             station=dict(name='Robit',latitude=11.6,longitude=37.36,elevation=1790),weather=frame.to_dict('list'),
             target_model_version='62.0.0',additional_weather=['solar_radiation','relative_humidity','wind_speed']))
        for filename in ['station.pcp','station.tem','station.slr','station.hmd','station.wnd','pcp.cli','tmp.cli','slr.cli','hmd.cli','wnd.cli']:
            source=temp/filename
            if not source.exists():raise ValueError('Writer output absent: '+filename)
            shutil.copy2(source,folder/filename)
        index.append(dict(case=row['case'],method=row['method'],folder=str(folder),days=len(frame)))
    save(OUT/'weather_handoff/index.json',index)


if __name__=='__main__':
    {'predict':predict,'score':score,'weather':weather}[sys.argv[1]]()
