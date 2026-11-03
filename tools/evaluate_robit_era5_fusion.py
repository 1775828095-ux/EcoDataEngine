"""Freeze all predictions before scoring hidden station values."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from time import perf_counter
import json,hashlib,sys
import numpy as np
import pandas as pd
import xarray as xr
from run_robit_era5_fusion import OUT,ROOT,PROTOCOL,VAR,ERA,load,raw_weather
from evaluate_open_complementary import recipe
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor,clean,canonical
from ecodata.algorithms.numerical_search import save
from ecodata.models.adapters.swat_plus import SWATPlusAdapter


def bench(kind):
    config=load(OUT/'cases'/kind/'config_v5.json')
    inputs=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in config['inputs']]
    context,_,_=build_context(inputs,config['request']);registry=field_registry(api.platform_registry(),context)
    return context,Executor(context,registry),config['request']


def labels(kind,phase):
    with xr.open_dataset(OUT/'cases'/kind/(phase+'_labels.nc')) as f:return f[VAR].load()


def origin(item):return {r['parameters']['source'] for r in item['records'].values() if r['operation']=='source'}


def fixed_controls():
    target=OUT/'controls_v6_frozen.json'
    if target.exists():return load(target)
    started=perf_counter();items={}
    for folder in (OUT/'runs_v6/development').iterdir():
        if not (folder/'summary.json').exists():continue
        for line in (folder/'attempts.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['status']=='ELIGIBLE_OUTPUT':
                item=recipe(folder,row['key']);items[item['identity']]=item
    benchmarks={kind:bench(kind) for kind in PROTOCOL['masks']}
    references={kind:labels(kind,'development') for kind in PROTOCOL['masks']}
    evaluations=[]
    for n,(identity,item) in enumerate(items.items()):
        scores={}
        for kind,(context,engine,_) in benchmarks.items():
            try:
                ds,_=engine.execute(item['records'],item['key'])
                scores[kind]=measured(ds,references[kind],context)['objective']
            except Exception:scores[kind]=None
        evaluations.append(dict(identity=identity,scores=scores,origins=sorted(origin(item))))
        if n%25==0:print('FIXED_REPLAY',n,'/',len(items),flush=True)
    feasible=[r for r in evaluations if all(x is not None for x in r['scores'].values())]
    rank=lambda r:(float(np.mean(list(r['scores'].values()))),sum(v['operation']!='source' for v in items[r['identity']]['records'].values()),r['identity'])
    selected={}
    for name,origins in [('fixed_global',None),('station_only',{'source0'}),('era5_calibrated',{'source1'})]:
        pool=[r for r in feasible if origins is None or set(r['origins'])==origins]
        if not pool:raise ValueError('No '+name+' baseline')
        selected[name]=items[min(pool,key=rank)['identity']]
    per_kind={kind:items[min([r for r in evaluations if r['scores'][kind] is not None],
        key=lambda r:(r['scores'][kind],r['identity']))['identity']] for kind in PROTOCOL['masks']}
    save(OUT/'control_development_evaluations_v6.json',evaluations)
    result=dict(controls=selected,fixed_per_regime=per_kind,discovered_recipe_count=len(items),
        feasible_all_regimes=len(feasible),chosen_using='Development hidden values only',test_labels_used=False,
        development_replay_seconds=perf_counter()-started,adapter_calls=sum(b[1].calls for b in benchmarks.values()))
    save(target,result);return result


def protect(ds,context):
    output=ds.copy(deep=True);raw=context.sources[0].dataset[VAR]
    output[VAR]=xr.where(np.isfinite(raw),raw,output[VAR]);output[VAR].attrs=dict(raw.attrs)
    output.attrs.update(protected_observed_source='Robit_station',fusion_method='observations_preserved_reconstruction',
        protected_visible_values=int(np.isfinite(raw).sum()))
    return output


def predict():
    controls=fixed_controls();manifest=[]
    for kind in PROTOCOL['masks']:
        context,engine,request=bench(kind)
        recipes=dict(controls['controls'],fixed_per_regime=controls['fixed_per_regime'][kind])
        recipes['era5_raw']=dict(records={'s1':dict(operation='source',parents=[],parameters={'source':'source1'})},key='s1')
        for method in ('linear','nearest','ffill','month_mean'):
            recipes['station_'+method]=dict(records={'s0':dict(operation='source',parents=[],parameters={'source':'source0'}),
                'g':dict(operation='reconstruction.daily_'+method,parents=['s0'],parameters={})},key='g')
        for seed in PROTOCOL['search_seeds']:
            for policy in ['contract_guided','random']:
                folder=OUT/'runs_v6/validation'/(kind+'_'+str(seed)+('_random' if policy=='random' else ''))
                result=load(folder/'summary.json')
                if result['best'] is None:raise ValueError('Failed selection retained, not silently replaced: '+str(folder))
                recipes[policy+'_'+str(seed)]=recipe(folder,result['best']['key'])
        for name,item in recipes.items():
            ds,_=engine.execute(item['records'],item['key']);ds=protect(ds,context)
            if not np.isfinite(ds[VAR]).all():raise ValueError('Frozen selected graph has incomplete application output: '+name)
            assert (ds[VAR]>=0).all()
            path=OUT/'predictions'/kind/(name+'.nc');path.parent.mkdir(parents=True,exist_ok=True)
            clean(ds).to_netcdf(path,engine='h5netcdf');save(path.with_suffix('.graph.json'),item)
            manifest.append(dict(case=kind,method=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('PREDICTED',kind,name,flush=True)
        alternate=context.sources[1].dataset.copy(deep=True)
        utc=pd.read_csv(ERA/'utc.csv',parse_dates=['time']).set_index('time')
        alternate[VAR].values[:,0,0]=utc.loc[pd.DatetimeIndex(alternate.time.values),VAR].to_numpy()
        alternate.attrs['day_boundary']='Sensitivity only: UTC calendar-day assumption'
        alternate=protect(alternate,context)
        path=OUT/'predictions'/kind/'era5_raw_utc.nc';clean(alternate).to_netcdf(path,engine='h5netcdf')
        manifest.append(dict(case=kind,method='era5_raw_utc',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(OUT/'predictions_frozen.json',manifest)
    print('ALL PREDICTIONS FROZEN, TEST NOT SCORED',flush=True)


def extra_metrics(prediction,reference,threshold):
    error=prediction-reference;wet=reference>=1.;predwet=prediction>=1.
    heavy=reference>=threshold
    return dict(wet_hit_rate=float(np.mean(predwet[wet])) if wet.any() else None,
        dry_false_alarm_rate=float(np.mean(predwet[~wet])) if (~wet).any() else None,
        heavy_threshold_from_training=threshold,heavy_RMSE=float(np.sqrt(np.mean(error[heavy]**2))) if heavy.any() else None,
        n=len(reference),coverage=float(np.isfinite(prediction).mean()))


def score():
    assert (OUT/'predictions_frozen.json').exists()
    weather=raw_weather();train=weather.loc[weather.time.dt.year<=1995,VAR].to_numpy()
    heavy=float(np.quantile(train[train>=1.],.9));rows=[]
    contexts={kind:bench(kind)[0] for kind in PROTOCOL['masks']}
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        reference=labels(entry['case'],'test')
        with xr.open_dataset(path) as f:ds=f.load()
        metrics=measured(ds,reference,contexts[entry['case']]);pred=ds[VAR].sel(time=reference.time).values.ravel()
        rows.append(dict(case=entry['case'],method=entry['method'],**metrics,
            **extra_metrics(pred,reference.values.ravel(),heavy)))
    names=sorted({r['method'] for r in rows});methods={}
    for name in names:
        selected=[r for r in rows if r['method']==name]
        methods[name]=dict(mean_regime_year_RMSE=float(np.mean([r['objective'] for r in selected])),
            regime_objectives={r['case']:r['objective'] for r in selected})
    adaptive=np.array([np.mean([r['window_RMSE'][str(y)] for r in rows if r['method'].startswith('contract_guided')]) for y in range(2008,2014)])
    comparisons={};rng=np.random.default_rng(20261027)
    for baseline in ['station_only','era5_raw','era5_calibrated','fixed_global','fixed_per_regime','random']:
        base=np.array([np.mean([r['window_RMSE'][str(y)] for r in rows if (r['method'].startswith('random_') if baseline=='random' else r['method']==baseline)]) for y in range(2008,2014)])
        diff=base-adaptive;ix=rng.integers(0,6,size=(10000,6))
        comparisons[baseline]=dict(relative_improvement_percent=float(100*(1-adaptive.mean()/base.mean())),
            year_wins=int((diff>1e-10).sum()),year_ties=int((abs(diff)<=1e-10).sum()),
            paired_year_differences=diff.tolist(),descriptive_bootstrap_95_interval=np.quantile(diff[ix].mean(axis=1),[.025,.975]).tolist())
    save(OUT/'test_metrics.json',rows);save(OUT/'effect_summary.json',dict(methods=methods,comparisons=comparisons,
        analysis_unit='Six years; regimes and search seeds are repeated measurements, not independent samples',
        test_read_after_prediction_freeze=True,global_optimum_proven=False))
    print(json.dumps(comparisons,ensure_ascii=False,indent=2),flush=True)


def weather_export():
    data=raw_weather();base=data[data.time.dt.year>=2008].copy();index=[]
    for kind in PROTOCOL['masks']:
        for name in ['station_only','era5_raw','fixed_global','contract_guided_11','random_11']:
            with xr.open_dataset(OUT/'predictions'/kind/(name+'.nc')) as f:ds=f.load().sel(time=pd.DatetimeIndex(base.time))
            frame=base.copy();frame[VAR]=ds[VAR].values[:,0,0]
            folder=OUT/'weather_candidates'/kind/name
            SWATPlusAdapter().save_package(folder,dict(project_name='Robit_ERA5_masked_recovery',start='2008-01-01',end='2013-12-31',
                station=dict(name='Robit',latitude=11.6,longitude=37.36,elevation=1790),weather=frame.to_dict('list'),
                target_model_version='62.0.0',additional_weather=['solar_radiation','relative_humidity','wind_speed']))
            read=pd.read_csv(folder/'station.pcp',sep=r'\s+',skiprows=3,header=None)
            np.testing.assert_allclose(read[2],frame[VAR],rtol=0,atol=.00051)
            handoff=OUT/'weather_handoff'/kind/name;handoff.mkdir(parents=True,exist_ok=True)
            for file in folder.iterdir():
                if file.name.startswith('station.') or file.name in ['pcp.cli','tmp.cli','slr.cli','hmd.cli','wnd.cli']:
                    (handoff/file.name).write_bytes(file.read_bytes())
            index.append(dict(case=kind,method=name,days=len(frame),folder=str(handoff),
                purpose='Artificially masked recovery experiment, not a replacement recommendation for complete original observations',native_SWAT_run=False))
    save(OUT/'weather_handoff/index.json',index)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='controls':print('CONTROLS',fixed_controls()['discovered_recipe_count'])
    elif stage=='predict':predict()
    elif stage=='score':score()
    elif stage=='weather':weather_export()
