"""Development-only controls, frozen public-entry predictions, then held-out test."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from time import perf_counter
import hashlib,json,sys,shutil
import numpy as np
import pandas as pd
import xarray as xr
from run_unseen_portrait_response import ROOT,OUT,DATA,STATIONS,ARMS,SEEDS,BUDGET,VAR,cases,load,freeze
from evaluate_open_complementary import recipe
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor,clean
from ecodata.algorithms.numerical_search import save
from ecodata.models.adapters.swat_plus import SWATPlusAdapter


def bench(case):
    config=load(OUT/'cases'/case/'config.json')
    inputs=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in config['inputs']]
    context,_,_=build_context(inputs,config['request'])
    return context,Executor(context,field_registry(api.platform_registry(),context)),config


def labels(case,phase):
    with xr.open_dataset(OUT/'cases'/case/(phase+'_labels.nc')) as f:return f[VAR].load()


def origins(item):return {r['parameters']['source'] for r in item['records'].values() if r['operation']=='source'}


def controls():
    freeze();target=OUT/'controls_frozen.json'
    if target.exists():raise ValueError('Controls already frozen')
    items={};started=perf_counter()
    for folder in sorted((OUT/'runs/development').iterdir()):
        for line in (folder/'attempts.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['status']=='ELIGIBLE_OUTPUT':
                item=recipe(folder,row['key']);items[item['identity']]=item
    # Include the previously frozen two-method recipe as an independent transfer baseline.
    legacy=dict(records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0')),
        's1':dict(operation='source',parents=[],parameters=dict(source='source1')),
        'a':dict(operation='reconstruction.daily_month_mean',parents=['s0'],parameters={}),
        'b':dict(operation='fusion.primary_weighted_mean',parents=['a','s1'],parameters=dict(primary_index=0,primary_share=.6273899184553271))},key='b',identity='prior_robit_fixed')
    items[legacy['identity']]=legacy
    contexts={c:bench(c) for c in cases()};references={c:labels(c,'development') for c in cases()}
    rows=[]
    for index,item in enumerate(items.values()):
        scores={};errors={}
        for case,(context,engine,_) in contexts.items():
            try:
                output,_=engine.execute(item['records'],item['key']);scores[case]=measured(output,references[case],context)['objective']
            except Exception as exc:scores[case]=None;errors[case]=str(exc)
        rows.append(dict(identity=item['identity'],scores=scores,source_roots=sorted(origins(item)),errors=errors))
        if index%50==0:print('DEVELOPMENT FIXED REPLAY',index,'/',len(items),flush=True)
    feasible=[r for r in rows if all(v is not None for v in r['scores'].values())]
    assert feasible
    best=lambda group:min(group,key=lambda r:(np.mean(list(r['scores'].values())),r['identity']))
    chosen={name:items[best(group)['identity']] for name,group in [
       ('global_fixed',feasible),('station_only',[r for r in feasible if r['source_roots']==['source0']]),
       ('era5_only',[r for r in feasible if r['source_roots']==['source1']])]}
    raw=dict(records={'s1':dict(operation='source',parents=[],parameters=dict(source='source1'))},key='s1',identity='raw_era5')
    chosen.update(era5_raw=raw,legacy_fixed=legacy)
    per_case={case:items[min((r for r in rows if r['scores'][case] is not None),key=lambda r:(r['scores'][case],r['identity']))['identity']] for case in cases()}
    save(OUT/'control_development_scores.json',rows)
    save(target,dict(chosen=chosen,per_case=per_case,discovered_recipes=len(items),all_cases_feasible=len(feasible),
         selected_on='2018-2019 development only',test_used=False,source_space_complete=False,
         replay_seconds=perf_counter()-started,replay_adapter_calls=sum(e.calls for _,e,_ in contexts.values())))
    print('CONTROLS FROZEN',len(items),flush=True)


def predict():
    freeze();target=OUT/'predictions_frozen.json'
    if target.exists():raise ValueError('Predictions already frozen')
    fixed=load(OUT/'controls_frozen.json');manifest=[]
    for case in cases():
        for arm in ARMS:
            for seed in SEEDS:
                folder=OUT/'runs/validation'/(case+'_'+str(seed)+'_'+arm);s=load(folder/'summary.json')
                if s['status']!='PASS':raise ValueError('Failed seed not silently replaced: '+str(folder))
                name=arm+'_'+str(seed);path=OUT/'predictions'/case/(name+'.nc');path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(s['product'],path)
                save(path.with_suffix('.graph.json'),recipe(folder,s['best']['key']))
                manifest.append(dict(case=case,method=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        recipes=dict(fixed['chosen'],case_fixed=fixed['per_case'][case])
        for name,item in recipes.items():
            config=load(OUT/'cases'/case/'config.json')
            config.update(mode='PortraitMutation',seed=11,budget=BUDGET,search_validation=dict(path=str(OUT/'cases'/case/'validation_labels.nc')))
            config['request'].update(mutation_control='fixed',fixed_graph=dict(records=item['records'],key=item['key']),response_scoring=dict(enabled=False))
            folder=OUT/'fixed_runs'/case/name
            save(folder/'client_config.json',config)
            s=api.submit_file_request(config,folder);assert s['status']=='PASS'
            path=OUT/'predictions'/case/(name+'.nc');shutil.copy2(s['product'],path)
            save(path.with_suffix('.graph.json'),item)
            manifest.append(dict(case=case,method=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('FROZEN CASE PREDICTIONS',case,flush=True)
    save(target,manifest);print('ALL PREDICTIONS FROZEN BEFORE TEST SCORING',len(manifest),flush=True)


def score():
    freeze();manifest=load(OUT/'predictions_frozen.json');rows=[]
    for entry in manifest:
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        reference=labels(entry['case'],'test');context,_,_=bench(entry['case'])
        with xr.open_dataset(path) as f:output=f.load()
        metric=measured(output,reference,context)
        rows.append(dict(case=entry['case'],method=entry['method'],**metric))
    groups={arm:(lambda r,a=arm:r['method'].startswith(a+'_')) for arm in ARMS}
    for name in load(OUT/'controls_frozen.json')['chosen']:groups[name]=lambda r,n=name:r['method']==n
    groups['case_fixed']=lambda r:r['method']=='case_fixed'
    results={};station_values={}
    for name,condition in groups.items():
        selected=[r for r in rows if condition(r)]
        by_site={sid:float(np.mean([r['objective'] for r in selected if r['case'].startswith(sid)])) for sid in STATIONS}
        station_values[name]=np.array(list(by_site.values()))
        results[name]=dict(RMSE=float(np.mean(list(by_site.values()))),by_station=by_site,
             by_gap={k:float(np.mean([r['objective'] for r in selected if r['case'].endswith(k)])) for k in ('short','long')},
             MAE=float(np.mean([r['MAE'] for r in selected])),Bias=float(np.mean([r['Bias'] for r in selected])))
    comparisons={};rng=np.random.default_rng(20261028)
    for name,base in station_values.items():
        if name=='full':continue
        delta=base-station_values['full'];indices=rng.integers(0,len(STATIONS),(10000,len(STATIONS)))
        comparisons[name]=dict(improvement_percent=float(100*(1-station_values['full'].mean()/base.mean())),
             paired_station_differences=delta.tolist(),station_wins=int((delta>1e-10).sum()),
             descriptive_station_bootstrap_95=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist())
    costs=[]
    for folder in sorted((OUT/'runs/validation').iterdir()):
        s=load(folder/'summary.json');meta=load(folder/'run_metadata.json');cost=s['total_cost']
        costs.append(dict(**meta,status=s['status'],calls=cost['used'],limit=cost['limit'],phases=cost['phases'],
             seconds=cost['service_seconds'],failed_calls=cost['failed_calls'],fit_adjust_calls=cost['fit_adjust_calls'],
             graph_requests=s['executed_candidates'],scientific_duplicates=s['scientific_state_deduplications'],
             selected_nodes=s['best']['algorithm_nodes'],response_intervals=s['response_score_intervals'],
             initial_validation=s['initial_best']['metrics']['objective'],final_validation=s['best']['metrics']['objective']))
    save(OUT/'test_metrics.json',rows);save(OUT/'effect_summary.json',dict(results=results,comparisons=comparisons,
        independent_clusters='Three stations; masks, seeds and years are repeated measures',new_data_test_not_used_for_tuning=True,
        global_optimum_proven=False,all_fixed_and_search_predictions_frozen=True))
    save(OUT/'execution_costs.json',costs)
    print(json.dumps(results,indent=2),flush=True)


def weather():
    index=[]
    for entry in load(OUT/'predictions_frozen.json'):
        if entry['method'] not in {'full_11','random_11','global_fixed','station_only','case_fixed'}:continue
        sid=entry['case'].split('_')[0];meta=load(DATA/sid/'metadata.json')
        with xr.open_dataset(entry['path']) as f:field=f[VAR].sel(time=slice('2022-01-01','2024-12-31T23:59')).load()
        dates=pd.DatetimeIndex(field.time.values);frame=pd.DataFrame(dict(year=dates.year,yday=dates.dayofyear,precipitation=field.values.ravel()))
        folder=OUT/'precipitation_handoff'/entry['case']/entry['method'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/'station.pcp';path.write_bytes(SWATPlusAdapter._weather_file(frame,meta,'pcp'))
        read=pd.read_csv(path,sep=r'\s+',skiprows=3,header=None)
        np.testing.assert_allclose(read[2],frame.precipitation,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(read[0],frame.year);np.testing.assert_array_equal(read[1],frame.yday)
        index.append(dict(case=entry['case'],method=entry['method'],path=str(path),days=len(frame),readback='PASS',
                          precipitation_only=True,other_weather_not_invented=True,native_SWAT_run=False))
    save(OUT/'precipitation_handoff/index.json',index)
    print('PRECIPITATION FORMAT READBACK',len(index),flush=True)


if __name__=='__main__':{'controls':controls,'predict':predict,'score':score,'weather':weather}[sys.argv[1]]()
