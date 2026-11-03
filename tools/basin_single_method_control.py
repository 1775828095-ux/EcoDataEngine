"""Pre-test best discovered single-method comparator; never changes search."""
from pathlib import Path
import sys,hashlib,json
import numpy as np
import xarray as xr
from run_basin_ecology import OUT,SEEDS,BUDGET,load,freeze
from evaluate_basin_ecology import cases,recipe,bench,labels
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save


def predict():
    freeze()
    if (OUT/'test_metrics.json').exists():raise ValueError('Single-method selection must precede all test scoring')
    if (OUT/'single_method_manifest.json').exists():raise ValueError('Single-method predictions already sealed')
    entries=[]
    for case in cases():
        for seed in SEEDS:
            source=OUT/'runs/validation'/(case+'_random_'+str(seed))
            rows=[json.loads(line) for line in (source/'attempts.jsonl').read_text().splitlines()]
            rows=[r for r in rows if r['status']=='ELIGIBLE_OUTPUT' and r['algorithm_nodes']<=1]
            if not rows:raise ValueError('No discovered complete single-method comparator')
            best=min(rows,key=lambda r:(r['metrics']['objective'],r['algorithm_nodes'],r['key']))
            graph=recipe(source,best['key']);config=load(OUT/'cases'/case/'config.json')
            config['request'].update(mutation_control='fixed',fixed_graph=graph)
            config.update(mode='PortraitMutation',seed=seed,budget=BUDGET,search_validation=dict(path=str(OUT/'cases'/case/'validation_labels.nc')))
            folder=OUT/'single_method_runs'/case/str(seed);save(folder/'client_config.json',config)
            result=api.submit_file_request(config,folder);assert result['status']=='PASS'
            path=Path(result['product']);entries.append(dict(case=case,seed=seed,path=str(path),
                graph=graph,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selection='2016 validation only; already executed <=1 node',
                calls=result['total_cost']['used'],seconds=result['total_cost']['service_seconds']))
    save(OUT/'single_method_manifest.json',entries);print('SINGLE METHOD SEALED BEFORE TEST',len(entries),flush=True)


def score():
    freeze();entries=load(OUT/'single_method_manifest.json');base=load(OUT/'test_metrics.json');rows=[];audit_calls=0
    for entry in entries:
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        context,executor,_=bench(entry['case']);replay,_=executor.execute(entry['graph']['records'],entry['graph']['key']);audit_calls+=executor.calls
        source=context.sources[0].dataset;replay.fvc.values[:]=np.where(np.isfinite(source.fvc),source.fvc,replay.fvc)
        with xr.open_dataset(path) as f:output=f.load()
        np.testing.assert_allclose(replay.fvc,output.fvc,rtol=1e-10,atol=1e-10)
        reference=labels(entry['case'],'test');pred=output.fvc.sel(time=reference.time,lat=reference.lat,lon=reference.lon).values
        error=pred-reference.values;rmse=float(np.sqrt(np.mean(error**2)))
        main=next(r for r in base if r['case']==entry['case'] and r['method']=='random_'+str(entry['seed']))
        rows.append(dict(case=entry['case'],seed=entry['seed'],single_RMSE=rmse,random_RMSE=main['objective'],
            difference=rmse-main['objective'],single_nodes=sum(r['operation']!='source' for r in entry['graph']['records'].values())))
    single=float(np.mean([r['single_RMSE'] for r in rows]));random=float(np.mean([r['random_RMSE'] for r in rows]))
    save(OUT/'single_method_comparison.json',dict(rows=rows,single_mean_RMSE=single,random_mean_RMSE=random,
       improvement_percent=100*(1-random/single),deployment_calls=sum(r['calls'] for r in entries),
       deployment_seconds=sum(r['seconds'] for r in entries),replay_adapter_calls=audit_calls,independent_replay='PASS',
       test_used_for_single_selection=False,no_additional_search_budget=True))
    print('SINGLE METHOD COMPARISON',single,random,100*(1-random/single),flush=True)


if __name__=='__main__':{'predict':predict,'score':score}[sys.argv[1]]()
