"""Development controls, sealed predictions, then test scoring. No route tuning."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from time import perf_counter
import hashlib,json,sys,shutil
import numpy as np
import xarray as xr
from run_basin_ecology import ROOT,OUT,SEEDS,BUDGET,load,freeze
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor,canonical
from ecodata.algorithms.numerical_search import save


def cases():return [b['case'] for b in load(OUT/'prepared.json')['blocks']]


def bench(case):
    config=load(OUT/'cases'/case/'config.json')
    inputs=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in config['inputs']]
    # Use the exact same public request preparation as the real search service.
    from ecodata.algorithms.profile_matching import prepare_model_request
    request,_=prepare_model_request(config['request']);context,_,_=build_context(inputs,request)
    return context,Executor(context,field_registry(api.platform_registry(),context)),config


def labels(case,phase):
    with xr.open_dataset(OUT/'cases'/case/(phase+'_labels.nc')) as f:return f.fvc.load()


def recipe(folder,key):
    all_records=load(folder/'graphs.json');records={}
    def visit(k):
        if k in records:return
        for parent in all_records[k]['parents']:visit(parent)
        records[k]=all_records[k]
    visit(key)
    def identity(k):
        r=records[k];return [r['operation'],r['parameters'],[identity(p) for p in r['parents']]]
    return dict(records=records,key=key,identity=canonical(identity(key)))


def controls():
    freeze();items={};started=perf_counter()
    for folder in sorted((OUT/'runs/development').iterdir()):
        for line in (folder/'attempts.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['status']=='ELIGIBLE_OUTPUT':
                item=recipe(folder,r['key']);items[item['identity']]=item
    contexts={c:bench(c) for c in cases()};references={c:labels(c,'development') for c in cases()}
    evaluations=[]
    for index,item in enumerate(items.values()):
        scores={};failures={}
        for case,(context,executor,_) in contexts.items():
            try:output,_=executor.execute(item['records'],item['key']);scores[case]=measured(output,references[case],context)['objective']
            except Exception as exc:scores[case]=None;failures[case]=str(exc)
        evaluations.append(dict(identity=item['identity'],scores=scores,failures=failures))
        if index%50==0:print('FIXED DEVELOPMENT REPLAY',index,'/',len(items),flush=True)
    valid=[r for r in evaluations if all(v is not None for v in r['scores'].values())]
    if not valid:raise ValueError('No cross-case development fixed graph')
    best=min(valid,key=lambda r:(np.mean(list(r['scores'].values())),r['identity']))
    chosen=dict(development_fixed=items[best['identity']],prior=dict(records={'s1':dict(operation='source',parents=[],parameters=dict(source='source1'))},key='s1'))
    # These simple reference methods are controls only, never search seeds or a pool.
    features=load(OUT/'prepared.json')['features']
    for name,operation,parameters in [('nearest','reconstruction.grid_gap_nearest',{}),('linear','reconstruction.grid_gap_linear',{}),
         ('forest_all','reconstruction.prior_grid_forest',dict(grid_predictor_features=features))]:
        chosen[name]=dict(records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0')),
            'fixed':dict(operation=operation,parents=['s0'],parameters=parameters)},key='fixed')
    save(OUT/'fixed_development_scores.json',evaluations)
    save(OUT/'controls_frozen.json',dict(chosen=chosen,discovered_recipes=len(items),cross_case_feasible=len(valid),
         selected_on='2015 development only',test_used=False,global_space_complete=False,
         replay_adapter_calls=sum(e.calls for _,e,_ in contexts.values()),seconds=perf_counter()-started))
    print('CONTROLS FROZEN',len(items),len(valid),flush=True)


def predict():
    freeze();manifest=[]
    if (OUT/'predictions_frozen.json').exists():raise ValueError('Final predictions already sealed')
    for case in cases():
        for control in ['random','none']:
            for seed in SEEDS:
                folder=OUT/'runs/validation'/(case+'_'+control+'_'+str(seed));s=load(folder/'summary.json')
                if s['status']!='PASS':raise ValueError('Failed run retained, not silently dropped: '+str(folder))
                path=OUT/'predictions'/case/(control+'_'+str(seed)+'.nc');path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(s['product'],path);save(path.with_suffix('.graph.json'),recipe(folder,s['best']['key']))
                manifest.append(dict(case=case,method=control+'_'+str(seed),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for name,item in load(OUT/'controls_frozen.json')['chosen'].items():
            config=load(OUT/'cases'/case/'config.json');config['request'].update(mutation_control='fixed',fixed_graph=item)
            config.update(mode='PortraitMutation',seed=11,budget=BUDGET,search_validation=dict(path=str(OUT/'cases'/case/'validation_labels.nc')))
            folder=OUT/'fixed_runs'/case/name;save(folder/'client_config.json',config)
            s=api.submit_file_request(config,folder);assert s['status']=='PASS'
            path=OUT/'predictions'/case/(name+'.nc');shutil.copy2(s['product'],path);save(path.with_suffix('.graph.json'),item)
            manifest.append(dict(case=case,method=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('PREDICTIONS',case,flush=True)
    assert len(manifest)==33
    save(OUT/'predictions_frozen.json',manifest);print('ALL PREDICTIONS FROZEN BEFORE TEST',len(manifest),flush=True)


def score():
    freeze();rows=[]
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        context,_,_=bench(entry['case'])
        with xr.open_dataset(path) as f:output=f.load()
        metric=measured(output,labels(entry['case'],'test'),context)
        rows.append(dict(case=entry['case'],method=entry['method'],**metric))
    names=['random','none',*load(OUT/'controls_frozen.json')['chosen']];results={}
    for name in names:
        selected=[r for r in rows if r['method'].startswith(name+'_')] if name in ['random','none'] else [r for r in rows if r['method']==name]
        by_case={case:float(np.mean([r['objective'] for r in selected if r['case']==case])) for case in cases()}
        results[name]=dict(RMSE=float(np.mean(list(by_case.values()))),by_case=by_case,
            MAE=float(np.mean([r['MAE'] for r in selected])),Bias=float(np.mean([r['Bias'] for r in selected])),
            x_edge_RMSE=float(np.mean([r['gradient_RMSE'] for r in selected])))
    comparisons={}
    for name,r in results.items():
        if name=='random':continue
        differences=np.array([r['by_case'][c]-results['random']['by_case'][c] for c in cases()])
        comparisons[name]=dict(improvement_percent=float(100*(1-results['random']['RMSE']/r['RMSE'])),
             paired_block_differences=differences.tolist(),block_wins=int((differences>1e-12).sum()))
    costs=[]
    for folder in sorted((OUT/'runs/validation').iterdir()):
        summary=load(folder/'summary.json');ledger=load(folder/'total_cost.json');meta=load(folder/'metadata.json')
        costs.append(dict(**meta,status=summary['status'],calls=ledger['used'],seconds=ledger['service_seconds'],
             training_capable_calls=ledger['training_capable_calls'],failed_calls=ledger['failed_calls'],
             nodes=summary['best']['algorithm_nodes'],initial_validation=summary['initial_best']['metrics']['objective'],
             final_validation=summary['best']['metrics']['objective'],mutation_used=summary['evolution_used']))
    save(OUT/'test_metrics.json',rows);save(OUT/'effect_summary.json',dict(results=results,comparisons=comparisons,
        scope='One basin, three dependent spatial blocks, one held-out year; repeated seeds do not enlarge independent sample size',
        true_ground_FVC_accuracy_assessed=False,test_used_for_tuning=False))
    save(OUT/'execution_costs.json',costs);print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':{'controls':controls,'predict':predict,'score':score}[sys.argv[1]]()
