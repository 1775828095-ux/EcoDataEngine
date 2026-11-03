"""Freeze discovered fixed controls and final predictions before scoring truth."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from time import perf_counter
import sys,json,hashlib
import numpy as np
import xarray as xr
from run_open_complementary import ROOT,OUT,PROTOCOL,materialize,run_one
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor,canonical,clean
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def recipe(folder,key):
    records=load(folder/'graphs.json');out={}
    def visit(k):
        if k in out:return
        for p in records[k]['parents']:visit(p)
        out[k]=records[k]
    visit(key)
    def identity(k):
        r=records[k];return [r['operation'],r['parameters'],[identity(p) for p in r['parents']]]
    return dict(records=out,key=key,identity=canonical(identity(key)))


def replay(item,inputs,request):
    context,_,_=build_context(inputs,request);registry=field_registry(api.platform_registry())
    engine=Executor(context,registry);output,_=engine.execute(item['records'],item['key'])
    return output,context


def controls():
    target=OUT/'controls_frozen.json'
    if target.exists():return load(target)
    items={}
    for folder in (OUT/'runs_v6/development').iterdir():
        if not (folder/'summary.json').exists():continue
        for line in (folder/'attempts.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['status']=='ELIGIBLE_OUTPUT':
                item=recipe(folder,row['key']);items[item['identity']]=item
    evaluations=[];benchmarks=[];started=perf_counter()
    for config in PROTOCOL['development']:
        inputs,request,reference,_,_=materialize(config)
        context,_,_=build_context(inputs,request)
        reg=field_registry(api.platform_registry(),context)
        benchmarks.append((Executor(context,reg),reference,context))
    for identity,item in items.items():
        scores=[]
        for engine,ref,context in benchmarks:
            try:
                output,_=engine.execute(item['records'],item['key']);scores.append(measured(output,ref,context)['objective'])
            except Exception as exc:scores.append(None)
        origins={r['parameters']['source'] for r in item['records'].values() if r['operation']=='source'}
        evaluations.append(dict(identity=identity,scores=scores,origins=sorted(origins)))
        if len(evaluations)%25==0:print('CONTROL_RECIPES',len(evaluations),'/',len(items),flush=True)
    feasible=[r for r in evaluations if all(s is not None for s in r['scores'])]
    if not feasible:raise ValueError('No cross-development fixed control')
    ranking=lambda r:(np.mean(r['scores']),r['identity'])
    selected={}
    for name,origins in [('strong_fixed',None),('fine_only',{'source0'}),('coarse_only',{'source1'})]:
        candidates=[r for r in feasible if origins is None or set(r['origins'])==origins]
        if not candidates:raise ValueError('No '+name+' control')
        winner=min(candidates,key=ranking);selected[name]=items[winner['identity']]
    save(OUT/'fixed_control_development_evaluations.json',evaluations)
    result=dict(controls=selected,discovered_recipe_count=len(items),feasible_count=len(feasible),
        selected_on='Mean held-out fine-anchor RMSE on development cases only',test_seen=False,
        development_control_replay_seconds=perf_counter()-started,development_control_adapter_calls=sum(b[0].calls for b in benchmarks))
    save(target,result);return result


def predict():
    fixed=controls()['controls'];manifest=[]
    for index,config in enumerate(PROTOCOL['test']):
        _,request,_,inputs,_=materialize(config)
        for method,item in fixed.items():
            path=OUT/'predictions'/str(index)/(method+'.nc')
            if not path.exists():
                output,_=replay(item,inputs,request);path.parent.mkdir(parents=True,exist_ok=True)
                clean(output).to_netcdf(path,engine='h5netcdf')
            manifest.append(dict(case=index,method=method,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks={pool.submit(run_one,'test',index,seed,policy):(index,seed,policy)
            for index in range(len(PROTOCOL['test'])) for seed in PROTOCOL['seeds'] for policy in ('contract_guided','random')}
        for task in as_completed(tasks):
            index,seed,policy=tasks[task];s=task.result()
            if not s['best']:raise ValueError('No eligible graph; keep failed case '+str(index))
            assert s['self_validation_from_visible_anchors']
            folder=Path(s['product']).parent;item=recipe(folder,s['best']['key'])
            method=policy+'_'+str(seed);path=OUT/'predictions'/str(index)/(method+'.nc')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(Path(s['product']).read_bytes())
            save(path.with_suffix('.graph.json'),item)
            manifest.append(dict(case=index,method=method,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                search_seconds=s['elapsed_seconds'],kernel_calls=s['kernel_calls'],executed_candidates=s['executed_candidates']))
            print('PREDICTED',index,method,s['best']['metrics']['objective'],flush=True)
    manifest.sort(key=lambda r:(r['case'],r['method']))
    target=OUT/'predictions_frozen.json'
    if target.exists():assert load(target)==manifest
    else:save(target,manifest)
    print('ALL PREDICTIONS FROZEN; final truth not scored',flush=True)


def score():
    manifest=load(OUT/'predictions_frozen.json');rows=[]
    for entry in manifest:
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        _,request,_,inputs,truth=materialize(PROTOCOL['test'][entry['case']])
        context,_,_=build_context(inputs,request)
        with xr.open_dataset(path) as f:output=f.load()
        rows.append(dict(case=entry['case'],method=entry['method'],**measured(output,truth,context)))
    methods=sorted({r['method'] for r in rows});summary={}
    for method in methods:
        selected=[r for r in rows if r['method']==method]
        summary[method]=dict(mean_case_objective=float(np.mean([r['objective'] for r in selected])),
            case_objectives=[r['objective'] for r in selected])
    adaptive=np.array([np.mean([r['objective'] for r in rows if r['case']==i and r['method'].startswith('contract_guided')]) for i in range(len(PROTOCOL['test']))])
    rng=np.random.default_rng(20261027);comparisons={}
    for baseline in ['fine_only','coarse_only','strong_fixed','random']:
        base=np.array([np.mean([r['objective'] for r in rows if r['case']==i and (r['method'].startswith('random') if baseline=='random' else r['method']==baseline)]) for i in range(len(PROTOCOL['test']))])
        difference=base-adaptive;resamples=rng.integers(0,len(base),size=(10000,len(base)))
        comparisons[baseline]=dict(relative_improvement_percent=float(100*(1-adaptive.mean()/base.mean())),
            case_wins=int((difference>0).sum()),case_differences=difference.tolist(),
            descriptive_case_bootstrap_95_interval=np.quantile(difference[resamples].mean(axis=1),[.025,.975]).tolist(),
            warning='Four case blocks only; descriptive interval, not population significance proof')
    save(OUT/'test_metrics.json',rows);save(OUT/'effect_summary.json',dict(methods=summary,comparisons=comparisons,
        final_truth_read_after_prediction_freeze=True,global_optimum_proven=False))
    print(json.dumps(comparisons,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='controls':print('FROZEN',controls()['discovered_recipe_count'],flush=True)
    elif stage=='predict':predict()
    elif stage=='score':score()
