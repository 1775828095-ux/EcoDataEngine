"""2015-only stopping calibration from real trajectories and repeated visible masks."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,sys,hashlib
import numpy as np
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor
from ecodata.algorithms.current_input import raster_validation
from ecodata.algorithms.search_stopping import SearchStopper,POLICY
from ecodata.algorithms.numerical_search import save
from run_halfdate_reconstruction import inputs,recipe

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'output/halfdate_reconstruction_20261028'
OUT=ROOT/'output/autonomous_search_closure_20261029';DEV=OUT/'development'
BUDGET=dict(objective_limit=360,proposal_limit=40000,wall_seconds=300,probe_call_limit=120)
SEEDS=[704,911];MASKS=[704,911,1237,1597]


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def cases():return [r['case'] for r in load(OLD/'prepared.json')['cases']]
def config(case):return load(OLD/'cases'/case/'config_development.json')


def job(case,seed):
    folder=DEV/f'{case}_{seed}'
    if (folder/'summary.json').exists():return dict(case=case,seed=seed,status=load(folder/'summary.json')['status'])
    cfg=config(case);request=cfg['request'];request['search_stopping']=dict(enabled=False,evidence='2015 development trajectory collection')
    result=api.process_current_input(inputs(cfg),request,folder,budget=BUDGET,seed=seed)
    return dict(case=case,seed=seed,status=result['status'],seconds=result['elapsed_seconds'])


def collect():
    OUT.mkdir(exist_ok=True)
    save(OUT/'calibration_protocol.json',dict(data='2015 development visible inputs only',cases=cases(),seeds=SEEDS,masks=MASKS,budget=BUDGET,
        noise='Sample standard deviation of one fixed initial platform graph under four independently masked visible cohorts',
        rule='Choose smallest patience retaining finite-budget validation best within one measured mask-standard-deviation on every development trajectory',
        caveat='Mask sensitivity heuristic, not an independence-based confidence interval or a global optimality certificate',
        min_evaluations=1,search='Existing public random mutation, early stopping disabled during trace collection',test_used=False))
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(job,c,s) for c in cases() for s in SEEDS]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)


def fit():
    noise={};relative=[]
    for c in cases():
        folder=DEV/f'{c}_{SEEDS[0]}';s=load(folder/'summary.json');assert s['status']=='PASS'
        graph=recipe(folder,s['initial_best']['key']);cfg=config(c);request,_=prepare_model_request(cfg['request'])
        original,_,_=build_context(inputs(cfg),request);values=[]
        for mask in MASKS:
            q=dict(request,current_input_validation=dict(seed=mask))
            changed,ref,protocol,confirmation=raster_validation(inputs(cfg),q,original)
            ctx,_,_=build_context(changed,q)
            out,_=Executor(ctx,field_registry(api.platform_registry(),ctx)).execute(graph['records'],graph['key'])
            values.append(measured(out,ref.reference,ctx)['objective'])
        sigma=float(np.std(values,ddof=1));noise[c]=dict(values=values,std=sigma,graph=graph)
        relative.append(sigma/max(np.mean(values),np.finfo(float).eps))
    delta=float(np.median(relative));traces=[];gaps=[];empty_gaps=[]
    for c in cases():
        for seed in SEEDS:
            p=DEV/f'{c}_{seed}';s=load(p/'summary.json');assert s['status']=='PASS'
            rows={r['key']:r for r in [json.loads(x) for x in (p/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]}
            transitions=load(p/'mutation_trace.json');resolution=s['objective_tie_resolution']
            best=s['initial_best'];anchor=best['metrics']['objective'];n=0;last_gain=0;last_iteration=0;events=[]
            def rank(row):return (round(row['metrics']['objective']/resolution),row['algorithm_nodes'],row['key'])
            for transition in transitions:
                actual=rows.get(transition['proposed_child']);row=rows.get(transition['child'])
                if actual is None or row is None:continue
                fresh=int(actual.get('total_new_calls',0)>0 and actual['origin']=='mutation')
                if fresh:
                    empty_gaps.append(transition['iteration']-last_iteration);last_iteration=transition['iteration'];n+=1
                if row['status']=='ELIGIBLE_OUTPUT' and not row.get('dominated') and rank(row)<rank(best):best=row
                value=best['metrics']['objective']
                if anchor-value>max(resolution,abs(anchor)*delta):gaps.append(max(1,n-last_gain));last_gain=n;anchor=value
                events.append(dict(iteration=transition['iteration'],evaluations=n,best=best))
            gaps.append(max(1,n-last_gain));empty_gaps.append(max(1,s['iterations']-last_iteration))
            traces.append(dict(case=c,seed=seed,summary=s,events=events,resolution=resolution,terminal=best))
    novelty=max(empty_gaps)+1
    candidates=sorted({1,*gaps,max(t['events'][-1]['evaluations'] if t['events'] else 0 for t in traces)+1})
    outcomes=[]
    for patience in candidates:
        declaration=dict(enabled=True,min_evaluations=1,patience_evaluations=patience,no_novel_iterations=novelty,min_relative_gain=delta)
        selected=[]
        for t in traces:
            stopper=SearchStopper(declaration,t['resolution']);stopper.start(t['summary']['initial_best']);best=t['summary']['initial_best'];evaluations=0;iteration=0
            for event in t['events']:
                for _ in range(iteration+1,event['iteration']):
                    stopper.observe(evaluations,best)
                    if stopper.reason:break
                if stopper.reason:break
                best=event['best'];evaluations=event['evaluations'];stopper.observe(evaluations,best);iteration=event['iteration']
                if stopper.reason:break
            loss=best['metrics']['objective']-t['terminal']['metrics']['objective']
            selected.append(dict(case=t['case'],seed=t['seed'],evaluations=evaluations,loss=loss,
                allowed=noise[t['case']]['std'],passed=loss<=max(t['resolution'],noise[t['case']]['std']),key=best['key']))
        outcomes.append(dict(policy=declaration,passed=all(r['passed'] for r in selected),evaluations=sum(r['evaluations'] for r in selected),runs=selected))
    valid=[r for r in outcomes if r['passed']];assert valid
    winner=min(valid,key=lambda r:(r['evaluations'],r['policy']['patience_evaluations']))
    published=dict(winner['policy'],version='visible_stagnation_2015_v1',calibration_data='2015 only',
        evidence='Four-mask initial-graph sensitivity and twelve real development trajectories; no 2016/2018 truth',
        calibration_sha256=hashlib.sha256(json.dumps(dict(noise=noise,outcomes=outcomes),sort_keys=True,default=str).encode()).hexdigest(),
        sources=['https://hub.optuna.org/callbacks/terminator/','https://proceedings.mlr.press/v188/makarova22a.html'],
        interpretation='Empirical stagnation tolerance; no GP regret-bound claim or global optimum certificate')
    assert not POLICY.exists(),'Do not overwrite a published calibration'
    save(POLICY,published);save(OUT/'calibration_results.json',dict(noise=noise,outcomes=outcomes,winner=winner,published=published))
    print(json.dumps(published,indent=2))


if __name__=='__main__':
    {'collect':collect,'fit':fit}[sys.argv[1]]()
