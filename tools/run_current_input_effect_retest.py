"""Same-data retrospective effect check: visible-only public decisions, sealed products."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib,json,sys,traceback,shutil
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,measured
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.numerical_search import save
from run_halfdate_reconstruction import inputs,recipe

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/halfdate_reconstruction_20261028'
BEFORE=ROOT/'output/library_update_retest_20261029'
OUT=ROOT/'output/current_input_effect_retest_20261029'
BUDGET=dict(objective_limit=1800,proposal_limit=400000,wall_seconds=3600,probe_call_limit=600)
SEEDS=[11,29,47]
WORKERS=6


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def cases():return [r['case'] for r in load(OLD/'prepared.json')['cases']]
def config(case):return load(OLD/'cases'/case/'config_test.json')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(exist_ok=True)
    files=list((ROOT/'src/ecodata').rglob('*.py'))+list((ROOT/'src/ecodata').rglob('*.json'))
    files+=list((ROOT/'data/attribute_registries').glob('*.json'))
    files+=[Path(__file__),ROOT/'tools/run_geo_complements.py',ROOT/'tools/run_starfm.py',ROOT/'tools/run_dineof.R',
        ROOT/'external/sinkr/R/dineof.R',ROOT/'external/starfm4py/src/starfm4py.py',ROOT/'external/geo-runtime/verified.json']
    manifest={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    if (OUT/'source_manifest.json').exists():
        assert manifest==load(OUT/'source_manifest.json'),'Source changed after protocol freeze'
        return
    for source in files:
        destination=OUT/'source_frozen'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    save(OUT/'source_manifest.json',manifest)
    save(OUT/'input_manifest.json',{p.relative_to(OLD).as_posix():digest(p) for p in (OLD/'cases').rglob('*') if p.is_file()})
    save(OUT/'historical_manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in [
        OLD/'controls_frozen.json',BEFORE/'effect_summary.json',BEFORE/'test_metrics.json',BEFORE/'deployment_failures.json']})
    save(OUT/'protocol.json',dict(status='RETROSPECTIVE_SAME_DATA_RETEST',cases=cases(),seeds=SEEDS,budget=BUDGET,workers=WORKERS,
        question='Does current-input autonomous reselection improve actual hidden-date reconstruction after repair?',
        library='Full current public registry, no hand-authored workflow pool or preference for new nodes',
        arms=['random','none','six original frozen fixed controls'],
        random_entry='submit_file_request -> process_current_input -> submit_search -> GraphMutation random',
        none_entry='submit_file_request -> submit_search PortraitMutation, mutation_control=none, same visible-only evaluation and deployment',
        controls='No mutation is explicit ablation only; fixed graphs selected in original 2015 development, unchanged',
        learning='Existing historical profiles/states reused without retraining or modifying weights; no external labels supplied',
        selection='Per 2018 visible input: existing deterministic disjoint internal search/confirmation holdouts; no hidden-date truth',
        confirmation='Not used for selection or acceptance; no tuning based on confirmation or test effects',
        test='All products, decisions and input hashes sealed before reading original 2018 hidden-date labels',
        primary='Same finite reference support: equal hidden-date RMSE, then equal case mean and seed mean',
        secondary=['MAE','Bias','gradient_RMSE','seed standard deviation','failure rate','registered calls','service seconds'],
        failure_policy='Retain all cases/seeds; any missing/invalid result makes arm incomplete, no favourable subset overall mean',
        comparisons=['random vs none','random vs original development_fixed','random vs old legacy_random',
            'random vs old expanded_random only on all-seed complete matched cases, not overall if historical arm incomplete'],
        inference='Descriptive retrospective result only: three spatially dependent patches, two masks, three seeds; no independent basin or p-value claim',
        before_after_boundary='Old searches selected on 2016 and froze graphs; new mode reselects from current visible input. Difference is full operational mode, not isolated code-only causal effect',
        policy_changes_after_freeze='None; all negative outcomes retained'))
    save(OUT/'planned_jobs.json',[(c,m,s) for c in cases() for m in ['random','none'] for s in SEEDS])


def run_job(case,mode,seed):
    folder=OUT/'runs'/f'{case}_{mode}_{seed}'
    if (folder/'client_result.json').exists():return load(folder/'client_result.json')
    cfg=config(case);cfg.pop('evaluation',None);cfg.pop('search_validation',None)
    cfg['request'].pop('mutation_control',None)
    if mode=='random':cfg['decision_policy']='current_input'
    else:cfg['request']['mutation_control']='none'
    cfg.update(mode='PortraitMutation',seed=seed,budget=BUDGET)
    save(folder/'client_config.json',cfg)
    reads=[];guard=dict(active=True)
    def audit(event,args):
        if event=='open' and guard['active'] and isinstance(args[0],(str,bytes,os.PathLike)):
            name=Path(os.fsdecode(args[0])).name
            if name in {'test_labels.nc','validation_labels.nc','development_labels.nc'}:
                reads.append(str(args[0]));raise RuntimeError('External reference forbidden during autonomous decisions')
    sys.addaudithook(audit)
    try:
        result=api.submit_file_request(cfg,folder/'platform')
        result=dict(status=result['status'],case=case,mode=mode,seed=seed,product=result.get('product'),
            key=(result.get('best') or {}).get('key'),backend=result['optimizer']['backend'],
            evolution_used=result['evolution_used'],total_cost=result['total_cost'],external_reference_reads=reads)
    except Exception as exc:
        result=dict(status='CLIENT_FAILED',case=case,mode=mode,seed=seed,error=str(exc),trace=traceback.format_exc(),external_reference_reads=reads)
    finally:guard['active']=False
    save(folder/'client_result.json',result)
    return {k:result.get(k) for k in ['case','mode','seed','status','backend','evolution_used']}


def search():
    freeze()
    jobs=load(OUT/'planned_jobs.json')
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(run_job,*job):job for job in jobs}
        for future in as_completed(futures):print('SEARCH',json.dumps(future.result()),flush=True)
    save(OUT/'search_complete.json',dict(jobs=len(jobs),all_returned=True))


def seal():
    freeze();assert (OUT/'search_complete.json').exists()
    if (OUT/'predictions_frozen.json').exists():return
    entries=[];failures=[]
    for case,mode,seed in load(OUT/'planned_jobs.json'):
        folder=OUT/'runs'/f'{case}_{mode}_{seed}';row=load(folder/'client_result.json');name=f'{mode}_{seed}'
        if row['status']!='PASS':failures.append(dict(case=case,method=name,stage='search/deploy',details=row));continue
        path=Path(row['product']);entries.append(dict(case=case,method=name,path=str(path),sha256=digest(path),
            graph=recipe(folder/'platform',row['key']),source='current_input'))
    for case in cases():
        cfg=config(case)
        for name,graph in load(OLD/'controls_frozen.json')['chosen'].items():
            folder=OUT/'fixed'/case/name
            try:
                result=load(folder/'deployment.json') if (folder/'deployment.json').exists() else api.execute_frozen_graph(inputs(cfg),cfg['request'],graph,folder)
                path=Path(result['product']);entries.append(dict(case=case,method=name,path=str(path),sha256=digest(path),graph=graph,source='2015_frozen_fixed'))
            except Exception as exc:
                row=dict(case=case,method=name,stage='fixed_deploy',error=str(exc),trace=traceback.format_exc());save(folder/'failure.json',row);failures.append(row)
    save(OUT/'failures.json',failures);save(OUT/'predictions_frozen.json',entries)
    print('ALL PREDICTIONS SEALED',len(entries),'FAILURES',len(failures),'BEFORE READING TEST LABELS',flush=True)


def score():
    freeze();assert (OUT/'predictions_frozen.json').exists()
    rows=[]
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert digest(path)==entry['sha256']
        case=entry['case'];cfg=config(case);request,_=prepare_model_request(cfg['request']);ctx,_,_=build_context(inputs(cfg),request)
        with xr.open_dataset(OLD/'cases'/case/'test_labels.nc') as f:truth=f.ndvi.load()
        with xr.open_dataset(path) as f:out=f.load()
        try:metric=measured(out,truth,ctx);status='PASS'
        except Exception as exc:metric=dict(error=str(exc));status='FAILED_OUTPUT'
        rows.append(dict(case=case,method=entry['method'],status=status,**metric))
    save(OUT/'test_metrics.json',rows)
    arms=['random','none',*load(OLD/'controls_frozen.json')['chosen']];results={}
    for arm in arms:
        selected=[r for r in rows if (r['method'].startswith(arm+'_') if arm in ['random','none'] else r['method']==arm)]
        expected=18 if arm in ['random','none'] else 6
        if len(selected)!=expected or any(r['status']!='PASS' for r in selected):
            results[arm]=dict(status='INCOMPLETE',observed=len(selected),passed=sum(r['status']=='PASS' for r in selected),expected=expected);continue
        by_case={c:float(np.mean([r['objective'] for r in selected if r['case']==c])) for c in cases()}
        values=[float(np.mean([r['objective'] for r in selected if r['method']==f'{arm}_{s}'])) for s in SEEDS] if arm in ['random','none'] else []
        results[arm]=dict(status='PASS',RMSE=float(np.mean(list(by_case.values()))),by_case=by_case,
            seed_RMSE=values,seed_SD=float(np.std(values,ddof=1)) if values else None,
            **{k:float(np.mean([r[k] for r in selected])) for k in ['MAE','Bias','gradient_RMSE']})
    save(OUT/'effect_summary.json',results);print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='all':search();seal();score()
    else:{'freeze':freeze,'search':search,'seal':seal,'score':score}[stage]()
