"""Same-data retrospective library comparison through public platform APIs.

Only the control library's membership is frozen; no candidate workflow pools.
No test labels are loaded until every selected graph and product is sealed.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import hashlib,json,sys,traceback,shutil
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.autonomous_field import build_context,measured
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.numerical_search import save
from run_halfdate_reconstruction import recipe,inputs

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/halfdate_reconstruction_20261028'
OUT=ROOT/'output/library_update_retest_20261029'
BUDGET=dict(objective_limit=1800,proposal_limit=400000,wall_seconds=3600,probe_call_limit=600)
SEEDS=[11,29,47]
MAX_WORKERS=12
BASE_FACTORY=api.platform_registry


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def cases():return [r['case'] for r in load(OLD/'prepared.json')['cases']]
def config(case,phase):return load(OLD/'cases'/case/f'config_{phase}.json')


def set_library(condition):
    legacy=set(load(OUT/'legacy_algorithm_ids.json'))
    def factory(*,weights=None):
        full=BASE_FACTORY(weights=weights)
        if condition=='expanded':return full
        registry=AlgorithmRegistry()
        for spec in full.specs():
            if spec.id in legacy:registry.register(full.get(spec.id))
        return registry
    api.platform_registry=factory


def freeze():
    OUT.mkdir(exist_ok=True)
    legacy=set(load(OUT/'legacy_algorithm_ids.json'));current=BASE_FACTORY()
    added=sorted({s.id for s in current.specs()}-legacy)
    assert len(added)==10
    files=list((ROOT/'src/ecodata').rglob('*.py'))+list((ROOT/'src/ecodata').rglob('*.json'))
    files+=list((ROOT/'data/attribute_registries').glob('*.json'))
    files+=[Path(__file__),ROOT/'tools/run_geo_complements.py',ROOT/'tools/run_starfm.py',ROOT/'tools/run_dineof.R',
        ROOT/'external/sinkr/R/dineof.R',ROOT/'external/starfm4py/src/starfm4py.py',ROOT/'external/geo-runtime/verified.json']
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=OUT/'source_manifest.json'
    if path.exists():
        assert load(path)==manifest,'Frozen retest source changed'
        return
    for file in files:
        dest=OUT/'source_frozen'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    save(path,manifest)
    input_hashes={str(p.relative_to(OLD)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD/'cases').rglob('*') if p.is_file()}
    save(OUT/'input_manifest.json',input_hashes)
    save(OUT/'protocol.json',dict(question='Does expanded executable library improve same-data reconstruction?',
        status='RETROSPECTIVE_RETEST',historical_test_already_seen=True,training='2013-2014 original QA only; no new learned ranker',
        development='2015 original fixed controls unchanged',validation=2016,test=2018,
        cases=cases(),seeds=SEEDS,libraries=['legacy','expanded'],legacy_membership_source='Original experiment frozen registry export; same current kernels and engine',
        added_algorithms=added,new_pool='Full current public registry; no handcrafted graph templates or method preference',
        controls=['random','none','original frozen fixed controls'],budget=BUDGET,parallel_workers=MAX_WORKERS,
        budget_purpose='10x previous numerical-call allowance, generous resource ceiling; accuracy primary, no efficiency claim',
        fixed_selection='Original 2015 development selection retained',primary_metric='Equal hidden-date RMSE, then case mean, then seed mean',
        failure_policy='All seeds and failed candidates/deployments retained; incomplete arm gets no favourable subset mean',
        test_policy='Seal all graphs and prediction products before loading 2018 labels; do not tune to test advantage',
        independent_units='Three spatially dependent patches, two masks; no claim seeds/pixels are independent basins'))
    save(OUT/'catalogs.json',dict(legacy_ids=sorted(legacy),expanded=[s.as_dict() for s in current.specs()]))
    # Explain applicability without filtering any expanded-library candidates.
    applicability=[]
    for case in cases():
        cfg=config(case,'validation');request,_=prepare_model_request(cfg['request']);ctx,_,_=build_context(inputs(cfg),request)
        from dataclasses import replace
        from itertools import permutations
        for identifier in added:
            plugin=current.get(identifier);arity=plugin.spec.input_contract.get('source_arity',1);rows=[]
            for parents in permutations(range(len(ctx.sources)),arity):
                local=replace(ctx,sources=tuple(ctx.sources[i] for i in parents))
                variants=plugin.parameter_proposals(local,np.random.default_rng(11)) if hasattr(plugin,'parameter_proposals') else [{}]
                checks=[plugin.compatibility(replace(local,parameters=dict(local.parameters,**p))) for p in variants]
                check=plugin.compatibility(local)
                rows.append(dict(parents=parents,initial_parameter_proposals=len(variants),eligible=any(x.eligible for x in checks),reasons=list(check.reasons)))
            applicability.append(dict(case=case,algorithm=identifier,initial_parents=rows,
                downstream='May become eligible after other registered nodes; never excluded solely by initial applicability'))
    save(OUT/'initial_applicability.json',applicability)


def search_job(condition,case,control,seed):
    set_library(condition);folder=OUT/'runs'/condition/f'{case}_{control}_{seed}'
    if (folder/'summary.json').exists():return load(folder/'summary.json')
    if (folder/'client_failure.json').exists():return load(folder/'client_failure.json')
    cfg=config(case,'validation');cfg['request']['mutation_control']=control
    cfg.update(mode='PortraitMutation',seed=seed,budget=BUDGET,search_validation=dict(path=str(OLD/'cases'/case/'validation_labels.nc')))
    save(folder/'client_config.json',cfg);save(folder/'metadata.json',dict(library=condition,case=case,control=control,seed=seed))
    try:return api.submit_file_request(cfg,folder)
    except Exception as exc:
        row=dict(status='BLOCKED',error=str(exc),trace=traceback.format_exc());save(folder/'client_failure.json',row);return row
    finally:api.platform_registry=BASE_FACTORY


def search():
    freeze();jobs=[(lib,c,control,s) for c in cases() for s in SEEDS for lib in ['legacy','expanded'] for control in ['random','none']]
    save(OUT/'planned_jobs.json',jobs)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures={pool.submit(search_job,*job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result();job=futures[future];best=row.get('best') or {}
            print('SEARCH',*job,row['status'],'calls',row.get('total_cost',{}).get('used'),'validation',best.get('metrics',{}).get('objective'),flush=True)
    save(OUT/'search_complete.json',dict(jobs=len(jobs),all_returned=True))


def predict():
    freeze();assert (OUT/'search_complete.json').exists()
    if (OUT/'predictions_frozen.json').exists():return
    manifest=[];failures=[]
    for case in cases():
        selected=[]
        for lib in ['legacy','expanded']:
            for control in ['random','none']:
                for seed in SEEDS:
                    folder=OUT/'runs'/lib/f'{case}_{control}_{seed}'
                    summary=load(folder/'summary.json') if (folder/'summary.json').exists() else load(folder/'client_failure.json')
                    name=f'{lib}_{control}_{seed}'
                    if not summary.get('best'):
                        failures.append(dict(case=case,method=name,stage='search',error=summary.get('error',summary['status'])));continue
                    selected.append((name,recipe(folder,summary['best']['key'])))
        selected+=list(load(OLD/'controls_frozen.json')['chosen'].items())
        cfg=config(case,'test')
        for name,graph in selected:
            folder=OUT/'deployments'/case/name
            try:
                result=load(folder/'deployment.json') if (folder/'deployment.json').exists() else api.execute_frozen_graph(inputs(cfg),cfg['request'],graph,folder)
                path=Path(result['product']);manifest.append(dict(case=case,method=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),graph=graph,
                    registered_calls=result['cost']['used'],service_seconds=result['cost'].get('service_seconds')))
            except Exception as exc:
                failure=dict(case=case,method=name,stage='deployment',error=str(exc),trace=traceback.format_exc())
                failures.append(failure);save(folder/'failure.json',failure)
        print('SEALED CASE',case,'products',len(manifest),'failures',len(failures),flush=True)
    save(OUT/'deployment_failures.json',failures);save(OUT/'predictions_frozen.json',manifest)
    print('ALL PREDICTIONS SEALED; NO TEST LABELS LOADED',flush=True)


def score():
    freeze();assert (OUT/'predictions_frozen.json').exists()
    rows=[]
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        case=entry['case'];cfg=config(case,'test');request,_=prepare_model_request(cfg['request']);ctx,_,_=build_context(inputs(cfg),request)
        with xr.open_dataset(OLD/'cases'/case/'test_labels.nc') as d:truth=d.ndvi.load()
        with xr.open_dataset(path) as d:product=d.load()
        try:metric=measured(product,truth,ctx);status='PASS'
        except Exception as exc:metric=dict(error=str(exc));status='FAILED_OUTPUT'
        rows.append(dict(case=case,method=entry['method'],status=status,**metric))
    results={}
    arms=[f'{lib}_{control}' for lib in ['legacy','expanded'] for control in ['random','none']]
    fixed=list(load(OLD/'controls_frozen.json')['chosen'])
    for method in [*arms,*fixed]:
        selected=[r for r in rows if (r['method'].startswith(method+'_') if method in arms else r['method']==method)]
        expected=18 if method in arms else 6
        if len(selected)!=expected or any(r['status']!='PASS' for r in selected):
            results[method]=dict(status='INCOMPLETE',observed=len(selected),passed=sum(r['status']=='PASS' for r in selected),expected=expected);continue
        by_case={c:float(np.mean([r['objective'] for r in selected if r['case']==c])) for c in cases()}
        results[method]=dict(status='PASS',RMSE=float(np.mean(list(by_case.values()))),by_case=by_case,
            **{k:float(np.mean([r[k] for r in selected])) for k in ['MAE','Bias','gradient_RMSE']})
    save(OUT/'test_metrics.json',rows);save(OUT/'effect_summary.json',results)
    trace=[];usage=[]
    for job in load(OUT/'planned_jobs.json'):
        lib,case,control,seed=job;folder=OUT/'runs'/lib/f'{case}_{control}_{seed}'
        if not (folder/'summary.json').exists():continue
        s=load(folder/'summary.json');graph=recipe(folder,s['best']['key']) if s.get('best') else None
        best_ops=[r['operation'] for r in graph['records'].values() if r['operation']!='source'] if graph else []
        trace.append(dict(library=lib,case=case,control=control,seed=seed,status=s['status'],best_operations=best_ops,
            validation=(s.get('best') or {}).get('metrics',{}).get('objective'),cost=s.get('total_cost'),seconds=s.get('total_service_seconds')))
        cost=load(folder/'total_cost.json')
        for event in cost.get('events',[]):usage.append(dict(library=lib,case=case,control=control,seed=seed,**event))
    save(OUT/'search_costs_and_selection.json',trace);save(OUT/'registered_call_usage.json',usage)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='all':
        search();predict();score()
    else:{'freeze':freeze,'search':search,'predict':predict,'score':score}[stage]()
