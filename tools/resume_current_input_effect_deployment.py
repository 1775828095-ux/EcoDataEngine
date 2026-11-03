"""Uniform deployment continuation using the platform's saved rankings, no new search."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,traceback
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,field_registry
from ecodata.algorithms.field_mutation import GraphMutation
from ecodata.algorithms.search_accounting import SearchAccount
from ecodata.algorithms.current_input import deploy_ranked
from ecodata.algorithms.autonomous_decision import clean
from ecodata.algorithms.numerical_search import save
from run_current_input_effect_retest import ROOT,OLD,OUT,config,load,inputs,digest,recipe,cases,freeze
from pathlib import Path
import json

DEPLOYMENT=dict(objective_limit=10000,wall_seconds=3600)


def run_one(case,mode,seed):
    parent=OUT/'runs'/f'{case}_{mode}_{seed}';folder=OUT/'deployment_continuation'/parent.name
    if (folder/'receipt.json').exists():return load(folder/'receipt.json')
    folder.mkdir(parents=True,exist_ok=True);original=load(parent/'platform/summary.json')
    request=load(parent/'platform/request.json')['request'];parsed=inputs(config(case));reads=[];guard=dict(active=True)
    def audit(event,args):
        if event=='open' and guard['active'] and isinstance(args[0],(str,bytes,os.PathLike)):
            if Path(os.fsdecode(args[0])).name in {'test_labels.nc','validation_labels.nc','development_labels.nc'}:
                reads.append(str(args[0]));raise RuntimeError('External reference forbidden in deployment continuation')
    sys.addaudithook(audit)
    try:
        context,_,_=build_context(parsed,request);registry=field_registry(api.platform_registry(),context)
        # Restore only persisted rank inputs. Both rank() and deployment remain platform code.
        search=GraphMutation.__new__(GraphMutation)
        search.records=load(parent/'platform/graphs.json')
        rows=[json.loads(line) for line in (parent/'platform/attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        search.evaluated={row['key']:row for row in rows}
        search.objective_resolution=original['objective_tie_resolution']
        search.account=SearchAccount(DEPLOYMENT['objective_limit'],DEPLOYMENT['wall_seconds'])
        best,out,receipt=deploy_ranked(search,context,registry,request)
        product=None
        if best:
            product=folder/'product.nc';clean(out).to_netcdf(product,engine='h5netcdf')
            with xr.open_dataset(product) as f:xr.testing.assert_equal(out,f.load())
        result=dict(case=case,mode=mode,seed=seed,status=receipt['status'],product=str(product) if product else None,
            key=best['key'] if best else None,original_status=original['status'],deployment=receipt,
            cost=search.account.receipt(),external_reference_reads=reads,new_search_calls=0,
            ranking='Unchanged GraphMutation.rank on saved real numerical evaluations; confirmation and test unused')
    except Exception as exc:
        result=dict(case=case,mode=mode,seed=seed,status='CONTINUATION_FAILED',error=str(exc),trace=traceback.format_exc(),external_reference_reads=reads)
    finally:guard['active']=False
    save(folder/'receipt.json',result)
    return {k:result.get(k) for k in ['case','mode','seed','status','original_status']}


def main():
    freeze();assert (OUT/'search_complete.json').exists()
    amendment=load(OUT/'deployment_amendment.json');assert amendment['external_test_scores_seen'] is False
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_one,*job) for job in load(OUT/'planned_jobs.json')]
        for future in as_completed(futures):print('DEPLOY',json.dumps(future.result()),flush=True)
    entries=[];failures=[];original_failures=[]
    for case,mode,seed in load(OUT/'planned_jobs.json'):
        parent=OUT/'runs'/f'{case}_{mode}_{seed}';row=load(OUT/'deployment_continuation'/parent.name/'receipt.json')
        original=load(parent/'client_result.json')
        if original['status']!='PASS':original_failures.append(original)
        if row['status']!='PASS':failures.append(dict(case=case,method=f'{mode}_{seed}',stage='continuation',details=row));continue
        entries.append(dict(case=case,method=f'{mode}_{seed}',path=row['product'],sha256=digest(row['product']),
            graph=recipe(parent/'platform',row['key']),source='current_input_uniform_deployment_continuation'))
    for case in cases():
        cfg=config(case)
        for name,graph in load(OLD/'controls_frozen.json')['chosen'].items():
            folder=OUT/'fixed'/case/name
            try:
                receipt=load(folder/'deployment.json') if (folder/'deployment.json').exists() else api.execute_frozen_graph(inputs(cfg),cfg['request'],graph,folder)
                entries.append(dict(case=case,method=name,path=receipt['product'],sha256=digest(receipt['product']),graph=graph,source='2015_frozen_fixed'))
            except Exception as exc:
                row=dict(case=case,method=name,stage='fixed_deploy',error=str(exc),trace=traceback.format_exc());save(folder/'failure.json',row);failures.append(row)
    save(OUT/'original_budget_failures.json',original_failures);save(OUT/'failures.json',failures)
    assert not (OUT/'predictions_frozen.json').exists(),'Predictions already sealed'
    save(OUT/'predictions_frozen.json',entries)
    print('ALL PREDICTIONS SEALED',len(entries),'FINAL FAILURES',len(failures),'ORIGINAL BUDGET FAILURES',len(original_failures),'NO TEST SCORES READ',flush=True)


if __name__=='__main__':main()
