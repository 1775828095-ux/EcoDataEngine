"""Pre-outcome supplemental random arm matched to guided elapsed time."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,math,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_open_complementary import OUT,PROTOCOL,materialize
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save


def worker(index,seed):
    folder=OUT/'cost_matched'/(str(index)+'_'+str(seed))
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    guided=json.loads((OUT/'runs_v6/test'/(str(index)+'_'+str(seed))/'summary.json').read_text(encoding='utf-8'))
    _,request,_,inputs,_=materialize(PROTOCOL['test'][index]);request['search_policy']='random'
    budget=dict(objective_limit=10000,proposal_limit=100000,wall_seconds=math.ceil(guided['elapsed_seconds']))
    return api.submit_search(inputs,request,None,folder,mode='AutonomousField',budget=budget,seed=seed)


def main():
    note=OUT/'cost_match_protocol.json'
    if not note.exists():
        assert not (OUT/'effect_summary.json').exists(),'Supplement must be frozen before final scoring'
        save(note,dict(reason='Equal candidate executions gave unequal internal kernel cost',
            comparator='Random exploration, same code and data; wall limit = ceiling paired guided elapsed seconds',
            objective_limit=10000,proposal_limit=100000,scientific_objective_unchanged=True,
            outcome_not_inspected=True,worker_processes=3,global_optimum_claim=False))
    manifest=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(worker,i,s):(i,s) for i in range(len(PROTOCOL['test'])) for s in PROTOCOL['seeds']}
        for task in as_completed(futures):
            index,seed=futures[task];r=task.result()
            if not r['product']:raise ValueError('Failed cost-matched case '+str(index))
            path=Path(r['product'])
            manifest.append(dict(case=index,seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                seconds=r['elapsed_seconds'],kernel_calls=r['kernel_calls'],executions=r['executed_candidates']))
            print('COST_MATCHED',index,seed,r['executed_candidates'],r['elapsed_seconds'],flush=True)
    manifest.sort(key=lambda x:(x['case'],x['seed']))
    save(OUT/'cost_matched_predictions_frozen.json',manifest)


if __name__=='__main__':main()
