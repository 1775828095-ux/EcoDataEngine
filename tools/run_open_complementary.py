"""Prospective complementary-field study through the public platform entry."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from scheduled_fusion_data import case
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor
from ecodata.algorithms.numerical_search import save

OUT=ROOT/'output/open_complementary_20261027'
PROTOCOL=dict(version='open-complementary-v6',
    development=[dict(start=s,lat=[0,8],lon=[0,8]) for s in [40,400]],
    validation=[dict(start=s,lat=[8,16],lon=[24,32]) for s in [1500,1850]],
    test=[dict(start=s,lat=[16,24],lon=[40,48]) for s in [2450,2600,2750,2850]],
    seeds=[11,29,47],budget=dict(objective_limit=180,proposal_limit=40000,wall_seconds=120),
    design='Existing NCEP controlled same-origin 2x spatial and 8x temporal degradation, not independent sensors',
    data='fine anchors 0,8,16; coarse frames all 17 at 6h; 8x8 fine and 4x4 coarse grids',
    self_validation='Withhold middle fine anchor; search observes only endpoint fine anchors and all coarse frames',
    test_labels='Hidden fine frames excluding all supplied anchors; opened only after selection manifest freezes',
    primary='Mean per-frame spatial RMSE across case blocks; paired effects, each case one cluster',
    controls=['fine temporal-only','coarse spatial-only','development strongest discovered fixed graph','random discovery-order candidate'],
    no_fixed_graph_pool=True,no_manual_max_depth=True,
    scientific_limits='Small controlled benchmark; previous use of parent NCEP dataset precludes claiming untouched external data',
    success='Fusion improves both one-source baselines; selection benefit separately compared with development strongest fixed; no global certificate')


def materialize(config):
    inputs,old,truth=case(config['start'],lat_slice=tuple(config['lat']),lon_slice=tuple(config['lon']))
    request={k:v for k,v in old.items() if k not in ['composition','scoring','portrait_policy','model_profile']}
    request['complementary_observation_operator']=True
    fine=xr.open_dataset(__import__('io').BytesIO(inputs[0].content)).load()
    coarse=xr.open_dataset(__import__('io').BytesIO(inputs[1].content)).load()
    request['target_times']=[str(t) for t in coarse.time.values]
    validation=fine.air_temperature.isel(time=[1]).copy();validation.attrs['split']='search_validation'
    search_fine=fine.isel(time=[0,2])
    search=[api.SearchInput('fine.nc','primary',search_fine.to_netcdf()),inputs[1]]
    return search,request,validation,inputs,truth


def run_one(phase,index,seed,policy='contract_guided'):
    config=PROTOCOL[phase][index];folder=OUT/'runs_v6'/phase/(str(index)+'_'+str(seed)+('_random' if policy=='random' else ''))
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    _,request,_,inputs,_=materialize(config);request['search_policy']=policy
    return api.submit_search(inputs,request,None,folder,mode='AutonomousField',seed=seed,budget=PROTOCOL['budget'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=OUT/'protocol_v6.json'
    if frozen.exists():assert json.loads(frozen.read_text(encoding='utf-8'))==PROTOCOL
    else:save(frozen,PROTOCOL)
    save(OUT/'data_lineage.json',dict(parent='data/ncep_xarray/air_temperature.nc',
        sha256=hashlib.sha256((ROOT/'data/ncep_xarray/air_temperature.nc').read_bytes()).hexdigest(),
        truthful_scope='Relative fine/coarse controlled observations, fine native 2.5 degrees; no native sensor independence'))
    phase=sys.argv[1] if len(sys.argv)>1 else 'development'
    if phase=='prepare':
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=3) as pool:
            tasks={pool.submit(run_one,stage,index,seed):(stage,index,seed)
                for stage in ('development','validation') for index in range(len(PROTOCOL[stage])) for seed in PROTOCOL['seeds']}
            for task in as_completed(tasks):
                result=task.result()
                print(*tasks[task],result['status'],result['executed_candidates'],
                    result['best']['metrics']['objective'] if result['best'] else None,flush=True)
        return
    index=int(sys.argv[2]) if len(sys.argv)>2 else 0
    seed=int(sys.argv[3]) if len(sys.argv)>3 else 11
    result=run_one(phase,index,seed)
    print(json.dumps({k:v for k,v in result.items() if k not in ['best','best_single']},ensure_ascii=False),flush=True)
    print('BEST',json.dumps(result.get('best',{}),default=str)[:800],flush=True)


if __name__=='__main__':main()
