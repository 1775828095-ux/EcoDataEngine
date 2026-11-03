"""Full relation grammar and validation-driven public search; test is post-freeze."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys,shutil
import numpy as np
import xarray as xr
from scheduled_fusion_data import ROOT,case
sys.path.insert(0,str(ROOT/'src'))
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.algorithms.composition_decision import Composition
from ecodata.models import portrait_policy as policy
OUT=ROOT/'output/scheduled_fusion_20261025'
DEV=[80,440,800,1160];PAIRS=[(1510,1540),(1870,1900),(2230,2260),(2590,2620)]
SEEDS=[11,29,47,71,101];METHODS=['alns','basic','random','none']

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stores():
    for source,target in [(ar.ROOT/'complementary-state-v1.json',OUT/'attributes/complementary-state-v1.json'),(policy.ROOT/'complementary-wrf-v1.json',OUT/'models/complementary-wrf-v1.json')]:
        if source==target:continue
        if target.exists():assert source.read_bytes()==target.read_bytes()
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    ar.ROOT=OUT/'attributes';policy.ROOT=OUT/'models'

def stationary(plan):
    return all(next(p['choice'] for p in plan if p['window']=='first' and p['branch']==branch)==next(p['choice'] for p in plan if p['window']=='second' and p['branch']==branch) for branch in ('fine_slow','coarse_fast','fusion','post'))

def prepare():
    stores();counts=[]
    for phase,starts in [('development',DEV),('validation',[a for a,b in PAIRS]),('test',[b for a,b in PAIRS])]:
        for start in starts:
            folder=OUT/'cases'/str(start)
            if (folder/'config.json').exists():
                counts.append(load(folder/'domain.json'));continue
            inputs,req,truth=case(start)
            parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
            contracts=[engine.preflight(engine.decode(i)) for i in range(engine.count)]
            info=dict(case=start,phase=phase,grammar_count=engine.count,structural_legal=sum(r['status']!='REJECT' for r in contracts),
                structural_rejected=[dict(index=i,**r) for i,r in enumerate(contracts) if r['status']=='REJECT'],
                stationary_count=sum(stationary(engine.plan(engine.decode(i))) for i in range(engine.count)),
                axes=[dict(slot=s,alternatives=len(a)) for s,a in zip(engine.slots,engine.axes)],
                numerical_legal_count='PENDING_REAL_EXECUTION',numeric_distinct_count='PENDING_REAL_EXECUTION')
            assert engine.count==1296 and info['structural_legal']==1296 and info['stationary_count']==36
            files=[]
            for item in inputs:
                p=folder/item.filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(item.content)
                files.append(dict(path=str(p),role=item.role,sha256=sha(p)))
            truth.attrs['split']='test' if phase=='test' else 'search_validation'
            truth.to_dataset(name=req['variable']).to_netcdf(folder/'labels.nc',engine='h5netcdf')
            save(folder/'config.json',dict(inputs=files,request=req,phase=phase,labels_sha256=sha(folder/'labels.nc')))
            save(folder/'domain.json',info);counts.append(info)
    frozen=dict(version='scheduled-real-validation-v1',development=DEV,validation_test_pairs=PAIRS,seeds=SEEDS,
        methods=METHODS,objective='RMSE on separate search-validation labels; tests evaluated after all choices freeze',
        budget=dict(objective_limit=64,proposal_limit=256,wall_seconds=900),
        oracle_budget=dict(objective_limit=1296,proposal_limit=1,wall_seconds=3600),
        candidate_grammar='per window: fine temporal linear=1; coarse grid bilinear/PCHIP with optional median in either order=6; fusion A-dominant/B-dominant 0.75 or half-detail projection=3; post median off/on=2. Two independent windows: (1*6*3*2)^2=1296',
        candidate_scope='Representative executable finite grammar, not all imaginable library combinations; median can precede/follow spatial interpolation and follow fusion; no training or weights',
        data_scope='Real NCEP temperature controlled relative-resolution degradation; same-origin complementary sources, not independent sensors; 3 fine anchors at 48h, 17 coarse observations at 6h, reconstruct 14 fine fields',
        parent_data=dict(path=str(ROOT/'data/ncep_xarray/air_temperature.nc'),sha256=sha(ROOT/'data/ncep_xarray/air_temperature.nc')),
        controls='Best development fixed graph (all 1296), best development stationary graph (36), shared-seed basic, pure legal uniform random (ranking ignored), only tri-portrait initial pair',
        success='Within 1e-9 K of exhaustive validation optimum; first unique execution hitting that value; success@64 and censored execution cost. No guarantee of test optimum.',
        tuning_after_freeze=False,test_seen_at_freeze=False,domains=counts)
    target=OUT/'frozen_protocol.json'
    if target.exists():assert load(target)==json.loads(json.dumps(frozen))
    else:save(target,frozen)
    # Freeze the actual source before any experiment execution.
    sources=[ROOT/'src/ecodata/algorithms'/name for name in ('composition_decision.py','composition_optimizers.py','platform_search.py','spatial_median.py','spatiotemporal_detail.py')]
    sources += [ROOT/'tools'/name for name in ('scheduled_fusion_data.py','run_scheduled_fusion_study.py')]
    manifest=[]
    for source in sources:
        target=OUT/'source_frozen'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==source.read_bytes(),'Source changed after protocol freeze'
        else:shutil.copy2(source,target)
        manifest.append(dict(path=str(source),sha256=sha(source)))
    save(OUT/'source_manifest.json',manifest)
    print(json.dumps(dict(stage='PREPARED',per_task_grammar=1296,per_task_structural_legal=1296,stationary=36,tasks=len(counts)),ensure_ascii=False),flush=True)

def bundle(start):
    config=load(OUT/'cases'/str(start)/'config.json')
    inputs=[]
    for i in config['inputs']:
        p=Path(i['path']);assert sha(p)==i['sha256'];inputs.append(api.SearchInput(p.name,i['role'],p.read_bytes()))
    return inputs,deepcopy(config['request'])

def feedback(start):
    config=load(OUT/'cases'/str(start)/'config.json');assert config['phase']!='test'
    path=OUT/'cases'/str(start)/'labels.nc';assert sha(path)==config['labels_sha256']
    with xr.open_dataset(path,engine='h5netcdf') as ds:labels=ds[config['request']['variable']].load()
    return api.SearchValidation(labels,config['request']['variable'])

def run(start,backend,seed,*,exact=False):
    folder=OUT/('oracles' if exact else 'search')/str(start)/backend/str(seed)
    if (folder/'summary.json').exists():return load(folder/'summary.json')
    inputs,req=bundle(start);req['composition']['search'].update(backend=backend,exact_check=exact)
    if backend=='none':req['composition']['search']['generations']=0
    if backend=='random':req['composition']['search']['initialization']='uniform_control'
    frozen=load(OUT/'frozen_protocol.json')
    return api.submit_search(inputs,req,None,folder,mode='Profile',seed=seed,validation=feedback(start),budget=frozen['oracle_budget' if exact else 'budget'])

def test_product(start,plan,folder):
    # No search or validation callback; the already frozen graph is executed once.
    inputs,req=bundle(start);parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
    genome=[]
    for slot,axis in zip(engine.slots,engine.axes):
        choice=next(p['choice'] for p in plan if (p['field'],p['window'],p['branch'])==slot)
        genome.append(next(i for i,a in enumerate(axis) if a==choice))
    product,matching,trace,checks=engine.execute(genome)
    # Same NetCDF attribute normalization as the shared composition service.
    for obj in [product,*product.variables.values()]:
        obj.attrs={k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in obj.attrs.items() if v is not None}
    path=folder/'product.nc';folder.mkdir(parents=True,exist_ok=True);product.to_netcdf(path,engine='h5netcdf')
    with xr.open_dataset(OUT/'cases'/str(start)/'labels.nc',engine='h5netcdf') as ds:truth=ds[req['variable']].load()
    # This independent error is never returned to search.
    evaluation=api.DatasetReference(truth,req['variable'])(product)
    save(folder/'evaluation.json',dict(phase='test',graph_frozen=True,plan=plan,product=str(path),product_sha256=sha(path),metrics=evaluation,matching=matching,trace=trace,checks=checks))
    return evaluation

def main():
    if '--prepare' in sys.argv:prepare();return
    if (OUT/'results.json').exists():
        print(json.dumps(dict(stage='ALREADY_COMPLETE',summary=load(OUT/'results.json')['summary'])),flush=True);return
    stores();protocol=load(OUT/'frozen_protocol.json')
    for item in load(OUT/'source_manifest.json'):
        source=Path(item['path'])
        if sha(source)!=item['sha256']:
            repair=load(OUT/'orchestration_repair.json')
            assert source.name=='run_scheduled_fusion_study.py' and (OUT/'frozen_selections.json').exists()
            assert repair['frozen_sha256']==item['sha256'] and repair['released_sha256']==sha(source)
            assert all((OUT/'search'/str(a)/m/str(seed)/'summary.json').exists() for a,b in PAIRS for seed in SEEDS for m in METHODS)
    dev=[]
    for start in DEV:
        report=run(start,'none',11,exact=True)
        assert report['counts']['exact_count_complete']
        dev.append({r['key']:r for r in report['rows'] if r['status']=='PASS'})
        print(json.dumps(dict(stage='development_oracle',case=start,counts=report['counts'])),flush=True)
    common=set.intersection(*(set(d) for d in dev))
    fixed=min(common,key=lambda k:(np.mean([d[k]['search_validation']['RMSE'] for d in dev]),k))
    stationary_keys=[k for k in common if stationary(dev[0][k]['plan'])]
    fixed_stationary=min(stationary_keys,key=lambda k:(np.mean([d[k]['search_validation']['RMSE'] for d in dev]),k))
    save(OUT/'fixed_from_development.json',dict(key=fixed,stationary_key=fixed_stationary,plan=dev[0][fixed]['plan'],stationary_plan=dev[0][fixed_stationary]['plan'],development_only=True))
    selections=[];search_results=[]
    for validation_start,test_start in PAIRS:
        for seed in SEEDS:
            for backend in METHODS:
                report=run(validation_start,backend,seed)
                assert report['best'] and report['selection_objective']=='search_validation_RMSE'
                row=dict(validation_case=validation_start,test_case=test_start,seed=seed,backend=backend,key=report['recommended_key'],plan=report['best']['plan'],validation_rmse=report['final_validation_rmse'],executions=report['counts']['evaluated'],seconds=report['total_service_seconds'])
                selections.append(row)
        oracle=run(validation_start,'none',11,exact=True)
        optimum=oracle['final_validation_rmse']
        for row in [r for r in selections if r['validation_case']==validation_start]:
            folder=OUT/'search'/str(validation_start)/row['backend']/str(row['seed']);report=load(folder/'summary.json')
            passing=[(i+1,r) for i,r in enumerate(report['rows']) if r['status']=='PASS']
            hits=[(i,r) for i,r in passing if abs(r['search_validation']['RMSE']-optimum)<=1e-9]
            row.update(oracle_rmse=optimum,hit=bool(hits),first_hit_execution=hits[0][0] if hits else None,first_hit_seconds=hits[0][1]['completed_seconds'] if hits else None)
            search_results.append(row)
        print(json.dumps(dict(stage='validation_completed',case=validation_start,oracle_rmse=optimum,selections=len(selections))),flush=True)
    # All choices frozen across every case before opening any final-test labels.
    save(OUT/'frozen_selections.json',dict(test_labels_read=False,rows=selections,fixed=fixed,fixed_stationary=fixed_stationary))
    tested=[];cache={}
    for row in selections:
        key=(row['test_case'],row['key'])
        if key not in cache:cache[key]=test_product(row['test_case'],row['plan'],OUT/'test'/str(key[0])/key[1])
        tested.append(dict(row,test_rmse=cache[key]['RMSE']))
    baselines=[]
    for _,test_start in PAIRS:
        for name,k in [('fixed',fixed),('fixed_stationary',fixed_stationary)]:
            key=(test_start,k)
            if key not in cache:cache[key]=test_product(test_start,dev[0][k]['plan'],OUT/'test'/str(test_start)/k)
            baselines.append(dict(case=test_start,baseline=name,key=k,rmse=cache[key]['RMSE']))
    summary={}
    for backend in METHODS:
        rows=[r for r in tested if r['backend']==backend];hits=[r for r in rows if r['hit']]
        summary[backend]=dict(success_rate=len(hits)/len(rows),successes=len(hits),runs=len(rows),
            mean_test_rmse=float(np.mean([r['test_rmse'] for r in rows])),mean_validation_rmse=float(np.mean([r['validation_rmse'] for r in rows])),
            mean_executions=float(np.mean([r['executions'] for r in rows])),
            expected_evaluations_to_success=float(sum(r['first_hit_execution'] if r['hit'] else r['executions'] for r in rows)/len(hits)) if hits else None,
            median_hit_evaluations=float(np.median([r['first_hit_execution'] for r in hits])) if hits else None)
    save(OUT/'results.json',dict(protocol=protocol,summary=summary,runs=tested,baselines=baselines,
        scope='Finite grammar validation global optima, not guaranteed final-test optima; no independent sensor or WRF claim'))
    print(json.dumps(dict(stage='COMPLETE',summary=summary),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
