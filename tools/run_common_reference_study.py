"""Development-frozen common reference, budget-prefix mutation study, fresh tests."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json,shutil,sys

import numpy as np
import xarray as xr

from scheduled_fusion_data import ROOT,case
import run_parallel_portrait_study as old
from run_scheduled_fusion_study import load,save,sha
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.composition_decision import Composition
from ecodata.algorithms.composition_scoring import reference_scope,gain_priority

OUT=ROOT/'output/common_reference_20261026'
SEEDS=[11,29,47,71,101];BUDGETS=[32,128,256]
METHODS=['common','legacy','uniform_alns','random']


def specs():
    return [dict(id=f'{r["id"]}_{a}',region=r,start=a,test_start=b) for r in old.REGIONS[2:] for a,b in [(2000,2070),(2700,2770)]]


def bundle(identifier,test=False):
    c=load(OUT/'cases'/identifier/('test_config.json' if test else 'config.json'))
    inputs=[]
    for x in c['inputs']:
        p=Path(x['path']);assert sha(p)==x['sha256'];inputs.append(api.SearchInput(p.name,x['role'],p.read_bytes()))
    return inputs,deepcopy(c['request'])


def feedback(identifier):
    p=OUT/'cases'/identifier/'labels.nc'
    with xr.open_dataset(p,engine='h5netcdf') as d:a=d.air_temperature.load()
    return api.SearchValidation(a,'air_temperature')


def restrict_request(req,plan):
    req=deepcopy(req);req['composition'].pop('parallel_scoring',None)
    def choice(branch):return next(p['choice'] for p in plan if p['branch']==branch)
    def stages(chain):return [dict(id='frozen_'+str(i),after=['frozen_'+str(i-1)] if i else [],options=[step],evidence='Frozen reference order') for i,step in enumerate(chain)]
    req['composition']['source_stages']={k:stages(choice(k)) for k in req['composition']['source_stages']}
    req['composition']['post_stages']=stages(choice('post'))
    req['composition']['fusion_options']=[choice('fusion')]
    return req


def prepare():
    old.stores();cal=load(old.OUT/'calibration.json');dev=old.specs()[0]
    # Retain the independently audited strongest development fixed combination.
    fixed=cal['fixed_plan'];single_candidates={};single_losses={}
    for spec in dev:
        inputs,req=old.bundle(spec['id']);parsed,_=api.parse_inputs(inputs)
        base=Composition(parsed,req,api.platform_registry())
        for name,source,grid in [('fine_linear','fine_slow','raster_task.pchip'),('coarse_bilinear','coarse_fast','raster_task.spatial.bilinear'),('coarse_pchip','coarse_fast','raster_task.pchip')]:
            plan=deepcopy(fixed)
            for p in plan:
                if p['branch']=='fusion':p['choice']=dict(algorithm='prepare.select_source',source=source,evidence='Single-algorithm development baseline')
                elif p['branch']=='post':p['choice']=[]
                elif p['branch']=='coarse_fast':p['choice']=[dict(algorithm=grid,parameters={})]
            folder=OUT/'single_development'/spec['id']/name
            if not (folder/'result.json').exists():
                engine=Composition(parsed,restrict_request(req,plan),api.platform_registry())
                ds,_,trace,checks=engine.execute(engine.decode(0))
                metric=old.validation(spec['id'])(ds)
                for obj in [ds,*ds.variables.values()]:obj.attrs={k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in obj.attrs.items() if v is not None}
                folder.mkdir(parents=True,exist_ok=True);ds.to_netcdf(folder/'product.nc',engine='h5netcdf')
                save(folder/'result.json',dict(metrics=metric,sha256=sha(folder/'product.nc'),trace=trace,checks=checks))
            single_candidates[name]=plan;single_losses.setdefault(name,[]).append(load(folder/'result.json')['metrics']['RMSE'])
    best_single=min(single_losses,key=lambda k:(np.mean(single_losses[k]),k))
    profile=dict(training_partition='development',test_used=False,training_groups=[x['id'] for x in dev],
        scope=reference_scope(base),plans=dict(fixed=fixed,single=single_candidates[best_single]),
        selected_single=best_single,single_development_losses=single_losses,
        fixed_provenance=dict(path=str(old.OUT/'calibration.json'),sha256=sha(old.OUT/'calibration.json'),fixed_key=cal['fixed']),
        evidence='Strongest development single algorithm and all-1296 development-fixed combination; no validation/test choice')
    if (OUT/'common_reference.json').exists():assert load(OUT/'common_reference.json')==profile
    else:save(OUT/'common_reference.json',profile)
    protocol=dict(version='common-reference-budget-v1',cases=specs(),seeds=SEEDS,budget_prefixes=BUDGETS,
        methods=METHODS,maximum_budget=dict(objective_limit=256,proposal_limit=4096,wall_seconds=1800),
        scope='Four fresh paired blocks in two eastern regions not used in weight training; same-origin controlled NCEP temperature, not independent sensors/WRF response',
        scoring='Common reference compared on same visible masked anchors; primary .5 data + .5 model, C is secondary gain gate/tiebreak only, never additive',
        stability='positive paired gain in both observed windows; descriptive, not confidence; absent support remains UNKNOWN',
        baseline_profile_sha256=sha(OUT/'common_reference.json'),
        strategy='One bounded mutation trajectory per seed; analyse nested 32/128/256 unique-execution prefixes; no exact_check, no optimum-aware stopping',
        fixed='Development-frozen fixed plan; test evaluation after all methods/prefix choices freeze',
        optimum_scope='Only exact finite-domain validation RMSE optimum; oracle separately executed and never passed to search',
        initial_vs_mutation='first hit <=2 is initial selection, >2 mutation; report both; failures and actual early stops retained',
        previous_tests='Older studied cases are not reused as fresh test; new starts 2000/2700, final dates2070/2770',
        training_provenance=dict(path=str(old.OUT/'calibration.json'),sha256=sha(old.OUT/'calibration.json')))
    if (OUT/'protocol.json').exists():assert load(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    for item in specs():
        for test in [False,True]:
            folder=OUT/'cases'/item['id'];config=folder/('test_config.json' if test else 'config.json')
            if config.exists():continue
            r=item['region'];inputs,req,truth=case(item['test_start'] if test else item['start'],length=33,lat_slice=r['lat'],lon_slice=r['lon'])
            files=[]
            for x in inputs:
                p=folder/(('test_' if test else '')+x.filename);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(x.content)
                files.append(dict(path=str(p),role=x.role,sha256=sha(p)))
            truth.attrs['split']='test' if test else 'search_validation'
            label=folder/('test_labels.nc' if test else 'labels.nc');truth.to_dataset().to_netcdf(label,engine='h5netcdf')
            save(config,dict(inputs=files,request=req,labels_sha256=sha(label),phase='test' if test else 'validation'))
    source=[ROOT/'src/ecodata/algorithms'/n for n in ['composition_scoring.py','composition_decision.py','composition_optimizers.py','platform_search.py']]+[Path(__file__)]
    manifest=[]
    for p in source:
        q=OUT/'source_frozen'/p.relative_to(ROOT)
        if q.exists():assert sha(q)==sha(p)
        else:q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        manifest.append(dict(path=str(p),sha256=sha(p)))
    save(OUT/'source_manifest.json',manifest)
    print(json.dumps(dict(stage='PREPARED',cases=4,methods=4,seeds=5,maximum_budget=256,best_single=best_single)),flush=True)


def search_job(job):
    old.stores();identifier,method,seed=job;folder=OUT/'search'/identifier/method/str(seed)
    if (folder/'summary.json').exists():return job
    inputs,req=bundle(identifier);req['composition']['search'].update(backend='alns',population=64,generations=100,exact_check=False)
    if method=='common':
        req['composition']['parallel_scoring']=dict(weights=dict(data=.5,model=.5,collaboration=0),evidence='Frozen development-supported primary; common gain secondary only',
            common_reference=dict(path=str(OUT/'common_reference.json'),sha256=sha(OUT/'common_reference.json')))
    elif method=='legacy':req['composition']['parallel_scoring']=dict(weights=dict(data=.5,model=.25,collaboration=.25),evidence='Previous frozen legacy control')
    else:
        req['composition']['search']['initialization']='uniform_control'
        if method=='random':req['composition']['search']['backend']='random'
    api.submit_search(inputs,req,None,folder,mode='Profile',validation=feedback(identifier),seed=seed,budget=load(OUT/'protocol.json')['maximum_budget'])
    return job


def oracle_job(identifier):
    old.stores();folder=OUT/'oracles'/identifier
    if (folder/'summary.json').exists():return identifier
    inputs,req=bundle(identifier);req['composition']['search'].update(backend='none',generations=0,exact_check=True)
    api.submit_search(inputs,req,None,folder,mode='Profile',validation=feedback(identifier),budget=dict(objective_limit=1296,proposal_limit=1,wall_seconds=1800))
    return identifier


def parallel(fn,jobs):
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(fn,x) for x in jobs]):print(json.dumps(dict(stage=fn.__name__,completed=f.result())),flush=True)


def freeze():
    rows=[]
    for item in specs():
        for method in METHODS:
            for seed in SEEDS:
                d=load(OUT/'search'/item['id']/method/str(seed)/'summary.json')
                assert not d['exact_audit_additional_executions'] and d['certification']=='NOT_CERTIFIED'
                for budget in BUDGETS:
                    passing=[r for r in d['rows'][:budget] if r['status']=='PASS']
                    best=min(passing,key=lambda r:(round(r['search_validation']['RMSE'],9),-round(r['matching']['score_interval'][0],9),*[-v for v in gain_priority(r['matching'])],r['scientific_steps'],r['index']))
                    rows.append(dict(case=item['id'],method=method,seed=seed,budget=budget,actual_executions=min(budget,len(d['rows'])),
                        key=best['key'],plan=best['plan'],validation_rmse=best['search_validation']['RMSE']))
    save(OUT/'frozen_selections.json',dict(rows=rows,test_labels_read=False,oracle_used_to_choose=False))


def test_product(identifier,plan):
    inputs,req=bundle(identifier,test=True);parsed,_=api.parse_inputs(inputs)
    engine=Composition(parsed,req,api.platform_registry())
    genome=[next(i for i,a in enumerate(axis) if a==next(p['choice'] for p in plan if (p['field'],p['window'],p['branch'])==slot)) for slot,axis in zip(engine.slots,engine.axes)]
    ds,_,_,checks=engine.execute(genome);key=__import__('ecodata.algorithms.attribute_registry',fromlist=['digest']).digest(plan)
    folder=OUT/'test'/identifier/key;folder.mkdir(parents=True,exist_ok=True)
    for obj in [ds,*ds.variables.values()]:obj.attrs={k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in obj.attrs.items() if v is not None}
    ds.to_netcdf(folder/'product.nc',engine='h5netcdf')
    with xr.open_dataset(OUT/'cases'/identifier/'test_labels.nc',engine='h5netcdf') as raw:truth=raw.air_temperature.load()
    metrics=api.DatasetReference(truth,'air_temperature')(ds)
    save(folder/'result.json',dict(metrics=metrics,plan=plan,sha256=sha(folder/'product.nc'),checks=checks))
    return metrics


def finish():
    old.stores();frozen=load(OUT/'frozen_selections.json');assert not frozen['test_labels_read']
    rows=[];cache={}
    for row in frozen['rows']:
        d=load(OUT/'search'/row['case']/row['method']/str(row['seed'])/'summary.json')
        oracle=load(OUT/'oracles'/row['case']/'summary.json');opt=oracle['final_validation_rmse']
        hits=[i+1 for i,r in enumerate(d['rows'][:row['budget']]) if r['status']=='PASS' and abs(r['search_validation']['RMSE']-opt)<=1e-9]
        key=(row['case'],row['key'])
        if key not in cache:cache[key]=test_product(row['case'],row['plan'])
        rows.append(dict(row,oracle_rmse=opt,first_hit=hits[0] if hits else None,
            initial_hit=bool(hits and hits[0]<=2),mutation_hit=bool(hits and hits[0]>2),hit=bool(hits),test_rmse=cache[key]['RMSE']))
    profile=load(OUT/'common_reference.json');fixed=[]
    for spec in specs():
        metric=test_product(spec['id'],profile['plans']['fixed']);fixed.append(dict(case=spec['id'],test_rmse=metric['RMSE']))
    summary={}
    for method in METHODS:
        summary[method]={}
        for budget in BUDGETS:
            group=[r for r in rows if r['method']==method and r['budget']==budget]
            summary[method][str(budget)]=dict(hits=sum(x['hit'] for x in group),runs=len(group),initial_hits=sum(x['initial_hit'] for x in group),
                mutation_hits=sum(x['mutation_hit'] for x in group),mean_test_rmse=float(np.mean([x['test_rmse'] for x in group])),
                incomplete_budget=sum(x['actual_executions']<budget for x in group))
    save(OUT/'results.json',dict(summary=summary,rows=rows,fixed=fixed,mean_fixed_rmse=float(np.mean([r['test_rmse'] for r in fixed])),
        scope='Four new paired cases x five seeds; 20 runs not 20 independent datasets; prefixes are nested, not independent repeats; no exhaustive completion in mutation runs'))
    print(json.dumps(dict(stage='RESULTS',summary=summary),ensure_ascii=False),flush=True)


if __name__=='__main__':
    phase=sys.argv[1]
    if phase=='prepare':prepare()
    elif phase=='search':parallel(search_job,[(x['id'],m,s) for x in specs() for m in METHODS for s in SEEDS]);freeze()
    elif phase=='oracles':parallel(oracle_job,[x['id'] for x in specs()])
    elif phase=='finish':finish()
