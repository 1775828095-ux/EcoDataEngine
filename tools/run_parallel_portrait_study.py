"""Frozen multi-region study through the public service, with no test tuning."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,shutil,sys,time

import numpy as np
import xarray as xr

from scheduled_fusion_data import ROOT,case
sys.path.insert(0,str(ROOT/'src'))
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.algorithms.composition_decision import Composition
from ecodata.models import portrait_policy as policy
from run_scheduled_fusion_study import load,save,sha,stationary

OUT=ROOT/'output/parallel_portrait_20261025'
REGIONS=[dict(id='south_west',lat=[0,8],lon=[0,8]),dict(id='north_west',lat=[12,20],lon=[0,8]),
    dict(id='south_east',lat=[0,8],lon=[32,40]),dict(id='north_east',lat=[12,20],lon=[32,40])]
DEV_TIMES=[120,600,900,1250]
PAIRS=[(1600,1670),(2400,2470)]
SEEDS=[11,29,47]
METHODS=['full','no_data','no_model','no_collaboration','unscored_alns','random','random_cost_allowance','score_only']


def stores():
    for module,version,folder in [(ar,'complementary-state-v1','attributes'),(policy,'complementary-wrf-v1','models')]:
        target=OUT/folder/(version+'.json');source=module.ROOT/(version+'.json')
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        module.ROOT=OUT/folder


def specs():
    dev=[dict(id=f'dev_{r["id"]}_{t}',region=r,start=t,phase='development') for r in REGIONS[:2] for t in DEV_TIMES]
    val=[dict(id=f'val_{r["id"]}_{a}',region=r,start=a,phase='validation',test_id=f'test_{r["id"]}_{b}') for r in REGIONS for a,b in PAIRS]
    test=[dict(id=f'test_{r["id"]}_{b}',region=r,start=b,phase='test') for r in REGIONS for a,b in PAIRS]
    return dev,val,test


def bundle(identifier):
    conf=load(OUT/'cases'/identifier/'config.json')
    inputs=[]
    for item in conf['inputs']:
        path=Path(item['path']);assert sha(path)==item['sha256']
        inputs.append(api.SearchInput(path.name,item['role'],path.read_bytes()))
    return inputs,deepcopy(conf['request'])


def validation(identifier):
    conf=load(OUT/'cases'/identifier/'config.json');assert conf['phase']!='test'
    path=OUT/'cases'/identifier/'labels.nc';assert sha(path)==conf['labels_sha256']
    with xr.open_dataset(path,engine='h5netcdf') as ds:labels=ds.air_temperature.load()
    return api.SearchValidation(labels,'air_temperature')


def prepare():
    stores();dev,val,test=specs()
    protocol=dict(version='three-parallel-visible-mask-v1',data_source=str(ROOT/'data/ncep_xarray/air_temperature.nc'),
        data_sha256=sha(ROOT/'data/ncep_xarray/air_temperature.nc'),regions=REGIONS,development=dev,validation=val,test=test,
        design='8 development blocks in two western regions; 8 validation/later-test pairs across four disjoint regions and two dates; eastern regions held out spatially from calibration',
        scope='Same-origin controlled relative-resolution NCEP air temperature, not independent sensors or measured WRF response; 8-day blocks, five fine 48h anchors and coarse 6h observations; 28 hidden fine targets',
        grammar='36 actual local policies in each of two windows: 1296 legal structures; local probes 36*2=72 actual graph executions at masked visible anchors',
        scoring=dict(data='1/(1+masked-visible-anchor RMSE / spatial SD of masked visible anchor)',
            model='(1 + existing scenario-weighted risk reduction)/2; empirical visible coarse/detail state, explicit engineering preferences',
            collaboration='(1+(best constituent RMSE - combination RMSE)/(best constituent RMSE + combination RMSE))/2; 0/0 defined as zero gain',
            unknown='[0,1] and provenance; never impute missing state effects; local evidence does not certify cross-window synergy',
            normalization='Routes in [0,1]; conservative lower bound; equal window importance; final joint weights fit only on eight development cases',
            weight_candidates='full: equal or permutations of (.5,.25,.25); ablations: remaining pair (.25,.75),(.5,.5),(.75,.25); no outcome-driven expansion'),
        controls=METHODS,seeds=SEEDS,search_budget=dict(objective_limit=32,proposal_limit=128,wall_seconds=900),
        random_cost_allowance='104 full-target executions instead of 32, granting random the count of 72 extra local probes; local and full costs differ, report kernels and time separately',
        fixed='Mean-development best among all 1296 whole graphs, plus strongest stationary among 36, chosen before validation/test',
        primary='Independent test RMSE and relative regret vs development fixed, averaged per case across seeds; eight case pairs, not seeds as datasets',
        secondary='Validation optimum success@32, score discrimination, development/validation score-error rank correlation, real kernel and execution counts',
        uncertainty='Case-level paired bootstrap descriptive 95% intervals; shared dates/regions reduce independence; no universal significance claim',
        freeze='Freeze model/weights/code before validation, all selections before any test evaluation; retain failed seeds and negatives; no post-test repair of scientific scoring')
    path=OUT/'protocol.json'
    if path.exists():assert load(path)==protocol
    else:save(path,protocol)
    for item in dev+val+test:
        folder=OUT/'cases'/item['id']
        if (folder/'config.json').exists():continue
        r=item['region'];inputs,req,truth=case(item['start'],length=33,lat_slice=r['lat'],lon_slice=r['lon'])
        req['composition']['parallel_scoring']=dict(weights=dict(data=1,model=1,collaboration=1),
            evidence='Equal initial engineering priorities; study calibration only on development, not measured WRF sensitivity')
        parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
        assert engine.count==1296
        assert all(engine.preflight(engine.decode(i))['status']!='REJECT' for i in range(engine.count))
        files=[]
        for inp in inputs:
            p=folder/inp.filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(inp.content)
            files.append(dict(path=str(p),role=inp.role,sha256=sha(p)))
        truth.attrs['split']='test' if item['phase']=='test' else 'search_validation'
        truth.to_dataset(name='air_temperature').to_netcdf(folder/'labels.nc',engine='h5netcdf')
        save(folder/'config.json',dict(item,request=req,inputs=files,labels_sha256=sha(folder/'labels.nc')))
    files=[ROOT/'src/ecodata/algorithms'/n for n in ('composition_scoring.py','composition_decision.py','composition_optimizers.py','spatiotemporal_detail.py','platform_search.py')]
    files += [Path(__file__),ROOT/'tools/scheduled_fusion_data.py']
    files += [ROOT/'src/ecodata/interface/search_page.py']
    manifest=[]
    for path in files:
        target=OUT/'source_frozen'/path.relative_to(ROOT)
        if target.exists():assert sha(path)==sha(target),'Code changed after scientific freeze'
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        manifest.append(dict(path=str(path),sha256=sha(path)))
    save(OUT/'source_manifest.json',manifest)
    print(json.dumps(dict(stage='PREPARED',development=len(dev),validation=len(val),test=len(test))),flush=True)


def oracle(identifier):
    stores();folder=OUT/'oracles'/identifier
    if (folder/'summary.json').exists():return identifier
    inputs,req=bundle(identifier);req['composition']['search'].update(backend='none',generations=0,exact_check=True)
    api.submit_search(inputs,req,None,folder,mode='Profile',validation=validation(identifier),
        budget=dict(objective_limit=1296,proposal_limit=1,wall_seconds=3600))
    return identifier


def weights_for(method):
    return load(OUT/'calibration.json')['weights'][method]


def calibrate():
    dev,_,_=specs();tables={};fixed_losses={};stationary_keys=None
    candidates=dict(full=[dict(zip(('data','model','collaboration'),v)) for v in [(1/3,1/3,1/3),(.5,.25,.25),(.25,.5,.25),(.25,.25,.5)]])
    for removed in ('data','model','collaboration'):
        kept=[r for r in ('data','model','collaboration') if r!=removed]
        candidates['no_'+removed]=[{removed:0,kept[0]:a,kept[1]:1-a} for a in (.25,.5,.75)]
    for item in dev:
        d=load(OUT/'oracles'/item['id']/'summary.json');assert d['counts']['evaluated']==1296
        scores=load(OUT/'oracles'/item['id']/'initial_screening.json')['rows']
        by_index={r['index']:r for r in d['rows'] if r['status']=='PASS'}
        tables[item['id']]=(scores,by_index)
        for row in by_index.values():fixed_losses.setdefault(row['key'],[]).append(row['search_validation']['RMSE'])
        if stationary_keys is None:stationary_keys={r['key'] for r in by_index.values() if stationary(r['plan'])}
    fixed=min(fixed_losses,key=lambda k:(np.mean(fixed_losses[k]),k))
    fixed_stationary=min(stationary_keys,key=lambda k:(np.mean(fixed_losses[k]),k))
    candidate_results={};weights={};loo={}
    for method,options in candidates.items():
        records=[]
        for w in options:
            losses={}
            for name,(scores,by_index) in tables.items():
                ranked=sorted(scores,key=lambda r:(-sum(w[k]*r['routes'][k][0] for k in w),r['scientific_steps'],r['index']))
                # Same initial pair as production; minimum training error in pair.
                chosen=min(by_index[r['index']]['search_validation']['RMSE'] for r in ranked[:2])
                best=min(r['search_validation']['RMSE'] for r in by_index.values())
                losses[name]=chosen/best-1
            records.append(dict(weights=w,relative_regret_by_group=losses,mean_regret=float(np.mean(list(losses.values())))))
        winner=min(range(len(records)),key=lambda i:(records[i]['mean_regret'],i));weights[method]=records[winner]['weights']
        candidate_results[method]=records
        loo[method]=[]
        for name in tables:
            index=min(range(len(records)),key=lambda i:(np.mean([v for k,v in records[i]['relative_regret_by_group'].items() if k!=name]),i))
            loo[method].append(dict(held_out=name,weights=records[index]['weights'],regret=records[index]['relative_regret_by_group'][name]))
    first=next(iter(tables.values()))[1]
    keyplans={r['key']:r['plan'] for r in first.values()}
    save(OUT/'calibration.json',dict(training_partition='development',weights=weights,candidate_results=candidate_results,
        leave_one_case_out=loo,fixed=fixed,fixed_plan=keyplans[fixed],fixed_stationary=fixed_stationary,
        stationary_plan=keyplans[fixed_stationary],training_groups=list(tables),test_used=False,
        scope='Small declared simplex grid calibration of three interpretable routes, not a replacement ranker; LOO descriptive only'))


def search_job(job):
    stores();identifier,method,seed=job;folder=OUT/'search'/identifier/method/str(seed)
    if (folder/'summary.json').exists():return job
    inputs,req=bundle(identifier);budget=load(OUT/'protocol.json')['search_budget']
    options=req['composition']['search'];options.update(backend='alns',population=4,generations=100)
    if method in {'unscored_alns','random','random_cost_allowance'}:
        req['composition'].pop('parallel_scoring');options['initialization']='uniform_control'
        if method.startswith('random'):options['backend']='random'
        if method=='random_cost_allowance':budget=dict(budget,objective_limit=104,proposal_limit=256)
    else:
        w=weights_for('full' if method=='score_only' else method)
        req['composition']['parallel_scoring']=dict(weights=w,evidence='Frozen eight-development-case simplex calibration: '+sha(OUT/'calibration.json'))
        if method=='score_only':options.update(backend='none',generations=0,survivors=1)
    api.submit_search(inputs,req,None,folder,mode='Profile',validation=validation(identifier),seed=seed,budget=budget)
    return job


def parallel(function,jobs,workers=3):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(function,job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(dict(stage=function.__name__,completed=future.result()),ensure_ascii=False),flush=True)


def test_product(identifier,plan,folder):
    if (folder/'result.json').exists():return load(folder/'result.json')
    inputs,req=bundle(identifier);req['composition'].pop('parallel_scoring')
    parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
    genome=[next(i for i,a in enumerate(axis) if a==next(p['choice'] for p in plan if (p['field'],p['window'],p['branch'])==slot)) for slot,axis in zip(engine.slots,engine.axes)]
    ds,matching,trace,checks=engine.execute(genome)
    for obj in [ds,*ds.variables.values()]:obj.attrs={k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in obj.attrs.items() if v is not None}
    folder.mkdir(parents=True,exist_ok=True);path=folder/'product.nc';ds.to_netcdf(path,engine='h5netcdf')
    with xr.open_dataset(OUT/'cases'/identifier/'labels.nc',engine='h5netcdf') as raw:truth=raw.air_temperature.load()
    result=dict(metrics=api.DatasetReference(truth,'air_temperature')(ds),plan=plan,product=str(path),sha256=sha(path),checks=checks)
    save(folder/'result.json',result);return result


def finish():
    stores();_,val,_=specs();rows=[]
    for case_spec in val:
        oracle_data=load(OUT/'oracles'/case_spec['id']/'summary.json');optimum=oracle_data['final_validation_rmse']
        for method in METHODS:
            for seed in ([SEEDS[0]] if method=='score_only' else SEEDS):
                folder=OUT/'search'/case_spec['id']/method/str(seed);d=load(folder/'summary.json')
                best=d['best'];assert best is not None
                hits=[i+1 for i,r in enumerate(d['rows']) if r['status']=='PASS' and abs(r['search_validation']['RMSE']-optimum)<=1e-9]
                rows.append(dict(case=case_spec['id'],test_id=case_spec['test_id'],method=method,seed=seed,
                    key=best['key'],plan=best['plan'],validation_rmse=d['final_validation_rmse'],optimum=optimum,
                    hit=bool(hits),first_hit=hits[0] if hits else None,executions=d['counts']['evaluated'],
                    probe_executions=d.get('parallel_scoring',{}).get('probe_graph_executions',0),
                    kernel_calls=sum(t.get('kernel_calls',0) for r in d['rows'] for t in r.get('trace',[]))+d.get('parallel_scoring',{}).get('probe_kernel_calls',0),
                    seconds=d['total_service_seconds']))
    frozen=OUT/'frozen_selections.json';cal=load(OUT/'calibration.json')
    choices=dict(rows=rows,fixed=cal['fixed'],fixed_plan=cal['fixed_plan'],fixed_stationary=cal['fixed_stationary'],stationary_plan=cal['stationary_plan'],test_labels_read=False)
    if frozen.exists():assert load(frozen)==choices
    else:save(frozen,choices)
    # No test-label read is permitted above this point.
    tested=[];baselines=[]
    for row in rows:
        result=test_product(row['test_id'],row['plan'],OUT/'test'/row['test_id']/row['key'])
        tested.append(dict(row,test_rmse=result['metrics']['RMSE']))
    for item in val:
        for name,key,plan in [('fixed',cal['fixed'],cal['fixed_plan']),('fixed_stationary',cal['fixed_stationary'],cal['stationary_plan'])]:
            result=test_product(item['test_id'],plan,OUT/'test'/item['test_id']/key)
            baselines.append(dict(case=item['id'],test_id=item['test_id'],method=name,rmse=result['metrics']['RMSE'],key=key))
    summary={}
    fixed={r['case']:r['rmse'] for r in baselines if r['method']=='fixed'}
    for method in METHODS:
        chosen=[r for r in tested if r['method']==method]
        cases={i['id']:float(np.mean([r['test_rmse'] for r in chosen if r['case']==i['id']])) for i in val}
        differences=np.array([cases[k]-fixed[k] for k in cases]);rng=np.random.default_rng(20261025)
        interval=np.quantile(rng.choice(differences,(10000,len(differences)),replace=True).mean(axis=1),[.025,.975]).tolist()
        summary[method]=dict(mean_test_rmse=float(np.mean(list(cases.values()))),case_rmse=cases,
            mean_difference_vs_fixed_K=float(differences.mean()),descriptive_case_bootstrap_95=interval,
            validation_hits=sum(r['hit'] for r in chosen),runs=len(chosen),
            mean_probe_graphs=float(np.mean([r['probe_executions'] for r in chosen])),
            mean_objective_graphs=float(np.mean([r['executions'] for r in chosen])),
            mean_kernel_calls=float(np.mean([r['kernel_calls'] for r in chosen])))
    save(OUT/'results.json',dict(summary=summary,runs=tested,baselines=baselines,
        mean_fixed_rmse=float(np.mean(list(fixed.values()))),test_choices_frozen=True,
        scope='Eight paired space-time cases; development-only weight fit; engineering model preferences, controlled same-source data; all negatives retained'))
    print(json.dumps(dict(stage='RESULTS',summary=summary),ensure_ascii=False),flush=True)


def main():
    phase=sys.argv[1] if len(sys.argv)>1 else 'all'
    if phase=='prepare':prepare();return
    stores();dev,val,_=specs()
    if phase in {'development','all'}:parallel(oracle,[x['id'] for x in dev]);calibrate()
    if phase in {'validation','all'}:
        assert (OUT/'calibration.json').exists()
        jobs=[(x['id'],m,s) for x in val for m in METHODS for s in ([SEEDS[0]] if m=='score_only' else SEEDS)]
        parallel(search_job,jobs);parallel(oracle,[x['id'] for x in val])
    if phase in {'finish','all'}:finish()


if __name__=='__main__':main()
