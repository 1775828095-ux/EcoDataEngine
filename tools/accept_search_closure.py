"""One-call public-service acceptance: no external deployment continuation."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import json,sys,hashlib,traceback
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save
from ecodata.algorithms.autonomous_field import build_context,measured
from ecodata.algorithms.profile_matching import prepare_model_request
from run_halfdate_reconstruction import recipe

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'output/halfdate_reconstruction_20261028'
OUT=ROOT/'output/autonomous_search_closure_20261029';RUN=OUT/'acceptance';SEEDS=[11,29,47]


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cases():return [r['case'] for r in load(OLD/'prepared.json')['cases']]
def cfg(case):return load(OLD/'cases'/case/'config_test.json')
def inputs(c):return [api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes(),{}) for i in c['inputs']]
def request(c):return dict(c['request'],purpose='same_variable',allow_training=False,allow_existing_weights=False)


def freeze():
    RUN.mkdir(exist_ok=True)
    files=list((ROOT/'src/ecodata').rglob('*.py'))+list((ROOT/'src/ecodata').rglob('*.json'))
    files+=list((ROOT/'data/attribute_registries').glob('*.json'))
    files+=[ROOT/'pyproject.toml',Path(__file__),ROOT/'tools/calibrate_search_closure.py',ROOT/'tools/verify_closure_page.py',ROOT/'tests/test_search_closure.py']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    path=RUN/'source_manifest.json'
    if path.exists():assert hashes==load(path),'Frozen source changed';return
    for p in files:
        target=RUN/'source_frozen'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    save(path,hashes)
    save(RUN/'input_manifest.json',{p.relative_to(OLD).as_posix():digest(p) for p in (OLD/'cases').rglob('*') if p.is_file()})
    jobs=[(c,m,s,0) for c in cases() for m in ['random','none'] for s in SEEDS]+[(c,'random',s,1) for c in cases() for s in SEEDS]
    save(RUN/'planned_jobs.json',jobs)
    save(RUN/'protocol.json',dict(cases=cases(),seeds=SEEDS,jobs=54,independent_main_jobs=36,duplicate_repro_jobs=18,
        policy=load(ROOT/'src/ecodata/algorithms/search_stop_policy.json'),budget=api.CURRENT_INPUT_BUDGET,
        calibrated_on='2015 visible data and internal-mask sensitivity only',test_year=2018,historical_test_already_seen=True,
        test='Seal every main/repeat/page product before numerical test-reference evaluation; no tuning after',
        service='Public entry performs initialization, qualified checkpoint, random mutation, calibrated stopping and final deployment in one call',
        repeats='Identical input bytes, goals, version, seed, limits and policy; compare graph ID, stop reason and all numeric products',
        primary='Hidden-date RMSE, then equal case/seed weighting; identical finite reference support; failures not dropped',
        baselines='Same public self-validation/no-mutation ablation; original 2015 frozen fixed controls',
        deployment='Inside the public service, separate bounded allowance recorded in the same ledger; no experiment-side continuation',
        inference='Descriptive retrospective same-data integration acceptance; no global-optimum or isolated stopping-only effect claim'))


def job(case,mode,seed,repeat):
    folder=RUN/'runs'/f'{case}_{mode}_{seed}_{repeat}'
    if (folder/'receipt.json').exists():return load(folder/'receipt.json')
    c=cfg(case);q=request(c);args=dict(inputs=c['inputs'],request=q,seed=seed,budget=api.CURRENT_INPUT_BUDGET)
    if mode=='random':args['decision_policy']='current_input'
    else:q['mutation_control']='none';args['mode']='PortraitMutation'
    save(folder/'client_config.json',args);reads=[];active=[True]
    def audit(event,values):
        if event=='open' and active[0] and isinstance(values[0],(str,bytes,os.PathLike)):
            if Path(os.fsdecode(values[0])).name in {'test_labels.nc','validation_labels.nc','development_labels.nc'}:
                reads.append(str(values[0]));raise RuntimeError('External truth forbidden during decisions')
    sys.addaudithook(audit)
    try:
        result=api.submit_file_request(args,folder/'platform')
        row=dict(case=case,mode=mode,seed=seed,repeat=repeat,status=result['status'],product=result.get('product'),
            key=(result.get('best') or {}).get('key'),stop_reason=result['stop_reason'],external_reads=reads,
            seconds=result['total_cost']['service_seconds'],calls=result['total_cost']['used'],
            policy=result['search_stopping'],manual_deployment_used=False)
    except Exception as e:row=dict(case=case,mode=mode,seed=seed,repeat=repeat,status='CLIENT_FAILED',error=str(e),trace=traceback.format_exc(),external_reads=reads)
    finally:active[0]=False
    save(folder/'receipt.json',row)
    return {k:row.get(k) for k in ['case','mode','seed','repeat','status','stop_reason','seconds']}


def run():
    freeze()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(job,*j) for j in load(RUN/'planned_jobs.json')]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    entries=[];failures=[];repeats=[]
    for c,m,s,repetition in load(RUN/'planned_jobs.json'):
        p=RUN/'runs'/f'{c}_{m}_{s}_{repetition}';row=load(p/'receipt.json')
        if repetition:
            original=load(RUN/'runs'/f'{c}_{m}_{s}_0/receipt.json')
            if row['status']==original['status']=='PASS':
                assert row['key']==original['key'] and row['stop_reason']==original['stop_reason']
                assert row['policy']['new_numerical_evaluations']==original['policy']['new_numerical_evaluations']
                bound=load(RUN/'runs'/f'{c}_{m}_{s}_0/platform/summary.json')['objective_tie_resolution']
                with xr.open_dataset(row['product']) as a,xr.open_dataset(original['product']) as b:
                    x,y=a.load(),b.load();xr.testing.assert_equal(x.coords.to_dataset(),y.coords.to_dataset())
                    xr.testing.assert_allclose(x,y,rtol=0.,atol=bound)
                    difference=float(np.max(abs(x.ndvi.values-y.ndvi.values)))
                repeats.append(dict(case=c,seed=s,status='PASS',graph_and_numeric_product_equal=True,
                    exact_values=difference==0,max_abs_difference=difference,preexisting_roundoff_bound=bound))
            else:repeats.append(dict(case=c,seed=s,status='FAILED',original=original['status'],repeat=row['status']))
            continue
        if row['status']!='PASS':failures.append(row);continue
        entries.append(dict(case=c,method=f'{m}_{s}',path=row['product'],sha256=digest(row['product']),graph=recipe(p/'platform',row['key'])))
    for c in cases():
        data=cfg(c)
        for method,graph in load(OLD/'controls_frozen.json')['chosen'].items():
            p=RUN/'fixed'/c/method
            try:
                r=load(p/'deployment.json') if (p/'deployment.json').exists() else api.execute_frozen_graph(inputs(data),request(data),graph,p)
                entries.append(dict(case=c,method=method,path=r['product'],sha256=digest(r['product']),graph=graph))
            except Exception as e:failures.append(dict(case=c,method=method,status='FIXED_FAILED',error=str(e)))
    save(RUN/'failures.json',failures);save(RUN/'reproducibility.json',repeats)
    save(RUN/'predictions_frozen.json',entries)
    print('SEALED',len(entries),'PRODUCTS; REPEATS',len(repeats),'NO TEST VALUES READ',flush=True)


def score():
    freeze();entries=load(RUN/'predictions_frozen.json');rows=[];checks=[]
    for e in entries:
        assert digest(e['path'])==e['sha256'];data=cfg(e['case']);q,_=prepare_model_request(request(data));ctx,_,_=build_context(inputs(data),q)
        with xr.open_dataset(OLD/'cases'/e['case']/'test_labels.nc') as f:truth=f.ndvi.load()
        with xr.open_dataset(e['path']) as f:out=f.load()
        metric=measured(out,truth,ctx)
        pred=out.ndvi.sel(time=truth.time,lat=truth.lat,lon=truth.lon).values.astype(float);ref=truth.values.astype(float);mask=np.isfinite(ref)
        assert np.isfinite(pred[mask]).all()
        independent=float(np.sqrt(np.nanmean(np.where(mask,(pred-ref)**2,np.nan),axis=(1,2))).mean())
        np.testing.assert_allclose(independent,metric['objective'],rtol=1e-12,atol=1e-12)
        with xr.open_dataset(data['inputs'][0]['path']) as f:visible=f.ndvi.load()
        known=np.isfinite(visible.values);np.testing.assert_array_equal(out.ndvi.sel(time=visible.time).values[known],visible.values[known])
        checks.append(dict(case=e['case'],method=e['method'],independent_objective=independent,support=int(mask.sum()),observations_preserved=True))
        rows.append(dict(case=e['case'],method=e['method'],**metric))
    arms=['random','none',*load(OLD/'controls_frozen.json')['chosen']];effects={}
    for m in arms:
        selected=[r for r in rows if (r['method'].startswith(m+'_') if m in ['random','none'] else r['method']==m)]
        count=18 if m in ['random','none'] else 6
        if len(selected)!=count:effects[m]=dict(status='INCOMPLETE',passed=len(selected),expected=count);continue
        by_case={c:float(np.mean([r['objective'] for r in selected if r['case']==c])) for c in cases()}
        effects[m]=dict(status='PASS',RMSE=float(np.mean(list(by_case.values()))),by_case=by_case)
    save(RUN/'test_metrics.json',rows);save(RUN/'effect_summary.json',effects);save(RUN/'independent_metrics.json',checks)
    print(json.dumps(effects,indent=2))


def audit():
    freeze();runs=[];masks={}
    for c,m,s,repetition in load(RUN/'planned_jobs.json'):
        p=RUN/'runs'/f'{c}_{m}_{s}_{repetition}';r=load(p/'receipt.json');assert r['status']=='PASS',r
        summary=load(p/'platform/summary.json');q=load(p/'platform/request.json');cost=load(p/'platform/total_cost.json')
        assert r['external_reads']==[] and not r['manual_deployment_used']
        assert summary['checkpoint_available'] and summary['current_input_deployment']['status']=='PASS'
        assert summary['optimizer']['backend']==m
        assert cost['used']==len(cost['events'])<=cost['limit'] and cost['search_limit']==1800 and cost['deployment_limit']==1800
        assert summary['search_stopping']['policy']['calibration_data']=='2015 only'
        assert summary['confirmation']['used_for_selection'] is False
        with xr.open_dataset(p/'platform/self_validation_input.nc') as f:masked=f.ndvi.load()
        with xr.open_dataset(p/'platform/self_validation_labels.nc') as f:search=f.ndvi.load()
        with xr.open_dataset(p/'platform/self_confirmation_labels.nc') as f:confirm=f.ndvi.load()
        left=np.isfinite(search.reindex(time=masked.time).values);right=np.isfinite(confirm.reindex(time=masked.time).values)
        assert not (left&right).any() and np.isnan(masked.values[left|right]).all()
        signature=hashlib.sha256(left.tobytes()+right.tobytes()).hexdigest()
        if c in masks:assert masks[c]==signature
        masks[c]=signature
        runs.append(dict(case=c,mode=m,seed=s,repeat=repetition,seconds=cost['service_seconds'],calls=cost['used'],
            reason=summary['stop_reason'],duplicates=summary['duplicate_proposals_skipped'],
            checkpoint_fallback=summary['current_input_deployment'].get('checkpoint_fallback',False)))
    for p,h in load(RUN/'input_manifest.json').items():assert digest(OLD/p)==h
    repeats=load(RUN/'reproducibility.json');assert len(repeats)==18 and all(r['status']=='PASS' for r in repeats)
    page=load(OUT/'page_acceptance/acceptance.json');assert page['status']=='PASS' and page['numeric_products_equal'] and not page['manual_deployment_used']
    main=[r for r in runs if r['repeat']==0]
    save(RUN/'audit.json',dict(status='PASS',jobs=len(runs),main_jobs=len(main),numeric_repeat_pairs=len(repeats),page_form_and_api_equal=True,
        manual_deployments=0,external_truth_decision_reads=0,source_inputs_unchanged=True,
        main_costs={m:dict(seconds=sum(r['seconds'] for r in main if r['mode']==m),calls=sum(r['calls'] for r in main if r['mode']==m),
            median_seconds=float(np.median([r['seconds'] for r in main if r['mode']==m])),max_seconds=max(r['seconds'] for r in main if r['mode']==m)) for m in ['random','none']},
        stopping_reasons=dict(Counter(r['reason'] for r in main if r['mode']=='random')),duplicate_checks_skipped=sum(r['duplicates'] for r in main),
        independent_metric_checks=len(load(RUN/'independent_metrics.json')),failures=load(RUN/'failures.json')))
    save(RUN/'runs_audit.json',runs);print('AUDIT PASS: 54 ONE-CALL RUNS, 18 REPEAT PAIRS, ACTUAL PAGE FORM, NO MANUAL DEPLOYMENT')


if __name__=='__main__':{'freeze':freeze,'run':run,'score':score,'audit':audit}[sys.argv[1]]()
