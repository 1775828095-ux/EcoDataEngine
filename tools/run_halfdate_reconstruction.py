"""Frozen cohort client for the public platform, no experiment-local routing."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import json,hashlib,sys,traceback
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor,canonical
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.numerical_search import save
from prepare_halfdate_reconstruction import ROOT,OUT,load

BUDGET=dict(objective_limit=180,proposal_limit=40000,wall_seconds=300,probe_call_limit=60)
SEEDS=[11,29,47]


def cases():return [r['case'] for r in load(OUT/'prepared.json')['cases']]


def freeze():
    p=OUT/'source_manifest.json'
    if p.exists():
        for name,digest in load(p).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+name)
        return
    paths=list((ROOT/'src/ecodata').rglob('*.py'))+list((ROOT/'src/ecodata').rglob('*.json'))
    paths+=list((ROOT/'data/attribute_registries').glob('*.json'))
    paths+=[ROOT/'tools'/n for n in ['run_halfdate_reconstruction.py','prepare_halfdate_reconstruction.py','acquire_halfdate_reconstruction.py','audit_halfdate_reconstruction.py']]
    paths+=[ROOT/'docs/半日期隐藏时空重建_冻结协议_20261028.md']
    paths+=[ROOT/'tests'/n for n in ['test_raster_temporal_hidden.py','test_raster_gap_application.py','test_field_mutation.py','test_autonomous_field.py','test_registered_joint_scoring.py']]
    paths+=[ROOT/'pyproject.toml',ROOT/'requirements-basin-data.txt',ROOT/'requirements-lock.txt']
    manifest={}
    for source in paths:
        name=source.relative_to(ROOT).as_posix();dest=OUT/'source_frozen'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes());manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    save(p,manifest)
    save(OUT/'input_manifest.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'cases').rglob('*') if p.is_file()})
    save(OUT/'budget.json',BUDGET)
    import platform,importlib.metadata
    save(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
         packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','xarray','rasterio','h5netcdf','planetary-computer']}))
    raw=ROOT/'data/basin_ecology_2013_2017/raw_subsets'
    receipts=[p for group in ['landsat','modis_reflectance'] for p in (raw/group).glob('*/*.receipt.json')]
    save(OUT/'source_provenance.json',[dict(path=str(p.relative_to(ROOT)),receipt=load(p)) for p in receipts])


def config(case,phase):return load(OUT/'cases'/case/f'config_{phase}.json')


def inputs(cfg):return [api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in cfg['inputs']]


def bench(case,phase):
    cfg=config(case,phase);request,_=prepare_model_request(cfg['request']);context,_,_=build_context(inputs(cfg),request)
    return context,Executor(context,field_registry(api.platform_registry(),context))


def labels(case,phase):
    with xr.open_dataset(OUT/'cases'/case/(phase+'_labels.nc')) as ds:return ds.ndvi.load()


def recipe(folder,key):
    all_records=load(folder/'graphs.json');records={}
    def visit(k):
        if k in records:return
        for p in all_records[k]['parents']:visit(p)
        records[k]=all_records[k]
    visit(key)
    def identity(k):
        r=records[k];return [r['operation'],r['parameters'],[identity(p) for p in r['parents']]]
    return dict(records=records,key=key,identity=canonical(identity(key)))


def run(case,control,seed,phase):
    folder=OUT/'runs'/phase/f'{case}_{control}_{seed}'
    if (folder/'summary.json').exists():return load(folder/'summary.json')
    cfg=config(case,phase);cfg['request']['mutation_control']=control
    cfg.update(mode='PortraitMutation',seed=seed,budget=BUDGET,search_validation=dict(path=str(OUT/'cases'/case/(phase+'_labels.nc'))))
    save(folder/'client_config.json',cfg);save(folder/'metadata.json',dict(case=case,control=control,seed=seed,phase=phase))
    try:return api.submit_file_request(cfg,folder)
    except Exception as exc:
        row=dict(status='BLOCKED',error=str(exc),trace=traceback.format_exc());save(folder/'client_failure.json',row);return row


def search(phase):
    freeze();seeds=[704] if phase=='development' else SEEDS
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(run,c,control,s,phase):(c,control,s) for c in cases() for control in ['random','none'] for s in seeds}
        for future in as_completed(futures):
            r=future.result();print(*futures[future],r['status'],r.get('total_cost',{}).get('used'),
                (r.get('best') or {}).get('metrics',{}).get('objective'),flush=True)


def fixed_controls():
    freeze();items={}
    # All unique numerically eligible development graphs, not test-selected winners.
    for folder in (OUT/'runs/development').iterdir():
        path=folder/'attempts.jsonl'
        if not path.exists():continue
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['status']=='ELIGIBLE_OUTPUT':
                r=recipe(folder,row['key']);items[r['identity']]=r
    contexts={c:bench(c,'development') for c in cases()};truth={c:labels(c,'development') for c in cases()}
    rows=[]
    for n,item in enumerate(items.values()):
        scores={};errors={}
        for c,(ctx,executor) in contexts.items():
            try:
                output,_=executor.execute(item['records'],item['key']);scores[c]=measured(output,truth[c],ctx)['objective']
            except Exception as exc:scores[c]=None;errors[c]=str(exc)
        rows.append(dict(identity=item['identity'],scores=scores,errors=errors))
        if n%25==0:print('FIXED REPLAY',n,'/',len(items),flush=True)
    valid=[r for r in rows if all(v is not None for v in r['scores'].values())]
    if not valid:raise ValueError('No cross-case fixed development graph')
    resolution=max(np.finfo(float).eps*int(np.isfinite(a).sum())*max(1.,float(abs(a).max(skipna=True))) for a in truth.values())
    best=min(valid,key=lambda r:(round(np.mean(list(r['scores'].values()))/resolution),
         sum(v['operation']!='source' for v in items[r['identity']]['records'].values()),r['identity']))
    roots={f's{i}':dict(operation='source',parents=[],parameters=dict(source=f'source{i}')) for i in range(2)}
    def graph(chain,key):return dict(records={**roots,**chain},key=key)
    controls=dict(development_fixed=items[best['identity']])
    for method in ['linear','nearest']:
        controls['fine_time_'+method]=graph({'t':dict(operation='temporal.raster_'+method,parents=['s0'],parameters={})},'t')
    controls['coarse_only']=graph({'t':dict(operation='temporal.raster_linear',parents=['s1'],parameters={}),
        'u':dict(operation='spatial.bilinear',parents=['t'],parameters={})},'u')
    controls['anchor_residual']=graph({'f':dict(operation='fusion.visible_anchor_residual',parents=['s0','s1'],
        parameters=dict(fine_index=0,residual_strength=1.))},'f')
    controls['anchor_residual_physical']=graph({'f':dict(operation='fusion.visible_anchor_residual',parents=['s0','s1'],
        parameters=dict(fine_index=0,residual_strength=1.)),
        'range':dict(operation='constraint.declared_range_projection',parents=['f'],parameters={})},'range')
    save(OUT/'fixed_development_scores.json',rows)
    save(OUT/'controls_frozen.json',dict(chosen=controls,unique_discovered=len(items),cross_case_feasible=len(valid),
        selection='2015 only, all eligible discovered graphs, no exhaustive global claim',
        numerical_tie_resolution=resolution,
        calls=sum(e.calls for _,e in contexts.values()),test_used=False))
    print('FROZEN CONTROLS',len(items),len(valid),flush=True)


def smoke_controls():
    """Check every predefined control on development data before formal runs."""
    freeze();rows=[]
    for case in cases():
        cfg=config(case,'development');context,_=bench(case,'development')
        for name,graph in load(OUT/'controls_frozen.json')['chosen'].items():
            folder=OUT/'control_smoke'/case/name
            try:
                receipt=api.execute_frozen_graph(inputs(cfg),cfg['request'],graph,folder)
                with xr.open_dataset(receipt['product']) as ds:metrics=measured(ds.load(),labels(case,'development'),context)
                rows.append(dict(case=case,method=name,status='PASS',metrics=metrics,calls=receipt['cost']['used']))
            except Exception as exc:rows.append(dict(case=case,method=name,status='FAILED',error=str(exc)))
    save(OUT/'development_control_acceptance.json',rows)
    required=[r for r in rows if r['method'] in ['coarse_only','anchor_residual_physical','development_fixed']]
    if any(r['status']!='PASS' for r in required):raise ValueError('Required development controls fail; inspect metadata/contracts before formal experiment')
    print('DEVELOPMENT CONTROL ACCEPTANCE',sum(r['status']=='PASS' for r in rows),'/',len(rows),flush=True)


def predict():
    freeze()
    if (OUT/'predictions_frozen.json').exists():raise ValueError('Predictions already sealed')
    manifest=[];failures=[]
    for case in cases():
        candidates=[]
        for control in ['random','none']:
            for seed in SEEDS:
                folder=OUT/'runs/validation'/f'{case}_{control}_{seed}';summary=load(folder/'summary.json')
                if summary['best'] is None:
                    failures.append(dict(case=case,method=f'{control}_{seed}',reason=summary['status']));continue
                candidates.append((f'{control}_{seed}',recipe(folder,summary['best']['key'])))
        candidates+=list(load(OUT/'controls_frozen.json')['chosen'].items())
        cfg=config(case,'test')
        for method,graph in candidates:
            folder=OUT/'deployments'/case/method
            try:
                if (folder/'deployment.json').exists():receipt=load(folder/'deployment.json')
                else:receipt=api.execute_frozen_graph(inputs(cfg),cfg['request'],graph,folder)
                path=Path(receipt['product']);manifest.append(dict(case=case,method=method,path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),graph=graph))
            except Exception as exc:
                row=dict(case=case,method=method,error=str(exc),trace=traceback.format_exc())
                save(folder/'failure.json',row);failures.append(row)
        print('DEPLOYED',case,flush=True)
    save(OUT/'deployment_failures.json',failures)
    save(OUT/'predictions_frozen.json',manifest)
    print('SEALED',len(manifest),'FAILED',len(failures),'TEST NOT OPENED',flush=True)


def score():
    freeze();rows=[]
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        c=entry['case'];ctx,_=bench(c,'test');truth=labels(c,'test')
        with xr.open_dataset(path) as ds:output=ds.load()
        try:metric=measured(output,truth,ctx);status='PASS'
        except Exception as exc:metric=dict(error=str(exc));status='FAILED_OUTPUT'
        rows.append(dict(case=c,method=entry['method'],status=status,**metric))
    results={}
    for method in ['random','none',*load(OUT/'controls_frozen.json')['chosen']]:
        selected=[r for r in rows if (r['method'].startswith(method+'_') if method in ['random','none'] else r['method']==method)]
        expected=18 if method in ['random','none'] else 6
        if len(selected)!=expected or any(r['status']!='PASS' for r in selected):
            results[method]=dict(status='INCOMPLETE',observed=len(selected),expected=expected);continue
        by_case={c:float(np.mean([r['objective'] for r in selected if r['case']==c])) for c in cases()}
        results[method]=dict(status='PASS',RMSE=float(np.mean(list(by_case.values()))),by_case=by_case,
            MAE=float(np.mean([r['MAE'] for r in selected])),Bias=float(np.mean([r['Bias'] for r in selected])),
            gradient_RMSE=float(np.mean([r['gradient_RMSE'] for r in selected])))
    save(OUT/'test_metrics.json',rows);save(OUT/'effect_summary.json',results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage in ['development','validation']:search(stage)
    else:{'controls':fixed_controls,'smoke':smoke_controls,'predict':predict,'score':score}[stage]()
