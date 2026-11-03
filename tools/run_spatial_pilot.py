"""Frozen exploratory ERA5 pilot; all graph execution uses the public platform."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from pyproj import Geod
import xarray as xr

from ecodata.algorithms.platform_search import DatasetReference, submit_search, SearchInput, make_context, train_profile_matching
from ecodata.algorithms.raster_reconstruction_task import degradation, edges


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/algorithm_library_expansion/v4/era5_bth_t2m_reference.nc'
BASES=['raster_task.spatial.nearest','raster_task.spatial.bilinear','raster_task.pchip','raster_task.spatial.xarray_cubic']
TAIL='raster_task.coarse_consistency'
MODEL='era5_spatial_pilot_20261023_v2'
SOURCES=['tools/run_spatial_pilot.py','src/ecodata/algorithms/platform_search.py','src/ecodata/algorithms/profile_matching.py',
    'src/ecodata/algorithms/portrait_rules.py','src/ecodata/algorithms/profile_pool.py',
    'src/ecodata/algorithms/raster_reconstruction_task.py','src/ecodata/interface/raster_upload.py',
    'src/ecodata/algorithms/open_source_spatial.py','src/ecodata/algorithms/golem_generation.py']


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def event(out,kind,**details):
    row=dict(time=datetime.now(timezone.utc).isoformat(),kind=kind,**details)
    with (out/'events.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def split(day):
    if day<=4:return 'fit'
    if day==5:return 'internal_holdout'
    if 7<=day<=9:return 'validation'
    if 11<=day<=14:return 'audit'
    return None


def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='spatial_pilot_v2',source=str(SOURCE),data_sha256=sha(SOURCE),
        source_sha256={f:sha(ROOT/f) for f in SOURCES},model_id=MODEL,variable='t2m',unit='K',
        graph_pool=BASES+[TAIL],scales=[2,3],target_shape=[24,24],hours=[0,12],
        split={str(day):split(day) for day in range(1,15)},
        primary='per-task spherical-cell-area weighted RMSE on fixed common coarse-centre interior; equal task averages',
        thresholds='high-temperature training-only pooled p90; no audit tuning',
        sampling='fixed hours and full region; no error-based selection or hidden-value filling',
        scenarios=dict(M1=dict(hard=[],H0=['raster_task.spatial.bilinear']),M2=dict(hard=['coarse_consistency'],H0=['raster_task.spatial.bilinear',TAIL])),
        strategies=['fixed','cost','rules','learned','random'],random_seeds=list(range(10)),
        inference_budget='top-1 choice frozen without current reference; shared eight-graph numerical matrix evaluated only afterwards',
        training='fit dates Aug1-4; Aug5 holdout diagnostics only; RF64/depth4/leaf2/seed17 from platform; no tuning',
        graph_budget=dict(objective_limit=8,wall_seconds=120),
        failures='retain; no replacement or gate relaxation; missing selections count failures, paired means on common successes disclosed',
        uncertainty='four audit date groups, paired day/scale effects; descriptive, no pixel-wise significance claim',
        evidence_scope='previously used ERA5 reanalysis; artificial degradation; exploratory, not untouched measured truth or simulator sensitivity',
        memory_scope='process RSS snapshots only, not measured per-kernel peak memory')
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol,'Frozen protocol/source changed'
    else:save(out/'protocol.json',protocol)
    return protocol


def prepare(out):
    manifest=out/'dataset_manifest.json'
    if manifest.exists():
        tasks=read(manifest)
        assert all(sha(t['coarse'])==t['coarse_sha256'] and sha(t['reference'])==t['reference_sha256'] for t in tasks)
        return tasks
    tasks=[];fit_values=[]
    with xr.open_dataset(SOURCE) as source:
        for index,timestamp in enumerate(pd.DatetimeIndex(source.time.values)):
            stage=split(timestamp.day)
            if stage is None or timestamp.hour not in {0,12}:continue
            fine=source.isel(time=index,drop=True)[['t2m']].transpose('lat','lon').reset_coords(drop=True).load()
            if not np.isfinite(fine.t2m).all():raise ValueError('Incomplete fixed reference; no target filling or replacement')
            if stage=='fit':fit_values.append(fine.t2m.values.copy())
            # Preserve the original file; only the experimental coarse product
            # receives an explicitly constructed average-support declaration.
            for scale in (2,3):
                token=timestamp.strftime('%Y%m%dT%H')+f'_x{scale}'
                refpath=out/'sealed_reference'/f'{token}.nc';refpath.parent.mkdir(parents=True,exist_ok=True)
                fine.to_netcdf(refpath,engine='h5netcdf')
                coarse,operator=degradation(fine,24//scale,'t2m')
                coarse.attrs=dict(crs='EPSG:4326',reference_kind='experimental average of ERA5 point-grid reference, not observed cell means',
                    original_spatial_semantics='UNKNOWN',degradation='spherical-area weighted discrete reference-cell averaging')
                coarse.t2m.attrs=dict(units='K',data_kind='continuous',standard_name='air_temperature',
                                      physical_variable='air_temperature',spatial_semantics='cell_mean')
                path=out/'uploads'/f'{token}.nc';path.parent.mkdir(parents=True,exist_ok=True);coarse.to_netcdf(path,engine='h5netcdf')
                request=dict(task='spatial',variable='t2m',unit='K',semantic_type='continuous',nonnegative=True,
                    spatial_semantics='cell_mean',target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),parameters={},
                    model_profile=dict(id='wrf_chem',role='wrf_initial_boundary',field='air_temperature'),
                    model_requirements=dict(priorities={'continuity':1},hard=[],evidence='Frozen experimental preparation task, not claimed WRF-Chem sensitivity'),
                    preferences=dict(weights={'RMSE':1},scales={'RMSE':1}),requirements=dict(try_coarse_consistency=True),
                    profile_options=dict(max_steps=2,candidate_limit=64,candidate_pool=dict(algorithms=BASES+[TAIL])))
                context,_,_=make_context([('primary',coarse)],request)
                delta=context.parameters['degradation_operator']-operator
                assert delta.nnz==0 or np.max(np.abs(delta.data))<1e-12
                save(out/'public_requests'/f'{token}.json',dict(inputs=[dict(path=str(path),role='primary')],request=request,mode='Profile'))
                tasks.append(dict(id=token,date=timestamp.strftime('%Y-%m-%d'),hour=timestamp.hour,scale=scale,split=stage,
                    coarse=str(path),reference=str(refpath),coarse_sha256=sha(path),reference_sha256=sha(refpath),
                    request=str(out/'public_requests'/f'{token}.json'),operator_parity=True))
    assert len(tasks)==48
    threshold=float(np.quantile(np.concatenate([v.ravel() for v in fit_values]),.9))
    save(out/'training_statistics.json',dict(high_temperature_p90_K=threshold,fit_dates=['2019-08-01','2019-08-02','2019-08-03','2019-08-04']))
    save(manifest,tasks);event(out,'DATA_PREPARATION_COMPLETE',tasks=len(tasks),role='preparation only; not recommendation')
    return tasks


def area_weights(lat,lon):
    return np.outer(np.diff(np.sin(np.deg2rad(edges(lat)))),np.deg2rad(np.diff(edges(lon))))


class AreaReference(DatasetReference):
    """Evaluation resource passed to the existing public execution service."""
    def __init__(self,reference,coarse,operator,threshold,stage):
        super().__init__(reference,'t2m')
        self.coarse=coarse;self.operator=operator;self.threshold=threshold
        self.weights=area_weights(reference.lat.values,reference.lon.values)
        self.interior=np.outer((reference.lat>=coarse.lat.min())&(reference.lat<=coarse.lat.max()),
                               (reference.lon>=coarse.lon.min())&(reference.lon<=coarse.lon.max()))
        self.protocol.update(objective='area_weighted_RMSE_common_interior_v1',evaluation_stage=stage,
            measure='spherical reference-cell area; fixed geometry mask; all reference cells finite',
            mask_sha256=hashlib.sha256(self.interior.tobytes()).hexdigest())
        lat=reference.lat.values;lon=reference.lon.values;geod=Geod(ellps='WGS84')
        yy,xx=np.meshgrid(lat,lon,indexing='ij')
        self.dy=geod.inv(xx[:-1],yy[:-1],xx[1:],yy[1:])[2]/1000
        self.dx=geod.inv(xx[:,:-1],yy[:,:-1],xx[:,1:],yy[:,1:])[2]/1000

    def __call__(self,ds):
        # The inherited checks enforce exact coordinates, units and a complete
        # reference cohort before any weighted or subgroup calculation.
        super().__call__(ds)
        p=ds.t2m.values;y=self.reference.values;d=p-y
        def average(values,mask):
            return float(np.sum(values[mask]*self.weights[mask])/self.weights[mask].sum()) if mask.any() else None
        def rmse(mask):
            value=average(d*d,mask);return float(np.sqrt(value)) if value is not None else None
        full=np.ones(y.shape,dtype=bool);mask=self.interior
        gradient=[];gradient_weights=[]
        for axis,dist in ((0,self.dy),(1,self.dx)):
            delta=np.diff(d,axis=axis)/dist
            valid=(mask[:-1]&mask[1:]) if axis==0 else (mask[:,:-1]&mask[:,1:])
            w=(self.weights[:-1]+self.weights[1:])/2 if axis==0 else (self.weights[:,:-1]+self.weights[:,1:])/2
            gradient.append(np.sum(delta[valid]**2*w[valid]));gradient_weights.append(w[valid].sum())
        residual=self.operator@p.ravel()-self.coarse.t2m.values.ravel()
        result=dict(objective=rmse(mask),RMSE=rmse(mask),MAE=average(abs(d),mask),Bias=average(d,mask),
            denominator=int(mask.sum()),accuracy_assessed=True,full_RMSE=rmse(full),boundary_RMSE=rmse(~mask),
            gradient_RMSE_K_per_km=float(np.sqrt(sum(gradient)/sum(gradient_weights))),
            high_temperature_RMSE=rmse(mask&(y>=self.threshold)),coarse_RMSE=float(np.sqrt(np.mean(residual**2))),
            coarse_max_abs_error=float(abs(residual).max()),coverage=float(np.isfinite(p).mean()),
            negative_fraction=float((p<0).mean()),range_violation_fraction=float(((p<self.coarse.t2m.min().item())|(p>self.coarse.t2m.max().item())).mean()))
        return result


def inputs(task):
    path=Path(task['coarse']);assert sha(path)==task['coarse_sha256']
    return [SearchInput(path.name,'primary',path.read_bytes())]


def matrix(out,task):
    folder=out/'runs'/task['id']
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    if task['split']=='audit':assert (out/'audit_choices_frozen.json').is_file()
    cfg=read(task['request']);request=deepcopy(cfg['request'])
    if task['split'] in {'fit','internal_holdout'}:
        request['profile_options'].update(evidence_role='development',group=task['date'])
    with xr.open_dataset(task['reference']) as source:fine=source.t2m.load()
    with xr.open_dataset(task['coarse']) as source:coarse=source.load()
    context,_,_=make_context([('primary',coarse)],request)
    evaluator=AreaReference(fine,coarse,context.parameters['degradation_operator'],read(out/'training_statistics.json')['high_temperature_p90_K'],task['split'])
    event(out,'REFERENCE_EVALUATION_OPEN',task=task['id'],stage=task['split'],choices_frozen=(out/'audit_choices_frozen.json').exists())
    clock=perf_counter();resources=[];process=psutil.Process()
    def progress(item):resources.append(dict(item,seconds=perf_counter()-clock,process_rss_bytes=process.memory_info().rss))
    result=submit_search(inputs(task),request,evaluator,folder,mode='Profile',budget=dict(objective_limit=8,wall_seconds=120),progress=progress)
    save(folder/'resource_snapshots.json',dict(scope='whole process RSS after each graph, not per-kernel peak',samples=resources))
    assert result['counts']['candidate_count']==8 and result['counts']['real_executions']==8
    event(out,'NUMERICAL_MATRIX_COMPLETE',task=task['id'],failures=result['counts']['quality_failures'])
    print('EXECUTED',task['id'],result['counts'],flush=True)
    return result


def recommendations(out,task,model):
    choices=[];cfg=read(task['request'])
    for scenario,hard in [('M1',[]),('M2',['coarse_consistency'])]:
        rankings={}
        for strategy in ['cost','rules','learned']:
            folder=out/'recommendations'/task['id']/scenario/strategy
            if (folder/'summary.json').exists():result=read(folder/'summary.json')
            else:
                request=deepcopy(cfg['request']);request['model_requirements']['hard']=hard
                if strategy=='cost':request['model_requirements']['priorities']={}
                if strategy=='learned':request['profile_options'].update(ranking_model_id=model,acknowledge_limited_development=True)
                result=submit_search(inputs(task),request,None,folder,mode='Profile')
            assert result['counts']['real_executions']==0 and result['best'] is None
            rankings[strategy]=result['recommendations']
            first=rankings[strategy][0]
            choices.append(dict(task=task['id'],scenario=scenario,strategy=strategy,seed=None,key=first['key'],steps=first['steps'],
                matching_status=result['matching_status'],learning_status=result['learning_status'],recommendation_seconds=result['seconds']))
        legal=sorted(rankings['cost'],key=lambda x:x['key'])
        wanted=['raster_task.spatial.bilinear']+([TAIL] if hard else [])
        fixed=next(c for c in legal if c['steps']==wanted)
        choices.append(dict(task=task['id'],scenario=scenario,strategy='fixed',seed=None,key=fixed['key'],steps=fixed['steps']))
        for seed in range(10):
            salt=int(hashlib.sha256(task['id'].encode()).hexdigest()[:8],16)
            picked=legal[int(np.random.default_rng(seed+salt).integers(len(legal)))]
            choices.append(dict(task=task['id'],scenario=scenario,strategy='random',seed=seed,key=picked['key'],steps=picked['steps']))
    return choices


def analyze(out,tasks,choices):
    graph_rows=[];strategy_rows=[];combo=[]
    for task in tasks:
        result=read(out/'runs'/task['id']/'summary.json');rows={r['key']:r for r in result['rows']}
        for row in rows.values():
            graph_rows.append(dict(task=task['id'],date=task['date'],split=task['split'],scale=task['scale'],
                method=' + '.join(row['steps']),status=row['status'],error=row.get('error'),product=row.get('product'),
                **{k:row.get(k) for k in ['RMSE','MAE','Bias','full_RMSE','boundary_RMSE','gradient_RMSE_K_per_km','high_temperature_RMSE',
                    'coarse_RMSE','coarse_max_abs_error','coverage','range_violation_fraction','seconds']}))
        for scenario in ['M1','M2']:
            legal=[r for r in rows.values() if scenario=='M1' or r['steps'][-1]==TAIL]
            valid=[r for r in legal if r['status']=='PASS']
            optimum=min((r['RMSE'] for r in valid),default=None)
            for choice in [c for c in choices if c['task']==task['id'] and c['scenario']==scenario]:
                row=rows[choice['key']]
                strategy_rows.append(dict(**choice,date=task['date'],split=task['split'],scale=task['scale'],status=row['status'],
                    RMSE=row.get('RMSE'),MAE=row.get('MAE'),Bias=row.get('Bias'),coarse_RMSE=row.get('coarse_RMSE'),
                    regret=row['RMSE']-optimum if row['status']=='PASS' and optimum is not None else None))
        for base in BASES:
            a=next(r for r in rows.values() if r['steps']==[base]);b=next(r for r in rows.values() if r['steps']==[base,TAIL])
            combo.append(dict(task=task['id'],date=task['date'],split=task['split'],scale=task['scale'],base=base,
                status='PASS' if a['status']==b['status']=='PASS' else 'FAILED_PAIR',
                base_RMSE=a.get('RMSE'),combination_RMSE=b.get('RMSE'),
                RMSE_gain=a['RMSE']-b['RMSE'] if a['status']==b['status']=='PASS' else None,
                coarse_RMSE_gain=a['coarse_RMSE']-b['coarse_RMSE'] if a['status']==b['status']=='PASS' else None))
    graphs=pd.DataFrame(graph_rows);strategies=pd.DataFrame(strategy_rows);combos=pd.DataFrame(combo)
    graphs.to_csv(out/'all_graph_results.csv',index=False);strategies.to_csv(out/'strategy_results.csv',index=False);combos.to_csv(out/'combination_results.csv',index=False)
    summaries=[]
    def mean(series):return float(series.mean()) if series.notna().any() else None
    for (stage,scenario,scale,strategy),frame in strategies.groupby(['split','scenario','scale','strategy']):
        summaries.append(dict(split=stage,scenario=scenario,scale=int(scale),strategy=strategy,
            n_tasks=int(frame.task.nunique()),n_records=len(frame),failures=int((frame.status!='PASS').sum()),
            mean_RMSE=mean(frame.RMSE),mean_regret=mean(frame.regret),
            learned_tasks=int((frame.matching_status=='LEARNED_DEVELOPMENT_RANK').sum()),
            chosen_methods=frame.steps.map(lambda x:' + '.join(x)).value_counts().to_dict()))
    save(out/'summary.json',dict(status='COMPLETED',graph_executions=len(graphs),graph_failures=int((graphs.status!='PASS').sum()),
        strategy_summaries=summaries,scope='Exploratory paired degradation study, four audit date groups; not independent measured-truth validation'))
    daily=strategies.groupby(['split','date','scenario','strategy']).RMSE.mean().reset_index()
    daily.to_csv(out/'daily_strategy_results.csv',index=False)
    return summaries


def run(out):
    started=perf_counter();freeze(out);tasks=prepare(out)
    for task in tasks:
        if task['split'] in {'fit','internal_holdout'}:matrix(out,task)
    model_path=ROOT/'output/plan_ranking_models'/MODEL
    if not model_path.exists():
        report=train_profile_matching([out/'runs'/t['id'] for t in tasks if t['split'] in {'fit','internal_holdout'}],MODEL,
            holdout_groups=['2019-08-05'])
        save(out/'training_report.json',report);event(out,'MODEL_TRAINED',model=MODEL)
    else:
        assert (out/'training_report.json').exists(),'Model ID already belongs to another run'
    all_choices=[]
    for task in tasks:
        if task['split']=='validation':
            all_choices+=recommendations(out,task,MODEL);matrix(out,task)
    save(out/'validation_choices.json',all_choices)
    audit=[]
    for task in tasks:
        if task['split']=='audit':audit+=recommendations(out,task,MODEL)
    if (out/'audit_choices_frozen.json').exists():assert read(out/'audit_choices_frozen.json')==audit
    else:save(out/'audit_choices_frozen.json',audit)
    event(out,'ALL_AUDIT_CHOICES_FROZEN',sha256=sha(out/'audit_choices_frozen.json'),choices=len(audit))
    for task in tasks:
        if task['split']=='audit':matrix(out,task)
    # Development choices are retrospective diagnostics, never audit evidence.
    for task in tasks:
        if task['split'] in {'fit','internal_holdout'}:all_choices+=recommendations(out,task,MODEL)
    all_choices+=audit
    summaries=analyze(out,tasks,all_choices)
    assert sha(SOURCE)==read(out/'protocol.json')['data_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in read(out/'protocol.json')['source_sha256'].items())
    event(out,'COMPLETE',seconds=perf_counter()-started,graph_executions=384)
    for row in summaries:
        if row['split']=='audit':print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output.resolve())
