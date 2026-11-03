"""Prospective development iterations; sealed audit opened only after selection."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.platform_search import DatasetReference,SearchInput,submit_search,make_context,train_profile_matching
from ecodata.algorithms.raster_reconstruction_task import degradation,edges

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/spatial_next_20261023'
SOURCE=ROOT/'data/ncep_xarray/air_temperature.nc'
BASES=['raster_task.spatial.nearest','raster_task.spatial.bilinear','raster_task.pchip','raster_task.spatial.xarray_cubic',
       'raster_task.fusion.idw','raster_task.station.metpy_linear','raster_task.station.pykrige_ordinary']
TAIL='raster_task.coarse_consistency'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def event(kind,**details):
    with (OUT/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),kind=kind,**details))+'\n')


def protocol():
    p=OUT/'protocol.json'
    if p.exists():assert read(p)['data_sha256']==sha(SOURCE);return
    save(p,dict(source=str(SOURCE),data_sha256=sha(SOURCE),source_url='https://github.com/pydata/xarray-data/blob/master/air_temperature.nc',
        provenance='NCEP/NMC reanalysis air temperature at sigma=.995, not ERA5 2m temperature or station truth',
        train='2013 all months days1/15 at00; enhancement uses days1/8/15/22 at00/12, chosen before validation',
        calibration='2014 Jan-Feb day1 at00, never fitted',validation='2014 Mar-Jun day1 at00; development-only iteration',
        audit='2014 Jul-Dec day1 at00; opened once after model/strategy freeze; no post-audit tuning',
        regions='two disjoint 24x24 longitude tiles, increasing lat; same date grouped across regions/scales',scales=[2,3],
        pool=BASES+[TAIL],graphs=14,metric='spherical-cell-area RMSE on common coarse-centre interior',
        scenarios=['M1: no hard coarse consistency','M2: hard coarse consistency'],
        controls=['each fixed graph','development-best fixed per scenario/scale/region','cost','rules','uniform random seeds0..9','learning'],
        validation_goals=dict(learning_coverage_min=.80,RMSE_ratio_vs_rules_max=.95,RMSE_ratio_vs_random_max=.95,
            RMSE_ratio_vs_development_fixed_max=1.02),
        adapt='If development validation fails: diagnose support and choices, enhance development coverage; then test reuse of existing rank head or relative-gain learner if needed. Never relax supports or choose audit winners.',
        iteration_scope='all attempted validation variants retained; stop only on measured acceptable evidence or explicit unresolved limitation',
        scientific_scope='exploratory degraded reanalysis, six monthly audit date groups; no fine physical truth or simulator-benefit claim'))


def prepare(enhanced=False,extras=None,manifest_name=None):
    manifest=OUT/(manifest_name or ('manifest_enhanced.json' if enhanced else 'manifest_initial.json'))
    if manifest.exists():return read(manifest)
    tasks=[]
    with xr.open_dataset(SOURCE) as raw:
        for timestamp in pd.DatetimeIndex(raw.time.values):
            train=timestamp.year==2013 and timestamp.day in ({1,8,15,22} if enhanced else {1,15}) and timestamp.hour in ({0,12} if enhanced else {0})
            regular=train;stamp=timestamp.strftime('%Y%m%dT%H')
            if extras and any(item[0]==stamp for item in extras):train=True
            if not train and not(timestamp.year==2014 and timestamp.day==1 and timestamp.hour==0):continue
            stage='fit' if train else 'calibration' if timestamp.month<=2 else 'validation' if timestamp.month<=6 else 'audit'
            for region,start in [('west',0),('east',29)]:
                fine=raw.sel(time=timestamp).isel(lat=slice(0,24),lon=slice(start,start+24)).sortby('lat')[['air']].reset_coords(drop=True).load()
                fine=fine.assign_coords(lon=fine.lon.values-360).rename(air='temperature')
                fine.attrs=dict(crs='EPSG:4326',source='NCEP sigma995 reference grid')
                fine.temperature.attrs=dict(units='K',standard_name='air_temperature',data_kind='continuous',spatial_semantics='point')
                assert np.isfinite(fine.temperature).all()
                for scale in [2,3]:
                    if train and not regular and (stamp,region,scale) not in extras:continue
                    token=timestamp.strftime('%Y%m%dT%H')+f'_{region}_x{scale}'
                    coarse,operator=degradation(fine,24//scale,'temperature')
                    coarse.temperature.attrs['spatial_semantics']='cell_mean'
                    coarse.attrs.update(average_support='experimental discrete reference-cell average, not observed cell means')
                    cp=OUT/'uploads'/f'{token}.nc';rp=OUT/'sealed_reference'/f'{token}.nc'
                    if not cp.exists():cp.parent.mkdir(parents=True,exist_ok=True);coarse.to_netcdf(cp,engine='h5netcdf')
                    if not rp.exists():rp.parent.mkdir(parents=True,exist_ok=True);fine.to_netcdf(rp,engine='h5netcdf')
                    request=dict(task='spatial',variable='temperature',unit='K',semantic_type='continuous',nonnegative=True,spatial_semantics='cell_mean',
                        target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),parameters={},
                        model_profile=dict(id='wrf_chem',role='wrf_initial_boundary',field='air_temperature'),
                        model_requirements=dict(priorities={'continuity':1},hard=[],evidence='Explicit preparation preference, not measured WRF-Chem sensitivity'),
                        preferences=dict(weights={'RMSE':1},scales={'RMSE':1}),requirements=dict(try_coarse_consistency=True),
                        profile_options=dict(max_steps=2,candidate_limit=64,candidate_pool=dict(algorithms=BASES+[TAIL])))
                    context,_,_=make_context([('primary',coarse)],request)
                    diff=context.parameters['degradation_operator']-operator
                    assert not diff.nnz or np.max(abs(diff.data))<1e-12
                    cfg=OUT/'public_requests'/f'{token}.json'
                    if not cfg.exists():save(cfg,request)
                    tasks.append(dict(id=token,date=timestamp.strftime('%Y-%m-%d'),stage=stage,region=region,scale=scale,
                        coarse=str(cp),reference=str(rp),coarse_sha256=sha(cp),reference_sha256=sha(rp),request=str(cfg)))
    save(manifest,tasks);return tasks


class AreaMetric(DatasetReference):
    def __init__(self,reference,coarse,operator):
        super().__init__(reference,'temperature');self.coarse=coarse;self.operator=operator
        self.mask=np.outer((reference.lat>=coarse.lat.min())&(reference.lat<=coarse.lat.max()),(reference.lon>=coarse.lon.min())&(reference.lon<=coarse.lon.max()))
        self.weights=np.outer(np.diff(np.sin(np.deg2rad(edges(reference.lat)))),np.diff(np.deg2rad(edges(reference.lon))))
        self.protocol.update(objective='spherical_area_RMSE_common_interior_v1')
    def __call__(self,ds):
        super().__call__(ds);d=ds.temperature.values-self.reference.values;w=self.weights;mask=self.mask
        avg=lambda a:float(np.sum((a*w)[mask])/np.sum(w[mask]))
        rmse=float(np.sqrt(avg(d*d)))
        residual=self.operator@ds.temperature.values.ravel()-self.coarse.temperature.values.ravel()
        return dict(objective=rmse,RMSE=rmse,MAE=avg(abs(d)),Bias=avg(d),coverage=1.,
            full_RMSE=float(np.sqrt(np.sum(d*d*w)/w.sum())),coarse_RMSE=float(np.sqrt(np.mean(residual**2))),
            denominator=int(mask.sum()))


def inputs(t):
    assert sha(t['coarse'])==t['coarse_sha256']
    return [SearchInput('coarse.nc','primary',Path(t['coarse']).read_bytes())]


def matrix(t):
    folder=OUT/'runs'/t['id']
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    if t['stage']=='audit':assert (OUT/'audit_choices_frozen.json').exists()
    with xr.open_dataset(t['reference']) as d:reference=d.temperature.load()
    with xr.open_dataset(t['coarse']) as d:coarse=d.load()
    req=read(t['request'])
    if t['stage'] in {'fit','calibration'}:req['profile_options'].update(evidence_role='development',group=t['date'])
    context,_,_=make_context([('primary',coarse)],req)
    event('REFERENCE_OPEN',task=t['id'],stage=t['stage'])
    result=submit_search(inputs(t),req,AreaMetric(reference,coarse,context.parameters['degradation_operator']),folder,
        mode='Profile',budget=dict(objective_limit=14,wall_seconds=120))
    assert result['counts']['real_executions']==14
    event('MATRIX_COMPLETE',task=t['id'],failures=result['counts']['quality_failures'])
    print('EXECUTED',t['id'],result['counts'],flush=True);return result


def train(tasks,round_name):
    model='spatial_next_'+round_name+'_20261023';path=OUT/(round_name+'_training.json')
    if path.exists():return model
    report=train_profile_matching([OUT/'runs'/t['id'] for t in tasks if t['stage'] in {'fit','calibration'}],model,
        holdout_groups=sorted({t['date'] for t in tasks if t['stage']=='calibration'}))
    save(path,report);event('TRAINED',round=round_name,model=model);return model


def choose(t,model,label,scoring='metric',fallback='rules',use_model_defaults=False):
    choices=[]
    for scenario in ['M1','M2']:
        for strategy in ['cost','rules','learning']:
            folder=OUT/'recommendations'/label/t['id']/scenario/strategy
            if (folder/'summary.json').exists():r=read(folder/'summary.json')
            else:
                req=read(t['request']);req['model_requirements']['hard']=[] if scenario=='M1' else ['coarse_consistency']
                if strategy=='cost':req['model_requirements']['priorities']={}
                if strategy=='learning':req['profile_options'].update(ranking_model_id=model,acknowledge_limited_development=True)
                if strategy=='learning' and not use_model_defaults:
                    req['profile_options'].update(learning_order=scoring,fallback_policy=fallback)
                r=submit_search(inputs(t),req,None,folder,mode='Profile')
            assert r['counts']['real_executions']==0 and r['best'] is None
            c=r['recommendations'][0]
            choices.append(dict(task=t['id'],scenario=scenario,strategy=strategy,key=c['key'],steps=c['steps'],
                matching_status=r['matching_status'],support=r['learning_status'],region=t['region'],scale=t['scale'],date=t['date']))
            if strategy=='cost':
                legal=sorted(r['recommendations'],key=lambda x:x['key'])
                for seed in range(10):
                    salt=int(hashlib.sha256(t['id'].encode()).hexdigest()[:8],16)
                    picked=legal[int(np.random.default_rng(seed+salt).integers(len(legal)))]
                    choices.append(dict(task=t['id'],scenario=scenario,strategy='random',seed=seed,key=picked['key'],
                        steps=picked['steps'],region=t['region'],scale=t['scale'],date=t['date']))
    return choices


def fixed_choices(tasks):
    records=[]
    for t in tasks:
        if t['stage']!='fit':continue
        for row in read(OUT/'runs'/t['id']/'summary.json')['rows']:
            if row['status']=='PASS':records.append(dict(region=t['region'],scale=t['scale'],steps=tuple(row['steps']),RMSE=row['RMSE']))
    df=pd.DataFrame(records);fixed={}
    for scenario in ['M1','M2']:
        x=df if scenario=='M1' else df[df.steps.map(lambda x:x[-1]==TAIL)]
        for (region,scale),part in x.groupby(['region','scale']):
            fixed[f'{scenario}/{region}/{scale}']=list(part.groupby('steps').RMSE.mean().idxmin())
    return fixed


def assess(tasks,choices,fixed,label):
    records=[]
    for t in tasks:
        rows=read(OUT/'runs'/t['id']/'summary.json')['rows'];mapping={r['key']:r for r in rows}
        for scenario in ['M1','M2']:
            legal=[r for r in rows if scenario=='M1' or r['steps'][-1]==TAIL]
            best=min(r['RMSE'] for r in legal if r['status']=='PASS')
            selections=[c for c in choices if c['task']==t['id'] and c['scenario']==scenario]
            steps=fixed[f"{scenario}/{t['region']}/{t['scale']}"]
            c=next(r for r in legal if r['steps']==steps)
            selections.append(dict(task=t['id'],scenario=scenario,strategy='development_fixed',key=c['key'],steps=steps,
                region=t['region'],scale=t['scale'],date=t['date']))
            for choice in selections:
                row=mapping[choice['key']]
                records.append(dict(**choice,status=row['status'],RMSE=row.get('RMSE'),regret=row.get('RMSE',best)-best))
    df=pd.DataFrame(records);df.to_csv(OUT/(label+'_results.csv'),index=False)
    summaries=[]
    for scenario,part in df.groupby('scenario'):
        means=part.groupby('strategy').RMSE.mean().to_dict();learn=part[part.strategy=='learning']
        coverage=float(learn.matching_status.eq('LEARNED_DEVELOPMENT_RANK').mean())
        ratios={strategy:means['learning']/means[strategy] for strategy in ['rules','random','development_fixed']}
        ok=coverage>=.8 and ratios['rules']<=.95 and ratios['random']<=.95 and ratios['development_fixed']<=1.02
        summaries.append(dict(scenario=scenario,tasks=int(learn.task.nunique()),coverage=coverage,mean_RMSE=means,ratios=ratios,
            failures=int(df.status.ne('PASS').sum()),goals_met=bool(ok)))
    save(OUT/(label+'_assessment.json'),summaries)
    print('ASSESSMENT',label,json.dumps(summaries),flush=True);return summaries


def development(round_name,enhanced=False):
    protocol();tasks=prepare(enhanced)
    for t in tasks:
        if t['stage'] in {'fit','calibration'}:matrix(t)
    model=train(tasks,round_name);fixed=fixed_choices(tasks);save(OUT/(round_name+'_fixed.json'),fixed)
    choices=[];validation=[t for t in tasks if t['stage']=='validation']
    for t in validation:choices+=choose(t,model,round_name);matrix(t)
    save(OUT/(round_name+'_validation_choices.json'),choices)
    assess(validation,choices,fixed,round_name+'_validation')


def variant(model,label,scoring='metric',fallback='rules',tasks=None):
    tasks=tasks or prepare(True);fixed=fixed_choices(tasks);save(OUT/(label+'_fixed.json'),fixed)
    validation=[t for t in tasks if t['stage']=='validation'];choices=[]
    for t in validation:choices+=choose(t,model,label,scoring,fallback)
    save(OUT/(label+'_validation_choices.json'),choices)
    return assess(validation,choices,fixed,label+'_validation')


def final_audit(model,label,scoring='metric',fallback='rules',manifest='manifest_extremes.json'):
    assert not (OUT/'audit_choices_frozen.json').exists(),'Audit can only be opened once'
    tasks=read(OUT/manifest);fixed=fixed_choices(tasks);audit=[t for t in tasks if t['stage']=='audit']
    save(OUT/'final_selection.json',dict(model=model,validation_variant=label,learning_order=scoring,fallback_policy=fallback,
        fixed=fixed,manifest=manifest,manifest_sha256=sha(OUT/manifest),selection_scope='development validation only',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'src/ecodata/algorithms').glob('*.py'))+
            [ROOT/'src/ecodata/interface/raster_upload.py',ROOT/'tools/run_spatial_next.py']}))
    choices=[]
    for t in audit:
        choices+=choose(t,model,'final_audit',scoring,fallback,use_model_defaults=True)
        for ablation,order in [('metric_ablation','metric'),('rank_ablation','task_rank')]:
            for choice in choose(t,model,'final_'+ablation,order,'rules'):
                if choice['strategy']=='learning':choices.append(dict(choice,strategy=ablation))
        event('AUDIT_RECOMMENDATIONS_ONLY',task=t['id'])
        print('AUDIT_RECOMMENDATIONS_ONLY',t['id'],flush=True)
    save(OUT/'audit_choices_frozen.json',choices);event('ALL_AUDIT_CHOICES_FROZEN',sha256=sha(OUT/'audit_choices_frozen.json'))
    for t in audit:matrix(t)
    assess(audit,choices,fixed,'final_audit');event('COMPLETE',audit_tasks=len(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',required=True);p.add_argument('--enhanced',action='store_true')
    a=p.parse_args();development(a.round,a.enhanced)
