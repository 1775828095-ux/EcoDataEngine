"""Small-basin experiment client; all reconstruction decisions use public search."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json,hashlib,sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import convolve2d
from ecodata.algorithms import platform_search as api
from ecodata.algorithms import attribute_registry as attributes
from ecodata.algorithms.numerical_search import save

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/basin_ecology_2013_2017/prepared_v1'
OUT=ROOT/'output/basin_ecology_2013_2017'
SEEDS=[11,29,47]
BUDGET=dict(objective_limit=180,proposal_limit=40000,wall_seconds=120,probe_call_limit=60)
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def prepare():
    if (OUT/'prepared.json').exists():raise ValueError('Experiment inputs already frozen')
    with xr.open_dataset(DATA/'standardized.nc') as f:ds=f.load()
    common=np.isfinite(ds.fvc).all('time').values&ds.basin_mask.values.astype(bool)
    blocks=[]
    for width in (8,6):
        starts=np.argwhere(convolve2d(common.astype(int),np.ones((width,width),int),mode='valid')==width*width)
        chosen=[]
        if len(starts):
            centre=np.mean(np.argwhere(common),axis=0)-width/2
            chosen=[starts[np.argmin(np.sum((starts-centre)**2,axis=1))]]
            for _ in range(2):
                choices=[s for s in starts if all(abs(s[0]-p[0])>=width+1 or abs(s[1]-p[1])>=width+1 for p in chosen)]
                if not choices:break
                chosen.append(max(choices,key=lambda s:min(float(np.sum((s-p)**2)) for p in chosen)))
        if len(chosen)==3:
            blocks=[dict(case='block'+str(i),lat=[int(p[0]),int(p[0]+width)],lon=[int(p[1]),int(p[1]+width)],width=width) for i,p in enumerate(chosen)]
            break
    if not blocks:raise ValueError('Insufficient quality-only full-basin blocks; do not choose by error')
    registry_version='ecological-grid-state-v1'
    if not (attributes.ROOT/(registry_version+'.json')).exists():
        definitions=deepcopy(attributes.load('builtin-v1')['attributes'])
        for metric in ['missing_fraction','negative_fraction']:
            definitions.append(dict(id='eco_'+metric,definition='Measured '+metric+' in bounded ecological raster',
                tables=['data','algorithm','model'],unit='1',role='scoring',scoring_enabled=True,hard_allowed=False,
                provider='state_metric',property=metric,activation='constant_one',
                evidence='Explicit complete nonnegative FVC task contract, not simulator sensitivity',history=['Initial ecological task state requirement']))
        attributes.publish(registry_version,definitions,evidence='Reuse registered state measurement; no empirical accuracy invented')
    contract=dict(id='ecological-fvc-reconstruction',version='1',field='fvc',unit='1',semantic_type='continuous',valid_range=[0.,1.],
        evidence='Explicit FVC product recovery target; bounded fraction, finite complete grid, no native ecological model claim',
        issues=[dict(attribute='eco_'+metric,metric=metric,acceptable=0.,bad=1.,importance=1.,
          evidence='Equal input-validity diagnostics only; real reconstruction error supplied by isolated validation') for metric in ['missing_fraction','negative_fraction']])
    train=ds.sel(time=slice('2013','2014'));app=ds.sel(time=slice('2015','2017'))
    # The ancillary port contains prior labels only. No future FVC is a feature.
    features=[n for n in ds if ds[n].attrs.get('feature_role')=='covariate']
    aux=ds[features].copy(deep=True)
    aux['training_target']=ds.fvc.where((ds.time.dt.year<=2014)&ds.basin_mask.astype(bool))
    aux.training_target.attrs=dict(ds.fvc.attrs,split='train')
    aux.attrs.update(training_end='2014-12-31',target_labels_after_training='all NaN')
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'data').mkdir(exist_ok=True)
    aux.to_netcdf(OUT/'data/ancillaries_prior_only.nc',engine='h5netcdf')
    climatology=train.fvc.mean('time',skipna=True).fillna(float(train.fvc.where(ds.basin_mask.astype(bool)).mean()))
    for block in blocks:
        folder=OUT/'cases'/block['case'];folder.mkdir(parents=True,exist_ok=True)
        primary=app[['fvc']].copy(deep=True);primary.attrs.update(source_id='landsat_visible',data_role='masked_observations')
        yy=slice(*block['lat']);xx=slice(*block['lon']);primary.fvc.values[:,yy,xx]=np.nan
        primary.to_netcdf(folder/'visible_fvc.nc',engine='h5netcdf')
        prior=app[['fvc']].copy(deep=True);prior.fvc.values[:]=climatology.values
        prior.attrs.update(source_id='prior_fvc',data_role='prior_estimate',training_start='2013',training_end='2014',
          source_independence='Prior same-target product climatology, not another independent sensor')
        prior.to_netcdf(folder/'prior_fvc.nc',engine='h5netcdf')
        for phase,year in [('development',2015),('validation',2016),('test',2017)]:
            label=ds.fvc.sel(time=[pd.Timestamp(f'{year}-07-01')]).isel(lat=yy,lon=xx)
            assert np.isfinite(label).all()
            label=label.copy();label.attrs=dict(label.attrs,split='test' if phase=='test' else 'search_validation')
            label.to_dataset(name='fvc').to_netcdf(folder/(phase+'_labels.nc'),engine='h5netcdf')
        request=dict(task='multisource',variable='fvc',unit='1',semantic_type='continuous',nonnegative=True,
            target_lat=ds.lat.values.tolist(),target_lon=ds.lon.values.tolist(),target_times=[str(t) for t in app.time.values],
            allow_training=True,protect_observed_source='landsat_visible',parameters=dict(grid_gap_recovery=True,require_complete_output=True),
            application_contract=contract,scoring=dict(registry_version=registry_version),
            response_scoring=dict(enabled=True,evidence='Point-series probes are unsupported for raster data and must remain UNKNOWN'))
        inputs=[dict(path=str(folder/'visible_fvc.nc'),role='primary'),dict(path=str(folder/'prior_fvc.nc'),role='background'),
          dict(path=str(OUT/'data/ancillaries_prior_only.nc'),role='auxiliary')]
        save(folder/'config.json',dict(inputs=inputs,request=request))
    save(OUT/'prepared.json',dict(blocks=blocks,features=features,train=[2013,2014],development=2015,validation=2016,test=2017,
      selection_basis='Source QA and basin geometry only; not reconstruction error',test_scored=False,seed_list=SEEDS,budget=BUDGET,
      no_handwritten_candidate_graph_pool=True))
    print('PREPARED',blocks,features,flush=True)


def freeze():
    path=OUT/'source_manifest.json'
    if path.exists():
        for name,digest in load(path).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        return
    paths=list((ROOT/'src/ecodata').rglob('*.py'))+list((ROOT/'src/ecodata').rglob('*.json'))+list((ROOT/'data/attribute_registries').glob('*.json'))
    paths+=[ROOT/'tools'/name for name in ['run_basin_ecology.py','evaluate_basin_ecology.py','audit_basin_ecology.py']]
    manifest={}
    for source in paths:
        name=source.relative_to(ROOT).as_posix();dest=OUT/'source_frozen'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes());manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    save(path,manifest)
    save(OUT/'input_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((OUT/'data').glob('*.nc'))+list((OUT/'cases').rglob('*.nc'))})
    save(OUT/'protocol_snapshot.json',load(OUT/'prepared.json'))


def run(case,control,seed,phase):
    config=load(OUT/'cases'/case/'config.json');config['request']['mutation_control']=control
    config.update(mode='PortraitMutation',seed=seed,budget=BUDGET,search_validation=dict(path=str(OUT/'cases'/case/(phase+'_labels.nc'))))
    folder=OUT/'runs'/phase/(case+'_'+control+'_'+str(seed))
    if (folder/'summary.json').exists():return load(folder/'summary.json')
    save(folder/'client_config.json',config);save(folder/'metadata.json',dict(case=case,control=control,seed=seed,phase=phase))
    try:return api.submit_file_request(config,folder)
    except Exception as exc:
        result=dict(status='BLOCKED',error=str(exc));save(folder/'failure.json',result);return result


if __name__=='__main__':
    phase=sys.argv[1]
    if phase=='prepare':prepare()
    else:
        freeze();seeds=[704] if phase=='development' else SEEDS
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs={pool.submit(run,b['case'],control,seed,phase):(b['case'],control,seed) for b in load(OUT/'prepared.json')['blocks'] for control in ['random','none'] for seed in seeds}
            for future in as_completed(jobs):
                result=future.result();print(*jobs[future],result['status'],result.get('total_cost',{}).get('used'),
                    (result.get('best') or {}).get('metrics',{}).get('objective'),flush=True)
