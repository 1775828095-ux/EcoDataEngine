"""Frozen unseen-station benchmark through the public platform entry."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,sys
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.quality_aware import fit_affine_source
from ecodata.algorithms.numerical_search import save
from run_robit_single_source import dataset as existing_daily_dataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/unseen_portrait_response_20261028'
DATA=ROOT/'data/wa_precipitation_2014_2024'
VAR='precipitation'
STATIONS=['ASN00009021','ASN00003003','ASN00012038']
ARMS=['full','random','no_response','local_only']
SEEDS=[11,29,47]
BUDGET=dict(objective_limit=180,probe_call_limit=60,proposal_limit=40000,wall_seconds=120)
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def daily(frame,meta,source_id):
    ds=existing_daily_dataset(frame)
    ds=ds.assign_coords(lat=[meta['latitude']],lon=[meta['longitude']])
    ds['time_bounds']=(('time','bounds'),np.stack([(frame.time-pd.Timedelta(days=1)).to_numpy(),frame.time.to_numpy()],axis=1))
    ds.attrs.update(source_id=source_id,day_boundary='09:00 Australia/Perth; date is accumulation end; time is UTC',
                    not_spatial_downscaling='true')
    ds[VAR].attrs.update(physical_definition='24-hour precipitation amount ending at station reporting time')
    return ds


def prepare():
    if (OUT/'prepared.json').exists():raise ValueError('Prepared data are immutable')
    started=perf_counter();fits=[]
    for station_index,sid in enumerate(STATIONS):
        meta=load(DATA/sid/'metadata.json');raw=pd.read_csv(DATA/sid/'gauge.csv',parse_dates=['date'])
        raw['time']=raw['date']+pd.Timedelta(hours=1)
        secondary=pd.read_csv(DATA/sid/'era5_daily_9am.csv',parse_dates=['time'])
        np.testing.assert_array_equal(raw.time,secondary.time)
        train=raw.time.dt.year<=2017;application=~train
        ref=raw.loc[train,VAR].to_numpy();hist=secondary.loc[train,VAR].to_numpy();valid=np.isfinite(ref)&np.isfinite(hist)
        fit_started=perf_counter();slope,intercept=fit_affine_source(hist,ref)
        month=raw.loc[train,'time'].dt.month.to_numpy()
        means=np.array([np.mean(ref[valid&(month==m)]) for m in range(1,13)])
        hmeans=np.array([np.mean(hist[valid&(month==m)]) for m in range(1,13)])
        variables=dict(month_mean=('month',means),scale=float(np.std(ref[valid])),affine_slope=slope,affine_intercept=intercept)
        if hist[valid].mean()>0:variables['mean_ratio']=float(ref[valid].mean()/hist[valid].mean())
        if (hmeans>0).all():variables['monthly_ratio']=('month',means/hmeans)
        source_id='ERA5_'+sid
        state=xr.Dataset(variables,coords={'month':np.arange(1,13)},attrs=dict(training_start='2014-01-01T01:00:00',
             training_end='2017-12-31T01:00:00',calibration_source_id=source_id,split='training'))
        shared=OUT/'data'/sid;shared.mkdir(parents=True,exist_ok=True)
        state.to_netcdf(shared/'state.nc',engine='h5netcdf')
        for filename,frame,identity in [('training_reference.nc',raw.loc[train],sid),('historical_simulation.nc',secondary.loc[train],source_id)]:
            ds=daily(frame,meta,identity);ds[VAR].attrs['split']='train';ds.to_netcdf(shared/filename,engine='h5netcdf')
        original=daily(raw.loc[application],meta,sid);proxy=daily(secondary.loc[application],meta,source_id)
        proxy.attrs.update(observation_kind='Nearest ERA5 cell proxy, not a second gauge',
            native_grid_latitude=meta['native_era5_latitude'],native_grid_longitude=meta['native_era5_longitude'])
        proxy.to_netcdf(shared/'era5_proxy.nc',engine='h5netcdf')
        original.to_netcdf(shared/'reference_OFFLINE_ONLY.nc',engine='h5netcdf')
        fits.append(dict(station=sid,valid_training_pairs=int(valid.sum()),seconds=perf_counter()-fit_started))
        for kind,length in [('short',3),('long',14)]:
            case=sid+'_'+kind;folder=OUT/'cases'/case;folder.mkdir(parents=True,exist_ok=True)
            ds=original.copy(deep=True);dates=pd.DatetimeIndex(ds.time.values);known=np.isfinite(ds[VAR].values[:,0,0]);mask=np.zeros(len(dates),bool);counts=[]
            for year in np.unique(dates.year):
                indices=np.flatnonzero(dates.year==year);available=known[indices];local=np.zeros(len(indices),bool)
                rng=np.random.default_rng(20261028+int(year)+10000*length+station_index*100000)
                target=int(available.sum()*.2)
                for begin in rng.permutation(np.arange(1,len(indices)-length-1)):
                    end=begin+length
                    if local[begin-1:end+1].any() or not available[begin-1:end+1].all():continue
                    local[begin:end]=True
                    if local.sum()>=target:break
                mask[indices]=local;counts.append(dict(year=int(year),target=target,hidden=int(local.sum()),natural_missing=int((~available).sum())))
            assert not (mask&~known).any()
            ds[VAR].values[mask]=np.nan;ds.to_netcdf(folder/'station_masked.nc',engine='h5netcdf')
            for phase,(start,end) in {'development':(2018,2019),'validation':(2020,2021),'test':(2022,2024)}.items():
                which=mask&(dates.year>=start)&(dates.year<=end)
                label=original[VAR].isel(time=np.flatnonzero(which));label.attrs=dict(original[VAR].attrs,split='test' if phase=='test' else 'search_validation')
                assert np.isfinite(label).all();label.to_dataset(name=VAR).to_netcdf(folder/(phase+'_labels.nc'),engine='h5netcdf')
            request=dict(task='multisource',variable=VAR,unit='mm/day',semantic_type='continuous',nonnegative=True,
                spatial_semantics='point',temporal_semantics='accumulation',target_lat=[meta['latitude']],target_lon=[meta['longitude']],
                target_times=[str(t) for t in dates],protect_observed_source=sid,allow_training=True,
                parameters=dict(target_time_frequency='1D',objective_scope='calendar_year_RMSE',require_complete_output=True,
                    calibration_source_id=source_id,adjustment_kind='*'),
                model_profile=dict(id='swat_plus',role='weather',field=VAR),
                portrait_policy=dict(version='robit-era5-multisource-swatplus-v1',scenario='daily-recovery'),
                scoring=dict(registry_version='robit-daily-state-v1'),
                response_scoring=dict(period=['2018-01-01T01:00:00','2019-12-31T01:00:00'],weights=dict(data=.5,model=.5,collaboration=0.),
                   evidence='Reuse existing frozen daily-response experimental priorities, not learned simulator sensitivity'))
            inputs=[dict(path=str(folder/'station_masked.nc'),role='primary'),dict(path=str(shared/'era5_proxy.nc'),role='background'),
               dict(path=str(shared/'state.nc'),role='state'),dict(path=str(shared/'training_reference.nc'),role='training_reference'),
               dict(path=str(shared/'historical_simulation.nc'),role='historical_simulation')]
            save(folder/'config.json',dict(inputs=inputs,request=request));save(folder/'mask_counts.json',counts)
    save(OUT/'prepared.json',dict(stations=STATIONS,station_selection_by_coverage_only=True,test_scored=False,
         common_training_fits=fits,total_preparation_seconds=perf_counter()-started,budget=BUDGET,seeds=SEEDS,arms=ARMS))
    print('PREPARED SIX CASES; NO TEST EFFECT CALCULATED',flush=True)


def freeze():
    mechanism=ROOT/'output/portrait_response_20261028'
    assert load(mechanism/'mechanism_acceptance.json')['status']=='PASS'
    for path,digest in load(mechanism/'mechanism_source_hashes.json').items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,'Mechanism gate source changed: '+path
    manifest=OUT/'source_manifest.json'
    if manifest.exists():
        for path,digest in load(manifest).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        return
    paths=list((ROOT/'src/ecodata').rglob('*.py'))+list((ROOT/'src/ecodata').rglob('*.json'))
    paths+=list((ROOT/'data/model_portraits').glob('*.json'))+list((ROOT/'data/attribute_registries').glob('*.json'))
    paths+=[ROOT/'tools'/name for name in ['run_unseen_portrait_response.py','evaluate_unseen_portrait_response.py','audit_unseen_portrait_response.py']]
    hashes={}
    for path in paths:
        name=path.relative_to(ROOT).as_posix();dest=OUT/'source_frozen'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(manifest,hashes)
    input_paths=list((OUT/'data').rglob('*.nc'))+list((OUT/'cases').rglob('*.nc'))
    save(OUT/'input_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths})
    save(OUT/'protocol.json',dict(stations=STATIONS,arms=ARMS,seeds=SEEDS,budget=BUDGET,train=[2014,2017],development=[2018,2019],validation=[2020,2021],test=[2022,2024],parent_policy='best',
        total_budget_includes=['probe','search','failed_registered_calls','deploy'],test_unseen_at_freeze=True,global_optimum_certified=False))
    print('FROZEN',len(hashes),flush=True)


def cases():return [sid+'_'+kind for sid in STATIONS for kind in ('short','long')]


def run(case,arm,seed,phase='validation'):
    config=load(OUT/'cases'/case/'config.json')
    config.update(mode='PortraitMutation',seed=seed,budget=BUDGET,search_validation=dict(path=str(OUT/'cases'/case/(phase+'_labels.nc'))))
    config['request']['mutation_control']='random' if arm=='random' else 'alns'
    config['request']['joint_destroy_repair']=arm!='local_only'
    if arm=='no_response':config['request']['response_scoring']['enabled']=False
    folder=OUT/'runs'/phase/(case+'_'+str(seed)+'_'+arm)
    if (folder/'summary.json').exists():return load(folder/'summary.json')
    save(folder/'client_config.json',config)
    save(folder/'run_metadata.json',dict(case=case,arm=arm,seed=seed,phase=phase))
    try:return api.submit_file_request(config,folder)
    except Exception as exc:
        failure=dict(status='BLOCKED',error=str(exc),case=case,arm=arm,seed=seed)
        save(folder/'run_failure.json',failure);return failure


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='prepare':prepare()
    elif stage in {'development','validation'}:
        freeze()
        arms=['full','random'] if stage=='development' else ARMS;seeds=[704] if stage=='development' else SEEDS
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs={pool.submit(run,case,arm,seed,stage):(case,arm,seed) for case in cases() for arm in arms for seed in seeds}
            for job in as_completed(jobs):
                r=job.result();print(*jobs[job],r['status'],r.get('response_score_intervals'),r.get('total_cost',{}).get('used'),
                    (r.get('best') or {}).get('metrics',{}).get('objective'),flush=True)
