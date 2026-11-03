"""Robit station + ERA5 recovery with isolated fitting/selection/test ports."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_robit_single_source import raw_weather,dataset
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.quality_aware import fit_affine_source
from ecodata.algorithms.numerical_search import save

OUT=ROOT/'output/robit_era5_fusion_20261027'
ERA=ROOT/'data/robit_era5_1990_2013'
VAR='precipitation';SITE='ERA5_Robit_site_proxy'
PROTOCOL=dict(version='robit-era5-v6',train=[1990,1995],development=[1996,2001],validation=[2002,2007],test=[2008,2013],
    masks=dict(random=1,short=3,long=14),mask_fraction=.2,mask_seed=20261027,search_seeds=[11,29,47],
    primary_timezone='local_utc3',sensitivity_timezone='utc',
    budget=dict(objective_limit=180,proposal_limit=40000,wall_seconds=120),
    objective='Mean calendar-year hidden-day RMSE; singleton spatial axes are not a spatial reconstruction',
    dependence='ERA5 does not directly assimilate Robit daily rain gauges; original Robit generation provenance unverified',
    controls=['station-only','raw ERA5','development-best calibrated ERA5','development-best fixed graph','random search'],
    test_not_for_fitting=True,protect_visible_station=True,handwritten_candidate_pool=False)


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol_v6.json').exists():assert load(OUT/'protocol_v6.json')==PROTOCOL
    else:save(OUT/'protocol_v6.json',PROTOCOL)
    weather=raw_weather();era=pd.read_csv(ERA/'local_utc3.csv',parse_dates=['time'])
    assert weather.time.equals(era.time)
    fit=weather.time.dt.year<=1995;reference=weather.loc[fit,VAR].to_numpy();historical=era.loc[fit,VAR].to_numpy()
    slope,intercept=fit_affine_source(historical,reference)
    months=weather.loc[fit].time.dt.month
    means=weather.loc[fit].groupby(months)[VAR].mean().to_numpy()
    hm=era.loc[fit].groupby(months)[VAR].mean().to_numpy()
    state=xr.Dataset(dict(month_mean=('month',means),monthly_ratio=('month',np.divide(means,hm)),
        scale=float(np.std(reference)),affine_slope=slope,affine_intercept=intercept,mean_ratio=float(reference.mean()/historical.mean())),
        coords={'month':np.arange(1,13)},attrs=dict(training_start='1990-01-01',training_end='1995-12-31',
            calibration_source_id=SITE,split='training',fit='Prior observations only; affine reused fit_affine_source'))
    state.to_netcdf(OUT/'state.nc',engine='h5netcdf')
    for filename,frame in [('training_reference.nc',weather.loc[fit]),('historical_simulation.nc',era.loc[fit])]:
        ds=dataset(frame);ds[VAR].attrs['split']='train';ds.to_netcdf(OUT/filename,engine='h5netcdf')
    app=weather.time.dt.year>=1996
    original=dataset(weather.loc[app]);proxy=dataset(era.loc[app]);proxy.attrs.update(source_id=SITE,
        observation_kind='Gridded reanalysis nearest-cell proxy, NOT a rain-gauge measurement',
        native_grid_latitude=11.5,native_grid_longitude=37.25,native_grid_degrees=.25,
        collocation='Nearest grid cell used as site estimate; representativeness error unknown',day_boundary='Assumed local UTC+03 midnight')
    original.attrs['day_boundary']='Original day boundary unknown; local UTC+03 assumption'
    proxy.to_netcdf(OUT/'era5_proxy.nc',engine='h5netcdf')
    original.to_netcdf(OUT/'reference_DO_NOT_PASS_TO_ALGORITHMS.nc',engine='h5netcdf')
    save(OUT/'source_lineage.json',dict(station='Existing official Robit example, station provenance unverified',
        era5=load(ERA/'local_utc3_receipt.json'),grid_site_mapping='Explicit nearest-cell proxy approximation',
        derived_observation_semantics='Both target station daily amount; native ERA5 grid footprint retained in metadata',
        day_boundary='Unknown on Robit; UTC+03 primary, UTC sensitivity; never choose timezone on test'))
    for kind,length in PROTOCOL['masks'].items():
        folder=OUT/'cases'/kind;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'config.json').exists():
            existing=load(folder/'config.json');existing['request']['parameters']['require_complete_output']=True
            save(folder/'config_v5.json',existing);continue
        ds=original.copy(deep=True);dates=pd.DatetimeIndex(ds.time.values);mask=np.zeros(len(dates),bool)
        for year in np.unique(dates.year):
            pos=np.flatnonzero(dates.year==year);local=np.zeros(len(pos),bool)
            rng=np.random.default_rng(PROTOCOL['mask_seed']+int(year)+length*10000)
            for start in rng.permutation(np.arange(1,len(pos)-length-1)):
                if local[start-1:start+length+1].any():continue
                local[start:start+length]=True
                if local.sum()>=int(len(pos)*PROTOCOL['mask_fraction']):break
            mask[pos]=local
        ds[VAR].values[mask]=np.nan
        ds.to_netcdf(folder/'station_masked.nc',engine='h5netcdf')
        labels=[]
        for stage in ['development','validation','test']:
            begin,end=PROTOCOL[stage];which=mask&(dates.year>=begin)&(dates.year<=end)
            ref=original[VAR].isel(time=np.flatnonzero(which)).copy(deep=True)
            ref.attrs['split']='test' if stage=='test' else 'search_validation'
            ref.to_dataset().to_netcdf(folder/(stage+'_labels.nc'),engine='h5netcdf')
            labels.append(dict(stage=stage,n=int(which.sum())))
        req=dict(task='multisource',variable=VAR,unit='mm/day',semantic_type='continuous',nonnegative=True,
            spatial_semantics='point',temporal_semantics='accumulation',target_lat=[11.6],target_lon=[37.36],
            target_times=[str(t) for t in dates],parameters=dict(target_time_frequency='1D',objective_scope='calendar_year_RMSE',require_complete_output=True,
                calibration_source_id=SITE,adjustment_kind='*'),protect_observed_source='Robit_station')
        inputs=[dict(path=str(folder/'station_masked.nc'),role='primary'),dict(path=str(OUT/'era5_proxy.nc'),role='background'),
            dict(path=str(OUT/'state.nc'),role='state'),dict(path=str(OUT/'training_reference.nc'),role='training_reference'),
            dict(path=str(OUT/'historical_simulation.nc'),role='historical_simulation')]
        save(folder/'config_v5.json',dict(inputs=inputs,request=req,labels=labels))
    print('PREPARED',flush=True)


def run_case(kind,phase,seed,policy='contract_guided'):
    folder=OUT/'runs_v6'/phase/(kind+'_'+str(seed)+('_random' if policy=='random' else ''))
    if (folder/'summary.json').exists():return load(folder/'summary.json')
    config=load(OUT/'cases'/kind/'config_v5.json');request=dict(config['request'],search_policy=policy)
    data=dict(inputs=config['inputs'],request=request,mode='AutonomousField',seed=seed,budget=PROTOCOL['budget'],
        search_validation=dict(path=str(OUT/'cases'/kind/(phase+'_labels.nc'))))
    return api.submit_file_request(data,folder)


if __name__=='__main__':
    stage=sys.argv[1] if len(sys.argv)>1 else 'prepare'
    if stage=='prepare':prepare()
    elif stage=='batch':
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=3) as pool:
            tasks={pool.submit(run_case,kind,phase,seed,policy):(kind,phase,seed,policy)
                for kind in PROTOCOL['masks'] for phase in ['development','validation']
                for seed in PROTOCOL['search_seeds'] for policy in (['contract_guided','random'] if phase=='validation' else ['contract_guided'])}
            for task in as_completed(tasks):
                r=task.result();print(*tasks[task],r['status'],r['executed_candidates'],
                    (r.get('best') or {}).get('metrics',{}).get('objective'),flush=True)
    else:
        result=run_case(sys.argv[2],stage,int(sys.argv[3]) if len(sys.argv)>3 else 11)
        print(json.dumps({k:result.get(k) for k in ['status','executed_candidates','kernel_calls','elapsed_seconds','failed_candidates']}),flush=True)
        best=result.get('best') or {}
        print('BEST',{k:best.get(k) for k in ['key','algorithm','parameters']},best.get('metrics',{}).get('objective'),flush=True)
