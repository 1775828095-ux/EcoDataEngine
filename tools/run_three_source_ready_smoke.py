"""Real NOAA AOD + ERA5 + EPA daily arrays; fixed spatial/time holdout."""
import hashlib
import argparse
from dataclasses import asdict
from time import perf_counter
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.append(str(ROOT/'.venv-golem/Lib/site-packages'))
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import ConvexHull, Delaunay
from ecodata.algorithms.dag import Artifact
from ecodata.algorithms.epa_task import EPATask
from ecodata.algorithms.registry import create_default_registry
from golem.core.optimisers.graph import OptNode, OptGraph


def golem_graph(nodes):
    by_id={}
    for node in nodes:
        params=dict(node.parameters)
        parents,bindings=[],{}
        for port,(parent,output) in node.inputs.items():
            if by_id[parent] not in parents:
                parents.append(by_id[parent])
            bindings[port]=[parents.index(by_id[parent]),output]
        params['_port_bindings']=bindings
        by_id[node.id]=OptNode({'name':node.operation,'params':params},nodes_from=parents)
    return OptGraph([by_id['quality']])


def main():
    base=ROOT/'output/three_experiments_20261019'
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=base/'joint_smoke')
    parser.add_argument('--skip-baselines',action='store_true')
    parser.add_argument('--fresh-holdouts',action='store_true')
    parser.add_argument('--nonnegative-projection',action='store_true')
    parser.add_argument('--satellite-snapshot',type=Path)
    args=parser.parse_args()
    out=args.output
    out.mkdir(exist_ok=False)
    region='northern_california' if args.fresh_holdouts else 'central_valley'
    data=pd.read_csv(base/region/'qualified_continuous_daily_2022.csv')
    sites=pd.read_csv(base/region/('station_metadata.csv' if args.fresh_holdouts else 'station_split.csv'))
    points=sites[['longitude','latitude']].to_numpy()
    hull=set(ConvexHull(points).vertices)
    interior=sorted([i for i in range(len(sites)) if i not in hull],
        key=lambda i:hashlib.sha256(('joint11:'+sites.iloc[i].station_id).encode()).hexdigest())
    if args.fresh_holdouts:
        exposed=set(pd.read_csv(base/'central_valley/station_split.csv').station_id)
        interior=[i for i in interior if sites.iloc[i].station_id not in exposed]
        if len(interior)<8:
            raise ValueError('Fewer than eight previously untouched interior stations; cannot promise clean split')
    val=set(sites.iloc[interior[:4]].station_id)
    test=set(sites.iloc[interior[4:8]].station_id)
    if args.fresh_holdouts:
        # Keep the same four untouched Bay test stations. Coastal-only
        # validation failed input-domain checks (no target-score selection).
        # A mixed inland/coastal validation set measures both regimes.
        val=set(sites.iloc[interior[:2]].station_id)|{'06-031-0004','06-067-0006'}
    sites['split']=['validation' if s in val else 'test' if s in test else 'train' for s in sites.station_id]
    sites.to_csv(out/'station_split.csv',index=False)
    data['split']=data.station_id.map(sites.set_index('station_id').split)
    # Independent spatial targets plus future dates. Contemporaneous INPUT
    # station observations are allowed; this is reconstruction, not forecasting.
    config=dict(seed=11,training_dates='2022-01-01/2022-03-31',validation_dates='2022-04-01/2022-08-31',
        final_test_dates='2022-09-01/2022-12-31',test_stations=sorted(test),
        features=['aod','temperature_2m'],satellite='NOAA GOES17 ABI AOD 550nm, DQF 0 or 1; low/no retrieval excluded',
        nonnegative_projection=args.nonnegative_projection,
        time_matching='EPA fixed UTC-8 daily mean; same-day ~12:02 local AOD predictor; ERA5 same fixed UTC-8 daily mean',
        limitation='midday AOD and daily PM2.5 are different temporal statistics; association model, not equality',
        test_targets_used_for_fitting_or_selection=False)
    (out/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    lat=np.linspace(sites.latitude.min()-.02,sites.latitude.max()+.02,55)
    lon=np.linspace(sites.longitude.min()-.02,sites.longitude.max()+.02,65)
    template=xr.Dataset({'pm25':(('lat','lon'),np.full((len(lat),len(lon)),np.nan))},
                        coords={'lat':lat,'lon':lon},attrs={'crs':'EPSG:4326'})
    template.pm25.attrs.update(units='ug m-3',data_kind='continuous')
    baseline=EPATask(create_default_registry(),template,sealed_test_stations=tuple(test),nonnegative_projection=args.nonnegative_projection)
    (out/'graph_contracts.json').write_text(json.dumps({method:[asdict(n) for n in baseline.graph(method)]
        for method in ('fusion.idw','station.pykrige_ordinary','route_a','route_b')},indent=2),encoding='utf-8')
    baseline_results=[]
    for date in list(pd.date_range('2022-04-15',periods=7))+list(pd.date_range('2022-07-15',periods=7)):
        if args.skip_baselines:
            break
        today=data.loc[data.time.eq(str(date.date())) & ~data.split.eq('test')]
        train=today[today.split.eq('train')]
        validation=today[today.split.eq('validation')]
        if len(train)<8 or not len(validation):
            continue
        supported=Delaunay(train[['longitude','latitude']].to_numpy()).find_simplex(validation[['longitude','latitude']].to_numpy())>=0
        validation=validation.loc[supported]
        if not len(validation):
            continue
        external={'stations':Artifact('epa_train',train)}
        for method in ('fusion.idw','station.pykrige_ordinary'):
            started=perf_counter()
            try:
                result=baseline.objective(method,external,validation,out/f'baseline_{date:%Y%m%d}_{method}',
                    golem_graph=golem_graph(baseline.graph(method)))
                result.update(date=str(date.date()),method=method,true_golem_dag=True)
            except Exception as exc:
                result=dict(date=str(date.date()),method=method,status='FAIL',error=str(exc),traceback=traceback.format_exc(),
                            seconds=perf_counter()-started,dag_trace=getattr(exc,'trace',[]))
            baseline_results.append(result)
    (out/'spatial_baseline_results.json').write_text(json.dumps(baseline_results,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(f'Spatial baseline executions: {len(baseline_results)}',flush=True)
    weather=xr.open_dataset(base/'anonymous/era5_daily_2022.nc').load()
    matched,coverage=[],[]
    covariates={}
    satellite_root=base/('anonymous_north' if args.fresh_holdouts else 'anonymous')
    files=sorted(satellite_root.glob('aod_????????.nc'))
    if args.satellite_snapshot:
        records=json.loads(args.satellite_snapshot.read_text(encoding='utf-8'))
        files=[Path(r['file']) for r in records]
        for file,record in zip(files,records):
            if hashlib.sha256(file.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError(f'Frozen satellite snapshot hash changed: {file}')
    (out/'source_snapshot.json').write_text(json.dumps([dict(file=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        for file in files],indent=2),encoding='utf-8')
    for file in files:
        date=pd.Timestamp(file.stem[4:])
        with xr.open_dataset(file) as sat:
            ds=sat[['aod']].load()
        ds['temperature_2m']=weather.temperature_2m.sel(time=date).interp(lat=ds.lat,lon=ds.lon)
        ds.temperature_2m.attrs.update(units='degC',data_kind='continuous')
        ds.attrs.update(crs='EPSG:4326',provenance_verified='true',epa_local_standard_date=str(date.date()))
        sample=data.loc[data.time.eq(str(date.date())) & ~data.split.eq('test')].copy()
        for feature in config['features']:
            sample[feature]=ds[feature].interp(lat=xr.DataArray(sample.latitude.to_numpy(),dims='station'),
                lon=xr.DataArray(sample.longitude.to_numpy(),dims='station')).values
        sample=sample.loc[np.isfinite(sample[config['features']]).all(axis=1)]
        # Record actual common coverage, including insufficient dates.
        coverage.append(dict(date=str(date.date()),common_non_test_stations=len(sample),
                             training_stations=int(sample.split.eq('train').sum()),
                             validation_stations=int(sample.split.eq('validation').sum()),
                             effective_grid_fraction=float(np.isfinite(ds.aod).mean())))
        matched.append(sample)
        covariates[str(date.date())]=ds
    pd.DataFrame(coverage).to_csv(out/'joint_coverage.csv',index=False)
    table=pd.concat(matched,ignore_index=True)
    table.to_csv(out/'matched_non_test_samples.csv',index=False)
    history=table.loc[(table.time<='2022-03-31') & table.split.eq('train')]
    if history.station_id.nunique()<4:
        raise ValueError('Insufficient true joint historical training stations')
    eligible=[r['date'] for r in coverage if '2022-04-01'<=r['date']<='2022-08-31'
              and r['training_stations']>=8 and r['validation_stations']>=1]
    (out/'selected_dates.json').write_text(json.dumps(dict(rule='first three data-eligible dates, fixed before numerical execution',
        dates=eligible[:3],historical_fit_samples=len(history)),indent=2),encoding='utf-8')
    results=[]
    # Fixed first three eligible dates by data availability, never by RMSE.
    attempted_dates=[]
    rejected_dates=[]
    for date in eligible:
        ds=covariates[date]
        today=table[table.time.eq(date)]
        train=today[today.split.eq('train')]
        validation=today[today.split.eq('validation')]
        # No extrapolation; report which fixed validation sites have support.
        supported=Delaunay(train[['longitude','latitude']].to_numpy()).find_simplex(validation[['longitude','latitude']].to_numpy())>=0
        validation=validation.loc[supported]
        if not len(validation):
            rejected_dates.append(dict(date=date,reason='fixed validation stations outside current input-station hull'))
            continue
        # Require the whole readback interpolation stencil to be supported.
        gy,gx=np.meshgrid(ds.lat.values,ds.lon.values,indexing='ij')
        hullmask=(Delaunay(train[['longitude','latitude']].to_numpy()).find_simplex(
            np.c_[gx.ravel(),gy.ravel()])>=0).reshape(gx.shape)
        validgrid=np.isfinite(ds.aod.values)&np.isfinite(ds.temperature_2m.values)&hullmask
        supported_grid=xr.DataArray(np.where(validgrid,1.,np.nan),dims=('lat','lon'),coords=ds.coords)
        common=supported_grid.interp(lat=xr.DataArray(validation.latitude.to_numpy(),dims='station'),
            lon=xr.DataArray(validation.longitude.to_numpy(),dims='station')).notnull().values
        validation=validation.loc[common]
        if not len(validation):
            rejected_dates.append(dict(date=date,reason='fixed validation stations lack complete valid output pixel stencil'))
            continue
        attempted_dates.append(date)
        template=xr.Dataset({'pm25':xr.full_like(ds.aod,np.nan)},attrs={'crs':'EPSG:4326'})
        template.pm25.attrs.update(units='ug m-3',data_kind='continuous')
        task=EPATask(create_default_registry(),template,features=tuple(config['features']),
                     train_end='2022-03-31',sealed_test_stations=tuple(test),nonnegative_projection=args.nonnegative_projection)
        external={'stations':Artifact('epa_train',train),'history':Artifact('epa_history',history),
                  'covariates':Artifact('epa_covariates',ds)}
        ds.to_netcdf(out/f'actual_covariates_{date}.nc',engine='h5netcdf')
        for method in ('fusion.idw','station.pykrige_ordinary','route_a','route_b'):
            started=perf_counter()
            try:
                result=task.objective(method,external,validation,out/f'{date}_{method}',
                    golem_graph=golem_graph(task.graph(method)))
                result.update(date=date,method=method,true_golem_dag=True)
            except Exception as exc:
                result=dict(date=date,method=method,status='FAIL',error=str(exc),traceback=traceback.format_exc(),
                            dag_trace=getattr(exc,'trace',[]),seconds=perf_counter()-started,
                            counters=getattr(exc,'objective_counters',dict(task.counters)))
            results.append(result)
            (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            print({k:v for k,v in result.items() if k not in ('trace','quality')},flush=True)
        if len(attempted_dates)>=3:
            break
    (out/'selected_dates.json').write_text(json.dumps(dict(rule='first three input-QA-and-geometry-supported dates; never selected by error',
        dates=attempted_dates,rejected_dates=rejected_dates,historical_fit_samples=len(history)),indent=2),encoding='utf-8')
    if not results:
        raise ValueError('No actual supported validation-date joint samples')


if __name__=='__main__':
    main()
