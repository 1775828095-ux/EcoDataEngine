"""Reuse annual station archive and existing anonymous regional downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.engines.units import UnitEngine
from run_real_pm25_three_source import SPLIT
from run_pm25_ablation_validation import digest

PACKAGE=Path(r'D:\codex\数据\京津冀_PM25多源融合_201908')
RAW=Path(r'D:\codex\数据\国控\gk2019')


def module(name):
    path=PACKAGE/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def ground(root):
    target=root/'stations_september.parquet'
    if target.exists():return str(target)
    split=pd.read_csv(SPLIT,dtype={'station_id':str})
    parts=[]
    for city in split.city.unique():
        source=RAW/f'{city}.csv'
        for chunk in pd.read_csv(source,usecols=['DATATIME','STATIONCODE','LON','LAT','PM25'],
                                  dtype={'STATIONCODE':str},chunksize=100000,low_memory=False):
            dates=pd.to_datetime(chunk.DATATIME,errors='coerce')
            selected=chunk.STATIONCODE.isin(split.station_id) & dates.between('2019-09-01 09:00','2019-10-01 08:00')
            chunk=chunk[selected].copy();chunk['time_local_end']=dates[selected]
            chunk['source_file']=str(source)
            parts.append(chunk)
    frame=pd.concat(parts,ignore_index=True).rename(columns={
        'STATIONCODE':'station_id','LON':'longitude','LAT':'latitude','PM25':'value'})
    frame['value']=pd.to_numeric(frame.value,errors='coerce')
    frame['time']=frame.time_local_end-pd.Timedelta(hours=8,minutes=30)
    frame=frame.merge(split[['station_id','split','longitude','latitude']],on='station_id',
        suffixes=('','_fixed'),validate='many_to_one')
    frame['coordinate_delta']=np.maximum(abs(frame.longitude-frame.longitude_fixed),abs(frame.latitude-frame.latitude_fixed))
    if frame.duplicated(['station_id','time']).any():
        raise RuntimeError('September station/time duplicates: resolve explicitly, no silent dropping')
    frame['unit']='ug m-3';frame['quality_weight']=1.0
    excluded=int((~np.isfinite(frame.value) | (frame.value<0)).sum())
    frame=frame[np.isfinite(frame.value) & (frame.value>=0)].sort_values(['time','station_id'])
    if set(frame.station_id)!=set(split.station_id):
        raise RuntimeError('September does not cover all fixed 60 stations')
    frame.to_parquet(target,index=False)
    (root/'ground_provenance.json').write_text(json.dumps({
        'source':'existing annual CNEMC/QuotSoft-crosschecked station archive',
        'source_files':[{'path':str(RAW/f'{city}.csv'),'sha256':digest(RAW/f'{city}.csv')} for city in split.city.unique()],
        'source_timezone':'Asia/Shanghai UTC+8; preceding-hour mean',
        'aligned_time':'stored end label minus 8h30m -> GEOS UTC half-hour center',
        'station_counts':frame.groupby('split').station_id.nunique().to_dict(),
        'valid_records':len(frame),'nonfinite_or_negative_excluded':excluded,
        'maximum_coordinate_change_degrees':float(frame.coordinate_delta.max()),
        'coordinate_datum':'unresolved; same approximate stored datum as August',
        'instrument_qa':'not supplied, equal weights; no fabricated QA'},ensure_ascii=False,indent=2),encoding='utf-8')
    return str(target)


def geos(root):
    target=root/'geos_cf_pm25_201909_bth.nc'
    if target.exists():return str(target)
    m=module('download_geos_cf')
    m.START_INDEX=int((pd.Timestamp('2019-09-01')-pd.Timestamp('2018-01-01'))/pd.Timedelta(hours=1))
    cache=root/'geos_daily';cache.mkdir(exist_ok=True)
    chunks=[]
    for day in range(1,31):
        path=cache/f'geos_cf_pm25_201909{day:02d}.nc'
        if not path.exists():
            raw=m.read_day(day)
            ds=xr.decode_cf(raw)
            decoded=pd.to_datetime(ds.time.values)
            expected=pd.date_range(f'2019-09-{day:02d} 00:30',periods=24,freq='h')
            if not np.array_equal(decoded.values,expected.values):
                raise RuntimeError('Remote native GEOS timestamps do not match September indices')
            ds.time.attrs.update(timezone='UTC',cell_methods='time: mean',
                temporal_semantics='one-hour mean centered HH:30 UTC')
            ds['pm25_rh35_gcc']=UnitEngine().convert(ds.pm25_rh35_gcc,'ug m-3')
            ds.attrs.update(source=m.URL,product='GEOS-CF v1 aqc_tavg_1hr_g1440x721_v1',
                crs='EPSG:4326',coordinate_status='same model grid as August; ground datum approximate')
            ds.pm25_rh35_gcc.attrs.update(physical_variable='PM2.5',humidity_basis='RH35',data_kind='continuous')
            ds.to_netcdf(path,engine='h5netcdf')
        with xr.open_dataset(path,engine='h5netcdf') as ds:chunks.append(ds.load())
        print(f'GEOS day {day}/30 complete',flush=True)
    full=xr.concat(chunks,dim='time')
    full.to_netcdf(target,engine='h5netcdf')
    return str(target)


def era(root):
    target=root/'era5_201909_bth.nc'
    if target.exists():return str(target)
    import icechunk
    m=module('download_era5_anonymous')
    storage=icechunk.s3_storage(bucket='earthmover-icechunk-era5',prefix='icechunkV2',region='us-east-1',anonymous=True)
    repository=icechunk.Repository.open(storage);session=repository.readonly_session('main')
    ds=xr.open_zarr(session.store,group='single/temporal',consolidated=False,chunks=None)
    lat_slice=slice(41.5,35.5) if ds.latitude[0]>ds.latitude[-1] else slice(35.5,41.5)
    subset=ds[m.VARIABLES].sel(valid_time=slice('2019-09-01 00:00','2019-10-01 00:00'),
        latitude=lat_slice,longitude=slice(113.5,119.25))
    subset=m.normalize(subset,'Earthmover Icechunk ERA5 AWS Open Data anonymous; same source as August').load()
    subset.attrs['source_snapshot']=str(getattr(session,'snapshot_id','unavailable'))
    expected=pd.date_range('2019-09-01', '2019-10-01',freq='h')
    if not np.array_equal(subset.valid_time.values,expected.values):raise RuntimeError('ERA5 native time coverage mismatch')
    subset.to_netcdf(target,engine='h5netcdf')
    print('ERA5 September regional subset complete',flush=True)
    return str(target)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not (a.output/'audit.json').exists():raise RuntimeError('Finish ablation/seed stage first')
    root=a.output/'september'/'data';root.mkdir(parents=True,exist_ok=True)
    plan={'scope':'August 1-7 fixed checkpoints -> September observation-guided fusion',
        'seeds':[17,29,43,71,101],'direct_backends':['full','original_fno','original_uno'],
        'direct_evaluation':'2019-09-04 00:30 to 2019-09-30 23:30 UTC',
        'finetune':'full model only, September 1-3 train40+validation10, then same September4-30 test10',
        'finetune_learning_rate':0.0005,'finetune_epochs':10,'minimum_epochs':4,'patience':3,
        'target_test_for_updates_or_selection':False,'target_train_live_observations_used':True,
        'pure_zero_shot_prediction':False,'normalization':'frozen August checkpoint statistics, even for finetuning',
        'final_test_policy':'save all direct and finetuned weights/fields, freeze transfer plan, then test',
        'fno_uno_september_forecast_condition':'reanalysis/assimilation fields and contemporaneous train observations; retrospective fusion',
        'period_separation':'tuning days excluded from all direct-vs-finetune evaluation'}
    (a.output/'cross_month_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    # Independent public regional downloads; no global data or new credentials.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={name:pool.submit(fun,root) for name,fun in [('station',ground),('geos',geos),('era5',era)]}
        results={name:job.result() for name,job in jobs.items()}
    manifest={name:{'path':path,'sha256':digest(path),'bytes':Path(path).stat().st_size} for name,path in results.items()}
    (a.output/'september'/'download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SEPTEMBER_DATA_READY',manifest,flush=True)


if __name__=='__main__':main()
