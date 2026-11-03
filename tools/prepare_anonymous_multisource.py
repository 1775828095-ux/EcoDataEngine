"""Bounded anonymous NOAA/ERA5 samples, immutable raw files and explicit QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import threading
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import requests
import xarray as xr


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stations', type=Path, required=True)
    p.add_argument('--raw', type=Path)
    p.add_argument('--phase', choices=['sample', 'extend', 'pilot', 'cached', 'recover'], default='sample')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    raw = a.raw or a.output/'original'
    raw.mkdir(exist_ok=True)
    sites = pd.read_csv(a.stations)
    old_manifest=a.output/'satellite_manifest_cached.json'
    previous={r['date']:r for r in json.loads(old_manifest.read_text(encoding='utf-8'))} if old_manifest.exists() else {}
    # Dates fixed before inspecting PM2.5 outcomes; no metric-dependent sampling.
    dates = (pd.to_datetime(['2022-05-15','2022-06-15','2022-07-15']) if a.phase == 'pilot' else
             pd.to_datetime(['2022-01-15','2022-04-15','2022-07-15','2022-10-15']) if a.phase == 'sample'
             else pd.date_range('2022-01-02','2022-12-31',freq='3D'))
    if a.phase=='cached':
        dates=sorted({pd.to_datetime(f.name.split('_s')[1][:7],format='%Y%j') for f in raw.glob('OR_ABI-L2-AODC*.nc')})
    if a.phase=='recover':
        failed=json.loads((a.output/'satellite_manifest_extend.json').read_text(encoding='utf-8'))
        dates=[pd.Timestamp(r['date']) for r in failed if r['status']=='FAIL']
    local=threading.local()
    def fetch(url, params=None):
        if a.phase=='pilot':
            url=requests.Request('GET',url,params=params).prepare().url
            r=subprocess.run(['curl.exe','--http1.1','--fail','--silent','--show-error',
                '--connect-timeout','10','--max-time','40',url],capture_output=True)
            if r.returncode:
                raise RuntimeError(r.stderr.decode(errors='replace'))
            return r.stdout
        if not hasattr(local,'session'):
            local.session=requests.Session()
        r=local.session.get(url,params=params,timeout=(5,20) if a.phase=='recover' else (10,40))
        r.raise_for_status()
        return r.content
    lat = np.arange(np.floor(sites.latitude.min()*4)/4-.25, np.ceil(sites.latitude.max()*4)/4+.26, .25)
    lon = np.arange(np.floor(sites.longitude.min()*4)/4-.25, np.ceil(sites.longitude.max()*4)/4+.26, .25)
    # Regional output retains QA mask; fine grid is not claimed to add information.
    qlat = np.linspace(sites.latitude.min()-.03,sites.latitude.max()+.03,200)
    qlon = np.linspace(sites.longitude.min()-.03,sites.longitude.max()+.03,160)
    def satellite(date):
        record = dict(date=str(date.date()), provider='NOAA NODD', product='GOES17 ABI-L2-AODC',
                      license='NOAA open data; attribution; no implied endorsement')
        old=previous.get(record['date'])
        if a.phase=='cached' and old and old.get('status')=='READ' and (a.output/f'aod_{date:%Y%m%d}.nc').exists():
            if hashlib.sha256(Path(old['file']).read_bytes()).hexdigest()==old['sha256']:
                return old
        try:
            prefix = f'ABI-L2-AODC/{date.year}/{date.dayofyear:03d}/20/'
            url = ('https://noaa-goes17.s3.us-east-1.amazonaws.com/' if a.phase=='recover'
                   else 'https://noaa-goes17.s3.amazonaws.com/')
            listing = raw/f'listing_{date:%Y%m%d}.xml'
            if not listing.exists():
                listing.write_bytes(fetch(url,{'list-type':'2','prefix':prefix,'max-keys':1}))
            tree = ET.fromstring(listing.read_bytes())
            key = tree.find('.//{*}Key')
            if key is None:
                raise ValueError('No actual granule in the fixed UTC 20 hour')
            url += key.text
            file = raw/Path(key.text).name
            if not file.exists():
                file.write_bytes(fetch(url))
            record.update(url=url,file=str(file),bytes=file.stat().st_size,
                          sha256=hashlib.sha256(file.read_bytes()).hexdigest(),accessed_utc=datetime.now(timezone.utc).isoformat())
            with xr.open_dataset(file) as ds:
                pr = ds.goes_imager_projection.attrs
                tx = Transformer.from_crs(4326,CRS.from_cf(pr),always_xy=True)
                def sample(y,x,dims):
                    xx,yy = tx.transform(x,y)
                    return ds[['AOD','DQF']].interp(x=xr.DataArray(xx/pr['perspective_point_height'],dims=dims),
                        y=xr.DataArray(yy/pr['perspective_point_height'],dims=dims),method='nearest').load()
                yy,xx = np.meshgrid(qlat,qlon,indexing='ij')
                grid = sample(yy,xx,('lat','lon'))
                point = sample(sites.latitude.to_numpy(),sites.longitude.to_numpy(),'station')
                limits=np.asarray(ds.AOD.attrs['valid_range'],float)*ds.AOD.encoding.get('scale_factor',1)+ds.AOD.encoding.get('add_offset',0)
                # High OR medium-quality auxiliary retrievals, never low/no retrieval.
                # Retain DQF for strict-high-quality sensitivity analysis.
                mask = np.isin(grid.DQF.values,[0,1]) & np.isfinite(grid.AOD.values) & (grid.AOD.values>=max(0,limits.min())) & (grid.AOD.values<=limits.max())
                pointmask = np.isin(point.DQF.values,[0,1]) & np.isfinite(point.AOD.values) & (point.AOD.values>=max(0,limits.min())) & (point.AOD.values<=limits.max())
                result = xr.Dataset({'aod':(('lat','lon'),np.where(mask,grid.AOD.values,np.nan)),
                                     'aod_qa':(('lat','lon'),grid.DQF.values)},
                                    coords={'lat':qlat,'lon':qlon},attrs={'crs':'EPSG:4326',
                                    'time_coverage_start':ds.attrs['time_coverage_start'],
                                    'time_coverage_end':ds.attrs['time_coverage_end'],'source_url':url,
                                    'information_scale':'GOES native fixed-grid pixels, not target-grid precision'})
                result.aod.attrs.update(units='1',data_kind='continuous')
                result.to_netcdf(a.output/f'aod_{date:%Y%m%d}.nc',engine='h5netcdf')
                station = sites[['station_id']].copy()
                station['date'],station['aod'],station['DQF'] = str(date.date()),np.where(pointmask,point.AOD.values,np.nan),point.DQF.values
                station.to_csv(a.output/f'aod_stations_{date:%Y%m%d}.csv',index=False)
                record.update(status='READ',valid_grid_fraction=float(mask.mean()),valid_stations=int(pointmask.sum()),
                              station_count=len(sites),qa=ds.DQF.attrs['flag_meanings'],accepted_dqf=[0,1],
                              strict_high_quality_stations=int(((point.DQF.values==0)&pointmask).sum()),
                              decoded_valid_range=limits.tolist(),cloud_handling='upstream cloud screening; no-retrieval is NOT measured cloud fraction',
                              time_start=ds.attrs['time_coverage_start'],time_end=ds.attrs['time_coverage_end'])
        except Exception as exc:
            record.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        print(f"AOD {record['date']} {record['status']} {record.get('valid_stations','-')} stations",flush=True)
        return record
    records = list(ThreadPoolExecutor(max_workers=3).map(satellite,dates))
    save_json(a.output/f'satellite_manifest_{a.phase}.json',records)
    print(pd.DataFrame(records).reindex(columns=['date','status','valid_stations','valid_grid_fraction']).to_string(index=False))
    if a.phase=='cached':
        return
    # A single real ERA5 point is the access/array/semantics probe, not a fake grid.
    weather = raw/'era5_access_sample.json'
    if (a.output/'era5_daily_2022.nc').exists():
        return
    if not weather.exists():
        r=requests.get('https://archive-api.open-meteo.com/v1/archive',params=dict(latitude=37,longitude=-120,
            start_date='2022-07-15',end_date='2022-07-16',timezone='UTC',models='era5',
            hourly='temperature_2m,dew_point_2m,surface_pressure,wind_speed_10m,precipitation'),timeout=40)
        r.raise_for_status()
        weather.write_bytes(r.content)
    obj=json.loads(weather.read_bytes())
    save_json(a.output/'weather_access_check.json',dict(provider='Open-Meteo ERA5 access',
        docs='https://open-meteo.com/en/docs/historical-weather-api',license='CC BY 4.0 data; free API noncommercial terms',
        units=obj['hourly_units'],hours=len(obj['hourly']['time']),latitude=obj['latitude'],longitude=obj['longitude'],
        sha256=hashlib.sha256(weather.read_bytes()).hexdigest(),scope='real single-point access sample, not yet regional coverage'))


if __name__=='__main__':
    main()
