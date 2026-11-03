"""Explicit ERA5 anonymous access; no automatic model blend or elevation correction."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]/'output/three_experiments_20261019/anonymous'


def main():
    raw=ROOT/'original'
    raw.mkdir(parents=True,exist_ok=True)
    lat=np.arange(35.,42.)
    lon=np.arange(-123.,-117.)
    nodes=[(y,x) for y in lat for x in lon]
    records=[]
    arrays=[]
    for start in range(0,len(nodes),6):
        batch=nodes[start:start+6]
        file=raw/f'era5_grid_batch_{start:02d}.json'
        params=dict(latitude=','.join(str(y) for y,x in batch),longitude=','.join(str(x) for y,x in batch),
            start_date='2022-01-01',end_date='2022-12-31',models='era5',timezone='Etc/GMT+8',
            elevation=','.join('nan' for _ in batch),
            hourly='temperature_2m,dew_point_2m,surface_pressure,wind_speed_10m,precipitation')
        if not file.exists():
            url=requests.Request('GET','https://archive-api.open-meteo.com/v1/archive',params=params).prepare().url
            # Different transport for this previously timed-out request; no retry loop.
            result=subprocess.run(['curl.exe','--http1.1','--fail','--silent','--show-error',
                '--connect-timeout','10','--max-time','45',url],capture_output=True)
            if result.returncode:
                (ROOT/f'era5_batch_{start:02d}_failure.json').write_text(json.dumps(dict(
                    error=result.stderr.decode(errors='replace'),returncode=result.returncode,
                    rule='one curl transport attempt, then at most one requests transport attempt')))
                try:
                    r=requests.get(url,timeout=(10,30))
                    r.raise_for_status()
                    payload=r.content
                except requests.RequestException as exc:
                    print(f'BLOCKED batch {start}: {type(exc).__name__}',flush=True)
                    continue
            else:
                payload=result.stdout
            file.write_bytes(payload)
        objects=json.loads(file.read_bytes())
        assert isinstance(objects,list) and len(objects)==len(batch)
        for (y,x),obj in zip(batch,objects):
            assert abs(obj['latitude']-y)<.26 and abs(obj['longitude']-x)<.26
            assert obj['utc_offset_seconds']==-28800
            table=pd.DataFrame(obj['hourly'])
            table['time']=pd.to_datetime(table.time)
            # All variables are separately retained; precipitation is an hourly
            # amount (previous hour), not an instantaneous concentration.
            daily=table.groupby(table.time.dt.normalize()).agg(
                temperature_2m=('temperature_2m','mean'),dew_point_2m=('dew_point_2m','mean'),
                surface_pressure=('surface_pressure','mean'),wind_speed_10m=('wind_speed_10m','mean'),
                hours=('temperature_2m','count'))
            # Preceding-hour amount belongs to the interval, not the endpoint
            # calendar day. Last day remains missing without next-year 00:00.
            rain=table.groupby((table.time-pd.Timedelta(nanoseconds=1)).dt.normalize()).precipitation.sum(min_count=24)
            daily['precipitation']=rain.reindex(daily.index)
            daily.loc[daily.hours<24,daily.columns!='hours']=np.nan
            daily['lat'],daily['lon']=y,x
            arrays.append(daily.reset_index())
        records.append(dict(file=str(file),params=params,bytes=file.stat().st_size,
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),units=objects[0]['hourly_units'],
            source='https://archive-api.open-meteo.com/v1/archive',model='ERA5 explicitly selected',
            license='Open-Meteo data CC BY 4.0; API free noncommercial terms'))
        print(f'ERA5 actual grid nodes {start+len(batch)}/{len(nodes)}',flush=True)
    table=pd.concat(arrays)
    ds=table.set_index(['time','lat','lon']).to_xarray()
    units={'temperature_2m':'degC','dew_point_2m':'degC','surface_pressure':'hPa','wind_speed_10m':'km h-1','precipitation':'mm'}
    for name,unit in units.items():
        ds[name].attrs.update(units=unit,data_kind='continuous')
    ds.attrs.update(crs='EPSG:4326',time_semantics='fixed local standard UTC-8 day; hourly instant variables averaged, preceding-hour precipitation summed',
        source_grid='ERA5 native 0.25 degree accessed on 1 degree sampling nodes; elevation correction disabled; not 1km information')
    ds.to_netcdf(ROOT/'era5_daily_2022.nc',engine='h5netcdf')
    with xr.open_dataset(ROOT/'era5_daily_2022.nc') as check:
        assert check.sizes['time']==365 and check.temperature_2m.notnull().any()
    (ROOT/'era5_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
