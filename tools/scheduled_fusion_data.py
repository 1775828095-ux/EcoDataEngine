"""Frozen representative grammar; observations and withheld labels are separate."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.raster_reconstruction_task import degradation


def case(start,*,synthetic=False,length=17,lat_slice=(6,18),lon_slice=(15,27)):
    v='air_temperature'
    if synthetic:
        lat=np.linspace(30,35,8);lon=np.linspace(110,115,8);times=pd.date_range('2013-01-01',periods=length,freq='6h')
        y,x=np.meshgrid(lat,lon,indexing='ij');t=np.arange(length)[:,None,None]
        values=280+np.sin(x*2)[None]+np.cos(y*2)[None]+t*.3+np.sin(t*.7)*np.sin(x)[None]
    else:
        with xr.open_dataset(ROOT/'data/ncep_xarray/air_temperature.nc') as source:
            ds=source.sortby('lat').isel(time=slice(start,start+length),lat=slice(*lat_slice),lon=slice(*lon_slice)).load()
        values=ds.air.values.astype(float);lat=ds.lat.values;lon=ds.lon.values;times=pd.DatetimeIndex(ds.time.values)
    high=xr.Dataset({v:(('time','lat','lon'),values)},coords=dict(time=times,lat=lat,lon=lon),attrs=dict(crs='EPSG:4326'))
    high[v].attrs=dict(units='K',physical_variable=v,spatial_semantics='cell_mean',temporal_semantics='instantaneous',data_kind='continuous',physical_definition='Controlled cell-mean observation of gridded temperature; not native sensor support')
    low=xr.concat([degradation(high.sel(time=t,drop=True),len(lat)//2,v)[0].expand_dims(time=[t]) for t in times],dim='time')
    if length not in {17,33}:raise ValueError('Frozen two-window fixtures support 17 or 33 frames')
    anchors=list(range(0,length,8));sparse=high.isel(time=anchors);sparse.attrs['source_id']='fine_slow';low.attrs['source_id']='coarse_fast'
    positions=[i for i in range(length) if i not in anchors];truth=high[v].isel(time=positions)
    middle=(length-1)//2
    windows=[dict(id=key,start=str(times[a]),end=str(times[b])) for key,a,b in [('first',0,middle),('second',middle,length-1)]]
    grid=[dict(algorithm='raster_task.spatial.bilinear'),dict(algorithm='raster_task.pchip')]
    median=[dict(algorithm=None),dict(algorithm='spatial.median_filter')]
    req=dict(task='multisource',variable=v,unit='K',semantic_type='continuous',nonnegative=True,
        spatial_semantics='cell_mean',temporal_semantics='instantaneous',target_lat=lat.tolist(),target_lon=lon.tolist(),target_times=[str(t) for t in truth.time.values],
        parameters=dict(target_time_frequency='6h'),model_profile=dict(id='wrf',role='temperature',field=v),
        scoring=dict(registry_version='complementary-state-v1'),portrait_policy=dict(version='complementary-wrf-v1',scenario='coarse-and-detail'),
        composition=dict(reconstruction=dict(fine_source='fine_slow',coarse_source='coarse_fast',evidence='Declared complementary controlled sources'),windows=windows,
            source_stages=dict(fine_slow=[dict(id='time',options=[dict(algorithm='temporal.semantic_resample',parameters=dict(target_time_frequency='6h',temporal_interpolation='linear'))],evidence='Required time reconstruction from supplied 48h anchors')],
                coarse_fast=[dict(id='grid',options=grid,evidence='Two reviewed spatial reconstruction alternatives'),dict(id='median',options=median,evidence='Optional nonlinear filter before or after grid reconstruction; no enforced order')]),
            fusion_options=[dict(algorithm='fusion.primary_weighted_mean',primary=primary,primary_share=.75,evidence='Frozen dominant/support role comparison, not fitted uncertainty') for primary in ('fine_slow','coarse_fast')]+[dict(algorithm='fusion.spatiotemporal_detail',detail_strength=.5,evidence='Previously available fixed half-detail method, tested alongside means; no forced winner')],
            post_stages=[dict(id='post_median',options=median,evidence='Optional fusion-output filter; can hurt structure/observation agreement')],
            search=dict(survivors=2,population=4,generations=100,backend='alns',screen_limit=100000)))
    return [api.SearchInput('fine.nc','primary',sparse.to_netcdf()),api.SearchInput('coarse.nc','background',low.to_netcdf())],req,truth
