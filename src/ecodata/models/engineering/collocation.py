"""Format-neutral point/grid collocation; no radius, dates or sites are fixed."""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from ecodata.algorithms.collocated import local_residual


def process_collocations(background,times,x,y,observations,transform,lon,lat,*,variable,unit,support_mask=None):
    times=pd.DatetimeIndex(times);x=np.asarray(x);y=np.asarray(y)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('原生时刻缺失、重复或非递增')
    if len(x)<2 or len(y)<2 or not np.all(np.diff(x)>0) or not np.all(np.diff(y)>0):raise ValueError('缺少递增二维中心网格')
    if np.shape(background)!=(len(times),len(y),len(x)):raise ValueError('背景时空维度与坐标不一致')
    if support_mask is not None and np.shape(support_mask)!=(len(y),len(x)):raise ValueError('有效域掩码形状不一致')
    records=[];rejected=[];occupied=set()
    for row in observations.itertuples():
        ti=times.get_indexer([row.time])[0]
        if ti<0:rejected.append(dict(station=row.station,time=str(row.time),reason='not an exact native instant'));continue
        px,py=transform(row.longitude,row.latitude)
        if not (x[0]<=px<=x[-1] and y[0]<=py<=y[-1]):
            rejected.append(dict(station=row.station,time=str(row.time),reason='no finite interpolation stencil; no extrapolation'));continue
        ci=int(np.argmin(abs(x-px)));ri=int(np.argmin(abs(y-py)));cell=ri*len(x)+ci
        if support_mask is not None and not support_mask[ri,ci]:
            rejected.append(dict(station=row.station,time=str(row.time),reason='outside computational support; buffer preserved'));continue
        if (ti,cell) in occupied:raise ValueError('同原生格点同一时刻有多条观测，缺少聚合契约，不平均或猜选')
        sampled=float(RegularGridInterpolator((y,x),background[ti],bounds_error=True)([[py,px]])[0])
        if not np.isfinite(sampled) or not np.isfinite(background[ti,ri,ci]):
            rejected.append(dict(station=row.station,time=str(row.time),reason='missing native interpolation support'));continue
        occupied.add((ti,cell));records.append(dict(ti=ti,cell=cell,station=row.station,time=str(row.time),
            observed=float(row.value),sampled=sampled,background=float(background[ti,ri,ci]),longitude=float(lon[ri,ci]),latitude=float(lat[ri,ci])))
    if not records:return None,dict(status='NO_COLLOCATED_SUPPORT',rejected=rejected)
    ds=xr.Dataset({variable:('sample',[r['background'] for r in records],dict(units=unit,physical_variable=variable,
        temporal_semantics='instantaneous',spatial_semantics='grid_estimate',data_kind='continuous'))},
        coords=dict(time=('sample',pd.to_datetime([r['time'] for r in records])),
            lat=('sample',[r['latitude'] for r in records]),lon=('sample',[r['longitude'] for r in records])),attrs=dict(crs='EPSG:4326'))
    ds.lat.attrs['units']='degrees_north';ds.lon.attrs['units']='degrees_east'
    out,report=local_residual(ds,np.array([r['observed'] for r in records]),np.array([r['sampled'] for r in records]),variable)
    ti=np.unique([r['ti'] for r in records]);cells=np.unique([r['cell'] for r in records])
    values=background[ti].reshape(len(ti),-1)[:,cells].copy()
    for r,value in zip(records,out[variable].values):values[np.searchsorted(ti,r['ti']),np.searchsorted(cells,r['cell'])]=value
    return (values,ti,cells,out),dict(report,status='LOCAL_RESIDUAL_EXECUTED',records=records,rejected=rejected,
        processed_observations=len(records),time_policy='exact native timestamps only; off-hour reports not shifted',
        target_scope='one nearest native centre inside its valid interpolation support; no wider bias field')
