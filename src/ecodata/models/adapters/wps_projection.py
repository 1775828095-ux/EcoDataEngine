"""Explicit spherical Lambert grid for WPS 4.6 intermediate IPROJ=3.

No regridding or wind rotation. Native DX/DY and earth radius are kilometres:
metgrid/src/read_met_module.F multiplies DX/DY by 1000 after the v5 record.
"""
import struct
import numpy as np
from pyproj import CRS, Transformer
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.wps_sst import _text


def lambert_record(dataset, config, axis_reader, unit_checker):
    p=config.get('projection')
    required={'type','earth_radius_m','latitude_origin','standard_longitude','true_latitudes'}
    if not isinstance(p,dict) or set(p)!=required or p['type']!='lambert':
        raise SchemaError('projection需type=lambert、earth_radius_m、latitude_origin、standard_longitude、true_latitudes；其它投影暂不支持')
    try:
        radius=float(p['earth_radius_m']);origin=float(p['latitude_origin']);central=float(p['standard_longitude'])
        parallels=np.asarray(p['true_latitudes'],dtype=float)
    except (ValueError,TypeError) as exc:raise SchemaError('Lambert投影参数必须为有限数值') from exc
    if parallels.shape!=(2,) or not np.isfinite([radius,origin,central,*parallels]).all():
        raise SchemaError('true_latitudes必须为两个有限真纬度')
    if not 6e6<=radius<=7e6 or not -90<origin<90 or not -180<=central<=180 or np.any(np.abs(parallels)>=90) or parallels[0]*parallels[1]<=0:
        raise SchemaError('Lambert球半径/原点/真纬度范围无效；真纬度需同半球且非0')
    expected=CRS.from_proj4(f'+proj=lcc +lat_0={origin} +lon_0={central} +lat_1={parallels[0]} +lat_2={parallels[1]} +R={radius} +units=m +no_defs')
    try:source=CRS.from_user_input(config.get('crs'))
    except Exception as exc:raise SchemaError('Lambert需明确与投影参数一致的球形米制crs') from exc
    if not source.equals(expected,ignore_axis_order=True):
        raise SchemaError('源crs与WPS球形Lambert参数不符；不把椭球投影直接标为球形')
    if dataset.attrs.get('crs'):
        try:matches=CRS.from_user_input(dataset.attrs['crs']).equals(source,ignore_axis_order=True)
        except Exception as exc:raise SchemaError('数据内crs不可解析') from exc
        if not matches:raise SchemaError('数据内crs与配置投影冲突')
    axes=config['coordinates'];values={}
    for role in ('x','y'):
        name=axes[role];raw=axis_reader(dataset,name,role)
        declared=config.get('coordinate_units',{}).get(role,dataset[name].attrs.get('units',''))
        unit_checker(dataset[name].attrs.get('units',''),declared,role)
        if declared not in {'m','km'}:raise SchemaError(f'{role}投影坐标仅接受明确m或km单位')
        scale,offset=unit_transform(declared,'m');values[role]=np.sort(raw*scale+offset)
        if not np.isfinite(values[role]).all():raise SchemaError('投影坐标换算溢出')
    x,y=values['x'],values['y']
    try:lon,lat=Transformer.from_crs(source,source.geodetic_crs,always_xy=True).transform(x[0],y[0],errcheck=True)
    except Exception as exc:raise SchemaError('左下角投影坐标不能反算为经纬度') from exc
    if not np.isfinite([lon,lat]).all() or not -90<lat<90:raise SchemaError('投影网格左下角无效')
    params=[lat,lon,(x[1]-x[0])/1000,(y[1]-y[0])/1000,central,*parallels,radius/1000]
    encoded=np.asarray(params,dtype='f4')
    if not np.isfinite(encoded).all() or encoded[2]<=0 or encoded[3]<=0:raise SchemaError('投影格距不能安全编码为float32')
    return _text('SWCORNER',8)+struct.pack('>8f',*params),len(y),len(x),{
        'iproj':3,'type':'spherical_lambert','source_crs':source.to_string(),
        'start_latitude':lat,'start_longitude':lon,'dx_km':params[2],'dy_km':params[3],
        'earth_radius_km':radius/1000,'coordinate_processing':'axis sorting and unit conversion only; no interpolation',
        'projection_float32_max_absolute_error':float(np.max(np.abs(np.asarray(params)-encoded)))}
