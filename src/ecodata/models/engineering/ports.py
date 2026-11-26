"""Model format drivers expose scientific input ports, not algorithm choices."""
from dataclasses import dataclass
from pathlib import PurePosixPath
import numpy as np
import pandas as pd
import xarray as xr

from ecodata.models.scientific_schema import identity


@dataclass
class InputPort:
    resource: str
    variable: str
    quantity: str
    unit: str
    temporal: str
    kind: str
    data: xr.Dataset
    binding: dict


# Published native format vocabulary. No algorithms, cases, dates or preferences.
NATIVE_QUANTITIES={
 'wrf':{'T2':'air_temperature','PSFC':'air_pressure','Q2':'humidity_mixing_ratio','SST':'sea_surface_temperature'},
 'camx':{'t2':'air_temperature','psfc':'air_pressure'},
 'cmaq':{'TEMP2':'air_temperature','PRES':'air_pressure','PRSFC':'air_pressure','LAI':'leaf_area_index','WSPD10':'wind_speed'}}
NATIVE_SUPPORT={
 'wrf':{'T2':2,'Q2':2,'U10':10,'V10':10,'PSFC':0,'SST':0},
 'camx':{'t2':2,'u10':10,'v10':10},
 'cmaq':{'TEMP2':2,'PRSFC':0,'WSPD10':10}}
WEATHER={'pcp': [('precipitation','mm','accumulation')],
 'tmp':[('air_temperature_max','degC','maximum'),('air_temperature_min','degC','minimum')],
 'slr':[('solar_radiation','MJ/m**2/day','mean')], 'hmd':[('relative_humidity','1','mean')],
 'wnd':[('wind_speed','m/s','mean')]}


def canonical(values,times,lon,lat,quantity,unit,temporal):
    xy=np.column_stack([np.asarray(lon).ravel(),np.asarray(lat).ravel()])
    a=np.asarray(values).reshape(len(times),len(xy))
    return xr.Dataset({quantity:(('time','point'),a,dict(units=unit,physical_variable=quantity,
        physical_definition=quantity,temporal_semantics=temporal,spatial_semantics='grid_estimate',data_kind='continuous'))},
        coords=dict(time=pd.DatetimeIndex(times),point=np.arange(len(xy)),lon=('point',xy[:,0]),lat=('point',xy[:,1])),attrs=dict(crs='EPSG:4326'))


def grid(ds):
    for lon,lat in [('XLONG','XLAT'),('XLONG_M','XLAT_M')]:
        if {lon,lat}<=set(ds.variables):
            for name in (lon,lat):
                if not np.allclose(ds[name].values,ds[name][0].values,rtol=0,atol=1e-9):
                    raise ValueError('移动网格需要逐时支持域适配，不能复用首时刻坐标')
            return np.asarray(ds[lon][0]),np.asarray(ds[lat][0])
    for lon,lat in [('lon','lat'),('longitude','latitude')]:
        if lon in ds and lat in ds:
            x,y=ds[lon].values,ds[lat].values
            if ds[lon].dims!=ds[lat].dims and np.ndim(x)==np.ndim(y)==1:return np.meshgrid(x,y)
            return x,y
    from ecodata.models.adapters.camx_emissions import _grid
    a=ds.attrs;kind=int(a.get('GDTYP',-1))
    if kind not in (1,2):raise ValueError('工程网格缺少已支持的经纬度/LCC坐标契约')
    if kind==2 and float(a.get('P_GAM',a['XCENT']))!=float(a['XCENT']):
        raise ValueError('现有球体LCC读入器不支持P_GAM与XCENT不同，不能猜转换')
    _,_,_,lon,lat=_grid(dict(projection='LATLON' if kind==1 else 'LCC',ncols=int(a['NCOLS']),nrows=int(a['NROWS']),
        xorig=a['XORIG'],yorig=a['YORIG'],xcell=a['XCELL'],ycell=a['YCELL'],coordinate_units='degree' if kind==1 else 'm',
        xcent=a.get('XCENT',0),ycent=a.get('YCENT',0),p_alp=a.get('P_ALP',0),p_bet=a.get('P_BET',0),itzon=0))
    return lon,lat


def spatial_dimensions(ds):
    for lon,lat in [('XLONG','XLAT'),('XLONG_M','XLAT_M')]:
        if {lon,lat}<=set(ds.variables):return ds[lon].dims[1:]
    for lon,lat in [('lon','lat'),('longitude','latitude')]:
        if lon in ds and lat in ds:
            return ds[lon].dims if ds[lon].dims==ds[lat].dims else (*ds[lat].dims,*ds[lon].dims)
    for pair in [('ROW','COL'),('y','x')]:
        if set(pair)<=set(ds.dims):return pair
    raise ValueError('缺少与实际网格对应的空间维度声明')


def netcdf_ports(files,model,metadata):
    from ecodata.models.native_meteorology import native_times
    from ecodata.models.wrf_meteorology import wrf_times
    # Resource roles come from actual active control edges, or explicit input
    # metadata when a shell's dynamic activation cannot be statically known.
    roles={n for n,m in metadata.items() if m.get('role')=='model_input'}
    # The caller supplies existing binding facts; no host files are followed.
    for m in metadata.values():
        roles.update(m.get('_active_resources',[]))
    ports=[];unknown=[]
    for name in sorted(roles):
        if not files.prefix(name,8).startswith((b'CDF',b'\x89HDF')):continue
        with files.netcdf(name,decode_times=False) as opened:
            ds=opened.load()
        from ecodata.models.native_source_inspection import native_container_evidence
        native_container_evidence(model,ds.attrs,ds.variables)
        times=wrf_times(ds.Times.values) if model=='wrf' else native_times(ds)
        if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError('原生时间轴缺失、重复或非递增')
        missing=set(metadata.get(name,{}).get('variables',{}))-set(ds.variables)
        if missing:raise ValueError('工程语义清单引用不存在的变量：'+str(sorted(missing)))
        lon,lat=grid(ds)
        spatial=spatial_dimensions(ds)
        time_dim=ds['Times' if model=='wrf' else 'TFLAG'].dims[0]
        for variable in sorted(ds.data_vars):
            a=ds[variable]
            declared=metadata.get(name,{}).get('variables',{}).get(variable,{})
            for key,value in declared.items():
                if key in a.attrs and a.attrs[key]!=value:
                    raise ValueError('工程声明与原生字段元数据冲突：'+variable+'.'+key)
            attrs=dict(a.attrs,**declared)
            height=NATIVE_SUPPORT.get(model,{}).get(variable)
            if height is not None:
                known=dict(height_m=height,vertical_reference='surface' if height==0 else 'above_ground')
                for key,value in known.items():
                    if key in attrs and attrs[key]!=value:raise ValueError('声明与原生字段垂直支持冲突：'+variable)
                    attrs[key]=value
            quantity,basis=identity(variable,attrs)
            # Wind rotation and mixing ratio are scientific definitions, not
            # spelling aliases. Component orientation requires explicit facts.
            if variable in {'U10','V10','u10','v10'} and basis not in {'standard_name','physical_variable'}:
                unknown.append(dict(resource=name,field=variable,reason='风分量方向基准未声明，不能猜成地理东西/南北风'));continue
            if variable in {'U10','V10','u10','v10'} and not attrs.get('vector_reference'):
                unknown.append(dict(resource=name,field=variable,reason='缺少风分量方向/投影基准契约'));continue
            known_quantity=NATIVE_QUANTITIES.get(model,{}).get(variable)
            if known_quantity and basis in {'standard_name','physical_variable'} and quantity!=known_quantity:
                raise ValueError('声明与已知原生物理变量定义冲突：'+variable)
            quantity=quantity or NATIVE_QUANTITIES.get(model,{}).get(variable)
            if not quantity or not attrs.get('units') or a.dtype.kind not in 'iuf':continue
            if a.size!=len(times)*np.size(lon) or a.dims[0]!=time_dim or a.dims[-len(spatial):]!=spatial:
                unknown.append(dict(resource=name,field=variable,reason='非二维时间场；需要明确垂直/边界支持适配'));continue
            temporal=attrs.get('temporal_semantics')
            if not temporal:
                unknown.append(dict(resource=name,field=variable,reason='缺时间统计语义，未默认瞬时量'));continue
            decoded=xr.decode_cf(ds[[variable]],decode_times=False)[variable].values
            unit=str(a.attrs['units']).strip()
            ports.append(InputPort(name,variable,quantity,unit,temporal,'netcdf',
                canonical(decoded,times,lon,lat,quantity,unit,temporal),
                dict(shape=list(a.shape),dimensions=list(a.dims),sha256=files.identity(name)['sha256'])))
            for key in ['height_m','vertical_reference','vector_reference','physical_definition','data_kind','interval_seconds','spatial_semantics']:
                if key in attrs:ports[-1].data[quantity].attrs[key]=attrs[key]
            ports[-1].data.attrs['time_basis']=attrs.get('time_basis',ds.attrs.get('time_basis','UNKNOWN'))
    return ports,unknown


def weather_ports(files,metadata=None,*,requested_quantities=None):
    from ecodata.models.swat_weather_project import _layout,_table,_weather_metadata,_records
    roots=[PurePosixPath(n).parent for n in files.files if PurePosixPath(n).name=='file.cio']
    if len(roots)!=1:raise ValueError('SWAT+输入端口需要唯一实际file.cio工程')
    root=roots[0];payloads={}
    for key in ['file.cio']:
        with files.files[(root/key).as_posix()][0]() as s:payloads[key]=s.read()
    # Existing layout parser validates the climate and simulation references.
    for line in payloads['file.cio'].decode('utf-8-sig').splitlines()[1:]:
        for token in line.split()[1:]:
            path=(root/token).as_posix()
            if path in files.files:
                with files.files[path][0]() as s:payloads[token]=s.read()
    _,_,climate,_=_layout(payloads);_,header,stations=_table(payloads[climate[1]])
    ports=[];unknown=[];missing_resources=set()
    for station in stations:
        for kind,variables in WEATHER.items():
            if kind not in header:continue
            filename=station[header.index(kind)];name=(root/filename).as_posix()
            if filename in {'null','sim','0'}:continue
            if name not in files.files:raise ValueError('天气端口缺少实际引用：'+name)
            with files.files[name][0]() as s:raw=s.read()
            info=_weather_metadata(raw)
            try:times=_records(raw,kind,allow_native_missing=True)
            except ValueError as exc:
                if requested_quantities is None or any(q in requested_quantities for q,_,_ in variables):raise
                if not any(i.get('resource')==name for i in unknown):
                    unknown.append(dict(resource=name,code='NON_TARGET_WEATHER_INVALID',
                        reason=str(exc)+'；非本次来源匹配变量，原文件保留，未修正或认证原生模拟可运行'))
                continue
            rows=[r.split() for r in raw.decode('utf8').splitlines()[3:] if r.strip()]
            for i,(quantity,unit,temporal) in enumerate(variables):
                values=np.array([float(r[i+2]) for r in rows])[:,None]
                missing=values<=-97
                values=np.where(missing,np.nan,values)
                ports.append(InputPort(name,quantity,quantity,unit,temporal,'swat_weather',
                    canonical(values,pd.to_datetime(times),[info['longitude']],[info['latitude']],quantity,unit,temporal),
                    dict(column=i+2,kind=kind,station=station[header.index('name')],sha256=files.identity(name)['sha256'])))
                ports[-1].data[quantity].attrs['interval_seconds']=86400
                declared=(metadata or {}).get(name,{}).get('variables',{}).get(quantity,{})
                ports[-1].data.attrs['time_basis']=declared.get('time_basis','model_calendar_day')
                ports[-1].binding['native_missing_cells']=int(missing.sum())
                if missing.any() and name not in missing_resources:
                    missing_resources.add(name)
                    unknown.append(dict(resource=name,code='NATIVE_MISSING_PRESERVED',missing_cells=int(missing.sum()),
                        reason='原生<=-97缺测保留为掩码，未补零/删日期；是否由WGN生成需原生模型验证'))
    if missing_resources:
        from ecodata.models.swat_project_dependencies import weather_dependencies
        dependencies=weather_dependencies(payloads)
        unknown.append(dict(code='WGN_DEPENDENCY_CHECK',details=dependencies,native_model_executed=False,
            reason='仅检查作者WGN引用及月统计结构；不等于发生器已执行或模拟已通过'))
    return ports,unknown


def surface_ports(files,metadata=None):
    from ecodata.models.project_inventory import _calpuff
    from existing_air_converters.calpuff_native_bundle import read_surface,FIELDS,UNITS
    plans=_calpuff(files);targets={b['target'] for p in plans for b in p['bindings'] if b.get('parameter')=='SRFDAT' and b['status']=='BOUND'}
    if not targets:
        targets={n for n,m in (metadata or {}).items() if m.get('role')=='model_input'
                 and files.prefix(n,32).startswith(b'SURF.DAT')}
    if len(targets)!=1:raise ValueError('CALPUFF输入端口需要唯一激活SRFDAT')
    name=targets.pop()
    with files.files[name][0]() as s:raw=s.read()
    values,times,stations,zone=read_surface(raw)
    # Native station identifiers are an exact support contract. Geographic
    # matching needs actual station coordinates, never an invented lat/lon.
    ports=[]
    quantities=dict(temperature='air_temperature',station_pressure='air_pressure')
    for i,(field,unit) in enumerate(zip(FIELDS,UNITS)):
        quantity=quantities.get(field,field)
        d=xr.Dataset({quantity:(('time','station_id'),values[:,:,i],dict(units=unit,physical_variable=quantity,
            temporal_semantics='hour_end_observation',data_kind='categorical' if field=='precip_code' else 'circular' if field=='wind_direction' else 'continuous'))},
            coords=dict(time=[pd.Timestamp(t).tz_convert('UTC').tz_localize(None) for t in times],station_id=stations),
            attrs=dict(time_basis='UTC',spatial_support='native_station_identifiers'))
        ports.append(InputPort(name,field,quantity,unit,'hour_end_observation','calmet_surface',d,
            dict(index=i,stations=stations,zone=zone,sha256=files.identity(name)['sha256'])))
    return ports,[]
