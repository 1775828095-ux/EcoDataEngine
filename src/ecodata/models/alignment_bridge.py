"""Common evidence-preserving spatial/time adapter before numerical fusion.

No fusion weights or algorithms are selected here. Coordinates are transformed,
not relabelled. Time interpolation is restricted to declared instantaneous data.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import xarray as xr


def can_keep_unchanged(dataset,variable,request,contract=None):
    """Complete values alone do not mean they already match the target."""
    from pyproj import CRS
    values=dataset[variable].values
    if not np.isfinite(values).all():
        # A reviewed native writer may explicitly preserve the source mask.
        # This never authorises filling missing cells or dropping the check.
        if (variable not in (contract or {}).get('allow_missing_fields', [])
                or dataset[variable].attrs.get('missing_policy') != 'retain source mask; WPS encodes missing as -1e30'
                or not np.isfinite(values).any() or np.isinf(values).any()):
            return False
    target_crs=(contract or {}).get('crs')
    if target_crs:
        if not dataset.attrs.get('crs') or not CRS.from_user_input(dataset.attrs['crs']).equals(CRS.from_user_input(target_crs)):
            return False
    for key,axis in [('target_lat','lat'),('target_lon','lon'),('target_times','time')]:
        if key not in request:continue
        if axis not in dataset or dataset[axis].ndim!=1:return False
        expected=np.asarray(request[key],dtype='datetime64[ns]' if axis=='time' else None)
        if not np.array_equal(dataset[axis].values,expected):return False
    return True


def spatial(dataset, variable):
    from pyproj import CRS, Transformer
    ds=dataset.copy(deep=True); operations=[]
    if not ds.attrs.get('crs'):
        raise ValueError('缺少坐标系或原生投影到坐标系的可靠解释，不能猜测')
    crs=CRS.from_user_input(ds.attrs['crs']); target=CRS.from_epsg(4326)
    if {'x','y'} <= set(ds.coords) and crs.is_projected:
        lon,lat=ds.x,ds.y
    elif {'lat','lon'} <= set(ds.coords):
        lon,lat=ds.lon,ds.lat
        if crs.is_projected and any('degree' in str(c.attrs.get('units','')).lower() for c in (lon,lat)):
            raise ValueError('投影坐标系与经纬度坐标单位冲突，需明确坐标所用基准')
    else:
        raise ValueError('缺少可对应数据维度的空间坐标')
    if lon.ndim==lat.ndim==1 and lon.dims!=lat.dims:
        yy,xx=xr.broadcast(lat,lon)
    elif lon.dims==lat.dims and lon.ndim in (1,2):
        xx,yy=lon,lat
    else:
        raise ValueError('空间坐标的维度不匹配')
    if not set(xx.dims)<=set(ds[variable].dims):
        raise ValueError('坐标维度与处理字段不对应')
    if not crs.equals(target):
        transform=Transformer.from_crs(crs,target,always_xy=True,allow_ballpark=False)
        x,y=transform.transform(xx.values,yy.values,errcheck=True)
        xx=xr.DataArray(x,dims=xx.dims);yy=xr.DataArray(y,dims=yy.dims)
        operations.append(dict(operation='coordinate_transform',source_crs=crs.to_string(),
            target_crs='EPSG:4326',ballpark_allowed=False,description=transform.description))
    elif bool(((xx>180)&(xx<=360)).any()) and bool(((xx>=0)&(xx<=360)).all()):
        xx=((xx+180)%360)-180
        ds=ds.assign_coords(lon=(ds.lon.dims,((ds.lon.values+180)%360)-180))
        if ds.lon.dims==('lon',):
            if len(np.unique(ds.lon.values))!=ds.lon.size:
                raise ValueError('经度规范化后出现重复位置，不能自动合并')
            ds=ds.sortby('lon')
        operations.append(dict(operation='longitude_convention',resampling=False))
    if not np.isfinite(xx).all() or not np.isfinite(yy).all() or (abs(xx)>180).any() or (abs(yy)>90).any():
        raise ValueError('坐标转换结果无效或超出经纬度范围')
    # Preserve an already geographic rectilinear grid without changing support.
    if crs.equals(target) and ds.lon.dims==('lon',) and ds.lat.dims==('lat',):
        return ds,operations
    dims=xx.dims
    ds=ds.assign_coords(lon=(dims,xx.values),lat=(dims,yy.values))
    if len(dims)==2:
        if ds[variable].attrs.get('spatial_semantics') not in {'grid_estimate','point'}:
            raise ValueError('二维单元平均或总量需要保守重网格与单元边界，不能当作点值')
        before=ds[variable].values.copy()
        ds=ds.stack(location=dims).reset_index('location',drop=True)
        # This changes indexing only, not spatial resolution or physical meaning.
        ds.attrs['native_grid_shape']=json.dumps([dataset.sizes[d] for d in dims])
        operations.append(dict(operation='curvilinear_to_paired_locations',original_dims=list(dims),
            source_values=int(before.size),resampling=False,missing_mask_preserved=True))
    ds.attrs.update(crs='EPSG:4326',source_crs=crs.to_string())
    ds.lon.attrs['units']='degrees_east';ds.lat.attrs['units']='degrees_north'
    return ds,operations


def temporal(dataset, variable, target_times):
    ds=dataset.copy(deep=True)
    if 'time' not in ds[variable].dims:
        raise ValueError('动态融合字段缺少时间轴')
    clock=pd.DatetimeIndex(ds.time.values); target=pd.DatetimeIndex(target_times)
    if clock.hasnans or target.hasnans or clock.has_duplicates or not clock.is_monotonic_increasing or target.has_duplicates or not target.is_monotonic_increasing:
        raise ValueError('时间轴必须递增且不重复')
    if not len(target) or not len(clock) or target.min()<clock.min() or target.max()>clock.max():
        raise ValueError('数据未覆盖目标时段；禁止时间外推')
    if target.isin(clock).all():
        return ds.sel(time=target),dict(operation='exact_time_selection',interpolation=False)
    semantics=ds[variable].attrs.get('temporal_semantics')
    if semantics!='instantaneous' or len(clock)<2:
        raise ValueError('只有明确的瞬时连续数据可内插；日分析、平均、累计量不得改时间标签')
    if ds[variable].attrs.get('data_kind','continuous')!='continuous':
        raise ValueError('分类时间数据不能线性内插')
    # Same xarray interpolation backend used by the existing public spatial task.
    out=ds.interp(time=target,method='linear')
    return out,dict(operation='instantaneous_time_interpolation',backend='xarray.interp',extrapolation=False)


def align_inputs(inputs,request,folder):
    """All automatic model routes use this gate; return freshly hashed inputs."""
    from ecodata.algorithms.platform_search import parse_inputs, SearchInput
    from ecodata.engines.units import UnitEngine
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    parsed,audit=parse_inputs(inputs)
    field=request['variable'];issues=[];result=[];records=[]
    primary=[ds for role,ds in parsed if role=='primary']
    if len(primary)!=1:raise ValueError('必须能唯一确定目标资料')
    times=request.get('target_times',primary[0].time.values if 'time' in primary[0] else [])
    dynamic=['time' in ds[field].dims for role,ds in parsed if role in {'primary','background'} and field in ds]
    if any(dynamic) and not all(dynamic):
        issues.append(dict(source='all',stage='time_structure',message='静态场与动态时间序列不能默认混合'))
    semantics={ds[field].attrs.get('temporal_semantics') for role,ds in parsed if role in {'primary','background'} and field in ds}
    if any(dynamic) and (len(semantics)!=1 or None in semantics):
        issues.append(dict(source='all',stage='time_semantics',message='各来源时间含义不同或未声明，不能自动混合：'+str(sorted(map(str,semantics)))))
    for index,((role,ds),item,source) in enumerate(zip(parsed,audit,inputs)):
        record=dict(source=source.filename,input_sha256=item['sha256'],operations=[])
        try:
            if role not in {'primary','background'}:
                raise ValueError('此适配入口尚未定义该资料角色的对齐方式：'+role)
            ds.attrs.setdefault('source_id','sha256:'+item['sha256'])
            ds[field]=UnitEngine().convert(ds[field],request['unit'])
            ds,ops=spatial(ds,field);record['operations'].extend(ops)
            if 'time' in ds[field].dims:
                ds,operation=temporal(ds,field,times);record['operations'].append(operation)
            path=folder/f'aligned_{index}.nc';ds.to_netcdf(path,engine='h5netcdf')
            payload=path.read_bytes();record.update(output=path.name,output_sha256=hashlib.sha256(payload).hexdigest())
            result.append(SearchInput(path.name,role,payload))
        except Exception as exc:
            issues.append(dict(source=source.filename,stage='alignment',message=str(exc)))
        records.append(record)
    report=dict(status='BLOCK' if issues else 'PASS',sources=records,issues=issues,
        algorithm_selection='unchanged public fusion engine',original_data_modified=False,
        observation_crs='EPSG:4326')
    (folder/'alignment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if issues:
        raise ValueError('自动对齐未通过：'+'；'.join(i['source']+'：'+i['message'] for i in issues))
    return result,report
