"""Read published product identity without inventing a complete fusion contract.

These adapters identify products, not independent errors, model-day alignment,
or a preferred processing algorithm. Original arrays and attributes stay intact.
"""
import numpy as np
import pandas as pd

POWER_DOC='https://power.larc.nasa.gov/docs/services/api/temporal/daily/'
CMORPH_DOC='https://www.ncei.noaa.gov/products/climate-data-records/precipitation-cmorph'


def inspect_product(ds):
    attrs=ds.attrs
    power=(attrs.get('source')=='Prediction Of Worldwide Energy Resource (POWER)'
           and 'NASA Langley' in str(attrs.get('institution','')))
    cmorph=('CMORPH' in str(attrs.get('title','')) and attrs.get('naming_authority')=='gov.noaa.ncei'
            and 'NOAA' in str(attrs.get('institution','')))
    if not (power or cmorph):return None
    family='NASA_POWER' if power else 'NOAA_CMORPH'
    ancillary={n for a in ds.data_vars.values() for n in str(a.attrs.get('ancillary_variables','')).split()}
    fields=[n for n,a in ds.data_vars.items() if n not in ancillary and ((
        power and a.attrs.get('standard_name')=='Precipitation_Corrected'
        and 'MERRA-2' in str(a.attrs.get('definition',''))) or (
        cmorph and a.attrs.get('standard_name')=='lwe_precipitation_rate'))]
    evidence={k:attrs[k] for k in ('source','institution','project','title','naming_authority',
        'version','product_version','id','time_coverage_start','time_coverage_end','time_coverage_resolution') if k in attrs}
    result=dict(product=family,source_id=family+':'+str(attrs.get('product_version',attrs.get('version','unspecified'))),
        evidence=evidence,documentation=POWER_DOC if power else CMORPH_DOC,
        independence='NOT_ESTABLISHED: product identity does not establish independent errors',fields=[])
    for name in fields:
        a=ds[name];issues=[]
        def issue(code,message):issues.append(dict(code=code,message=message))
        from ecodata.engines.units import UnitEngine
        from ecodata.core.exceptions import SchemaError
        try:UnitEngine().affine(a.attrs.get('units',''),'mm/day')
        except SchemaError:issue('PRODUCT_UNIT_UNSUPPORTED','产品单位不能按量纲转换为降水率mm/day，未猜测水密度或物理定义')
        if set(a.dims)!={'time','lat','lon'}:issue('GRID_SUPPORT_UNCONFIRMED','降水维度不是已核查的time/lat/lon支持，未按数组大小猜测')
        coordinates={}
        for axis,unit in [('lat','degrees_north'),('lon','degrees_east')]:
            if axis not in ds or ds[axis].ndim!=1 or ds[axis].attrs.get('units')!=unit:
                issue('GRID_SUPPORT_UNCONFIRMED','缺少明确的一维地理坐标及角度单位：'+axis);continue
            values=ds[axis].values
            if not np.isfinite(values).all() or not np.all(np.diff(values)>0):
                issue('GRID_SUPPORT_UNCONFIRMED','产品坐标非有限或非递增：'+axis)
            coordinates[axis]=dict(min=float(np.min(values)),max=float(np.max(values)),units=unit,
                bounds=ds[axis].attrs.get('bounds'),size=len(values))
        time_basis=a.attrs.get('time_basis',attrs.get('time_basis','UNKNOWN'))
        if cmorph and time_basis not in {'UNKNOWN','UTC'}:
            issue('TIME_BASIS_CONFLICT','声明的时间基准与CMORPH官方UTC覆盖冲突')
        if 'time' not in ds:
            issue('TIME_SUPPORT_UNCONFIRMED','缺少时间坐标')
            times=pd.DatetimeIndex([])
        else:
            times=pd.DatetimeIndex(ds.time.values)
            if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
                issue('TIME_SUPPORT_UNCONFIRMED','时间轴缺失、重复或顺序错误')
        bounds=ds.time.attrs.get('bounds') if 'time' in ds else None
        if cmorph:
            # An inclusive 23:59:59 endpoint is retained, not silently stretched.
            start=str(attrs.get('time_coverage_start',''));end=str(attrs.get('time_coverage_end',''))
            if not (start.endswith('T00:00:00Z') and end.endswith('T23:59:59Z')
                    and attrs.get('time_coverage_resolution')=='P1D' and bounds in ds):
                issue('TIME_SUPPORT_UNCONFIRMED','CMORPH UTC日覆盖或原始时间边界不完整')
            else:
                edges=np.asarray(ds[bounds].values)
                valid=(edges.shape==(len(times),2) and len(times)>0)
                if valid:
                    valid=(np.array_equal(edges[:,0],times.values)
                        and np.all(edges[:,1]-edges[:,0]==np.timedelta64(86399,'s'))
                        and pd.Timestamp(start)==pd.Timestamp(edges[0,0]).tz_localize('UTC')
                        and pd.Timestamp(end)==pd.Timestamp(edges[-1,1]).tz_localize('UTC'))
                if valid:time_basis='UTC'
                else:issue('TIME_SUPPORT_UNCONFIRMED','CMORPH实际时间边界与坐标/P1D覆盖声明冲突；未改变原始区间')
        elif time_basis not in {'UTC','LST'}:
            issue('SOURCE_DAY_UNCONFIRMED','POWER日产品可为UTC或LST；当前文件未声明，需提供与此文件对应的原始请求/时间基准证据')
        daily=(cmorph and not any(b['code']=='TIME_SUPPORT_UNCONFIRMED' for b in issues)) or (
            power and 'time' in ds and ds.time.attrs.get('delta')=='0000-00-01 00:00:00')
        if daily and a.attrs.get('interval_seconds',86400)!=86400:
            issue('INTERVAL_CONFLICT','声明的区间长度与官方日产品不一致')
        if daily and a.attrs.get('temporal_semantics') not in {None,'mean','mean_rate','accumulation'}:
            issue('TEMPORAL_SEMANTICS_CONFLICT','声明的时间统计与已核查日产品不一致')
        if not daily and not a.attrs.get('temporal_semantics'):
            issue('RATE_ACCUMULATION_CONTRACT_REQUIRED','原始单位mm/day：需核实日平均降水率/日累计定义与区间积分契约；不能仅改单位或添加字段当作转换完成')
        values=np.asarray(a.values)
        if np.any(values[np.isfinite(values)]<0):issue('INVALID_PRECIPITATION','解码后的有效降水为负；未补零或删除记录')
        ancillary=str(a.attrs.get('ancillary_variables','')).split()
        if any(any(k in v.attrs for k in ('flag_values','flag_masks')) for v in ds.data_vars.values()):
            issue('QA_RULE_UNCONFIRMED','来源带质量标记，但尚未绑定其科学处理规则，不能忽略后融合')
        for n in ancillary:
            if n not in ds or ds[n].attrs.get('uncertainty_type')!='standard_deviation':continue
            u=ds[n]
            try:UnitEngine().affine(u.attrs.get('units',''),a.attrs.get('units',''))
            except SchemaError:issue('UNCERTAINTY_UNIT_UNCONFIRMED','标准差单位缺失或与原始降水率不兼容')
            if u.dims!=a.dims:issue('UNCERTAINTY_SUPPORT_UNCONFIRMED','标准差与降水的时空支持不同，不能默认广播或插值')
        result['fields'].append(dict(field=name,quantity='precipitation',unit=a.attrs.get('units'),
            original_attributes=dict(a.attrs),time_basis=time_basis,
            interval_seconds=86400 if daily else None,
            value_semantics='daily_mean_precipitation_rate' if daily else 'UNCONFIRMED',
            time_start=str(times.min()) if len(times) else None,time_end=str(times.max()) if len(times) else None,
            time_bounds=bounds,coordinates=coordinates,missing_cells=int((~np.isfinite(values)).sum()),
            finite_cells=int(np.isfinite(values).sum()),blockers=issues))
    if not fields:result['blockers']=[dict(code='PRODUCT_FIELD_UNCONFIRMED',message='产品提供方已识别，但无已核实的降水字段语义')]
    return result
