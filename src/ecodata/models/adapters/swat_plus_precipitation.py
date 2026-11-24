"""One actual daily gauge -> existing SWAT+ writer, no watershed required."""
import io
import numpy as np
import pandas as pd
import xarray as xr
from .swat_plus import SWATPlusWriter


def export_precipitation(raw,config):
    with xr.open_dataset(io.BytesIO(raw)) as opened:ds=opened.load()
    if ds.attrs.get('crs')!='EPSG:4326':raise ValueError('降水组件缺少明确EPSG:4326坐标基准')
    if 'precipitation' not in ds or 'time' not in ds.coords:raise ValueError('降水组件缺precipitation/time')
    a=ds.precipitation
    if a.attrs.get('units')!='mm' or a.attrs.get('temporal_semantics')!='accumulation' or a.attrs.get('interval_seconds')!=86400:
        raise ValueError('降水组件需要已确认的mm逐日累计及86400秒区间；不能猜测日界或把降水率当累计量')
    if not (a.attrs.get('time_basis') or ds.attrs.get('time_basis')) or (a.attrs.get('time_basis') or ds.attrs.get('time_basis'))=='UNKNOWN':
        raise ValueError('降水组件缺少原始日统计时间基准')
    times=pd.DatetimeIndex(ds.time.values)
    if not len(times) or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing or any(times!=times.normalize()):
        raise ValueError('降水日期标签缺失/重复/非递增或非日标签')
    if len(times)>1 and not np.all(np.diff(times.values)==np.timedelta64(1,'D')):raise ValueError('降水日序列不连续，不能填造日期或观测')
    if pd.Timestamp(config['start'])!=times[0] or pd.Timestamp(config['end'])!=times[-1]:
        raise ValueError('声明的转换日期与实际降水范围不一致，不能忽略目标窗口')
    if a.size!=len(times):raise ValueError('单站降水组件不能猜选多站或网格位置')
    station=config.get('station',{})
    if any(n not in station for n in ('latitude','longitude','elevation')):raise ValueError('缺少必要的站点经纬度/海拔；不能用默认地理位置替代')
    lat,lon,elev=(float(station[n]) for n in ('latitude','longitude','elevation'))
    if not np.isfinite([lat,lon,elev]).all() or not -90<=lat<=90 or not -180<=lon<=180:raise ValueError('站点位置无效')
    values=a.transpose('time',*[d for d in a.dims if d!='time']).values.reshape(-1)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('缺失或负降水不得伪造补齐')
    frame=pd.DataFrame(dict(year=times.year,yday=times.dayofyear,precipitation=values))
    payload=SWATPlusWriter._weather_file(frame,station,'pcp')
    from ecodata.models.swat_weather_project import _records
    _records(payload,'pcp',allow_native_missing=False)
    files={'station.pcp':payload,'pcp.cli':SWATPlusWriter._cli('station.pcp')}
    report=dict(status='PASS',scope='single_precipitation_component',records=len(times),unit='mm',
        unused_source_variables=[n for n in ds.data_vars if n!='precipitation'],
        source_time_basis=a.attrs.get('time_basis',ds.attrs.get('time_basis')),missing_policy='reject',
        complete_model_input=False,native_model_executed=False,
        required_next_steps=['Bind pcp.cli and weather station into the user watershed for full model execution'],
        format_source='https://docs.swat.tamu.edu/62.0.0/input-reference/pcp/')
    return files,report
