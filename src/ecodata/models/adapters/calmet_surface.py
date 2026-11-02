"""CALMET 5.8.5 formatted SURF.DAT 2.0 (NONE projection) ONLY.

This is one CALMET preprocessing input for a CALPUFF workflow. It is not
CALMET.DAT, emissions, a complete CALMET project, or a runnable CALPUFF case.
Header follows CALMET 5.8.5 RDHD, data follows RDS, and SMERGE 5.57 WRS.
"""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError

FIELDS = ('wind_speed', 'wind_direction', 'ceiling', 'opaque_cover',
          'temperature', 'relative_humidity', 'station_pressure', 'precip_code')
TARGET_UNITS = ('m/s', 'degree', 'hundred_ft', 'tenths', 'K', '%', 'hPa', 'code')


def _numeric(frame, column):
    if column not in frame:
        raise SchemaError(f'缺少字段：{column}；请提供该列，缺失值用空单元格表示')
    try:
        value = pd.to_numeric(frame[column], errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f'{column} 含非数值内容') from exc
    if np.isinf(value).any():
        raise SchemaError(f'{column} 含无穷值')
    return value


def _convert(values, source, field):
    conversions = {
        'wind_speed': {'m/s': (1,0), 'km/h': (1/3.6,0), 'knot': (.5144444444444445,0)},
        'wind_direction': {'degree': (1,0)},
        'ceiling': {'hundred_ft': (1,0), 'm': (1/30.48,0), 'ft': (.01,0)},
        'opaque_cover': {'tenths': (1,0)},
        'temperature': {'K': (1,0), 'degC': (1,273.15), 'degF': (5/9,255.3722222222222)},
        'relative_humidity': {'%': (1,0), 'fraction': (100,0)},
        'station_pressure': {'hPa': (1,0), 'mb': (1,0), 'Pa': (.01,0)},
        'precip_code': {'code': (1,0)},
    }
    if source not in conversions[field]:
        raise SchemaError(f'{field} 单位 {source!r} 不支持；不能猜测或仅改标签')
    scale, offset = conversions[field][source]
    return values * scale + offset


def export_surface(frame, config):
    if config.get('version') != 'CALMET-5.8.5-SURF-2.0':
        raise SchemaError('仅支持 CALMET-5.8.5-SURF-2.0；不混用6/7版')
    zone = config.get('timezone_west_hours')
    if isinstance(zone, bool) or not isinstance(zone, int) or not -12 <= zone <= 12:
        raise SchemaError('timezone_west_hours 必须为 -12..12 整数，西区为正；固定标准时，不采用夏令时')
    if config.get('time_label') != 'hour_end':
        raise SchemaError('必须明确 time_label=hour_end；本模块不猜测或移动小时标签')
    policy = config.get('missing_policy', 'reject')
    if policy not in {'reject', 'preserve'}:
        raise SchemaError('missing_policy 仅允许 reject/preserve')
    cols, units = config.get('columns', {}), config.get('units', {})
    for name in FIELDS:
        if name not in cols or name not in units:
            raise SchemaError(f'需明确配置 {name} 的源字段和单位')
    if len(set(cols[name] for name in FIELDS)) != len(FIELDS):
        raise SchemaError('不同物理变量不能重复映射到同一源字段')
    station_col, time_col = config.get('station_column','station_id'), config.get('time_column','time')
    if station_col not in frame or time_col not in frame or not len(frame):
        raise SchemaError('缺少站点/时间字段或记录为空')
    stations = config.get('station_ids')
    if not stations or len(set(stations)) != len(stations) or any(isinstance(s,bool) or not isinstance(s,int) or not 1 <= s <= 999999 for s in stations):
        raise SchemaError('station_ids 需明确给出不重复整数站点ID和顺序 (1..999999)')
    ids = _numeric(frame, station_col)
    if not np.isfinite(ids).all() or not np.equal(ids, np.rint(ids)).all() or set(ids.astype(int)) != set(stations):
        raise SchemaError('数据站点与配置不一致；不能丢弃未知站点或截断站点ID')
    dates=[]
    for raw in frame[time_col]:
        try:
            value=datetime.fromisoformat(str(raw).replace('Z','+00:00'))
        except ValueError as exc:
            raise SchemaError('时间需有效 ISO8601 日期与明确 UTC 偏移') from exc
        if value.utcoffset() is None:
            raise SchemaError('时间缺少UTC偏移；不猜测本地时区')
        value=value.astimezone(timezone(timedelta(hours=-zone))).replace(tzinfo=None)
        if value.minute or value.second or value.microsecond or not 1900 <= value.year <= 2099:
            raise SchemaError('需整点小时末标签，年份限1900..2099；不静默舍入时间')
        dates.append(value)
    work=pd.DataFrame({'station_id':ids.astype(int),'time':dates})
    missing={};rounding={}
    ranges={'wind_speed':(0,150),'wind_direction':(0,360),'ceiling':(0,999),
            'opaque_cover':(0,10),'temperature':(150,350),'relative_humidity':(0,100),
            'station_pressure':(100,1100),'precip_code':(0,45)}
    for name in FIELDS:
        raw=_numeric(frame,cols[name])
        value=_convert(raw,units[name],name)
        missing[name]=int(np.isnan(value).sum())
        if missing[name] and policy=='reject':
            raise SchemaError(f'{name} 有{missing[name]}个缺失值；reject策略阻止输出')
        lo,hi=ranges[name]
        if ((value<lo)|(value>hi)).any():
            raise SchemaError(f'{name} 超出本次工程检查范围[{lo},{hi}]；保留原数据，请核实单位/异常值')
        if name in {'opaque_cover','precip_code'} and not np.equal(value[np.isfinite(value)],np.rint(value[np.isfinite(value)])).all():
            raise SchemaError(f'{name} 必须整数编码；不以连续量代替类别')
        if name in {'ceiling','opaque_cover','relative_humidity','precip_code'}:
            rounded=np.rint(value)
            rounding[name]=float(np.nanmax(np.abs(rounded-value))) if np.isfinite(value).any() else 0.
            value=rounded
        work[name]=np.where(np.isnan(value),9999,value)
    if work.duplicated(['station_id','time']).any():
        raise SchemaError('重复站点时间记录；不能任取最后一条')
    times=sorted(work.time.unique())
    expected=pd.date_range(times[0],times[-1],freq='h')
    if len(times)!=len(expected) or len(work)!=len(expected)*len(stations):
        raise SchemaError('存在缺小时或站点时间矩阵不完整；需显式缺失记录，不能填补缺时刻')
    matrix=work.set_index(['time','station_id'])
    start,end=expected[0],expected[-1]
    lines=[f'{"SURF.DAT":16s}{"2.0":16s}{"EcoData Engine scoped CALMET 5.8.5 surface exporter":64s}',
           '1','Partial preprocessing input; coordinates must be set in CALMET.INP','NONE    ',
           f'{start.year} {start.dayofyear} {start.hour} {end.year} {end.dayofyear} {end.hour} {zone} {len(stations)}',
           ' '.join(map(str,stations))]
    for t in expected:
        lines.append(f'{t.year} {t.dayofyear} {t.hour}')
        for station in stations:
            row=matrix.loc[(t,station)]
            tokens=[str(int(row[n])) if n in {'ceiling','opaque_cover','relative_humidity','precip_code'} else f'{row[n]:.6f}' for n in FIELDS]
            lines.append(' '.join(tokens))
    report={'scope':'CALMET 5.8.5 SURF.DAT 2.0 formatted NONE only',
            'complete_model_input':False,'native_model_input_verified':False,
            'hours':len(expected),'stations':stations,'records':len(work),
            'timezone_west_hours':zone,'time_label':'hour_end','missing_policy':policy,
            'missing_counts':missing,'rounding_max_in_target_units':rounding,
            'required_next_steps':['CALMET.INP station locations/grid and NSSTA/IFORMS=2',
                                   'GEO.DAT, upper-air/prognostic meteorology as configured',
                                   'CALMET run, then CALPUFF sources/receptors/control'],
            'range_basis':'provisional engineering bounds; missing codes from official RDS/SMERGE',
            'format_source':'CALMET 5.8.5 RDHD and RDS; SMERGE 5.57 sample surf.dat'}
    return {'SURF.DAT':('\n'.join(lines)+'\n').encode('ascii')},report
