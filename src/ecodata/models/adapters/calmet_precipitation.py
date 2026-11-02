"""CALMET5.8.5 PRECIP.DAT2.0 formatted NONE, hourly observed rain only.

Own writer based on official RDHD/RDP read structure. Does not generate a
weather code, interpolate rain, distribute daily totals, or create controls.
"""
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform


def export_precipitation(frame,config):
    if config.get('mode')!='precipitation' or config.get('version')!='CALMET-5.8.5-PRECIP-2.0':
        raise SchemaError('降水模式需mode=precipitation与version=CALMET-5.8.5-PRECIP-2.0')
    if config.get('time_label')!='hour_end':
        raise SchemaError('PRECIP.DAT记录为小时结束时刻，需time_label=hour_end；不猜测或移动时刻')
    zone=config.get('timezone_west_hours')
    if type(zone) is not int or not -12<=zone<=12:raise SchemaError('timezone_west_hours需-12..12整数，固定标准时区西正东负')
    stations=config.get('station_ids')
    if not isinstance(stations,list) or not 1<=len(stations)<=700 or len(set(stations))!=len(stations) or any(type(s) is not int or not 1<=s<=99999999 for s in stations):
        raise SchemaError('station_ids需1–700个不重复正整数站点ID，顺序必须与CALMET.INP一致')
    columns=config.get('fields')
    if not isinstance(columns,dict) or set(columns)!={'station','time','precipitation'} or any(not isinstance(v,str) or not v for v in columns.values()) or len(set(columns.values()))!=3:
        raise SchemaError('fields需独立映射station/time/precipitation三列')
    if frame.empty or set(columns.values())-set(frame.columns):raise SchemaError('降水表为空或缺必需列：'+','.join(sorted(set(columns.values())-set(frame.columns))))
    markers=config.get('missing_values')
    if not isinstance(markers,list):raise SchemaError('需明确missing_values列表；不补缺站、缺时或降水')
    data=frame[list(columns.values())].replace(markers,np.nan)
    if data.isna().any().any() or data.map(lambda v:isinstance(v,(bool,np.bool_)) or (isinstance(v,str) and not v.strip())).any().any():
        raise SchemaError('降水必需字段缺测、空白或布尔值，不能填零')
    quantity=config.get('quantity');unit=config.get('source_units')
    if not isinstance(unit,str) or not unit.strip():raise SchemaError('降水source_units不明')
    if quantity=='one_hour_amount':
        if type(config.get('accumulation_seconds')) is not int or config['accumulation_seconds']!=3600:
            raise SchemaError('one_hour_amount必须是原始前一小时累计量accumulation_seconds=3600；日雨量不能拆成小时')
        scale,offset=unit_transform(unit,'mm')
        method='Original preceding-one-hour accumulated depth converted to its one-hour mean mm/hour; ending time unchanged.'
    elif quantity=='hour_mean_rate':
        scale,offset=unit_transform(unit,'mm/hour')
        method='Original hourly mean precipitation rate converted to mm/hour; ending time unchanged.'
    else:raise SchemaError('quantity需one_hour_amount或hour_mean_rate；不接受未说明的瞬时雨量或日累计')
    if offset!=0 or scale<=0:raise SchemaError('降水单位需正比例深度/时间，不得偏移')
    try:
        ids=pd.to_numeric(data[columns['station']],errors='raise').to_numpy(dtype=float)
        values=pd.to_numeric(data[columns['precipitation']],errors='raise').to_numpy(dtype=float)*scale
    except (ValueError,TypeError) as exc:raise SchemaError('站点或降水数值无效') from exc
    if not np.isfinite(ids).all() or (ids!=np.rint(ids)).any() or set(ids)!=set(stations):raise SchemaError('源站点与station_ids不符，不能丢弃未知站点')
    if not np.isfinite(values).all() or (values<0).any() or (values>1000).any():raise SchemaError('降水存在NaN/负值/缺测或超过暂定0–1000mm/hour物理检查范围')
    if ((values!=0)&(values.astype('f4')==0)).any():raise SchemaError('降水单精度下溢，不能静默变0')
    times=[]
    for raw in data[columns['time']]:
        try:
            t=datetime.fromisoformat(str(raw).replace('Z','+00:00'))
            if t.utcoffset() is None:raise ValueError('缺UTC偏移')
            t=t.astimezone(timezone(timedelta(hours=-zone))).replace(tzinfo=None)
            if t.minute or t.second or t.microsecond or not 1900<=t.year<=2199:raise ValueError('需1900–2199年整点结束时刻')
        except ValueError as exc:raise SchemaError(f'降水时间{raw!r}无效：{exc}') from exc
        times.append(t)
    work=pd.DataFrame({'time':times,'station':ids.astype('int64'),'value':values})
    if work.duplicated(['time','station']).any():raise SchemaError('重复时间×站点降水记录')
    hours=sorted(set(times));expected=pd.date_range(hours[0],hours[-1],freq='h')
    keys=pd.MultiIndex.from_frame(work[['time','station']]);allkeys=pd.MultiIndex.from_product([expected,stations])
    if len(keys)!=len(allkeys) or len(allkeys.difference(keys)):raise SchemaError('缺小时或缺站点记录；显式零降水也必须提供，不能自动补0')
    table=work.set_index(['time','station']);start,end=expected[0],expected[-1]
    lines=[f'{"PRECIP.DAT":16s}{"2.0":16s}{"EcoDataEngine CALMET5.8.5 hourly observed precipitation":64s}',
           '1','Observed hourly rates mm/hr; NONE locations require CALMET.INP','NONE    ',
           f'{start.year} {start.dayofyear} {start.hour} {end.year} {end.dayofyear} {end.hour} {zone} {len(stations)}',
           ' '.join(map(str,stations))]
    for t in expected:lines.append(f'{t.year} {t.dayofyear} {t.hour} '+' '.join(format(table.loc[(t,s),'value'],'.17g') for s in stations))
    scope=('CALMET5.8.5 PRECIP.DAT2.0 formatted NONE only; rate mm/hour at hour END. '
           'Set IFORMP=2, NPSTA, station ID order, IBTZ and station locations in existing CALMET.INP. '
           'No control file, station-location datum, rain interpolation or daily-total distribution generated. '
           'This precipitation file is not CALMET.DAT or a complete CALPUFF project.')
    return {'PRECIP.DAT':('\n'.join(lines)+'\n').encode('ascii'),'README_CALMET_precipitation.txt':scope.encode('ascii')}, {
        'scope':scope,'version':config['version'],'hours':len(expected),'station_ids':stations,'records':len(work),
        'start_local_standard':start.isoformat(),'end_local_standard':end.isoformat(),'timezone_west_hours':zone,
        'time_label':'hour_end','time_processing':'none','unit_processing':method,'scale':scale,'units':'mm/hour',
        'missing_policy':'reject','source':config.get('source'),
        'unmapped_columns':sorted(set(frame.columns)-set(columns.values())),
        'format_source':'Official CALMET5.8.5 RDHD formatted NONE and RDP; PARAMS.MET MXPS=700',
        'engineering_bounds_source':'Predefined provisional 0–1000 mm/hour safety checks, not model calibration thresholds',
        'native_program_read_verified':False,'complete_model_input':False}
