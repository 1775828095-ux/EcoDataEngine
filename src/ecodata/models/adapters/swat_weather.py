"""Traditional SWAT 2012 single-gauge daily .pcp/.tmp. Not SWAT+."""
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.calmet_surface import _numeric


def _fixed(value,width,precision):
    text=f'{value:{width}.{precision}f}'
    if len(text)!=width:
        raise SchemaError('数值不能装入SWAT固定宽度字段；禁止截断或输出星号')
    return text


def export_weather(frame,config):
    if config.get('version')!='SWAT-2012-daily-single-gauge':
        raise SchemaError('仅支持传统SWAT 2012单站日天气格式；不是SWAT+')
    if config.get('date_semantics')!='reported_daily_date':
        raise SchemaError('必须明确 date_semantics=reported_daily_date；确认源日统计窗口，不重排日期')
    if config.get('source_frequency') == 'hourly':
        return _export_hourly(frame, config)
    if config.get('source_frequency', 'daily') != 'daily':
        raise SchemaError('source_frequency 仅允许 daily 或 hourly')
    policy=config.get('missing_policy','reject')
    if policy not in {'reject','swat_generator'}:
        raise SchemaError('missing_policy 仅允许 reject/swat_generator')
    cols=config.get('columns',{});units=config.get('units',{})
    names=('precipitation','tmax','tmin')
    if any(n not in cols or n not in units for n in names) or len(set(cols.values()))!=len(cols):
        raise SchemaError('需明确配置不同源列与三项单位')
    date_col=config.get('date_column','date')
    if date_col not in frame or not len(frame): raise SchemaError('缺少日期或记录为空')
    try:
        dates=pd.to_datetime(frame[date_col],format='%Y-%m-%d',errors='raise')
    except ValueError as exc: raise SchemaError('SWAT日日期必须为YYYY-MM-DD') from exc
    if dates.isna().any() or dates.duplicated().any() or (dates.dt.hour!=0).any():
        raise SchemaError('日期缺失、重复或不是日标签')
    if not dates.is_monotonic_increasing or not np.equal(dates.diff().dropna(),pd.Timedelta(days=1)).all():
        raise SchemaError('日记录必须按日期连续递增；不静默补日或重排')
    station=config.get('station',{})
    lat,lon,elev=(station.get(n) for n in ('latitude','longitude','elevation_m'))
    if station.get('crs')!='EPSG:4326' or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in (lat,lon,elev)) or not -90<=lat<=90 or not -180<=lon<=180:
        raise SchemaError('需可靠EPSG:4326站点经纬度与海拔；经纬度不得交换')
    values={};missing={};rounding={}
    for name in names:
        raw=_numeric(frame,cols[name]);unit=units[name]
        factors=({'mm':(1,0),'inch':(25.4,0)} if name=='precipitation' else {'degC':(1,0),'degF':(5/9,-32*5/9),'K':(1,-273.15)})
        if unit not in factors:raise SchemaError(f'{name} 单位{unit!r}不支持')
        a,b=factors[unit];value=raw*a+b
        missing[name]=int(np.isnan(value).sum())
        if policy=='reject' and missing[name]:raise SchemaError(f'{name}缺失；拒绝触发SWAT天气生成器')
        if name=='precipitation' and (value<0).any():raise SchemaError('负降水不合法；缺失应留空，不能把-99当实测值')
        if name!='precipitation' and ((value<-90)|(value>65)).any():raise SchemaError('温度超出本次工程范围[-90,65] degC')
        values[name]=value
        rounding[name]=float(np.nanmax(np.abs(np.round(value,1)-value))) if np.isfinite(value).any() else 0.
    if (values['tmax']<values['tmin']).any():raise SchemaError('最高温低于最低温；不交换原值')
    # Chapter 6 uses 5-character station fields; Chapter 7 uses 10 characters.
    def header(width):
        elevation=int(np.rint(elev));etext=f'{elevation:{width}d}'
        if len(etext)!=width:raise SchemaError('海拔不能装入固定宽度')
        return ['EcoData SWAT 2012 single gauge daily weather',
                'Lati   '+_fixed(lat,width,1 if width==5 else 4),
                'Long   '+_fixed(lon,width,1 if width==5 else 4),
                'Elev   '+etext]
    pcp,tmp=header(5),header(10)
    for i,date in enumerate(dates):
        prefix=f'{date.year:04d}{date.dayofyear:03d}'
        output={n:_fixed(-99. if np.isnan(values[n][i]) else values[n][i],5,1) for n in names}
        pcp.append(prefix+output['precipitation']);tmp.append(prefix+output['tmax']+output['tmin'])
    report={'scope':'traditional SWAT 2012 single gauge daily weather only','records':len(frame),
            'missing_counts':missing,'missing_policy':policy,'date_semantics':config['date_semantics'],
            'station_source':station,'station_header_rounding':{'pcp_degrees':0.05,'tmp_degrees':0.00005,'elevation_m':0.5},
            'rounding_max':rounding,'native_model_input_verified':False,'complete_model_input':False,
            'warning':'-99.0 explicitly invokes SWAT weather generation; NOAA GSOD reporting windows require project review',
            'format_source':'SWAT Input/Output Documentation 2012 Chapters 6 and 7',
            'required_next_steps':['file.cio NRGAGE/NRGFIL NTGAGE/NTGFIL and file names', 'watershed soil/landuse/HRU/project inputs and SWAT executable']}
    return {'weather.pcp':('\n'.join(pcp)+'\n').encode('ascii'),'weather.tmp':('\n'.join(tmp)+'\n').encode('ascii')},report


def _export_hourly(frame, config):
    """Explicit hourly-to-daily aggregation for traditional SWAT weather."""
    aggregation = config.get('aggregation')
    required = {'precipitation': 'sum', 'tmax': 'max', 'tmin': 'min'}
    if not isinstance(aggregation, dict) or aggregation.get('precipitation') != 'sum' or aggregation.get('tmax') != 'max' or aggregation.get('tmin') != 'min':
        raise SchemaError('小时→日聚合必须显式指定 precipitation=sum、tmax=max、tmin=min；不允许静默聚合')
    zone = config.get('source_time_zone')
    if not isinstance(zone, str):
        raise SchemaError('小时数据必须提供 source_time_zone；不猜测本地时区')
    time_col = config.get('time_column', 'time')
    if time_col not in frame or not len(frame):
        raise SchemaError('缺少时间字段或记录为空')
    try:
        times = pd.to_datetime(frame[time_col], errors='raise')
    except (ValueError, TypeError) as exc:
        raise SchemaError('小时时间需为可解析日期时间') from exc
    if times.isna().any() or times.duplicated().any():
        raise SchemaError('时间缺失或重复')
    if times.dt.tz is None:
        try:
            times = times.dt.tz_localize(zone)
        except (TypeError, ValueError) as exc:
            raise SchemaError('无法按 source_time_zone 本地化时间') from exc
    else:
        try:
            times = times.dt.tz_convert(zone)
        except (TypeError, ValueError) as exc:
            raise SchemaError('无法按 source_time_zone 转换时间') from exc
    if (times.dt.minute != 0).any() or (times.dt.second != 0).any() or (times.dt.microsecond != 0).any():
        raise SchemaError('小时聚合只接受整点小时标签')
    work = frame.copy()
    work[time_col] = times.dt.tz_localize(None)
    work = work.sort_values(time_col, kind='stable')
    hourly_index = pd.date_range(work[time_col].min(), work[time_col].max(), freq='h')
    if len(work) != len(hourly_index) or not work[time_col].isin(hourly_index).all() or not hourly_index.isin(work[time_col]).all():
        raise SchemaError('存在缺小时；reject 策略不自动补日')

    cols = config.get('columns', {})
    units = config.get('units', {})
    names = ('precipitation', 'tmax', 'tmin')
    if any(n not in cols or n not in units for n in names) or len(set(cols.values())) != len(cols):
        raise SchemaError('需明确配置不同源列与三项单位')
    converted = {}
    for name in names:
        raw = _numeric(work, cols[name])
        factors = ({'mm': (1, 0), 'inch': (25.4, 0)} if name == 'precipitation' else {'degC': (1, 0), 'degF': (5/9, -32*5/9), 'K': (1, -273.15)})
        unit = units[name]
        if unit not in factors:
            raise SchemaError(f'{name} 单位{unit!r}不支持')
        scale, offset = factors[unit]
        value = raw * scale + offset
        if np.isnan(value).any():
            raise SchemaError(f'{name} 有缺小时；小时→日聚合要求完整日数据')
        if name == 'precipitation' and (value < 0).any():
            raise SchemaError('负降水不合法')
        if name != 'precipitation' and ((value < -90) | (value > 65)).any():
            raise SchemaError('温度超出本次工程范围[-90,65] degC')
        converted[name] = value
    work = work.assign(**{name: converted[name] for name in names})
    work['date'] = work[time_col].dt.normalize()
    daily = work.groupby('date').agg(
        precipitation=(cols['precipitation'], 'sum'),
        tmax=(cols['tmax'], 'max'),
        tmin=(cols['tmin'], 'min'),
    ).reset_index()
    daily = daily.rename(columns={
        cols['precipitation']: 'precipitation',
        cols['tmax']: 'tmax',
        cols['tmin']: 'tmin',
    })
    if (daily.tmax < daily.tmin).any():
        raise SchemaError('最高温低于最低温；不交换原值')
    daily['date'] = daily.date.dt.strftime('%Y-%m-%d')
    files, report = export_weather(daily, {
        **config,
        'source_frequency': 'daily',
        'columns': {'precipitation': 'precipitation', 'tmax': 'tmax', 'tmin': 'tmin'},
    })
    report.update({
        'source_frequency': 'hourly',
        'source_time_zone': zone,
        'aggregation': {'precipitation': 'sum', 'tmax': 'max', 'tmin': 'min'},
        'aggregation_status': 'AGGREGATED',
    })
    return files, report
