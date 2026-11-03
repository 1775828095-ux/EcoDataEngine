"""SWAT+ 62.0.0 cli_wgnread input, not a climate-statistics estimator."""
import calendar
import csv
import io
import json
import re
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError

FIELDS = ('tmpmx', 'tmpmn', 'tmpstdmx', 'tmpstdmn', 'pcpmm', 'pcpstd',
          'pcpskw', 'pr_wd', 'pr_ww', 'pcpd', 'rainhmx', 'solarav', 'dewpt', 'windav')
UNITS = dict(zip(FIELDS, ('degC', 'degC', 'degC', 'degC', 'mm', 'mm/day',
                        '1', '1', '1', 'day', 'mm', 'MJ/m2/day', 'degC', 'm/s')))
STATION_UNITS = {'latitude': 'degree', 'longitude': 'degree', 'elevation': 'm', 'rain_years': 'year'}


def _number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise SchemaError(f'{label}必须为有限数值，不能为布尔值')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f'{label}必须为数值') from exc
    if not np.isfinite(result):
        raise SchemaError(f'{label}包含缺失/NaN/无穷值')
    return result


def _identifier(value, label):
    # Fortran list-directed text: reject separators, comments and repetition syntax.
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,49}', value) or value in {'null', 'sim'}:
        raise SchemaError(f'{label}需1–50字符ASCII站点名（字母数字及_.-），不能为null/sim')
    return value


def _unit_transform(name, unit):
    if unit == UNITS[name]:
        return 1., 0.
    if name in {'tmpmx', 'tmpmn', 'dewpt'}:
        if unit == 'K': return 1., -273.15
        if unit == 'degF': return 5./9., -32.*5./9.
    if name in {'tmpstdmx', 'tmpstdmn'}:
        if unit == 'K': return 1., 0.
        if unit == 'degF': return 5./9., 0.  # A standard deviation has no offset.
    if name in {'pr_wd', 'pr_ww'} and unit == '%': return .01, 0.
    if name in {'pcpmm', 'rainhmx'} and unit == 'm': return 1000., 0.
    if name == 'pcpstd' and unit == 'm/day': return 1000., 0.
    if name == 'solarav' and unit == 'kWh/m2/day': return 3.6, 0.
    if name == 'windav' and unit == 'km/h': return 1./3.6, 0.
    raise SchemaError(f'{name}单位{unit!r}不明或不支持；目标单位为{UNITS[name]}，不能只改标签')


def check_station_references(content, station_names):
    """WGN slot is a generator NAME, not the name of a file."""
    if content is None:
        return {'status': 'not_supplied', 'warning': '未提供weather-sta.cli；尚未核对目标工程站点引用'}
    try:
        lines = content.decode('utf-8-sig').splitlines()
    except UnicodeError as exc:
        raise SchemaError('weather-sta.cli需UTF-8/ASCII') from exc
    if len(lines) < 3 or any(not line.strip() for line in lines):
        raise SchemaError('weather-sta.cli需两行表头及完整站点行；不能有空行')
    rows, seen = [], set()
    for line in lines[2:]:
        row = line.split()
        if len(row) != 9:
            raise SchemaError('weather-sta.cli站点记录必须9字段')
        weather_name = _identifier(row[0], '天气站名称')
        if weather_name in seen:
            raise SchemaError(f'weather-sta.cli重复天气站：{weather_name}')
        seen.add(weather_name)
        if row[1] not in station_names:
            raise SchemaError(f'天气站{weather_name}的WGN引用{row[1]!r}未对应本次统计站点；需在目标工程明确关联，不自动猜测')
        rows.append({'weather_station': weather_name, 'wgn_station': row[1]})
    return {'status': 'verified_wgn_names_only', 'links': rows,
            'warning': '只核对WGN站点名称；未校验该工程其它天气文件、空间代表性或完整工程'}


def export_wgn(content, config, weather_station_index=None):
    required = {'mode', 'target_model_version', 'stations', 'fields', 'units', 'station_units', 'source', 'missing_values'}
    if not required <= config.keys() or config.keys() - required - {'encoding', 'title'}:
        raise SchemaError('WGN配置需mode、target_model_version、stations、fields、units、station_units、source、missing_values；可选encoding/title')
    if not isinstance(config['missing_values'], list):
        raise SchemaError('missing_values需来源定义的缺测标记列表；无数字标记请明确填[]')
    missing_markers = [_number(v, 'missing_values') for v in config['missing_values']]
    if config['mode'] != 'wgn' or config['target_model_version'] != '62.0.0':
        raise SchemaError('WGN仅支持mode=wgn、target_model_version=62.0.0')
    source = config['source']
    if not isinstance(source, dict) or not {'kind', 'url', 'institution', 'license', 'description'} <= source.keys():
        raise SchemaError('source需kind、url、institution、license、description；不能把单年日天气冒充多年统计')
    if not isinstance(source['kind'], str) or source['kind'] not in {'climate_statistics', 'format_reference', 'synthetic_test'} or any(not isinstance(source[n], str) or not source[n].strip() for n in ('url','institution','license','description')):
        raise SchemaError('统计来源信息不完整；kind需climate_statistics/format_reference/synthetic_test')
    if source['kind'] == 'climate_statistics':
        start, end = source.get('period_start_year'), source.get('period_end_year')
        if type(start) is not int or type(end) is not int or not 1 <= start < end <= 9999:
            raise SchemaError('真实气候统计需明确至少跨两年的period_start_year/period_end_year；不接受2021单年冒充多年统计')
    if config['station_units'] != STATION_UNITS:
        raise SchemaError(f'站点单位必须明确为{STATION_UNITS}')
    station_list = config['stations']
    if not isinstance(station_list, list) or not station_list:
        raise SchemaError('stations需非空站点列表')
    stations = {}
    for station in station_list:
        if not isinstance(station, dict) or set(station) != {'id', *STATION_UNITS}:
            raise SchemaError('每个WGN站点需id、latitude、longitude、elevation、rain_years')
        name = _identifier(station['id'], 'WGN站点id')
        if name in stations: raise SchemaError(f'重复WGN站点：{name}')
        vals = {n: _number(station[n], f'{name}.{n}') for n in STATION_UNITS}
        if not -90 <= vals['latitude'] <= 90 or not -180 <= vals['longitude'] <= 180:
            raise SchemaError(f'{name}经纬度范围错误（纬度±90、经度±180）')
        if vals['rain_years'] < 1:
            raise SchemaError(f'{name}.rain_years需≥1；不允许触发模型默认为10年')
        if source['kind'] == 'climate_statistics' and not 2 <= vals['rain_years'] <= source['period_end_year']-source['period_start_year']+1:
            raise SchemaError(f'{name}.rain_years需为声明时段内至少2年的半小时最大降雨统计记录年数')
        stations[name] = vals
    mapping = config['fields']
    keys = {'station', 'month', *FIELDS}
    if not isinstance(mapping, dict) or set(mapping) != keys or any(not isinstance(v,str) or not v for v in mapping.values()) or len(set(mapping.values())) != len(keys):
        raise SchemaError('fields需station、month及全部14逐月字段的唯一列映射')
    units = config['units']
    if not isinstance(units, dict) or set(units) != set(FIELDS):
        raise SchemaError('units需明确全部14逐月字段单位')
    transforms = {n: _unit_transform(n, units[n]) for n in FIELDS}
    encoding = config.get('encoding', 'utf-8-sig')
    if encoding not in {'utf-8', 'utf-8-sig', 'gb18030'}: raise SchemaError('CSV编码需UTF-8/UTF-8 BOM/GB18030')
    try:
        decoded = content.decode(encoding)
        header = next(csv.reader(io.StringIO(decoded)))
        if len(set(header)) != len(header): raise SchemaError('统计CSV表头有重复字段')
        missing = set(mapping.values()) - set(header)
        if missing: raise SchemaError(f'统计CSV缺少必需列：{sorted(missing)}')
        if set(header) != set(mapping.values()): raise SchemaError('统计CSV有未映射字段，拒绝静默丢列')
        data = pd.read_csv(io.StringIO(decoded), dtype=str, keep_default_na=False, skip_blank_lines=False)
    except (ValueError, UnicodeError, StopIteration) as exc:
        raise SchemaError(f'统计CSV无法读取：{exc}') from exc
    data = data.rename(columns={v:k for k,v in mapping.items()})
    if set(data.station) != set(stations):
        raise SchemaError(f'CSV统计站点与配置不对应：CSV={sorted(set(data.station))}，配置={sorted(stations)}')
    months = [_number(v, f'行{i+2}.month') for i,v in enumerate(data.month)]
    if any(v != int(v) or not 1 <= v <= 12 for v in months): raise SchemaError('month必须为1–12整数')
    data['month'] = list(map(int,months))
    title = config.get('title', 'weather-wgn.cli: SWAT+ 62.0.0 climate statistics')
    if not isinstance(title,str) or not title.isascii() or not title.strip() or len(title)>80 or '\n' in title or '\r' in title:
        raise SchemaError('title需1–80字符单行ASCII')
    lines = [title]; warnings = []
    for name, coords in stations.items():
        frame = data[data.station==name].sort_values('month')
        if frame.month.duplicated().any(): raise SchemaError(f'{name}有重复月份')
        missing = sorted(set(range(1,13))-set(frame.month))
        if missing: raise SchemaError(f'{name}缺月份：{missing}；不能补零')
        lines.append(' '.join([name]+[repr(coords[n]) for n in STATION_UNITS]))
        lines.append(' '.join(n.upper() for n in FIELDS))
        for row in frame.to_dict('records'):
            mo = row['month']; label = f'{name}月{mo}'
            original = {n:_number(row[n], f'{label}.{n}') for n in FIELDS}
            for n,v in original.items():
                if v in missing_markers: raise SchemaError(f'{label}.{n}为来源缺测标记{v}；不能填零')
            values = {n:original[n]*transforms[n][0]+transforms[n][1] for n in FIELDS}
            if not all(np.isfinite(v) for v in values.values()): raise SchemaError(f'{label}单位转换溢出')
            if values['tmpmx'] < values['tmpmn']: raise SchemaError(f'{label}最高温低于最低温')
            for n in ('tmpstdmx','tmpstdmn','pcpmm','pcpstd','pcpd','rainhmx','solarav','windav'):
                if values[n] < 0: raise SchemaError(f'{label}.{n}不能为负值/缺测标记')
            for n in ('pr_wd','pr_ww'):
                if not 0 <= values[n] <= 1: raise SchemaError(f'{label}.{n}概率必须为0–1')
            max_days = calendar.monthrange(2000, mo)[1]  # Allow February climatologies spanning leap years.
            if values['pcpd'] > max_days: raise SchemaError(f'{label}.pcpd超过该月最多{max_days}天')
            if values['pcpskw'] < .2 or values['pr_ww'] <= values['pr_wd'] or values['pr_wd'] <= 0:
                warnings.append(f'{label}：SWAT+初始化可能调整偏度/湿日概率/湿日数；本转换保留原值')
            lines.append(' '.join(repr(values[n]) for n in FIELDS))
    reference = check_station_references(weather_station_index, stations)
    report = {'scope':'SWAT+ 62.0.0 weather-wgn.cli statistics only',
              'stations':stations, 'months_per_station':12, 'monthly_fields':list(FIELDS),
              'units':UNITS, 'station_units':STATION_UNITS,
              'unit_conversions':{n:{'input':units[n],'output':UNITS[n],'multiply_by':transforms[n][0],'add':transforms[n][1]} for n in FIELDS},
              'source':source, 'provenance_verification':'source metadata supplied by user, not independently certified by converter',
              'station_reference_check':reference, 'warnings':warnings,
              'random_weather_generation':False, 'climate_statistics_calculated':False,
              'complete_model_input':False, 'native_model_input_verified':False,
              'model_run_verified':False, 'missing_policy':'reject; no filling, defaults or estimation'}
    readme = ('SWAT+ 62.0.0 weather-wgn.cli: station headers and 12 monthly rows, NOT an empty index.\n'
              'Statistics only; no random weather generation, no climate statistics calculation.\n'
              'No file.cio is provided. In your complete target project, weather-sta.cli WGN slot must match a generator name here.\n'
              'Optional uploaded weather-sta.cli is checked for WGN names only, not modified or repackaged.\n'
              'Without that file, station association is unverified. Check the report before using it.\n'
              'SWAT initialization may change skewness, wet-day probabilities and wet-day counts; exporter preserves supplied values.\n'
              'Not an independently runnable basin project; no official SWAT run verified.\n')
    return {'weather-wgn.cli': ('\n'.join(lines)+'\n').encode('ascii'),
            'README_WGN.txt':readme.encode('ascii')}, report
