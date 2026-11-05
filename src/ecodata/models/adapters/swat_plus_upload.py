"""Single-station daily weather upload; explicit SWAT+ 62.0.0 additions."""
import hashlib
import json
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import read_structure
from ecodata.models.adapters.swat_plus import SWATPlusWriter, ADDITIONAL_WEATHER

UNIT_FACTORS = {
    'solar_radiation': {'MJ/m2/day': 1., 'MJ/m^2/day': 1., 'kWh/m2/day': 3.6},
    'relative_humidity': {'1': 1., '%': .01},
    'wind_speed': {'m/s': 1., 'km/h': 1./3.6},
}


def _validate_weather_references(files):
    """Check delivered single-station indexes; sim/null are SWAT+ tokens."""
    variable_indexes = {'pcp.cli', 'tmp.cli', 'slr.cli', 'hmd.cli', 'wnd.cli'}
    checked = {}
    for name in sorted(n for n in files if n.endswith('.cli')):
        rows = files[name].decode('utf-8').splitlines()
        if len(rows) != 3:
            raise SchemaError(f'天气索引{name}需表头及一个站点记录')
        if name in variable_indexes:
            refs = rows[2].split()
            if rows[1].split() != ['FILENAME'] or len(refs) != 1:
                raise SchemaError(f'天气索引{name}格式错误')
        elif name == 'weather-sta.cli':
            record = rows[2].split()
            if len(record) != 9 or record[1] != 'null':
                raise SchemaError('天气包不能引用未提供的WGN；weather-sta.cli需9字段')
            refs = record[1:]
        else:
            raise SchemaError(f'天气包包含未支持的索引：{name}')
        for ref in refs:
            if ref in {'null', 'sim'} and name == 'weather-sta.cli':
                continue
            if ref not in files:
                raise SchemaError(f'天气索引{name}引用了包内不存在的文件：{ref}')
        checked[name] = refs
    if not {'pcp.cli', 'tmp.cli', 'weather-sta.cli'} <= checked.keys():
        raise SchemaError('天气包缺少降水、温度或站点索引')
    return checked


def _weather_download_bundle(files):
    files = dict(files)
    files.pop('file.cio', None)
    checked = _validate_weather_references(files)
    files['README_weather_only.txt'] = (
        'SWAT+ 单站逐日天气资料包，不是可运行的流域工程。\n'
        '包含所选 station.* 天气文件、实际存在的 .cli 索引及报告；不交付 file.cio。\n'
        'time.sim 仅保留日期控制片段，须由目标工程配置采用。\n'
        '未生成 WGN、HRU、土壤、河道、含水层、管理措施或流域文件。\n'
        'weather-sta.cli 中 null 表示无引用；sim 是天气生成器选项，不是文件。\n'
        '未导出的变量如使用 sim，须另提供适用 WGN 并在目标工程中关联，不能凭本包运行。\n'
        '索引存在性校验不代表官方 SWAT+ 已读取或完整模型已运行。\n'
    ).encode('utf-8')
    validation = json.loads(files['validation_report.json'])
    validation.update(scope='swat_plus_weather_only', complete_model_input=False,
                      index_references_verified=True, index_references=checked,
                      generated_files=sorted(n for n in files if n not in {'manifest.json', 'validation_report.json'}),
                      warning='仅天气资料包；不交付file.cio；未生成WGN或流域工程，未用官方模型读取')
    manifest = json.loads(files['manifest.json'])
    manifest.update(reference_case=None, validation=validation,
                    files={n: {'size': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                           for n, b in files.items() if n not in {'manifest.json', 'validation_report.json'}})
    files['validation_report.json'] = json.dumps(validation, ensure_ascii=False, indent=2).encode('utf-8')
    files['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    return files


def export_uploaded_weather(content, config):
    required = {'project_name', 'start', 'end', 'station'}
    if not required <= set(config) or set(config) - required - {'additional_weather', 'target_model_version'}:
        raise SchemaError('SWAT+配置需project_name、start、end、station；可选additional_weather及target_model_version；weather从上传NC读取')
    additional = config.get('additional_weather', [])
    if not isinstance(additional, list) or any(not isinstance(n, str) or n not in ADDITIONAL_WEATHER for n in additional) or len(set(additional)) != len(additional):
        raise SchemaError('additional_weather只能选择solar_radiation、relative_humidity、wind_speed，不能重复')
    if additional and config.get('target_model_version') != '62.0.0':
        raise SchemaError('新增天气需target_model_version=62.0.0；其他版本尚未验证')
    if not isinstance(config['station'], dict) or set(config['station']) != {'name', 'latitude', 'longitude', 'elevation'}:
        raise SchemaError('站点配置需name、latitude、longitude、elevation')
    station = config['station']
    if not isinstance(config['project_name'], str) or not config['project_name'].strip() or not isinstance(station['name'], str) or not station['name'].strip():
        raise SchemaError('工程名和站点名不能为空')
    if any(c.isspace() for c in station['name']):
        raise SchemaError('站点名不能含空格或换行')
    try:
        coords = np.array([station[n] for n in ('latitude', 'longitude', 'elevation')], float)
    except (ValueError, TypeError) as exc:
        raise SchemaError('站点坐标和海拔必须为数值') from exc
    if not np.isfinite(coords).all() or not -90 <= coords[0] <= 90 or not -180 <= coords[1] <= 180:
        raise SchemaError('站点经纬度/海拔无效')
    try:
        ds = read_structure('weather.nc', content)
    except Exception as exc:
        raise SchemaError(f'无法读取天气NetCDF：{exc}') from exc
    fields = {'precipitation': 'mm', 'air_temperature_max': 'degC', 'air_temperature_min': 'degC'}
    if 'time' not in ds.coords or ds.time.dims != ('time',):
        raise SchemaError('天气NC需一维time日日期')
    values, conversions = {}, {}
    for name in list(fields) + additional:
        if name not in ds:
            raise SchemaError(f'天气NC缺少字段：{name}')
        if ds[name].dims != ('time',):
            raise SchemaError(f'{name}必须为单站一维日数据，不能静默选择网格或层')
        unit = ds[name].attrs.get('units')
        if name in fields:
            if unit != fields[name] and not (name=='precipitation' and unit=='mm/day'):
                raise SchemaError(f'{name}单位需{fields[name]}，不能只改单位标签')
            factor = 1.
        else:
            if unit not in UNIT_FACTORS[name]:
                hint = '；W/m2为功率，缺少积分时段不能当日总能量' if name == 'solar_radiation' else ''
                raise SchemaError(f'{name}单位需{list(UNIT_FACTORS[name])}{hint}')
            factor = UNIT_FACTORS[name][unit]
            fields[name] = unit
        try:
            v = np.asarray(ds[name].values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f'{name}需数值') from exc
        if not np.isfinite(v).all():
            raise SchemaError(f'{name}包含缺失/无效值，本界面不自动填补')
        values[name] = v * factor
        conversions[name] = {'input_unit': unit, 'output_unit': ADDITIONAL_WEATHER[name][1] if name in ADDITIONAL_WEATHER else fields[name], 'multiply_by': factor}
        if name=='precipitation' and unit=='mm/day':
            from ecodata.engines.units import UnitEngine
            # This writer's explicit contract is one complete daily record;
            # its existing calendar validation still rejects gaps/duplicates.
            amount=UnitEngine().integrate_rate(ds[name],np.full(len(v),86400.),'mm')
            values[name]=amount.values
            conversions[name]['interval_integration']=json.loads(amount.attrs['interval_integration'])
    if (values['precipitation'] < 0).any():
        raise SchemaError('降水不能为负值')
    if (values['air_temperature_max'] < values['air_temperature_min']).any():
        raise SchemaError('最高温不能低于最低温')
    for name, value in zip(('latitude', 'longitude', 'elevation'), coords):
        if name in ds.attrs:
            try:
                valid = np.isclose(float(ds.attrs[name]), value, rtol=0, atol=1e-8)
            except (ValueError, TypeError) as exc:
                raise SchemaError(f'源NC的{name}无效') from exc
            if not valid:
                raise SchemaError(f'配置站点{name}与源NC不一致')
    frame = pd.DataFrame(values)
    frame.insert(0, 'time', pd.DatetimeIndex(ds.time.values))
    writer_config = {**config, 'weather': frame.to_dict(orient='records')}
    if additional:
        writer_config['station'] = {**station, **dict(zip(('latitude', 'longitude', 'elevation'), map(float, coords)))}
    files = _weather_download_bundle(SWATPlusWriter().write(writer_config))
    report = {'scope': 'SWATPlusWriter single-station daily weather via upload',
              'writer': 'SWATPlusWriter', 'records': len(frame), 'start': config['start'], 'end': config['end'],
              'station': station, 'input_units': fields,
              'output_units': {n: c['output_unit'] for n, c in conversions.items()},
              'unit_conversions': conversions, 'additional_weather': additional,
              'weather_format_target': 'SWAT+ 62.0.0 / Editor 4.0.2' if additional else 'legacy precipitation/temperature writer',
              'time_standard_from_source': ds.attrs.get('time_standard', 'not specified; daily date labels retained'),
              'wind_height_m_from_source': float(ds.wind_speed.attrs['measurement_height_m']) if 'wind_speed' in additional and 'measurement_height_m' in ds.wind_speed.attrs else None,
              'extra_source_fields_not_exported': [n for n in ds.data_vars if n not in fields],
              'missing_policy': 'reject; no filling or invented dates/values',
              'complete_model_input': False, 'model_run_verified': False,
              'index_references_verified': True,
              'warning': '仅天气文件及实际索引；不交付file.cio；未生成WGN、HRU、河道、含水层或完整流域工程，未用官方模型读取'}
    return files, report
