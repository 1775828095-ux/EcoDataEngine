"""Mapped sounding tables -> CALMET 5.8.5 UP.DAT 2.0, comma data.

An observational CALMET input, not CALMET.DAT or a complete CALPUFF case.
Structure is based on RDHDU/RDUP of the official 5.8.5 source. No official
software code is embedded. Heights are reported MSL heights; temperatures K.
"""
from datetime import datetime, timezone
import re

import numpy as np
import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform

FIELDS = ('pressure', 'height', 'temperature', 'wind_direction', 'wind_speed')
UNITS = dict(zip(FIELDS, ('hPa', 'm', 'K', 'degree', 'm/s')))


def _real(value, label, bounds):
    if isinstance(value, bool):
        raise SchemaError(f'{label} 必须为有限数值')
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f'{label} 必须为有限数值') from exc
    if not np.isfinite(result) or not bounds[0] <= result <= bounds[1]:
        raise SchemaError(f'{label} 必须在 {bounds} 范围内')
    return result


def _integer(value, label, bounds):
    result = _real(value, label, bounds)
    if result != int(result):
        raise SchemaError(f'{label} 必须为整数')
    return int(result)


def export_upper_air(frame, config):
    """Return (files, report); one UPn.DAT per configured station.

    Input is a long table, one observed level per row. Column order and units
    are configured; levels are ordered by explicit level_column, not inferred
    by silently dropping/reordering bad heights. No missing values are filled.
    """
    if config.get('version') != 'CALMET-5.8.5-UP-2.0':
        raise SchemaError('高空气象仅支持 version=CALMET-5.8.5-UP-2.0')
    if config.get('height_reference') != 'MSL':
        raise SchemaError('height_reference 必须明确为 MSL；不能把 AGL 高度当作海拔')
    if config.get('wind_direction_reference') != 'meteorological_from_true_north':
        raise SchemaError('wind_direction_reference 必须为 meteorological_from_true_north')
    projection = config.get('projection', 'NONE')
    if projection not in {'NONE', 'LL'}:
        raise SchemaError('UP.DAT 仅支持 NONE 或 LL 表头投影')
    if projection == 'LL' and config.get('datum') not in {'WGS-84', 'NAD-83', 'NAD-27'}:
        raise SchemaError('LL 表头需明确 datum=WGS-84/NAD-83/NAD-27，不能猜测坐标基准')
    cadence = config.get('sounding_interval_hours')
    if cadence is not None:
        cadence = _integer(cadence, 'sounding_interval_hours', (1, 24 * 31))
    if not isinstance(config.get('missing_values'), list):
        raise SchemaError('需明确 missing_values 列表（无缺测标记时提供 []）；高空转换不填补缺失值')
    stations = config.get('stations')
    if not isinstance(stations, list) or not stations:
        raise SchemaError('需 stations 列表，明确站点 ID、四字符名称及坐标/海拔')
    metadata = []
    for station in stations:
        if not isinstance(station, dict):
            raise SchemaError('每个 stations 项须为对象')
        station_id = _integer(station.get('id'), '站点 id', (1, 99999999))
        name = station.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,4}', name):
            raise SchemaError('站点 name 必须为 1..4 个 ASCII 字母、数字或下划线')
        metadata.append(dict(id=station_id, name=name,
            latitude=_real(station.get('latitude'), '纬度', (-90, 90)),
            longitude=_real(station.get('longitude'), '经度', (-180, 180)),
            elevation_m=_real(station.get('elevation_m'), '站点海拔 m', (-500, 9000))))
    if len({s['id'] for s in metadata}) != len(metadata):
        raise SchemaError('stations 中站点 ID 重复')
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SchemaError('高空源数据表为空')
    cols, units = config.get('columns', {}), config.get('units', {})
    mappings = {name: cols.get(name) for name in FIELDS}
    for name in ('station', 'time', 'level'):
        mappings[name] = config.get(f'{name}_column', {'station': 'station_id', 'time': 'time', 'level': 'level'}[name])
    if any(not isinstance(c, str) or not c for c in mappings.values()):
        raise SchemaError('必须明确映射 pressure/height/temperature/wind_direction/wind_speed')
    if len(set(mappings.values())) != len(mappings):
        raise SchemaError('站点、时间、层号及物理变量不能重复映射同一列')
    absent = [c for c in mappings.values() if c not in frame]
    if absent:
        raise SchemaError('缺少源字段：' + '、'.join(absent))
    if frame[list(mappings.values())].map(lambda v:isinstance(v,(bool,np.bool_))).any().any():
        raise SchemaError('高空数值字段不能为布尔值')
    selected = frame[list(mappings.values())].replace(config['missing_values'], np.nan)
    if selected.isna().any().any() or selected.map(lambda v: isinstance(v, str) and not v.strip()).any().any():
        raise SchemaError('高空必需字段存在缺失值/缺测标记；不能填零、删除层或插值')
    work = pd.DataFrame(index=frame.index)
    for name in ('station', 'level'):
        try:
            value = pd.to_numeric(selected[mappings[name]], errors='raise').to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f'{name} 列须为整数') from exc
        if not np.isfinite(value).all() or (value != np.rint(value)).any() or (value < 1).any() or (value > 99999999).any():
            raise SchemaError(f'{name} 列须为正整数')
        work[name] = value.astype(np.int64)
    if set(work.station) != {s['id'] for s in metadata}:
        raise SchemaError('源数据站点与 stations 配置不一致；不会丢弃未知站点')
    dates = []
    for raw in selected[mappings['time']]:
        try:
            value = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
            if value.utcoffset() is None:
                raise ValueError('时间缺少 UTC 偏移')
            value = value.astimezone(timezone.utc)
            if value.minute or value.second or value.microsecond or not 1900 <= value.year <= 2199:
                raise ValueError('需整点探测时间，年份限 1900..2199')
            dates.append(value)
        except ValueError as exc:
            raise SchemaError(f'时间 {raw!r} 无效：{exc}') from exc
    work['time'] = dates
    # RDUP treats height >=9998 m as missing, even for comma data.
    ranges = {'pressure': (0.01, 1100), 'height': (-500, 9997.999999),
              'temperature': (100, 400), 'wind_direction': (0, 360), 'wind_speed': (0, 200)}
    for name in FIELDS:
        if not isinstance(units.get(name), str) or not units[name].strip():
            raise SchemaError(f'需明确 {name} 的源单位')
        try:
            scale, offset = unit_transform(units[name], UNITS[name])
            value = pd.to_numeric(selected[mappings[name]], errors='raise').to_numpy(dtype=float) * scale + offset
        except (ValueError, TypeError, KeyError) as exc:
            raise SchemaError(f'{name} 数值或单位不正确：{exc}') from exc
        lo, hi = ranges[name]
        if not np.isfinite(value).all() or ((value < lo) | (value > hi)).any():
            raise SchemaError(f'{name} 超出暂定工程检查范围 [{lo}, {hi}] {UNITS[name]}，请核实单位或缺测标记')
        work[name] = value
    if work.duplicated(['station', 'time', 'level']).any():
        raise SchemaError('重复站点、探测时间、层号；不能任取一条')
    top = _real(config.get('top_pressure_hpa'), 'top_pressure_hpa', (1, 1099))
    # Header F5.0 retains a configured pressure only if exactly representable.
    if top != int(top):
        raise SchemaError('top_pressure_hpa 须为整数，符合官方 F5.0 头记录精度；不静默舍入')
    files, summaries = {}, []
    for number, station in enumerate(metadata, 1):
        records = work[work.station == station['id']]
        times = sorted(records.time.unique())
        if cadence is not None and len(times) > 1 and any((b - a).total_seconds() != cadence * 3600 for a, b in zip(times, times[1:])):
            raise SchemaError(f"站点 {station['id']} 探测时间不连续，配置间隔 {cadence} 小时；不会补造探测")
        start, end = times[0], times[-1]
        # RDUP uses YY, interpreted by QAYR4 relative to CALMET.INP IBYR.
        if end.year >= start.year + 50:
            raise SchemaError('数据跨越 50 年或以上；RDUP 两位年份无法按当前运行起始年无歧义读取')
        lines = [f'{"UP.DAT":16s}{"2.0":16s}{"EcoDataEngine CALMET 5.8.5 observed upper-air":64s}',
                 '1', 'Observed levels only; no vertical or temporal interpolation performed.', f'{projection:8s}']
        if projection == 'LL':
            if station['elevation_m'] != int(station['elevation_m']):
                raise SchemaError('LL 站点海拔须为整数，官方读取头记录为整数 m；NONE 模式保留外部站点精度')
            lines.extend([f"{config['datum']:8s}{'':10s}", 'DEG '])
        lines.append(' ' + ''.join(f'{int(v):5d}' for v in (start.year, start.dayofyear, start.hour, end.year, end.dayofyear, end.hour)) + f'{top:5.0f}{3:5d}{2:5d}')
        lines.append(' ' + '    F' * 4)
        if projection == 'LL':
            lat = f"{abs(station['latitude']):.8f}{'N' if station['latitude'] >= 0 else 'S'}"
            lon = f"{abs(station['longitude']):.8f}{'E' if station['longitude'] >= 0 else 'W'}"
            lines.append(f"{station['id']} {station['name']} {lat} {lon} {int(station['elevation_m'])}")
        counts = []
        for time in times:
            sounding = records[records.time == time].sort_values('level')
            n = len(sounding)
            if not 2 <= n <= 99:
                raise SchemaError(f"站点 {station['id']} {time} 有 {n} 层；本目标官方编译参数 MXLEV=99，需 2..99 层")
            if not np.array_equal(sounding.level.to_numpy(), np.arange(1, n + 1)):
                raise SchemaError('每次探测层号必须从 1 开始连续；不能隐藏缺层')
            if (np.diff(sounding.height) <= 0).any() or (np.diff(sounding.pressure) >= 0).any():
                raise SchemaError('按层号，高度须严格增加、压力须严格下降；不会自动排序修复异常剖面')
            if sounding.pressure.iloc[-1] > top:
                raise SchemaError(f'探测顶层压力大于声明的 {top} hPa，不能声称已覆盖该顶层')
            if abs(sounding.height.iloc[0] - station['elevation_m']) > 100:
                raise SchemaError('探测第一层与站点海拔相差超过暂定 100 m 检查阈值；需提供近地面观测，不补造地面层')
            lines.append(' ' * 9 + f"{station['id']:8d}" + ' ' * 5 + time.strftime('%y%m%d%H') + ' ' * 35 + f'{n:5d}')
            # One level per record is legal Fortran list-directed continuation;
            # RDUP reads exactly 5*n values, then the next fixed header.
            for row in sounding.itertuples():
                lines.append(','.join(format(float(getattr(row, f)), '.17g') for f in FIELDS))
            counts.append(n)
        filename = f'UP{number}.DAT'
        files[filename] = ('\n'.join(lines) + '\n').encode('ascii')
        summaries.append(dict(filename=filename, **station, soundings=len(times), levels=counts,
                              start_utc=start.isoformat(), end_utc=end.isoformat(), calmet_inp_ibyr=start.year))
    report = dict(scope='CALMET 5.8.5 observed UP.DAT 2.0 formatted comma data',
        complete_model_input=False, native_model_input_verified=False, stations=summaries,
        target_units=UNITS, height_reference='MSL', time_reference='UTC',
        sounding_interval_hours=cadence, interpolation='none', missing_policy='reject',
        format_source='Official CALMET 5.8.5 RDHDU, RDUP, QAYR4; PARAMS.MET MXLEV=99',
        required_next_steps=['CALMET.INP: NUSTA, UPDAT file order, IDUSTA and IBYR matching report',
                             'NONE header requires station coordinates/datum configured in CALMET.INP',
                             'Soundings must bracket simulation times and cover model top; CALMET performs its own interpolation',
                             'GEO.DAT, surface/prognostic weather and control configuration; CALMET then CALPUFF project inputs'],
        range_basis='Provisional engineering ranges declared in adapter; no post-test relaxation')
    files['README_upper_air.txt'] = ('CALMET 5.8.5 UP.DAT observational preprocessing input only.\n'
        'No values/layers/soundings were generated or interpolated.\n'
        'Pressure hPa; height m MSL; temperature K; wind FROM true north degrees; speed m/s.\n'
        'Use NUSTA, UPDAT, IDUSTA and IBYR as listed in conversion_report.json.\n'
        'NONE headers require external station coordinates and datum in CALMET.INP.\n'
        'A converted file is not a complete CALMET/CALPUFF project or proof of a successful model run.\n').encode('ascii')
    return files, report
