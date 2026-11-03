"""Attach converted single-station weather to an existing SWAT+ 61/62 project.

Never extract archives or alter physical parameters. Only selected weather-sta
columns, the corresponding filename indices and new measurement files change.
"""
import calendar
import datetime as dt
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import PurePosixPath

RESERVED = {'null', 'sim'}
TYPES = {'pcp': ('station.pcp', 'pcp.cli', 'pcp'),
         'tmp': ('station.tem', 'tmp.cli', 'tmp'),
         'slr': ('station.slr', 'slr.cli', 'slr'),
         'hmd': ('station.hmd', 'hmd.cli', 'hmd'),
         'wnd': ('station.wnd', 'wnd.cli', 'wnd')}


def read_weather_zip(payload):
    """Read only known weather measurement files without extracting anything."""
    if len(payload) > 100 * 1024**2:
        raise ValueError('天气ZIP超过100 MiB')
    accepted = {v[0] for v in TYPES.values()}
    result = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        infos = z.infolist()
        if len(infos) > 3000 or sum(i.file_size for i in infos) > 100 * 1024**2:
            raise ValueError('天气ZIP解压大小或数量超限')
        for item in infos:
            if item.filename in accepted:
                if item.filename in result:
                    raise ValueError('天气ZIP有重复文件')
                result[item.filename] = z.read(item)
    if not {'station.pcp', 'station.tem'} <= result.keys():
        raise ValueError('请上传平台日天气转换ZIP，需要station.pcp和station.tem')
    return result


def _read_zip(payload):
    if len(payload) > 100 * 1024**2:
        raise ValueError('工程ZIP超过100 MiB限制')
    files = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        infos = z.infolist()
        if len(infos) > 3000 or sum(i.file_size for i in infos) > 100 * 1024**2:
            raise ValueError('工程解压大小或文件数量超过限制')
        for i in infos:
            path = PurePosixPath(i.filename)
            reserved = {'con', 'prn', 'aux', 'nul', *(f'com{n}' for n in range(1, 10)), *(f'lpt{n}' for n in range(1, 10))}
            if ('\\' in i.filename or path.is_absolute() or '..' in path.parts
                    or any(c in i.filename for c in ':<>"|?*') or any(ord(c) < 32 for c in i.filename)
                    or any(p.endswith(('.', ' ')) or p.split('.')[0].lower() in reserved for p in path.parts)
                    or (i.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('工程ZIP包含不安全路径或符号链接')
            if i.is_dir():
                continue
            if path.suffix.lower() in {'.exe', '.dll', '.bat', '.cmd', '.ps1', '.py', '.sh', '.com', '.scr', '.lnk'}:
                raise ValueError('工程ZIP只能包含输入资料，不能包含程序或脚本')
            if i.filename.casefold() in {n.casefold() for n in files}:
                raise ValueError('工程ZIP存在重复文件名')
            files[i.filename] = z.read(i)
    # macOS ZIPs carry Finder/resource-fork metadata alongside the real folder.
    # Validate their paths above, then exclude only these known metadata entries.
    files = {n: v for n, v in files.items() if not (
        PurePosixPath(n).parts[0] == '__MACOSX'
        or PurePosixPath(n).name == '.DS_Store'
        or PurePosixPath(n).name.startswith('._'))}
    roots = [n[:-len('file.cio')] for n in files if PurePosixPath(n).name == 'file.cio']
    if len(roots) != 1:
        raise ValueError('工程ZIP必须恰有一个file.cio；请上传单个流域工程')
    root = roots[0]
    if any(not n.startswith(root) or '/' in n[len(root):] for n in files):
        raise ValueError('仅支持一个平铺工程目录，或ZIP内单层工程文件夹')
    return {n[len(root):]: v for n, v in files.items()}


def _table(data):
    lines = data.decode('utf-8-sig').splitlines()
    if len(lines) < 3:
        raise ValueError('weather-sta.cli缺少表头或站点')
    header = lines[1].split()
    if not {'name', 'wgn', *TYPES} <= set(header) or len(set(header)) != len(header):
        raise ValueError('weather-sta.cli字段不符合SWAT+62天气站表')
    rows = [line.split() for line in lines[2:] if line.strip()]
    if any(len(row) != len(header) for row in rows):
        raise ValueError('weather-sta.cli站点记录列数错误')
    names = [r[header.index('name')] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError('weather-sta.cli站点名称重复')
    return lines, header, rows


def project_stations(payload):
    files = _read_zip(payload)
    _, _, climate, _ = _layout(files)
    _, header, rows = _table(files[climate[1]])
    return [r[header.index('name')] for r in rows]


def _layout(files):
    """Resolve native references, without guessing from example filenames."""
    cio = files['file.cio'].decode('utf-8-sig').splitlines()
    groups = {}
    for i, line in enumerate(cio[1:], 1):
        tokens = line.split('!', 1)[0].split()
        if not tokens:
            continue
        if tokens[0] in groups:
            raise ValueError(f'file.cio分组重复：{tokens[0]}')
        groups[tokens[0]] = (i, tokens)
    ci, climate = groups.get('climate', (None, []))
    if len(climate) != 10:
        raise ValueError('file.cio climate行不符合支持的SWAT+61/62十列结构')
    simulation = groups.get('simulation', (None, []))[1]
    time_name = simulation[1] if len(simulation) > 1 else 'time.sim'
    for name in [time_name, *climate[1:]]:
        if name == 'null':
            continue
        if PurePosixPath(name).name != name or name in {'.', '..'}:
            raise ValueError(f'仅支持平铺工程内的文件引用：{name}')
    for name in [climate[1], time_name]:
        if name == 'null' or name not in files:
            raise ValueError(f'工程引用缺失：{name}')
    return cio, ci, climate, time_name


def inspect_weather_project(payload):
    from .project_dependencies import weather_dependencies
    files = _read_zip(payload)
    cio, _, climate, time_name = _layout(files)
    _, header, rows = _table(files[climate[1]])
    return {'declared_header': cio[0], 'station_table': climate[1],
            'time_file': time_name, 'weather_indices': dict(zip(TYPES, climate[4:9])),
            'stations': [r[header.index('name')] for r in rows],
            'file_count': len(files),
            'weather_dependencies': weather_dependencies(files),
            'scope': '识别实际引用；不代表完整工程深层引用、版本兼容或原生运行已通过'}


def _weather_metadata(data):
    lines = data.decode('utf-8-sig').splitlines()
    if len(lines) < 3 or [v.upper() for v in lines[1].split()] != ['NBYR','TSTEP','LAT','LONG','ELEV']:
        raise ValueError('转换天气表头需NBYR TSTEP LAT LONG ELEV，不猜测单位或列含义')
    row = lines[2].split()
    if len(row) != 5:
        raise ValueError('转换天气站点元数据需5字段')
    years, step = int(row[0]), int(row[1])
    lat, lon, elev = map(float, row[2:])
    if years < 1 or step != 0:
        raise ValueError('转换天气仅支持NBYR>=1且TSTEP=0逐日资料')
    if not all(math.isfinite(v) for v in (lat,lon,elev)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('转换天气经纬度或高程无效')
    return dict(years=years, step=step, latitude=lat, longitude=lon, elevation_m=elev)


def _records(data, kind):
    metadata = _weather_metadata(data)
    lines = data.decode('utf-8-sig').splitlines()
    if len(lines) < 4:
        raise ValueError('转换天气文件缺少表头或日记录')
    records = [line.split() for line in lines[3:] if line.strip()]
    if not records:
        raise ValueError('转换天气文件没有日记录')
    dates = []
    for row in records:
        if len(row) != (4 if kind == 'tmp' else 3):
            raise ValueError('天气日记录列数错误')
        year, day = int(row[0]), int(row[1])
        values = [float(v) for v in row[2:]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('天气数值包含非有限值')
        if kind == 'tmp' and values[0] < values[1]:
            raise ValueError('最高温小于最低温')
        if kind != 'tmp' and any(v < 0 for v in values):
            raise ValueError('天气数值为负值或缺测标记；不会自动补值')
        if kind == 'hmd' and values[0] > 1:
            raise ValueError('相对湿度必须使用0至1比例')
        if not 1 <= day <= (366 if calendar.isleap(year) else 365):
            raise ValueError('天气日期无效')
        dates.append(dt.date(year, 1, 1) + dt.timedelta(days=day-1))
    if any(b-a != dt.timedelta(days=1) for a, b in zip(dates, dates[1:])):
        raise ValueError('天气日期缺失、重复或顺序错误')
    if metadata['years'] != len({d.year for d in dates}):
        raise ValueError('转换天气NBYR与实际日记录年份数不一致')
    return dates


def _check_refs(files):
    _, _, climate, _ = _layout(files)
    _, header, rows = _table(files[climate[1]])
    for kind, index in zip(TYPES, climate[4:9]):
        refs = {r[header.index(kind)] for r in rows} - RESERVED
        indexed = set()
        if index != 'null' and index in files:
            lines = files[index].decode('utf-8-sig').splitlines()
            if len(lines) < 2 or [token.casefold() for token in lines[1].split()] != ['filename']:
                raise ValueError(f'{index}表头必须为filename')
            indexed = {line.strip() for line in lines[2:] if line.strip()}
            if any(len(n.split()) != 1 for n in indexed):
                raise ValueError(f'{index}包含非法文件名')
        for name in refs | indexed:
            if name not in files:
                raise ValueError(f'天气引用缺失：{index}或weather-sta.cli → {name}')
        if not refs <= indexed:
            raise ValueError(f'{index}未登记站点引用：{sorted(refs-indexed)}')
    if any(r[header.index('wgn')] not in RESERVED for r in rows):
        if climate[2] == 'null' or climate[2] not in files:
            raise ValueError('工程站点引用WGN，但缺少所引用WGN文件')


def repair_missing_weather_indices(payload):
    """Recreate only missing, explicitly named lists from existing references.

    Never infer a measurement, station association, or inactive index. The
    caller receives an explicit manifest, and the source archive is unchanged.
    """
    files = _read_zip(payload)
    _, _, climate, _ = _layout(files)
    _, header, rows = _table(files[climate[1]])
    rebuilt = []
    for kind, name in zip(TYPES, climate[4:9]):
        if name == 'null' or name in files:
            continue
        refs = sorted({r[header.index(kind)] for r in rows} - RESERVED)
        if not refs:
            continue  # Generated-only/version-specific behavior is not guessed.
        missing = [ref for ref in refs if ref not in files]
        if missing:
            raise ValueError(f'索引重建缺少已有观测文件：{name} → {missing}')
        files[name] = ('Rebuilt from explicit weather-station references; no values changed\nfilename\n'+'\n'.join(refs)+'\n').encode()
        rebuilt.append(dict(file=name, variable=kind, references=refs, sha256=hashlib.sha256(files[name]).hexdigest()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return out.getvalue(), dict(reconstructed_indices=rebuilt, original_sha256=hashlib.sha256(payload).hexdigest(),
                                scope='仅重建file.cio已启用但缺失的索引；沿用站点表中明确存在的文件，不改测量数据或配置')


def attach_weather(project_zip, weather_files, station, *, simulation_period=None):
    files = _read_zip(project_zip)
    original = dict(files)
    repaired, index_repairs = repair_missing_weather_indices(project_zip)
    files = _read_zip(repaired)
    cio, ci, climate, time_name = _layout(files)
    original_climate = list(climate)
    station_table = climate[1]
    lines, header, rows = _table(files[station_table])
    selected = [r for r in rows if r[header.index('name')] == station]
    if len(selected) != 1 or not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', station):
        raise ValueError('请选择工程中唯一且名称合法的目标天气站')
    if not {'station.pcp', 'station.tem'} <= set(weather_files):
        raise ValueError('仅支持已通过writer校验的单站日天气转换结果')
    for kind, name in zip(TYPES, climate[4:9]):
        if name != 'null' and name not in files:
            if (kind in {'hmd', 'wnd'} and re.search(r'\brev\.?\s*61(?:\.|\b)', cio[0], re.I)
                    and all(r[header.index(kind)] in RESERVED for r in rows)):
                continue
            raise ValueError(f'天气索引引用缺失：{name}')
    if climate[2] != 'null' and climate[2] not in files:
        raise ValueError(f'file.cio引用缺失：{climate[2]}')
    time_rows = [l.split() for l in files[time_name].decode('utf-8-sig').splitlines()[2:] if l.strip()]
    if len(time_rows) != 1 or len(time_rows[0]) != 5:
        raise ValueError('工程time.sim格式不受支持')
    ds, ys, de, ye, step = map(int, time_rows[0])
    if step != 0:
        raise ValueError('仅支持time.sim step=0的逐日工程')
    ds = ds or 1
    de = de or (366 if calendar.isleap(ye) else 365)
    if not 1 <= ds <= (366 if calendar.isleap(ys) else 365) or not 1 <= de <= (366 if calendar.isleap(ye) else 365):
        raise ValueError('工程time.sim日期无效')
    start = dt.date(ys, 1, 1) + dt.timedelta(days=ds-1)
    end = dt.date(ye, 1, 1) + dt.timedelta(days=de-1)
    if end < start:
        raise ValueError('工程起止日期倒置')
    original_period = [str(start), str(end)]
    if simulation_period is not None:
        start, end = simulation_period
        if type(start) is not dt.date or type(end) is not dt.date or start > end:
            raise ValueError('请选择有效且顺序正确的本次模拟起止日期')
        time_lines = files[time_name].decode('utf-8-sig').splitlines()
        time_lines[2:] = [f'{start.timetuple().tm_yday} {start.year} {end.timetuple().tm_yday} {end.year} 0']
        files[time_name] = ('\n'.join(time_lines)+'\n').encode('utf-8')
    changes = []
    all_dates = None
    weather_metadata = {}
    for kind, (source, index, ext) in TYPES.items():
        if source not in weather_files:
            continue
        pos = 4 + list(TYPES).index(kind)
        if climate[pos] != 'null':
            index = climate[pos]
        elif index in files:
            raise ValueError(f'新启用索引名称冲突：{index}；请先明确工程引用')
        dates = _records(weather_files[source], kind)
        weather_metadata[kind] = _weather_metadata(weather_files[source])
        if all_dates is not None and dates != all_dates:
            raise ValueError('转换天气各变量日期不一致')
        all_dates = dates
        if dates[0] > start or dates[-1] < end:
            raise ValueError(f'天气日期不足：工程需要{start}至{end}，输入只有{dates[0]}至{dates[-1]}；不会补日期或自动改变工程运行期')
        target = f'ecodata_{station}.{ext}'
        if any(r is not selected[0] and r[header.index(kind)] == target for r in rows):
            raise ValueError(f'{target}还被其他站点引用，拒绝影响其他站点')
        if target.casefold() in {n.casefold() for n in files} and selected[0][header.index(kind)] != target:
            raise ValueError(f'自动目标文件名冲突：{target}')
        old = selected[0][header.index(kind)]
        selected[0][header.index(kind)] = target
        files[target] = weather_files[source]
        if index in files:
            idx = files[index].decode('utf-8-sig').splitlines()
            if len(idx) < 2 or [token.casefold() for token in idx[1].split()] != ['filename']:
                raise ValueError(f'{index}格式不受支持')
        else:
            idx = [f'{index}: EcoDataEngine SWAT+62 weather attachment', 'filename']
            # Enabling an index in a multi-station project must register all
            # existing station references, including stations not updated here.
            for row in rows:
                ref = row[header.index(kind)]
                if ref.lower() not in {'null', 'sim', '0'} and ref not in idx[2:]:
                    idx.append(ref)
        if target not in {l.strip() for l in idx[2:]}:
            idx.append(target)
        files[index] = ('\n'.join(idx)+'\n').encode()
        climate[4 + list(TYPES).index(kind)] = index
        changes.append({'variable': kind, 'source': source, 'previous_reference': old,
                        'target': target, 'sha256': hashlib.sha256(weather_files[source]).hexdigest()})
    files[station_table] = ('\n'.join(lines[:2])+ '\n' + '\n'.join(' '.join(r) for r in rows)+'\n').encode()
    cio[ci] = ' '.join(climate)
    # Preserve file.cio byte-for-byte when no climate reference changes are needed.
    if original_climate != climate:
        files['file.cio'] = ('\n'.join(cio)+'\n').encode()
    _check_refs(files)
    from .project_dependencies import weather_dependencies
    dependencies = weather_dependencies(files, station)
    if dependencies['errors']:
        raise ValueError('；'.join(dependencies['errors']))
    for pos, kind in enumerate(TYPES, 4):
        if climate[pos] == 'null' and any(r[header.index(kind)] not in RESERVED for r in rows):
            raise ValueError(f'weather-sta.cli引用{kind}实测文件，但file.cio没有启用对应索引')
    for pos, token in enumerate(climate[1:], 1):
        if token != 'null' and token not in files:
            # rev.61's humidity/wind readers allow absent indexes when all
            # stations explicitly request generated weather. Keep that choice.
            declared61 = bool(re.search(r'\brev\.?\s*61(?:\.|\b)', cio[0], re.I))
            kind = {7: 'hmd', 8: 'wnd'}.get(pos)
            if declared61 and kind and all(r[header.index(kind)] in RESERVED for r in rows):
                continue
            raise ValueError(f'file.cio天气引用缺失：{token}')
    report = {'target_format': 'SWAT+61/62 shared daily weather layout; native validation required', 'station': station, 'changes': changes,
              'weather_dependencies': dependencies,
              'index_repairs': index_repairs,
              'weather_metadata': weather_metadata,
              'unit_contract': {'pcp': 'mm/day', 'tmp': 'degC (max,min)', 'slr': 'MJ/m2/day', 'hmd': 'fraction 0..1', 'wnd': 'm/s'},
              'unit_verification': '按平台日天气writer格式接入；原生文本未逐项声明单位，不能独立认证来源单位或地理代表性',
              'resolved_layout': inspect_weather_project(project_zip),
              'file_manifest': [{'file': name, 'action': 'added' if name not in original else 'modified' if content != original[name] else 'unchanged',
                                 'before_sha256': hashlib.sha256(original[name]).hexdigest() if name in original else None,
                                 'after_sha256': hashlib.sha256(content).hexdigest()} for name, content in sorted(files.items()) if name != 'ecodata_weather_attachment.json'],
              'manifest_scope': '除本报告自身外的全部工程文件；原报告如存在将由本次核查报告替换',
              'simulation_period': [str(start), str(end)], 'weather_records': len(all_dates),
              'original_simulation_period': original_period,
              'simulation_period_selection': '用户明确选择本次副本日期' if simulation_period is not None else '保留原工程日期',
              'weather_period': [str(all_dates[0]), str(all_dates[-1])],
              'original_project_sha256': hashlib.sha256(project_zip).hexdigest(),
              'preserved': 'WGN、其他站点及土壤河道管理参数不变；原天气文件保留；file.cio仅在必要时更新climate索引引用；时间文件仅按用户明确选择在副本中调整，详见日期及文件清单',
              'verification': '已检查天气文件与变量索引引用、所选日天气日期覆盖；未验证完整工程深层引用或模型运行',
              'warning': '用户须确认目标站点与原始天气的实际地理适用性；不重新分配HRU站点、不补造气象或工程参数'}
    files['ecodata_weather_attachment.json'] = json.dumps(report, ensure_ascii=False, indent=2).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return out.getvalue(), report
