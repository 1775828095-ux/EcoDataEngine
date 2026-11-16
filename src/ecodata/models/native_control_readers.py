"""Read-only parser functions reused verbatim from the existing local platform.
No runner, uploaded script, or scientific configuration is executed.
See native_control_reader_origins.json for source hashes.
"""

import re, math

from datetime import datetime

TOKEN=re.compile(r'!(.*?)!',re.S)

def _groups(raw):
    """Only inspect scalar keys needed here, never rewrite a Fortran namelist."""
    text=raw.decode('utf-8-sig')
    # Quoted strings may contain !. Preserve them while removing real comments.
    text=re.sub(r"'[^']*'|\"[^\"]*\"|![^\n]*",lambda m:'' if m[0].startswith('!') else m[0],text)
    result={}
    for match in re.finditer(r'(?is)&([a-z][a-z0-9_]*)\b(.*?)(?:^\s*/\s*$)',text,re.M):
        name,body=match[1].lower(),match[2]
        if name in result:raise ValueError('重复namelist组：'+name)
        keys=list(re.finditer(r'(?i)\b([a-z][a-z0-9_]*)\s*=',body))
        group={}
        for i,key in enumerate(keys):
            label=key[1].lower()
            if label in group:raise ValueError('重复namelist参数：'+name+'.'+label)
            group[label]=body[key.end():keys[i+1].start() if i+1<len(keys) else len(body)].strip().rstrip(',').strip()
        result[name]=group
    return result

def _integer(group,key):
    value=group.get(key,'').split(',')[0].strip()
    if not re.fullmatch(r'[+-]?\d+',value):raise ValueError('缺少或不能识别整数参数：'+key)
    return int(value)

def _stamp(group,prefix):
    return datetime(*[_integer(group,prefix+'_'+key) for key in ('year','month','day','hour','minute','second')])

SPECIAL = {'null', 'sim', '0'}

def weather_dependencies(files, selected_station=None, *, unread_files=()):
    errors, warnings, links = [], [], []
    groups = {}
    for line in files.get('file.cio', b'').decode('utf-8-sig').splitlines()[1:]:
        row = line.split('!', 1)[0].split()
        if row:
            groups[row[0]] = row[1:]
    climate = groups.get('climate', [])
    if not climate:
        return dict(errors=[], warnings=['没有可检查的天气站表，站点依赖未验证'],
                    stations={}, connection_tables=[], scope='weather-name dependencies only')
    station_file = climate[0]
    if station_file not in files:
        return dict(errors=[f'file.cio引用的天气站文件不存在：{station_file}'], warnings=[], stations={}, connection_tables=[])
    if station_file in unread_files:
        return dict(errors=[f'天气站文件正文未读取：{station_file}'], warnings=[], stations={}, connection_tables=[])
    lines = files[station_file].decode('utf-8-sig').splitlines()
    if len(lines) < 2:
        return dict(errors=[f'{station_file}缺少站点表头'], warnings=[], stations={}, connection_tables=[])
    header = lines[1].split()
    if len(set(header)) != len(header) or not {'name', 'wgn'} <= set(header):
        return dict(errors=[f'{station_file}站点表头不能识别或字段重复'], warnings=[], stations={}, connection_tables=[])
    stations = {}
    for line_number, line in enumerate(lines[2:], 3):
        row = line.split()
        if not row:
            continue
        if len(row) != len(header):
            errors.append(f'{station_file}:{line_number}站点记录列数与表头不符')
            continue
        name = row[header.index('name')]
        if name in stations:
            errors.append(f'{station_file}天气站名称重复：{name}')
        stations[name] = dict(wgn=row[header.index('wgn')], object_count=0,
                              generated_variables=[k for k in ('pcp','tmp','slr','hmd','wnd')
                                                   if k in header and row[header.index(k)].lower() in SPECIAL])
    checked = []
    for name in groups.get('connect', []):
        if name.lower() in SPECIAL:
            continue
        if name not in files:
            errors.append(f'file.cio/connect引用缺失：{name}')
            continue
        if name in unread_files:
            warnings.append(f'{name}正文未读取，未验证其中的天气站关联')
            continue
        rows = files[name].decode('utf-8-sig').splitlines()
        cols = rows[1].split() if len(rows) > 1 else []
        if not {'id','name','wst'} <= set(cols) or len(set(cols)) != len(cols):
            warnings.append(f'{name}未识别id/name/wst表头，未验证其中的天气站关联')
            continue
        summary = dict(file=name, objects=0, explicit_station_links=0, unresolved=0)
        seen = set()
        for line_number, line in enumerate(rows[2:], 3):
            row = line.split()
            if not row:
                continue
            if len(row) <= max(cols.index(k) for k in ('id','name','wst')):
                errors.append(f'{name}:{line_number}缺少对象或天气站字段')
                continue
            ident, obj, station = (row[cols.index(k)] for k in ('id','name','wst'))
            if ident in seen:
                errors.append(f'{name}对象ID重复：{ident}')
            seen.add(ident)
            summary['objects'] += 1
            if station in stations:
                stations[station]['object_count'] += 1
                summary['explicit_station_links'] += 1
                links.append(dict(file=name, id=ident, name=obj, station=station))
            elif station.lower() in SPECIAL:
                summary['unresolved'] += 1
            else:
                errors.append(f'{name}:{line_number}对象{obj}引用不存在的天气站：{station}')
        if summary['unresolved']:
            warnings.append(f'{name}有{summary["unresolved"]}个特殊wst引用，未推断其自动分配结果')
        checked.append(summary)
    if not checked:
        warnings.append('未检查到可识别的对象连接表，不能证明天气站实际被计算对象使用')
    if selected_station in stations and checked and not stations[selected_station]['object_count']:
        warnings.append(f'所选天气站{selected_station}没有已识别的显式对象引用；不能保证替换会影响模拟')

    required = {v['wgn'] for v in stations.values() if v['wgn'].lower() not in SPECIAL}
    wgn_file = climate[1] if len(climate) > 1 else 'null'
    wgn_stations = {}
    if required:
        if wgn_file not in files:
            errors.append(f'WGN引用缺失：{wgn_file}；需要统计站{sorted(required)}')
        elif wgn_file in unread_files:
            errors.append(f'WGN文件正文未读取：{wgn_file}')
        else:
            rows = [r.split() for r in files[wgn_file].decode('utf-8-sig').splitlines()[1:] if r.strip()]
            pos = 0
            while pos < len(rows):
                station = rows[pos]
                if len(station) != 5:
                    errors.append(f'{wgn_file}第{pos+2}行WGN站点头需名称/纬度/经度/高程/统计年数5项')
                    break
                name = station[0]
                try:
                    lat, lon, elev, years = map(float, station[1:])
                    if not all(math.isfinite(v) for v in (lat,lon,elev,years)) or not (-90 <= lat <= 90 and -180 <= lon <= 180 and years >= 1):
                        raise ValueError()
                except ValueError:
                    errors.append(f'{wgn_file}统计站{name}位置或统计年数无效')
                if name in wgn_stations:
                    errors.append(f'{wgn_file}WGN统计站名称重复：{name}')
                month_rows = rows[pos+2:pos+14]
                count = len(month_rows)
                if count != 12:
                    errors.append(f'{wgn_file}统计站{name}需要12个月统计，实际只有{count}行；不会补值')
                if pos+1 >= len(rows) or len(rows[pos+1]) != 14:
                    errors.append(f'{wgn_file}统计站{name}月统计表头需14字段')
                for mo, values in enumerate(month_rows, 1):
                    try:
                        if len(values) != 14 or not all(math.isfinite(float(v)) for v in values):
                            raise ValueError()
                    except ValueError:
                        errors.append(f'{wgn_file}统计站{name}第{mo}个月需14个有限数值')
                wgn_stations[name] = dict(months=count)
                pos += 14
            for name in sorted(required - wgn_stations.keys()):
                errors.append(f'天气站引用的WGN统计站不存在：{name}（{wgn_file}）')
    for name, station in stations.items():
        if station['generated_variables'] and station['wgn'].lower() in SPECIAL:
            warnings.append(f'天气站{name}存在生成/特殊气象引用，但未指定具名WGN统计站；生成行为尚未验证')
    return dict(errors=sorted(set(errors)), warnings=sorted(set(warnings)),
                stations=stations, connection_tables=checked, object_station_links=links,
                wgn_file=wgn_file, wgn_stations=wgn_stations,
                scope='仅检查天气站、显式wst关联及具名WGN的12月/14字段结构；不检查完整拓扑、参数精度或空间代表性')

def control_values(text):
    result={}
    for t in TOKEN.finditer(text):
        m=re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*',t[1],re.S)
        if m:result.setdefault(m[1].upper(),[]).append(m[2].strip())
    return result

PATTERN=re.compile(r'\s*([A-Za-z][A-Za-z_0-9]*(?:\([\d, ]+\))?)\s*=\s*(.*?)\s*,\s*(?:!.*)?$')
