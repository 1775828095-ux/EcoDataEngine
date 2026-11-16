"""Read SWAT+ 62 text outputs without fabricating dates, units or values."""
from __future__ import annotations
import csv
from datetime import date, timedelta
import math
from pathlib import Path
import re

LABELS = {
    'precip': '降水', 'et': '蒸散发', 'surq_gen': '地表径流',
    'wateryld': '产水量', 'flo_out': '流出流量', 'flo_in': '流入流量',
    'pet': '潜在蒸散发', 'sw_final': '期末土壤水量',
}

def _select_output_lines(path,selected_identity=None,index_only=False):
    """Stream the identity index; retain at most one object's physical rows.

    The index is not a physical-value quality certificate. read_output then
    validates every variable and date for the selected object as usual.
    """
    path=Path(path)
    if path.stat().st_size>512*1024*1024:raise ValueError('单个结果文件超过512 MiB，请缩小输出范围。')
    lines=[];line_numbers=[];objects={};header=None
    with path.open(encoding='utf-8-sig',errors='strict') as handle:
        for number,line in enumerate(handle,1):
            if header is None:
                if number>20:raise ValueError('未找到SWAT+日期表头（jday mon day yr）')
                lines.append(line.rstrip('\r\n'));line_numbers.append(number)
                if line.split()[:4]==['jday','mon','day','yr']:header=number
                continue
            if number==header+1:
                lines.append(line.rstrip('\r\n'));line_numbers.append(number);continue
            if not line.strip():continue
            tokens=line.split()
            try:
                identity=(int(tokens[4]),int(tokens[5]),tokens[6])
            except (ValueError,IndexError) as exc:raise ValueError(f'第{number}行对象标识无效') from exc
            objects.setdefault(identity,None)
            if not index_only and identity==selected_identity:
                lines.append(line.rstrip('\r\n'));line_numbers.append(number)
    if header is None:raise ValueError('未找到SWAT+日期表头（jday mon day yr）')
    return lines,line_numbers,list(objects)


def output_objects(path):
    """Candidate object index only; numerical validation occurs on selection."""
    return _select_output_lines(path,index_only=True)[2]


def has_output_header(lines):
    return any(line.split()[:4]==["jday","mon","day","yr"] for line in lines[:20])


def output_header(lines):
    """Read field names and declared units from a SWAT+ result header."""
    header_index = next((i for i, line in enumerate(lines[:20])
                         if line.split()[:4] == ['jday', 'mon', 'day', 'yr']), None)
    if header_index is None:
        raise ValueError('未找到SWAT+日期表头（jday mon day yr）；不把输入文件当作结果。')
    raw_columns = lines[header_index].split()
    counts = {}
    columns = []
    for key in raw_columns:
        counts[key] = counts.get(key, 0) + 1
        columns.append(key if counts[key] == 1 else f'{key}[{counts[key]}]')
    # SWAT prints units only for physical variables; identity columns have no units.
    identity = ['jday', 'mon', 'day', 'yr', 'unit', 'gis_id', 'name']
    if columns[:7] != identity:
        raise ValueError('暂不支持此结果身份列结构；需要jday mon day yr unit gis_id name。')
    if len(lines)<=header_index+1:
        raise ValueError('SWAT+ 结果缺少单位行。')
    unit_tokens = lines[header_index + 1].split()
    # 62.0.0 water-balance header has two trailing non-physical diagnostics
    # (plant_cov, mgt_ops) without printed units. Keep them out of the chart selector.
    no_unit_tail = raw_columns[-2:] == ['plant_cov', 'mgt_ops'] and len(unit_tokens) == len(columns)-9
    unit_columns = [col for col,raw in zip(columns[7:],raw_columns[7:]) if raw != 'null']
    if no_unit_tail:
        unit_columns = columns[7:-2]
    if len(unit_tokens) != len(unit_columns):
        raise ValueError('单位行与变量数不一致，拒绝猜测单位。')
    units = dict(zip(unit_columns, unit_tokens))
    return header_index,columns,units,no_unit_tail



def read_output(path,selected_identity=None):
    path = Path(path)
    if selected_identity is None and path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError('单个结果文件超过100 MiB，请先缩小输出范围。')
    if selected_identity is None:
        lines = path.read_text(encoding='utf-8-sig', errors='strict').splitlines()
        line_numbers=list(range(1,len(lines)+1))
    else:
        lines,line_numbers,_=_select_output_lines(path,selected_identity)
    header_index, columns, units, no_unit_tail = output_header(lines)
    rows = []
    for lineno, line in enumerate(lines[header_index + 2:], header_index + 3):
        lineno=line_numbers[lineno-1]
        if not line.strip():
            continue
        values = line.split()
        # Basin rows omit the two trailing HRU-only diagnostic strings in 62.
        omitted_tail = no_unit_tail and path.name.startswith('basin_') and len(values) == len(columns)-2
        if len(values) != len(columns) and not omitted_tail:
            raise ValueError(f'第{lineno}行字段数错误或数值溢出。')
        row = dict(zip(columns, values))
        try:
            for key in columns[:6]:
                row[key] = int(row[key])
            for key in units:
                row[key] = float(row[key].replace('D', 'E'))
                if not math.isfinite(row[key]) or abs(row[key]) >= 1e30:
                    raise ValueError('非有限值或缺测标记')
            # Explicit daily date, not an inferred row counter.
            if '_day.' in path.name:
                dt = date(row['yr'], row['mon'], row['day'])
                if dt.timetuple().tm_yday != row['jday']:
                    raise ValueError('年积日与日期不一致')
            elif '_mon.' in path.name:
                dt = date(row['yr'], row['mon'], 1)
            elif '_yr.' in path.name:
                dt = date(row['yr'], 1, 1)
            else:
                raise ValueError('仅支持逐日、逐月、逐年结果，不为多年平均编造日期')
            row['_date'] = dt
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'第{lineno}行无效：{exc}') from exc
        rows.append(row)
    if not rows:
        raise ValueError('此文件没有数据行。请检查输出开关和跳过年数。')
    identities = {}
    for row in rows:
        key = (row['unit'], row['gis_id'], row['name'])
        identities.setdefault(key, []).append(row)
    for group in identities.values():
        dates = [r['_date'] for r in group]
        if dates != sorted(set(dates)):
            raise ValueError('同一对象有重复或倒序日期。')
        if '_day.' in path.name and any(b-a != timedelta(days=1) for a,b in zip(dates, dates[1:])):
            raise ValueError('逐日结果存在缺日期；未填值或连接缺口。')
    return {'path': str(path), 'columns': columns, 'units': units,
            'validation_scope':'selected_object' if selected_identity is not None else 'all_objects',
            'rows': rows, 'objects': list(identities)}

def discover(directory):
    valid, rejected = [], []
    for path in sorted(Path(directory).glob('*.txt')):
        if not re.search(r'_(day|mon|yr)\.txt$', path.name):
            continue
        try:
            data = read_output(path)
            valid.append(data)
        except (ValueError, UnicodeError, OSError) as exc:
            rejected.append(f'{path.name}：{exc}')
    return valid, rejected

def series(data, variable, identity):
    if variable not in data['units']:
        raise ValueError('变量没有明确单位。')
    rows = [r for r in data['rows'] if (r['unit'], r['gis_id'], r['name']) == identity]
    if not rows:
        raise ValueError('该对象没有记录。')
    return [r['_date'] for r in rows], [r[variable] for r in rows]

def export_csv(path, data, variable, identity):
    dates, values = series(data, variable, identity)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['date', 'unit', 'gis_id', 'name', variable, 'physical_unit'])
        for dt, value in zip(dates, values):
            writer.writerow([dt.isoformat(), *identity, repr(value), data['units'][variable]])

def enable_charts(files, engine_id=None):
    """Only change explicit print controls in a new run copy, never physics."""
    from ecodata.interface import swat_runner as runner
    report = runner.check_project(files, engine_id=engine_id)
    if report['errors']:
        raise ValueError('；'.join(report['errors']))
    # Resolve print file from actual simulation group, not a hardcoded filename.
    cio = files['file.cio'].decode('utf-8-sig').splitlines()
    simulation = next(line.split('!')[0].split()[1:] for line in cio[1:]
                      if line.split() and line.split()[0] == 'simulation')
    if len(simulation) < 2 or simulation[1] not in files:
        raise ValueError('simulation组未提供有效print文件，无法开启图表输出。')
    name = simulation[1]
    lines = files[name].decode('utf-8-sig').splitlines()
    control = next((i for i,line in enumerate(lines) if line.split()[:2] == ['nyskip', 'day_start']), None)
    if control is None or control+1 >= len(lines):
        raise ValueError('无法识别62.0.0打印设置，不自动改写未知格式。')
    tokens = lines[control+1].split()
    if len(tokens) != 6:
        raise ValueError('打印日期设置需要6个字段。')
    # Include the selected simulation dates. Do not alter model warmup physics.
    lines[control+1] = '0 0 0 0 0 1'
    changed = []
    for i,line in enumerate(lines):
        tokens = line.split()
        if tokens and tokens[0] in {'basin_wb', 'hru_wb', 'channel_sd', 'channel'}:
            if len(tokens) != 5 or any(x not in {'y','n'} for x in tokens[1:]):
                raise ValueError(f'未知打印开关结构：{tokens[0]}')
            tokens[1] = 'y'
            lines[i] = ' '.join(tokens)
            changed.append(tokens[0])
    if not changed:
        raise ValueError('打印设置不含支持的水量或河道输出，未创建空图表。')
    result = dict(files)
    result[name] = ('\n'.join(lines)+'\n').encode('utf-8')
    return result, {'changed_file': name, 'daily_outputs': changed,
                    'nyskip': 0, 'print_date_filter': 'all simulation dates',
                    'physical_parameters_changed': False}
