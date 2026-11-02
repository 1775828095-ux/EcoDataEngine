"""SWAT+ 62 aquifer.aqu parameter database from supplied engineering parameters.

The writer does not derive aquifer geometry, aquifer.con connectivity, or initial
chemistry.  It checks the direct aqu_ini name against a supplied initial.aqu.
"""
import csv
from decimal import Decimal, InvalidOperation
import io
import re
import zipfile

from ecodata.core.exceptions import SchemaError

FIELDS = ('id', 'name', 'init', 'gw_flo', 'dep_bot', 'dep_wt', 'no3_n', 'sol_p',
          'carbon', 'flo_dist', 'bf_max', 'alpha_bf', 'revap', 'rchg_dp',
          'spec_yld', 'hl_no3n', 'flo_min', 'revap_min')
TARGET_UNITS = {'gw_flo': 'mm', 'dep_bot': 'm', 'dep_wt': 'm',
                'no3_n': 'ppm NO3-N', 'sol_p': 'ppm P', 'carbon': '%',
                'flo_dist': 'm', 'bf_max': 'mm', 'alpha_bf': '1/day',
                'revap': '1', 'rchg_dp': '1', 'spec_yld': 'm3/m3',
                'hl_no3n': 'day', 'flo_min': 'm', 'revap_min': 'm'}
FACTORS = {('cm', 'mm'): Decimal('10'), ('km', 'm'): Decimal('1000'),
           ('fraction', '%'): Decimal('100'), ('percent', '%'): Decimal('1')}


def _name(value, field):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,15}', value):
        raise SchemaError(f'{field}须为1..16字符安全ASCII名称')
    return value


def _read_csv(content, expected, encoding):
    if encoding not in {'utf-8', 'utf-8-sig', 'gb18030'}:
        raise SchemaError('CSV编码不支持')
    try:
        reader = csv.DictReader(io.StringIO(content.decode(encoding)))
        header = reader.fieldnames
        if not header or len(header) != len(set(header)) or set(header) != set(expected):
            raise SchemaError('CSV列与映射不一致或有重复，不静默丢字段')
        records = list(reader)
    except UnicodeError as exc:
        raise SchemaError('CSV编码与配置不符') from exc
    if not records or any(None in row for row in records):
        raise SchemaError('源CSV为空或记录列数多于表头')
    return records


def export_aquifer_bundle(content, config):
    if config.get('target_model_version') != '62.0.0':
        raise SchemaError('aquifer.aqu仅核对SWAT+62.0.0读取结构')
    fields = config.get('fields')
    units = config.get('units')
    source = config.get('source')
    markers = config.get('missing_values')
    if not isinstance(fields, dict) or set(fields) != set(FIELDS) or len(set(fields.values())) != len(FIELDS):
        raise SchemaError('18个aquifer.aqu字段需逐项唯一映射；不填默认值')
    if not isinstance(units, dict) or set(units) != set(TARGET_UNITS):
        raise SchemaError('15个含水层物理字段均需明确来源单位')
    if not isinstance(markers, list):
        raise SchemaError('需明确缺测标记列表')
    if not isinstance(source, dict) or not source.get('url') or not source.get('description'):
        raise SchemaError('需提供来源URL和说明')
    table = config.get('table')
    initial = config.get('initial_aquifer_file')
    if not isinstance(table, str) or not isinstance(initial, str) or table == initial:
        raise SchemaError('需分别指定源参数CSV和现有initial.aqu')
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise SchemaError('含水层输入需ZIP：参数CSV和现有initial.aqu') from exc
    with archive:
        info = archive.infolist()
        names = [item.filename for item in info]
        if (len(info) > 100 or sum(item.file_size for item in info) > 32 * 1024**2
                or len(names) != len(set(names)) or set(names) != {table, initial}
                or any('/' in name or '\\' in name or ':' in name or '..' in name for name in names)):
            raise SchemaError('ZIP只允许安全的参数CSV与initial.aqu两个文件，最多32MB')
        records = _read_csv(archive.read(table), fields.values(), config.get('encoding', 'utf-8-sig'))
        try:
            initial_lines = archive.read(initial).decode('ascii').splitlines()
        except UnicodeError as exc:
            raise SchemaError('initial.aqu需目标版本ASCII文本') from exc
    if len(initial_lines) < 3:
        raise SchemaError('initial.aqu缺标题、表头或实际记录')
    init_names = set()
    for line in initial_lines[2:]:
        parts = line.split()
        if len(parts) < 6:
            raise SchemaError('initial.aqu记录字段不足')
        name = _name(parts[0], 'initial.aqu.name')
        if name in init_names:
            raise SchemaError('initial.aqu名称重复')
        init_names.add(name)
    converted = []
    markers = {str(x).strip() for x in markers}
    names_seen = set()
    ids_seen = set()
    for index, record in enumerate(records, start=1):
        values = {}
        for field in FIELDS:
            raw = record[fields[field]]
            if raw is None or not str(raw).strip() or str(raw).strip() in markers:
                raise SchemaError(f'第{index}条{field}缺测；不填默认值')
            raw = raw.strip()
            if field == 'id':
                if not raw.isdigit() or int(raw) < 1:
                    raise SchemaError('含水层id需正整数')
                values[field] = int(raw)
            elif field in {'name', 'init'}:
                values[field] = _name(raw, field)
            else:
                unit = units[field]
                target = TARGET_UNITS[field]
                factor = Decimal(1) if unit == target else FACTORS.get((unit, target))
                if factor is None:
                    raise SchemaError(f'{field}不支持{unit}到{target}；单位不明不可猜测')
                try:
                    value = Decimal(raw) * factor
                except InvalidOperation as exc:
                    raise SchemaError(f'{field}不是有效数值') from exc
                if not value.is_finite() or value < 0:
                    raise SchemaError(f'{field}需非负有限数值，不截断')
                if field in {'revap', 'rchg_dp', 'spec_yld'} and value > 1:
                    raise SchemaError(f'{field}须位于0..1')
                values[field] = value
        if values['name'] in names_seen or values['id'] in ids_seen:
            raise SchemaError('含水层名称或id重复')
        if values['init'] not in init_names:
            raise SchemaError(f"init引用{values['init']}不在所给initial.aqu中")
        names_seen.add(values['name']); ids_seen.add(values['id'])
        converted.append(values)
    if ids_seen != set(range(1, len(converted) + 1)):
        raise SchemaError('含水层id须连续1..N；不重新编号已有工程')
    lines = ['aquifer.aqu EcoDataEngine SWAT+62 parameter database', ' '.join(FIELDS)]
    for row in sorted(converted, key=lambda item: item['id']):
        lines.append(' '.join(str(row[field]) for field in FIELDS))
    scope = ('仅生成aquifer.aqu含水层参数数据库，检查init名称在所给initial.aqu中存在。'
             '不生成aquifer.con、流域/HRU/河道连接、初始化学数据或file.cio。'
             '所给initial.aqu中更深层的外部引用未验证；本包不包含完整工程。')
    return {'aquifer.aqu': ('\n'.join(lines) + '\n').encode('ascii'),
            'README_aquifer.txt': scope.encode('utf-8')}, {
                'scope': scope, 'target': 'SWAT+62.0.0 aqu_read/aquifer_database',
                'aquifer_records': len(converted), 'initial_name_refs': sorted(init_names),
                'target_units': TARGET_UNITS, 'source': source, 'missing_policy': 'reject',
                'complete_model_input': False, 'model_run_verified': False}
