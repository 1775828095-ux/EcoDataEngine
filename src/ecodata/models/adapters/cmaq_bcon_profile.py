"""CMAQ 5.5 BCON/profile static ASCII profile (BC_PROFILE), not BCON output."""
import csv
from decimal import Decimal, InvalidOperation
import io
import re

from ecodata.core.exceptions import SchemaError

VERSION = 'CMAQ-5.5-BCON-profile'
FACTORS = {('ppbV', 'ppmV'): Decimal('0.001'),
           ('hPa', 'Pa'): Decimal('100'), ('km', 'm'): Decimal('1000')}
UNITS = {'ppmV', 'ug m-3', 'm-3', 'm2 m-3', 'Pa', 'm'}


def _number(value, label, markers):
    if value is None or str(value).strip() in markers or not str(value).strip():
        raise SchemaError(f'{label}缺测；不填补或补零')
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise SchemaError(f'{label}不是有效数值') from exc
    if not number.is_finite():
        raise SchemaError(f'{label}不是有限值')
    return number


def export_bcon_profile(content, config):
    if config.get('version') != VERSION:
        raise SchemaError('BCON剖面仅支持CMAQ 5.5 BCON/profile；version需明确')
    mechanism = config.get('mechanism')
    if not isinstance(mechanism, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,64}', mechanism):
        raise SchemaError('需明确CMAQ化学机制，物种匹配由后续BCON配置核对')
    units = config.get('target_units')
    if (not isinstance(units, dict) or not 2 <= len(units) <= 251 or units.get('ZH') != 'm'
            or any(not re.fullmatch(r'[A-Z][A-Z0-9_]{0,15}', name) or unit not in UNITS
                   for name, unit in units.items())
            or any(units.get(name) != unit for name, unit in {'PRES': 'Pa', 'ZF': 'm'}.items() if name in units)
            or not set(units) - {'ZH', 'PRES', 'ZF', 'WVEL'}):
        raise SchemaError('target_units须含ZH/m及至少一个化学物种；PRES/Pa和ZF/m可选')
    layers = config.get('layers')
    if type(layers) is not int or not 1 <= layers <= 100:
        raise SchemaError('layers须为1..100；层数来自实测或上游剖面，不插值')
    fields = config.get('fields')
    layout = config.get('layout')
    if not isinstance(fields, dict) or layout not in {'long', 'wide'}:
        raise SchemaError('需明确long或wide布局及字段映射')
    source = config.get('source')
    if not isinstance(source, dict) or not source.get('url') or not source.get('description'):
        raise SchemaError('需给出来源URL和说明')
    markers = config.get('missing_values')
    if not isinstance(markers, list):
        raise SchemaError('missing_values须明确列出，包括数据源缺测标记')
    markers = {str(x).strip() for x in markers}
    encoding = config.get('encoding', 'utf-8-sig')
    if encoding not in {'utf-8', 'utf-8-sig', 'gb18030'}:
        raise SchemaError('CSV编码仅支持UTF-8、UTF-8 BOM或GB18030')
    try:
        text = content.decode(encoding)
        reader = csv.DictReader(io.StringIO(text))
        header = reader.fieldnames
        if not header or len(header) != len(set(header)):
            raise SchemaError('CSV表头为空或有重复列')
        records = list(reader)
    except UnicodeError as exc:
        raise SchemaError('CSV编码与配置不符') from exc
    if not records or any(None in row for row in records):
        raise SchemaError('CSV为空或记录列数多于表头')
    expected = {'variable', 'unit', 'layer', 'value'} if layout == 'long' else {'variable', 'unit', 'layers'}
    if set(fields) != expected:
        raise SchemaError(f'{layout}布局fields须含{sorted(expected)}')
    if layout == 'long':
        columns = [fields[k] for k in ('variable', 'unit', 'layer', 'value')]
    else:
        if not isinstance(fields['layers'], list) or len(fields['layers']) != layers:
            raise SchemaError('wide布局须逐层给出layers列名')
        columns = [fields['variable'], fields['unit'], *fields['layers']]
    if len(columns) != len(set(columns)) or set(header) != set(columns):
        raise SchemaError('CSV列与映射不一致；不得静默丢字段')
    values = {}
    declared_units = {}
    for row in records:
        name = row[fields['variable']]
        if name not in units:
            raise SchemaError(f'源变量{name}未在target_units声明')
        actual = row[fields['unit']]
        if not actual or actual.strip() in markers:
            raise SchemaError(f'{name}来源单位缺失')
        if name in declared_units and declared_units[name] != actual:
            raise SchemaError(f'{name}混合来源单位')
        declared_units[name] = actual
        factor = Decimal(1) if actual == units[name] else FACTORS.get((actual, units[name]))
        if factor is None:
            raise SchemaError(f'{name}不支持{actual}到{units[name]}；不猜测物种或摩尔/质量转换')
        if layout == 'long':
            try:
                layer = int(row[fields['layer']])
            except (ValueError, TypeError) as exc:
                raise SchemaError(f'{name}层号无效') from exc
            if not 1 <= layer <= layers or str(layer) != str(row[fields['layer']]).strip():
                raise SchemaError(f'{name}层号超范围或非整数')
            pairs = [(layer, row[fields['value']])]
        else:
            pairs = enumerate((row[column] for column in fields['layers']), start=1)
        for layer, raw in pairs:
            key = (name, layer)
            if key in values:
                raise SchemaError(f'{name}第{layer}层重复')
            value = _number(raw, f'{name}第{layer}层', markers) * factor
            if value < 0:
                raise SchemaError(f'{name}第{layer}层负值；不截断')
            values[key] = value
    if set(declared_units) != set(units) or len(values) != len(units) * layers:
        raise SchemaError('变量或层记录不完整；不填补')
    for name in units:
        series = [values[name, layer] for layer in range(1, layers + 1)]
        if name == 'ZH' and (series[0] <= 0 or any(b <= a for a, b in zip(series, series[1:]))):
            raise SchemaError('ZH层中心高度须为正且严格递增')
        if name == 'ZF' and (any(b <= a for a, b in zip(series, series[1:]))
                             or any(zf <= values['ZH', layer] for layer, zf in enumerate(series, start=1))):
            raise SchemaError('ZF层顶高度须严格递增且高于ZH')
        if name == 'PRES' and (series[0] <= 0 or any(b >= a for a, b in zip(series, series[1:]))):
            raise SchemaError('PRES压力须为正且严格递减')
    header = 'VNAME,UNITS,' + ','.join(f'LAYER{i:02d}' for i in range(1, layers + 1))
    if len(header) > 1000:
        raise SchemaError('BCON/profile Fortran头记录超过1000字符')
    lines = ['# EcoDataEngine CMAQ 5.5 BCON/profile static input; no geographic grid or timestamps', header]
    for name, unit in units.items():
        line = f'"{name}","{unit}",' + ','.join(str(values[name, i]) for i in range(1, layers + 1))
        lines.append(line)
    payload = ('\n'.join(lines) + '\n').encode('ascii')
    report = {'scope': 'CMAQ 5.5 BCON/profile BC_PROFILE ASCII input; not BCON boundary NetCDF',
              'version': VERSION, 'mechanism': mechanism, 'layers': layers,
              'variables': len(units), 'values': len(values), 'source_units': declared_units,
              'target_units': units, 'source': source, 'missing_policy': 'reject',
              'time': 'static vertical profile; BCON controls output time separately',
              'coordinates': 'ZH metres above ground; BCON obtains target boundary grid from MET_BDY_3D_FIN',
              'BCON_run_verified': False, 'complete_model_input': False}
    return {'BC_PROFILE_CMAQ55.csv': payload}, report
