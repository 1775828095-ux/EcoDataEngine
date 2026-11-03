"""Mapped layered soil tables to SWAT+ 62.0.0 soil_db_read, no parameter estimation."""
import csv
import io
import re
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError

META = ('soil', 'layer', 'hyd_grp', 'zmx', 'anion_excl', 'crk', 'texture')
LAYERS = ('depth', 'bd', 'awc', 'k', 'cbn', 'clay', 'silt', 'sand', 'rock', 'alb', 'usle_k', 'ec', 'cal', 'ph')
UNITS = dict(zip(('zmx', 'anion_excl', 'crk') + LAYERS,
                 ('mm', '1', '1', 'mm', 'Mg/m3', '1', 'mm/h', '%', '%', '%', '%', '%', '1', 'SWAT_USLE_K', 'dS/m', '%', 'pH')))


def _factor(name, unit):
    target = UNITS[name]
    if unit == target:
        return 1., 0.
    if target in {'pH', 'SWAT_USLE_K'}:
        raise SchemaError(f'{name}需明确单位{target}；不猜测USLE不同单位制或pH换算')
    # Explicit fractions avoid Pint's dimensionless aliases hiding unknown labels.
    if target in {'1', '%'}:
        if unit not in {'1', '%', 'mm/mm'}:
            raise SchemaError(f'{name}单位{unit!r}不支持，目标{target}')
        return ((.01 if unit == '%' else 1.) * (100. if target == '%' else 1.)), 0.
    from ecodata.engines.units import UnitEngine
    aliases = {'Mg/m3': 'megagram/meter**3', 'kg/m3': 'kilogram/meter**3',
               'g/cm3': 'gram/centimeter**3', 'dS/m': 'decisiemens/meter',
               'mS/cm': 'millisiemens/centimeter'}
    try:
        return UnitEngine().affine(aliases.get(unit, unit), aliases.get(target, target))
    except Exception as exc:
        raise SchemaError(f'{name}单位{unit!r}不能转换为{target}：{exc}') from exc


def export_soils(content, config):
    if config.get('target_model_version') != '62.0.0':
        raise SchemaError('土壤转换仅核对SWAT+ 62.0.0；请设置target_model_version=62.0.0')
    if config.get('depth_basis') != 'bottom':
        raise SchemaError('depth_basis必须为bottom：各层底部累计深度，不是层厚；不得猜测')
    mapping = config.get('fields', {})
    required = META + LAYERS
    if not isinstance(mapping, dict) or set(required) - set(mapping):
        raise SchemaError(f'缺少字段映射：{sorted(set(required)-set(mapping)) if isinstance(mapping,dict) else required}')
    if any(not isinstance(mapping[k], str) for k in required) or len({mapping[k] for k in required}) != len(required):
        raise SchemaError('每个必需字段需独立且有效的源列映射')
    units = config.get('units', {})
    if not isinstance(units, dict) or set(UNITS) - set(units):
        raise SchemaError('所有数值字段必须明确units，不能使用默认单位')
    source = config.get('source', {})
    if not isinstance(source, dict) or source.get('kind') not in {'measured', 'derived_parameters', 'format_reference', 'synthetic_test'} or not isinstance(source.get('description'), str) or not source.get('description'):
        raise SchemaError('必须提供source.kind和source.description，区分实测/格式参考/合成数据')
    fmt = config.get('source_format', 'csv')
    try:
        if fmt == 'csv':
            decoded = content.decode(config.get('encoding', 'utf-8-sig'))
            header = next(csv.reader(io.StringIO(decoded)))
            if len(header) != len(set(header)):
                raise SchemaError('CSV表头存在重复字段')
            frame = pd.read_csv(io.StringIO(decoded), dtype=str, keep_default_na=False)
        elif fmt in {'xlsx', 'parquet'}:
            from ecodata.io.structures import read_structure, table_frame
            frame = table_frame(read_structure('source.' + fmt, content))
        else:
            raise SchemaError('source_format只支持csv、xlsx、parquet')
    except (UnicodeError, ValueError, StopIteration) as exc:
        raise SchemaError(f'土壤表读取失败：{exc}') from exc
    missing = set(mapping[k] for k in required) - set(frame.columns)
    if missing:
        raise SchemaError(f'源数据缺列：{sorted(missing)}')
    if frame.empty:
        raise SchemaError('土壤表没有数据记录')
    data = frame[[mapping[k] for k in required]].copy()
    data.columns = required
    markers = config.get('missing_values')
    if not isinstance(markers, list):
        raise SchemaError('必须明确missing_values列表；即使无缺测标记也需提供[]')
    for name in required:
        col = data[name]
        if col.isna().any() or col.astype(str).str.strip().eq('').any() or col.isin(markers).any() or col.astype(str).isin([str(m) for m in markers]).any():
            raise SchemaError(f'{name}含缺失值或源缺测标记；不填零')
    for name in ('layer',) + tuple(UNITS):
        if data[name].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise SchemaError(f'{name}不能为布尔值')
        values = pd.to_numeric(data[name], errors='coerce').to_numpy(dtype=float)
        for marker in markers:
            try:
                if (values == float(marker)).any():
                    raise SchemaError(f'{name}含源缺测标记{marker!r}；不填零')
            except (TypeError, ValueError):
                pass
        if not np.isfinite(values).all():
            raise SchemaError(f'{name}包含无效数值/NaN/无穷值')
        scale, offset = (1., 0.) if name == 'layer' else _factor(name, units[name])
        data[name] = values * scale + offset
        if not np.isfinite(data[name]).all():
            raise SchemaError(f'{name}单位换算后溢出')
    lines = ['soils.sol EcoDataEngine SWAT+ 62.0.0',
             'name nly hyd_grp zmx anion_excl crk texture | depth bd awc k cbn clay silt sand rock alb usle_k ec cal ph']
    warnings = []
    counts = {}
    for soil, group in data.groupby('soil', sort=True):
        if not isinstance(soil, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,19}', soil) or soil.lower() in {'null', 'sim'}:
            raise SchemaError(f'土壤名称{soil!r}需1–20字符ASCII字母数字及_.-，不能为null/sim')
        group = group.sort_values('layer')
        if list(group.layer) != list(range(1, len(group) + 1)):
            raise SchemaError(f'{soil}层号必须唯一连续1..N')
        for field in META[2:]:
            if group[field].nunique() != 1:
                raise SchemaError(f'{soil}各层的{field}站点/土壤级信息不一致')
        first = group.iloc[0]
        if first.hyd_grp not in {'A', 'B', 'C', 'D'}:
            raise SchemaError(f'{soil}水文组需A/B/C/D；双组土壤需项目明确排水条件')
        texture = str(first.texture)
        if not re.fullmatch(r'[A-Za-z0-9 _.+-]{1,16}', texture):
            raise SchemaError(f'{soil}texture需1–16字符安全ASCII文本')
        if first.zmx <= 0 or not 0 <= first.anion_excl <= 1 or not 0 <= first.crk <= 1:
            raise SchemaError(f'{soil}zmx须正值；anion_excl和crk须0..1')
        depths = group.depth.to_numpy()
        if depths[0] <= 0 or (np.diff(depths) <= 0).any():
            raise SchemaError(f'{soil}层底累计深度必须为正且递增')
        for field in LAYERS:
            v = group[field]
            if (v < 0).any() or (field in {'bd','depth'} and (v <= 0).any()):
                raise SchemaError(f'{soil}/{field}数值超出允许范围')
            upper = 100 if field in {'cbn','clay','silt','sand','rock','cal'} else (1 if field in {'awc','alb'} else (14 if field == 'ph' else None))
            if upper is not None and (v > upper).any():
                raise SchemaError(f'{soil}/{field}超过{upper}{UNITS[field]}')
        # Provisional engineering input consistency tolerance, fixed before comparison.
        if (abs(group.clay + group.silt + group.sand - 100) > .1).any():
            raise SchemaError(f'{soil}clay+silt+sand应为100%，本次固定容差0.1百分点；不归一化')
        if depths[0] < 20:
            warnings.append(f'{soil}: 官方62读取器可能调整首层深度；本转换保留原始层底深度')
        if (group.alb < .1).any() or first.anion_excl <= 1e-6 or first.anion_excl >= 1:
            warnings.append(f'{soil}: 官方初始化可能重设alb/anion_excl；转换器未修改这些值')
        text = lambda v: repr(float(v))
        lines.append(f'{soil} {len(group)} {first.hyd_grp} {text(first.zmx)} {text(first.anion_excl)} {text(first.crk)} "{texture}"')
        for _, row in group.iterrows():
            lines.append(' '.join(text(row[k]) for k in LAYERS))
        counts[soil] = len(group)
    notes = ('仅soils.sol土壤物理参数数据库，不提供file.cio或HRU。目标工程的hru-data.hru土壤名称必须对应本文件。'
             '没有时间轴或地理网格；位置与HRU关系由项目提供。未推算水力参数、未填缺失值。'
             'zmx是最大根系深度，depth是各层底累计深度，两者不能混用。'
             'USLE_K须已采用目标SWAT定义，不能将任意SI/US制数值只改标签。')
    return {'soils.sol': ('\n'.join(lines)+'\n').encode('ascii'), 'README_soils_only.txt': notes.encode('utf-8')}, {
        'target': 'SWAT+ 62.0.0 soil_db_read', 'soils': counts, 'layer_records': len(data),
        'output_units': UNITS, 'depth_basis': 'bottom', 'missing_policy': 'reject',
        'texture_sum_tolerance_percentage_points': .1, 'warnings': warnings,
        'scope': notes, 'unmapped_source_columns': [str(c) for c in frame.columns if c not in mapping.values()],
        'hru_reference_check': 'not_supplied; must be checked in target project'}
