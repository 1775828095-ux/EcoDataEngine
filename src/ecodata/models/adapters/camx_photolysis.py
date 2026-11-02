"""CAMx 7.40 CB6r5 clear-sky photolysis lookup table, not a radiation solver.

Interface: READPHT (1x,10f12.0), readchm.f label, camx.prm.v7.40
dimensions, Mod_src/chmstry.f zenith angles. No other mechanism is supported.
"""
from itertools import product
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError

REACTIONS = (
    'NO2.PHF', 'O3O3P.PHF', 'O3O1D.PHF', 'H2O2.PHF', 'NO3_NO2.PHF',
    'NO3_NO.PHF', 'N2O5.PHF', 'HONO.PHF', 'HNO3.PHF', 'PNA.PHF',
    'PAN.PHF', 'MEPX.PHF', 'NTR.PHF', 'FORM_Rr5.PHF', 'FORM_Mr5.PHF',
    'ALD2_Rr5.PHF', 'ALDX_Rr5.PHF', 'GLYDr5.PHF', 'GLY_Rr5.PHF',
    'MGLY.PHF', 'KET.PHF', 'ACET.PHF', 'ISPD.PHF',
)
ZENITH_DEGREES = [0, 10, 20, 30, 40, 50, 60, 70, 78, 86]
COLUMNS = ('block_id', 'ozone_column_native', 'albedo', 'terrain_native',
           'height_km', 'reaction', 'zenith_index', 'photolysis_native_value')
AXES = ('ozone_column_native', 'albedo', 'terrain_native', 'height_km')
VERSION = 'CAMx-7.40-CB6r5-photolysis'
TITLE = 'TUV4.8CAMx7.40_CB6r5'


def export_photolysis(frame, config):
    """Repack a complete, explicitly unit-labelled long table into native ASCII."""
    if set(config) != {'version', 'units', 'zenith_degrees'} or config.get('version') != VERSION:
        raise SchemaError('仅支持CAMx 7.40 CB6r5光解表；配置需version、units、zenith_degrees')
    if config['zenith_degrees'] != ZENITH_DEGREES:
        raise SchemaError('zenith_index须对应CAMx固定天顶角0/10/20/30/40/50/60/70/78/86度')
    units = config['units']
    expected = {'ozone_column_native': 'DU', 'albedo': '1',
                'terrain_native': 'km', 'height_km': 'km'}
    if (not isinstance(units, dict) or set(units) != set(expected) | {'photolysis_native_value'}
            or any(units.get(k) != v for k, v in expected.items())
            or units.get('photolysis_native_value') not in {'min^-1', 's^-1'}):
        raise SchemaError('需明确DU、无量纲反照率、km高度及min^-1或s^-1光解率单位')
    if set(frame.columns) != set(COLUMNS) or len(frame.columns) != len(COLUMNS):
        raise SchemaError('CSV字段必须完整且无额外字段，禁止静默丢字段')
    data = frame.copy()
    if set(data['reaction']) != set(REACTIONS):
        raise SchemaError('反应标签必须精确匹配CB6r5的23个主光解反应')
    for name in COLUMNS:
        if name == 'reaction':
            continue
        try:
            data[name] = pd.to_numeric(data[name], errors='raise')
        except (ValueError, TypeError) as exc:
            raise SchemaError(f'{name}包含非数值') from exc
        if not np.isfinite(data[name].to_numpy(dtype=float)).all():
            raise SchemaError(f'{name}存在缺失或非有限值；禁止填零')
    for name in ('block_id', 'zenith_index'):
        if (data[name] < 0).any() or (data[name] % 1 != 0).any():
            raise SchemaError(f'{name}必须是非负整数')
    if set(data.zenith_index) != set(range(10)):
        raise SchemaError('必须保留完整10个天顶角索引')
    if (data.photolysis_native_value < 0).any():
        raise SchemaError('光解率不得为负；不允许静默截断')
    if (not data.ozone_column_native.between(250, 600).all()
            or not data.albedo.between(0, 1).all()
            or (data.terrain_native < 0).any() or (data.height_km < 0).any()):
        raise SchemaError('臭氧柱须250..600 DU，反照率0..1，高度不得为负')
    axes = [sorted(data[a].unique().tolist()) for a in AXES]
    if [len(a) for a in axes] != [5, 5, 3, 11] or axes[3][0] != 0:
        raise SchemaError('需完整5臭氧×5反照率×3地形×11高度（含地表0km）')
    context = data[['block_id', *AXES]].drop_duplicates()
    if (len(context) != 825 or context.block_id.duplicated().any()
            or context[list(AXES)].duplicated().any()):
        raise SchemaError('block_id与条件/高度须一一对应，禁止丢失或混用块上下文')
    keys = [*AXES, 'reaction', 'zenith_index']
    if data.duplicated(keys).any():
        raise SchemaError('存在重复条件/高度/反应/天顶角记录')
    index = pd.MultiIndex.from_product([*axes, REACTIONS, range(10)], names=keys)
    values = data.set_index(keys).photolysis_native_value.reindex(index)
    if len(data) != len(index) or values.isna().any():
        raise SchemaError('条件笛卡尔积不完整，缺少光解率记录；不补造数值')
    factor = 60.0 if units['photolysis_native_value'] == 's^-1' else 1.0
    array = values.to_numpy(dtype=float).reshape(5, 5, 3, 11, 23, 10) * factor
    if not np.isfinite(array).all():
        raise SchemaError('单位转换溢出')
    lines = [TITLE]
    for oi, ai, ti in product(range(5), range(5), range(3)):
        condition = (f' O3 Column = {axes[0][oi]} Albedo= {axes[1][ai]}'
                     f' Terrain = {axes[2][ti]}')
        if len(condition) > 80:
            raise SchemaError('条件头超出CAMx的80字符读取范围')
        lines.append(condition)
        for hi, height in enumerate(axes[3]):
            lines.append(f' {height} km above surface')
            for ri, reaction in enumerate(REACTIONS):
                fields = [f'{v:12.6E}' for v in array[oi, ai, ti, hi, ri]]
                if any(len(s) != 12 for s in fields):
                    raise SchemaError('光解率超出12字符定宽字段，不截断')
                lines.append(' ' + ''.join(fields) + f'  CBX     : {reaction}')
    report = {
        'target': VERSION, 'output_type': 'native ASCII photolysis lookup candidate',
        'dimensions': {'ozone': 5, 'albedo': 5, 'terrain': 3, 'height': 11,
                       'reaction': 23, 'zenith': 10},
        'records': len(data), 'units': {**expected, 'photolysis_native_value': 'min^-1'},
        'rate_conversion_factor': factor, 'axes': dict(zip(AXES, axes)),
        'reaction_order': list(REACTIONS), 'zenith_degrees': ZENITH_DEGREES,
        'block_context_map': context.sort_values(list(AXES)).to_dict(orient='records'),
        'time_scope': 'static conditional lookup, no timestamps or generated time series',
        'coordinate_scope': 'height and terrain in km; no geographic grid or projection',
        'missing_policy': 'reject; no fill or clipping',
        'complete_model_input': False, 'camx_run_verified': False,
        'scope': 'Repack supplied clear-sky lookup values; does not run TUV or generate atmospheric photolysis.',
        'next_steps': 'Set Photolysis_Rates in an existing matching CB6r5 CAMx.in; preserve matching CHEMPARAM primary-reaction order. Other inputs and actual CAMx reading are still required.',
    }
    return {'photolysis.CAMx7.40.CB6r5.txt': ('\n'.join(lines) + '\n').encode('ascii')}, report
