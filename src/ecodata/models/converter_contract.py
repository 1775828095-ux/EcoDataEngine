"""Field-level handoff metadata and fail-closed converter preflight.

No fusion algorithms, engineering settings, or original source files are edited.
The entry point remains convert(manifest_path, output_dir). Optional requirements
are supplied by the installed converter, rather than guessed from model names.
"""
from pathlib import Path
from importlib.metadata import entry_points
import hashlib
import io
import json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_converter(model_id, finder=None):
    entries = list((finder or entry_points)(group='ecodata.model_converters', name=model_id))
    if len(entries) > 1:
        raise ValueError('同一模型注册了多个转换器：' + model_id)
    if entries:
        return entries[0].load()
    # Source-tree desktop launches include the bundled package on sys.path
    # without its installed entry-point metadata.
    if finder is None:
        return bundled_converter(model_id)
    return None


def bundled_converter(model_id):
    if model_id=="swat_plus":
        try:
            from local_swat_converter import convert
        except ModuleNotFoundError as exc:
            if exc.name != "local_swat_converter":
                raise
        else:
            return convert
    if model_id in {"wrf","camx","cmaq","calpuff"}:
        try:
            import existing_air_converters
        except ModuleNotFoundError as exc:
            if exc.name != "existing_air_converters":
                raise
        else:
            return getattr(existing_air_converters,model_id)
    return None


def converter_requirements(model_id, converter=None):
    converter = converter or load_converter(model_id)
    provider = getattr(converter, 'requirements', None)
    if not provider:
        return None
    contract = provider()
    if contract.get('model_id') != model_id or contract.get('schema') != 'ecodata.converter.requirements.v1':
        raise ValueError('转换器返回的需求契约与所选模型不符')
    return contract


def preparation_table(model_id, fallback):
    contract = converter_requirements(model_id)
    if not contract:
        return fallback
    if contract.get('component_dispatch'):
        return fallback
    return dict(id=model_id, roles=[dict(role=contract['role'], record_only=False,
                fields=contract['fields'])], converter_contract=contract)


def coordinate_references(dataset, contract):
    """Keep declared identifiers/axes as coordinates, with values untouched."""
    if not contract:
        return dataset
    renames = {}
    for source, target in contract.get('coordinate_aliases', {}).items():
        if source in dataset:
            if target in dataset:
                raise ValueError('同一坐标存在多个候选，不能自动选择：'+source+' / '+target)
            renames[source] = target
    if renames:
        dataset = dataset.rename(renames)
    references = [name for name in contract.get('reference_fields', []) if name in dataset]
    return dataset.set_coords(references)


def describe_product(folder, product, task):
    import numpy as np
    import pandas as pd
    import xarray as xr
    root = Path(folder).resolve(); path = Path(product).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('处理产品不存在或超出本次任务目录')
    field = task.get('field')
    with xr.open_dataset(io.BytesIO(path.read_bytes())) as ds:
        if field not in ds.data_vars:
            raise ValueError('本次处理字段不在产品中：' + str(field))
        axes = {}
        for name, axis in ds.coords.items():
            values = np.asarray(axis.values)
            description = dict(dimensions=list(axis.dims), size=int(axis.size),
                               dtype=str(axis.dtype), unit=axis.attrs.get('units'),
                               sha256=hashlib.sha256(json.dumps(values.tolist(), default=str,
                                   ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            if values.size:
                if values.dtype.kind == 'M':
                    clock = pd.DatetimeIndex(values.reshape(-1))
                    description.update(start=str(clock.min()), end=str(clock.max()),
                        monotonic=bool(clock.is_monotonic_increasing), unique=bool(clock.is_unique))
                    delta = np.diff(clock.asi8)
                    description['interval_seconds'] = float(delta[0]/1e9) if len(delta) and np.all(delta == delta[0]) else None
                elif values.dtype.kind in 'iuf':
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        description.update(min=float(finite.min()), max=float(finite.max()))
            axes[name] = description
        metadata = {key: value for key, value in ds.attrs.items()
                    if key in {'crs', 'source_crs', 'time_basis', 'source_timezone', 'wind_reference',
                               'grid', 'elevation_vertical_reference', 'model_role'}}
        value = ds[field]
        return dict(path=path.relative_to(root).as_posix(), sha256=digest(path),
                    role=task.get('role'), field=field, unit=value.attrs.get('units'),
                    dimensions=dict(value.sizes), axes=axes, metadata=metadata,
                    finite=bool(np.isfinite(value.values).all()),
                    preparation_result=task.get('result', {}).get('status'))


def preflight(manifest_path, contract):
    """Accumulate actionable omissions/conflicts. Native writers check further."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from rasterio.crs import CRS
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf8'))
    checks = []; loaded = {}
    def check(key, passed, reason, **extra):
        checks.append(dict(key=key, status='PASS' if passed else 'BLOCK', message=reason, **extra))
    check('protocol', manifest.get('protocol') == 'ecodata.converter.v1' and
          manifest.get('model_id') == contract['model_id'], '交接协议与目标模型必须一致')
    for item in contract.get('dependencies', []):
        dependency = Path(item['path'])
        exists = dependency.is_file() if item.get('sha256') else dependency.exists()
        check('dependency.'+item['name'], exists, '需要本机依赖：'+str(dependency))
        if exists and item.get('sha256'):
            check('dependency_hash.'+item['name'], digest(dependency) == item['sha256'], '本机依赖必须与绑定校验值一致：'+item['name'])
    for index, item in enumerate(manifest.get('products', [])):
        try:
            relative = Path(item['path']); product = (path.parent/relative).resolve()
            if relative.is_absolute() or not product.is_relative_to(path.parent) or not product.is_file():
                raise ValueError('产品路径缺失或越界')
            if digest(product) != item['sha256']:
                raise ValueError('产品 SHA256 不匹配')
            with xr.open_dataset(io.BytesIO(product.read_bytes())) as source:
                ds = source.load()
            field = item.get('field')
            if field not in ds.data_vars:
                raise ValueError('清单必须指明本次已处理字段')
            if field in loaded:
                raise ValueError('处理字段重复，不能选择性覆盖')
            if item.get('role') != contract['role']:
                raise ValueError('产品用途与绑定转换场景不一致')
            loaded[field] = (ds, item)
            check(f'product.{index}', True, '本次处理产品路径与哈希已核对', path=item['path'], sha256=item['sha256'])
        except Exception as exc:
            check(f'product.{index}', False, str(exc))
    required = {field['name']: field for field in contract['fields']}
    for extra in set(loaded)-set(required):
        check('unsupported.'+extra, False, '转换器不接收该处理字段：'+extra)
    coordinate_signatures = {}
    for name, requirement in required.items():
        check('field.'+name, name in loaded, '需要处理字段：'+name, expected_unit=requirement['unit'])
        if name not in loaded:
            continue
        ds, item = loaded[name]; value = ds[name]
        check('unit.'+name, value.attrs.get('units') == requirement['unit'], '处理字段单位必须与转换契约一致：'+name)
        native_mask=contract.get('native_missing_policies',{}).get(name)
        if native_mask:
            ok=(value.dtype.kind in 'iuf' and np.isfinite(value.values).any()
                and not np.isinf(value.values).any()
                and value.attrs.get('missing_policy')==native_mask)
            check('finite.'+name,ok,'必须保留契约明确允许的原始缺测掩码，不得补造：'+name)
        elif name in contract.get('allow_missing_fields', []):
            ok = (value.dtype.kind in 'iuf' and np.isfinite(value.values).any()
                  and not np.isinf(value.values).any()
                  and value.attrs.get('missing_policy')=='retain source mask; WPS encodes missing as -1e30')
            check('finite.'+name, ok, '允许原始SST掩码但必须明确原生缺失编码、保留有效值且无无穷值：'+name)
        else:
            check('finite.'+name, value.dtype.kind in 'iuf' and np.isfinite(value.values).all(), '转换前不允许剩余缺失或非有限值：'+name)
        crs = ds.attrs.get('crs')
        try:
            crs_ok = bool(crs) and (not contract.get('crs') or CRS.from_user_input(crs) == CRS.from_user_input(contract['crs']))
        except Exception:
            crs_ok = False
        if contract.get('spatial_reference_mode')=='native_station_identifiers':
            reference=contract.get('station_binding_sha256')
            crs_ok=(isinstance(reference,str) and len(reference)==64 and
                    ds.attrs.get('station_binding_sha256')==reference and
                    'station_id' in ds.coords)
            check('station_binding.'+name,crs_ok,'原生无坐标站表必须与本次控制文件站点绑定一致')
        else:
            check('crs.'+name, crs_ok, '需要明确且与绑定工程一致的坐标系，不能从文件名推测')
        for key, expected in contract.get('metadata', {}).items():
            actual = ds.attrs.get(key)
            if isinstance(expected, dict) and isinstance(actual, str):
                try:
                    actual = json.loads(actual)
                except ValueError:
                    pass
            check('metadata.'+name+'.'+key, actual == expected,
                  '需要明确的 '+key+' = '+str(expected), actual=actual)
        for axis_name, rule in contract.get('coordinates', {}).items():
            axis = ds.coords.get(axis_name)
            check('axis.'+name+'.'+axis_name, axis is not None, '需要明确坐标：'+axis_name)
            if axis is None:
                continue
            data = np.asarray(axis.values)
            if 'unit' in rule:
                check('axis_unit.'+name+'.'+axis_name, axis.attrs.get('units') == rule['unit'], '坐标单位需要 '+rule['unit']+'：'+axis_name)
            if rule.get('dimension'):
                check('axis_dimension.'+name+'.'+axis_name, axis_name in value.dims, '处理变量必须包含维度：'+axis_name)
            if 'size' in rule:
                check('axis_size.'+name+'.'+axis_name, axis.size == rule['size'], '坐标长度必须与绑定工程一致：'+axis_name)
            if 'value' in rule:
                check('axis_value.'+name+'.'+axis_name, axis.size == 1 and np.allclose(data, rule['value'], atol=1e-6, rtol=0), '站点位置必须与绑定工程一致：'+axis_name)
            if 'start' in rule and 'step' in rule:
                expected = rule['start']+np.arange(rule['size'])*rule['step']
                check('axis_grid.'+name+'.'+axis_name, data.shape == expected.shape and bool(np.allclose(data, expected, atol=1e-6, rtol=0)), '坐标应与绑定网格逐点一致：'+axis_name)
            if data.dtype.kind in 'iuf':
                check('axis_finite.'+name+'.'+axis_name, np.isfinite(data).all(), '坐标不能包含缺失值：'+axis_name)
            if rule.get('datetime'):
                if data.dtype.kind != 'M':
                    check('axis_time.'+name, False, '时间必须是已解码日期，不能把任意数字解释为日期')
                else:
                    clock = pd.DatetimeIndex(data.reshape(-1))
                    good = not clock.hasnans and clock.is_unique and clock.is_monotonic_increasing
                    check('axis_time.'+name, good, '时间应有效、唯一且递增')
                    delta = np.diff(clock.asi8)/1e9
                    if rule.get('interval_seconds') is not None:
                        check('time_interval.'+name, bool(len(delta)) and bool(np.all(delta == rule['interval_seconds'])), '时间间隔必须满足转换配置')
                    if rule.get('start') and rule.get('end'):
                        expected = pd.date_range(rule['start'], rule['end'], freq=pd.Timedelta(seconds=rule['interval_seconds']))
                        check('time_coverage.'+name, clock.equals(expected), '时间必须连续覆盖绑定工程的完整要求')
            if rule.get('aligned', True):
                signature = json.dumps(dict(dims=list(axis.dims), values=data.tolist(), units=axis.attrs.get('units')), default=str, sort_keys=True)
                if axis_name in coordinate_signatures:
                    check('alignment.'+name+'.'+axis_name, coordinate_signatures[axis_name] == signature, '不同产品的同名坐标必须一致：'+axis_name)
                else:
                    coordinate_signatures[axis_name] = signature
        for reference in contract.get('required_references', []):
            check('reference.'+name+'.'+reference, reference in ds.coords, '需要保留不参与数值融合的标识或坐标：'+reference)
    blocked = [item for item in checks if item['status'] == 'BLOCK']
    return dict(schema='ecodata.converter.preflight.v1', model_id=contract['model_id'],
                passed=not blocked, checks=checks, missing_or_conflicting=blocked,
                scope=contract.get('scope'), native_model_executed=False,
                deferred_checks=contract.get('deferred_checks', []))
