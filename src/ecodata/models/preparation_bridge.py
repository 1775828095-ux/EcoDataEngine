"""Shared converter-to-fusion requirement resolution. No numerical algorithms."""
from copy import deepcopy
import hashlib
import json


def resolve_target(model_id, preparation=None):
    from ecodata.models.converter_contract import load_converter, converter_requirements
    converter = load_converter(model_id)
    if converter is None:
        raise ValueError('目标模型未安装转换器：' + model_id)
    provider = getattr(converter, 'requirements_for_manifest', None)
    manifest = dict(model_id=model_id, preparation=deepcopy(preparation or {}))
    contract = provider(manifest) if provider else None
    if contract is None:
        contract = converter_requirements(model_id, converter)
    if (not contract or contract.get('model_id') != model_id or
            contract.get('schema') != 'ecodata.converter.requirements.v1'):
        raise ValueError('转换器未提供有效的目标需求：' + model_id)
    contract = deepcopy(contract)
    fields = contract.get('fields', [])
    if (not contract.get('role') or not fields or
            len({f['name'] for f in fields}) != len(fields) or
            any(not f.get('unit') or f['unit'].casefold() in {'unknown',''} for f in fields)):
        raise ValueError('转换器的用途、字段或单位不完整，不能自动推定')
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()
    return dict(model_id=model_id, version='installed_converter_v1',
        roles=[dict(role=contract['role'], required=True, record_only=False, fields=fields)],
        converter_contract=contract, profile_sha256=fingerprint,
        sources=['installed ecodata.model_converters entry point'],
        status='CONVERTER_DECLARED_REQUIREMENTS', native_acceptance='NOT_ASSESSED',
        scope=contract.get('scope','field-level conversion requirements'),
        sensitivity=None, sensitivity_status='UNKNOWN')


def build_request(model_id, role, field, source_count, *, preparation=None):
    """Resolve from installed code, never trust a contract supplied in user data."""
    table = resolve_target(model_id, preparation)
    if table['roles'][0]['role'] != role:
        raise ValueError('识别用途与转换目标不一致：' + role)
    matches = [f for f in table['roles'][0]['fields'] if f['name'] == field]
    if len(matches) != 1:
        raise ValueError('转换器未声明该处理字段：' + field)
    spec = matches[0]
    request = dict(task='multisource' if source_count > 1 else 'spatial',
        variable=field, unit=spec['unit'], semantic_type=spec.get('data_type','continuous'),
        model_profile=dict(id=model_id, role=role, field=field),
        converter_context=deepcopy(preparation or {}),
        soft_guidance=False, mutation_control='random', allow_training=True)
    if source_count > 1:
        request['processing_goal'] = 'spatial_fusion'
    # Promote only exact machine-readable target axes, never a guessed grid.
    import numpy as np
    coordinates=table['converter_contract'].get('coordinates',{})
    axes={}
    for key in ('lat','lon'):
        rule=coordinates.get(key,{})
        if 'values' in rule:
            axes[key]=list(rule['values'])
        elif all(k in rule for k in ('start','step','size')):
            axes[key]=(rule['start']+np.arange(rule['size'])*rule['step']).tolist()
    if set(axes)=={'lat','lon'}:
        request.update(target_lat=axes['lat'],target_lon=axes['lon'])
    clock=coordinates.get('time',{})
    if all(k in clock for k in ('start','end','interval_seconds')):
        import pandas as pd
        request['target_times']=[str(v) for v in pd.date_range(clock['start'],clock['end'],
            freq=pd.Timedelta(seconds=clock['interval_seconds']))]
    return request, table


def source_report(sources, field, table):
    """Record all source facts and known capability gaps without inventing fixes.

    These diagnostics do not replace the numerical engine's quality gates.
    """
    import numpy as np
    rows=[]; issues=[]
    for name, ds in sources:
        if field not in ds:
            issues.append(dict(source=name,code='MISSING_VARIABLE',message='缺少字段：'+field))
            continue
        a=ds[field]
        row=dict(source=name,field=field,unit=a.attrs.get('units'),dimensions=dict(a.sizes),
            crs=ds.attrs.get('crs'),temporal_semantics=a.attrs.get('temporal_semantics'),
            spatial_semantics=a.attrs.get('spatial_semantics'),
            finite_values=int(np.isfinite(a.values).sum()),missing_values=int(a.size-np.isfinite(a.values).sum()),
            coordinates={k:dict(dimensions=list(v.dims),size=int(v.size)) for k,v in ds.coords.items()})
        if 'time' in ds and ds.time.size:
            row['time_range']=[str(ds.time.values.min()),str(ds.time.values.max())]
        if not row['crs']:
            projection=ds.attrs.get('native_projection')
            issues.append(dict(source=name,code='NATIVE_CRS_ADAPTER' if projection else 'MISSING_CRS',
                evidence=projection,message='资料带有原生投影参数，但平台尚未转换为融合引擎支持的坐标定义' if projection else '缺少可验证的坐标系定义'))
        if not a.attrs.get('physical_definition'):
            issues.append(dict(source=name,code='PHYSICAL_DEFINITION',message='缺少融合引擎要求的物理量定义证据，不能仅凭字段名认定不同产品含义一致'))
        if {'lat','lon'} <= set(ds.coords) and (ds.lat.ndim>1 or ds.lon.ndim>1):
            issues.append(dict(source=name,code='CURVILINEAR_GRID',message='二维网格需经过公共坐标适配；须先有可验证的坐标定义'))
        if row['temporal_semantics'] not in {'instantaneous','accumulation'}:
            issues.append(dict(source=name,code='TIME_SUPPORT',message='现有空间融合入口尚不支持或未确定该时间含义：'+str(row['temporal_semantics'])))
        rows.append(row)
    return dict(target=table, sources=rows, issues=issues,
        decision_owner='existing public fusion engine',
        target_grid_policy='Only converter-declared requirements; no invented engineering grid',
        algorithm_prescribed=False, original_data_modified=False)


def inventory_report(model_id, groups):
    """Find missing target fields/dependencies together before numerical work."""
    from pathlib import Path
    table=resolve_target(model_id)
    contract=table['converter_contract']
    missing=[dict(code='MISSING_REQUIRED_FIELD',message='缺少必要字段：'+f['name'])
        for f in contract['fields'] if f.get('required',True) and
        (contract['role'],f['name']) not in groups]
    for dep in contract.get('dependencies',[]):
        path=Path(dep['path'])
        if not path.exists():
            missing.append(dict(code='MISSING_DEPENDENCY',message='缺少工程依赖：'+str(path)))
    # Scientific declarations cannot be supplied by a numeric fusion algorithm.
    # Report their absence before spending time on alignment/search.
    for (_,field),sources in groups.items():
        for name,ds,_ in sources:
            for key,expected in contract.get('metadata',{}).items():
                actual=ds.attrs.get(key)
                if isinstance(expected,dict) and isinstance(actual,str):
                    try:actual=json.loads(actual)
                    except ValueError:pass
                if actual!=expected:
                    missing.append(dict(code='MISSING_OR_CONFLICTING_METADATA',source=name,field=field,
                        message=name+'：需要明确的 '+key+' = '+str(expected)+'；不能由融合算法猜测'))
    return dict(target=table,missing=missing,fields=[
        source_report([(name,ds) for name,ds,_ in sources],field,table)
        for (_,field),sources in groups.items()])
