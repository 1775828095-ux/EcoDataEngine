"""Shared scientific metadata and matching, independent of case/model names.

Use installed cf-xarray/Pint, not filename guesses or inferred numerical ranges.
Metadata matches are candidates; they do not prove independent provenance,
compatible time support, or permission to replace an engineering input.
"""
import re


def normal(value):
    return re.sub(r'[^a-z0-9]', '', str(value).casefold())


# Scientific quantities, shared across ALL model adapters. This is not a list
# of datasets, products, resolutions or accepted filenames. Explicit CF names
# remain usable without registering additional quantities here.
QUANTITIES = {
    'leaf_area_index': ('LAI', 'leaf area index'),
    'sea_surface_temperature': ('sst', 'sea surface temperature'),
    'air_temperature': ('air temperature', 'temperature at 2 m', '2 m air temperature'),
    'eastward_wind': ('eastward wind',),
    'northward_wind': ('northward wind',),
    'geopotential_height': ('geopotential height',),
}
ALIASES = {normal(v): key for key, values in QUANTITIES.items()
           for v in (key, *values)}


def identity(name, attrs):
    # A valid explicit declaration outranks descriptive strings. Do not turn
    # e.g. "standard deviation for temperature" into temperature itself.
    for key in ('standard_name', 'physical_variable'):
        value = str(attrs.get(key, '')).strip()
        if value:
            return ALIASES.get(normal(value), value.casefold()), key
    for value, origin in ((attrs.get('long_name', ''), 'long_name'),
                          (name, 'variable_name')):
        if normal(value) in ALIASES:
            return ALIASES[normal(value)], origin
    return None, None


REFERENCE_ATTRIBUTES=("coordinates","bounds","climatology","grid_mapping","ancillary_variables",
                      "cell_measures","formula_terms")
_TERMS=re.compile(r"([^\s:]+)\s*:\s*")


def variable_references(facts):
    """Resolve CF references in the same group, or by an explicit group path."""
    from posixpath import dirname, join, normpath

    variables=facts["variables"]
    graph={}
    for name,item in variables.items():
        references=[]
        seen=set()

        def add(relation,token,term=None):
            token=str(token).strip()
            if not token:
                return
            if token.startswith("/"):
                candidates=[token.lstrip("/")]
            else:
                candidates=[]
                group=dirname(name)
                while group:
                    candidates.append(normpath(join(group,token)))
                    if "/" in token:
                        break
                    group=dirname(group)
                if "/" not in token or not candidates:
                    candidates.append(normpath(token))
            target=next((candidate for candidate in candidates if candidate in variables),None)
            key=(relation,target or token,term)
            if key not in seen:
                seen.add(key)
                row=dict(relation=relation,reference=token,target=target)
                if term:
                    row["term"]=term
                references.append(row)

        attrs=item["attributes"]
        for dim in item.get("dimensions",[]):
            group=dirname(name)
            candidates=[join(group,dim),dim] if group else [dim]
            if any(key in variables and key!=name and variables[key]["dimensions"]==[dim] for key in candidates):
                add("dimension_coordinate",dim)
        for scale in item.get("dimension_scales",[]):
            add("dimension_coordinate",scale)
        for relation in REFERENCE_ATTRIBUTES:
            value=str(attrs.get(relation) or "").strip()
            groups=list(_TERMS.finditer(value)) if relation in {"cell_measures","formula_terms","grid_mapping"} else []
            if not groups:
                for token in value.split():
                    add(relation,token)
                continue
            for index,match in enumerate(groups):
                term=match[1]
                end=groups[index+1].start() if index+1<len(groups) else len(value)
                tokens=value[match.end():end].split()
                if relation=="grid_mapping":
                    add(relation,term)
                    for token in tokens:
                        add("grid_mapping_coordinate",token,term)
                elif tokens:
                    add(relation,tokens[0],term)
        if references:
            graph[name]=references
    return graph


def describe_variables(facts, *, references=None):
    """Header-only variable roles; never load arrays or assume data are valid."""
    variables = facts['variables']; dimensions = facts.get('dimensions', {})
    links = {}; coordinates = set(dimensions)
    graph=variable_references(facts) if references is None else references
    for name,refs in graph.items():
        for ref in refs:
            target=ref["target"]
            if target is None:
                continue
            relation=ref["relation"]
            if relation in {"coordinates","dimension_coordinate","grid_mapping_coordinate"}:
                coordinates.add(target)
            else:
                links.setdefault(target,[]).append(dict(parent=name,relation=relation))
    rows = []
    for name, item in variables.items():
        attrs = item['attributes']; evidence = []
        text = str(attrs.get('long_name', '')) + ' ' + str(attrs.get('standard_name', ''))
        role = 'physical_candidate'
        if name in coordinates or attrs.get('axis') in ('X', 'Y', 'Z', 'T'):
            role = 'coordinate'
        elif attrs.get('grid_mapping_name'):
            role = 'coordinate_system'
        elif name in links:
            role = 'ancillary'; evidence.extend(links[name])
        elif any(k in attrs for k in ('flag_values', 'flag_masks', 'flag_meanings')):
            role = 'quality_flag'
        elif re.search(r'\b(quality|standard deviation|uncertainty|variance)\b', text, re.I):
            role = 'ancillary'
        elif name.casefold().endswith('_raw') and 'source_fill_value' in attrs:
            # Retain, but do not fuse the unscaled encoding alongside its
            # decoded scientific field. This is evidence, not suffix alone.
            role = 'raw_encoding'
        elif attrs.get('cf_role') or name == 'TFLAG':
            role = 'identifier_or_time_control'
        quantity, basis = identity(name, attrs)
        rows.append(dict(name=name, role=role, dimensions=item['dimensions'],
                         unit=str(attrs.get('units', '')).strip(), quantity=quantity,
                         identity_basis=basis, evidence=evidence,
                         attributes=attrs))
    return rows


def dataset_facts(ds):
    # cf-xarray's coordinate discovery is used in addition to CF references;
    # this handles coordinate names that are not latitude/longitude/time.
    import cf_xarray  # noqa: F401
    facts = dict(dimensions=dict(ds.sizes), attributes=dict(ds.attrs),
                 variables={n: dict(dimensions=list(v.dims), attributes=dict(v.attrs))
                            for n, v in ds.variables.items()})
    cf_coords = set()
    for names in ds.cf.coordinates.values():
        cf_coords.update(names)
    for names in ds.cf.axes.values():
        cf_coords.update(names)
    return facts, set(ds.coords) | cf_coords


def match_fields(ds, fields):
    """Return unambiguous mappings plus evidence, without touching data values.

    Physical identity AND convertible units are required. Dimensionless
    quantities such as LAI and albedo cannot match by units alone. Missing or
    ambiguous identities/units are explicit; raw/QC fields cannot win a match.
    """
    from ecodata.engines.units import UnitEngine
    from ecodata.core.exceptions import SchemaError
    facts, coords = dataset_facts(ds)
    rows = {r['name']: r for r in describe_variables(facts)}
    mapping = {}; reports = []
    for field in fields:
        wanted = [field['name'], field.get('standard_name', ''),
                  *field.get('aliases', [])]
        wanted_names = {normal(x) for x in wanted if x}
        wanted_quantities = {ALIASES.get(normal(x), str(x).casefold()) for x in wanted if x}
        known_quantities = {ALIASES[normal(x)] for x in wanted if normal(x) in ALIASES}
        if field.get('standard_name'):
            known_quantities.add(field['standard_name'].casefold())
        candidates = []; rejected = []
        for name in ds.data_vars:
            row = rows[name]
            if name in coords or row['role'] != 'physical_candidate' or ds[name].dtype.kind not in 'iuf':
                continue
            identities = {normal(name), normal(ds[name].attrs.get('standard_name', '')),
                          normal(ds[name].attrs.get('physical_variable', ''))} - {''}
            if row['identity_basis'] in ('standard_name', 'physical_variable') and known_quantities and row['quantity'] not in known_quantities:
                continue  # Explicit conflicting physics cannot be overridden by a familiar filename/alias.
            matched = bool(wanted_names & identities or row['quantity'] in wanted_quantities)
            if not matched:
                continue
            item = dict(source=name, quantity=row['quantity'],
                        identity_basis=row['identity_basis'] or 'declared_name_or_alias',
                        source_unit=row['unit'], target_unit=field.get('unit'))
            try:
                scale, offset = UnitEngine().affine(row['unit'], field['unit'])
                item.update(scale=scale, offset=offset)
                candidates.append(item)
            except (SchemaError, ValueError, TypeError, KeyError) as exc:
                rejected.append(dict(**item, reason=str(exc)))
        status = 'MATCHED' if len(candidates) == 1 else 'AMBIGUOUS' if candidates else 'MISSING_OR_INCOMPATIBLE'
        if status == 'MATCHED':
            mapping[field['name']] = candidates[0]['source']
        reports.append(dict(target=field['name'], status=status, candidates=candidates, rejected=rejected))
    return mapping, reports


def relationships(records):
    """Cross-file physical candidates, not a claim that fusion is executable."""
    from ecodata.engines.units import UnitEngine
    from ecodata.core.exceptions import SchemaError
    grouped = {}
    for record in records:
        for field in record.get('scientific_variables', []):
            if field['role'] == 'physical_candidate' and field['quantity']:
                grouped.setdefault(field['quantity'], []).append((record, field))
    result = []
    for quantity, items in sorted(grouped.items()):
        if len({r['file'] for r, _ in items}) < 2:
            continue
        sources = []; issues = []
        target_unit = items[0][1]['unit']
        for record, field in items:
            facts = record.get('facts', {}); attrs = facts.get('attributes', {})
            support = field['attributes'].get('temporal_semantics') or attrs.get('time_support') or attrs.get('temporal_semantics')
            cells = field['attributes'].get('cell_methods')
            if not support and cells:
                support = str(cells)
            try:
                scale, offset = UnitEngine().affine(field['unit'], target_unit)
                conversion = dict(scale=scale, offset=offset)
            except (SchemaError, ValueError, TypeError) as exc:
                conversion = None; issues.append(dict(file=record['file'], code='UNIT', message=str(exc)))
            if not support:
                issues.append(dict(file=record['file'], code='TIME_SUPPORT_UNDECLARED',
                                   message='未声明瞬时/平均/累计或合成时间支持，不能只按时间标签混合'))
            if field['attributes'].get('quality_note'):
                issues.append(dict(file=record['file'], code='SOURCE_QUALITY_POLICY',
                                   message=str(field['attributes']['quality_note'])))
            sources.append(dict(file=record['file'], variable=field['name'], unit=field['unit'],
                unit_conversion=conversion, dimensions=field['dimensions'],
                temporal_support=support, cell_methods=cells,
                time_coverage={k: attrs[k] for k in ('time_coverage_start', 'time_coverage_end') if k in attrs},
                source_role=record.get('role', 'not_classified'),
                source_origin=attrs.get('source_url') or attrs.get('source')))
        result.append(dict(quantity=quantity, status='CANDIDATE_ONLY', sources=sources,
            issues=issues, independent_sources_verified=False, algorithm_executed=False,
            required_checks=['独立来源/工程用途', '空间覆盖、坐标与重网格守恒要求',
                             '时间支持及覆盖相容', '质量标记处理规则', '目标工程写入契约']))
    return result


def compare_units(left_quantity, left_unit, right_quantity, right_unit):
    """Match physical identity before comparing representation or scale."""
    from math import isfinite, isclose
    from ecodata.core.exceptions import SchemaError
    from ecodata.engines.units import UnitEngine

    units = [str(unit or "").strip() for unit in (left_unit, right_unit)]
    target, source = units
    report = dict(quantity=left_quantity if left_quantity == right_quantity else None,
                  source_unit=source, target_unit=target)
    if not report["quantity"]:
        return dict(report, status="physical_identity_unconfirmed")
    if not all(units):
        return dict(report, status="unit_unknown")
    try:
        scale, offset = UnitEngine().affine(source, target)
        if not isfinite(scale) or not isfinite(offset) or scale == 0:
            raise ValueError("invalid affine conversion")
    except (SchemaError, ValueError, TypeError, OverflowError):
        return dict(report, status="unit_not_convertible")
    equivalent = isclose(scale, 1., rel_tol=1e-12) and isclose(offset, 0., abs_tol=1e-12)
    return dict(report, status="equivalent" if equivalent else "convertible",
                scale=scale, offset=offset)


def convert_matched_field(field, quantity, target_unit, *, verified_quantity=None):
    """Harmonize a verified field prior to passing it to numerical processing."""
    from ecodata.engines.units import UnitEngine

    actual, _ = identity(field.name, field.attrs)
    if actual is None:
        actual = verified_quantity
    report = compare_units(quantity, target_unit, actual, field.attrs.get("units"))
    if report["status"] not in {"equivalent", "convertible"}:
        raise ValueError("物理量或单位未通过核验：" + report["status"])
    return UnitEngine().convert(field, target_unit)
