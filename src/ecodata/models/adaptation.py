"""Selected-input preparation, separate from whole-model completeness checks."""
from copy import deepcopy

import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.models.requirements import GENERIC_UNITS, _unsupported_reason


def input_profile(profile, role):
    selected = next(item for item in profile["inputs"] if item["name"] == role)
    result = deepcopy(profile)
    result["inputs"] = [deepcopy(selected)]
    result["inputs"][0]["depends_on"] = []
    result["required_variables"] = result["inputs"]
    result["selected_input"] = role
    result["selected_input_dependencies"] = list(selected["depends_on"])
    frequency = selected.get("temporal_resolution")
    try:
        pd.tseries.frequencies.to_offset(frequency) if frequency else None
    except ValueError:
        frequency = None
    result["temporal_resolution"] = frequency
    result["spatial_resolution_km"] = selected.get("spatial_resolution_km")
    return result


def field_requirements(requirement):
    if "fields" in requirement:
        return requirement["fields"]
    if requirement["unit"].casefold() in GENERIC_UNITS or requirement["data_type"] in {"geometry", "multidimensional_array"}:
        return []
    return [dict(requirement)]


def suggest_mapping(fields, descriptor):
    mapping = {}
    for field in fields:
        names = {field["name"].casefold(), field.get("standard_name", field["name"]).casefold(),
                 *(item.casefold() for item in field.get("aliases", []))}
        mapping[field["name"]] = next((name for name in descriptor.variables if name.casefold() in names), "")
    return mapping


def unit_transform(source, target):
    """Retain the public affine API; Pint owns physical unit conversions."""
    if target.casefold() in GENERIC_UNITS or source == target:
        return 1., 0.
    from ecodata.engines.units import UnitEngine
    return UnitEngine().affine(source, target)


def prepare_input(dataset, descriptor, profile, mapping, selected_variables=()):
    requirement = profile["inputs"][0]
    fields = field_requirements(requirement)
    missing, unsupported, satisfied, convertible, unit_issues, operations = [], [], [], [], [], []
    from ecodata.fusion.planning import DataRoleClassifier
    role_check = DataRoleClassifier().compatibility(dataset, {**requirement, "fields": fields}, mapping, selected_variables)
    unsupported.extend({"requirement": requirement["name"], "reason": reason} for reason in role_check["reasons"])
    if reason := _unsupported_reason(requirement, descriptor):
        unsupported.append({"requirement": requirement["name"], "reason": reason})
    output = dataset.copy(deep=True)
    rename = {}
    for field in fields:
        target, source = field["name"], mapping.get(field["name"], "")
        if not source or source not in dataset.data_vars:
            if field.get("required", True):
                missing.append(target)
            continue
        if field.get("data_type", requirement["data_type"]) in {"continuous", "integer_count"} and dataset[source].dtype.kind not in "biuf":
            unsupported.append({"requirement": target, "reason": f"{source} 不是数值字段，不能作为连续物理变量"})
            continue
        unit = field.get("unit", "")
        actual = descriptor.units.get(source, "")
        try:
            scale, offset = unit_transform(actual, unit)
            if source != target and target in dataset.variables:
                raise SchemaError(f"字段映射目标 {target} 已存在，不能覆盖原字段")
            if scale != 1 or offset != 0:
                from ecodata.engines.units import UnitEngine
                output[source] = UnitEngine().convert(output[source], unit)
                operations.append({"字段": source, "处理": "单位换算", "源单位": actual, "目标单位": unit,
                                   "scale": scale, "offset": offset})
                convertible.append({"requirement": target, "conversion": f"{actual} → {unit}"})
            else:
                satisfied.append({"requirement": target, "reason": f"字段 {source} 已匹配"})
            if unit.casefold() not in GENERIC_UNITS:
                output[source].attrs["units"] = unit
            output[source].attrs["source_field_name"] = source
            output[source].attrs["source_units"] = actual
            if source != target:
                rename[source] = target
                operations.append({"字段": source, "处理": "字段映射", "目标字段": target})
        except SchemaError as exc:
            unit_issues.append({"variable": target, "expected": unit, "actual": actual, "reason": str(exc)})
    if len(set(mapping.values())-{""}) != len([value for value in mapping.values() if value]):
        unsupported.append({"requirement": requirement["name"], "reason": "同一源字段不能同时映射到多个物理变量"})
    if not fields and not selected_variables:
        missing.append("请选择当前数据中要作为该输入类型内容的变量")
    if not missing and not unsupported and not unit_issues:
        output = output.rename(rename)
    report = {"scope": "selected_input", "model_id": profile["id"], "input_type": requirement["name"],
              "field_schema_verified": bool(fields), "field_mapping": mapping,
              "selected_variables": list(selected_variables), "satisfied": satisfied, "convertible": convertible,
              "missing": missing, "unsupported": unsupported, "unit_issues": unit_issues,
              "preparation_operations": operations,
              "dependencies_not_checked": profile.get("selected_input_dependencies", []),
              "can_generate_native_model_input": False,
              "role_compatibility": role_check,
              "ready": not missing and not unsupported and not unit_issues}
    return output, report
