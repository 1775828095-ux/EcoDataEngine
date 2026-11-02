"""One versioned schema for built-in and user-defined model requirements."""
from __future__ import annotations

from copy import deepcopy
import re


SCHEMA_VERSION = "1.0"
MODEL_STATUSES = {
    "requirements_index",
    "standardized_preprocessing",
    "verified_native_adapter",
}
INPUT_DATA_TYPES = {
    "continuous",
    "integer_count",
    "categorical",
    "string_identifier",
    "datetime",
    "geometry",
    "multidimensional_array",
    "quality_metadata",
}

STATUS_LABELS = {
    "requirements_index": "需求索引",
    "standardized_preprocessing": "标准化预处理支持",
    "verified_native_adapter": "已验证原生适配",
}


_PRIORITY_OVERRIDES = {
    "cmaq": {"output_format": "NetCDF", "temporal_resolution": "1h"},
    "wrf": {"output_format": "NetCDF"},
    "swat_plus": {"output_format": "CSV", "temporal_resolution": "1D"},
    "wasp": {"output_format": "NetCDF", "status": "standardized_preprocessing"},
    "modflow6": {"output_format": "CSV"},
}

_WASP_INPUTS = {
    "segment_geometry": dict(standard_name="segment_geometry", unit="model-specific length", data_type="geometry", temporal_resolution=None, spatial_resolution_km=None, formats=["CSV", "GeoJSON", "Shapefile ZIP"], required=True, depends_on=[]),
    "flow": dict(standard_name="water_volume_flow_rate", unit="model/project-specific", data_type="continuous", temporal_resolution="simulation-specific", spatial_resolution_km=None, formats=["CSV", "NetCDF", "HDF5-NetCDF"], required=True, depends_on=["segment_geometry"]),
    "boundary_concentrations": dict(standard_name="boundary_concentration", unit="state-variable-specific", data_type="continuous", temporal_resolution="simulation-specific", spatial_resolution_km=None, formats=["CSV", "NetCDF", "HDF5-NetCDF"], required=True, depends_on=["segment_geometry"]),
    "loads": dict(standard_name="external_load", unit="state/parameter-specific", data_type="continuous", temporal_resolution="simulation-specific", spatial_resolution_km=None, formats=["CSV", "NetCDF", "HDF5-NetCDF"], required=True, depends_on=["segment_geometry"]),
    "water_quality_state": dict(standard_name="water_quality_state", unit="state-variable-specific", data_type="continuous", temporal_resolution="initial-condition or simulation-specific", spatial_resolution_km=None, formats=["CSV", "NetCDF", "HDF5-NetCDF"], required=True, depends_on=["segment_geometry"]),
    "kinetic_parameters": dict(standard_name="kinetic_parameter", unit="parameter-specific", data_type="multidimensional_array", temporal_resolution=None, spatial_resolution_km=None, formats=["CSV", "JSON", "NetCDF"], required=True, depends_on=["water_quality_state"]),
}


def normalize_profile(raw: dict) -> dict:
    """Normalize legacy profile JSON or a user profile into the public schema."""
    source = deepcopy(raw)
    if source.get("schema_version") == SCHEMA_VERSION and "inputs" in source:
        validate_profile(source)
        source["required_variables"] = [dict(item) for item in source["inputs"]]
        source["adapter_status"] = source["status"]
        return source

    model_id = source["id"]
    aliases = source.get("variable_aliases", {})
    inputs = []
    for item in source.get("required_variables", []):
        if isinstance(item, dict):
            name = item["name"]
            requirement = {
                "name": name,
                "standard_name": item.get("standard_name", name),
                "unit": item.get("unit", "model-specific"),
                "data_type": item.get("data_type", "multidimensional_array"),
                "temporal_resolution": item.get("temporal_resolution"),
                "spatial_resolution_km": item.get("spatial_resolution_km"),
                "formats": list(item.get("formats", source.get("input_formats", []))),
                "required": item.get("required", True),
                "depends_on": list(item.get("depends_on", [])),
                "aliases": list(item.get("aliases", aliases.get(name, []))),
            }
        else:
            name = item
            specific = _WASP_INPUTS.get(name) if model_id == "wasp" else None
            requirement = {
                "name": name,
                "standard_name": name,
                "unit": "model-specific",
                "data_type": "multidimensional_array",
                "temporal_resolution": None,
                "spatial_resolution_km": None,
                "formats": list(source.get("input_formats", [])),
                "required": True,
                "depends_on": [],
                "aliases": list(aliases.get(name, [])),
                **(specific or {}),
            }
        inputs.append(requirement)
        if isinstance(item, dict) and "fields" in item:
            requirement["fields"] = deepcopy(item["fields"])

    override = _PRIORITY_OVERRIDES.get(model_id, {})
    old_status = source.get("status") or source.get("adapter_status", "index_only")
    status = {
        "index_only": "requirements_index",
        "requirements_only": "requirements_index",
        "verified": "verified_native_adapter",
    }.get(old_status, old_status)
    profile = {
        **source,
        "schema_version": SCHEMA_VERSION,
        "id": model_id,
        "name": source["name"],
        "category": source.get("category", "用户自定义"),
        "version": source.get("version", source.get("reference_version", "未指定")),
        "reference_sources": list(source.get("reference_sources", source.get("official_sources", []))),
        "inputs": inputs,
        "status": override.get("status", status),
        "output_format": source.get("output_format", override.get("output_format", _default_output(source.get("input_formats", [])))),
        "temporal_resolution": source.get("temporal_resolution", override.get("temporal_resolution")),
        "spatial_resolution_km": source.get("spatial_resolution_km"),
    }
    # Compatibility view for the existing Pipeline; it is derived, not a second database.
    profile["required_variables"] = [dict(item) for item in inputs]
    profile["adapter_status"] = profile["status"]
    validate_profile(profile)
    return profile


def validate_profile(profile: dict) -> None:
    required = {"schema_version", "id", "name", "category", "version", "reference_sources", "inputs", "status", "output_format", "temporal_resolution", "spatial_resolution_km"}
    missing = required - set(profile)
    if missing:
        raise ValueError("模型需求缺少字段：" + "、".join(sorted(missing)))
    if profile["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"不支持的模型Schema版本：{profile['schema_version']}")
    if not re.fullmatch(r"[a-z0-9_\-]+", str(profile["id"])):
        raise ValueError("模型 ID 只能使用小写字母、数字、下划线或连字符")
    if profile["status"] not in MODEL_STATUSES:
        raise ValueError("模型状态必须是需求索引、标准化预处理支持或已验证原生适配")
    if not isinstance(profile["reference_sources"], list) or not profile["reference_sources"]:
        raise ValueError("模型必须提供至少一个参考来源")
    if not isinstance(profile["inputs"], list) or not profile["inputs"]:
        raise ValueError("模型必须包含输入需求")
    names = {item.get("name") for item in profile["inputs"]}
    input_fields = {"name", "standard_name", "unit", "data_type", "temporal_resolution", "spatial_resolution_km", "formats", "required", "depends_on", "aliases"}
    for item in profile["inputs"]:
        if "fields" in item:
            if not isinstance(item["fields"], list) or any(not isinstance(field, dict) or not field.get("name") for field in item["fields"]):
                raise ValueError("输入类型 fields 必须为含 name 的字段配置列表")
            if len({field["name"] for field in item["fields"]}) != len(item["fields"]):
                raise ValueError("输入类型 fields 不允许重复字段名称")
        if missing_input := input_fields - set(item):
            raise ValueError(f"输入需求 {item.get('name', '?')} 缺少字段：{sorted(missing_input)}")
        if item["data_type"] not in INPUT_DATA_TYPES:
            raise ValueError(f"输入需求 {item['name']} 的数据类型无效")
        if not isinstance(item["formats"], list) or not item["formats"]:
            raise ValueError(f"输入需求 {item['name']} 必须声明输入格式")
        unknown_dependencies = set(item["depends_on"]) - names
        if unknown_dependencies:
            raise ValueError(f"输入需求 {item['name']} 引用了不存在的依赖：{sorted(unknown_dependencies)}")
    if profile["status"] == "verified_native_adapter" and not profile.get("verified_sample"):
        raise ValueError("已验证原生适配必须记录对应版本真实样例证据")
    scenario = profile.get("preparation_scenario")
    if scenario:
        if not {"id", "name", "status", "inputs", "target_crs", "native_gaps", "native_model_input"} <= set(scenario):
            raise ValueError("模型准备场景缺少必要配置")
        roles = [item.get("role") for item in scenario["inputs"]]
        if len(roles) != len(set(roles)):
            raise ValueError("模型准备场景输入类别重复")
        for item in scenario["inputs"]:
            if not {"role", "name", "required", "formats", "dimensions", "crs", "time", "space", "fields"} <= set(item):
                raise ValueError("场景输入需声明格式、维度、CRS、时间、空间与字段")
            if not item["fields"] or any(not {"name", "unit", "data_type"} <= set(field) for field in item["fields"]):
                raise ValueError("场景字段需声明名称、单位与数据类型")


def editable_template() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "my_model",
        "name": "自定义模型",
        "category": "用户自定义",
        "version": "1.0",
        "reference_sources": ["请填写官方文档或项目说明来源"],
        "inputs": [{
            "name": "temperature", "standard_name": "air_temperature", "unit": "K",
            "data_type": "continuous", "temporal_resolution": "1h",
            "spatial_resolution_km": None, "formats": ["CSV", "NetCDF"],
            "required": True, "depends_on": [], "aliases": ["T", "温度"],
        }],
        "status": "requirements_index",
        "output_format": "NetCDF",
        "temporal_resolution": "1h",
        "spatial_resolution_km": None,
    }


def _default_output(formats: list[str]) -> str:
    joined = " ".join(formats).lower()
    if "netcdf" in joined:
        return "NetCDF"
    if "csv" in joined or "ascii" in joined or "text" in joined:
        return "CSV"
    return "NetCDF"
