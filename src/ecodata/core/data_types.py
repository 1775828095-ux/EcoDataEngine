from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticType:
    id: str
    name_zh: str
    safe_operations: tuple[str, ...]
    prohibited_operations: tuple[str, ...]


SEMANTIC_TYPES = {
    item.id: item for item in (
        SemanticType("continuous", "连续数值", ("linear_interpolation", "fusion", "regrid", "qp_when_applicable"), ()),
        SemanticType("integer_count", "整数及计数", ("sum", "conservative_allocation"), ("linear_interpolation_without_physical_rule",)),
        SemanticType("categorical", "分类变量", ("nearest", "majority"), ("linear_interpolation", "qp")),
        SemanticType("string_identifier", "字符串和标识符", ("preserve", "explicit_mapping"), ("numeric_interpolation", "qp")),
        SemanticType("datetime", "时间和日期", ("calendar_conversion", "timezone_conversion"), ("numeric_interpolation_as_value",)),
        SemanticType("geometry", "空间坐标及几何对象", ("crs_transform", "topology_preserving_conversion"), ("field_interpolation_as_measurement",)),
        SemanticType("multidimensional_array", "多维数组", ("dimension_aware_transform",), ("dimension_drop_without_confirmation",)),
        SemanticType("quality_metadata", "缺失值、质量标记及元数据", ("preserve", "explicit_quality_mapping"), ("silent_drop",)),
    )
}

DATA_DOMAINS = (
    "大气", "气象", "水文", "水质", "地下水", "土壤地质", "地理空间",
    "遥感", "生态", "污染排放", "模型初始条件", "模型边界条件",
)

FORMAT_CAPABILITIES = {
    "CSV": "read_write_verified",
    "TSV": "read_write_verified",
    "JSON": "read_write_verified",
    "Excel": "read_write_verified",
    "NetCDF": "read_write_verified",
    "HDF5-NetCDF": "read_write_verified",
    "Parquet": "read_write_verified",
    "GeoJSON": "read_write_verified",
    "GeoTIFF": "structure_page_verified",
    "Shapefile": "structure_page_verified",
    "HDF5 (2D dataset)": "structure_page_verified",
    "GRIB": "registered_not_implemented",
}


def recommended_spatial_method(semantic_type: str, *, downscaling: bool) -> str:
    if semantic_type == "continuous":
        return "bilinear" if downscaling else "conservative"
    if semantic_type == "categorical":
        return "nearest"
    if semantic_type == "integer_count":
        return "conservative"
    raise ValueError(f"数据类型 {semantic_type!r} 不允许自动空间插值")
