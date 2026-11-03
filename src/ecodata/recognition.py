"""Format-independent structural and semantic inspection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.crs import CRS
from shapely.geometry import shape

from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import GEOMETRY, read_structure, projected_coordinate_names


@dataclass(frozen=True)
class DataDescriptor:
    filename: str
    format: str
    structure: str
    variables: tuple[str, ...]
    units: dict[str, str]
    semantic_types: dict[str, str]
    time_coordinate: str | None
    temporal_resolution: str | None
    x_coordinate: str | None
    y_coordinate: str | None
    crs: str | None
    dimensions: dict[str, int]
    confirmations_required: tuple[str, ...]
    spatial_resolution_km: float | None = None
    pixel_width_km: float | None = None
    pixel_height_km: float | None = None
    spatial_resolution_reason: str | None = None
    geometry_types: tuple[str, ...] = ()
    extent: tuple[float, float, float, float] | None = None
    field_types: dict[str, str] | None = None
    rasterizable_fields: tuple[str, ...] = ()
    missing_values: dict[str, int] | None = None
    cf_metadata: dict | None = None
    location_metadata: dict | None = None

    def as_dict(self):
        return asdict(self)


def _format(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".csv": "CSV", ".tsv": "TSV", ".xlsx": "Excel", ".parquet": "Parquet",
        ".json": "JSON", ".geojson": "GeoJSON", ".nc": "NetCDF", ".nc4": "NetCDF",
        ".h5": "HDF5", ".hdf5": "HDF5", ".hdf": "HDF5", ".tif": "GeoTIFF",
        ".tiff": "GeoTIFF", ".zip": "Shapefile ZIP",
    }.get(suffix, suffix.lstrip(".").upper() or "未知")


def _split_variable_unit(name: str) -> tuple[str, str] | None:
    """Return a clean variable name and unit for `name(unit)` headers."""
    match = re.fullmatch(r"\s*(.+?)\s*[（(]\s*([^()（）]+?)\s*[）)]\s*", name)
    if not match:
        return None
    variable, unit = (item.strip() for item in match.groups())
    return (variable, unit) if variable and unit else None


def _normalize_variable_headers(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    parsed = {}
    occupied = set(dataset.variables)
    for name in dataset.data_vars:
        result = _split_variable_unit(name)
        if result is None:
            continue
        variable, unit = result
        if variable != name and variable in occupied:
            raise SchemaError(f"字段 {name} 拆分后与已有字段 {variable} 重名，请先明确改名")
        rename[name] = variable
        parsed[variable] = (name, unit)
        occupied.add(variable)
    if rename:
        dataset = dataset.rename(rename)
    for variable, (source_name, unit) in parsed.items():
        dataset[variable].attrs.setdefault("units", unit)
        dataset[variable].attrs["source_field_name"] = source_name
    return dataset


def _semantic(value: xr.DataArray, name: str) -> str:
    if name == GEOMETRY:
        return "geometry"
    lower = name.lower()
    if lower in {"time", "date", "datetime", "timestamp", "year", "month", "month_start"} or any(token in lower for token in ("时间", "日期", "年份", "月份")) or value.dtype.kind in "Mm":
        return "datetime"
    if lower.startswith("source_") or lower.endswith("_flag") or any(token in lower for token in ("来源", "源文件", "源表", "原始行", "质量标记")):
        return "quality_metadata"
    if lower in {"id", "station_id", "name", "label", "code", "qc_flag", "quality_flag"}:
        return "string_identifier" if "qc" not in lower and "quality" not in lower else "quality_metadata"
    if any(token in lower for token in ("编码", "代码", "编号", "_id", "_code")):
        return "string_identifier"
    if value.dtype.kind in "OUS":
        if any(token in lower for token in ("类别", "类型", "地区", "省份", "城市", "流域", "河流", "名称", "class", "type", "region", "province", "city", "basin", "river", "_name")):
            return "categorical"
        try:
            raw = pd.Series(value.values.ravel())
            meaningful = raw[~raw.astype(str).str.strip().str.lower().isin({"", "nan", "none", "null", "na", "n/a"})]
            converted = pd.to_numeric(meaningful, errors="coerce")
            if len(converted) and converted.notna().all():
                return "continuous"
        except Exception:
            pass
        return "string_identifier"
    if value.attrs.get("flag_values") is not None or value.attrs.get("data_kind") == "categorical":
        return "categorical"
    if value.dtype.kind in "iu" and any(token in lower for token in ("count", "number", "frequency", "records", "samples")):
        return "integer_count"
    return "continuous" if value.dtype.kind in "fiu" else "multidimensional_array"


def _coerce_recognised_types(dataset: xr.Dataset) -> xr.Dataset:
    """Apply only conversions justified by the recognised field semantics."""
    dataset = dataset.copy()
    null_tokens = {"", "nan", "none", "null", "na", "n/a"}
    for name in dataset.data_vars:
        value = dataset[name]
        lower = name.lower()
        semantic = _semantic(value, name)
        is_year_or_month = lower in {"year", "month"} or any(token in lower for token in ("年份", "月份"))
        if semantic != "continuous" and not is_year_or_month:
            continue
        raw = pd.Series(value.values.ravel())
        cleaned = raw.mask(raw.astype(str).str.strip().str.lower().isin(null_tokens))
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric[cleaned.notna()].isna().any():
            continue
        attrs = dict(value.attrs)
        if is_year_or_month:
            present = numeric.dropna()
            limits = np.iinfo(np.int32)
            if not np.equal(present, np.floor(present)).all() or ((present <= limits.min) | (present > limits.max)).any():
                raise SchemaError(f"年份/月份字段 {name} 必须为可表示的整数或缺失值")
            dtype = np.float64 if numeric.isna().any() else np.int32
            converted = numeric.to_numpy(dtype=dtype).reshape(value.shape)
        elif semantic == "continuous":
            converted = numeric.to_numpy(dtype=np.float64).reshape(value.shape)
        else:
            continue
        dataset[name] = xr.DataArray(converted, dims=value.dims, coords=value.coords, attrs=attrs)
        if is_year_or_month and numeric.isna().any():
            dataset[name].encoding.update(dtype="int32", _FillValue=np.iinfo(np.int32).min)
    return dataset


def inspect_dataset(filename: str, dataset: xr.Dataset) -> DataDescriptor:
    from ecodata.engines.semantics import SemanticEngine
    semantic_engine = SemanticEngine()
    cf = semantic_engine.inspect(dataset)
    variables = tuple(name for name in dataset.data_vars if name != GEOMETRY)
    units = {}
    for name in variables:
        explicit = str(dataset[name].attrs.get("units", ""))
        match = re.search(r"[（(]\s*([^()（）]+?)\s*[）)]\s*$", name)
        suffix_unit = "mg/L" if re.search(r"(?:^|_)mg_L(?:$|_)", name, flags=re.IGNORECASE) else ""
        units[name] = explicit or (match.group(1).strip() if match else suffix_unit)
    semantics = {name: _semantic(dataset[name], name) for name in dataset.data_vars}
    geometry_types, extent, rasterizable = (), None, ()
    if GEOMETRY in dataset:
        geometries = [shape(json.loads(value)) for value in dataset[GEOMETRY].values]
        if not geometries or any(g.is_empty or not g.is_valid for g in geometries):
            raise SchemaError("矢量包含空几何或无效几何，无法自动确定栅格化范围")
        bounds = np.asarray([g.bounds for g in geometries])
        extent = (float(bounds[:, 0].min()), float(bounds[:, 1].min()),
                  float(bounds[:, 2].max()), float(bounds[:, 3].max()))
        geometry_types = tuple(sorted({g.geom_type for g in geometries}))
        rasterizable = tuple(name for name in variables if dataset[name].dtype.kind in "biuf"
                            and semantics[name] != "string_identifier" and np.isfinite(dataset[name].values).all())
    available = list(dataset.coords) + list(dataset.data_vars)
    time = next((name for name in available if name.lower() in {"time", "date", "datetime", "timestamp", "month_start"} or any(token in name.lower() for token in ("时间", "日期"))), None)
    if 'time' in dataset and dataset['time'].dtype.kind=='M':
        time='time'
    if time is None:
        time = next((name for name in available if name.lower() in {"year", "month"} or any(token in name.lower() for token in ("年份", "月份"))), None)
    temporal_resolution = None
    time = semantic_engine.coordinate(cf, "T", "time") or time
    if time and time in dataset and dataset[time].size > 1:
        try:
            if dataset[time].dtype.kind in "biuf":
                raise ValueError("数值年份/月号不能当作纳秒时间戳")
            values = pd.DatetimeIndex(pd.to_datetime(dataset[time].values)).dropna().unique().sort_values()
            delta = np.diff(values.as_unit("ns").asi8)
            if len(delta) and np.all(delta == delta[0]):
                temporal_resolution = str(pd.to_timedelta(int(delta[0]), unit="ns"))
        except Exception:
            pass
    x = dataset.attrs.get("spatial_x") or next((name for name in available if name.lower() in {"lon", "longitude", "x", "经度"}), None)
    y = dataset.attrs.get("spatial_y") or next((name for name in available if name.lower() in {"lat", "latitude", "y", "纬度"}), None)
    crs = dataset.attrs.get("crs")
    x = semantic_engine.coordinate(cf, "X", "longitude") or x
    y = semantic_engine.coordinate(cf, "Y", "latitude") or y
    crs = crs or cf["crs"]
    resolution = dataset.attrs.get("grid_resolution_km")
    pixel_x, pixel_y = dataset.attrs.get("pixel_width_km"), dataset.attrs.get("pixel_height_km")
    if x in dataset and y in dataset and dataset[x].dtype.kind in "biuf" and dataset[y].dtype.kind in "biuf":
        xs, ys = dataset[x].values.ravel(), dataset[y].values.ravel()
        xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
        if xs.size and ys.size:
            extent = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            if dataset.attrs.get("structure") == "grid" and x in dataset.dims and y in dataset.dims and len(xs) > 1 and len(ys) > 1:
                dx, dy = np.abs(np.diff(xs)), np.abs(np.diff(ys))
                if np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]) and dx[0] > 0 and dy[0] > 0:
                    extent = (extent[0]-dx[0]/2, extent[1]-dy[0]/2, extent[2]+dx[0]/2, extent[3]+dy[0]/2)
                    if crs and CRS.from_user_input(crs).is_projected:
                        factor = CRS.from_user_input(crs).linear_units_factor[1]/1000
                        pixel_x, pixel_y = float(dx[0]*factor), float(dy[0]*factor)
                        if np.isclose(pixel_x, pixel_y, rtol=.001):
                            resolution = (pixel_x+pixel_y)/2
    confirmations = []
    if cf["ambiguities"]:
        confirmations.append("CF 坐标存在多个候选，请确认坐标映射")
    if variables and all(not value for value in units.values()):
        confirmations.append("变量单位")
    if x or y or GEOMETRY in dataset:
        if not crs:
            confirmations.append("CRS")
    if time and not temporal_resolution:
        confirmations.append("时间分辨率或时间语义")
    return DataDescriptor(
        filename, _format(filename), dataset.attrs.get("structure", "grid"), variables,
        units, semantics, time, temporal_resolution, x, y, str(crs) if crs else None,
        dict(dataset.sizes), tuple(confirmations),
        spatial_resolution_km=resolution,
        pixel_width_km=pixel_x,
        pixel_height_km=pixel_y,
        spatial_resolution_reason=dataset.attrs.get("spatial_resolution_reason"),
        geometry_types=geometry_types, extent=extent,
        field_types={name: str(dataset[name].dtype) for name in variables},
        rasterizable_fields=rasterizable,
        missing_values={name: int(pd.isna(dataset[name].values).sum()) for name in variables},
        cf_metadata=cf,
        location_metadata=__import__("ecodata.location_metadata", fromlist=["declared_location"]).declared_location(dataset),
    )


def recognise_file(filename: str, content: bytes, **read_options) -> tuple[xr.Dataset, DataDescriptor]:
    dataset = read_structure(filename, content, **read_options)
    if not dataset.attrs.get("crs"):
        from ecodata.engines.semantics import SemanticEngine
        cf_crs = SemanticEngine().inspect(dataset).get("crs")
        if cf_crs:
            dataset.attrs["crs"] = cf_crs
            dataset.attrs["crs_identification_source"] = "CF grid_mapping / PROJ.from_cf"
    dataset = projected_coordinate_names(dataset)
    dataset = _normalize_variable_headers(dataset)
    if Path(filename).suffix.lower() in {".csv", ".tsv", ".xlsx", ".txt", ".rdb"}:
        dataset = _coerce_recognised_types(dataset)
        from ecodata.io.raw_semantics import enrich_raw_semantics
        dataset = enrich_raw_semantics(dataset)
    if not dataset.attrs.get("crs") and "crs" in dataset:
        values = pd.Series(dataset["crs"].values.ravel()).dropna().astype(str).str.strip()
        unique = values[values.ne("")].unique()
        if len(unique) == 1:
            try:
                dataset.attrs["crs"] = CRS.from_user_input(unique[0]).to_wkt()
            except Exception:
                pass
    from ecodata.location_metadata import recognise_location
    dataset = recognise_location(dataset)
    return dataset, inspect_dataset(filename, dataset)
