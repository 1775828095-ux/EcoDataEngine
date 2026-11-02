"""Independent, task-aware validation using existing reports and diagnostics."""
from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Geod

from ecodata.core.exceptions import SchemaError, ValidationError
from ecodata.core.schemas import InformationLossRecord, ValidationReport
from ecodata.evaluation.output_quality import dataset_quality
from ecodata.evaluation.metrics import spherical_cell_areas, _coordinate_edges, fidelity_metrics
from ecodata.io.structures import GEOMETRY, spatial_axes, projected_coordinate_names
from ecodata.recognition import inspect_dataset


@dataclass(frozen=True)
class ValidationThresholds:
    rtol: float = 1e-6
    atol: float = 1e-8
    max_new_missing_ratio: float = .5
    conservation_relative_error: float = 1e-6
    peak_loss_warning_percent: float = 10.


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _time_bounds(dataset):
    candidates = [dataset.time.attrs.get("bounds")] if "time" in dataset else []
    return next((dataset[name] for name in [*candidates, "time_bnds", "time_bounds"] if name and name in dataset), None)


def profile_dataset(dataset):
    """A JSON baseline with actual data statistics; no physical metadata guessed."""
    dataset = projected_coordinate_names(dataset)
    result = dataset_quality(dataset)
    info = inspect_dataset("dataset", dataset)
    for name, field in dataset.data_vars.items():
        stats = result["variables"][name]
        values = field.values
        finite = np.isfinite(values) if values.dtype.kind in "biuf" else ~pd.isna(values)
        stats.update(dimensions=list(field.dims), semantic_type=info.semantic_types.get(name),
                     mean=float(values[finite].mean()) if values.dtype.kind in "biuf" and finite.any() else None,
                     std=float(values[finite].std()) if values.dtype.kind in "biuf" and finite.any() else None,
                     sum=float(values[finite].sum()) if values.dtype.kind in "biuf" and finite.any() else None,
                     valid_pixel_count=int(finite.sum()), metadata=dict(field.attrs))
        is_quality = "flag" in name.lower() or name.lower().endswith("_qc") or field.attrs.get("flag_values") is not None
        if info.semantic_types.get(name) in {"categorical", "quality_metadata"} or is_quality:
            stats["categorical_values"] = sorted(set(str(v) for v in values[~pd.isna(values)].ravel()))
        time_dims = dataset[info.time_coordinate].dims if info.time_coordinate else ()
        has_time = bool(time_dims) and set(time_dims) <= set(field.dims)
        stats["temporal_semantics"] = field.attrs.get("temporal_semantics", dataset.attrs.get("temporal_semantics") if has_time else None)
        stats["spatial_semantics"] = field.attrs.get("spatial_semantics", dataset.attrs.get("spatial_semantics"))
    row = dataset.attrs.get("row_dimension", "observation")
    result["record_count"] = dataset.sizes.get(row, max((v.size for n, v in dataset.data_vars.items() if n != "time_bnds"), default=0))
    result.update(variable_count=len(dataset.data_vars), field_names=list(dataset.data_vars),
                  coordinate_metadata={n: {"dtype": str(v.dtype), "dimensions": list(v.dims), "metadata": dict(v.attrs)} for n, v in dataset.coords.items()},
                  metadata=dict(dataset.attrs), structure=info.structure, time_coordinate=info.time_coordinate,
                  time_range=None, source={k: v for k, v in dataset.attrs.items() if "source" in k or k in {"dataset_id", "user_confirmations"}},
                  quality_flags={n: result["variables"][n] for n in dataset.data_vars if "flag" in n.lower() or n.lower().endswith("_qc") or dataset[n].attrs.get("flag_values") is not None or info.semantic_types.get(n) == "quality_metadata"})
    if info.time_coordinate and dataset[info.time_coordinate].dtype.kind in "MOUS":
        values = pd.to_datetime(dataset[info.time_coordinate].values.ravel(), errors="coerce")
        result["invalid_time_count"] = int(values.isna().sum())
        if values.notna().any():
            result["time_range"] = [pd.Timestamp(values.min()).isoformat(), pd.Timestamp(values.max()).isoformat()]
    bounds = _time_bounds(dataset)
    if bounds is not None:
        result["time_coverage"] = [pd.Timestamp(bounds.values.min()).isoformat(), pd.Timestamp(bounds.values.max()).isoformat()]
    else:
        result["time_coverage"] = result["time_range"]
    result["pixel_extent"] = None
    try:
        x, y = spatial_axes(dataset)
        if dataset[x].ndim == dataset[y].ndim == 1 and x in dataset.dims and y in dataset.dims:
            xe, ye = _coordinate_edges(np.sort(dataset[x].values)), _coordinate_edges(np.sort(dataset[y].values))
            result["pixel_extent"] = [float(xe[0]), float(ye[0]), float(xe[-1]), float(ye[-1])]
    except (SchemaError, ValueError, KeyError):
        pass
    return json_safe(result)


def task_kind(history):
    methods = [record.get("method", "") for record in history]
    if any("fusion" in m for m in methods):
        return "fusion"
    if any(m.startswith(("temporal.", "spatial.", "downscaling", "optimization.", "downscale.")) for m in methods):
        return "resampling"
    if any(record.get("parameters", {}).get("operation", "preserve") != "preserve" or record.get("parameters", {}).get("slices") for record in history):
        return "structure"
    return "format"


def _equal(a, b):
    return json_safe(a) == json_safe(b)


def _paired(before, after):
    """Compare the same coordinate traversal, not unrelated array flattenings."""
    if set(before.dims) != set(after.dims):
        return None
    for dim in before.dims:
        if before.sizes[dim] != after.sizes[dim]:
            return None
        if dim in before.coords and dim in after.coords:
            if before[dim].ndim == after[dim].ndim == 1:
                before, after = before.sortby(dim), after.sortby(dim)
                if not np.array_equal(before[dim].values, after[dim].values):
                    return None
    return before.values, after.transpose(*before.dims).values


def _measure(dataset, name):
    """Integrals are comparable only over equivalent full support and semantics."""
    field = dataset[name]
    spatial = field.attrs.get("spatial_semantics", dataset.attrs.get("spatial_semantics"))
    temporal = field.attrs.get("temporal_semantics", dataset.attrs.get("temporal_semantics"))
    if not np.isfinite(field.values).all():
        return None, "存在缺失/无穷值，不能声称全域积分守恒"
    measured = field.astype(float)
    if spatial in {"cell_mean", "flux"}:
        if not dataset.attrs.get("crs"):
            return None, "缺少CRS，不能计算面积积分"
        try:
            x, y = spatial_axes(dataset)
            if not {x, y} <= set(field.dims):
                return None, "不是可积规则空间场"
            crs = CRS.from_user_input(dataset.attrs["crs"])
            ordered = dataset.sortby(x).sortby(y)
            measured = ordered[name].astype(float)
            if crs.is_projected:
                width = np.diff(_coordinate_edges(ordered[x].values))
                height = np.diff(_coordinate_edges(ordered[y].values))
                factor = crs.axis_info[0].unit_conversion_factor
                area = xr.DataArray(height[:, None] * width[None, :] * factor**2, dims=(y, x), coords={y: ordered[y], x: ordered[x]})
            else:
                area = spherical_cell_areas(ordered[y].rename({y: "lat"}) if y != "lat" else ordered[y],
                                            ordered[x].rename({x: "lon"}) if x != "lon" else ordered[x])
                area = area.rename({n: d for n, d in (("lat", y), ("lon", x)) if n != d})
            measured = measured * area
        except (SchemaError, ValueError, KeyError) as exc:
            return None, str(exc)
    elif spatial != "cell_total" and temporal not in {"mean", "accumulation"}:
        return None, "未声明守恒/可积语义，不虚构积分指标"
    if "time" in measured.dims and temporal == "mean":
        time_bounds = _time_bounds(dataset)
        if time_bounds is None:
            return None, "平均值没有时间边界，不能计算时间积分"
        bounds = time_bounds.values.astype("datetime64[ns]")
        duration = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
        measured = measured * xr.DataArray(duration, dims="time", coords={"time": measured.time})
    return float(measured.sum(skipna=False)), None


def _peak_position(dataset, name):
    field = dataset[name]
    if not np.isfinite(field.values).any():
        return None
    indices = np.unravel_index(np.nanargmax(field.values), field.shape)
    try:
        x, y = spatial_axes(dataset)
    except SchemaError:
        return None
    if not {x, y} <= set(field.dims):
        return None
    return {axis: float(dataset[axis].values[indices[field.dims.index(axis)]]) for axis in (x, y)}


def _station_placement(dataset, frame):
    """Coordinate-to-grid distance is geometry QA, NOT predictive accuracy."""
    required = {"latitude", "longitude", "qc_flag"}
    if not required <= set(frame) or not dataset.attrs.get("crs"):
        return {"available": False, "reason": "缺少站点坐标/质控/CRS"}
    crs = CRS.from_user_input(dataset.attrs["crs"])
    if not crs.equals(CRS.from_epsg(4326), ignore_axis_order=True):
        return {"available": False, "reason": "站点坐标距离检查需先明确转换至EPSG:4326"}
    x, y = spatial_axes(dataset)
    distances, outside = [], 0
    for row in frame.loc[frame.qc_flag.eq("VALID")].itertuples():
        if not np.isfinite([row.latitude, row.longitude]).all():
            outside += 1
            continue
        if not (float(dataset[x].min()) <= row.longitude <= float(dataset[x].max()) and float(dataset[y].min()) <= row.latitude <= float(dataset[y].max())):
            outside += 1
            continue
        lon = dataset[x].values[np.argmin(abs(dataset[x].values-row.longitude))]
        lat = dataset[y].values[np.argmin(abs(dataset[y].values-row.latitude))]
        distances.append(Geod(ellps="WGS84").inv(row.longitude, row.latitude, lon, lat)[2])
    return {"available": bool(distances), "matched_count": len(distances), "outside_or_invalid_count": outside,
            "max_nearest_cell_distance_m": max(distances) if distances else None,
            "mean_nearest_cell_distance_m": float(np.mean(distances)) if distances else None,
            "meaning": "站点到最近网格中心距离，不是站点真实定位误差或独立精度"}


class PostConversionValidator:
    """Independent assessment, not another conversion or interpolation engine."""
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or ValidationThresholds()

    def validate(self, before, after, *, kind=None, history=(), critical_fields=(), field_mapping=None,
                 sources=(), before_profile=None, model_check=None, reference=None, validation_stations=None, observation_stations=None, metadata_sidecar=False):
        history = [r.as_dict() if hasattr(r, "as_dict") else r for r in history]
        before, after = projected_coordinate_names(before), projected_coordinate_names(after)
        kind = kind or task_kind(history)
        pure = kind == "format"
        first = before_profile or profile_dataset(before)
        last = profile_dataset(after)
        mapping = field_mapping or {name: name for name in before.data_vars}
        critical = set(critical_fields)
        levels = {key: "PASS" for key in ("numerical_integrity", "variables", "units", "time", "CRS", "missing_values", "metadata", "physical_range", "conservation", "structure", "model_compatibility")}
        errors, warnings, numeric, changes = [], [], {}, {}
        loss = InformationLossRecord()

        def flag(category, message, fail=False):
            if levels[category] != "FAIL":
                levels[category] = "FAIL" if fail else "WARN"
            (errors if fail else warnings).append(message)

        loss.lost_fields = [name for name in before.data_vars if mapping.get(name, name) not in after.variables]
        if loss.lost_fields:
            flag("variables", "未保留字段：" + ", ".join(loss.lost_fields), pure or bool(set(loss.lost_fields) & critical))
        loss.dropped_records = max(0, first["record_count"] - last["record_count"]) if pure or first["structure"] == "table" else 0
        if first["record_count"] != last["record_count"]:
            flag("structure", f"记录/样本数量 {first['record_count']} → {last['record_count']}", pure)
        if not after.data_vars or any(size == 0 for size in after.sizes.values()):
            flag("structure", "结果为空", True)
        numeric_output = [field.values for field in after.data_vars.values() if field.dtype.kind in "biuf"]
        if numeric_output and not any(np.isfinite(values).any() for values in numeric_output):
            flag("missing_values", "输出数值全部无效/NaN", True)
        for name, field in after.data_vars.items():
            if field.dtype.kind in "biuf":
                if not np.isfinite(field.values).any():
                    source_name = next((source for source in before.data_vars if mapping.get(source, source) == name), None)
                    preexisting_missing = source_name is not None and before[source_name].dtype.kind in "biuf" and not np.isfinite(before[source_name].values).any()
                    flag("missing_values", name + " 全部无效/NaN" + ("（原始非关键字段已完全缺失）" if preexisting_missing else ""), not preexisting_missing or source_name in critical or name in critical)
                if np.isinf(field.values).any():
                    flag("physical_range", name + " 出现无穷值", True)
        for key, value in before.attrs.items():
            if key not in after.attrs:
                loss.lost_metadata.append("dataset:" + key)
            elif not _equal(value, after.attrs[key]) and key not in {"crs", "transform_history", "processing_stage", "export_variable", "spatial_x", "spatial_y"}:
                flag("metadata", "元数据改变：" + key, pure and not metadata_sidecar)
        for name, field in before.data_vars.items():
            target = mapping.get(name, name)
            if target not in after:
                continue
            for key in field.attrs:
                if key not in after[target].attrs:
                    loss.lost_metadata.append(name + ":" + key)
                elif key not in {"units", "temporal_semantics", "spatial_semantics"} and not _equal(field.attrs[key], after[target].attrs[key]):
                    flag("metadata", name + " 元数据改变：" + key, pure and not metadata_sidecar)
        if loss.lost_metadata:
            flag("metadata", "丢失元数据：" + ", ".join(loss.lost_metadata), pure and not metadata_sidecar)
        bc, ac = first["CRS"], last["CRS"]
        if bc and not ac:
            flag("CRS", "目标文件未保留CRS" + ("，必须同时使用JSON旁车报告" if metadata_sidecar else ""), not metadata_sidecar)
            loss.changed_crs = {"before": bc, "after": ac}
        elif bc and ac and not CRS.from_user_input(bc).equals(CRS.from_user_input(ac), ignore_axis_order=True):
            loss.changed_crs = {"before": bc, "after": ac}
            declared = kind in {"structure", "model_adaptation"} and any(r.get("parameters", {}).get("crs") or "reproject" in r.get("method", "") for r in history)
            flag("CRS", "CRS发生改变" + ("，按声明投影处理" if declared else "，未声明重投影"), not declared)
        for key in ("resolution_km", "time_resolution"):
            if not _equal(first[key], last[key]):
                loss.changed_resolution[key] = {"before": first[key], "after": last[key]}
                flag("structure", key + " 改变", pure)
        if not _equal(first["time_coverage"], last["time_coverage"]):
            flag("time", "时间覆盖范围改变", pure)
        if not _equal(first["extent"], last["extent"]):
            flag("structure", "空间范围改变；中心域与像元边界均需参考具体结构", pure)
        if GEOMETRY in before and GEOMETRY not in after:
            flag("structure", "原几何不能由目标栅格/表格无损表达，geometry已舍弃", pure)
        elif GEOMETRY in before and GEOMETRY in after and before[GEOMETRY].size == after[GEOMETRY].size:
            from shapely.geometry import shape
            from shapely.ops import transform
            from pyproj import Transformer
            projection = Transformer.from_crs(bc, ac, always_xy=True).transform if bc and ac and loss.changed_crs else None
            for a, b in zip(before[GEOMETRY].values.ravel(), after[GEOMETRY].values.ravel()):
                source_geometry, output_geometry = shape(json.loads(a)), shape(json.loads(b))
                source_geometry = transform(projection, source_geometry) if projection else source_geometry
                if not source_geometry.equals_exact(output_geometry, tolerance=1e-8):
                    flag("structure", "保留的几何位置/拓扑发生非投影解释的改变", True)
                    break
        methods = [r.get("method", "") for r in history]
        fusion_task = kind == "fusion" or any("fusion" in method for method in methods)
        unit_mapping_only = kind == "model_adaptation" and not fusion_task and task_kind(history) == "format"
        loss.interpolation_applied = any(any(token in m for token in ("linear", "bilinear", "nearest", "downscal", "residual", "interpolat")) for m in methods)
        loss.extrapolation_applied = any(r.get("parameters", {}).get("extrapolation") is True for r in history)
        if loss.extrapolation_applied:
            flag("numerical_integrity", "发生声明的外推；不是新观测")
        for name, field in before.data_vars.items():
            target = mapping.get(name, name)
            if target not in after:
                continue
            result = after[target]
            if target in after.coords:
                flag("structure", name + " 已从表格列提升为空间/时间坐标", pure)
                continue
            bs, ads = first["variables"][name], last["variables"][target]
            unit, new_unit = field.attrs.get("units"), result.attrs.get("units")
            comparison_field = field
            comparison_dataset = before
            if unit != new_unit:
                loss.changed_units[name] = {"before": unit, "after": new_unit}
                declared = result.attrs.get("source_units") == unit and kind == "model_adaptation"
                declared = declared or (not unit and bool(new_unit))
                flag("units", name + " 单位改变/丢失" + ("，单位已记录在JSON旁车" if metadata_sidecar else ""), not metadata_sidecar and (not new_unit or not declared))
                if declared and unit and new_unit and field.dtype.kind in "biuf":
                    from ecodata.models.adaptation import unit_transform
                    conversion = json.loads(result.attrs["unit_conversion"]) if result.attrs.get("unit_conversion") else None
                    try:
                        scale, offset = unit_transform(unit, new_unit)
                    except SchemaError:
                        if conversion is None:
                            flag("units", name + " 没有可验证的单位换算记录", True)
                            scale, offset = 1., 0.
                        else:
                            scale, offset = conversion["scale"], conversion["offset"]
                    comparison_field = field.astype(float)*scale+offset
                    comparison_field.attrs = dict(field.attrs, units=new_unit)
                    comparison_dataset = before.copy()
                    comparison_dataset[name] = comparison_field
                    bs = profile_dataset(comparison_dataset)["variables"][name]
                    loss.changed_units[name].update(scale=scale, offset=offset, statistics_normalized_to=new_unit)
            if bs["temporal_semantics"] != ads["temporal_semantics"]:
                loss.changed_time_semantics[name] = {"before": bs["temporal_semantics"], "after": ads["temporal_semantics"]}
                flag("time", name + " 时间语义改变/补充，需保留声明来源", bs["temporal_semantics"] is not None and not metadata_sidecar)
            if field.dtype.kind in "biuf":
                if result.dtype.kind not in "biuf":
                    flag("numerical_integrity", name + " 数值类型变为字符串", True)
                    continue
                values = result.values
                if np.isinf(values).any():
                    flag("physical_range", name + " 出现无穷值", True)
                increase = ads["NaN_ratio"] - bs["NaN_ratio"]
                if increase > 0:
                    flag("missing_values", f"{name} 缺失比例增加 {increase:.1%}", kind != "structure" and increase > self.thresholds.max_new_missing_ratio)
                changes[name] = {key: {"before": bs[key], "after": ads[key]} for key in ("min", "max", "mean", "std", "sum", "NaN_ratio", "valid_pixel_count")}
                changes[name]["comparison_unit"] = new_unit
                if bs["semantic_type"] == "continuous" and bs["max"] is not None and ads["max"] is not None:
                    denominator = abs(bs["max"])
                    peak = 100 * max(0, bs["max"] - ads["max"]) / denominator if denominator else None
                    loss.peak_loss[name] = {"amplitude_percent": peak, "before_max": bs["max"], "after_max": ads["max"]}
                    position_before, position_after = _peak_position(comparison_dataset, name), _peak_position(after, target)
                    loss.peak_loss[name].update(before_position=position_before, after_position=position_after,
                                              location_comparable=not bool(loss.changed_crs),
                                              meaning="源/目标最大值位置变化；并非独立真值位置误差，多个并列峰时只记录首个")
                    if position_before and position_after and bc and ac and not loss.changed_crs:
                        a, b = list(position_before.values()), list(position_after.values())
                        crs = CRS.from_user_input(bc)
                        distance = Geod(ellps="WGS84").inv(*a, *b)[2] if crs.is_geographic else np.linalg.norm(np.array(a)-b)*crs.axis_info[0].unit_conversion_factor
                        loss.peak_loss[name]["position_shift_m"] = float(distance)
                    loss.peak_loss[name]["extreme_region"] = {
                        "definition": "数值达到各自最大值90%以上的像元比例；非独立真值比较",
                        "before_fraction": float(np.mean(comparison_field.values >= .9*bs["max"])) if bs["max"] > 0 else None,
                        "after_fraction": float(np.mean(values >= .9*ads["max"])) if ads["max"] > 0 else None}
                    if not pure and peak and peak > self.thresholds.peak_loss_warning_percent:
                        flag("numerical_integrity", f"{name} 峰值降低 {peak:.3f}%")
                    limit = result.attrs.get("max_peak_loss_percent", field.attrs.get("max_peak_loss_percent"))
                    if not pure and limit is not None and peak is not None and peak > float(limit):
                        flag("numerical_integrity", f"{name} 峰值损失超出声明的允许上限 {limit}%", True)
                for bound, comparison in (("valid_min", np.less), ("valid_max", np.greater)):
                    limit = result.attrs.get(bound, field.attrs.get(bound))
                    if limit is not None and np.any(comparison(values[np.isfinite(values)], limit)):
                        flag("physical_range", f"{name} 违反声明的{bound}={limit}", True)
                negatives = int((values < 0).sum())
                if negatives and bs["min"] is not None and bs["min"] >= 0:
                    flag("physical_range", f"{name} 新出现{negatives}个负值；未声明非负范围时不擅自截断")
                pairs = _paired(comparison_field, result)
                if pairs is None and fusion_task:
                    pairs = _paired(comparison_field.squeeze(drop=True), result.squeeze(drop=True))
                if pairs is not None:
                    av, bv = pairs
                    missing_same = np.array_equal(pd.isna(av), pd.isna(bv))
                    if not missing_same:
                        flag("missing_values", name + " 缺失位置改变", pure)
                        loss.filled_values += int((pd.isna(av) & ~pd.isna(bv)).sum())
                    valid = np.isfinite(av) & np.isfinite(bv)
                    delta = (bv[valid].astype(object) - av[valid].astype(object)).astype(float) if av.dtype.kind in "iu" else bv[valid].astype(float) - av[valid].astype(float)
                    unacceptable = np.abs(delta) > self.thresholds.atol + self.thresholds.rtol * np.abs(av[valid].astype(float))
                    if av.dtype.kind in "biu" and bv.dtype.kind in "biu":
                        unacceptable = delta != 0
                    numeric[name] = {"max_abs_error": float(np.abs(delta).max()) if delta.size else None,
                                     "mean_abs_error": float(np.abs(delta).mean()) if delta.size else None,
                                     "changed_value_count": int((delta != 0).sum()), "unacceptable_value_count": int(unacceptable.sum()),
                                     "missing_positions_equal": missing_same}
                    if (pure or unit_mapping_only) and unacceptable.any():
                        flag("numerical_integrity", name + " 格式/字段/单位转换数值超出浮点容差", True)
                    if fusion_task and delta.size:
                        numeric[name]["correction_distribution"] = {"min": float(delta.min()), "max": float(delta.max()), "mean": float(delta.mean()), "std": float(delta.std()), "p50_abs": float(np.quantile(abs(delta), .5)), "p95_abs": float(np.quantile(abs(delta), .95))}
                elif pure:
                    flag("numerical_integrity", name + " 维度或坐标/时间错位，无法逐值比较", True)
                spatial = bs["spatial_semantics"]
                temporal = bs["temporal_semantics"]
                if not pure and not fusion_task and (spatial in {"cell_mean", "cell_total", "flux"} or temporal in {"mean", "accumulation"}):
                    a, ar = _measure(comparison_dataset, name)
                    b, br = _measure(after, target)
                    supports_match = _equal(first["time_coverage"], last["time_coverage"])
                    if spatial in {"cell_mean", "cell_total", "flux"}:
                        supports_match = supports_match and bool(first["pixel_extent"] and last["pixel_extent"]) and np.allclose(first["pixel_extent"], last["pixel_extent"], rtol=1e-8, atol=1e-8) and not loss.changed_crs
                    if temporal == "instantaneous" and first["time_coordinate"] and last["time_coordinate"]:
                        supports_match = supports_match and np.array_equal(before[first["time_coordinate"]].values, after[last["time_coordinate"]].values)
                    if a is not None and b is not None and supports_match:
                        relative = abs(b - a) / abs(a) if a else None
                        loss.conservation_error[name] = {"before_integral": a, "after_integral": b, "relative_error": relative, "absolute_error": abs(b-a), "zero_reference": a == 0}
                        required = spatial in {"cell_total", "flux"} or temporal == "accumulation" or any("conservative" in m for m in methods)
                        exceeded = relative > self.thresholds.conservation_relative_error if relative is not None else abs(b-a) > self.thresholds.atol
                        if exceeded:
                            flag("conservation", name + " 积分/总量误差超过阈值", required)
                    else:
                        loss.conservation_error[name] = {"relative_error": None, "reason": ar or br or "时空积分支持域/瞬时采样不同，不能冒充全域守恒"}
                        flag("conservation", name + " 守恒未能验证：" + loss.conservation_error[name]["reason"])
            else:
                pairs = _paired(field, result)
                if pairs is None:
                    if pure:
                        flag("time" if field.dtype.kind == "M" else "structure", name + " 记录位置或维度改变", True)
                elif pure:
                    a, b = pairs
                    if field.dtype.kind == "M":
                        a, b = pd.to_datetime(a.ravel()).to_numpy(), pd.to_datetime(b.ravel()).to_numpy()
                    if not _equal(a, b):
                        flag("time" if field.dtype.kind == "M" else "numerical_integrity", name + " 时间/分类/文字/质量标记改变", True)
                if "categorical_values" in bs and set(bs["categorical_values"]) - set(ads.get("categorical_values", [])):
                    flag("structure", name + " 分类值集合缩减", pure)
        for name in after.data_vars:
            if "bound_flag" in name or "clip_flag" in name:
                prior = int(np.count_nonzero(before[name].values)) if name in before else 0
                loss.clipped_values += max(0, int(np.count_nonzero(after[name].values)) - prior)
        placement = _station_placement(after, observation_stations) if observation_stations is not None else None
        if placement and placement.get("outside_or_invalid_count"):
            flag("structure", f"{placement['outside_or_invalid_count']}个VALID观测不在目标网格域内或坐标无效，未参与域内匹配")
        extrapolation_count = None
        if fusion_task and observation_stations is not None and any(r.get("method") == "fusion.residual_field" for r in history):
            from scipy.spatial import Delaunay, QhullError
            x, y = spatial_axes(after)
            frame = observation_stations.loc[observation_stations.qc_flag.eq("VALID")]
            frame = frame.loc[frame.longitude.between(float(after[x].min()), float(after[x].max())) & frame.latitude.between(float(after[y].min()), float(after[y].max()))]
            points = frame[["longitude", "latitude"]].to_numpy(dtype=float)
            tx, ty = np.meshgrid(after[x].values, after[y].values)
            try:
                extrapolation_count = int(np.count_nonzero(Delaunay(points).find_simplex(np.column_stack([tx.ravel(), ty.ravel()]), tol=1e-8) < 0))
                loss.extrapolation_applied |= extrapolation_count > 0
                if extrapolation_count:
                    flag("numerical_integrity", f"{extrapolation_count}个网格中心在有效站点凸包外；已有残差算法使用nearest，属于观测覆盖域外推断")
            except (ValueError, QhullError):
                flag("numerical_integrity", "站点无法形成二维凸包，不能验证空间外推范围")
        if loss.clipped_values:
            flag("numerical_integrity", f"显式截断{loss.clipped_values}个值；原值和标记应随输出保留")
        if kind != "format":
            flag("numerical_integrity", "本任务包含结构转换、重采样、融合或适配；不等同无损格式转换")
        if model_check is not None and not model_check.get("complete", False):
            flag("model_compatibility", "模型需求复查未通过/不完整；禁止完整模型输入", True)
        accuracy = {"available": False, "reason": "没有独立真值/验证站，不生成精度成功率"}
        if reference is not None:
            metrics = {}
            for name in after.data_vars:
                if name in reference and after[name].dtype.kind in "biuf":
                    actual, expected = xr.align(after[name], reference[name], join="exact")
                    from ecodata.engines.validation import ValidationEngine
                    metrics[name] = {**asdict(fidelity_metrics(actual, expected)), **ValidationEngine.skill(actual, expected, independent=True)}
            accuracy = {"available": bool(metrics), "variables": metrics}
        stations = self._stations(after, validation_stations) if validation_stations is not None else None
        if stations is not None:
            accuracy = stations
        status = "FAIL" if errors else "WARN" if warnings else "PASS"
        loss.warnings = warnings
        return ValidationReport(status != "FAIL", {key: value != "FAIL" for key, value in levels.items()}, json_safe({
            "status": status, "task_kind": kind, "checks_by_category": levels, "thresholds": asdict(self.thresholds),
            "before_profile": first, "after_profile": last, "source_profiles": [profile_dataset(ds) for ds in sources],
            "information_loss": loss.as_dict(), "numerical_comparison": numeric, "statistical_changes": changes,
            "accuracy": accuracy, "observation_placement": placement, "extrapolated_spatial_cell_count": extrapolation_count, "field_mapping": field_mapping,
            "critical_fields": list(critical_fields), "model_compatibility_check": model_check, "warnings": warnings, "errors": errors}))

    @staticmethod
    def _stations(after, frame):
        if "split" not in frame or not frame["split"].isin(["validation", "test"]).all():
            raise SchemaError("精度计算只接受显式独立validation/test站；训练站不得作为独立精度")
        errors, predictions, references = [], [], []
        for row in frame.itertuples():
            if row.qc_flag != "VALID":
                continue
            field = after[getattr(row, "output_variable", "value")]
            if field.attrs.get("units") != row.unit:
                raise SchemaError("独立验证站单位不匹配")
            if "time" in field.dims:
                if np.datetime64(pd.Timestamp(row.time)) not in field.time.values:
                    continue
                field = field.sel(time=pd.Timestamp(row.time))
            elif "time" in field.coords and pd.Timestamp(field.time.item()) != pd.Timestamp(row.time):
                continue
            if getattr(row, "variable", None) and field.attrs.get("physical_variable") and row.variable != field.attrs["physical_variable"]:
                raise SchemaError("独立验证站物理变量不匹配")
            prediction = field.interp(lat=row.latitude, lon=row.longitude).item()
            if np.isfinite(prediction):
                errors.append(prediction - row.value)
                predictions.append(prediction)
                references.append(row.value)
        if not errors:
            return {"available": False, "reason": "独立站没有有效时空匹配"}
        from ecodata.engines.validation import ValidationEngine
        return {**ValidationEngine.skill(xr.DataArray(predictions, dims="sample"), xr.DataArray(references, dims="sample"), independent=True),
                "independent_station_count": len(predictions)}


def require_usable(report):
    if not report.passed:
        raise ValidationError("自动自查FAIL：" + "；".join(report.details.get("errors", [])), report)


def validation_files(report):
    """Persist/download profiles and report together, without an extra UI action."""
    detail = report.as_dict() if isinstance(report, ValidationReport) else report
    payloads = {"before_profile.json": detail["before_profile"], "after_profile.json": detail["after_profile"], "validation_report.json": detail}
    return {name: json.dumps(json_safe(value), ensure_ascii=False, indent=2, default=str, allow_nan=False).encode("utf-8") for name, value in payloads.items()}
