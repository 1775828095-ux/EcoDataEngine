"""Standardize sources onto an explicitly chosen reference grid before fusion."""
import json
import time

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord
from ecodata.fusion.weighted import UncertaintyWeightedFusion, WeightedFusionConfig
from ecodata.io.structures import spatial_axes, raster_plane
from ecodata.models.adaptation import unit_transform
from ecodata.spatial.regridding import normalize_grid_axes
from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler


def fusion_compatibility(datasets, variable):
    """Read-only preflight: no weights, interpolation or data mutation."""
    problems, steps = [], []
    first = datasets[0]
    unit = first[variable].attrs.get("units", "") if variable in first else ""
    physical = first[variable].attrs.get("physical_variable") if variable in first else None
    for index, ds in enumerate(datasets, 1):
        prefix = f"数据 {index}："
        if variable not in ds or "uncertainty" not in ds:
            problems.append(prefix + "缺少所选变量或 uncertainty")
            continue
        if set(ds["uncertainty"].dims) != set(ds[variable].dims) or not np.isfinite(ds.uncertainty.values).all() or np.any(ds.uncertainty.values <= 0):
            problems.append(prefix + "uncertainty 必须与变量同维度且为有限正数")
        if ds[variable].dtype.kind not in "biuf" or ds[variable].attrs.get("data_kind") == "categorical":
            problems.append(prefix + "加权融合只适用于连续数值物理变量")
        if not physical or ds[variable].attrs.get("physical_variable") != physical:
            problems.append(prefix + "physical_variable 未声明或不一致")
        try:
            if unit == "unknown" or ds[variable].attrs.get("units") == "unknown":
                raise SchemaError("未知单位")
            scale, offset = unit_transform(ds[variable].attrs.get("units", ""), unit)
            if scale != 1 or offset != 0:
                steps.append(prefix + f"单位换算到 {unit}")
        except Exception as exc:
            problems.append(prefix + str(exc))
        if not ds.attrs.get("crs"):
            problems.append(prefix + "缺少 CRS，不能自动猜测")
        else:
            try:
                CRS.from_user_input(ds.attrs["crs"])
                x, y = spatial_axes(ds)
                if not {x, y}.issubset(ds[variable].dims):
                    problems.append(prefix + "当前加权融合需要规则二维网格，不是未网格化站点表")
            except Exception as exc:
                problems.append(prefix + str(exc))
        if "time" in ds[variable].dims and ds[variable].attrs.get("temporal_semantics", ds.attrs.get("temporal_semantics")) != "instantaneous":
            problems.append(prefix + "请先明确瞬时值时间语义；平均/累计量需按时间边界标准化")
    has_time = [variable in ds and "time" in ds[variable].dims for ds in datasets]
    if any(has_time) and not all(has_time):
        problems.append("不能默认融合静态场和动态时间序列")
    elif all(has_time):
        if max(pd.Timestamp(ds.time.min().values) for ds in datasets) > min(pd.Timestamp(ds.time.max().values) for ds in datasets):
            problems.append("各源时间范围没有交集，禁止外推后融合")
    steps.append("共同时间范围内插；按主数据 CRS、范围与目标网格对齐；无有效交集时停止")
    return {"ready": not problems, "reasons": problems, "plan": steps}


def align_and_fuse(datasets, variable, *, frequency=None, resolution=None, method="bilinear"):
    if len(datasets) < 2:
        raise SchemaError("融合至少需要两个数据源")
    if method not in {"nearest", "bilinear"}:
        raise SchemaError("CRS/网格对齐目前支持最近邻和双线性；保守重网格、QP 请先在单文件任务执行")
    preflight = fusion_compatibility(datasets, variable)
    if not preflight["ready"]:
        raise SchemaError("；".join(preflight["reasons"]))
    started = time.perf_counter()
    sources, plan = [], []
    reference = datasets[0]
    unit = reference[variable].attrs.get("units", "") if variable in reference else ""
    physical = reference[variable].attrs.get("physical_variable") if variable in reference else None
    if not unit or unit == "unknown" or not physical:
        raise SchemaError("融合需要明确物理变量 physical_variable 和可靠单位，不能猜测语义")
    for index, dataset in enumerate(datasets):
        if variable not in dataset or "uncertainty" not in dataset:
            raise SchemaError(f"数据 {index+1} 缺少 {variable} 或 uncertainty；不能虚构观测不确定度")
        if dataset[variable].attrs.get("physical_variable") != physical:
            raise SchemaError(f"数据 {index+1} 物理变量语义不兼容")
        if not dataset.attrs.get("crs"):
            raise SchemaError(f"数据 {index+1} 缺少 CRS，无法融合")
        x, y = spatial_axes(dataset)
        ds = dataset.rename({name: target for name, target in ((x, "lon"), (y, "lat")) if name != target})
        ds.attrs = dict(dataset.attrs, spatial_x="lon", spatial_y="lat")
        ds, _ = normalize_grid_axes(ds)
        ds = ds.copy(deep=True)
        if set(ds["uncertainty"].dims) != set(ds[variable].dims):
            raise SchemaError("uncertainty 必须与融合变量具有相同维度")
        if np.any(~np.isfinite(ds.uncertainty.values)) or np.any(ds.uncertainty.values <= 0):
            raise SchemaError("uncertainty 必须为有限正数")
        scale, offset = unit_transform(ds[variable].attrs.get("units", ""), unit)
        from ecodata.engines.units import UnitEngine
        ds[variable] = UnitEngine().convert(ds[variable], unit)
        ds["uncertainty"] = ds.uncertainty*abs(scale)
        ds.uncertainty.attrs.update(units=unit)
        if scale != 1 or offset != 0:
            plan.append({"source": index+1, "operation": "单位换算", "target_unit": unit, "scale": scale, "offset": offset})
        sources.append(ds)
        if "qc_flag" in ds:
            if set(ds.qc_flag.dims) != set(ds[variable].dims):
                raise SchemaError("qc_flag 必须与融合变量同维度；其他标记需要先明确映射")
            flags, counts = np.unique(ds.qc_flag.values.astype(str), return_counts=True)
            if not set(flags).issubset({"VALID", "INVALID", "MISSING"}):
                raise SchemaError("未知 qc_flag 定义，请先映射为 VALID/INVALID/MISSING，不能猜测质控含义")
            ds[variable] = ds[variable].where(ds.qc_flag == "VALID")
            ds[variable].attrs.update(units=unit)
            plan.append({"source": index+1, "operation": "质控筛选，仅 VALID 参与", "original_flags": dict(zip(flags.tolist(), counts.tolist()))})
    has_time = ["time" in ds[variable].dims for ds in sources]
    if any(has_time) and not all(has_time):
        raise SchemaError("不能将静态场与动态时间序列默认融合")
    if all(has_time):
        low = max(pd.Timestamp(ds.time.values.min()) for ds in sources)
        high = min(pd.Timestamp(ds.time.values.max()) for ds in sources)
        if low > high:
            raise SchemaError("各源时间范围没有交集，禁止外推后融合")
        for index, ds in enumerate(sources):
            semantics = ds[variable].attrs.get("temporal_semantics", ds.attrs.get("temporal_semantics"))
            if semantics != "instantaneous":
                raise SchemaError("自动融合对齐仅接受已明确为瞬时值的序列；平均/累计量请先按时间边界标准化")
            if frequency:
                parts = []
                for field in (variable, "uncertainty"):
                    item = ds.copy()
                    item[field].attrs.update(data_kind="continuous", temporal_semantics="instantaneous")
                    parts.append(SemanticTimeResampler(SemanticTimeConfig(frequency, field)).apply(EcoDataBundle(item)).data.dataset[[field]])
                ds = xr.merge(parts)
                ds.attrs = dict(sources[index].attrs)
            sources[index] = ds
        target_time = sources[0].time.sel(time=slice(low, high))
        if not target_time.size:
            raise SchemaError("共同时间范围中没有目标时间点")
        for index, ds in enumerate(sources):
            if not np.array_equal(ds.time.values, target_time.values):
                attrs = {name: dict(ds[name].attrs) for name in (variable, "uncertainty")}
                ds = ds[[variable, "uncertainty"]].interp(time=target_time)
                for name, meta in attrs.items():
                    ds[name].attrs = meta
                plan.append({"source": index+1, "operation": "瞬时值时间插值/共同范围裁剪", "frequency": frequency, "extrapolation": False})
            sources[index] = ds
    target = sources[0]
    if resolution:
        from ecodata.interface.workflow import target_grid
        from ecodata.optimization.downscaling import supported_regrid
        from ecodata.recognition import inspect_dataset
        source_resolution = target.attrs.get("grid_resolution_km") or inspect_dataset("reference.nc", target).spatial_resolution_km
        if not source_resolution:
            raise SchemaError("无法可靠识别参考网格的 km 分辨率，请先确认源分辨率")
        axes = target_grid(target, source_resolution, resolution)
        outputs = [supported_regrid(EcoDataBundle(target), method, axes[0], axes[1], name).dataset[[name]] for name in (variable, "uncertainty")]
        target = xr.merge(outputs)
        target.attrs = dict(sources[0].attrs, grid_resolution_km=resolution)
        sources[0] = target
        plan.append({"source": 1, "operation": "目标网格尺度转换", "resolution_km": resolution, "method": method})
    target_crs = CRS.from_user_input(target.attrs["crs"])
    template = target[[variable]].isel(time=0, drop=True) if all(has_time) else target[[variable]]
    _, target_transform, _ = raster_plane(template, variable)
    target_shape = (target.sizes["lat"], target.sizes["lon"])
    for index, ds in enumerate(sources):
        crs = CRS.from_user_input(ds.attrs["crs"])
        same = crs == target_crs and np.array_equal(ds.lat, target.lat) and np.array_equal(ds.lon, target.lon)
        if same:
            continue
        arrays = {}
        for name in (variable, "uncertainty"):
            frames = []
            count = ds.sizes["time"] if all(has_time) else 1
            for slot in range(count):
                plane_ds = ds[[name]].isel(time=slot, drop=True) if all(has_time) else ds[[name]]
                plane, transform, _ = raster_plane(plane_ds, name)
                output = np.full(target_shape, np.nan)
                reproject(plane, output, src_transform=transform, src_crs=crs,
                          dst_transform=target_transform, dst_crs=target_crs,
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=getattr(Resampling, method))
                frames.append(output[::-1])
            arrays[name] = (("time", "lat", "lon") if all(has_time) else ("lat", "lon"), np.stack(frames) if all(has_time) else frames[0])
        aligned = xr.Dataset(arrays, coords={axis: target[axis] for axis in target[variable].dims}, attrs=dict(target.attrs))
        for name in arrays:
            aligned[name].attrs = dict(ds[name].attrs)
        if not np.isfinite(aligned[variable]).any():
            raise SchemaError(f"数据 {index+1} 与目标空间范围没有有效重叠")
        sources[index] = aligned
        plan.append({"source": index+1, "operation": "CRS/网格/范围对齐", "source_crs": crs.to_string(), "target_crs": target_crs.to_string(), "method": method})
    bundles = []
    for index, ds in enumerate(sources):
        try:
            history = tuple(TransformRecord(**item) for item in json.loads(datasets[index].attrs.get("transform_history", "[]")))
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"数据 {index+1} 转换历史无法解析") from exc
        record = TransformRecord("fusion.source_standardization", {"source": index+1, "steps": [step for step in plan if step.get("source") == index+1]},
                                 int(datasets[index][variable].size), int(ds[variable].size))
        bundles.append(EcoDataBundle(ds, history + (record,)))
    result = UncertaintyWeightedFusion(WeightedFusionConfig(variable=variable)).apply(bundles).data
    if any("qc_flag" in ds for ds in datasets):
        result.dataset["qc_flag"] = xr.where(np.isfinite(result.dataset[variable]), "FUSED_VALID", "MISSING")
        result.dataset.qc_flag.attrs["long_name"] = "derived fusion availability; original source flags recorded in alignment_plan"
    result.dataset.attrs["alignment_plan"] = json.dumps(plan, ensure_ascii=False)
    result.dataset.attrs["transform_history"] = json.dumps([record.as_dict() for record in result.history], ensure_ascii=False)
    result.dataset.attrs["sources"] = json.dumps([dict(ds.attrs) for ds in datasets], default=str, ensure_ascii=False)
    return result, {"steps": plan, "reference": "第一个/主数据网格", "target_unit": unit,
                    "target_CRS": target_crs.to_string(), "time_policy": "共同时间范围内插，不外推",
                    "uncertainty_note": "不确定度同步换算和插值；未估计跨源相关性或重采样协方差", "seconds": time.perf_counter()-started}
