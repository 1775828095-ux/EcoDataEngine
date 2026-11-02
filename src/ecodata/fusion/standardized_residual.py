"""Validate and prepare station inputs for the existing P2 residual algorithm."""
import time
import json
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord
from ecodata.fusion.residual import ResidualFieldFusion, ResidualFusionConfig
from ecodata.io.structures import spatial_axes, table_frame
from ecodata.io.readers import _frame_to_dataset, REQUIRED_COLUMNS
from ecodata.models.adaptation import unit_transform
from ecodata.spatial.regridding import normalize_grid_axes


def run_residual_task(datasets, request):
    from ecodata.interface.workflow import TrialResult, _report
    if len(datasets) != 2:
        raise SchemaError("背景场+监测站残差融合要求恰好两个文件，主数据为背景网格")
    if request.target_time_frequency or request.target_spatial_resolution_km:
        raise SchemaError("当前残差融合接受已对齐的单时间点背景场；请先完成时空转换，再融合，不能忽略所选处理设置")
    bg, station = datasets
    if not bg.attrs.get("crs") or not station.attrs.get("crs"):
        raise SchemaError("背景网格和监测站都必须声明 CRS")
    if CRS.from_user_input(bg.attrs["crs"]) != CRS.from_epsg(4326) or CRS.from_user_input(station.attrs["crs"]) != CRS.from_epsg(4326):
        raise SchemaError("当前 P2 残差融合使用经纬度，请先将两源转换为 EPSG:4326")
    frame = table_frame(station)
    if missing := set(REQUIRED_COLUMNS)-set(frame):
        raise SchemaError(f"监测站缺少字段：{sorted(missing)}")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    for name in ("value", "latitude", "longitude"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame.loc[frame.qc_flag.eq("VALID"), name]).all():
            raise SchemaError(f"VALID 站点的 {name} 包含无效值，请先完成质控")
    if "time" not in bg.coords or bg.time.size != 1 or frame.time.nunique() != 1 or pd.Timestamp(bg.time.values.ravel()[0]) != frame.time.iloc[0]:
        raise SchemaError("当前残差融合只接受同一明确时间点的二维背景场和监测站，不能跨时间混合")
    variable = request.variable or "value"
    if variable not in bg:
        raise SchemaError("背景场缺少所选变量")
    if not bg[variable].attrs.get("physical_variable"):
        raise SchemaError("背景场未声明 physical_variable，不能猜测它与站点变量相同")
    target_unit = bg[variable].attrs.get("units", "")
    if not target_unit or target_unit == "unknown":
        raise SchemaError("背景场单位未知，不能进行残差融合")
    steps = []
    for source_unit in frame.unit.unique():
        scale, offset = unit_transform(source_unit, target_unit)
        chosen = frame.unit.eq(source_unit)
        frame.loc[chosen, "value"] = pd.to_numeric(frame.loc[chosen, "value"])*scale+offset
        frame.loc[chosen, "unit"] = target_unit
        steps.append({"operation": "站点单位检查/换算", "source_unit": source_unit, "target_unit": target_unit})
    x, y = spatial_axes(bg)
    prepared = bg.rename({name: target for name, target in ((x, "lon"), (y, "lat")) if name != target})
    prepared.attrs = dict(bg.attrs, spatial_x="lon", spatial_y="lat")
    if "time" in prepared.dims:
        prepared = prepared.isel(time=0)
    prepared, _ = normalize_grid_axes(prepared)
    started = time.perf_counter()
    method = request.spatial_method if request.spatial_method in {"linear", "nearest"} else "linear"
    try:
        histories = [tuple(TransformRecord(**item) for item in json.loads(ds.attrs.get("transform_history", "[]"))) for ds in datasets]
    except (TypeError, ValueError) as exc:
        raise SchemaError("输入转换历史无法解析") from exc
    result = ResidualFieldFusion(ResidualFusionConfig(variable, method)).apply(EcoDataBundle(prepared, histories[0]), EcoDataBundle(_frame_to_dataset(frame), histories[1])).data
    history = tuple(record.as_dict() for record in result.history)
    result.dataset.attrs.update(export_variable=variable, transform_history=json.dumps(history, ensure_ascii=False),
                                source_station_metadata=json.dumps(dict(station.attrs), default=str, ensure_ascii=False))
    report = _report(bg, result.dataset, steps, history, time.perf_counter()-started, reference=None)
    report["标准化与融合"] = {"steps": steps, "method": "existing_P2_residual", "time_policy": "同一时间点",
                                "quality_filter": "仅 VALID 且在背景域内的站点"}
    report["目标输出格式"] = request.output_format
    return TrialResult(bg, result.dataset, history, report, time.perf_counter()-started)
