"""Compose existing datasets/CRS/unit handling with the GSTools engine."""
import json
import time
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from ecodata.core.exceptions import SchemaError
from ecodata.engines.fusion import SpatialFusionEngine
from ecodata.engines.units import UnitEngine
from ecodata.fusion.planning import FusionPlanner
from ecodata.io.structures import spatial_axes, table_frame
from ecodata.interface.workflow import TrialResult
from ecodata.recognition import inspect_dataset


def sample_field(dataset, variable, x, y, crs, timestamp=None):
    if not dataset.attrs.get("crs"):
        raise SchemaError(f"{variable} 缺少 CRS，不能猜测坐标系")
    field = dataset[variable]
    if "time" in field.dims:
        if timestamp is None or np.datetime64(timestamp) not in field.time.values:
            raise SchemaError(f"{variable} 没有同一时刻的有效辅助场；请先按时间语义标准化")
        field = field.sel(time=timestamp)
    xaxis, yaxis = spatial_axes(dataset)
    tx, ty = Transformer.from_crs(crs, dataset.attrs["crs"], always_xy=True).transform(x, y)
    sampled = field.interp({xaxis: xr.DataArray(tx, dims="point"), yaxis: xr.DataArray(ty, dims="point")})
    if set(sampled.dims) != {"point"}:
        raise SchemaError(f"{variable} 含额外维度，需先明确层/维度选择")
    return sampled.values


def run_kriging_task(datasets, request):
    started = time.perf_counter()
    variable = request.variable
    plan = FusionPlanner().plan(datasets, variable, request.fusion_method, request.covariates)
    if not plan["ready"]:
        raise SchemaError("；".join(plan["reasons"]))
    target, observations = datasets[plan["target_grid"]], datasets[plan["observations"][0]]
    if request.target_time_frequency or request.target_spatial_resolution_km:
        raise SchemaError("Kriging 先使用上传的目标网格/时间；请先用现有转换任务完成分辨率统一")
    if not target.attrs.get("crs") or not observations.attrs.get("crs"):
        raise SchemaError("训练站/目标网格缺少 CRS")
    xaxis, yaxis = spatial_axes(target)
    xs, ys = np.meshgrid(target[xaxis].values, target[yaxis].values)
    if target[xaxis].ndim != 1 or target[yaxis].ndim != 1:
        raise SchemaError("当前 Kriging 任务输出需要二维规则目标网格")
    info = inspect_dataset("station.csv", observations)
    frame = table_frame(observations)
    if "split" not in frame or not frame["split"].isin(["train", "validation", "test"]).all():
        raise SchemaError("站点必须有明确 split=train/validation/test；不能自动当成训练数据")
    if "qc_flag" not in frame or not frame.qc_flag.isin(["VALID", "INVALID", "MISSING"]).all():
        raise SchemaError("站点需要明确 qc_flag=VALID/INVALID/MISSING，不能猜测质控")
    train = frame[(frame.split == "train") & (frame.qc_flag == "VALID")].copy()
    held = frame[frame.split.isin(["validation", "test"])]
    if "station_id" not in frame:
        raise SchemaError("需要 station_id 检查训练与独立验证隔离")
    if set(train.station_id) & set(held.station_id):
        raise SchemaError("训练/验证 station_id 重叠，禁止数据泄漏")
    if set(zip(train[info.x_coordinate], train[info.y_coordinate])) & set(zip(held[info.x_coordinate], held[info.y_coordinate])):
        raise SchemaError("训练/验证站位置重叠，不能称为独立空间验证")
    value_name = variable if variable in train else "value"
    attrs = dict(target[variable].attrs) if variable in target else dict(observations[value_name].attrs)
    unit = attrs.get("units", "")
    if value_name == "value" and ("variable" not in train or not (train.variable == attrs.get("physical_variable", variable)).all()):
        raise SchemaError("站点目标变量语义与背景场不匹配")
    if variable in observations and attrs.get("physical_variable", variable) != observations[variable].attrs.get("physical_variable", variable):
        raise SchemaError("训练观测与背景场不是同一物理变量")
    dynamic = "time" in target.dims
    if not dynamic and "time" in train and pd.to_datetime(train.time).nunique() > 1:
        raise SchemaError("静态目标网格不能混合多个观测时刻；请先按时间语义聚合或提供带 time 的目标网格")
    times = target.time.values if dynamic else [None]
    if dynamic and ("time" not in train or attrs.get("temporal_semantics", target.attrs.get("temporal_semantics")) != "instantaneous"):
        raise SchemaError("动态 Kriging 需要显式瞬时值时间语义和同一时刻站点，平均/累计值请先标准化")
    target_crs = CRS.from_user_input(target.attrs["crs"])
    metric_crs = target_crs
    if target_crs.is_geographic:
        metric_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={float(ys.mean())} +lon_0={float(xs.mean())} +datum=WGS84 +units=m")
    project = Transformer.from_crs(target_crs, metric_crs, always_xy=True)
    mx, my = project.transform(xs.ravel(), ys.ravel())
    fields, variances, diagnostics = [], [], []
    covariate_sources = []
    for name in request.covariates:
        matches = [ds for ds in datasets if name in ds and ds[name].dtype.kind in "biuf" and name != variable]
        if len(matches) != 1 or inspect_dataset("auxiliary.nc", matches[0]).semantic_types.get(name) != "continuous":
            raise SchemaError(f"辅助字段 {name} 需要唯一、连续、非分类数值场；类别编码不能当连续 drift")
        if matches[0].attrs.get("data_role") == "quality_auxiliary":
            raise SchemaError(f"{name} 已声明为质控辅助数据，不能自动当作物理 external drift")
        UnitEngine().parse(matches[0][name].attrs.get("units", ""))
        covariate_sources.append((name, matches[0]))
    for timestamp in times:
        rows = train if timestamp is None else train[pd.to_datetime(train.time) == pd.Timestamp(timestamp)]
        ox, oy = rows[info.x_coordinate].to_numpy(float), rows[info.y_coordinate].to_numpy(float)
        px, py = Transformer.from_crs(observations.attrs["crs"], metric_crs, always_xy=True).transform(ox, oy)
        values = rows[value_name].to_numpy(float).copy()
        source_unit = observations[value_name].attrs.get("units", "")
        if "unit" in rows and len(rows.unit.unique()) > 1:
            raise SchemaError("同一批站点含不同单位，请先显式统一；禁止把数值直接混用")
        if "unit" in rows and len(rows.unit.unique()) == 1:
            source_unit = rows.unit.iloc[0]
        values = UnitEngine().convert(xr.DataArray(values, attrs={"units": source_unit}), unit).values
        drift, target_drift = [], []
        for name, ds in covariate_sources:
            drift.append(sample_field(ds, name, ox, oy, observations.attrs["crs"], timestamp))
            target_drift.append(sample_field(ds, name, xs.ravel(), ys.ravel(), target_crs, timestamp))
        field, variance, detail, _ = SpatialFusionEngine().estimate([px, py], values, [mx, my], method=plan["method"],
            drift=drift or None, target_drift=target_drift or None, split=rows.split.values)
        fields.append(field.reshape(xs.shape)); variances.append(variance.reshape(xs.shape))
        detail.update(time=str(timestamp), metric_crs=metric_crs.to_wkt(), covariates=[name for name, _ in covariate_sources])
        detail["covariate_units"] = {name: ds[name].attrs["units"] for name, ds in covariate_sources}
        diagnostics.append(detail)
    output = target.copy(deep=True)
    shape = (len(times), *xs.shape) if dynamic else xs.shape
    dims = ("time", yaxis, xaxis) if len(shape) == 3 else (yaxis, xaxis)
    if variable in target:
        output[f"original_background_{variable}"] = target[variable]
    output[variable] = xr.DataArray(np.asarray(fields).reshape(shape), dims=dims, attrs=attrs)
    output["kriging_variance"] = xr.DataArray(np.asarray(variances).reshape(shape), dims=dims, attrs={"units": f"({unit})^2", "long_name": "GSTools kriging estimation variance (not independent accuracy)"})
    output.attrs.update(fusion_method=plan["method"], kriging_diagnostics=json.dumps(diagnostics), processing_stage="fused")
    if not dynamic and "time" in train:
        output.attrs["observation_time"] = str(pd.to_datetime(train.time).iloc[0])
    history = ({"method": "fusion.GSTools."+plan["method"], "parameters": {"diagnostics": diagnostics,
                "covariates": list(request.covariates), "independent_validation_used_for_fit": False},
                "input_rows": int(len(train)), "output_rows": int(output[variable].size)},)
    result = TrialResult(target, output, history, {"fusion_plan": plan, "GSTools": diagnostics,
        "精度指标可用": False, "精度说明": "训练站只用于估计；独立验证另行计算，kriging_variance 不是精度评分",
        "辅助数据说明": "External Drift 进入 GSTools 约束方程，不与目标值直接相加"}, time.perf_counter()-started)
    predictions, references = [], []
    for timestamp in times:
        rows = held[held.qc_flag == "VALID"]
        if timestamp is not None:
            rows = rows[pd.to_datetime(rows.time) == pd.Timestamp(timestamp)]
        if rows.empty:
            continue
        prediction = sample_field(output, variable, rows[info.x_coordinate].values, rows[info.y_coordinate].values,
                                  observations.attrs["crs"], timestamp)
        source_unit = observations[value_name].attrs.get("units", "")
        if "unit" in rows and len(rows.unit.unique()) == 1:
            source_unit = rows.unit.iloc[0]
        expected = UnitEngine().convert(xr.DataArray(rows[value_name].values, attrs={"units": source_unit}), unit).values
        predictions.extend(prediction); references.extend(expected)
    if predictions:
        from ecodata.engines.validation import ValidationEngine
        score = ValidationEngine.skill(xr.DataArray(predictions, dims="station"), xr.DataArray(references, dims="station"), independent=True)
        result.report["独立验证"] = dict(score, independent_station_count=score.get("sample_count", 0))
        result.report["精度指标可用"] = score["available"]
        result.report["精度说明"] = "仅明确独立 validation/test 站用于评估；没有参与拟合或可靠性筛选"
    return result
