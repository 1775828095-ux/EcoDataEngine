"""CMAQ PM2.5 auxiliary adapter; NOT a native I/O API exporter."""
import json
import time

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform, transform_bounds

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.output_quality import validate_export
from ecodata.io.structures import export_structure, raster_plane, spatial_axes, table_dataset, table_frame, StructureRequest
from ecodata.interface.workflow import TrialRequest
from ecodata.models.adaptation import unit_transform
from ecodata.optimization.downscaling import supported_regrid
from ecodata.recognition import inspect_dataset
from ecodata.task import ConversionTaskEngine
from ecodata.models.contracts import ModelSchema, ScenarioResult
from ecodata.models.adapters.standardized import ModelAdapter


def scenario_schema(profile):
    if not profile.get("preparation_scenario"):
        raise SchemaError("该模型尚未配置工程准备场景，请使用已有数据转换；不假装支持原生适配")
    return ModelSchema.from_profile(profile).preparation


def _field_mapping(dataset, role):
    mapping = {}
    for field in role["fields"]:
        names = {str(name).casefold() for name in [field["name"], *field.get("aliases", [])]}
        candidates = [name for name in dataset.data_vars if name.casefold() in names]
        if len(candidates) == 1:
            mapping[field["name"]] = candidates[0]
    wind = next((field for field in role["fields"] if field["name"] == "wind"), {})
    if wind.get("allow_derived") and "wind" not in mapping and {"u10m", "v10m"} <= set(dataset):
        mapping["wind"] = "derived:u10m,v10m"
    return mapping


def _canonical_unit(unit):
    return {"UG/M3": "ug/m3", "ug m-3": "ug/m3", "µg/m3": "ug/m3", "μg/m3": "ug/m3",
            "m s-1": "m/s", "m s^-1": "m/s", "degrees_Celsius": "degC", "percent": "%"}.get(unit, unit)


def _convert(field, target):
    actual = _canonical_unit(field.attrs.get("units", ""))
    from ecodata.engines.units import UnitEngine
    source = field.copy()
    source.attrs = dict(field.attrs, units=actual)
    return UnitEngine().convert(source, target)


def plan_scenario(profile, inputs, assignments):
    schema = scenario_schema(profile)
    by_name = {item.filename: item for item in inputs}
    analysis, matches, errors = [], [], []
    for item in inputs:
        info = inspect_dataset(item.filename, item.dataset).as_dict()
        info.update(sha256=item.sha256, source_metadata=dict(item.dataset.attrs))
        if info["time_coordinate"]:
            times = pd.to_datetime(item.dataset[info["time_coordinate"]].values.ravel())
            info["time_range"] = [str(times.min()), str(times.max())]
        analysis.append(info)
    selected_files = [name for role, name in assignments.items() if role != "boundary" and name]
    for role in schema["inputs"]:
        name = assignments.get(role["role"], "")
        match = {"role": role["role"], "name": role["name"], "file": name, "can_process": False}
        if role["role"] == "boundary":
            match.update(status="原生依赖，本辅助场景不生成", required_for_native=True)
        elif name not in by_name:
            match.update(status="缺少", reason="请上传并确认对应数据类别")
            if role["required"]:
                errors.append(f"缺少{role['name']}")
        else:
            ds = by_name[name].dataset
            mapping = _field_mapping(ds, role)
            issues = []
            for field in role["fields"]:
                source = mapping.get(field["name"])
                if not source:
                    issues.append(f"缺少或歧义字段：{field['name']}")
                    continue
                try:
                    if source.startswith("derived:"):
                        for component in ("u10m", "v10m"):
                            _convert(ds[component], "m/s")
                    else:
                        if ds[source].dtype.kind not in "biuf":
                            raise SchemaError("非数值字段不能作为连续物理变量")
                        _convert(ds[source], field["unit"])
                except Exception as exc:
                    issues.append(f"{field['name']}：{exc}")
            info = inspect_dataset(name, ds)
            if not info.crs or not (info.x_coordinate and info.y_coordinate):
                issues.append("缺少可靠CRS或空间坐标，需用户确认，不能猜测")
            elif role["role"] != "observation":
                for axis in (info.x_coordinate, info.y_coordinate):
                    coordinates = ds[axis].values
                    if coordinates.ndim != 1 or len(coordinates) < 2 or not np.isfinite(coordinates).all() or not (np.all(np.diff(coordinates) > 0) or np.all(np.diff(coordinates) < 0)):
                        issues.append(f"{axis}需至少两个严格单调的有限规则网格坐标")
            if role["role"] != "terrain" and not info.time_coordinate:
                issues.append("缺少可靠时间字段")
            if role["role"] in {"background", "meteorology"}:
                reference_semantics = None
                for variable in mapping.values():
                    variables = ("u10m", "v10m") if variable.startswith("derived:") else (variable,)
                    for variable in variables:
                        try:
                            semantics = _time_semantics(ds, variable)
                            if reference_semantics is not None and semantics != reference_semantics:
                                issues.append(f"{variable}时间语义与同一角色其他变量不同")
                            reference_semantics = semantics
                            expected_dims = {info.time_coordinate, info.x_coordinate, info.y_coordinate}
                            if set(ds[variable].dims) != expected_dims:
                                issues.append(f"{variable}需time×y×x；不能静默选择或广播额外垂直层")
                        except SchemaError as exc:
                            issues.append(str(exc))
                variable = mapping.get("PM25" if role["role"] == "background" else "temperature")
                if variable:
                    try:
                        _time_semantics(ds, variable)
                        expected_dims = {info.time_coordinate, info.x_coordinate, info.y_coordinate}
                        if set(ds[variable].dims) != expected_dims:
                            issues.append("本辅助场景仅处理time×y×x规则场；不能静默丢弃垂直层")
                    except SchemaError as exc:
                        issues.append(str(exc))
            if role["role"] == "observation" and ds.attrs.get("temporal_semantics") not in {"mean", "instantaneous"}:
                issues.append("站点时间语义需用户明确确认mean/instantaneous")
            if role["role"] == "observation" and "unit" in ds and mapping.get("PM25"):
                declared = _canonical_unit(ds[mapping["PM25"]].attrs.get("units", ""))
                row_units = {_canonical_unit(str(value).strip()) for value in ds.unit.values}
                if row_units != {declared}:
                    issues.append("观测单位列与变量单位不一致或含混合单位，请先标准化，不能统一标签而不换算数值")
            if role["role"] in {"background", "observation"} and mapping.get("PM25"):
                field = ds[mapping["PM25"]]
                declared_physical = field.attrs.get("physical_variable")
                if declared_physical and declared_physical not in {"PM2.5", "PM25"}:
                    issues.append("物理变量声明不是PM2.5，不能按同单位强行融合")
                if role["role"] == "observation" and "variable" in ds and not set(ds.variable.values.astype(str)) <= {"PM2.5", "PM25"}:
                    issues.append("观测variable列不是单一PM2.5，不能覆盖原物理变量标签")
                if role["role"] == "observation" and mapping["PM25"] == "value" and not declared_physical and "variable" not in ds:
                    issues.append("通用value列没有物理变量声明，请明确标注PM25而不是猜测")
            if role["role"] == "terrain" and mapping.get("elevation") and set(ds[mapping["elevation"]].dims) != {info.x_coordinate, info.y_coordinate}:
                issues.append("地形需二维规则场，不能静默选择波段或垂直层")
            if role["role"] == "observation" and "qc_flag" not in ds:
                confirmations = json.loads(ds.attrs.get("user_confirmations", "{}"))
                if not confirmations.get("observations_valid"):
                    issues.append("缺少质量标记，请明确确认哪些观测有效")
            match.update(mapping=mapping, can_process=not issues, status="可以转换/标准化" if not issues else "当前无法处理", reasons=issues)
            if issues:
                errors.extend(f"{role['name']}：{issue}" for issue in issues)
        matches.append(match)
    compatibility = {}
    if not errors:
        background = by_name[assignments["background"]].dataset
        meteorology = by_name[assignments["meteorology"]].dataset
        observations = by_name[assignments["observation"]].dataset
        infos = {role: inspect_dataset(assignments[role], by_name[assignments[role]].dataset) for role in ("background", "meteorology", "observation")}
        try:
            if CRS.from_user_input(background.attrs["crs"]) != CRS.from_epsg(4326):
                errors.append("背景目标网格需先转为EPSG:4326；本阶段不猜测目标经纬度网格")
            times = {role: pd.to_datetime(by_name[assignments[role]].dataset[infos[role].time_coordinate].values.ravel()).to_numpy() for role in infos}
            for role, values in times.items():
                if pd.isna(values).any() or (role != "observation" and len(values) != len(np.unique(values))):
                    errors.append(f"{role}存在空时间或重复网格时间，请先明确处理")
            common = np.intersect1d(np.intersect1d(times["background"], times["meteorology"]), times["observation"])
            compatibility["common_times"] = [str(value) for value in common]
            if not len(common):
                errors.append("三类动态数据没有共同真实时间，不能生成融合计划")
            else:
                bg_role = next(item for item in schema["inputs"] if item["role"] == "background")
                bg_var = _field_mapping(background, bg_role)["PM25"]
                met_role = next(item for item in schema["inputs"] if item["role"] == "meteorology")
                met_var = _field_mapping(meteorology, met_role)["temperature"]
                sem = _time_semantics(background, bg_var)
                if sem != _time_semantics(meteorology, met_var) or sem != observations.attrs.get("temporal_semantics"):
                    errors.append("背景、气象和观测时间语义不一致")
                if sem == "mean":
                    a = background.sel({infos["background"].time_coordinate: common}).time_bnds
                    b = meteorology.sel({infos["meteorology"].time_coordinate: common}).time_bnds
                    if not np.array_equal(a.values, b.values):
                        errors.append("背景和气象平均时间窗口不一致")
                    if not {"time_start", "time_end"} <= set(observations):
                        errors.append("平均值观测缺少time_start/time_end")
            target_info = infos["background"]
            target_box = [float(background[target_info.x_coordinate].min()), float(background[target_info.y_coordinate].min()),
                          float(background[target_info.x_coordinate].max()), float(background[target_info.y_coordinate].max())]
            compatibility["target_center_extent"] = target_box
            for role in ("meteorology", "terrain"):
                source_info = inspect_dataset(assignments[role], by_name[assignments[role]].dataset)
                extent = transform_bounds(source_info.crs, "EPSG:4326", *source_info.extent)
                compatibility[f"{role}_extent_in_target_crs"] = extent
                if not (extent[0] <= target_box[0] and extent[1] <= target_box[1] and extent[2] >= target_box[2] and extent[3] >= target_box[3]):
                    errors.append(f"{role}空间范围未覆盖目标网格；不能静默外推或填补")
        except (KeyError, ValueError, SchemaError) as exc:
            errors.append(f"时空兼容性检查失败：{exc}")
    steps = [
        {"stage": "时间统一", "operation": "选取背景、气象、有效站点共同时间；检查平均窗口", "no_invention": "不补造缺测时刻"},
        {"stage": "空间/坐标统一", "operation": "以背景网格为目标，气象重网格、DEM/站点重投影至EPSG:4326"},
        {"stage": "单位统一", "operation": "PM25 ug/m3；temperature degC；wind m/s；humidity %；elevation m"},
        {"stage": "融合", "operation": "每个实际时段调用已有Pipeline背景+站点残差融合；保留背景和原始融合"},
        {"stage": "模型准备输出", "operation": "生成CF风格辅助NetCDF及质量标记，不声称I/O API原生输入"},
    ]
    return analysis, {"model": profile["name"], "version": profile["version"], "scenario": schema["id"],
                      "scope": "auxiliary_preparation", "can_process": not errors, "assignments": assignments,
                      "matches": matches, "errors": errors, "steps": steps, "compatibility": compatibility,
                      "unassigned_files": sorted(set(by_name)-set(selected_files)), "native_model_input": False,
                      "native_gaps": schema["native_gaps"]}


def _grid(ds, mapping):
    x, y = spatial_axes(ds)
    rename = {name: target for name, target in ((x, "lon"), (y, "lat")) if name != target}
    time_coordinate = inspect_dataset("grid.nc", ds).time_coordinate
    if time_coordinate and time_coordinate != "time":
        rename[time_coordinate] = "time"
    grid = ds.rename(rename).sortby("lat").sortby("lon")
    result = xr.Dataset(coords={name: grid[name] for name in ("time", "lat", "lon") if name in grid})
    for target, source in mapping.items():
        if not source.startswith("derived:"):
            result[target] = grid[source]
    if "time_bnds" in grid:
        result["time_bnds"] = grid.time_bnds
    result.attrs.update(grid.attrs, spatial_x="lon", spatial_y="lat")
    return result


def _on_target(source, field, target, method):
    if CRS.from_user_input(source.attrs["crs"]) == CRS.from_epsg(4326):
        value = source[field].copy()
        value.attrs["data_kind"] = "continuous"
        prepared = value.to_dataset(name=field)
        prepared.attrs.update(source.attrs)
        return supported_regrid(EcoDataBundle(prepared), method, tuple(target.lat.values), tuple(target.lon.values), field).dataset[field]
    _, target_transform, target_crs = raster_plane(target.isel(time=0, drop=True), "PM25")
    parts = []
    for index in range(source.sizes.get("time", 1)):
        plane = source.isel(time=index, drop=True) if "time" in source.dims else source
        values, affine, crs = raster_plane(plane, field)
        output = np.full((target.lat.size, target.lon.size), np.nan)
        if method not in {"nearest", "bilinear"}:
            raise SchemaError("跨CRS重投影目前使用nearest/bilinear；不能冒充保守重网格")
        reproject(values, output, src_transform=affine, src_crs=crs, src_nodata=np.nan,
                  dst_transform=target_transform, dst_crs=target_crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest if method == "nearest" else Resampling.bilinear)
        parts.append(output[::-1])
    dims = ("time", "lat", "lon") if "time" in source.dims else ("lat", "lon")
    coords = {name: source[name] if name == "time" else target[name] for name in dims}
    return xr.DataArray(np.stack(parts) if "time" in source.dims else parts[0], dims=dims, coords=coords, attrs=source[field].attrs)


def _time_semantics(ds, variable):
    value = ds[variable].attrs.get("temporal_semantics", ds.attrs.get("temporal_semantics"))
    if value not in {"mean", "instantaneous"}:
        raise SchemaError(f"{variable} 时间语义未确认或不支持；请明确mean/instantaneous")
    if value == "mean" and "time_bnds" not in ds:
        raise SchemaError(f"{variable} 平均值缺少time_bnds，不能猜测测量窗口")
    return value


def execute_scenario(profile, inputs, assignments, *, spatial_method="bilinear", residual_method="nearest", negative_policy="clip"):
    started = time.perf_counter()
    if spatial_method not in {"nearest", "bilinear", "conservative"} or residual_method not in {"nearest", "linear"} or negative_policy not in {"clip", "reject"}:
        raise SchemaError("场景算法或负值策略不支持")
    analysis, plan = plan_scenario(profile, inputs, assignments)
    if not plan["can_process"]:
        raise SchemaError("；".join(plan["errors"]))
    by_name = {item.filename: item.dataset for item in inputs}
    matches = {item["role"]: item for item in plan["matches"]}
    grids = {role: _grid(by_name[assignments[role]], matches[role]["mapping"]) for role in ("background", "meteorology", "terrain")}
    bg, met, dem = grids["background"], grids["meteorology"], grids["terrain"]
    if CRS.from_user_input(bg.attrs["crs"]) != CRS.from_epsg(4326):
        raise SchemaError("背景目标网格本阶段需EPSG:4326；请先用已有转换生成经纬度目标网格")
    for name, unit in {"PM25": "ug/m3"}.items():
        bg[name] = _convert(bg[name], unit)
    physical = bg.PM25.attrs.get("physical_variable")
    if physical and physical not in {"PM2.5", "PM25"}:
        raise SchemaError("背景声明的物理变量不是PM2.5")
    bg.PM25.attrs["physical_variable"] = "PM2.5"
    frame = table_frame(by_name[assignments["observation"]]).copy()
    observation_ds = by_name[assignments["observation"]]
    info = inspect_dataset(assignments["observation"], observation_ds)
    frame = frame.rename(columns={info.time_coordinate: "time", info.x_coordinate: "longitude", info.y_coordinate: "latitude"})
    frame.time = pd.to_datetime(frame.time)
    station_field = matches["observation"]["mapping"]["PM25"]
    station_values = _convert(observation_ds[station_field], "ug/m3").values
    if station_values.size != len(frame):
        raise SchemaError("观测必须是单记录维度的站点表格")
    frame["value"] = station_values
    frame["variable"], frame["unit"] = "PM2.5", "ug/m3"
    if "station_id" not in frame:
        frame["station_id"] = [f"input_record_{i}" for i in range(len(frame))]
    if "qc_flag" not in frame:
        frame["qc_flag"] = "VALID"  # Explicit user confirmation already checked.
    source_crs = CRS.from_user_input(observation_ds.attrs["crs"])
    if source_crs != CRS.from_epsg(4326):
        lon, lat = transform(source_crs, "EPSG:4326", frame.longitude.to_numpy(), frame.latitude.to_numpy())
        frame["longitude"], frame["latitude"] = lon, lat
    common = np.intersect1d(bg.time.values, met.time.values)
    common = np.intersect1d(common, pd.to_datetime(frame.loc[frame.qc_flag.eq("VALID"), "time"]).to_numpy())
    if not len(common):
        raise SchemaError("背景、气象与有效观测没有共同真实时间；不补造时刻")
    bg, met = bg.sel(time=common), met.sel(time=common)
    semantics = _time_semantics(bg, "PM25")
    if _time_semantics(met, "temperature") != semantics:
        raise SchemaError("背景与气象时间语义不同，不能直接匹配")
    if semantics == "mean":
        if observation_ds.attrs.get("temporal_semantics") != "mean":
            raise SchemaError("背景为平均值，站点时间语义不同，不能直接融合")
        if not np.array_equal(bg.time_bnds.values, met.time_bnds.values):
            raise SchemaError("背景和气象平均时间窗口不一致")
        if not {"time_start", "time_end"} <= set(frame):
            raise SchemaError("平均值站点需time_start/time_end，不能从采样间隔猜测平均窗口")
        for i, center in enumerate(common):
            rows = frame.loc[frame.time.eq(center)]
            if not pd.to_datetime(rows.time_start).eq(pd.Timestamp(bg.time_bnds.values[i, 0])).all() or not pd.to_datetime(rows.time_end).eq(pd.Timestamp(bg.time_bnds.values[i, 1])).all():
                raise SchemaError("站点与背景平均时间窗口不一致")
    else:
        confirmed = json.loads(observation_ds.attrs.get("user_confirmations", "{}"))
        if observation_ds.attrs.get("temporal_semantics") != "instantaneous" and confirmed.get("temporal_semantics") != "instantaneous":
            raise SchemaError("站点瞬时值语义需明确确认")
    output = xr.Dataset()
    met_source = by_name[assignments["meteorology"]]
    for field, unit in (("temperature", "degC"), ("humidity", "%")):
        met[field] = _convert(met[field], unit)
        output[field] = _on_target(met, field, bg, spatial_method)
    if matches["meteorology"]["mapping"].get("wind", "").startswith("derived:"):
        components = _grid(met_source, {"u10m": "u10m", "v10m": "v10m"}).sel(time=common)
        for field in ("u10m", "v10m"):
            components[field] = _convert(components[field], "m/s")
            output[field] = _on_target(components, field, bg, spatial_method)
        output["wind"] = np.hypot(output.u10m, output.v10m)
        output.wind.attrs.update(units="m/s", derivation="magnitude of aligned source wind vector; not mean wind speed")
    else:
        met["wind"] = _convert(met.wind, "m/s")
        output["wind"] = _on_target(met, "wind", bg, spatial_method)
    dem["elevation"] = _convert(dem.elevation, "m")
    output["elevation"] = _on_target(dem, "elevation", bg, "bilinear" if spatial_method == "conservative" else spatial_method)
    engine, fused, histories, fusion_checks = ConversionTaskEngine(), [], [], []
    for center in common:
        rows = frame.loc[frame.time.eq(center)].copy()
        result = engine.execute([bg.sel(time=[center]).drop_vars("time_bnds", errors="ignore"), table_dataset(rows, crs="EPSG:4326")],
                                StructureRequest(variable="PM25"), TrialRequest(variable="PM25", spatial_method=residual_method, fusion_method="residual"), fuse=True)
        fused.append(result.output.drop_vars("time", errors="ignore").expand_dims(time=[center]))
        histories.append({"time": str(center), "history": result.history})
        fusion_checks.append({"time": str(center), "validation_report": result.report["validation_report"]})
    fused = xr.concat(fused, dim="time")
    raw = fused.PM25
    negative_count = int((raw < 0).sum())
    if negative_count and negative_policy == "reject":
        raise SchemaError(f"融合产生{negative_count}个负浓度像元，按所选策略阻止模型准备输出；未修改原始数据")
    output["PM25"] = raw.clip(min=0) if negative_policy == "clip" else raw
    output.PM25.attrs.update(raw.attrs, negative_policy=negative_policy)
    output["PM25_background"] = fused.background_value
    output["PM25_raw_fused"] = raw
    output["PM25_bound_flag"] = (raw < 0).astype(np.uint8)
    output.PM25_bound_flag.attrs.update(units="1", flag_values=np.array([0, 1], dtype=np.uint8), flag_meanings="unchanged clipped_to_zero")
    fused["qc_flag"] = (raw < 0).astype(np.uint8)
    fused.qc_flag.attrs.update(units="1", flag_meanings="nonnegative negative_unbounded_result", flag_values=np.array([0, 1], dtype=np.uint8))
    for ds in (output, fused):
        if "time_bnds" in bg:
            ds["time_bnds"] = bg.time_bnds
            ds.time.attrs["bounds"] = "time_bnds"
            ds.time_bnds.encoding.update(units="minutes since 1970-01-01", calendar="proleptic_gregorian")
        ds.time.encoding.update(units="minutes since 1970-01-01", calendar="proleptic_gregorian")
        ds.lat.attrs.update(units="degrees_north", standard_name="latitude")
        ds.lon.attrs.update(units="degrees_east", standard_name="longitude")
        ds.attrs.update(crs="EPSG:4326", spatial_x="lon", spatial_y="lat", Conventions="CF-1.8",
                        output_scope="CMAQ auxiliary preparation; NOT native I/O API input",
                        native_model_input="false", temporal_semantics=semantics,
                        source_metadata=json.dumps(analysis, ensure_ascii=False, default=str),
                        processing_history=json.dumps(histories, ensure_ascii=False, default=str))
    field_quality = {}
    expected_units = {"PM25": "ug/m3", "temperature": "degC", "wind": "m/s", "humidity": "%", "elevation": "m"}
    for field, unit in expected_units.items():
        values = output[field].values
        if not np.isfinite(values).all() or output[field].attrs.get("units") != unit:
            raise SchemaError(f"{field}存在缺失/无效值或单位不匹配，不能输出合格候选文件")
        field_quality[field] = {"dtype": str(values.dtype), "unit": unit, "dims": output[field].dims,
                                "nan_ratio": float(np.isnan(values).mean()), "min": float(values.min()),
                                "max": float(values.max()), "negative_count": int((values < 0).sum())}
    if float(output.humidity.min()) < 0 or float(output.humidity.max()) > 100 or float(output.wind.min()) < 0:
        raise SchemaError("湿度或风速超过明确物理范围")
    quality = {"passed": True, "native_model_input": False, "scope": "auxiliary_standardization_only", "fields": field_quality,
               "fusion_validation": fusion_checks,
               "required_variables": list(expected_units), "missing_variables": sorted(set(expected_units)-set(output)),
               "crs": output.attrs["crs"], "dimensions": dict(output.sizes),
               "time_range": [str(output.time.values[0]), str(output.time.values[-1])],
               "negative_fused_cells": negative_count, "negative_policy": negative_policy,
               "maximum_pm25_correction": float(max(0, -raw.min().item())) if negative_policy == "clip" else 0.,
               "warnings": ["原始负浓度保留；截断不证明融合误差降低", "地形/气象仅配套对齐，未参与融合权重或复杂同化", "无独立真值，不报告MAE/RMSE", "背景小时平均场不能冒充三维瞬时初始场"],
               "input_dimensions": {item.filename: dict(item.dataset.sizes) for item in inputs}}
    plan.update(algorithm_options={"spatial_method": spatial_method, "residual_method": residual_method, "negative_policy": negative_policy,
                                   "terrain_spatial_method": "bilinear" if spatial_method == "conservative" else spatial_method},
                common_times=[str(value) for value in common], original_time_counts={"background": grids["background"].time.size, "meteorology": grids["meteorology"].time.size},
                boundary_supplied=bool(assignments.get("boundary")), boundary_processed=False)
    files = {}
    for filename, ds in (("model_input.nc", output), ("fused_PM25.nc", fused)):
        content, name, _ = export_structure(ds, "NetCDF")
        quality[filename] = validate_export(ds, content, name, "NetCDF")
        files[filename] = content
    plan["runtime_seconds"] = time.perf_counter()-started
    for filename, report in (("input_analysis.json", analysis), ("processing_plan.json", plan), ("quality_report.json", quality)):
        files[filename] = json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return ScenarioResult(output, fused, files, analysis, plan, quality)


class CMAQAdapter(ModelAdapter):
    """All air-quality-specific physical checks stay behind this boundary."""
    field_mapping = staticmethod(_field_mapping)
    supports_partial = False

    @staticmethod
    def unit_transform(source, target):
        source = _canonical_unit(source)
        return unit_transform(source, target)

    @staticmethod
    def inspect_ioapi(path, *, format="ioapi", python=None):
        from ecodata.models.adapters.ioapi_reader import CmaqIOAPIReader
        return CmaqIOAPIReader(python).inspect(path, format=format)

    def validate_field(self, ds, source, field):
        if source.startswith("derived:"):
            for component in ("u10m", "v10m"):
                _convert(ds[component], "m/s")
        else:
            if ds[source].dtype.kind not in "biuf":
                raise SchemaError("非数值字段不能作为连续物理变量")
            _convert(ds[source], field["unit"])

    def validate_plan(self, schema, inputs, assignments, plan):
        _, checks = plan_scenario(schema.profile, inputs, assignments)
        plan.update(scope=checks["scope"], compatibility=checks["compatibility"], steps=checks["steps"])
        plan["errors"] = list(dict.fromkeys([*plan["errors"], *checks["errors"]]))
        by_role = {item["role"]: item for item in plan["matches"]}
        for checked in checks["matches"]:
            if checked.get("reasons"):
                match = by_role[checked["role"]]
                match.update(can_process=False, reasons=checked["reasons"], status=checked["status"])

    def execute(self, schema, inputs, assignments, analysis, plan, **options):
        if not plan["can_process"]:
            raise SchemaError("；".join(plan["errors"]))
        result = execute_scenario(schema.profile, inputs, assignments, **options)
        result.plan.update(adapter=schema.preparation["adapter"], matches=plan["matches"], execution_order=plan["execution_order"])
        result.plan["information_loss"] = [
            {"operation": "time_intersection", "loss": "仅共同实际时段，未输出时段计数见original_time_counts"},
            {"operation": "spatial_regrid", "loss": "重采样可能平滑极值；不代表新增真实观测"},
            {"operation": "residual_fusion", "loss": "推断改变背景；原始背景和融合值独立保留"},
            {"operation": "negative_policy", "loss": "截断改变数值；原始值、修正量和质量标记保留"}]
        source = next(i.dataset for i in inputs if i.filename == assignments["observation"])
        self.observation = source.copy(deep=True)
        mapping = _field_mapping(source, next(r for r in schema.inputs if r["role"] == "observation"))
        self.observation["PM25"] = _convert(source[mapping["PM25"]], "ug/m3")
        return result

    def output_roles(self, result, schema):
        return {"background": result.output[["PM25"]], "meteorology": result.output[["temperature", "wind", "humidity"]],
                "terrain": result.output[["elevation"]], "observation": self.observation}

    def validation_history(self, result, role):
        options = result.plan["algorithm_options"]
        if role in {"meteorology", "terrain"}:
            method = options["terrain_spatial_method"] if role == "terrain" else options["spatial_method"]
            return [{"method": "spatial.reproject_and_regrid." + method, "parameters": {"crs": "EPSG:4326"}}]
        if role == "background":
            return [{"method": "fusion.residual." + options["residual_method"], "parameters": {"negative_policy": options["negative_policy"]}}]
        return []
