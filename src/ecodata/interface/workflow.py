from __future__ import annotations

import io
import json
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.crs import CRS

from ecodata.core.exceptions import SchemaError
from ecodata.core.pipeline import P3Pipeline, SequentialPipeline
from ecodata.core.schemas import EcoDataBundle, TransformRecord
from ecodata.io.readers import CsvReadConfig, StationCsvReader
from ecodata.fusion.weighted import UncertaintyWeightedFusion, WeightedFusionConfig
from ecodata.optimization.downscaling import QPOptions, supported_regrid
from ecodata.temporal.resampling import LinearTimeConfig, LinearTimeInterpolator
from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler
from ecodata.spatial.regridding import normalize_grid_axes

QP_METHODS = {"constrained", "peak_aware", "adaptive_smoothing", "adaptive_peak"}


@dataclass(frozen=True)
class Recognition:
    filename: str
    format: str
    variables: tuple[str, ...]
    selected_variable: str | None
    unit: str | None
    temporal_semantics: str | None
    spatial_semantics: str | None
    time_coordinate: str | None
    latitude_coordinate: str | None
    longitude_coordinate: str | None
    temporal_resolution: str | None
    spatial_resolution_km: float | None
    confirmations_required: tuple[str, ...]


@dataclass(frozen=True)
class TrialRequest:
    target_time_frequency: str | None = None
    target_spatial_resolution_km: float | None = None
    spatial_method: str = "auto"
    variable: str | None = None
    variables: tuple[str, ...] = ()
    temporal_semantics: str | None = None
    unit: str | None = None
    target_model: str = "generic_cf_netcdf"
    output_format: str = "NetCDF"
    semantic_type: str = "continuous"
    target_fields: tuple[str, ...] = ()
    optimization_options: QPOptions | None = None
    spatial_semantics: str | None = None
    fusion_method: str = "weighted"
    covariates: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrialResult:
    original: xr.Dataset
    output: xr.Dataset
    history: tuple[dict[str, Any], ...]
    report: dict[str, Any]
    runtime_seconds: float

    def __post_init__(self):
        from ecodata.evaluation.self_check import PostConversionValidator, require_usable
        context = self.report.get("validation_report", {})
        audit = PostConversionValidator().validate(self.original, self.output, kind=context.get("task_kind"), history=self.history,
            field_mapping=context.get("field_mapping"), critical_fields=context.get("critical_fields", ()), model_check=context.get("model_compatibility_check"))
        self.report["validation_report"] = audit.as_dict()
        require_usable(audit)

    def validation_files(self):
        from ecodata.evaluation.self_check import validation_files
        return validation_files(self.report["validation_report"])

    def _processing_validation(self):
        from ecodata.evaluation.self_check import PostConversionValidator, require_usable
        from ecodata.core.schemas import ValidationReport
        context = self.report["validation_report"]
        require_usable(ValidationReport(context["passed"], context["checks"], context))
        audit = PostConversionValidator().validate(self.original, self.output, kind=context["task_kind"], history=self.history,
            field_mapping=context.get("field_mapping"), critical_fields=context.get("critical_fields", ()), model_check=context.get("model_compatibility_check"))
        require_usable(audit)
        return audit

    def netcdf_bytes(self) -> bytes:
        return self.export("NetCDF")[0]

    def csv_bytes(self) -> bytes:
        self._processing_validation()
        return self.output.to_dataframe().reset_index().to_csv(index=False).encode("utf-8-sig")

    def report_bytes(self) -> bytes:
        return json.dumps(self.report, ensure_ascii=False, indent=2, default=str).encode("utf-8")

    def export(self, output_format: str) -> tuple[bytes, str, str]:
        from ecodata.evaluation.output_quality import validate_export, processing_quality
        from ecodata.evaluation.self_check import PostConversionValidator, require_usable, profile_dataset
        from ecodata.recognition import recognise_file
        from ecodata.io.structures import projected_coordinate_names
        from copy import deepcopy
        audit = self.report["validation_report"]
        processing = self._processing_validation()
        restored = None
        try:
            content, filename, mime = self._export_unchecked(output_format)
            restored, _ = recognise_file(filename, content, record_read_source=False)
            export_history = [{"method": "serialization.reproject", "parameters": {"crs": "EPSG:4326"}}] if output_format == "GeoJSON" else []
            file_check = PostConversionValidator().validate(projected_coordinate_names(self.output), restored,
                kind="format" if output_format in {"NetCDF", "HDF5-NetCDF"} else "structure",
                history=export_history, metadata_sidecar=output_format in {"CSV", "TSV", "JSON", "Excel", "GeoJSON"})
            require_usable(file_check)
            self.report["自动质量检查"] = validate_export(self.output, content, filename, output_format)
            self.report["自动质量检查"]["processing"] = processing_quality(self.original, self.output)
        except Exception as exc:
            from ecodata.core.exceptions import ValidationError
            from ecodata.core.schemas import ValidationReport
            failed = deepcopy(processing.as_dict())
            failed.update(status="FAIL", passed=False, after_profile_available=restored is not None,
                          processing_after_profile=failed["after_profile"],
                          after_profile=profile_dataset(restored) if restored is not None else profile_dataset(xr.Dataset()))
            failure = getattr(exc, "report", None)
            failed["file_validation"] = failure.as_dict() if failure else {"status": "FAIL", "reason": str(exc)}
            failed["errors"].append("输出文件自查失败：" + str(exc))
            failed["checks_by_category"]["structure"] = "FAIL"
            if failure:
                for key, level in failure.details["checks_by_category"].items():
                    if level != "PASS":
                        failed["checks_by_category"][key] = level
                failed["information_loss"] = failure.details["information_loss"]
            failed["checks"] = {key: value != "FAIL" for key, value in failed["checks_by_category"].items()}
            self.report["validation_report"] = failed
            raise ValidationError(failed["errors"][-1], ValidationReport(False, failed["checks"], failed)) from exc
        combined = deepcopy(processing.as_dict())
        combined["source_profiles"] = audit.get("source_profiles", [])
        if audit.get("accuracy", {}).get("available"):
            combined["accuracy"] = audit["accuracy"]
        combined["observation_placement"] = audit.get("observation_placement")
        combined["extrapolated_spatial_cell_count"] = audit.get("extrapolated_spatial_cell_count")
        if audit["information_loss"].get("extrapolation_applied"):
            combined["information_loss"]["extrapolation_applied"] = True
            combined["warnings"].extend(warning for warning in audit["warnings"] if warning not in combined["warnings"])
        combined["processing_after_profile"] = combined["after_profile"]
        combined["after_profile"] = profile_dataset(restored)
        combined["file_validation"] = file_check.as_dict()
        combined["warnings"].extend(file_check.details["warnings"])
        combined["status"] = "WARN" if combined["warnings"] else "PASS"
        for key, level in file_check.details["checks_by_category"].items():
            if level == "WARN":
                combined["checks_by_category"][key] = level
        for key in ("lost_fields", "lost_metadata"):
            combined["information_loss"][key] = sorted(set(combined["information_loss"][key]) | set(file_check.details["information_loss"][key]))
        for key in ("changed_units", "changed_resolution", "changed_time_semantics", "changed_crs"):
            if file_check.details["information_loss"][key]:
                combined["information_loss"][key].update(file_check.details["information_loss"][key])
        for key in ("interpolation_applied", "extrapolation_applied"):
            combined["information_loss"][key] |= file_check.details["information_loss"][key]
        combined["information_loss"]["warnings"] = combined["warnings"]
        if output_format in {"NetCDF", "HDF5-NetCDF"}:
            from ecodata.engines.validation import ValidationEngine
            conventions = str(self.output.attrs.get("Conventions", ""))
            standards = ValidationEngine.compliance(content, acdd="ACDD" in conventions or bool(self.output.attrs.get("summary")))
            combined["standards_compliance"] = standards
            # Integrity and standards are distinct: a lossless copy may not be CF.
            combined["standardized_preprocessing_status"] = combined["status"] if standards["status"] == "PASS" else "WARN"
            combined["native_model_input_verified"] = False
        self.report["validation_report"] = combined
        return content, filename, mime

    def _export_unchecked(self, output_format: str) -> tuple[bytes, str, str]:
        if self.report.get("结构转换"):
            from ecodata.io.structures import export_structure
            return export_structure(self.output, output_format)
        if output_format in {"NetCDF", "HDF5-NetCDF", "GeoTIFF", "Shapefile ZIP", "Parquet"}:
            from ecodata.io.structures import export_structure
            dataset = self.output.copy()
            dataset.attrs["transform_history"] = json.dumps(list(self.history), ensure_ascii=False)
            return export_structure(dataset, output_format)
        frame = self.output.to_dataframe().reset_index()
        if output_format == "CSV":
            return frame.to_csv(index=False).encode("utf-8-sig"), "ecodata_result.csv", "text/csv"
        if output_format == "TSV":
            return frame.to_csv(index=False, sep="\t").encode("utf-8-sig"), "ecodata_result.tsv", "text/tab-separated-values"
        if output_format == "JSON":
            return frame.to_json(orient="records", date_format="iso", force_ascii=False).encode("utf-8"), "ecodata_result.json", "application/json"
        if output_format == "Excel":
            if len(frame) > 1_048_576:
                raise SchemaError("Excel 单工作表最多 1,048,576 行，请选择其他格式")
            buffer = io.BytesIO()
            frame.to_excel(buffer, index=False, engine="openpyxl")
            return buffer.getvalue(), "ecodata_result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if output_format == "NetCDF":
            return self.netcdf_bytes(), "ecodata_result.nc", "application/x-netcdf"
        if output_format == "HDF5-NetCDF":
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
                path = Path(handle.name)
            try:
                dataset = self.output.copy()
                dataset.attrs["transform_history"] = json.dumps(list(self.history), ensure_ascii=False)
                dataset.to_netcdf(path, engine="h5netcdf")
                content = path.read_bytes()
            finally:
                path.unlink(missing_ok=True)
            return content, "ecodata_result.h5", "application/x-hdf5"
        if output_format == "GeoJSON":
            return _geojson_bytes(frame), "ecodata_result.geojson", "application/geo+json"
        raise SchemaError(f"尚未实现输出格式：{output_format}")


def load_netcdf(content: bytes) -> xr.Dataset:
    errors: list[Exception] = []
    for engine in ("h5netcdf", "scipy"):
        try:
            with xr.open_dataset(io.BytesIO(content), engine=engine) as opened:
                return opened.load()
        except Exception as exc:
            errors.append(exc)
    raise SchemaError("文件不是可读取的 NetCDF3 或 HDF5-NetCDF") from errors[-1]


def run_structure_trial(dataset: xr.Dataset, request, output_format: str) -> TrialResult:
    """Use the shared Pipeline for explicit format/structure conversions."""
    from ecodata.io.structures import StructureConverter, dumps, export_structure, metadata
    started = time.perf_counter()
    prior_history = ()
    if dataset.attrs.get("transform_history"):
        try:
            prior_history = tuple(TransformRecord(**item) for item in json.loads(dataset.attrs["transform_history"]))
        except (TypeError, ValueError) as exc:
            raise SchemaError("输入转换历史无法解析，请核对 transform_history 元数据") from exc
    transformed = SequentialPipeline().run(EcoDataBundle(dataset, prior_history), [StructureConverter(request)]).data
    output = transformed.dataset.copy()
    output.attrs["export_variable"] = request.variable
    history = tuple(record.as_dict() for record in transformed.history)
    output.attrs["transform_history"] = dumps(history)
    content, filename, _ = export_structure(output, output_format)
    elapsed = time.perf_counter() - started
    decisions = [{"步骤": "格式与数据结构转换", "算法": request.operation, "原因": "用户明确选择字段、坐标和处理规则"}]
    report = _report(dataset, output, decisions, history, elapsed, reference=None)
    report.update({
        "结构转换": True,
        "目标模型": "自定义格式与数据结构转换",
        "目标输出格式": output_format,
        "导出文件": filename,
        "导出字节数": len(content),
        "输入元数据": metadata(dataset),
        "输出元数据": metadata(output),
        "注意事项": [warning for record in history for warning in record["warnings"]],
    })
    if output_format in {"CSV", "TSV", "JSON"}:
        report["注意事项"].append("此表格格式不完整携带类型/单位/CRS，须同时保存本 JSON 报告；重新导入时确认类型和坐标。")
    if output_format == "GeoJSON":
        report["注意事项"].append("GeoJSON 按 RFC 7946 输出 WGS84 经度/纬度；已有投影坐标会通过 Rasterio 转换。")
    if output_format == "GeoTIFF":
        from ecodata.io.structures import validate_geotiff
        report["GeoTIFF输出检查"] = validate_geotiff(content)
        report["导出波段字段"] = [request.variable]
        report["未作为波段输出的字段"] = [key for key in output.data_vars if key != request.variable]
        report["注意事项"].append("GeoTIFF 仅导出所选字段的二维切片，其他字段不写入波段；时间切片、单位与历史保存在标签及报告中。")
    return TrialResult(dataset, output, history, report, elapsed)


def recognise_netcdf(filename: str, dataset: xr.Dataset) -> Recognition:
    numeric = tuple(name for name, value in dataset.data_vars.items() if np.issubdtype(value.dtype, np.number))
    auxiliary = {"uncertainty", "quality_flag", "qc_flag"}
    candidates = tuple(
        name for name in numeric
        if {"lat", "lon"}.issubset(dataset[name].dims) and name.lower() not in auxiliary
    )
    selected = candidates[0] if len(candidates) == 1 else None
    confirmations: list[str] = []
    if selected is None:
        confirmations.append("目标数据变量")
    unit = dataset[selected].attrs.get("units") if selected else None
    temporal = _metadata(dataset, selected, "temporal_semantics")
    spatial = _metadata(dataset, selected, "spatial_semantics")
    if not unit:
        confirmations.append("数据单位")
    if "time" in dataset.dims and temporal not in {"instantaneous", "mean", "accumulation"}:
        confirmations.append("时间语义")
    lat_name = "lat" if "lat" in dataset.coords else None
    lon_name = "lon" if "lon" in dataset.coords else None
    if not lat_name or not lon_name:
        confirmations.append("纬度/经度坐标")
    temporal_resolution = None
    if "time" in dataset.coords and dataset.sizes.get("time", 0) > 1:
        delta = np.diff(dataset["time"].values.astype("datetime64[ns]").astype("int64"))
        if np.all(delta == delta[0]):
            temporal_resolution = str(pd.to_timedelta(int(delta[0]), unit="ns"))
    spatial_resolution = dataset.attrs.get("grid_resolution_km")
    if spatial_resolution is not None:
        spatial_resolution = float(spatial_resolution)
    return Recognition(
        filename, "NetCDF", numeric, selected, unit, temporal, spatial,
        "time" if "time" in dataset.coords else None, lat_name, lon_name,
        temporal_resolution, spatial_resolution, tuple(dict.fromkeys(confirmations)),
    )


def recognise_csv(filename: str, frame: pd.DataFrame) -> Recognition:
    required = {"station_id", "time", "latitude", "longitude", "variable", "value", "unit", "qc_flag"}
    missing = sorted(required - set(frame.columns))
    confirmations = [f"缺少标准字段：{', '.join(missing)}"] if missing else []
    variables = tuple(sorted(frame["variable"].dropna().astype(str).unique())) if "variable" in frame else ()
    units = tuple(sorted(frame["unit"].dropna().astype(str).unique())) if "unit" in frame else ()
    return Recognition(
        filename, "CSV", variables, "value" if not missing else None,
        units[0] if len(units) == 1 else None, None, "point",
        "time" if "time" in frame else None,
        "latitude" if "latitude" in frame else None,
        "longitude" if "longitude" in frame else None,
        None, None, tuple(confirmations + (["时间语义"] if not missing else [])),
    )


def run_grid_trial(
    dataset: xr.Dataset, request: TrialRequest, initial_history=(), *,
    training_observations: pd.DataFrame | None = None,
) -> TrialResult:
    started = time.perf_counter()
    variable = request.variable
    if not variable or variable not in dataset:
        raise SchemaError("请确认要处理的数据变量")
    is_qp = request.spatial_method in QP_METHODS
    if is_qp and request.semantic_type != "continuous":
        raise SchemaError("QP 只适用于连续物理变量")
    if is_qp and not request.target_spatial_resolution_km:
        raise SchemaError("QP 需要设置比源网格更小的目标空间分辨率")
    if is_qp and request.unit and dataset[variable].attrs.get("units") not in {None, request.unit}:
        raise SchemaError("QP 输入单位与确认单位不同；请先进行明确的单位转换")
    working = dataset.copy(deep=True)
    working, reversed_axes = normalize_grid_axes(working)
    if request.unit:
        working[variable].attrs["units"] = request.unit
    working[variable].attrs["data_kind"] = "continuous" if request.semantic_type == "continuous" else request.semantic_type
    working.attrs["data_kind"] = working[variable].attrs["data_kind"]
    if request.spatial_semantics:
        working[variable].attrs["spatial_semantics"] = request.spatial_semantics
    if not working[variable].attrs.get("units"):
        raise SchemaError("缺少数据单位，请输入实际单位；无法确定时可填写 unknown，无量纲可填写 1 或 无量纲")
    if request.temporal_semantics:
        working.attrs["temporal_semantics"] = request.temporal_semantics
        working[variable].attrs["temporal_semantics"] = request.temporal_semantics
    if "time" in working[variable].dims and _metadata(working, variable, "temporal_semantics") not in {
        "instantaneous", "mean", "accumulation"
    }:
        raise SchemaError("缺少可靠的时间语义，请先确认瞬时值、平均值或累计值")
    if not {"lat", "lon"}.issubset(working[variable].dims):
        raise SchemaError("当前版本仅支持名为 lat/lon 的规则经纬度网格")

    history_records = tuple(initial_history)
    if reversed_axes:
        history_records += (TransformRecord("spatial.coordinate_orientation", {"reversed_axes": reversed_axes}, int(dataset[variable].size), int(working[variable].size)),)
    bundle = EcoDataBundle(working[[variable]], history_records)
    transforms = []
    spatial_plan = None
    decisions: list[dict[str, str]] = []
    if request.target_time_frequency:
        if request.semantic_type != "continuous":
            raise SchemaError("当前自动时间转换仅支持连续物理量；其他类型需要明确的聚合或映射规则")
        transforms.append(SemanticTimeResampler(SemanticTimeConfig(request.target_time_frequency, variable)))
        decisions.append({"步骤": "时间尺度转换", "算法": "语义感知重采样", "原因": "用户设置了目标时间分辨率"})
    if request.target_spatial_resolution_km:
        source_km = working.attrs.get("grid_resolution_km")
        if source_km is None:
            raise SchemaError("源数据未提供 grid_resolution_km，无法可靠生成目标网格")
        target_lat, target_lon = target_grid(working, float(source_km), request.target_spatial_resolution_km)
        method = request.spatial_method
        if is_qp and request.target_spatial_resolution_km >= float(source_km):
            raise SchemaError("QP 仅用于低分辨率到高分辨率：目标 km 必须小于源 km")
        if is_qp and (len(target_lat) % working.sizes["lat"] or len(target_lon) % working.sizes["lon"] or len(target_lat) // working.sizes["lat"] != len(target_lon) // working.sizes["lon"]):
            raise SchemaError("QP 需要两方向一致的整数倍细化；任意分辨率请使用最近邻、双线性或保守重网格")
        if method == "auto":
            if request.semantic_type == "categorical":
                method = "nearest"
            elif request.semantic_type == "continuous":
                method = "conservative" if request.target_spatial_resolution_km > float(source_km) else "bilinear"
            else:
                raise SchemaError("该数据语义类型没有可安全自动选择的空间转换算法")
        spatial_plan = (method, target_lat, target_lon)
        reason = "训练站约束优化，保持非负性和粗网格算术平均一致性" if is_qp else (
            "保持单元平均量" if method == "conservative" else "用户选择或自动确定的空间插值"
        )
        decisions.append({"步骤": "空间尺度转换", "算法": method, "原因": reason})
    if not transforms and spatial_plan is None:
        raise SchemaError("请至少设置一个目标时间或空间分辨率")
    current = SequentialPipeline().run(bundle, transforms).data if transforms else bundle
    if spatial_plan is not None:
        method, target_lat, target_lon = spatial_plan
        if is_qp:
            observations = _qp_training(current.dataset, variable, training_observations)
            current = EcoDataBundle(current.dataset.transpose("time", "lat", "lon"), current.history)
            current = P3Pipeline().run(
                current, method, target_lat, target_lon, observations,
                variable=variable, optimization_options=request.optimization_options,
            )
        else:
            current = supported_regrid(current, method, target_lat, target_lon, variable)
        actual_x = float(source_km) * working.sizes["lon"] / len(target_lon)
        actual_y = float(source_km) * working.sizes["lat"] / len(target_lat)
        crs = current.dataset.attrs.get("crs")
        if crs and CRS.from_user_input(crs).is_projected:
            _, factor = CRS.from_user_input(crs).linear_units_factor
            width = abs(target_lon[1] - target_lon[0])
            height = abs(target_lat[1] - target_lat[0])
            actual_x, actual_y = width * factor / 1000, height * factor / 1000
            current.dataset.attrs.update(pixel_width=width, pixel_height=height, pixel_width_km=actual_x, pixel_height_km=actual_y)
        current.dataset.attrs.update(grid_resolution_km=(actual_x + actual_y) / 2,
                                     requested_grid_resolution_km=request.target_spatial_resolution_km,
                                     actual_grid_resolution_x_km=actual_x, actual_grid_resolution_y_km=actual_y)
    current.dataset.attrs.setdefault("Conventions", "CF-1.10")
    elapsed = time.perf_counter() - started
    history = tuple(record.as_dict() for record in current.history)
    report = _report(dataset, current.dataset, decisions, history, elapsed, reference=None)
    if spatial_plan is not None:
        report["空间分辨率"] = {"请求_km": request.target_spatial_resolution_km,
                             "实际x_km": actual_x, "实际y_km": actual_y,
                             "说明": "保持源像元边界范围，目标行列数取整；实际分辨率已记录。"}
    report["目标模型"] = "自定义转换" if request.target_model == "custom" else "通用 CF-NetCDF"
    report["目标输出格式"] = request.output_format
    if is_qp:
        report["优化诊断"] = {
            "每时刻实际迭代次数": history[-1]["parameters"]["solver_iterations"],
            "求解状态": history[-1]["parameters"]["solver_statuses"],
            "粗网格算术平均一致性最大误差": current.dataset.attrs["coarse_consistency_max_abs_error"],
            "非负约束最大违反量": current.dataset.attrs["nonnegativity_max_violation"],
            "峰值软约束最大松弛量": current.dataset.attrs["peak_constraint_max_slack"],
            "说明": "约束误差不是精度指标；球面面积加权守恒不由此处的算术平均约束保证。",
        }
    return TrialResult(dataset, current.dataset, history, report, elapsed)


def _qp_training(dataset: xr.Dataset, variable: str, observations: pd.DataFrame | None) -> pd.DataFrame:
    if observations is None or observations.empty:
        raise SchemaError("所选 QP 需要训练监测站数据；请上传表格并明确选择训练站文件")
    required = {"station_id", "time", "latitude", "longitude", "variable", "value", "unit", "qc_flag", "split"}
    if missing := sorted(required - set(observations.columns)):
        raise SchemaError("训练站缺少字段：" + "、".join(missing))
    if set(observations["split"]) != {"train"}:
        raise SchemaError("QP 只能使用 split=train 的训练站；validation/test 数据不得参与优化")
    if not observations["qc_flag"].astype(str).str.upper().eq("VALID").all():
        raise SchemaError("训练站包含非 VALID 质量标记，请先完成质控；本次不会静默删除或修改观测")
    if _metadata(dataset, variable, "temporal_semantics") != "instantaneous":
        raise SchemaError("上传 QP 当前要求粗网格与站点均为同一时刻的瞬时值")
    if set(dataset[variable].dims) != {"time", "lat", "lon"}:
        raise SchemaError("上传 QP 当前要求 time、lat、lon 三个维度")
    values = dataset[variable].values
    if not np.isfinite(values).all() or np.any(values < 0):
        raise SchemaError("非负 QP 要求粗网格值有限且非负")
    if not dataset[variable].attrs.get("physical_variable"):
        raise SchemaError("粗网格缺少 physical_variable，无法核对训练站的物理变量")
    if set(observations["variable"]) != {dataset[variable].attrs["physical_variable"]}:
        raise SchemaError("训练站与网格的 physical_variable 不一致")
    if set(observations["unit"]) != {dataset[variable].attrs["units"]}:
        raise SchemaError("训练站与网格单位不一致")
    frame = observations.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    if frame["time"].isna().any() or frame["station_id"].isna().any():
        raise SchemaError("训练站的 time 或 station_id 存在缺失/无效值")
    if frame.duplicated(["time", "station_id"]).any():
        raise SchemaError("训练站存在重复的 station_id/time 记录")
    for column, axis in (("latitude", "lat"), ("longitude", "lon")):
        coords = dataset[axis].values
        steps = np.diff(coords)
        if len(coords) < 2 or not np.isfinite(coords).all() or not np.all(steps > 0) or not np.allclose(steps, steps[0]):
            raise SchemaError("上传 QP 要求升序、等间隔的规则经纬度网格")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all() or not frame[column].between(coords[0] - steps[0] / 2, coords[-1] + steps[0] / 2).all():
            raise SchemaError("训练站坐标缺失或超出粗网格范围")
        if frame.groupby("station_id")[column].nunique().gt(1).any():
            raise SchemaError("同一 station_id 的坐标随时间变化；当前 QP 要求固定站点")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if not np.isfinite(frame["value"]).all() or frame["value"].lt(0).any():
        raise SchemaError("非负 QP 要求训练站 value 有限且非负")
    times = pd.DatetimeIndex(dataset.time.values)
    if times.has_duplicates or times.hasnans:
        raise SchemaError("QP 网格时间必须有效且不重复")
    frame = frame.loc[frame.time.isin(times)].copy()
    if frame.empty or set(frame.time) != set(times):
        raise SchemaError("训练站未覆盖转换后的全部时间点；不会自动插补站点观测")
    if frame.groupby("time").size().ne(frame.station_id.nunique()).any():
        raise SchemaError("每个转换时刻必须包含全部训练站观测")
    return frame


def run_multi_grid_trial(
    datasets: list[xr.Dataset], request: TrialRequest, *, training_observations: pd.DataFrame | None = None,
) -> TrialResult:
    """Fuse user-confirmed compatible grids, then apply the requested transforms."""
    if len(datasets) < 2:
        raise SchemaError("多源融合至少需要两个网格文件")
    variable = request.variable or "value"
    fused = UncertaintyWeightedFusion(WeightedFusionConfig(variable=variable)).apply(
        [EcoDataBundle(dataset) for dataset in datasets]
    ).data
    result = run_grid_trial(fused.dataset, request, fused.history, training_observations=training_observations)
    decisions = [{"步骤": "多源融合", "算法": "反方差加权", "原因": "用户确认多个同变量、同网格且含 uncertainty 的来源"}]
    result.report["算法决策"] = decisions + result.report["算法决策"]
    return result


def run_fusion_trial(datasets: list[xr.Dataset], request: TrialRequest) -> TrialResult:
    """Fuse compatible grids without forcing an unrelated scale conversion."""
    if len(datasets) < 2:
        raise SchemaError("多源融合至少需要两个网格文件")
    variable = request.variable or "value"
    started = time.perf_counter()
    from ecodata.fusion.alignment import align_and_fuse
    fused, alignment = align_and_fuse(datasets, variable, frequency=request.target_time_frequency, resolution=request.target_spatial_resolution_km,
                                     method=request.spatial_method if request.spatial_method in {"nearest", "bilinear"} else "bilinear")
    elapsed = time.perf_counter() - started
    history = tuple(record.as_dict() for record in fused.history)
    fused.dataset.attrs["export_variable"] = variable
    decisions = [{"步骤": "多源融合", "算法": "反方差加权", "原因": "物理语义检查后自动对齐，按有效观测的不确定度加权"}]
    report = _report(datasets[0], fused.dataset, decisions, history, elapsed, reference=None)
    report["目标模型"] = "自定义转换" if request.target_model == "custom" else request.target_model
    report["目标输出格式"] = request.output_format
    report["标准化与融合"] = alignment
    return TrialResult(datasets[0], fused.dataset, history, report, elapsed)


def run_wide_table_time_trial(dataset: xr.Dataset, request: TrialRequest, *, station_field: str, time_field: str) -> TrialResult:
    """Resample several wide-table measurements by station without inventing coordinates."""
    from ecodata.io.structures import table_dataset, table_frame
    if not request.target_time_frequency:
        raise SchemaError("宽表时间处理需要目标时间分辨率")
    variables = request.variables or ((request.variable,) if request.variable else ())
    if not variables:
        raise SchemaError("请至少选择一个数值变量")
    frame = table_frame(dataset)
    for field in (station_field, time_field, *variables):
        if field not in frame:
            raise SchemaError(f"缺少字段：{field}")
    frame = frame.copy()
    frame[time_field] = pd.to_datetime(frame[time_field], errors="coerce")
    if frame[time_field].isna().any() or frame[station_field].isna().any():
        raise SchemaError("站点标识或时间包含空值/无效值")
    if frame.duplicated([station_field, time_field]).any():
        raise SchemaError("同一站点同一时间存在重复记录，请先明确聚合规则")
    for variable in variables:
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    metadata_fields = [name for name in frame.columns if name not in {*variables, time_field}]
    rows = []
    started = time.perf_counter()
    for station, group in frame.groupby(station_field, sort=False, dropna=False):
        group = group.sort_values(time_field).set_index(time_field)
        start = group.index.min().ceil(request.target_time_frequency)
        end = group.index.max().floor(request.target_time_frequency)
        if start > end:
            continue
        target = pd.date_range(start, end, freq=request.target_time_frequency)
        union = group.index.union(target).sort_values()
        output = pd.DataFrame(index=target)
        for variable in variables:
            original = group[variable]
            values = original.reindex(union).interpolate(method="time", limit_area="inside").reindex(target)
            output[variable] = values
            original_times = set(original.dropna().index)
            output[f"{variable}_qc_flag"] = ["ORIGINAL" if stamp in original_times else "INTERPOLATED" if pd.notna(value) else "MISSING" for stamp, value in values.items()]
        for field in metadata_fields:
            unique = group[field].dropna().astype(str).unique()
            if len(unique) > 1 and field != station_field:
                raise SchemaError(f"同一站点的标识字段 {field} 随时间变化，请先明确映射规则")
            output[field] = station if field == station_field else (unique[0] if len(unique) else None)
        output[time_field] = output.index
        rows.append(output.reset_index(drop=True))
    if not rows:
        raise SchemaError("没有可生成的目标时间点")
    combined = pd.concat(rows, ignore_index=True)
    column_order = [*metadata_fields, time_field, *[item for variable in variables for item in (variable, f"{variable}_qc_flag")]]
    output_dataset = table_dataset(combined[column_order])
    output_dataset.attrs.update({key: value for key, value in dataset.attrs.items() if key not in {"structure", "row_dimension"}})
    for variable in variables:
        output_dataset[variable].attrs.update(dataset[variable].attrs)
    elapsed = time.perf_counter() - started
    record = TransformRecord(
        method="temporal.wide_table_interpolation", parameters={"frequency": request.target_time_frequency, "station_field": station_field, "time_field": time_field, "variables": list(variables)},
        input_rows=len(frame), output_rows=len(combined), elapsed_seconds=elapsed,
        warnings=("仅在每个站点已有观测的时间范围内插值，不进行外推。",),
    )
    history = (record.as_dict(),)
    report = _report(dataset, output_dataset, [{"步骤": "多变量时间转换", "算法": "站点内时间线性插值", "原因": "识别到宽表时间和多个连续数值字段"}], history, elapsed, reference=None)
    report["目标输出格式"] = request.output_format
    return TrialResult(dataset, output_dataset, history, report, elapsed)


def run_station_csv_trial(content: bytes, request: TrialRequest, filename: str = "uploaded.csv") -> TrialResult:
    if not request.target_time_frequency:
        raise SchemaError("站点 CSV 处理需要设置目标时间分辨率")
    if request.semantic_type != "continuous" or request.temporal_semantics != "instantaneous":
        raise SchemaError("当前标准站点 CSV 仅支持连续瞬时值的时间插值")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        config = CsvReadConfig("uploaded", filename, temporal_semantics="instantaneous")
        original = StationCsvReader().read(temporary, config)
    finally:
        temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    output = LinearTimeInterpolator(LinearTimeConfig(request.target_time_frequency)).apply(original).data
    if request.target_fields:
        unknown = sorted(set(request.target_fields) - set(output.dataset.data_vars))
        if unknown:
            raise SchemaError("目标字段不存在：" + "、".join(unknown))
        output = EcoDataBundle(output.dataset[list(request.target_fields)], output.history)
    elapsed = time.perf_counter() - started
    history = tuple(record.as_dict() for record in output.history)
    decisions = [{"步骤": "时间尺度转换", "算法": "线性插值", "原因": "连续瞬时站点观测"}]
    report = _report(original.dataset, output.dataset, decisions, history, elapsed, reference=None)
    report["目标模型"] = "自定义转换" if request.target_model == "custom" else "通用 CF-NetCDF"
    report["目标输出格式"] = request.output_format
    return TrialResult(original.dataset, output.dataset, history, report, elapsed)


def run_p3_sample(method: str, resolution_km: float, hour: int, data_dir: Path) -> tuple[TrialResult, xr.Dataset]:
    coarse = xr.load_dataset(data_dir / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf").isel(time=slice(hour, hour + 1))
    truth = xr.load_dataset(data_dir / "p3_pm25_truth_2p5km_48h.nc", engine="h5netcdf").isel(time=slice(hour, hour + 1))
    if resolution_km == 5.0:
        truth = truth.coarsen(lat=2, lon=2, boundary="exact").mean()
    target_lat, target_lon = tuple(truth.lat.values), tuple(truth.lon.values)
    train = None
    if method in {"constrained", "peak_aware", "adaptive_smoothing", "adaptive_peak"}:
        train = pd.read_csv(data_dir / "p3_pm25_train_stations.csv", parse_dates=["time"])
        train = train.loc[train.time.eq(pd.Timestamp(coarse.time.values[0]))]
        if resolution_km == 5.0:
            raise SchemaError("QP 对照实验当前仅支持内置 2.5 km 目标网格")
    started = time.perf_counter()
    output = P3Pipeline().run(EcoDataBundle(coarse), method, target_lat, target_lon, train)
    elapsed = time.perf_counter() - started
    history = tuple(record.as_dict() for record in output.history)
    decisions = [{"步骤": "空间降尺度实验", "算法": method, "原因": "用户在算法实验页明确选择"}]
    report = _report(coarse, output.dataset, decisions, history, elapsed, reference=truth)
    return TrialResult(coarse, output.dataset, history, report, elapsed), truth


def target_grid(dataset: xr.Dataset, source_km: float, target_km: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not np.isfinite([source_km, target_km]).all() or min(source_km, target_km) <= 0:
        raise SchemaError("源和目标空间分辨率必须是有限正数")
    ratio = source_km / target_km
    dataset, _ = normalize_grid_axes(dataset)
    lat = np.asarray(dataset["lat"].values, dtype=float)
    lon = np.asarray(dataset["lon"].values, dtype=float)
    if len(lat) < 2 or len(lon) < 2:
        raise SchemaError("空间转换至少需要 2×2 网格")
    from ecodata.io.structures import MAX_CELLS
    extra_cells = int(np.prod([size for name, size in dataset.sizes.items() if name not in {"lat", "lon"}]))
    if not np.isfinite(ratio) or ratio > np.sqrt(MAX_CELLS / (len(lat) * len(lon) * extra_cells)):
        raise SchemaError("目标网格超过本地 200 万单元上限，请增大目标分辨率或先裁剪")
    ny = int(round(len(lat) * ratio))
    nx = int(round(len(lon) * ratio))
    if min(ny, nx) < 2:
        raise SchemaError("目标分辨率过大，输出每个方向至少需要两个像元")
    if nx * ny * extra_cells > MAX_CELLS:
        raise SchemaError("目标网格超过本地 200 万单元上限，请增大目标分辨率或先裁剪")
    if not np.allclose(np.diff(lat), np.diff(lat)[0]) or not np.allclose(np.diff(lon), np.diff(lon)[0]):
        raise SchemaError("固定分辨率转换要求等间距坐标，请先明确非规则网格的处理方式")
    lat_step, lon_step = np.diff(lat).mean(), np.diff(lon).mean()
    lat_edges = (lat[0] - lat_step / 2, lat[-1] + lat_step / 2)
    lon_edges = (lon[0] - lon_step / 2, lon[-1] + lon_step / 2)
    target_lat = np.linspace(lat_edges[0], lat_edges[1], ny, endpoint=False) + (lat_edges[1] - lat_edges[0]) / ny / 2
    target_lon = np.linspace(lon_edges[0], lon_edges[1], nx, endpoint=False) + (lon_edges[1] - lon_edges[0]) / nx / 2
    return tuple(target_lat), tuple(target_lon)


def _metadata(dataset: xr.Dataset, variable: str | None, key: str) -> Any:
    return dataset[variable].attrs.get(key, dataset.attrs.get(key)) if variable else dataset.attrs.get(key)


def recommended_spatial_resolutions(dataset: xr.Dataset, source_km: float) -> tuple[float, ...]:
    """Offer only candidate grids that pass the same bounds and size checks."""
    x = dataset.attrs.get("spatial_x", "lon")
    y = dataset.attrs.get("spatial_y", "lat")
    if x not in dataset.dims or y not in dataset.dims or not np.isfinite(source_km) or source_km <= 0:
        return ()
    rename = {axis: target for axis, target in ((x, "lon"), (y, "lat")) if axis != target}
    grid = dataset.rename(rename) if rename else dataset
    candidates = []
    for multiplier in (0.5, 0.25, 0.1, 2.0, 4.0):
        target_km = source_km * multiplier
        try:
            target_grid(grid, source_km, target_km)
        except SchemaError:
            continue
        candidates.append(target_km)
    return tuple(candidates)


def _report(source: xr.Dataset, output: xr.Dataset, decisions, history, elapsed: float, reference: xr.Dataset | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "状态": "成功",
        "目标模型": "通用 CF-NetCDF 示例模型",
        "输入维度": dict(source.sizes),
        "输出维度": dict(output.sizes),
        "算法决策": decisions,
        "转换历史": history,
        "运行时间_秒": elapsed,
        "元数据": dict(output.attrs),
        "精度指标可用": reference is not None,
    }
    if reference is None:
        report["精度说明"] = "未提供独立参考真值，因此不计算或展示精度指标。"
    else:
        variable = "value"
        error = np.asarray(output[variable]) - np.asarray(reference[variable])
        report["精度指标"] = {
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
        }
    return report


def _geojson_bytes(frame: pd.DataFrame) -> bytes:
    lat_name = "latitude" if "latitude" in frame else "lat" if "lat" in frame else None
    lon_name = "longitude" if "longitude" in frame else "lon" if "lon" in frame else None
    if not lat_name or not lon_name:
        raise SchemaError("GeoJSON 输出需要 latitude/longitude 或 lat/lon 坐标")
    features = []
    for record in frame.to_dict(orient="records"):
        latitude, longitude = record.pop(lat_name), record.pop(lon_name)
        properties = {key: _json_value(value) for key, value in record.items()}
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(longitude), float(latitude)]}, "properties": properties})
    return json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False).encode("utf-8")


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value
