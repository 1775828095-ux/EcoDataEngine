"""One public orchestration path for structure, scale, fusion and model matching."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.evaluation.self_check import PostConversionValidator, profile_dataset, require_usable
from ecodata.core.schemas import TransformRecord
from ecodata.interface.workflow import TrialRequest, TrialResult, run_fusion_trial, run_grid_trial, run_structure_trial, run_wide_table_time_trial
from ecodata.io.structures import StructureRequest
from ecodata.spatial.regridding import normalize_grid_axes


@dataclass(frozen=True)
class PlanStep:
    order: int
    stage: str
    operation: str
    reason: str

    def as_dict(self):
        return {"顺序": self.order, "阶段": self.stage, "处理": self.operation, "原因": self.reason}


class ConversionTaskEngine:
    """Compose existing engines; it contains no duplicate interpolation logic."""

    def plan(self, structure: StructureRequest, processing: TrialRequest, *, fuse: bool, model: dict | None) -> tuple[PlanStep, ...]:
        steps = [PlanStep(1, "数据识别", "识别结构、变量、单位、时间、坐标、CRS 和数据类型", "所有后续处理的前置检查")]
        if structure.operation != "preserve":
            steps.append(PlanStep(len(steps) + 1, "结构转换", structure.operation, "将输入整理为后续算法可处理的数据结构"))
        if processing.target_time_frequency:
            steps.append(PlanStep(len(steps) + 1, "时间处理", f"转换为 {processing.target_time_frequency}", "用户或目标需求指定"))
        if processing.target_spatial_resolution_km:
            operation = f"{processing.spatial_method} → {processing.target_spatial_resolution_km:g} km"
            if processing.optimization_options is not None:
                operation += f"（{processing.optimization_options.solver} 最大 {processing.optimization_options.max_iter} 次迭代）"
            steps.append(PlanStep(len(steps) + 1, "空间处理", operation, "用户或目标需求指定"))
        if fuse:
            operation = "GSTools " + processing.fusion_method if processing.fusion_method.endswith("kriging") else "背景场 + 站点残差融合" if processing.fusion_method == "residual" else "时间/CRS/网格/单位对齐 → 反方差加权融合"
            steps.append(PlanStep(len(steps) + 1, "标准化与融合", operation, "检查变量语义、单位、重叠范围和有效数据，禁止猜测物理信息"))
        if model:
            steps.append(PlanStep(len(steps) + 1, "模型需求匹配", model["name"], "检查必需变量、单位、格式和时空要求"))
        steps.append(PlanStep(len(steps) + 1, "输出", processing.output_format, "生成转换文件和可追溯 JSON 报告"))
        return tuple(steps)

    def execute(
        self, datasets: list[xr.Dataset], structure: StructureRequest, processing: TrialRequest,
        *, fuse: bool = False, model: dict | None = None,
        training_observations: pd.DataFrame | None = None,
    ) -> TrialResult:
        if not datasets:
            raise SchemaError("没有可处理的数据")
        if model and model.get("selected_input"):
            from ecodata.fusion.planning import DataRoleClassifier
            from ecodata.models.adaptation import field_requirements
            requirement = model["inputs"][0]
            fields = field_requirements(requirement)
            check = DataRoleClassifier().compatibility(datasets[0], {**requirement, "fields": fields},
                {field["name"]: field["name"] for field in fields}, processing.variables or (processing.variable,))
            if not check["passed"]:
                raise SchemaError("；".join(check["reasons"]))
        originals = [ds.copy(deep=True) for ds in datasets]
        baseline = profile_dataset(originals[0])
        result = self._execute(datasets, structure, processing, fuse=fuse, model=model, training_observations=training_observations)
        kind = "fusion" if fuse else "resampling" if processing.target_time_frequency or processing.target_spatial_resolution_km else "structure" if structure.operation != "preserve" else "format"
        selected = processing.variables or ((processing.variable,) if processing.variable else ())
        model_check = None
        if model and model.get("selected_input"):
            from ecodata.models.adaptation import prepare_input, field_requirements
            from ecodata.recognition import inspect_dataset
            requirements = field_requirements(model["inputs"][0])
            _, check = prepare_input(result.output, inspect_dataset("output.nc", result.output), model,
                                     {field["name"]: field["name"] for field in requirements}, selected)
            model_check = {**check, "complete": check["ready"], "scope": "selected_input_standardized_preprocessing", "native_verified": False}
        stations = None
        if fuse and processing.fusion_method == "residual":
            from ecodata.io.structures import table_frame
            stations = table_frame(originals[1])
        audit = PostConversionValidator().validate(originals[0], result.output, kind=kind, history=result.history, critical_fields=selected, sources=originals[1:], before_profile=baseline, model_check=model_check, observation_stations=stations)
        if "独立验证" in result.report:
            audit.details["accuracy"] = result.report["独立验证"]
        require_usable(audit)
        result = replace(result, original=originals[0])
        result.report["validation_report"] = audit.as_dict()
        return result

    def _execute(
        self, datasets, structure, processing, *, fuse=False, model=None, training_observations=None,
    ):
        if not datasets:
            raise SchemaError("没有可处理的数据")
        fusion_plan = None
        if fuse and processing.fusion_method == "auto":
            from ecodata.fusion.planning import FusionPlanner
            fusion_plan = FusionPlanner().plan(datasets, processing.variable, "auto", processing.covariates)
            if not fusion_plan["ready"]:
                raise SchemaError("；".join(fusion_plan["reasons"]))
            processing = replace(processing, fusion_method=fusion_plan["method"], covariates=tuple(fusion_plan["covariates"]))
        plan = self.plan(structure, processing, fuse=fuse, model=model)
        primary = datasets[0]
        variables = processing.variables or ((processing.variable,) if processing.variable else ())
        if fuse:
            if processing.fusion_method.endswith("kriging"):
                from ecodata.fusion.kriging_task import run_kriging_task
                result = run_kriging_task(datasets, processing)
            elif processing.fusion_method == "residual":
                from ecodata.fusion.standardized_residual import run_residual_task
                result = run_residual_task(datasets, processing)
            else:
                result = run_fusion_trial(datasets, processing)
            result.report["自动处理流程"] = [item.as_dict() for item in plan]
            if fusion_plan is not None:
                result.report["fusion_plan"] = fusion_plan
            if model:
                result.report["目标模型需求"] = model
            return result
        if processing.target_time_frequency and primary.attrs.get("structure") == "table" and structure.operation == "preserve":
            if not structure.time:
                raise SchemaError("识别到表格时间处理，但未指定时间字段")
            result = run_wide_table_time_trial(primary, processing, station_field=structure.x, time_field=structure.time)
            result.report["自动处理流程"] = [item.as_dict() for item in plan]
            if model:
                result.report["目标模型需求"] = model
                if model.get("status") != "verified_native_adapter":
                    result.report["模型输出说明"] = "已按需求生成标准化预处理数据；该模型原生运行文件适配器尚未验证。"
            return result
        structure_result = run_structure_trial(primary, structure, "NetCDF")
        working = structure_result.output
        history = structure_result.history
        if working.attrs.get("structure") == "grid":
            axes = (working.attrs.get("spatial_y", "lat"), working.attrs.get("spatial_x", "lon"))
            working, reversed_axes = normalize_grid_axes(working, axes)
            if reversed_axes:
                record = TransformRecord("spatial.coordinate_orientation", {"reversed_axes": reversed_axes},
                                         int(sum(v.size for v in primary.data_vars.values())), int(sum(v.size for v in working.data_vars.values())))
                history = (*history, record.as_dict())
                working.attrs["coordinate_orientation_history"] = str(reversed_axes)
        has_scale = bool(processing.target_time_frequency or processing.target_spatial_resolution_km)
        if has_scale:
            rename = {}
            if structure.x in working.dims and structure.x != "lon":
                rename[structure.x] = "lon"
            if structure.y in working.dims and structure.y != "lat":
                rename[structure.y] = "lat"
            if rename:
                working = working.rename(rename)
                working.attrs.update(spatial_x="lon", spatial_y="lat")
        if has_scale:
            if len(variables) <= 1:
                result = run_grid_trial(
                    working, processing, tuple(TransformRecord(**item) for item in history),
                    training_observations=training_observations,
                )
            else:
                outputs = []
                histories = []
                elapsed = 0.0
                for variable in variables:
                    item = run_grid_trial(
                        working, replace(processing, variable=variable, variables=()),
                        tuple(TransformRecord(**entry) for entry in history),
                        training_observations=training_observations,
                    )
                    outputs.append(item.output[[variable]])
                    histories.extend(item.history)
                    elapsed += item.runtime_seconds
                merged = xr.merge(outputs, compat="override")
                report = item.report
                report["多变量处理"] = list(variables)
                result = TrialResult(primary, merged, tuple(histories), report, elapsed)
        else:
            result = run_structure_trial(working, StructureRequest(operation="preserve", variable=processing.variable or structure.variable), processing.output_format)
        result.report["自动处理流程"] = [item.as_dict() for item in plan]
        if model:
            result.report["目标模型需求"] = model
            if model.get("status") != "verified_native_adapter":
                result.report["模型输出说明"] = "已按需求生成标准化预处理数据；该模型原生运行文件适配器尚未验证，不能声称可直接启动模型。"
        return result
