"""Default adapter composes existing Pipeline operations only."""
import json
import time

import numpy as np
import xarray as xr
from rasterio.crs import CRS

from ecodata.core.exceptions import SchemaError
from ecodata.evaluation.output_quality import validate_export
from ecodata.interface.workflow import TrialRequest
from ecodata.io.structures import FORMATS, StructureRequest, export_structure
from ecodata.models.adaptation import unit_transform
from ecodata.models.contracts import ScenarioResult
from ecodata.models.matching import field_mapping
from ecodata.recognition import inspect_dataset
from ecodata.task import ConversionTaskEngine


class ModelAdapter:
    """Unknown/native formats are deliberately not treated as CF-NetCDF."""
    output_formats = FORMATS
    supports_partial = True
    field_mapping = staticmethod(field_mapping)
    unit_transform = staticmethod(unit_transform)

    def validate_field(self, ds, source, field):
        if source not in ds:
            raise SchemaError("当前Adapter不提供该衍生变量")
        kind = field.get("data_type", "continuous")
        if kind in {"continuous", "integer_count"} and ds[source].dtype.kind not in "biuf":
            raise SchemaError("非数值字段不能作为数值物理变量")
        if kind in {"continuous", "integer_count"} and not np.isfinite(ds[source].values).any():
            raise SchemaError('变量全部缺失，不能匹配为已提供的物理变量')
        physical = ds[source].attrs.get("physical_variable")
        accepted = {str(v).casefold() for v in [field["name"], field.get("standard_name", field["name"]), *field.get("aliases", [])]}
        if physical and str(physical).casefold() not in accepted:
            raise SchemaError("物理变量声明不兼容，不能仅按相同单位强行映射")
        self.unit_transform(ds[source].attrs.get("units", ""), field.get("unit", ""))

    def role_issues(self, ds, role, mapping):
        issues = []
        settings = role["processing"] if isinstance(role["processing"], dict) else {}
        if settings.get("target_time_frequency") and ds.attrs.get("structure") == "table" and settings.get("structure_operation", "preserve") == "preserve":
            station = settings.get("station_field", ds.attrs.get("station_field", "station_id"))
            if station not in ds:
                issues.append("表格时间处理需配置可靠的processing.station_field分组字段")
            semantics = ds.attrs.get("temporal_semantics") or next((ds[v].attrs.get("temporal_semantics") for v in mapping.values() if v in ds), None)
            if semantics != "instantaneous":
                issues.append("现有宽表时间流程仅支持瞬时值内插；平均/累计量需明确时间边界并先整理为可用科学数据结构")
        if settings.get("target_spatial_resolution_km") and ds.attrs.get("crs") and CRS.from_user_input(ds.attrs["crs"]).is_projected:
            issues.append("投影坐标需先重投影为现有尺度模块支持的经纬度规则网格，不能仅将x/y改名为lat/lon")
        return issues

    def validate_plan(self, schema, inputs, assignments, plan):
        return None

    def execute(self, schema, inputs, assignments, analysis, plan, **options):
        started = time.perf_counter()
        files, outputs, records, failures = {}, {}, [], []
        by_file = {item.filename: item for item in inputs}
        roles = {role["role"]: role for role in schema.inputs}
        matches = {item["role"]: item for item in plan["matches"]}
        for name in plan["execution_order"]:
            role, match = roles[name], matches[name]
            if not match["can_process"]:
                continue
            if any(dep not in outputs for dep in role["depends_on"]):
                failures.append({"role": name, "reason": "前置输入处理失败，禁止绕过依赖"})
                continue
            settings = role["processing"] if isinstance(role["processing"], dict) else {}
            fusion = settings.get("fusion")
            if fusion:
                try:
                    variable = role["fields"][0]["name"]
                    request = TrialRequest(variable=variable, output_format=role["output_format"], fusion_method=fusion["method"],
                                           covariates=tuple(fusion.get("covariates", ())),
                                           spatial_method=settings.get("spatial_method", "bilinear"),
                                           target_time_frequency=settings.get("target_time_frequency"),
                                           target_spatial_resolution_km=settings.get("target_spatial_resolution_km"))
                    result = ConversionTaskEngine().execute([outputs[r] for r in fusion["sources"]], StructureRequest(variable=variable), request, fuse=True)
                    output = result.output
                    record = {"role": name, "source_roles": fusion["sources"], "operations": result.history,
                              "information_loss": "融合改变源数值；源角色分别输出，历史与来源SHA保留；不代表精度提高", "complete_model_input": False}
                    output.attrs.update(native_model_input="false", preparation_history=json.dumps(record, ensure_ascii=False, default=str))
                    content, filename, _ = export_structure(output, role["output_format"])
                    record["quality"] = validate_export(output, content, filename, role["output_format"])
                    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
                    output_filename = f"prepared_{safe_name}.{filename.rsplit('.', 1)[-1]}"
                    files[output_filename] = content
                    record["output_filename"] = output_filename
                    outputs[name] = output
                    records.append(record)
                except (SchemaError, ValueError, TypeError) as exc:
                    failures.append({"role": name, "reason": str(exc)})
                continue
            item = by_file[assignments[name]]
            ds = item.dataset.copy(deep=True)
            try:
                mapping = match["mapping"]
                if len(set(mapping.values())) != len(mapping):
                    raise SchemaError("同一源字段不能映射为多个不同物理变量")
                history = []
                for field in role["fields"]:
                    source = mapping.get(field["name"])
                    if not source:
                        continue
                    if source != field["name"] and field["name"] in ds.variables:
                        raise SchemaError("字段映射目标已存在，禁止覆盖")
                    actual = ds[source].attrs.get("units", "")
                    scale, offset = self.unit_transform(actual, field.get("unit", ""))
                    if scale != 1 or offset != 0:
                        from ecodata.engines.units import UnitEngine
                        ds[source] = UnitEngine().convert(ds[source], field["unit"])
                    ds[source].attrs.update(units=field.get("unit", actual), source_units=actual, source_field_name=source)
                    history.append({"operation": "field_and_unit_mapping", "source": source, "target": field["name"],
                                    "source_unit": actual, "target_unit": field.get("unit", ""), "scale": scale, "offset": offset,
                                    "information_loss": "单位仿射换算，无物理信息补造；浮点舍入受回读容差检查"})
                ds = ds.rename({source: target for target, source in mapping.items() if source != target})
                info = inspect_dataset(item.filename, ds)
                structure = StructureRequest(operation=settings.get("structure_operation", "preserve"), variable=role["fields"][0]["name"],
                                             x=info.x_coordinate or "", y=info.y_coordinate or "", time=info.time_coordinate or "", crs=info.crs or "")
                if settings.get("target_time_frequency") and info.structure == "table" and structure.operation == "preserve":
                    from dataclasses import replace
                    station = settings.get("station_field", ds.attrs.get("station_field", "station_id"))
                    if station not in ds:
                        raise SchemaError("表格时间处理需明确站点/分组字段：在processing.station_field配置，不可把不同站点混为一条时间序列")
                    structure = replace(structure, x=station)
                request = TrialRequest(variable=structure.variable, variables=tuple(mapping), output_format=role["output_format"],
                                       target_time_frequency=settings.get("target_time_frequency"),
                                       target_spatial_resolution_km=settings.get("target_spatial_resolution_km"),
                                       temporal_semantics=ds.attrs.get("temporal_semantics"), spatial_method=settings.get("spatial_method", "auto"))
                result = ConversionTaskEngine().execute([ds], structure, request)
                output = result.output
                history.extend(result.history)
                record = {"role": name, "source_file": item.filename, "sha256": item.sha256, "source_dimensions": dict(item.dataset.sizes),
                          "output_dimensions": dict(output.sizes), "source_metadata": dict(item.dataset.attrs), "operations": history,
                          "information_loss": "重采样/聚合改变采样与极值，不等同原始测量" if settings.get("target_time_frequency") or settings.get("target_spatial_resolution_km") else "格式/字段/单位转换；未请求丢弃原字段",
                          "complete_model_input": False}
                output.attrs.update(model_id=schema.profile["id"], preparation_role=name, native_model_input="false",
                                    preparation_history=json.dumps(record, ensure_ascii=False, default=str))
                content, filename, _ = export_structure(output, role["output_format"])
                check = validate_export(output, content, filename, role["output_format"])
                # Role identifiers are configuration keys, not filesystem paths.
                safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
                extension = filename.rsplit(".", 1)[-1]
                files[f"prepared_{safe_name}.{extension}"] = content
                record["output_filename"] = f"prepared_{safe_name}.{extension}"
                outputs[name] = output
                record["quality"] = check
                records.append(record)
            except (SchemaError, ValueError, TypeError) as exc:
                failures.append({"role": name, "reason": str(exc)})
        plan.update(runtime_seconds=time.perf_counter() - started, executed_steps=records, failed_inputs=failures)
        quality = {"passed": bool(outputs) and not failures, "scope": "standardized_preparation", "native_model_input": False,
                   "fields": {}, "failures": failures, "output_checks": {record["role"]: record["quality"] for record in records}}
        self.outputs = outputs
        return ScenarioResult(next(iter(outputs.values()), xr.Dataset()), xr.Dataset(), files, analysis, plan, quality)

    def output_roles(self, result, schema):
        return self.outputs

    def validation_history(self, result, role):
        return [operation for step in result.plan.get("executed_steps", []) if step["role"] == role
                for operation in step.get("operations", [])]

    def native_validation(self, result, schema):
        # A profile flag or a matching file extension is not validation evidence.
        return {"verified": False, "reason": "本Adapter未实现对应版本原生读取验收"}
