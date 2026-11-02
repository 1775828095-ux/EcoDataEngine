"""Requirements are matched to data, not file suffixes or model names."""
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from ecodata.core.exceptions import SchemaError

from ecodata.recognition import inspect_dataset


def field_mapping(dataset, role):
    result = {}
    for field in role["fields"]:
        aliases = {str(value).casefold() for value in [field["name"], field.get("standard_name", field["name"]), *field.get("aliases", [])]}
        candidates = [name for name in dataset.data_vars if name.casefold() in aliases or str(dataset[name].attrs.get("standard_name", "")).casefold() in aliases or str(dataset[name].attrs.get("physical_variable", "")).casefold() in aliases]
        numeric = [name for name in candidates if dataset[name].dtype.kind in 'biuf' and np.isfinite(dataset[name].values).any()]
        if numeric:candidates=numeric
        if len(candidates) == 1:
            result[field["name"]] = candidates[0]
    return result


class ModelRequirementMatcher:
    def __init__(self, schema, adapter):
        self.schema, self.adapter = schema, adapter

    def suggest(self, inputs):
        result = {}
        for role in self.schema.inputs:
            candidates = []
            if role.get("record_only") or not role["fields"]:
                result[role["role"]] = ""
                continue
            for item in inputs:
                if not self.role_evidence_matches(item.dataset,role):
                    continue
                mapping = self.adapter.field_mapping(item.dataset, role)
                info = inspect_dataset(item.filename, item.dataset)
                dimensions = set(role.get("dimensions", []))
                table = info.structure == "table"
                fits = (not dimensions or ("record" in dimensions and table) or ("x" in dimensions and info.x_coordinate and info.y_coordinate))
                if "time" in dimensions:
                    fits = fits and bool(info.time_coordinate)
                if dimensions == {"x", "y"}:
                    fits = fits and not info.time_coordinate
                if fits and mapping and all(field["name"] in mapping for field in role["fields"] if field.get("required", True)):
                    exact_structure = ("record" in dimensions and table) or ("x" in dimensions and not table)
                    candidates.append((int(exact_structure), item.filename))
            best = [filename for score, filename in candidates if score == max((s for s, _ in candidates), default=0)]
            result[role["role"]] = best[0] if len(best) == 1 else ""
        return result

    @staticmethod
    def role_evidence_matches(dataset,role):
        return all(str(dataset.attrs.get(key,'')) in values for key,values in role.get('source_evidence',{}).items())

    def match(self, inputs, assignments):
        sources = {item.filename: item for item in inputs}
        matches = []
        for role in self.schema.inputs:
            filename = assignments.get(role["role"], "")
            missing, issues, operations = [], [], []
            match = {"role": role["role"], "name": role["name"], "file": filename,
                     "required": role["required"], "can_process": False, "mapping": {}, "fields": []}
            if role.get("record_only"):
                match.update(status="原生依赖，本辅助场景不生成", classification="missing" if not filename else "convertible",
                             reason="仅记录原生依赖，本准备场景不处理", required_for_native=True)
                matches.append(match)
                continue
            settings = role["processing"] if isinstance(role["processing"], dict) else {}
            if filename in sources and not self.role_evidence_matches(sources[filename].dataset,role):
                match.update(status='角色证据不足/不兼容',classification='missing',missing=['role_evidence'],reasons=['数据用途未确认，不能把观测自动当初始/边界/负荷'],operations=[])
                matches.append(match)
                continue
            if settings.get("fusion") and role["allow_derived"]:
                match.update(status="可以转换/标准化", classification="convertible", can_process=True,
                             missing=[], reasons=[], derived_from=settings["fusion"]["sources"],
                             operations=[{"operation": "fusion", **settings["fusion"]}])
                if len(role["fields"]) != 1 or role["fields"][0].get("data_type") != "continuous":
                    match.update(can_process=False, reasons=["现有融合任务一次处理一个明确的连续物理变量"])
                matches.append(match)
                continue
            if filename not in sources:
                match.update(status="缺少", classification="missing", missing=[role["name"]], reason="请上传数据；存在歧义时请确认输入角色")
                matches.append(match)
                continue
            ds = sources[filename].dataset
            info = inspect_dataset(filename, ds)
            mapping = self.adapter.field_mapping(ds, role)
            from ecodata.fusion.planning import DataRoleClassifier
            compatibility = DataRoleClassifier().compatibility(ds, role, mapping)
            match["role_compatibility"] = compatibility
            issues.extend(compatibility["reasons"])
            if not role["fields"]:
                issues.append("需求目前只有类别索引，需在模型配置中明确变量、单位及输出约束；不是认定物理数据缺失")
            for field in role["fields"]:
                operation_start = len(operations)
                target = field["name"]
                source = mapping.get(target)
                record = {"variable": target, "source": source, "required": field.get("required", True)}
                if not source:
                    aliases = {str(value).casefold() for value in [target, field.get("standard_name", target), *field.get("aliases", [])]}
                    candidates = [name for name in ds.data_vars if name.casefold() in aliases]
                    if len(candidates) > 1:
                        issues.append(f"{target} 对应多个字段，请明确字段映射，不能自动猜测")
                        record["classification"] = "convertible"
                    else:
                        if field.get("required", True):
                            missing.append(target)
                        record["classification"] = "missing"
                else:
                    try:
                        self.adapter.validate_field(ds, source, field)
                        if source.startswith("derived:"):
                            operations.append({"operation": "adapter_derivation", "variable": target, "source": source})
                        else:
                            actual = ds[source].attrs.get("units", "")
                            scale, offset = self.adapter.unit_transform(actual, field.get("unit", ""))
                            if scale != 1 or offset != 0 or actual != field.get("unit", ""):
                                operations.append({"operation": "unit_conversion", "variable": target, "source_unit": actual,
                                                   "target_unit": field.get("unit", ""), "scale": scale, "offset": offset})
                            if source != target:
                                operations.append({"operation": "field_mapping", "source": source, "variable": target})
                        record["classification"] = "convertible" if len(operations) > operation_start else "satisfied"
                    except (SchemaError, ValueError, TypeError) as exc:
                        issues.append(f"{target}：{exc}")
                        record["classification"] = "convertible"
                match["fields"].append(record)
            settings = role["processing"] if isinstance(role["processing"], dict) else {}
            if info.format != role["output_format"]:
                operations.append({"operation": "format_conversion", "source_format": info.format, "target_format": role["output_format"]})
            if role["output_format"] not in self.adapter.output_formats:
                issues.append(f"当前Adapter不输出 {role['output_format']}；文件有数据，不等于缺少物理变量")
            operation = settings.get("structure_operation", "preserve")
            if operation == "preserve" and info.structure == "table" and {"x", "y"} <= set(role.get("dimensions", [])) and info.x_coordinate and info.y_coordinate:
                operation = "table_to_grid"
                settings["structure_operation"] = operation
            if operation != "preserve":
                operations.append({"operation": "structure_conversion", "method": operation})
            expected = set(role.get("dimensions", []))
            actual_dims = set(ds[mapping[next(iter(mapping))]].dims) if mapping and not next(iter(mapping.values())).startswith("derived:") else set(ds.dims)
            axis_names = {"x": info.x_coordinate, "y": info.y_coordinate, "time": info.time_coordinate, "record": ds.attrs.get("row_dimension")}
            if expected and operation == "preserve":
                resolved = {axis_names.get(name, name) for name in expected}
                if None in resolved or resolved != actual_dims:
                    issues.append("维度不匹配；需配置已有结构转换，不能仅更改文件后缀")
            target_crs = settings.get("target_crs")
            if target_crs:
                if not info.crs:
                    issues.append("源CRS未知，需用户确认；不能补造坐标系")
                elif CRS.from_user_input(info.crs) != CRS.from_user_input(target_crs):
                    issues.append("需要CRS重投影；本Adapter未配置可执行的重投影步骤，请先用已有转换任务")
            for key in ("target_time_frequency", "target_spatial_resolution_km"):
                if settings.get(key):
                    operations.append({"operation": key, "target": settings[key]})
            if settings.get("target_time_frequency"):
                if not info.time_coordinate:
                    issues.append("缺少时间字段，不能重采样")
                if ds.attrs.get("temporal_semantics") not in {"instantaneous", "mean", "accumulation"} and not all(ds[v].attrs.get("temporal_semantics") in {"instantaneous", "mean", "accumulation"} for v in mapping.values() if v in ds):
                    issues.append("时间语义未确认，不能猜测瞬时/平均/累计")
            if settings.get("target_spatial_resolution_km") and not (info.crs and info.x_coordinate and info.y_coordinate):
                issues.append("缺少可靠空间坐标/CRS，不能进行空间重采样")
            issues.extend(self.adapter.role_issues(ds, role, mapping))
            classification = "missing" if missing else "convertible" if operations or issues else "satisfied"
            match.update(mapping=mapping, classification=classification, missing=missing, reasons=issues,
                         operations=operations, can_process=not missing and not issues,
                         status="缺少" if missing else "待确认/当前无法处理" if issues else "可以转换/标准化" if operations else "已满足")
            matches.append(match)
        return matches

    def check_outputs(self, outputs):
        """Recheck actual processed data, not the previously planned promises."""
        reports, missing = [], []
        for role in self.schema.inputs:
            if role.get("record_only"):
                continue
            ds = outputs.get(role["role"])
            issues = []
            if ds is None:
                if role["required"]:
                    missing.append(role["role"])
                continue
            for field in role["fields"]:
                name = field["name"]
                if name not in ds:
                    if field.get("required", True):
                        issues.append("缺少变量：" + name)
                    continue
                try:
                    self.adapter.validate_field(ds, name, field)
                    scale, offset = self.adapter.unit_transform(ds[name].attrs.get("units", ""), field.get("unit", ""))
                    if scale != 1 or offset != 0:
                        issues.append(name + " 单位尚未转换")
                except (SchemaError, ValueError, TypeError) as exc:
                    issues.append(str(exc))
                if ds[name].dtype.kind in "biuf" and not np.isfinite(ds[name].values).any():
                    issues.append(name + " 全部无效")
                if ds[name].dtype.kind in "biuf":
                    values = ds[name].values
                    for key, compare in (("valid_min", np.less), ("valid_max", np.greater)):
                        if field.get(key) is not None and np.any(compare(values[np.isfinite(values)], field[key])):
                            issues.append(name + " 不符合模型物理范围：" + key)
            info = inspect_dataset("output.nc", ds)
            expected_dims = role.get("dimensions", [])
            axes = {"x": info.x_coordinate, "y": info.y_coordinate, "time": info.time_coordinate, "record": ds.attrs.get("row_dimension")}
            resolved = {axes.get(name, name) for name in expected_dims}
            for field in role["fields"]:
                if expected_dims and field["name"] in ds and set(ds[field["name"]].dims) != resolved:
                    issues.append(field["name"] + " 输出维度不匹配")
            settings = role["processing"] if isinstance(role["processing"], dict) else {}
            if settings.get("target_time_frequency"):
                if not info.time_coordinate:
                    issues.append("输出时间字段丢失")
                else:
                    times = pd.DatetimeIndex(pd.to_datetime(ds[info.time_coordinate].values.ravel())).unique().sort_values()
                    if len(times) < 2 or not times.equals(pd.date_range(times[0], times[-1], freq=settings["target_time_frequency"])):
                        issues.append("输出时间分辨率与配置不一致或不足以验证")
            if settings.get("target_spatial_resolution_km"):
                actual = info.spatial_resolution_km
                if actual is None or not np.isclose(actual, settings["target_spatial_resolution_km"], rtol=.05):
                    issues.append("输出空间分辨率与配置不一致")
            if settings.get("target_crs") and (not info.crs or CRS.from_user_input(info.crs) != CRS.from_user_input(settings["target_crs"])):
                issues.append("输出CRS不匹配")
            reports.append({"role": role["role"], "passed": not issues, "issues": issues})
        return {"complete": not missing and all(item["passed"] for item in reports), "missing_roles": missing, "roles": reports}
