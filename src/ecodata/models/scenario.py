"""Generic model preparation orchestrator, retaining the previous public API."""
import hashlib
import json

import pandas as pd
from rasterio.crs import CRS

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters import get_adapter
from ecodata.models.contracts import ModelSchema, ScenarioInput, ScenarioResult
from ecodata.models.matching import ModelRequirementMatcher
from ecodata.models.planning import ProcessingPlanner
from ecodata.recognition import recognise_file, inspect_dataset
from ecodata.evaluation.self_check import PostConversionValidator, require_usable, validation_files


def scenario_schema(profile):
    return ModelSchema.from_profile(profile).preparation


def recognise_inputs(files, declarations=None):
    """Declarations confirm missing metadata; they do not create physical data."""
    declarations, inputs = declarations or {}, []
    for filename, content in files:
        if filename in {item.filename for item in inputs}:
            raise SchemaError("上传文件重名，请改名后上传，避免角色混淆")
        ds, info = recognise_file(filename, content)
        confirmed = declarations.get(filename, {})
        ds.attrs["user_confirmations"] = json.dumps(confirmed, ensure_ascii=False)
        if confirmed.get("crs"):
            ds.attrs["source_crs_declared"] = ds.attrs.get("crs", "")
            ds.attrs["crs"] = CRS.from_user_input(confirmed["crs"]).to_wkt()
        for variable, unit in confirmed.get("units", {}).items():
            if variable not in ds:
                raise SchemaError(f"{filename} 没有字段 {variable}")
            ds[variable].attrs.update(source_units_declared=ds[variable].attrs.get("units", ""), units=unit)
        if info.structure == "table" and "unit" in ds:
            units = {str(value).strip() for value in ds.unit.values if not pd.isna(value)}
            candidates = [v for v in info.variables if info.semantic_types.get(v) == "continuous" and v not in {info.x_coordinate, info.y_coordinate}]
            # A single measurement's explicit unit column is reliable; a mixed
            # table's one unit must never be assigned to all physical variables.
            if len(units) == 1 and len(candidates) == 1 and not ds[candidates[0]].attrs.get("units"):
                ds[candidates[0]].attrs.update(units=next(iter(units)), unit_evidence="explicit unit column; unique measurement")
        if confirmed.get("temporal_semantics"):
            ds.attrs["source_temporal_semantics_declared"] = ds.attrs.get("temporal_semantics", "")
            ds.attrs["temporal_semantics"] = confirmed["temporal_semantics"]
        inputs.append(ScenarioInput(filename, ds, hashlib.sha256(content).hexdigest()))
    return inputs


def _components(profile):
    schema = ModelSchema.from_profile(profile)
    adapter = get_adapter(schema)
    return schema, adapter, ModelRequirementMatcher(schema, adapter)


def suggest_roles(profile, inputs):
    _, _, matcher = _components(profile)
    return matcher.suggest(inputs)


def apply_field_mappings(profile, inputs, assignments, mappings):
    """Confirm aliases only after checking original physical evidence."""
    from ecodata.fusion.planning import DataRoleClassifier
    schema, _, _ = _components(profile)
    result = []
    for item in inputs:
        ds = item.dataset.copy(deep=False)
        rename = {}
        for role in schema.inputs:
            if assignments.get(role["role"]) != item.filename:
                continue
            mapping = mappings.get(role["role"], {})
            check = DataRoleClassifier().compatibility(ds, role, mapping)
            if not check["passed"]:
                raise SchemaError("；".join(check["reasons"]))
            for target, source in mapping.items():
                if not source or source.startswith("derived:") or source == target:
                    continue
                if source not in ds or target in ds or (source in rename and rename[source] != target):
                    raise SchemaError("字段映射缺失、冲突或覆盖已有变量，禁止继续")
                rename[source] = target
        if rename:
            ds = ds.rename(rename)
            ds.attrs = dict(ds.attrs, confirmed_field_mapping=json.dumps(rename, ensure_ascii=False))
            for source, target in rename.items():
                ds[target].attrs = dict(ds[target].attrs, source_field_name=source)
        result.append(ScenarioInput(item.filename, ds, item.sha256))
    return result


def _plan(schema, adapter, matcher, inputs, assignments):
    analysis = []
    for item in inputs:
        info = inspect_dataset(item.filename, item.dataset).as_dict()
        info.update(sha256=item.sha256, source_metadata=dict(item.dataset.attrs))
        info["variable_metadata"] = {name: {"attrs": dict(value.attrs), "dimensions": list(value.dims), "dtype": str(value.dtype)} for name, value in item.dataset.data_vars.items()}
        if info["time_coordinate"]:
            times = pd.to_datetime(item.dataset[info["time_coordinate"]].values.ravel())
            info["time_range"] = [str(times.min()), str(times.max())]
        analysis.append(info)
    matches = matcher.match(inputs, assignments)
    plan = ProcessingPlanner(schema, adapter).build(inputs, assignments, matches)
    # Missing other files must not prevent preparing a compatible single role.
    # Never use this path to bypass incompatibility or a failed alignment check.
    provided = [m for m in matches if m["file"] and not next(r for r in schema.inputs if r["role"] == m["role"]).get("record_only")]
    if (not adapter.supports_partial and not plan["can_process"] and not plan["role_conflicts"]
            and provided and all(m["can_process"] and not any(v.startswith("derived:") for v in m["mapping"].values()) for m in provided)
            and any(m["required"] and m["classification"] == "missing" for m in matches)):
        plan.update(can_process_partial=True, partial_adapter="standardized",
                    steps=[{"stage": m["name"], "operation": "仅准备已匹配输入；不执行跨文件对齐或融合"} for m in provided])
    return analysis, plan


def plan_scenario(profile, inputs, assignments):
    return _plan(*_components(profile), inputs, assignments)


def execute_scenario(profile, inputs, assignments, **options):
    schema, adapter, matcher = _components(profile)
    analysis, plan = _plan(schema, adapter, matcher, inputs, assignments)
    if not plan["can_process"] and not plan["can_process_partial"]:
        raise SchemaError("；".join(plan["errors"]) or "没有可执行的准备任务")
    if plan.get("partial_adapter"):
        from ecodata.models.adapters.standardized import ModelAdapter
        adapter = ModelAdapter()
    result = adapter.execute(schema, inputs, assignments, analysis, plan, **options)
    postcheck = matcher.check_outputs(adapter.output_roles(result, schema))
    outputs = adapter.output_roles(result, schema)
    file_checks, reopened_outputs = {}, dict(outputs)
    expected_files = {step["output_filename"]: (step["role"], outputs[step["role"]]) for step in result.plan.get("executed_steps", []) if step.get("output_filename")}
    if "model_input.nc" in result.files:
        expected_files["model_input.nc"] = (None, result.output)
    if "fused_PM25.nc" in result.files:
        expected_files["fused_PM25.nc"] = (None, result.fused)
    for filename, (role, expected) in expected_files.items():
        restored, _ = recognise_file(filename, result.files[filename], record_read_source=False)
        check = PostConversionValidator().validate(expected, restored,
            kind="format" if filename.endswith((".nc", ".h5", ".hdf5")) else "structure",
            metadata_sidecar=filename.endswith((".csv", ".tsv", ".json", ".xlsx")))
        require_usable(check)
        file_checks[filename] = check.as_dict()
        if filename.endswith(".nc"):
            from ecodata.engines.validation import ValidationEngine
            file_checks[filename]["standards_compliance"] = ValidationEngine.compliance(result.files[filename],
                acdd="ACDD" in str(expected.attrs.get("Conventions", "")))
        if role:
            reopened_outputs[role] = restored
    postcheck = matcher.check_outputs(reopened_outputs)
    by_file = {item.filename: item.dataset for item in inputs}
    role_checks = {}
    for match in result.plan["matches"]:
        if match["role"] not in outputs:
            continue
        step = next((step for step in result.plan.get("executed_steps", []) if step["role"] == match["role"]), {})
        derived_sources = [outputs[role] for role in step.get("source_roles", [])]
        role_source = by_file.get(match["file"])
        if role_source is None and not derived_sources:
            continue
        role_source = role_source if role_source is not None else derived_sources[0]
        audit = PostConversionValidator().validate(role_source, outputs[match["role"]], kind="fusion" if derived_sources else "model_adaptation", sources=derived_sources[1:],
            history=adapter.validation_history(result, match["role"]),
            field_mapping={source: target for target, source in match.get("mapping", {}).items() if not source.startswith("derived:")},
            model_check={"complete": next(item["passed"] for item in postcheck["roles"] if item["role"] == match["role"]), "scope": "preparation_only", "native_verified": False})
        require_usable(audit)
        role_checks[match["role"]] = audit.as_dict()
    primary_role = schema.preparation.get("target_grid")
    if assignments.get(primary_role) not in by_file:
        primary_role = next((role for role in result.plan["execution_order"] if assignments.get(role) in by_file and role in outputs), None)
    source = by_file[assignments[primary_role]] if primary_role else result.output
    primary_match = next((match for match in result.plan["matches"] if match["role"] == primary_role), {})
    audit = PostConversionValidator().validate(source, result.output, kind="model_adaptation", sources=[item.dataset for item in inputs if item.dataset is not source],
        history=adapter.validation_history(result, primary_role),
        field_mapping={source: target for target, source in primary_match.get("mapping", {}).items() if not source.startswith("derived:")},
        model_check={**postcheck, "scope": "configured_preparation_requirements", "native_verified": False})
    audit.details["role_validation"] = role_checks
    audit.details["file_validation"] = file_checks
    for check in file_checks.values():
        audit.details["warnings"].extend(check["warnings"])
    if audit.passed and audit.details["warnings"]:
        audit.details["status"] = "WARN"
    audit.details["information_loss"]["warnings"] = audit.details["warnings"]
    if any(level == "FAIL" for key, level in audit.details["checks_by_category"].items() if key != "model_compatibility") or any(not item["passed"] for item in postcheck["roles"]):
        require_usable(audit)
    result.quality["validation_report"] = audit.as_dict()
    result.quality["validation_status"] = audit.status
    result.files.update(validation_files(audit))
    native = adapter.native_validation(result, schema)
    result.quality.update(postcheck=postcheck, completion={
        "standardized_preprocessing_completed": postcheck["complete"] and result.quality["passed"] and audit.passed,
        "model_format_matched": postcheck["complete"] and result.quality["passed"] and schema.preparation.get("format_scope") == "declared_requirements",
        "native_model_input_verified": native["verified"],
        "complete_model_input": postcheck["complete"] and native["verified"],
        "scope": "configured_preparation_requirements_only", "native_validation": native})
    if not postcheck["complete"]:
        result.quality["completion"]["model_format_matched"] = False
        result.plan["completion_warning"] = "仅输出可处理部分；未满足全部准备需求，禁止称为完整模型输入"
    if any(not item["passed"] for item in postcheck["roles"]):
        raise SchemaError("处理后需求复查失败：" + json.dumps(postcheck, ensure_ascii=False))
    result.plan["complete_model_input"] = result.quality["completion"]["complete_model_input"]
    # Reports are finalized only after output revalidation, including partial
    # results and failed tasks. No complete-model file is generated by default.
    for name, report in (("input_analysis.json", result.analysis), ("processing_plan.json", result.plan), ("quality_report.json", result.quality)):
        result.files[name] = json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return result
