"""Export evidence-based coverage, including gaps and model-only capabilities."""
import csv
import json
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

from ecodata.algorithms.capabilities import capability_catalog
from ecodata.algorithms.registry import create_default_registry

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/algorithm_library_expansion"
CAPABILITIES = {
    "C1": ("Continuous reconstruction", ["continuous_spatial_resampling"]),
    "C2": ("Categorical resampling", ["categorical_scale_processing", "categorical_aggregation"]),
    "C3": ("Category fractions", ["fractional_categorical_aggregation"]),
    "C4": ("Conservative regridding", ["spatial_conservation"]),
    "C5": ("Learned environmental reconstruction; C5a/C5b distinguished", ["arbitrary_scale_image_reconstruction", "sentinel5p_hyperspectral_super_resolution", "learned_climate_reconstruction_candidate", "learning_based_spatial_reconstruction", "learned_arbitrary_environmental_reconstruction"]),
    "C6": ("Station-to-grid", ["observation_interpolation"]),
    "C7": ("Topography-assisted downscaling", ["topography_assisted_climate_downscaling", "terrain_learning_downscaling_candidate", "terrain_auxiliary", "terrain_pressure_adjustment", "terrain_aware_weather_downscaling", "meteorological_downscaling"]),
    "C8": ("Multi-grid regridding", ["multi_grid_regridding"]),
    "C9": ("Time interpolation", ["temporal_interpolation"]),
    "C10": ("Time statistical aggregation", ["temporal_mean_resampling", "temporal_statistical_aggregation", "temporal_event_statistics"]),
    "C11": ("Accumulation intervals", ["temporal_accumulation"]),
    "C12": ("Extreme-value correction", ["extreme_value_correction", "extreme_quantile_extrapolation_candidate", "extreme_precipitation_downscaling_candidate"]),
    "C13": ("Irregular-time processing", ["irregular_temporal_processing", "irregular_time_interpolation", "continuous_time_representation", "irregular_sampling"]),
    "C14": ("Vertical interpolation", ["vertical_interpolation"]),
    "C15": ("Soil profile semantics", ["soil_profile_processing"]),
    "C16": ("Range validation", ["nonnegative_validation", "physical_range_validation"]),
    "C17": ("Conservation", ["spatial_conservation", "temporal_accumulation"]),
    "C18": ("Coarse-scale consistency", ["coarse_scale_consistency"]),
    "C19": ("Multi-source fusion", ["multi_source_fusion", "multi_source_precipitation_fusion"]),
    "C20": ("Bias correction", ["bias_correction", "distribution_adjustment"]),
    "C21": ("Residual fusion", ["observation_constraint"]),
    "C22": ("Quality-aware fusion", ["quality_aware_fusion"]),
    "C23": ("Auxiliary features / estimation", ["auxiliary_variable_estimation", "external_drift_estimation", "multiscale_feature_preparation", "pretrained_feature_encoding"]),
    "C24": ("Emission spatial allocation", ["emission_spatial_allocation"]), "C25": ("Emission temporal allocation", ["emission_temporal_allocation"]),
    "C26": ("Chemical species mapping", []), "C27": ("Initial conditions", []),
    "C28": ("Boundary conditions", []), "C29": ("Landuse/soil model mapping", []),
    "C30": ("Crop/management model mapping", []),
    "C5a": ("Learned climate spatial downscaling", ["scale_adaptive_climate_downscaling", "learning_based_spatial_reconstruction", "learned_climate_reconstruction_candidate"]),
    "C5b": ("Arbitrary/continuous learned reconstruction", ["arbitrary_scale_image_reconstruction", "learned_arbitrary_resolution_candidate", "learned_arbitrary_environmental_reconstruction"]),
    "C7a": ("Terrain auxiliary", ["terrain_auxiliary"]),
    "C7b": ("Terrain pressure adjustment", ["terrain_pressure_adjustment"]),
    "C7c": ("Terrain-aware weather downscaling", ["terrain_learning_downscaling_candidate", "terrain_aware_weather_downscaling", "topography_assisted_climate_downscaling"]),
    "C12a": ("Ordinary bias correction", ["bias_correction", "distribution_adjustment"]),
    "C12b": ("Extreme bias correction", ["extreme_value_correction", "extreme_quantile_extrapolation_candidate"]),
    "C12c": ("Generative extreme reconstruction", ["extreme_precipitation_downscaling_candidate"]),
    "C13a": ("Conventional temporal interpolation", ["temporal_interpolation"]),
    "C13b": ("Irregular temporal interpolation", ["irregular_time_interpolation", "irregular_temporal_processing", "irregular_sampling"]),
    "C13c": ("Continuous-time representation", ["continuous_time_representation"]),
}


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def evidence_status(spec, upstream):
    name = spec.id
    if name.startswith("v5.") or name == "soil.mpspline2":
        from ecodata.algorithms.expansion_v5 import verified
        executed = verified(name)
        passed = tests_pass(OUTPUT / "v5/targeted_tests.xml")
        return ("VERIFIED" if executed and passed and spec.status == "available" else "PARTIAL"), {
            "actual_official_execution": executed, "adapter_tests": passed,
            "scope": "Exact V5 input contract; emiproc is numerical fixture validation, not native model verification",
            "model_adapter_verified": False, "independent_environmental_accuracy": False}
    if name in {"downscale.climatelearn_t2m", "terrain.fords_temperature"}:
        manifest = OUTPUT / "v4/execution_manifest.json"
        expected = "climatelearn_pipeline_output.nc" if name == "downscale.climatelearn_t2m" else "fords_temperature_pipeline_output.nc"
        rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
        executed = any(r["algorithm"] == name and r["path"] == expected and r["status"] in {"PASS", "WARN"}
            and (OUTPUT / "v4" / r["path"]).is_file() for r in rows)
        verified = executed and spec.status == "available" and spec.verification_status == "VERIFIED"
        return ("VERIFIED" if verified else "PARTIAL"), {"official_source": bool(upstream), "pipeline_executed": executed,
            "scope": "specific real ERA5 temperature task only; no arbitrary scale or full ForDs claim", "independent_measurement_accuracy": False}
    if spec.verification_status == "RESEARCH_CANDIDATE":
        return "RESEARCH_CANDIDATE", {"official_source": bool(upstream), "actual_execution": False}
    if name == "temporal.dualtimesfield":
        return spec.verification_status, {"official_source": bool(upstream), "actual_execution": spec.verification_status == "VERIFIED",
            "evidence": "v3/dualtimesfield_execution.json", "environmental_accuracy_verified": False}
    manifest = OUTPUT / "v2_examples/execution_manifest.json"
    if manifest.exists() and name in {"bias.xsdba_extreme", "terrain.fords_pressure", "temporal.xclim_statistics"}:
        executions = [r for r in json.loads(manifest.read_text(encoding="utf-8")) if r["algorithm"] == name]
        valid = bool(executions) and all(r["status"] == "PASS" and (ROOT / r["path"]).is_file() for r in executions)
        checks = {"actual_official_execution": valid, "pinned_source": bool(upstream),
                  "explicit_license": spec.license in {"MIT", "Apache-2.0"},
                  "adapter_tests": tests_pass(OUTPUT / "v2_tests.xml"),
                  "independent_environmental_accuracy": False,
                  "scope": "executed component only; synthetic examples do not prove predictive skill"}
        return ("VERIFIED" if valid and checks["pinned_source"] and checks["adapter_tests"] and spec.status == "available" else "PARTIAL"), checks
    real = OUTPUT / "real_examples" / (name + ".json")
    official = False
    if name.startswith("spatial.xarray_"):
        official = tests_pass(OUTPUT / "xarray_official_tests.xml")
        if name.endswith("cubic"):
            official = False  # Downloaded reduction suite does not cover cubic interpolation.
    elif name.startswith("station.pykrige_"):
        proof = OUTPUT / "pykrige_official_examples.json"
        official = proof.exists() and all(item["returncode"] == 0 for item in json.loads(proof.read_text(encoding="utf-8"))) and name.endswith(("ordinary", "universal"))
    elif name.startswith("station.metpy_") or name == "vertical.metpy":
        official = tests_pass(OUTPUT / "metpy_official_tests.xml")
    elif name == "bias.xsdba_eqm":
        official = (OUTPUT / "xsdba_official_example.json").exists()
    checks = {"github_source": bool(upstream), "pinned_version_or_commit": bool(upstream and upstream.get("commit")),
              "explicit_license": bool(upstream and upstream.get("license")), "installed": spec.status == "available",
              "official_example_or_test": official, "real_input": real.exists(), "real_output": real.with_suffix(".nc").exists(),
              "adapter_called": real.exists(), "capability_registered": bool(spec.capabilities),
              "adapter_test_suite": tests_pass(OUTPUT / "adapter_tests.xml")}
    if spec.status != "available":
        return "BLOCKED" if spec.status not in {"research_only"} else "UNVERIFIED", checks
    return ("VERIFIED" if all(checks.values()) else "PARTIAL"), checks


def tests_pass(path):
    if not path.exists():
        return False
    suites = ET.parse(path).findall(".//testsuite")
    return bool(suites) and all(int(item.get("failures", 0)) == 0 and int(item.get("errors", 0)) == 0 for item in suites)


def requirements_for_role(role):
    role = role.lower()
    if "emission" in role:
        return ["C24", "C25", "C26"], "MODEL_SPECIFIC"
    if "initial" in role:
        return ["C1", "C9", "C27"], "MODEL_SPECIFIC"
    if "boundary" in role:
        return ["C1", "C9", "C28"], "MODEL_SPECIFIC"
    if "landuse" in role or "land_use" in role:
        return ["C2", "C3", "C29"], "MODEL_SPECIFIC"
    if "soil" in role:
        return ["C15", "C29"], "MODEL_SPECIFIC"
    if any(key in role for key in ("crop", "plant", "management")):
        return ["C30"], "MODEL_SPECIFIC"
    if any(key in role for key in ("hydrodynamic", "geometry", "segment", "grid_definition")):
        return [], "UPSTREAM_MODEL_OR_PROJECT_CONFIGURATION"
    if any(key in role for key in ("meteor", "weather", "forcing")):
        return ["C1", "C9", "C10", "C14"], "GENERIC_PREPROCESSING_ONLY"
    if any(key in role for key in ("terrain", "topograph")):
        return ["C1", "C7"], "GENERIC_PREPROCESSING_ONLY"
    if any(key in role for key in ("observation", "concentration", "inflow", "flow", "quality")):
        return ["C6", "C9", "C10", "C20"], "GENERIC_PREPROCESSING_ONLY"
    return [], "NEEDS_SEMANTIC_REVIEW"


def main():
    registry = create_default_registry()
    research = {r["id"]: r for r in json.loads((ROOT / "src/ecodata/algorithms/expansion_v2_catalog.json").read_text(encoding="utf-8"))}
    research.update({r["id"]: r for r in json.loads((ROOT / "src/ecodata/algorithms/expansion_v3_catalog.json").read_text(encoding="utf-8"))})
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in (OUTPUT / "upstream").glob("*/source_evidence.json")]
    lookup = {item["repository"].lower(): item for item in sources}
    records = {item.algorithm_id: item for item in capability_catalog(registry)}
    rows, lifecycle = [], []
    for spec in registry.specs():
        upstream = lookup.get((spec.source_url or "").lower())
        status, checks = evidence_status(spec, upstream)
        record = records[spec.id]
        lifecycle.append({"algorithm": spec.id, "status": status, "ten_checks": checks,
                          "repository": spec.source_url, "upstream": upstream.get("commit") if upstream else None})
        capabilities = record.capabilities or ("NO_CAPABILITY_DECLARED",)
        for capability in capabilities:
            codes = [code for code, (_, labels) in CAPABILITIES.items() if capability in labels]
            if spec.id in research:
                codes = list(dict.fromkeys(codes + research[spec.id]["capability_ids"]))
            row_status = "PARTIAL" if (capability == "external_drift_estimation" and spec.id == "station.pykrige_universal") or spec.id == "terrain.fords_pressure" else status
            rows.append({"Algorithm": spec.id, "Capability": capability, "CapabilityCode": ";".join(codes),
                "Input Type": ";".join(spec.input_kinds), "Output Type": json.dumps(spec.output_contract, ensure_ascii=False),
                "Domain": spec.category, "Constraints": ";".join(spec.preconditions + spec.physical_constraints),
                "Hardware": spec.compute, "Status": row_status, "Executable": record.executable,
                "Repository": spec.source_url, "Version": spec.version, "Commit": upstream.get("commit", "") if upstream else "",
                "License": spec.license, "Environment": "Windows/Python " + platform.python_version(),
                "Evidence": ";".join(spec.verification_evidence), "Notes": spec.notes,
                "Capability ID": ";".join(codes), "Capability Name": ";".join(CAPABILITIES[c][0] for c in codes),
                "Required Data Properties": ";".join(spec.preconditions), "Candidate Algorithms": spec.id,
                "Algorithm Type": spec.category, "Version/Commit": upstream.get("commit", spec.version) if upstream else spec.version,
                "Official Entry Point": research.get(spec.id, {}).get("entry", spec.official_entry_point or spec.backend),
                "Input Requirements": json.dumps(spec.input_contract, ensure_ascii=False),
                "Output": json.dumps(spec.output_contract, ensure_ascii=False), "Runtime Status": spec.status,
                "VERIFIED/PARTIAL/BLOCKED/UNVERIFIED": row_status,
                "Applicable Data Types": ";".join(spec.input_kinds),
                "Applicable Models": "Requirements-compatible preprocessing only; no native-model validation claimed",
                "Test Result": "v2_tests.xml and v2_examples/" if spec.id in {"bias.xsdba_extreme", "terrain.fords_pressure", "temporal.xclim_statistics"} else ("NOT_RUN" if spec.status != "available" else "See verification evidence")})
    # Separate broad and narrow capabilities: a verified pressure component must
    # not silently make the whole terrain-downscaling capability VERIFIED.
    expanded = []
    component_status = {item["algorithm"]: item["status"] for item in lifecycle}
    for row in rows:
        for code in row["CapabilityCode"].split(";") or [""]:
            item = dict(row)
            item["CapabilityCode"] = item["Capability ID"] = code
            item["Capability Name"] = CAPABILITIES[code][0] if code else row["Capability"]
            if row["Algorithm"] == "terrain.fords_pressure" and code == "C7b" and row["Executable"] and component_status.get(row["Algorithm"]) == "VERIFIED":
                item["Status"] = item["VERIFIED/PARTIAL/BLOCKED/UNVERIFIED"] = "VERIFIED"
            expanded.append(item)
    rows = expanded
    # Model-only execution evidence is not an unconditional GOLEM algorithm.
    model_manifest=ROOT/"output/capability_final_acceptance/model_execution_manifest.json"
    if model_manifest.exists():
        model_cases=json.loads(model_manifest.read_text(encoding="utf-8"))
        for case in model_cases:
            if not (ROOT/"output/capability_final_acceptance"/case["output"]).is_file():
                continue
            codes=case["capability"].split("/")
            for code in codes:
                if code not in CAPABILITIES:continue
                rows.append({"Algorithm":"MODEL_PREPROCESSOR:"+code,"Capability":CAPABILITIES[code][0],"CapabilityCode":code,
                    "Status":"VERIFIED","Executable":False,"Domain":"MODEL_PREPROCESSING",
                    "Capability ID":code,"Capability Name":CAPABILITIES[code][0],"Runtime Status":"explicit model preprocessor only",
                    "Evidence":"output/capability_final_acceptance/model_execution_manifest.json",
                    "NativeModelStatus":"NATIVE_UNVERIFIED",
                    "Notes":"Restricted S2S/legacy static profile/EDGAR contract only; not all models or current CMAQ5.5 mechanism",
                    "VERIFIED/PARTIAL/BLOCKED/UNVERIFIED":"VERIFIED"})
    for code, (description, _) in CAPABILITIES.items():
        if not any(code in row["CapabilityCode"].split(";") for row in rows):
            model_only = code[1:].isdigit() and int(code[1:]) >= 24
            rows.append({"Algorithm": "MODEL_SPECIFIC" if model_only else "NO_VERIFIED_CANDIDATE",
                "Capability": description, "CapabilityCode": code, "Status": "MODEL_SPECIFIC" if model_only else "UNVERIFIED",
                "Executable": False, "Domain": "MODEL_SPECIFIC" if model_only else "CAPABILITY_GAP",
                "Capability ID": code, "Capability Name": description,
                "Required Data Properties": "Model-specific schema/configuration" if model_only else "No verified candidate contract",
                "Candidate Algorithms": "MODEL_ADAPTER_ONLY" if model_only else "NONE_VERIFIED",
                "Runtime Status": "not a runnable algorithm node",
                "VERIFIED/PARTIAL/BLOCKED/UNVERIFIED": "MODEL_SPECIFIC" if model_only else "UNVERIFIED",
                "Test Result": "No generic algorithm execution evidence",
                "Notes": "Existing model adapters are separate; this row is not a GOLEM algorithm node"})
    csv_write(OUTPUT / "algorithm_capability_matrix.csv", rows)
    model_rows = []
    for path in sorted((ROOT / "src/ecodata/models/profiles").glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        inputs = profile.get("preparation_scenario", {}).get("inputs", []) or [
            {"role": role, "fields": []} for role in profile.get("required_variables", [])]
        for item in inputs:
            role = item.get("role", item.get("name", "UNKNOWN"))
            codes, scope = requirements_for_role(role)
            candidates = [row["Algorithm"] for row in rows if row["Executable"] and set(codes) & set(row["CapabilityCode"].split(";"))]
            model_rows.append({"Model": profile.get("name", path.stem), "Version": profile.get("reference_version", "UNKNOWN"),
                "Input": role, "Fields": ";".join(f.get("name", "") for f in item.get("fields", []) if isinstance(f, dict)),
                "DataNature": ";".join(sorted({f.get("data_type", "UNKNOWN") for f in item.get("fields", []) if isinstance(f, dict)})),
                "RequiredCapabilities": ";".join(codes), "CurrentAlgorithms": ";".join(sorted(set(candidates))),
                "Scope": scope, "Status": "PARTIAL" if candidates else "UNVERIFIED",
                "SourceSchema": str(path.relative_to(ROOT)), "NativeInputVerified": "not evaluated in this phase",
                "Notes": "Role-based routing audit; per-variable schema, units, topology and native tools remain necessary"})
    csv_write(OUTPUT / "model_data_capability_matrix.csv", model_rows)
    (OUTPUT / "algorithm_verification_evidence.json").write_text(json.dumps({"python": platform.python_version(),
        "environment": "Windows CPU; torch.cuda.is_available=False", "algorithms": lifecycle}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Registry", len(registry.specs()), "algorithm rows", len(rows), "model rows", len(model_rows))
    print(pd_summary(lifecycle))


def pd_summary(lifecycle):
    from collections import Counter
    return dict(Counter(item["status"] for item in lifecycle))


if __name__ == "__main__":
    main()
