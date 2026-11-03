#!/usr/bin/env python3
"""Run conservative standardized preparation for the five target models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ecodata.models import RequirementStore
from ecodata.models.scenario import (
    apply_field_mappings,
    execute_scenario,
    plan_scenario,
    recognise_inputs,
    suggest_roles,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path(r"C:\Users\ASUS\Documents\Codex\2026-10-02\wo\outputs")
GENERAL = OUTPUTS / "general_test_data"
RESULTS = OUTPUTS / "platform_results"

INPUTS = {
    "cmaq": [
        GENERAL / "netcdf" / "pm25_background.nc",
        GENERAL / "netcdf" / "geos_cf_meteorology.nc",
        GENERAL / "geotiff" / "dem.tif",
        GENERAL / "csv" / "pm25_station.csv",
    ],
    "calpuff": [
        GENERAL / "netcdf" / "geos_cf_meteorology.nc",
        GENERAL / "geotiff" / "dem.tif",
        GENERAL / "geotiff" / "NLCD_2021_02070009_small_100x100.tif",
        GENERAL / "csv" / "calpuff_receptors_latlon.csv",
    ],
    "swat_plus": [
        GENERAL / "geotiff" / "dem.tif",
        GENERAL / "geotiff" / "NLCD_2021_02070009_small_100x100.tif",
        GENERAL / "netcdf" / "soilgrids_02070009_small_100x100.nc",
        GENERAL / "netcdf" / "daymet_2020_2021.nc",
    ],
    "wrf": [
        GENERAL / "geotiff" / "dem.tif",
        GENERAL / "netcdf" / "geos_cf_meteorology.nc",
    ],
    "camx": [
        GENERAL / "netcdf" / "geos_cf_meteorology.nc",
        GENERAL / "netcdf" / "pm25_background.nc",
    ],
}

DECLARATIONS = {
    "pm25_station.csv": {
        "temporal_semantics": "mean",
        "units": {"longitude": "degree_east", "latitude": "degree_north"},
    },
    "calpuff_receptors_latlon.csv": {
        "units": {"longitude": "degree_east", "latitude": "degree_north"},
    },
    "NLCD_2021_02070009_small_100x100.tif": {"units": {"value": "class_id"}},
}

FIXED_ASSIGNMENTS = {
    "calpuff": {
        "terrain": "dem.tif",
        "land_use": "NLCD_2021_02070009_small_100x100.tif",
        "receptors": "calpuff_receptors_latlon.csv",
    },
    "swat_plus": {
        "topography": "dem.tif",
        "landuse": "NLCD_2021_02070009_small_100x100.tif",
        "weather": "daymet_2020_2021.nc",
    },
}

FIXED_MAPPINGS = {
    "calpuff": {
        "land_use": {"land_cover_class": "value"},
        "receptors": {"longitude": "longitude", "latitude": "latitude"},
    },
    "swat_plus": {
        "landuse": {"land_cover_class": "value"},
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_one(model_id: str) -> dict[str, Any]:
    profile = RequirementStore().get(model_id)
    target = RESULTS / model_id
    target.mkdir(parents=True, exist_ok=True)
    paths = INPUTS[model_id]
    files = [(path.name, path.read_bytes()) for path in paths]
    inputs = recognise_inputs(files, DECLARATIONS)
    assignments = suggest_roles(profile, inputs)
    assignments.update(FIXED_ASSIGNMENTS.get(model_id, {}))
    mappings = FIXED_MAPPINGS.get(model_id, {})
    inputs = apply_field_mappings(profile, inputs, assignments, mappings)
    analysis, plan = plan_scenario(profile, inputs, assignments)
    write_json(target / "input_analysis.json", analysis)
    write_json(target / "processing_plan.json", plan)

    status: dict[str, Any] = {
        "model_id": model_id,
        "model_status": profile["status"],
        "scenario_status": profile.get("preparation_scenario", {}).get("status"),
        "native_model_input": False,
        "assignments": assignments,
        "can_process": plan["can_process"],
        "can_process_partial": plan["can_process_partial"],
        "errors": plan["errors"],
        "native_gaps": plan["native_gaps"],
    }
    try:
        result = execute_scenario(profile, inputs, assignments)
        for name, content in result.files.items():
            (target / name).write_bytes(content)
        write_json(target / "quality_report.json", result.quality)
        status.update(
            executed=True,
            standardized_preprocessing_completed=result.quality["completion"]["standardized_preprocessing_completed"],
            native_model_input_verified=result.quality["completion"]["native_model_input_verified"],
            validation_status=result.quality["validation_status"],
            output_files=sorted(result.files),
        )
    except Exception as exc:
        status.update(executed=False, exception_type=type(exc).__name__, exception=str(exc))
    write_json(target / "status.json", status)
    return status


def main() -> None:
    summary = {}
    for model_id in INPUTS:
        try:
            summary[model_id] = run_one(model_id)
        except Exception as exc:
            summary[model_id] = {"executed": False, "exception_type": type(exc).__name__, "exception": str(exc)}
    write_json(RESULTS / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
