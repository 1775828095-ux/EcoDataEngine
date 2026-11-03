"""Read-only round-two readiness audit; never fit models or inspect test targets."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from audit_golem_pm25_readiness import fingerprint, station_profile


ROOT = Path(__file__).resolve().parents[1]


def utc_center(label):
    # Existing provenance report supports Beijing time and preceding-hour means.
    return datetime.fromisoformat(label) - timedelta(hours=8, minutes=30)


def profile_grid(path):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        times = ds.get("time", ds.get("valid_time"))
        result = {**fingerprint(path), "dimensions": dict(ds.sizes),
                  "attributes": dict(ds.attrs), "variables": {},
                  "time_start_utc": str(times.values[0]), "time_end_utc": str(times.values[-1]),
                  "time_unique": bool(times.to_index().is_unique),
                  "time_increasing": bool(times.to_index().is_monotonic_increasing),
                  "coordinates": {}}
        for name in ("lat", "lon", "latitude", "longitude", "lev"):
            if name in ds.coords:
                a = ds[name]
                result["coordinates"][name] = {"min": float(a.min()), "max": float(a.max()),
                    "attrs": dict(a.attrs), "step": float(a[1]-a[0]) if a.size > 1 else None}
        for name, a in ds.data_vars.items():
            if np.issubdtype(a.dtype, np.number):
                v = a.values
                result["variables"][name] = {"dims": list(a.dims), "attrs": dict(a.attrs),
                    "dtype": str(a.dtype), "nonfinite": int((~np.isfinite(v)).sum()),
                    "size": int(v.size), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
        return result, {str(v.astype("datetime64[s]")) for v in times.values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--station", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output filename; previous evidence is not overwritten")
    geos, geos_times = profile_grid(args.data / "background_geos_cf/subset/geos_cf_pm25_201908_bth.nc")
    era, era_times = profile_grid(args.data / "meteorology_era5/anonymous_public/era5_201908_bth.nc")
    station = station_profile(args.station)
    with args.station.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Read target values only for completeness, never calculate held-out errors.
    counts = Counter()
    labels = set()
    for row in rows:
        center = utc_center(row["time_as_stored"])
        key = center.isoformat(timespec="seconds")
        if key not in geos_times:
            counts["outside_geos_rows"] += 1
            continue
        before, after = center-timedelta(minutes=30), center+timedelta(minutes=30)
        if any(t.isoformat(timespec="seconds") not in era_times for t in (before, after)):
            counts["missing_era5_endpoint_rows"] += 1
            continue
        labels.add(key)
        try:
            valid = np.isfinite(float(row["pm25_as_stored"]))
        except ValueError:
            valid = False
        counts[f"overlap_rows_{row['split']}"] += 1
        counts[f"finite_rows_{row['split']}"] += int(valid)
    satellite_path = args.data / "satellite_sentinel2_aot/sentinel2_aot_station_samples_201908.csv"
    with satellite_path.open(encoding="utf-8-sig", newline="") as stream:
        satellite = list(csv.DictReader(stream))
    clear = [r for r in satellite if r["qa_clear"].lower() == "true"]
    frozen = ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"
    catalog = json.loads(frozen.read_text(encoding="utf-8"))
    protected = [frozen, ROOT / "src/ecodata/algorithms/golem_search.py", ROOT / "tools/run_golem_search.py",
                 ROOT / "src/ecodata/algorithms/pipeline.py", ROOT / "tools/run_meteorology_golem_numeric.py"]
    split_ids = {role: sorted({r["station_id"] for r in rows if r["split"] == role})
                 for role in ("train", "validation", "test")}
    inner = split_ids["train"].copy()
    random.Random(20261018).shuffle(inner)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_directory": str(args.data.resolve()), "station_is_external_to_requested_directory": True,
        "station": station, "geos_cf": geos, "era5": era,
        "temporal_join_diagnostic": {"status": "TIME_COVERAGE_ONLY_NOT_FUSION_READY",
            "station_time_rule": "UTC+8 ending label -> UTC interval center: label minus 8h30m",
            "geos_rule": "HH:30 center of one-hour mean; exact center join, no time interpolation",
            "era5_rule": "Both adjacent instantaneous endpoints required; not an hourly observation mean",
            "precipitation_excluded": "tp accumulation convention must be separately verified",
            "counts": dict(counts), "overlap_hours": len(labels),
            "first_center_utc": min(labels), "last_center_utc": max(labels)},
        "satellite": {**fingerprint(satellite_path), "pairs": len(satellite), "qa_clear_pairs": len(clear),
            "qa_clear_stations": len({r['station_id'] for r in clear}),
            "qa_clear_dates": len({r['observation_time_utc'][:10] for r in clear}),
            "physical_variable": "dimensionless AOT at 550nm, NOT surface PM2.5",
            "station_extraction_datum_uncertain": True},
        "frozen_pool": {**fingerprint(frozen), "count": catalog["count"],
                        "ids": [a["id"] for a in catalog["algorithms"]]},
        "proposed_split": {"status": "PLANNED_NOT_SEARCHED", "station_ids": split_ids,
            "inner_fit_station_ids": sorted(inner[:28]), "inner_weight_station_ids": sorted(inner[28:]),
            "train_dates_beijing": ["2019-08-02", "2019-08-14"],
            "validation_dates_beijing": ["2019-08-15", "2019-08-21"],
            "test_dates_beijing": ["2019-08-22", "2019-08-31"], "inner_seed": 20261018,
            "input_at_evaluation_time": "40 input stations permitted; evaluation station targets excluded",
            "historical_test_exposure": "retrospective within-round isolation, not a new untouched dataset"},
        "proposed_graph": {"status": "DESIGN_ONLY_NOT_EXECUTABLE_DAG",
            "output": "station-observation-defined hourly PM2.5 analysis field, not RH35 output",
            "nodes": [
                {"id": "stations", "role": "observation", "parents": [], "operations": ["QA", "UTC interval bounds", "confirmed CRS"]},
                {"id": "geos", "role": "RH35 covariate", "parents": [], "operations": ["interval matching", "spatial covariate query"]},
                {"id": "era5", "role": "meteorological covariates", "parents": [], "operations": ["instantaneous-time interpolation explicitly labelled", "spatial covariate query"]},
                {"id": "estimate", "algorithm": "station.pykrige_universal", "parents": ["stations", "geos", "era5"],
                 "ports": {"observations": "stations", "auxiliaries": ["geos", "era5"]}},
                {"id": "quality", "parents": ["estimate"], "operations": ["nonnegative", "hull mask", "units/time/grid", "NetCDF readback"]}],
            "prospective_parameter_recipes": 24,
            "recipe_factors": "2 GEOS resampling x 2 ERA5 resampling x 2 covariate sets x 3 variogram models",
            "composition_diversity": "one established multivariate estimator topology; 24 settings are NOT 24 validated distinct algorithm compositions",
            "currently_verified_end_to_end_searchable_graphs": 0},
        "protected_code": [fingerprint(p) for p in protected],
        "readiness": {"status": "BLOCKED", "end_to_end_pilot_run": False, "objective_calls": 0,
            "blockers": ["STATION_DATUM_UNCONFIRMED", "MULTIBRANCH_NUMERICAL_EXECUTOR_NOT_IMPLEMENTED",
                         "SUFFICIENT_DISTINCT_COMPOSITION_SPACE_NOT_YET_VERIFIED"],
            "direct_fusion_additional_blockers": ["RH35_TO_OBSERVATION_TARGET_CONTRACT_UNRESOLVED",
                         "TRAIN_ONLY_BIAS_AND_WEIGHT_STATE_ROUTING_NOT_VERIFIED",
                         "RESIDUAL_EXTRAPOLATION_MUST_BE_MASKED"],
            "covariate_alternative": "Existing PyKrige Universal accepts GEOS RH35 and ERA5 as DISTINCT predictors; no physical equivalence or humidity conversion claimed",
            "historical_test_is_prospective": False, "test_metrics_calculated": False},
        "source_evidence": [str(args.data / "source_manifest.json"),
            str(args.station.parents[1] / "06_数据质量与匹配报告/地面监测站来源与时间语义核实报告.md"),
            str(ROOT / "output/golem_real_pm25_round1_preflight_checked/preflight_audit.json")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "overlap": result["temporal_join_diagnostic"],
                      "readiness": result["readiness"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
