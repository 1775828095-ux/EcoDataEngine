"""Execute available real-data preparation and stop formal fusion on missing evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xarray as xr

from audit_golem_pm25_readiness import fingerprint, readiness
from ecodata.algorithms.pm25_alignment import align_native_covariates, observation_centers, formal_blockers, require_formal_spatial_evidence
from ecodata.algorithms.dag import DAGError
from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def run(output):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    data = Path("D:/codex/数据/京津冀_PM25多源融合_201908")
    station_root = Path("D:/codex/数据/京津冀_PM25融合测试_201908/01_地面监测站")
    station_path, metadata_path = station_root/"ground_pm25_201908_60stations.csv", station_root/"ground_metadata_status.json"
    geos_path = data/"background_geos_cf/subset/geos_cf_pm25_201908_bth.nc"
    era_path = data/"meteorology_era5/anonymous_public/era5_201908_bth.nc"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_paths = [station_path, metadata_path, geos_path, era_path]
    sources_before = [fingerprint(p) for p in source_paths]
    frozen_pool = ROOT/"output/capability_final_acceptance/frozen_generic_pool.json"
    save(output/"experiment_config.json", {
        "scope":"Formal A-E evaluation only after evidence passes; no implicit exploratory CRS",
        "source_manifest": sources_before, "frozen_pool": fingerprint(frozen_pool),
        "station_split":"unchanged historical 40 train / 10 validation / 10 test",
        "time_window":"ground ending-hour label UTC+8 -> UTC midpoint; no index-only match",
        "grid":"GEOS native 0.25 degree, no upsampling",
        "test_labels":"never parsed before all prerequisites and final configurations freeze",
        "requested_search":{"seeds":[11,22,33],"candidate_evaluation_limit":8,"executed":False}})
    # Only coordinates, time and split are read for all roles. Test target values are never parsed.
    frame = pd.read_csv(station_path, usecols=["time_as_stored", "station_id", "split", "longitude_as_stored", "latitude_as_stored"])
    if frame.duplicated(["station_id", "time_as_stored"]).any():
        raise DAGError("Duplicate station/hour; do not average silently")
    split = frame[["station_id", "split", "longitude_as_stored", "latitude_as_stored"]].drop_duplicates()
    if split.station_id.duplicated().any() or split.groupby("split").size().to_dict() != {"test": 10, "train": 40, "validation": 10}:
        raise DAGError("Frozen split changed or coordinates unstable")
    old = json.loads((ROOT/"output/golem_multisource_round2_preparation/preparation_audit.json").read_text(encoding="utf-8"))
    previous = {r["station_id"]:r["splits"][0] for r in old["station"]["per_station"]}
    if dict(zip(split.station_id, split.split)) != previous:
        raise DAGError("Station roles differ from historical 40/10/10 split")
    split.to_csv(output/"station_split_unchanged.csv", index=False)
    frame["center_utc"] = observation_centers(frame.time_as_stored, metadata)
    with xr.open_dataset(geos_path, engine="h5netcdf") as handle:
        geos = handle.load()
    with xr.open_dataset(era_path, engine="h5netcdf") as handle:
        era = handle.load()
    centers = pd.DatetimeIndex(frame.center_utc.unique()).sort_values().intersection(geos.time.to_index())
    ends = era.valid_time.to_index()
    centers = centers[(centers-pd.Timedelta(minutes=30)).isin(ends)&(centers+pd.Timedelta(minutes=30)).isin(ends)]
    frame["time_eligible"] = frame.center_utc.isin(centers)
    frame["reason"] = np.where(frame.time_eligible, "time matched only; spatial not verified", "outside actual grid time coverage")
    frame.to_csv(output/"station_time_matches_no_test_labels.csv", index=False)
    # Numeric QA is restricted to train/validation. No test concentration diagnostics or metrics.
    visible = []
    with station_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test":
                continue
            visible.append({"station_id": row["station_id"], "time_as_stored": row["time_as_stored"],
                            "split": row["split"], "value": pd.to_numeric(row["pm25_as_stored"], errors="coerce")})
    visible = pd.DataFrame(visible)
    visible["center_utc"] = observation_centers(visible.time_as_stored, metadata)
    visible["time_eligible"] = visible.center_utc.isin(centers)
    visible["numeric_valid"] = np.isfinite(visible.value)&visible.value.ge(0)
    visible["eligible_valid"] = visible.time_eligible & visible.numeric_valid
    summary = visible.groupby("split").agg(rows=("value", "size"), numeric_valid=("numeric_valid", "sum"),
        time_eligible=("time_eligible", "sum"), eligible_valid=("eligible_valid", "sum"))
    summary.to_csv(output/"train_validation_quality.csv")
    aligned = align_native_covariates(geos, era, centers)
    for name in ["pm25_rh35_gcc", "t2m", "d2m", "u10", "v10", "sp", "tp"]:
        # Check temporal/physical input integrity without inventing a station CRS.
        raw = RawInputQualityGate().validate(aligned, variable=name, requires_spatial=False)
        if raw["status"] == "BLOCK":
            raise DAGError(f"Native input quality blocked: {name}")
        save(output/f"quality_{name}.json", raw)
    target = output/"native_grid_covariates_NOT_FUSION.nc"
    aligned.to_netcdf(target, engine="h5netcdf")
    checks = {}
    with xr.open_dataset(target, engine="h5netcdf") as reopened:
        xr.testing.assert_equal(aligned, reopened.load())
        for name in ["pm25_rh35_gcc", "t2m", "d2m", "u10", "v10", "sp", "tp"]:
            checks[name] = ResultQualityGate().validate(aligned, reopened, variable=name)
            if checks[name]["status"] == "BLOCK":
                raise DAGError(f"Roundtrip quality blocked: {name}")
    save(output/"native_grid_readback_quality.json", checks)
    blockers = formal_blockers(metadata, geos)
    guard_observed = False
    try:
        require_formal_spatial_evidence(metadata, geos)
    except DAGError as exc:
        guard_observed = True
        save(output/"formal_execution_rejection.json", {"error": str(exc), "prediction_attempted": False})
    if not guard_observed:
        raise DAGError("Evidence changed: inspect supported execution path before releasing test labels")
    candidates = [("A", "station IDW/GWR"), ("B", "raw GEOS RH35; diagnostic, NOT identical target"),
                  ("C", "training-only empirical GEOS-to-station calibration"),
                  ("D", "GEOS RH35 + ERA5 external-drift UK"), ("E", "automatic executable DAG selection"),
                  ("ablation_no_GEOS", "ERA5 + stations"), ("ablation_no_ERA5", "GEOS + stations")]
    pd.DataFrame([dict(id=k, method=v, status="BLOCKED_NOT_RUN",
                       reason="; ".join(b for b in blockers if "GEOS_VERTICAL" not in b or k not in {"A", "ablation_no_GEOS"}),
                       RMSE=None, MAE=None, Bias=None, R2=None, coverage=None, seconds=None) for k,v in candidates]).to_csv(output/"method_status_and_metrics.csv", index=False)
    sources_after = [fingerprint(p) for p in source_paths]
    assert sources_before == sources_after
    cells = int(aligned.sizes["time"]*aligned.sizes["lat"]*aligned.sizes["lon"])
    report = dict(status="PARTIAL_REAL_TEMPORAL_PREPARATION; FORMAL_SPATIAL_FUSION_BLOCKED", blockers=blockers,
                  source_manifest=sources_before, sources_unchanged=True, test_targets_parsed=False,
                  split_unchanged=True, common_windows=len(centers), grid_shape=dict(aligned.sizes),
                  native_cells_per_variable=cells, station_rows_outside_time=int((~frame.time_eligible).sum()),
                  variable_valid_fraction={n:float(aligned[n].mean()) for n in aligned if n.endswith("_valid")},
                  actual_first_center=str(centers.min()), actual_last_center=str(centers.max()),
                  spatial_overlap_not_computed=True, final_test_executions=0, search_runs=0,
                  geos_layer_metadata=dict(geos.lev.attrs), geos_layer_values=geos.lev.values.tolist(),
                  independent_full_field_truth=False, aligned_file=fingerprint(target), seconds=perf_counter()-started,
                  code_manifest=[fingerprint(Path(p)) for p in [__file__, ROOT/"src/ecodata/algorithms/pm25_alignment.py"]],
                  conclusions={"fusion_effectiveness":"NOT_EVALUATED", "automatic_composition_advantage":"NOT_EVALUATED",
                               "semantic_search_advantage":"NOT_EVALUATED"})
    save(output/"run_result.json", report)
    print(json.dumps(report, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
