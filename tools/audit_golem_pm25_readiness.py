"""Read-only data preflight for the first real PM2.5 GOLEM experiment.

This does not assign an unproven CRS, evaluate held-out predictions or run search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


def station_profile(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = Counter((row["station_id"], row["time_as_stored"]) for row in rows)
    coordinates = defaultdict(set)
    stations = defaultdict(lambda: {"rows": 0, "valid_pm25": 0, "times": set(), "splits": set()})
    invalid_coordinates = negative = missing = 0
    times = set()
    for row in rows:
        sid, timestamp = row["station_id"], row["time_as_stored"]
        value = number(row["pm25_as_stored"])
        missing += value is None
        negative += value is not None and value < 0
        x, y = number(row["longitude_as_stored"]), number(row["latitude_as_stored"])
        invalid_coordinates += x is None or y is None or not (-180 <= x <= 180 and -90 <= y <= 90)
        if x is not None and y is not None:
            coordinates[sid].add((x, y))
        entry = stations[sid]
        entry["rows"] += 1
        entry["valid_pm25"] += value is not None and value >= 0
        entry["times"].add(timestamp)
        entry["splits"].add(row["split"])
        times.add(timestamp)
    ordered_times = sorted(times)
    from datetime import datetime
    gaps = Counter(int((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())
                   for a, b in zip(ordered_times, ordered_times[1:]))
    per_station = [{"station_id": sid, "rows": item["rows"], "valid_pm25": item["valid_pm25"],
                    "missing_hour_labels": len(times) - len(item["times"]),
                    "splits": sorted(item["splits"]),
                    "coordinate_pairs": sorted(coordinates[sid])}
                   for sid, item in sorted(stations.items())]
    positions = Counter(position for pairs in coordinates.values() for position in pairs)
    return {"source": fingerprint(path), "rows": len(rows), "stations": len(stations),
            "valid_numeric_pm25": len(rows) - missing, "missing_or_nonfinite_pm25": missing,
            "missing_fraction": missing / len(rows) if rows else None, "negative_pm25": negative,
            "duplicate_station_time_extra_rows": sum(count - 1 for count in keys.values()),
            "invalid_coordinate_rows": invalid_coordinates,
            "stations_with_changing_coordinates": [sid for sid, pairs in coordinates.items() if len(pairs) != 1],
            "shared_coordinate_positions": [position for position, count in positions.items() if count > 1],
            "split_station_counts": dict(Counter(next(iter(item["splits"])) for item in stations.values()
                                                 if len(item["splits"]) == 1)),
            "split_overlap_stations": [sid for sid, item in stations.items() if len(item["splits"]) != 1],
            "stored_time_start": min(times) if times else None, "stored_time_end": max(times) if times else None,
            "hour_labels": len(times), "time_interval_counts_seconds": dict(gaps),
            "crs": None, "crs_reason": "No datum declaration in source; numeric plausibility is not CRS evidence",
            "instrument_qa": "UNKNOWN; AQI category is not instrument QA", "per_station": per_station}


def readiness(profile, semantics):
    blockers = []
    if semantics.get("coordinate_datum") != "CONFIRMED" or not semantics.get("crs") or not semantics.get("coordinate_evidence"):
        blockers.append("STATION_COORDINATE_DATUM_UNCONFIRMED")
    if not semantics.get("unit_supported"):
        blockers.append("PM25_UNIT_UNCONFIRMED")
    if not semantics.get("time_supported"):
        blockers.append("OBSERVATION_TIME_WINDOW_UNCONFIRMED")
    for key in ("duplicate_station_time_extra_rows", "negative_pm25", "invalid_coordinate_rows"):
        if profile[key]:
            blockers.append(key.upper())
    if profile["split_overlap_stations"]:
        blockers.append("STATION_SPLIT_LEAKAGE")
    return {"status": "BLOCKED" if blockers else "PREFLIGHT_ONLY",
            "blockers": blockers, "search_permitted": not blockers,
            "prediction_accuracy_claimed": False}


def audit(root, data, consolidated, output):
    output.mkdir(parents=True, exist_ok=False)
    station = consolidated / "01_地面监测站" / "ground_pm25_201908_60stations.csv"
    metadata = consolidated / "01_地面监测站" / "ground_metadata_status.json"
    status = json.loads(metadata.read_text(encoding="utf-8"))
    profile = station_profile(station)
    evidence = status["evidence_status"]
    semantics = {"coordinate_datum": evidence["coordinate_datum"].upper(), "crs": status.get("confirmed_crs"),
                 "coordinate_evidence": status.get("coordinate_datum_evidence"),
                 "unit_supported": evidence["pm25_unit"].startswith("supported:"),
                 "time_supported": evidence["timezone"].startswith("supported:") and
                                   evidence["hour_window"].startswith("supported:"),
                 "unit": "ug m-3", "time_basis": "Asia/Shanghai",
                 "time_window": "one hour ending at stored label",
                 "evidence_file": fingerprint(metadata)}
    raw_paths = json.loads((data / "station_observations/metadata/station_schema_audit.json").read_text(encoding="utf-8"))["source_files"]
    with station.open(encoding="utf-8-sig", newline="") as stream:
        expected = {(r["station_id"], r["time_as_stored"]): number(r["pm25_as_stored"])
                    for r in csv.DictReader(stream)}
    raw_seen, raw_mismatches = set(), []
    for raw in raw_paths:
        with Path(raw).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["STATIONCODE"], row["DATATIME"]
                if key in expected:
                    raw_seen.add(key)
                    if number(row["PM25"]) != expected[key]:
                        raw_mismatches.append(key)
    if raw_mismatches or len(raw_seen) != len(expected):
        raise ValueError("Original city CSV and consolidated station values do not match")
    frozen_path = root / "output/capability_final_acceptance/frozen_generic_pool.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    candidates = [{"id": item["id"], "potential_station_candidate": item["id"].startswith("station."),
                   "executable_this_experiment": False,
                   "reason": "Dataset prerequisite BLOCK; frozen membership alone is not task compatibility"}
                  for item in frozen["algorithms"]]
    inventory = []
    for path in sorted(data.rglob("*")):
        if path.is_file():
            relative = path.relative_to(data)
            category = relative.parts[0]
            inventory.append({"path": str(path), "bytes": path.stat().st_size, "format": path.suffix.lower(),
                              "category": category,
                              "observation_kind": "model_computation" if category in ("background_geos_cf", "meteorology_era5")
                              else "satellite_auxiliary_not_pm25" if category.startswith("satellite_")
                              else "metadata_or_script"})
    import xarray as xr
    gridded = []
    for folder in ("background_geos_cf", "meteorology_era5"):
        for path in sorted((data / folder).rglob("*.nc")):
            if "daily_cache" in path.parts:
                continue
            with path.open("rb") as stream:
                engine = "h5netcdf" if stream.read(8) == b"\x89HDF\r\n\x1a\n" else "scipy"
            with xr.open_dataset(path, engine=engine) as ds:
                gridded.append({**fingerprint(path), "dimensions": dict(ds.sizes),
                    "variables": {name: {"units": value.attrs.get("units"), "dims": list(value.dims),
                                           "missing_fraction": float(value.isnull().mean().values)}
                                  for name, value in ds.data_vars.items()},
                    "coordinates": {name: {"first": str(value.values.flat[0]), "last": str(value.values.flat[-1])}
                                    for name, value in ds.coords.items() if value.size},
                    "use_in_this_experiment": "EXCLUDED: not independent station observations"})
    airnow = root / "examples/engineering_case_air_quality/observation/pm25_station.csv"
    with airnow.open(encoding="utf-8-sig", newline="") as stream:
        airnow_rows = list(csv.DictReader(stream))
    payload = {"requested_data_directory": str(data), "readiness": readiness(profile, semantics),
        "station_audit": profile, "semantics": semantics,
        "raw_crosscheck": {"original_files": [fingerprint(Path(path)) for path in raw_paths],
                           "matched_station_time_keys": len(raw_seen), "value_mismatches": len(raw_mismatches)},
        "frozen_pool": {"source": fingerprint(frozen_path), "count": len(candidates), "candidates": candidates,
                        "idw_in_frozen_pool": any("idw" in item["id"].lower() for item in candidates)},
        "alternate_local_observations": {"source": fingerprint(airnow), "rows": len(airnow_rows),
            "stations": len({r["station_id"] for r in airnow_rows}),
            "decision": "REJECT: two stations cannot identify 2D surface and independent train/validation/test station sets"},
        "gridded_file_audit": gridded, "requested_directory_inventory": inventory,
        "test_policy": "No test-station accuracy metric, tuning, fitting, selection or search was performed",
        "existing_use": "2019 station source already used in historical PM25 experiments; final test no longer prospective across project history",
        "golem_objective": "Existing tools/run_golem_search.py scores capability coverage, conflicts and costs, not real validation RMSE",
        "execution": {"pilot": "NOT_RUN_DATA_BLOCKED", "main_abc": "NOT_RUN_DATA_BLOCKED",
                      "ablations": "NOT_RUN_DATA_BLOCKED", "objective_calls": 0,
                      "validation_rmse": None, "test_rmse": None},
        "protected_files": [fingerprint(root / "src/ecodata/algorithms/golem_search.py"),
                            fingerprint(root / "tools/run_golem_search.py"), fingerprint(frozen_path)]}
    target = output / "preflight_audit.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    json.loads(target.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(target), "readiness": payload["readiness"],
                      "rows": profile["rows"], "stations": profile["stations"],
                      "valid_numeric_pm25": profile["valid_numeric_pm25"], "raw_keys_checked": len(raw_seen)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--consolidated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(Path(__file__).resolve().parents[1], args.data, args.consolidated, args.output)
