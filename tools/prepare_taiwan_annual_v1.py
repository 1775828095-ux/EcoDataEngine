"""Full-year read-only matching under the frozen working contract."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from taiwan_working_contract import CONTRACT, CONFIG, DATA, OUT, PROJECTION, aod_values, ground, save, scan_hour, sha, sites


def satellite():
    station = sites()
    inventory = json.loads((DATA / "verified_taiwan_source_manifest.json").read_text(encoding="utf-8"))
    files = sorted([item for item in inventory["taiwan_members"] if item["member"].endswith(".nc")], key=lambda item:item["member"])
    cache, rows, audits = {}, [], []
    for index, entry in enumerate(files):
        path = DATA / entry["file"]
        with h5py.File(path) as ds:
            instant, hour, origin = scan_hour(path, dict(ds.attrs))
            lon, lat = ds["Longitude"][...].ravel(), ds["Latitude"][...].ravel()
            grid = hashlib.sha256(lon.tobytes()+lat.tobytes()).hexdigest()
            if grid not in cache:
                valid_coords = np.isfinite(lon) & np.isfinite(lat) & (abs(lon) <= 180) & (abs(lat) <= 90)
                indices = np.flatnonzero(valid_coords)
                x, y = PROJECTION.transform(lon[valid_coords], lat[valid_coords])
                tree = cKDTree(np.column_stack([x, y]))
                neighbors = tree.query_ball_point(station[["easting", "northing"]].to_numpy(), CONTRACT["satellite"]["neighborhood_metres"])
                cache[grid] = [indices[near] for near in neighbors]
            values, valid = aod_values(ds["AOD550"][...], ds["QCAll"][...], dict(ds["AOD550"].attrs))
            values, valid = values.ravel(), valid.ravel()
            audits.append(dict(file=entry["member"], sha256=entry["sha256"], nominal_time=instant,
                time=hour, time_source=origin, valid_pixels=int(valid.sum()), total_pixels=len(valid), grid_sha256=grid))
            for station_id, near in zip(station.station_id, cache[grid]):
                accepted = near[valid[near]]
                rows.append(dict(station_id=station_id, time=hour, nominal_time=instant,
                    aod=float(values[accepted].mean()) if len(accepted) else np.nan,
                    valid_pixels=len(accepted), source_file=entry["member"], time_source=origin,
                    reason="VALID" if len(accepted) else "NO_QA_AOD_IN_NEIGHBORHOOD"))
        if (index+1) % 500 == 0:
            print(f"AHI read and matched {index+1}/{len(files)}", flush=True)
    scans = pd.DataFrame(rows).sort_values(["station_id", "time", "nominal_time", "source_file"]).reset_index(drop=True)
    if scans.duplicated(["station_id", "nominal_time"]).any():
        raise ValueError("Duplicate satellite scan for station/time")
    good = scans.loc[scans.reason.eq("VALID")]
    hourly = good.groupby(["station_id", "time"], as_index=False, sort=True).agg(
        aod=("aod", "mean"), valid_scan_count=("aod", "size"), valid_pixel_count=("valid_pixels", "sum"))
    return scans, hourly, pd.DataFrame(audits)


def canonical_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Actually recompute complete raw matching and compare")
    args = parser.parse_args()
    verification = json.loads((DATA / "download_verification.json").read_text())
    assert verification["verified"] is True and verification["md5"] == "d7329012a175b3ad52012d1f3d6dd50f"
    assert (DATA/"aod_pm25_data.tar.gz").stat().st_size == verification["bytes"] == 13435034939
    target = OUT / "data"
    if not args.verify:
        target.mkdir(parents=True, exist_ok=False)
        save(OUT / "contract_frozen.json", dict(contract=CONTRACT, contract_sha256=sha(CONFIG),
            frozen_before_this_round_model_results=True, reason="Explicit user rules, not historical metadata proof"))
    else:
        assert json.loads((OUT / "contract_frozen.json").read_text(encoding="utf-8"))["contract_sha256"] == sha(CONFIG)
    station = sites()
    observed = ground()
    scans, hourly, source_audit = satellite()
    matched = observed.merge(hourly, on=["station_id", "time"], how="left", validate="one_to_one")
    matched["match_reason"] = np.select([matched.reason.ne("VALID"), matched.aod.isna()],
        [matched.reason, "NO_QA_AOD_IN_SAME_HOUR"], default="VALID")
    matched = matched.sort_values(["station_id", "time"]).reset_index(drop=True)
    products = {"station_metadata": station, "ground_non_test": observed, "scans": scans,
        "aod_hourly_all_sites": hourly, "matched_non_test": matched, "source_scan_audit": source_audit}
    fingerprints = {name:canonical_hash(frame) for name, frame in products.items()}
    if args.verify:
        previous = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        assert fingerprints == previous["canonical_table_sha256"], "Raw re-computation is not identical"
        save(OUT / "reproducibility.json", dict(status="PASS", full_raw_scans_recomputed=len(source_audit),
            canonical_table_sha256=fingerprints, same_source_and_contract=True, contract_sha256=sha(CONFIG)))
        print("PASS: all full-year products independently recomputed identically", flush=True)
        return
    for name, frame in products.items():
        frame.to_parquet(target / f"{name}.parquet", index=False)
    matched["quarter"] = ((matched.month-1)//3+1)
    for key in ["month", "quarter", "station_id", "split"]:
        grouped = matched.groupby(key, observed=True).agg(records=("source_row", "size"),
            valid_ground=("reason", lambda x:int(x.eq("VALID").sum())),
            matched_samples=("match_reason", lambda x:int(x.eq("VALID").sum())))
        grouped["coverage_of_valid_ground"] = grouped.matched_samples/grouped.valid_ground
        grouped.to_csv(target / f"coverage_by_{key}.csv")
    matched.groupby(["month", "split", "match_reason"], observed=True).size().to_csv(target / "monthly_exclusions.csv")
    save(target / "manifest.json", dict(contract_version=CONTRACT["version"], contract_sha256=sha(CONFIG),
        source_archive_md5="d7329012a175b3ad52012d1f3d6dd50f", full_input_scans=len(source_audit),
        canonical_table_sha256=fingerprints, output_file_sha256={name:sha(target/f"{name}.parquet") for name in products},
        non_test_ground_records=len(observed), common_valid_samples=int(matched.match_reason.eq("VALID").sum()),
        test_targets_parsed=False, test_aod_covariates_prepared=True, author_definitions_verified=False,
        original_fields_and_source_rows_preserved=True, no_imputation=True))
    print("Full-year working data saved", target, "matched", matched.match_reason.eq("VALID").sum(), flush=True)


if __name__ == "__main__":
    main()
