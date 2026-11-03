"""Read-only revalidation of preserved results; never fit or unseal test targets."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/taiwan_final_20261019"
OLD = ROOT / "output/taiwan_2017_preparation_20261019"
RUN = OLD / "golem_dag_smoke"
DATA = ROOT / "data/taiwan_2017_zenodo4420335"
EXPECTED_SIZE = 13435034939
EXPECTED_MD5 = "d7329012a175b3ad52012d1f3d6dd50f"


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def formal_gate(*, archive_verified, ground_time_verified, satellite_time_verified, coordinate_verified):
    conditions = dict(official_archive_md5=archive_verified is True, ground_timestamp_and_window=ground_time_verified is True,
        satellite_timestamp=satellite_time_verified is True, author_coordinate_datum=coordinate_verified is True)
    return dict(allowed=all(conditions.values()), conditions=conditions,
        blockers=[key for key, value in conditions.items() if not value])


def main():
    OUT.mkdir(exist_ok=True)
    inventory = json.loads((DATA / "source_manifest.json").read_text(encoding="utf-8"))
    members = []
    for entry in inventory["members"]:
        path = DATA / "provisional_members" / entry["member"]
        assert path.stat().st_size == entry["bytes"]
        assert digest(path) == entry["sha256"]
        members.append(dict(member=entry["member"], sha256=entry["sha256"], unchanged=True))
    archive = DATA / "aod_pm25_data.tar.gz"
    official = dict(expected_bytes=EXPECTED_SIZE, expected_md5=EXPECTED_MD5, status="INCOMPLETE", verified=False)
    if archive.exists():
        actual = digest(archive, "md5")
        official.update(bytes=archive.stat().st_size, md5=actual, verified=archive.stat().st_size == EXPECTED_SIZE and actual == EXPECTED_MD5)
        official["status"] = "VERIFIED" if official["verified"] else "CHECKSUM_FAILED"
    progress = json.loads((DATA / "range_progress.json").read_text(encoding="utf-8"))
    completed = sorted(set(progress["completed"]))
    official["accepted_http_range_bytes"] = progress["prefix_bytes"] + sum(min(16*1024*1024, EXPECTED_SIZE-start) for start in completed)
    official["completed_range_count"] = len(completed)
    official["historical_failed_range_attempts"] = len(progress["errors"])
    official["errors_are_history_not_current_missing_count"] = True
    partial = DATA / "aod_pm25_data.tar.gz.part"
    official["partial_file_bytes"] = partial.stat().st_size if partial.exists() else None

    prediction = pd.read_csv(RUN / "predictions.csv")
    comparison = pd.read_csv(RUN / "comparison.csv")
    split = pd.read_csv(RUN / "station_split.csv")
    roles = split.set_index("station_id").split
    assert prediction.station_id.map(roles).eq("validation").all()
    assert not set(prediction.station_id) & set(split.loc[split.split.eq("test"), "station_id"])
    summaries = []
    for (experiment, method), g in prediction.groupby(["experiment", "method"]):
        e = g.predicted-g.observed
        values = dict(RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(abs(e))), Bias=float(e.mean()))
        saved = comparison.loc[comparison.experiment.eq(experiment) & comparison.method.eq(method)].iloc[0]
        for name, value in values.items():
            np.testing.assert_allclose(value, saved[name], rtol=1e-10)
        total = float(np.sum((g.observed-g.observed.mean())**2))
        r2 = 1-float(np.sum(e**2))/total
        np.testing.assert_allclose(r2, saved.R2, rtol=1e-10)
        summaries.append(dict(experiment=experiment, method=method, n=len(g), **values, R2=r2))

    files = []
    for file in sorted(RUN.rglob("predictions.nc")):
        method, timestamp, experiment = file.parent.name, file.parent.parent.name, file.parent.parent.parent.name
        time = pd.to_datetime(timestamp, format="%Y-%m-%d_%H%M%S")
        rows = prediction.loc[prediction.experiment.eq(experiment) & prediction.method.eq(method) & pd.to_datetime(prediction.time).eq(time)]
        with xr.open_dataset(file) as ds:
            assert ds.pm25.attrs["units"] == "ug m-3"
            assert np.isfinite(ds.pm25).all() and (ds.pm25 >= 0).all()
            np.testing.assert_allclose(ds.pm25.values, rows.predicted.to_numpy())
            metadata = split.set_index("station_id").loc[rows.station_id]
            np.testing.assert_allclose(ds.longitude.values, metadata.longitude.to_numpy())
            np.testing.assert_allclose(ds.latitude.values, metadata.latitude.to_numpy())
            quality = json.loads((file.parent / "quality.json").read_text(encoding="utf-8"))
            assert quality["result"]["status"] == "WARN"
            files.append(dict(file=str(file.relative_to(ROOT)), points=ds.sizes["sample"], finite=True,
                unit_correct=True, coordinates_match_station_table=True, datum_verified=False,
                sha256=digest(file), grid_coverage="NOT_APPLICABLE_POINT_OUTPUT"))
    assert len(files) == 32

    uci = ROOT / "output/three_experiments_20261019/uci"
    previous = json.loads((OLD / "uci_reproducibility_check.json").read_text(encoding="utf-8"))
    for name, expected in previous["hashes"].items():
        assert digest(uci / name) == expected
    uci_result = dict(status="REUSED_VERIFIED_OUTPUTS", metrics_rows=previous["metric_rows"], retrained=False,
        hashes_unchanged=True, independent_recompute_evidence=str((OLD / "uci_reproducibility_check.json").relative_to(ROOT)))
    pool = ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"
    assert digest(pool) == "3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28"
    old_code = json.loads((OLD / "actual_output_verification.json").read_text(encoding="utf-8"))["code_sha256"]
    unchanged = {path:digest(ROOT / path) == checksum for path, checksum in old_code.items()}
    preserved_before_fixes = {}
    for path, same in unchanged.items():
        if not same:
            backup = OUT / "source_before_fixes" / Path(path).name
            assert backup.exists() and digest(backup) == old_code[path]
            preserved_before_fixes[path] = dict(file=str(backup.relative_to(ROOT)), sha256=old_code[path])
    regression = []
    for file in OUT.glob("full_regression*.xml"):
        root = ElementTree.parse(file).getroot()
        suite = root.find("testsuite")
        regression.append(dict(file=str(file.relative_to(ROOT)), **suite.attrib,
            skipped_tests=[dict(name=case.attrib["name"], reason=case.find("skipped").attrib.get("message"))
                           for case in root.iter("testcase") if case.find("skipped") is not None]))
    contract = formal_gate(archive_verified=official["verified"], ground_time_verified=False,
        satellite_time_verified=False, coordinate_verified=False)
    full_sources = None
    full_manifest = DATA / "verified_taiwan_source_manifest.json"
    if full_manifest.exists():
        complete = json.loads(full_manifest.read_text(encoding="utf-8"))
        assert official["verified"] and complete["archive_verified"] is True
        assert complete["official_md5"] == EXPECTED_MD5
        for entry in complete["taiwan_members"] + complete["documentation_members"]:
            assert (DATA / entry["file"]).stat().st_size == entry["bytes"]
        headers = json.loads((OUT / "header_audit_verified_archive.json").read_text(encoding="utf-8"))
        assert headers["all_arrays_read"] is True
        assert headers["files"] == sum(entry["member"].endswith(".nc") for entry in complete["taiwan_members"])
        full_sources = dict(taiwan_files=len(complete["taiwan_members"]),
            taiwan_bytes=sum(entry["bytes"] for entry in complete["taiwan_members"]),
            documentation_files=len(complete["documentation_members"]),
            actual_satellite_array_read_count=headers["files"],
            filename_dates_not_verified_observation_times=len(headers["filename_dates_not_verified_observation_times"]),
            mean_valid_pixel_fraction=headers["mean_valid_pixel_fraction"],
            clock_variables=headers["files_with_clock_variables"],
            clock_attributes=headers["files_with_clock_attributes"],
            explicit_global_datum=headers["files_with_explicit_global_datum"],
            formal_time_and_datum_verified=False)
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), official_archive=official,
        full_verified_source_inventory=full_sources,
        local_member_fingerprints=dict(count=len(members), all_unchanged=True, not_substitute_for_official_md5=True),
        uci=uci_result, readback_files=files, metrics=summaries, immutable_prior_code=unchanged, prior_source_backups=preserved_before_fixes,
        frozen_pool_unchanged=True, final_test_targets_used=False, source_datum_or_time_inferred_from_numeric_correlation=False,
        formal_contract=contract, regression=regression, disk_free_bytes=shutil.disk_usage(ROOT).free,
        new_deletions=0, formal_experiments_rerun=False, formal_abc_run=False)
    (OUT / "delivery_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(readbacks=len(files), source_members=len(members), formal_allowed=contract["allowed"], blockers=contract["blockers"])), flush=True)


if __name__ == "__main__":
    main()
