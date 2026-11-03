"""Final independent file checks and six tiny validation-only deterministic replays."""
import json
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd

import taiwan_annual_adapter as adapter
from taiwan_working_contract import CONFIG, CONTRACT, OUT, ROOT, frozen_selection, ground, save, sha, sites
from run_taiwan_annual_v1 import METHODS


def main():
    target = OUT/"final_verification.json"
    if target.exists():
        raise ValueError("Final verification already exists; preserve deterministic replay evidence")
    frozen = frozen_selection()
    data_manifest = json.loads((OUT/"data/manifest.json").read_text())
    assert data_manifest["contract_sha256"] == sha(CONFIG)
    assert json.loads((OUT/"reproducibility.json").read_text())["status"] == "PASS"
    for name, expected in data_manifest["output_file_sha256"].items():
        assert sha(OUT/f"data/{name}.parquet") == expected
    scan_audit = pd.read_parquet(OUT/"data/source_scan_audit.parquet")
    assert len(scan_audit) == 7231
    matched = pd.read_parquet(OUT/"data/matched_non_test.parquet")
    assert not matched.duplicated(["station_id","time"]).any()
    assert matched.groupby("station_id").time.apply(lambda x:x.is_monotonic_increasing).all()
    assert not matched.split.eq("test").any()
    assert matched.loc[matched.match_reason.eq("VALID"),["value","aod"]].notna().all().all()

    adapter.configure()
    replay = []
    for experiment in METHODS:
        receipt = next(row for row in json.loads((OUT/"validation_run_manifest.json").read_text())["completed"]
                       if row["experiment"] == experiment and row["month"] == 1)
        saved = pd.read_parquet(OUT/receipt["output"]/"predictions.parquet")
        stamp = saved.time.min()
        # Replay the original source-row order used by month_job. The sorted
        # matching export is not the same RF bootstrap input ordering.
        table = pd.read_parquet(OUT/"data/ground_non_test.parquet")
        table = table.loc[table.month.eq(1) & table.reason.eq("VALID")].copy()
        if experiment == "experiment3":
            table = table.merge(pd.read_parquet(OUT/"data/aod_hourly_all_sites.parquet"),
                on=["station_id","time"],how="left",validate="one_to_one")
            table = table.loc[table.aod.notna()]
        table = table.loc[table.time.eq(stamp)]
        columns = ["station_id","longitude","latitude","value","time","split","unit","easting","northing"]
        if experiment == "experiment3":
            columns.append("aod")
        train, query, status = adapter.qualified(table[columns],"validation")
        assert status == "ELIGIBLE"
        for method in METHODS[experiment]:
            actual = adapter.task.execute(method,train,query.drop(columns=["value","split"]))["outputs"]["prediction"].value
            expected = saved.loc[saved.method.eq(method) & saved.time.eq(stamp)].set_index("station_id").loc[query.station_id].predicted
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
            replay.append(dict(experiment=experiment,method=method,time=str(stamp),n=len(query),status="PASS",test_targets_used=False))
    save(OUT/"validation_algorithm_reproducibility.json",dict(actual_fresh_replay_graphs=6,checks=replay,
        full_year_baselines_not_repeated=True, all_alternative_sensitivity_runs_not_repeated=True))

    nc_checks = []
    for role in ["validation","test"]:
        receipt = json.loads((OUT/f"{role}_evaluation_receipt.json").read_text())
        folder = OUT/receipt["output"]
        assert sha(folder/"comparison.csv") == receipt["comparison_sha256"]
        proof = json.loads((folder/"verification.json").read_text())
        for item in proof["readbacks"]:
            file = OUT/item["file"]
            assert sha(file) == item["sha256"]
            with h5py.File(file) as ds:
                values = ds["pm25"][...]
                valid = ds["prediction_valid"][...]
                assert np.array_equal(np.isfinite(values),valid.astype(bool))
                assert (values[np.isfinite(values)] >= 0).all()
                assert ds.attrs["author_original_definitions_verified"] in ["false",b"false"]
            nc_checks.append(dict(file=item["file"],status="PASS_INDEPENDENT_H5PY_READ",rows=len(values)))

    pool = ROOT/"output/capability_final_acceptance/frozen_generic_pool.json"
    assert sha(pool) == "3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28"
    historic = json.loads((ROOT/"output/taiwan_final_20261019/delivery_verification.json").read_text(encoding="utf-8"))
    for item in historic["readback_files"]:
        assert sha(ROOT/item["file"]) == item["sha256"]
    uci = json.loads((ROOT/"output/taiwan_2017_preparation_20261019/uci_reproducibility_check.json").read_text())
    for file, expected in uci["hashes"].items():
        assert sha(ROOT/"output/three_experiments_20261019/uci"/file) == expected
    historical_sensitive = 0
    for name in ["taiwan_contract_sensitivity_20261019","taiwan_contract_sensitivity_paired_20261019"]:
        old_root = ROOT/"output"/name
        old = json.loads((old_root/"verification.json").read_text())
        for item in old["readbacks"]:
            file = old_root/item["file"]
            assert file.exists() and file.stat().st_mtime < (OUT/"contract_frozen.json").stat().st_mtime
            historical_sensitive += 1
    assert historical_sensitive == 576

    # Post-selection QC counts do not feed back into selection or parameters.
    test_ground = ground(include_test=True)
    all_ground = pd.concat([pd.read_parquet(OUT/"data/ground_non_test.parquet"),test_ground],ignore_index=True)
    covariates = pd.read_parquet(OUT/"data/aod_hourly_all_sites.parquet")
    counts = all_ground.merge(covariates,on=["station_id","time"],how="left",validate="one_to_one")
    counts["valid"] = counts.reason.eq("VALID")
    counts["matched"] = counts.valid & counts.aod.notna()
    by_station = counts.groupby(["station_id","split"],as_index=False).agg(records=("source_row","size"),
        valid_ground=("valid","sum"),matched_samples=("matched","sum"))
    by_station.merge(sites(),on=["station_id","split"],validate="one_to_one").to_csv(OUT/"all_station_roles_and_counts.csv",index=False)
    counts.groupby("split").agg(records=("source_row","size"),valid_ground=("valid","sum"),matched_samples=("matched","sum")).to_csv(OUT/"all_role_counts.csv")
    summary = dict(verified_utc=datetime.now(timezone.utc).isoformat(),status="PASS_UNDER_FROZEN_WORKING_CONVENTIONS",
        contract_version=CONTRACT["version"],contract_sha256=sha(CONFIG),historical_author_definitions_verified=False,
        full_year_matching_recomputed_twice=True,independent_netcdf_readbacks=nc_checks,actual_extra_validation_replays=6,
        test_selected_configs_only=True,selection_hash_verified=True,old_32_sha_unchanged=True,
        old_576_sensitivity_files_preserved=historical_sensitive,UCI_hashes_unchanged=True,frozen_34_pool_unchanged=True,
        new_deletions=0,new_downloads=0,full_regression_rerun=False,
        chosen={experiment:frozen[experiment]["method"] for experiment in METHODS},GOLEM_advantage="NOT_PROVEN_SMALL_SPACE")
    save(target,summary)
    print(json.dumps({key:value for key,value in summary.items() if key != "independent_netcdf_readbacks"}),flush=True)


if __name__ == "__main__":
    main()
