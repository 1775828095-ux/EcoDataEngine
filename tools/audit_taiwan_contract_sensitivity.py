"""Recompute paired sensitivity metrics and independently read numerical outputs."""
import itertools
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from run_taiwan_contract_sensitivity import OUT, METHODS, OLD, task


def metrics(rows):
    return task.validation_metrics(rows.predicted, rows.rename(columns={"observed": "value"}).assign(split="validation"))


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", action="store_true")
    if parser.parse_args().paired:
        OUT = OUT.parent / "taiwan_contract_sensitivity_paired_20261019"
    p = pd.read_csv(OUT / "predictions.csv")
    corrected = OUT / "executions_window_labels_corrected.csv"
    r = pd.read_csv(corrected if corrected.exists() else OUT / "executions.csv")
    summary = pd.read_csv(OUT / "comparison.csv")
    summary["coverage_of_valid_ground"] = summary.actual_predictions / summary.ground_valid
    summary["coverage_of_qualified"] = summary.actual_predictions / summary.qualified_validation
    summary["aod_coverage_of_valid_ground"] = summary.aod_matched_validation / summary.ground_valid
    attempt_counts = r.dropna(subset=["method"]).groupby(["hypothesis", "method"]).size()
    summary["attempted_windows"] = [int(attempt_counts.get((row.hypothesis, row.method), 0)) for row in summary.itertuples()]
    summary["execution_failure_rate"] = summary.failures / summary.attempted_windows.replace(0, np.nan)
    summary.to_csv(OUT / "comparison_with_coverage.csv", index=False)
    split = pd.read_csv(OUT / "station_split.csv")
    pd.testing.assert_frame_equal(split, pd.read_csv(OLD / "station_split.csv"))
    assert set(p.station_id).issubset(set(split.loc[split.split.eq("validation"), "station_id"]))
    assert not p.duplicated(["hypothesis", "method", "station_id", "raw_DT"]).any()
    assert np.isfinite(p[["observed", "predicted", "before_projection"]]).all().all()
    assert (p.predicted >= 0).all()
    keys = {}
    for hyp, g in p.groupby("hypothesis"):
        keys[hyp] = set.intersection(*[set(zip(v.raw_DT, v.station_id)) for method in METHODS for v in [g.loc[g.method.eq(method)]]])
        for method in METHODS:
            v = g.loc[g.method.eq(method)]
            v = v.loc[[(t, s) in keys[hyp] for t, s in zip(v.raw_DT, v.station_id)]]
            result = metrics(v)
            recorded = summary.loc[summary.hypothesis.eq(hyp) & summary.method.eq(method)].iloc[0]
            for name, value in result.items():
                np.testing.assert_allclose(value, recorded[name])
    paired, pair_counts = [], []
    for a, b in itertools.combinations(sorted(keys), 2):
        common = keys[a] & keys[b]
        pair_counts.append(dict(hypothesis_a=a, hypothesis_b=b, common_same_ground_labels=len(common)))
        if len(common) < 2:
            continue
        truth = []
        for hyp in [a, b]:
            v = p.loc[p.hypothesis.eq(hyp) & p.method.eq(METHODS[0])]
            v = v.loc[[(t, s) in common for t, s in zip(v.raw_DT, v.station_id)]]
            truth.append(v.set_index(["raw_DT", "station_id"]).observed.sort_index())
        pd.testing.assert_series_equal(*truth)
        for hyp in [a, b]:
            for method in METHODS:
                v = p.loc[p.hypothesis.eq(hyp) & p.method.eq(method)]
                v = v.loc[[(t, s) in common for t, s in zip(v.raw_DT, v.station_id)]]
                paired.append(dict(pair_a=a, pair_b=b, hypothesis=hyp, method=method,
                    common_samples=len(v), **metrics(v)))
    pair = pd.DataFrame(paired)
    pair["rank"] = pair.groupby(["pair_a", "pair_b", "hypothesis"]).RMSE.rank(method="min")
    pair.to_csv(OUT / "paired_comparison.csv", index=False)
    pd.DataFrame(pair_counts).to_csv(OUT / "pairwise_intersection_counts.csv", index=False)
    checks = []
    for (a, b), rows in pair.groupby(["pair_a", "pair_b"]):
        aa, bb = rows.loc[rows.hypothesis.eq(a)].set_index("method"), rows.loc[rows.hypothesis.eq(b)].set_index("method")
        changed = not aa["rank"].sort_index().equals(bb["rank"].sort_index())
        checks.append(dict(hypothesis_a=a, hypothesis_b=b, n=int(rows.common_samples.iloc[0]),
            rank_changed=changed, a_best=aa.RMSE.idxmin(), b_best=bb.RMSE.idxmin(),
            maximum_rmse_change=float((aa.RMSE-bb.RMSE).abs().max())))
    pd.DataFrame(checks).to_csv(OUT / "paired_rank_changes.csv", index=False)
    readbacks = []
    for file in sorted(OUT.rglob("predictions.nc")):
        relative = file.relative_to(OUT)
        hypothesis, _, time, method, _ = relative.parts
        stamp = str(pd.to_datetime(time, format="%Y-%m-%d_%H%M%S"))
        values = p.loc[p.hypothesis.eq(hypothesis) & p.method.eq(method) & p.time.eq(stamp)]
        assert len(values) > 0
        with xr.open_dataset(file) as ds:
            np.testing.assert_allclose(ds.pm25.values, values.predicted.to_numpy())
            assert ds.pm25.attrs["units"] == task.UNIT
            assert ds.pm25.dims == ("sample",)
            locations = split.set_index("station_id").loc[values.station_id]
            np.testing.assert_allclose(ds.longitude.values, locations.longitude.to_numpy())
            np.testing.assert_allclose(ds.latitude.values, locations.latitude.to_numpy())
        quality = json.loads((file.parent / "quality.json").read_text(encoding="utf-8"))
        assert quality["result"]["status"] == "WARN"
        readbacks.append(dict(file=str(relative), n=len(values), finite=True, nonnegative=True,
            source_time_datum_unverified=True, grid_coverage="NOT_APPLICABLE_POINT_OUTPUT"))
    global_common = set.intersection(*keys.values())
    result = dict(hypotheses=len(keys), total_predictions=len(p), netcdf_readbacks=len(readbacks),
        real_algorithm_executions=int(r.loc[r.reused_from.isna(), "method"].notna().sum()),
        cached_algorithm_executions=int(r.loc[r.reused_from.notna(), "method"].notna().sum()),
        all_hypotheses_common_ground_labels=len(global_common), paired_comparisons=len(checks),
        paired_rank_changed=sum(row["rank_changed"] for row in checks),
        pairs_without_common_labels=sum(row["common_same_ground_labels"] < 2 for row in pair_counts),
        final_test_targets_used=False, formal_contract_frozen=False,
        projection_count=int(summary.projected.sum()),
        failed_algorithm_runs=int(r.status.eq("FAIL").sum()), insufficient_windows=int(r.status.str.startswith("INSUFFICIENT").sum()),
        interpretation="Ranking changes on different cohorts are not paired proof; only pairwise common-label rows are paired diagnostics, not provenance evidence",
        readbacks=readbacks)
    pool = task.ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"
    assert hashlib.sha256(pool.read_bytes()).hexdigest() == "3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28"
    historical = json.loads((task.ROOT / "output/taiwan_final_20261019/delivery_verification.json").read_text(encoding="utf-8"))
    for old in historical["readback_files"]:
        assert hashlib.sha256((task.ROOT / old["file"]).read_bytes()).hexdigest() == old["sha256"]
    result["old_32_outputs_and_frozen_pool_unchanged"] = True
    result["new_deletions"] = 0
    task.save(OUT / "verification.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "readbacks"}), flush=True)


if __name__ == "__main__":
    main()
