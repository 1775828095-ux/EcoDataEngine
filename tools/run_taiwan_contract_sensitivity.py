"""Bounded real-data sensitivity audit, never a provenance inference or test evaluation."""
import hashlib
import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pyproj import Transformer

import run_taiwan_preparation as task

ROOT = task.ROOT
OUT = ROOT / "output/taiwan_contract_sensitivity_20261019"
OLD = ROOT / "output/taiwan_2017_preparation_20261019/golem_dag_smoke"
METHODS = ["idw", "ok", "linear", "gwr", "uk", "rf"]
DATES = ["20170115", "20170415", "20170715", "20171015"]
HOURS = [1, 5]


def hour_start(labels, offset, convention):
    if offset not in [0, 8] or convention not in ["start", "end"]:
        raise ValueError("Unsupported explicit time hypothesis")
    return pd.to_datetime(labels) - pd.Timedelta(hours=offset + (convention == "end"))


def case_key(stamp, eligible):
    # Empty rejected inputs still belong to different requested windows.
    return hashlib.sha256((str(stamp) + "\n" + eligible.sort_values("station_id").to_csv(index=False)).encode()).hexdigest()


def main():
    global OUT, HOURS
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", action="store_true", help="Fixed adjacent/8-hour-boundary design, not metric-selected")
    args = parser.parse_args()
    if args.paired:
        OUT = ROOT / "output/taiwan_contract_sensitivity_paired_20261019"
        HOURS = [0, 1, 8, 9]
    OUT.mkdir(exist_ok=False)
    archive = json.loads((task.DATA / "download_verification.json").read_text())
    assert archive["verified"] is True
    split = pd.read_csv(OLD / "station_split.csv")
    split.to_csv(OUT / "station_split.csv", index=False)
    sealed = set(split.loc[split.split.eq("test"), "station_id"])
    sites = split.loc[~split.split.eq("test")].copy()
    assert len(sealed) == 14 and len(sites) == 63
    base = task.DATA / "provisional_members/zhang_data"
    # Select identifiers before loading target columns: sealed station values are not parsed.
    raw = pd.read_csv(base / "pm25_taiwan/Taiwan_PM_2017.csv", dtype=str)
    raw = raw.loc[~raw.stn_eng.isin(sealed), ["stn_eng", "DT", "PM2.5"]]
    ground = raw.rename(columns={"stn_eng": "station_id", "DT": "raw_DT", "PM2.5": "value"})
    ground["raw_DT"] = pd.to_datetime(ground.raw_DT)
    ground["value"] = pd.to_numeric(ground.value, errors="coerce")
    assert not ground.duplicated(["station_id", "raw_DT"]).any()
    plan = dict(exploratory=True, dates=DATES, satellite_filename_hours=HOURS, methods=METHODS,
        time_hypotheses=8, coordinate_hypotheses=["EPSG:4326", "EPSG:3824"],
        fixed_split_sha256=hashlib.sha256((OLD / "station_split.csv").read_bytes()).hexdigest(),
        interval="Ground start: [DT,DT+1h); end: [DT-1h,DT); AHI nominal scan instant assigned to left-closed hour",
        ground_offsets=[0, 8], satellite_offsets=[0, 8], hour_conventions=["start", "end"],
        algorithm_parameters="Unchanged run_taiwan_preparation.py; no tuning; same-hour training stations only",
        qualification="valid nonnegative ground, QA AOD, >=15 train, >=3 validation in training hull; same cohort all six methods",
        cross_hypothesis_key="station_id + original ground raw_DT; never compare different truth labels as paired samples",
        test_station_targets_used=False, formal_freeze=False,
        satellite_navigation="Conditional WGS84 interpretation supported by JMA HSD guide; author output lineage unresolved",
        omitted_hypotheses="No evidence for GCJ02/TWD67; no fabricated scan duration or arbitrary clock offsets",
        stopping_rule="Four fixed seasonal days; requested filename hours frozen before this run, no performance-selected dates",
        design_amendment="Paired supplement: adjacent hours and 8-hour-offset boundary address empty paired intersections in initial 01/05-hour design; not selected using RMSE" if args.paired else None)
    task.save(OUT / "analysis_plan.json", plan)
    predictions, runs, denominators, matches, projections = [], [], [], [], []
    cache = {}
    for datum in [4326, 3824]:
        current = sites.copy()
        transform = Transformer.from_crs(datum, 4326, always_xy=True)
        current["longitude"], current["latitude"] = transform.transform(sites.longitude.to_numpy(), sites.latitude.to_numpy())
        projections.append(dict(source_crs=f"EPSG:{datum}", target_crs="EPSG:4326", operation=transform.description,
            accuracy_metres=transform.accuracy, definition=transform.definition,
            max_coordinate_degree_change=float(np.max(abs(current[["longitude", "latitude"]].to_numpy()-sites[["longitude", "latitude"]].to_numpy())))))
        task.OUT = OUT / f"spatial_{datum}"
        task.OUT.mkdir()
        files = [base / "ahi_aod_taiwan" / day / f"results_Taiwan_{day}_{hour:02d}{minute:02d}.nc"
                 for day in DATES for hour in HOURS for minute in [0, 30]]
        missing = [str(path) for path in files if not path.exists()]
        task.save(task.OUT / "missing_source_files.json", missing)
        aod = task.satellite_samples([file for file in files if file.exists()], current).rename(columns={"utc_hour": "filename_hour"})
        for goff, soff, convention in itertools.product([0, 8], [0, 8], ["start", "end"]):
            hypothesis = f"g{goff}_s{soff}_{convention}_epsg{datum}"
            folder = OUT / hypothesis
            folder.mkdir()
            task.OUT = folder
            g = ground.merge(current, on="station_id", validate="many_to_one")
            g["time"] = hour_start(g.raw_DT, goff, convention)
            g["unit"] = task.UNIT
            sat = aod.copy()
            sat["time"] = sat.filename_hour - pd.Timedelta(hours=soff)
            anchors = [pd.Timestamp(day)+pd.Timedelta(hours=h-soff) for day in DATES for h in HOURS]
            candidate = g.loc[g.time.isin(anchors)].copy()
            paired = candidate.merge(sat, on=["station_id", "time"], validate="one_to_one")
            paired = paired.loc[paired.value.ge(0) & np.isfinite(paired.value)].copy()
            assert not set(paired.station_id) & sealed
            paired.to_csv(folder / "matched_non_test.csv", index=False)
            for stamp in anchors:
                eligible = paired.loc[paired.time.eq(stamp)].copy()
                original = candidate.loc[candidate.time.eq(stamp) & candidate.split.eq("validation")]
                targets = eligible.loc[eligible.split.eq("validation")]
                denom = dict(hypothesis=hypothesis, time=str(stamp), possible_validation=13,
                    ground_valid=int((original.value.ge(0) & np.isfinite(original.value)).sum()),
                    aod_matched_validation=len(targets), train=len(eligible.loc[eligible.split.eq("train")]))
                # Identical actual full inputs may reuse a previously verified numerical execution.
                key = case_key(stamp, eligible)
                if key in cache:
                    source_hyp, p, r = cache[key]
                    p, r = [dict(row) for row in p], [dict(row) for row in r]
                    for row in r:
                        row.update(reused_from=source_hyp, audit_seconds=0.)
                else:
                    start = perf_counter()
                    p, r = task.evaluate_case("sensitivity", stamp, eligible, METHODS)
                    for row in r:
                        row.update(reused_from=None, audit_seconds=perf_counter()-start)
                    cache[key] = (hypothesis, p, r)
                label_map = targets.set_index("station_id").raw_DT.to_dict()
                for row in p:
                    row = dict(row, hypothesis=hypothesis, raw_DT=str(label_map[row["station_id"]]))
                    predictions.append(row)
                for row in r:
                    runs.append(dict(row, hypothesis=hypothesis))
                denom["in_hull_qualified"] = max([row.get("expected", 0) for row in r] or [0])
                denominators.append(denom)
            matches.append(dict(hypothesis=hypothesis, matched_non_test=len(paired), validation=int(paired.split.eq("validation").sum())))
            print(hypothesis, "matched", len(paired), flush=True)
    p, r, d = pd.DataFrame(predictions), pd.DataFrame(runs), pd.DataFrame(denominators)
    p.to_csv(OUT / "predictions.csv", index=False)
    r.to_csv(OUT / "executions.csv", index=False)
    d.to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(matches).to_csv(OUT / "matching_counts.csv", index=False)
    task.save(OUT / "coordinate_operations.json", projections)
    summaries = []
    for hyp in [row["hypothesis"] for row in matches]:
        part, executions = p.loc[p.hypothesis.eq(hyp)], r.loc[r.hypothesis.eq(hyp)]
        keys = [set(zip(part.loc[part.method.eq(method), "raw_DT"], part.loc[part.method.eq(method), "station_id"])) for method in METHODS]
        common = set.intersection(*keys)
        for method in METHODS:
            rows = part.loc[part.method.eq(method)].copy()
            rows = rows.loc[[(t, s) in common for t, s in zip(rows.raw_DT, rows.station_id)]]
            run = executions.loc[executions.get("method", pd.Series(index=executions.index, dtype=str)).eq(method)]
            denom = d.loc[d.hypothesis.eq(hyp)]
            result = dict(hypothesis=hyp, method=method, common_samples=len(rows),
                possible_validation=int(denom.possible_validation.sum()), ground_valid=int(denom.ground_valid.sum()),
                aod_matched_validation=int(denom.aod_matched_validation.sum()),
                qualified_validation=int(denom.in_hull_qualified.sum()),
                actual_predictions=int(len(part.loc[part.method.eq(method)])),
                failures=int(run.status.eq("FAIL").sum()), insufficient_windows=int(executions.status.str.startswith("INSUFFICIENT").sum()),
                projected=int(run.projection_count.fillna(0).sum()) if "projection_count" in run else 0,
                algorithm_seconds=float(run.seconds.fillna(0).sum()) if "seconds" in run else 0.,
                real_executions=int(run.reused_from.isna().sum()), cached_executions=int(run.reused_from.notna().sum()))
            if len(rows) >= 2:
                validation = rows.rename(columns={"observed": "value"}).assign(split="validation")
                result.update(task.validation_metrics(rows.predicted, validation))
            summaries.append(result)
    summary = pd.DataFrame(summaries)
    summary["rank"] = summary.groupby("hypothesis").RMSE.rank(method="min")
    summary.to_csv(OUT / "comparison.csv", index=False)
    task.save(OUT / "completion.json", dict(status="EXPLORATORY_EXECUTED", formal_freeze=False,
        predictions=len(p), hypotheses=len(matches), readbacks="Every newly executed case uses existing evaluate_case NetCDF readback",
        final_test_targets_used=False, frozen_pool_modified=False))
    print(summary[["hypothesis", "method", "common_samples", "RMSE", "rank"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
