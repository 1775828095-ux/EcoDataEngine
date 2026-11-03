"""Monthly real DAG execution; final tests require a pre-existing frozen selection."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import pandas as pd

import taiwan_annual_adapter as adapter
from taiwan_working_contract import CONFIG, CONTRACT, OUT, frozen_selection, ground, save, sha, sites

METHODS = {"experiment2":["idw", "ok"], "experiment3":["linear", "gwr", "uk", "rf"]}


def month_job(job):
    experiment, month, role, chosen = job
    adapter.configure()
    task = adapter.task
    root = OUT / "runs" / role / experiment / f"month_{month:02d}"
    if (root / "completion.json").exists():
        previous = json.loads((root / "completion.json").read_text())
        assert previous["contract_sha256"] == sha(CONFIG)
        return previous
    folder = root
    retry = 0
    while folder.exists():
        retry += 1
        folder = root.parent / f"month_{month:02d}_attempt_{retry}"
        if (folder / "completion.json").exists():
            return json.loads((folder / "completion.json").read_text())
    folder.mkdir(parents=True)
    frame = pd.read_parquet(OUT / "data/ground_non_test.parquet")
    if role == "test":
        frame = pd.concat([frame.loc[frame.split.eq("train")], ground(include_test=True)], ignore_index=True)
    frame = frame.loc[frame.month.eq(month) & frame.reason.eq("VALID")].copy()
    methods = chosen if role == "test" else METHODS[experiment]
    if experiment == "experiment3":
        covariates = pd.read_parquet(OUT / "data/aod_hourly_all_sites.parquet")
        frame = frame.merge(covariates, on=["station_id", "time"], how="left", validate="one_to_one")
    station = sites()
    potential = int(station.split.eq(role).sum())
    predictions, logs, coverage = [], [], []
    started = perf_counter()
    groups = dict(tuple(frame.groupby("time", sort=True)))
    first = pd.Timestamp(year=2017, month=month, day=1, tz="Asia/Taipei")
    requested = pd.date_range(first, first+pd.offsets.MonthBegin(1), freq="h", inclusive="left").tz_convert("UTC")
    with (folder / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for counter, stamp in enumerate(requested):
            raw = groups.get(stamp, frame.iloc[:0])
            raw_targets = raw.loc[raw.split.eq(role)]
            table = raw.loc[raw.aod.notna()].copy() if experiment == "experiment3" else raw.copy()
            columns = ["station_id", "longitude", "latitude", "value", "time", "split", "unit", "easting", "northing"]
            if experiment == "experiment3":
                columns.append("aod")
            table = table[columns]
            train, target, status = adapter.qualified(table, role)
            coverage.append(dict(time=stamp, month=month, possible=potential, ground_valid=len(raw_targets),
                covariate_matched=int(table.split.eq(role).sum()), qualified=len(target), train=len(train), status=status))
            if status != "ELIGIBLE":
                continue
            query = target.drop(columns=["value", "split"])
            assert set(train.station_id).isdisjoint(query.station_id)
            for method in methods:
                begin = perf_counter()
                try:
                    result = task.execute(method, train, query)
                    values = result["outputs"]["prediction"].value
                    before = result["outputs"]["raw_prediction"].value
                    quality = result["outputs"]["quality"].value
                    assert np.isfinite(values).all() and (values >= 0).all() and len(values) == len(target)
                    assert quality["result"]["status"] != "BLOCK"
                    line = json.dumps(dict(time=str(stamp), method=method, trace=result["trace"], quality=quality), default=str)
                    traces.write(line+"\n")
                    logs.append(dict(time=stamp, method=method, expected=len(target), predicted=len(target),
                        status="EXECUTED_WORKING_CONTRACT", seconds=perf_counter()-begin, projection_count=int((before < 0).sum()),
                        quality=quality["result"]["status"], trace_sha256=hashlib.sha256(line.encode()).hexdigest(),
                        graph_executed=True, reason=None))
                except Exception as error:
                    values, before = np.full(len(target), np.nan), np.full(len(target), np.nan)
                    logs.append(dict(time=stamp, method=method, expected=len(target), predicted=0, status="FAIL",
                        seconds=perf_counter()-begin, projection_count=0, quality="BLOCK", graph_executed=True, reason=repr(error)))
                    traces.write(json.dumps(dict(time=str(stamp), method=method, error=repr(error), trace=getattr(error,"trace",[])),default=str)+"\n")
                for (_, row), value, raw_value in zip(target.iterrows(), values, before):
                    predictions.append(dict(time=stamp, station_id=row.station_id, month=month, method=method,
                        observed=row.value, predicted=value, before_projection=raw_value, split=role,
                        longitude=row.longitude, latitude=row.latitude))
            if counter and counter % 200 == 0:
                print(role, experiment, month, "hours examined", counter, flush=True)
    for name, rows in [("predictions",predictions),("executions",logs),("coverage",coverage)]:
        pd.DataFrame(rows).to_parquet(folder / f"{name}.parquet", index=False)
    receipt = dict(experiment=experiment, month=month, split=role, output=str(folder.relative_to(OUT)),
        contract_sha256=sha(CONFIG), predictions=len(predictions), algorithm_calls=len(logs),
        failures=sum(row["status"] == "FAIL" for row in logs), elapsed_seconds=perf_counter()-started,
        source_implementation_sha256=sha(task.ROOT / "tools/run_taiwan_preparation.py"))
    save(folder / "completion.json", receipt)
    print("COMPLETE", role, experiment, month, "calls", len(logs), "failures", receipt["failures"], flush=True)
    return receipt


def main():
    overall_start = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    assert json.loads((OUT / "reproducibility.json").read_text())["status"] == "PASS"
    previous_manifest = OUT / f"{args.split}_run_manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text())
        if len(previous["completed"]) == 24 and not previous["failures"]:
            print("Already completed; original run manifest retained", flush=True)
            return
    chosen = {}
    if args.split == "test":
        selection = frozen_selection()
        assert selection["contract_sha256"] == sha(CONFIG)
        chosen = {experiment:[selection[experiment]["method"]] for experiment in METHODS}
    else:
        plan = OUT / "baseline_plan.json"
        if not plan.exists():
            save(plan, dict(contract_sha256=sha(CONFIG), methods=METHODS, months=list(range(1,13)),
                task="same-hour training-station interpolation, not forecasting", parameters="unchanged previous method parameters; common EPSG32651 adapter",
                selection="quality-compatible common-cohort validation RMSE, elapsed cost as tie break only",
                search_space="four non-equivalent multisource configurations; exhaustive baseline, no fabricated GOLEM search advantage",
                test_unseal="separate command requires immutable selection_frozen.json", new_algorithms=False))
    jobs = [(experiment,month,args.split,chosen.get(experiment)) for experiment in METHODS for month in range(1,13)]
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(month_job, job):job for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(job=futures[future], error=repr(error)))
                print("MONTH_FAILURE", futures[future], repr(error), flush=True)
    save(OUT / f"{args.split}_run_manifest.json", dict(completed=results, failures=failures, contract_sha256=sha(CONFIG), wall_seconds=perf_counter()-overall_start))
    if failures:
        raise RuntimeError("Monthly failures retained; fix and resume only affected jobs")


if __name__ == "__main__":
    main()
