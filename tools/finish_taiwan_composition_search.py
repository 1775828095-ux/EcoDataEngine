"""Freeze searched graphs before reading reused held-out labels; audit artifacts."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

import h5py
import numpy as np
import pandas as pd

from run_taiwan_composition_search import (OUT, ANNUAL, ROOT, POOL, CONFIG, save, sha,
    setup, restore, fingerprint, Evaluator, cases, metrics)


def freeze():
    path=OUT/"search_selection_frozen.json"
    if path.exists(): return json.loads(path.read_text())
    results=json.loads((OUT/"search_summary.json").read_text())
    assert len(results)==15 and all(r["best"] and not r["optimizer_error"] for r in results)
    selected=[dict(mode=r["mode"],seed=r["seed"],key=r["best"]["key"],graph=r["best"]["graph"],
        validation_subset_RMSE=r["best"]["RMSE"]) for r in results]
    data=dict(frozen_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/"protocol.json"),
        contract_sha256=sha(CONFIG),pool_sha256=sha(POOL),selection=selected,
        selection_rule="minimum hard-quality-compatible RMSE within each search run; no final test used",
        caveat="These 14 stations were already evaluated in chapter29. This is reused held-out evaluation, not a fresh blind confirmatory test.")
    save(path,data)
    save(OUT/"search_selection_freeze_receipt.json",dict(sha256=sha(path),test_read_after_freeze_only=True))
    return data


def full_month(key,rows,role,month):
    folder=OUT/"final_frozen"/role/f"month_{month:02d}"
    path=folder/key/"result.json"
    if path.exists(): return str(path)
    targets=cases(role)
    targets=[c for c in targets if c[0].tz_convert("Asia/Taipei").month==month]
    evaluator=Evaluator(targets,folder,role)
    result=evaluator.execute(restore(rows),key)
    print("FINAL",role,month,key[:8],result["status"],result.get("RMSE"),flush=True)
    return str(path)


def aggregate():
    selected=freeze()
    summary=[]
    for role in ["validation","test"]:
        for key in sorted({s["key"] for s in selected["selection"]}):
            folders=[OUT/"final_frozen"/role/f"month_{m:02d}"/key for m in range(1,13)]
            values=pd.concat([pd.read_parquet(p/"predictions.parquet") for p in folders],ignore_index=True)
            run=[json.loads((p/"result.json").read_text()) for p in folders]
            assert not values.duplicated(["station_id","time"]).any()
            valid=np.isfinite(values.prediction)
            row=dict(split=role,key=key,coverage=float(valid.mean()),expected=len(values),
                failure_hours=sum(r["failure_hours"] for r in run),clips=sum(r["clips"] for r in run),
                seconds=sum(r["seconds"] for r in run),stations=values.station_id.nunique(),
                days=pd.to_datetime(values.time,utc=True).dt.tz_convert("Asia/Taipei").dt.date.nunique())
            if valid.all(): row.update(metrics(values.prediction,values.observed))
            summary.append(row)
            values.to_parquet(OUT/"final_frozen"/f"{role}_{key}.parquet",index=False)
            for folder in folders:
                with h5py.File(folder/"predictions.nc") as file:
                    assert len(file["pm25"])==json.loads((folder/"result.json").read_text())["expected"]
                    np.testing.assert_allclose(file["pm25"][:],pd.read_parquet(folder/"predictions.parquet").prediction,equal_nan=True)
    pd.DataFrame(summary).to_csv(OUT/"full_validation_and_reused_test.csv",index=False)
    return summary


def analysis():
    results=json.loads((OUT/"search_summary.json").read_text())
    rows,curves=[],[]
    for result in results:
        detail=json.loads((OUT/"search"/f"{result['mode']}_{result['seed']}"/"summary.json").read_text())
        best=float("inf")
        for evaluation_index,r in enumerate(detail["objective_records"],1):
            if r.get("RMSE") is not None and r["score"]<1e6:
                best=min(best,r["RMSE"])
            curves.append(dict(mode=result["mode"],seed=result["seed"],evaluation=evaluation_index,
                source_evaluation_index=r["index"],
                best_RMSE=best if np.isfinite(best) else None,real_execution=r["real_execution"],
                cache_hit=r["cache_hit"],gap=r.get("gap"),status=r["status"],elapsed=r["elapsed"]))
        c=result["counts"]
        rows.append(dict(mode=result["mode"],seed=result["seed"],RMSE=result["best"]["RMSE"],
            MAE=result["best"]["MAE"],Bias=result["best"]["Bias"],key=result["best"]["key"],
            evaluations=len(detail["objective_records"]),real=c.get("real_graph_evaluations",0),
            cache=c.get("cache_hits",0),unique_legal=result["unique_legal_graphs"],
            proposals=c.get("proposals",0),rejections=c.get("proposal_rejections",0),
            posterior_rejections=c.get("posterior_rejections",0),gap_completions=result["gap_completions"],
            wall_seconds=result["wall_seconds"],numeric_failures=c.get("numeric_failures",0),
            first_feasible=min((r["index"] for r in detail["objective_records"] if r.get("score",1e6)<1000),default=None)))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/"seed_results.csv",index=False)
    curve_path=OUT/"budget_curves.csv"
    if curve_path.exists() and "source_evaluation_index" not in pd.read_csv(curve_path,nrows=0).columns:
        shutil.copy2(curve_path,OUT/"budget_curves_before_cache_accounting_fix.csv")
    pd.DataFrame(curves).to_csv(curve_path,index=False)
    frame.groupby("mode")[["RMSE","MAE","Bias","unique_legal","real","proposals","rejections","wall_seconds"]].agg(["mean","std"]).to_csv(OUT/"mode_means_std.csv")
    catalog=json.loads((OUT/"catalog.json").read_text())
    enum=[]
    signatures={}
    for key,graph in catalog.items():
        folder=OUT/"enumeration"/key
        result=json.loads((folder/"result.json").read_text())
        values=pd.read_parquet(folder/"predictions.parquet")
        behavior=__import__("hashlib").sha256(np.asarray(values.prediction).round(10).tobytes()).hexdigest()
        signatures.setdefault(behavior,[]).append(key)
        enum.append(dict(key=key,behavior=behavior,**{k:v for k,v in result.items() if k!="failures"}))
    pd.DataFrame(enum).to_csv(OUT/"enumeration_results.csv",index=False)
    save(OUT/"behavior_equivalence.json",dict(round_decimals=10,groups=signatures,
        note="Observed numeric equivalence on this fixed validation subset is not a proof of universal equivalence; identities/bindings already canonicalized before execution"))
    # A fixed no-search random ordering reference uses the very same real
    # enumeration executions. Replay/cache costs never counted as fresh fits.
    random_rows=[]
    keys=list(catalog)
    scores={r["key"]:r["score"] for r in enum}
    for seed in [11,22,33,44,55]:
        order=keys.copy(); random.Random(seed).shuffle(order)
        best=1e6
        for index,key in enumerate(order[:20],1):
            best=min(best,scores[key])
            random_rows.append(dict(seed=seed,evaluation=index,key=key,best_RMSE=best,new_execution=False,
                cache_source="same-fingerprint real enumeration; replay, not timed search"))
    pd.DataFrame(random_rows).to_csv(OUT/"random_order_cached_reference.csv",index=False)
    # Four historical fixed baselines on the identical new search cohort.
    protocol=json.loads((OUT/"protocol.json").read_text())
    old=pd.concat([pd.read_parquet(ANNUAL/"runs/validation/experiment3"/f"month_{m:02d}"/"predictions.parquet") for m in range(1,13)],ignore_index=True)
    old=old.loc[old.time.isin(pd.to_datetime(protocol["times"],utc=True))]
    baseline=[]
    for method,g in old.groupby("method"):
        baseline.append(dict(method=method,**metrics(g.predicted,g.observed),
            cache_source="chapter29 actual full-year execution filtered to the same fixed 96 hours; not a new fit"))
    pd.DataFrame(baseline).to_csv(OUT/"four_fixed_baseline_reference.csv",index=False)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    selection=freeze()
    analysis()
    unique={s["key"]:s["graph"] for s in selection["selection"]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(full_month,key,rows,role,m) for key,rows in unique.items()
               for role in ["validation","test"] for m in range(1,13)]
        for future in as_completed(tasks): future.result()
    summary=aggregate()
    save(OUT/"final_checks.json",dict(selection_sha256=sha(OUT/"search_selection_frozen.json"),
        receipt_matches=json.loads((OUT/"search_selection_freeze_receipt.json").read_text())["sha256"]==sha(OUT/"search_selection_frozen.json"),
        frozen_pool_unchanged=sha(POOL)==selection["pool_sha256"],contract_unchanged=sha(CONFIG)==selection["contract_sha256"],
        full_results=summary,deleted_files=0,downloads=0,third_party_source_modified=False))


if __name__=="__main__": main()
