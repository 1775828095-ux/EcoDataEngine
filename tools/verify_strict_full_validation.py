"""Post-selection annual validation coverage audit; never reselect by its scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import run_taiwan_composition_search as run


def month_job(out, key, graph, month):
    run.OUT=Path(out)
    folder=run.OUT/"postfreeze_full_validation"/f"month_{month:02d}"
    result_path=folder/key/"result.json"
    if not result_path.exists():
        cohort=[c for c in run.cases() if c[0].tz_convert("Asia/Taipei").month==month]
        run.Evaluator(cohort,folder).execute(run.restore(graph),key)
    return month,key


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=run.ROOT/"output/taiwan_frozen_pool_search_final")
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args(); out=args.output.resolve()
    selected_path=out/"strict_validation_selection_frozen.json"
    selection=json.loads(selected_path.read_text(encoding="utf-8"))
    graphs={r["key"]:r["graph"] for r in selection["selection"]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(month_job,str(out),key,graph,month)
                 for key,graph in graphs.items() for month in range(1,13)]
        for future in as_completed(futures):
            month,key=future.result()
            print("VALIDATION",month,key[:8],flush=True)
    rows=[]
    for key in graphs:
        folders=[out/"postfreeze_full_validation"/f"month_{m:02d}"/key for m in range(1,13)]
        frames=[]; results=[]
        for folder in folders:
            frame=pd.read_parquet(folder/"predictions.parquet")
            with h5py.File(folder/"predictions.nc") as file:
                np.testing.assert_allclose(file["pm25"][:],frame.prediction,equal_nan=True)
                np.testing.assert_array_equal(file["residual_usage"][:],frame.residual_usage)
            frames.append(frame)
            results.append(json.loads((folder/"result.json").read_text(encoding="utf-8")))
        values=pd.concat(frames,ignore_index=True)
        assert not values.duplicated(["station_id","time"]).any()
        valid=np.isfinite(values.prediction)
        row=dict(key=key,expected=len(values),coverage=float(valid.mean()),
            hours=values.time.nunique(),stations=values.station_id.nunique(),
            days=values.time.dt.tz_convert("Asia/Taipei").dt.date.nunique(),
            status="WARN_WORKING_CONTRACT" if valid.all() else "BLOCK_INCOMPLETE_COVERAGE",
            failure_hours=sum(r["failure_hours"] for r in results),
            clips=sum(r["clips"] for r in results),seconds=sum(r["seconds"] for r in results),
            selection_sha256=run.sha(selected_path),used_for_reselection=False,test_labels_read=False)
        if valid.all(): row.update(run.metrics(values.prediction,values.observed))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"postfreeze_full_validation_metrics.csv",index=False)
    run.save(out/"postfreeze_full_validation_receipt.json",dict(results=rows,
        note="Frozen selection audited on the remaining validation dates, no graph or strategy changed; no test labels read",
        script_sha256=run.sha(Path(__file__))))
