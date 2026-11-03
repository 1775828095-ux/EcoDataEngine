"""One post-freeze evaluation on the existing, previously exposed holdout.

No claim of a fresh blind confirmation; no parameter/graph reselection occurs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import json
from pathlib import Path

import pandas as pd
import run_taiwan_composition_search as run


def freeze_plan(out):
    path=out/"holdout_plan.json"
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    if not path.exists():
        run.save(path,dict(frozen_at=datetime.now(timezone.utc).isoformat(),times=protocol["times"],
            protocol_sha256=run.sha(out/"protocol.json"),station_split_sha256=run.sha(run.ROOT/"output/taiwan_2017_preparation_20261019/golem_dag_smoke/station_split.csv"),
            rule="same prespecified 96 hours as validation; original 14 test stations; frozen QA and geometry; no post-score adjustment",
            status="PREVIOUSLY_EXPOSED_HOLDOUT_NOT_NEW_BLIND_TEST",
            evaluation="one evaluation per unique selected graph; report every arm/seed, never select by test error"))
    else:
        plan=json.loads(path.read_text(encoding="utf-8"))
        assert plan["times"]==protocol["times"] and plan["protocol_sha256"]==run.sha(out/"protocol.json")


def worker(key,rows,out):
    run.OUT=Path(out)
    target=run.OUT/"postfreeze_reused_holdout"/key/"result.json"
    if target.exists(): return json.loads(target.read_text(encoding="utf-8"))
    selection_path=run.OUT/"strict_validation_selection_frozen.json"
    assert selection_path.exists()
    plan=json.loads((run.OUT/"holdout_plan.json").read_text(encoding="utf-8"))
    cohort=run.cases(role="test",selected=plan["times"])
    evaluator=run.Evaluator(cohort,run.OUT/"postfreeze_reused_holdout",role="test")
    result=evaluator.execute(run.restore(rows),key)
    result.update(key=key,selection_sha256=run.sha(selection_path),qualified_hours=len(cohort),requested_hours=96,
        exposure_status=plan["status"])
    run.save(target,result)
    print("REUSED HOLDOUT",key[:8],result["status"],result.get("RMSE"),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=run.ROOT/"output/taiwan_frozen_pool_search_final")
    parser.add_argument("--freeze-plan",action="store_true")
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args(); out=args.output.resolve()
    freeze_plan(out)
    if not args.freeze_plan:
        selection=json.loads((out/"strict_validation_selection_frozen.json").read_text(encoding="utf-8"))
        selected={r["key"]:r["graph"] for r in selection["selection"]}
        receipts=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker,key,rows,out) for key,rows in selected.items()]
            for future in as_completed(futures): receipts.append(future.result())
        run.save(out/"holdout_receipts.json",receipts)
        scalar=[{k:v for k,v in r.items() if k not in {"failures"}} for r in receipts]
        pd.DataFrame(scalar).to_csv(out/"reused_holdout_metrics.csv",index=False)
        pd.DataFrame([{k:r[k] for k in ["mode","seed","key"]} for r in selection["selection"]]).merge(
            pd.DataFrame(scalar),on="key",validate="many_to_one").to_csv(out/"seed_reused_holdout_metrics.csv",index=False)
