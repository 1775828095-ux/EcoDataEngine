"""Audit integrated validation searches; no final-test access or new fitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import run_taiwan_composition_search as run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=run.ROOT/"output/taiwan_composition_search_v3")
    args=parser.parse_args()
    out=args.output.resolve()
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    results=json.loads((out/"search_summary.json").read_text(encoding="utf-8"))
    assert len(results)==15
    rows,curves,selections=[],[],[]
    canonical=None
    for s in results:
        folder=out/"search"/f"{s['mode']}_{s['seed']}"
        detail=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        assert not detail["optimizer_error"] and not detail["test_labels_read"]
        records=detail["objective_records"]
        assert len(records)<=20 and s["counts"]["proposals"]<=160
        best=float("inf")
        for index,row in enumerate(records,1):
            if row.get("RMSE") is not None and row["score"]<1e6: best=min(best,row["RMSE"])
            curves.append(dict(mode=s["mode"],seed=s["seed"],evaluation=index,
                best_RMSE=best if np.isfinite(best) else None,elapsed=row["elapsed"],
                real_execution=row["real_execution"],cache_hit=row["cache_hit"],gap=row.get("gap"),status=row["status"]))
            if not row["real_execution"]: continue
            values=pd.read_parquet(folder/"executions"/row["key"]/"predictions.parquet")
            target=values[["time","station_id","observed"]].copy()
            if canonical is None: canonical=target
            else: pd.testing.assert_frame_equal(canonical,target)
            assert len(values)==688 and np.isfinite(values.prediction).all()
            np.testing.assert_allclose(np.sqrt(np.mean((values.prediction-values.observed)**2)),row["RMSE"],rtol=0,atol=1e-12)
        assert s["best"] is not None
        c=s["counts"]
        rows.append(dict(mode=s["mode"],seed=s["seed"],RMSE=s["best"]["RMSE"],MAE=s["best"]["MAE"],
            Bias=s["best"]["Bias"],key=s["best"]["key"],evaluations=len(records),
            real=c.get("real_graph_evaluations",0),cache=c.get("cache_hits",0),proposals=c.get("proposals",0),
            rejected=c.get("proposal_rejections",0),posterior_rejections=c.get("posterior_rejections",0),
            gap_completions=s["gap_completions"],wall_seconds=s["wall_seconds"],stop_reason=s["stop_reason"]))
        selections.append(dict(mode=s["mode"],seed=s["seed"],key=s["best"]["key"],graph=s["best"]["graph"]))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"seed_results.csv",index=False)
    curve=pd.DataFrame(curves)
    curve.to_csv(out/"budget_curves.csv",index=False)
    common=[]
    for seed,group in frame.groupby("seed"):
        n=int(group.evaluations.min())
        for mode in "ABC":
            row=curve.loc[curve.seed.eq(seed)&curve["mode"].eq(mode)&curve.evaluation.eq(n)].iloc[0]
            common.append(dict(seed=int(seed),mode=mode,common_callback_budget=n,
                best_RMSE=float(row.best_RMSE),elapsed=float(row.elapsed)))
    pd.DataFrame(common).to_csv(out/"common_budget_comparison.csv",index=False)
    frame.groupby("mode")[["RMSE","evaluations","real","proposals","rejected","wall_seconds"]].agg(["mean","std"]).to_csv(out/"mode_means_std.csv")
    selection_path=out/"validation_selections_frozen.json"
    if not selection_path.exists():
        run.save(selection_path,dict(frozen_at=datetime.now(timezone.utc).isoformat(),selection=selections,
            protocol_sha256=run.sha(out/"protocol.json"),test_access="NOT_RUN: prior test targets already exposed; no renewed test-driven optimization"))
    for relative,expected in protocol["code_hashes"].items():
        source=run.ROOT/relative
        target=out/"code_snapshot"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            assert run.sha(source)==expected
            shutil.copy2(source,target)
        assert run.sha(target)==expected
    tests=ET.parse(run.ROOT/"output/composition_integrated_fixed_tests.xml").getroot()
    totals={k:sum(int(s.attrib.get(k,0)) for s in tests.iter("testsuite")) for k in ["tests","failures","errors","skipped"]}
    assert not totals["failures"] and not totals["errors"]
    integration=json.loads((out/"registry_pipeline_acceptance.json").read_text(encoding="utf-8"))
    assert integration["status"]=="PASS_REAL_VALIDATION_ENGINEERING"
    run.save(out/"integrated_search_acceptance.json",dict(status="PASS_ENGINEERING_EXPLORATORY_SEARCH",
        actual_search_runs=len(frame),same_validation_targets=len(canonical),new_graph_evaluations=int(frame.real.sum()),
        scored_callbacks=int(frame.evaluations.sum()),cache_hits=int(frame.cache.sum()),tests=totals,
        source_hashes_verified=True,selection_sha256=run.sha(selection_path),test_labels_read=False,
        scientific_limits=["Source clock/datum are working assumptions", "Only 21 meaningful complete graphs",
            "688 targets across 96 prespecified validation hours", "Previous final test set is not fresh",
            "Wall/proposal ceilings may stop before 20 callbacks; common-prefix comparison also supplied",
            "No claim of significant or universal superiority"],
        mode_means=frame.groupby("mode")[["RMSE","real","proposals","wall_seconds"]].mean().to_dict("index")))
    print(frame[["mode","seed","evaluations","real","RMSE"]].sort_values(["seed","mode"]).to_string(index=False))


if __name__=="__main__": main()
