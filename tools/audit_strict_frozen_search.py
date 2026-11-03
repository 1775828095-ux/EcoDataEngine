"""Strict pool identity, real generation and numerical search acceptance."""
import argparse
import json
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
import run_taiwan_composition_search as run
from golem.core.optimisers.genetic.operators import base_mutations


def generation(out):
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    backend,factory,frozen=run.setup()
    allowed={r["id"] for r in frozen}
    adapters=backend.adapters()
    assert ("fusion.residual","fit") not in adapters
    assert ("fusion.residual","direct") not in adapters
    assert ("fusion.residual","residual_interpolate") not in adapters
    assert backend.idw is None and backend.rf_fit is None
    manifest=[]
    for (operation,mode),adapter in adapters.items():
        assert operation in allowed or operation in {"source.train","source.query","quality.station"}
        manifest.append(dict(operation=operation,mode=mode,adapter=adapter.execute.__module__+"."+adapter.execute.__qualname__,
            inputs={n:p.kind for n,p in adapter.inputs.items()},outputs={n:p.kind for n,p in adapter.outputs.items()}))
    run.save(out/"actual_kernel_binding_manifest.json",dict(frozen_ids=sorted(allowed),task_modes=manifest,
        hidden_rf=False,hidden_idw=False,baseline_only=["IDW","RF+residual IDW","GWR","linear regression"],
        estimator_backend="official PyKrige OrdinaryKriging/UniversalKriging; no new learned model",
        residual_backend="existing project residual_difference/residual_addition with explicit availability masks"))
    first=run.cases(selected=[protocol["times"][0]])
    results=[]
    for operation,mode in [("station.pykrige_universal","direct"),("fusion.residual","sum")]:
        random.seed(11); np.random.seed(11)
        _,f,_=run.setup()
        initial=f.initial()
        f.preferred["raw_prediction"]=(operation,mode)
        graph=base_mutations.add_separate_parent_node(initial,f)
        status=f.inspect(graph,complete=True)
        assert status["gap"]==0
        evaluator=run.Evaluator(first,out/"generation_acceptance"/operation)
        key=run.fingerprint(graph,f)
        result=evaluator.execute(graph,key)
        assert result["score"]<1e6 and result["coverage"]==1.
        states=[a.value for engine in evaluator.pipeline._dag_executors.values() for outputs in engine.cache.values()
            for a in outputs.values() if a.kind=="background_state"]
        for state in states:
            assert state["backend_class"].startswith("pykrige.")
            assert set(state["fit_stations"])==set(first[0][1].station_id)
            for fold in state["folds"]: assert set(fold["fit_stations"]).isdisjoint(fold["heldout_stations"])
        results.append(dict(operation=operation,mode=mode,native_mutation="add_separate_parent_node",attempts=1,
            graph=run.record(graph),gap_after=0,result=result,
            fitted_backends=[s["backend_class"] for s in states],factory_trace=f.trace))
    run.save(out/"strict_generation_acceptance.json",dict(status="PASS_REAL_GENERATION_AND_EXECUTION",results=results,
        test_labels_read=False,initial_graph="same incomplete quality node with fixed real data leaves"))
    snapshot=out/"code_snapshot"
    for relative,expected in protocol["code_hashes"].items():
        source=run.ROOT/relative
        assert run.sha(source)==expected
        target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(source,target)
    print("PASS strict kernels and two native-generated real routes",flush=True)


def final(out):
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    runs=json.loads((out/"search_summary.json").read_text(encoding="utf-8"))
    assert len(runs)==15 and all(r["best"] and not r["optimizer_error"] for r in runs)
    catalog=json.loads((out/"catalog.json").read_text(encoding="utf-8"))
    enum=[]; behavior={}
    for key in catalog:
        folder=out/"enumeration"/key
        result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
        values=pd.read_parquet(folder/"predictions.parquet")
        result.pop("failures",None)
        enum.append(dict(key=key,**result))
        digest=__import__("hashlib").sha256(values.prediction.to_numpy().round(10).tobytes()).hexdigest()
        behavior.setdefault(digest,[]).append(key)
        with h5py.File(folder/"predictions.nc") as ds:
            np.testing.assert_allclose(ds["pm25"][:],values.prediction,equal_nan=True)
            np.testing.assert_array_equal(ds["residual_usage"][:],values.residual_usage)
    pd.DataFrame(enum).to_csv(out/"enumeration_results.csv",index=False)
    run.save(out/"numeric_behavior_groups.json",dict(round_decimals=10,groups=behavior))
    rows=[]; curves=[]; feedback=[]
    for s in runs:
        detail=json.loads((out/"search"/f"{s['mode']}_{s['seed']}"/"summary.json").read_text(encoding="utf-8"))
        best=float("inf")
        for i,r in enumerate(detail["objective_records"],1):
            if r.get("RMSE") is not None and r["score"]<1e6: best=min(best,r["RMSE"])
            curves.append(dict(mode=s["mode"],seed=s["seed"],evaluation=i,best_RMSE=best if np.isfinite(best) else None,
                real_execution=r["real_execution"],cache_hit=r["cache_hit"],status=r["status"],elapsed=r["elapsed"]))
        c=s["counts"]
        qualified=[r for r in detail["objective_records"] if r.get("RMSE") is not None and r["score"]<1e6]
        feedback.append(dict(mode=s["mode"],seed=s["seed"],
            scored_callbacks=len(detail["objective_records"]),qualified_numeric_callbacks=len(qualified),
            qualified_callback_rate=len(qualified)/len(detail["objective_records"]),
            unique_qualified_graphs=s["unique_legal_graphs"],
            first_qualified_callback=next((i for i,r in enumerate(detail["objective_records"],1) if r in qualified),None),
            newly_executed_failures=c.get("numeric_failures",0),
            incomplete_callbacks=sum(r["status"]=="INCOMPLETE" for r in detail["objective_records"]),
            proposal_rejections=c.get("proposal_rejections",0),posterior_rejections=c.get("posterior_rejections",0)))
        rows.append(dict(mode=s["mode"],seed=s["seed"],RMSE=s["best"]["RMSE"],MAE=s["best"]["MAE"],Bias=s["best"]["Bias"],
            key=s["best"]["key"],evaluations=len(detail["objective_records"]),real=c.get("real_graph_evaluations",0),
            cache=c.get("cache_hits",0),proposals=c.get("proposals",0),rejected=c.get("proposal_rejections",0),
            posterior_rejections=c.get("posterior_rejections",0),gap_completions=s["gap_completions"],
            wall_seconds=s["wall_seconds"],stop_reason=s["stop_reason"]))
    frame=pd.DataFrame(rows); curve=pd.DataFrame(curves)
    random_runs=json.loads((out/"random_reference_summary.json").read_text(encoding="utf-8"))
    assert len(random_runs)==5 and all(r["best"] and not r["test_labels_read"] for r in random_runs)
    for s in random_runs:
        detail=json.loads((out/"random_reference"/str(s["seed"])/"summary.json").read_text(encoding="utf-8"))
        best=float("inf")
        for i,r in enumerate(detail["records"],1):
            if r.get("RMSE") is not None and r["score"]<1e6: best=min(best,r["RMSE"])
            curves.append(dict(mode="R",seed=s["seed"],evaluation=i,best_RMSE=best if np.isfinite(best) else None,
                real_execution=r["real_execution"],cache_hit=r["cache_hit"],status=r["status"],elapsed=r["elapsed"]))
        rows.append(dict(mode="R",seed=s["seed"],RMSE=s["best"]["RMSE"],MAE=s["best"]["MAE"],Bias=s["best"]["Bias"],
            key=s["best"]["key"],evaluations=s["evaluations"],real=s["real"],cache=s["cache"],
            proposals=s["proposals"],rejected=0,posterior_rejections=0,gap_completions=0,
            wall_seconds=s["wall_seconds"],stop_reason="SAME_CALLBACK_OR_WALL_LIMIT"))
    frame=pd.DataFrame(rows); curve=pd.DataFrame(curves)
    frame.to_csv(out/"seed_results.csv",index=False)
    pd.DataFrame(feedback).to_csv(out/"search_feedback_metrics.csv",index=False)
    curve.to_csv(out/"budget_curves.csv",index=False)
    common=[]
    for seed,group in frame.groupby("seed"):
        n=int(group.evaluations.min())
        for mode in "ABCR":
            r=curve.loc[curve.seed.eq(seed)&curve["mode"].eq(mode)&curve.evaluation.eq(n)].iloc[0]
            common.append(dict(seed=int(seed),mode=mode,common_callback_budget=n,best_RMSE=r.best_RMSE,elapsed=r.elapsed))
    pd.DataFrame(common).to_csv(out/"common_budget_comparison.csv",index=False)
    frame.groupby("mode")[["RMSE","evaluations","real","proposals","wall_seconds"]].agg(["mean","std"]).to_csv(out/"mode_means_std.csv")
    selection=out/"strict_validation_selection_frozen.json"
    if not selection.exists():
        chosen=[dict(mode=s["mode"],seed=s["seed"],key=s["best"]["key"],graph=s["best"]["graph"]) for s in runs]
        chosen += [dict(mode="R",seed=s["seed"],key=s["best"]["key"],graph=s["best"]["graph"]) for s in random_runs]
        from datetime import datetime,timezone
        run.save(selection,dict(protocol_sha256=run.sha(out/"protocol.json"),selection=chosen,
            frozen_at=datetime.now(timezone.utc).isoformat(),
            test_status="Only post-freeze reused-holdout evaluation allowed; original test targets were previously exposed"))
    tests=ET.parse(run.ROOT/"output/composition_strict_final_tests.xml").getroot()
    totals={k:sum(int(t.attrib.get(k,0)) for t in tests.iter("testsuite")) for k in ["tests","failures","errors","skipped"]}
    assert not totals["failures"] and not totals["errors"]
    for relative,expected in protocol["code_hashes"].items(): assert run.sha(run.ROOT/relative)==expected
    run.save(out/"strict_final_acceptance.json",dict(status="PASS_ENGINEERING_STRICT_POOL_EXPLORATORY_SEARCH",
        frozen_pool_sha256=run.sha(run.POOL),contract_sha256=run.sha(run.CONFIG),graphs=len(catalog),numeric_behavior_groups=len(behavior),
        new_search_graph_evaluations=int(frame.real.sum()),scored_callbacks=int(frame.evaluations.sum()),cache_hits=int(frame.cache.sum()),
        test_labels_read=False,unfrozen_estimators_used=False,tests=totals,
        science="Engineering execution verified; no automatic claim of superiority or fresh independent confirmation"))
    print(frame[["mode","seed","evaluations","real","RMSE"]].sort_values(["seed","mode"]).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["generation","final"],default="generation")
    parser.add_argument("--output",type=Path,default=run.ROOT/"output/taiwan_frozen_pool_search_final")
    args=parser.parse_args(); run.OUT=args.output.resolve()
    (generation if args.phase=="generation" else final)(run.OUT)
