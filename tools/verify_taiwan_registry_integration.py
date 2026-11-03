"""Real validation-only checks for Registry/Pipeline/semantics integration.

Never reads test labels. The failure probe deliberately uses a previously known
validation coverage failure for engineering QA, not accuracy/model selection.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import run_taiwan_composition_search as run


def state_scope_probe(out):
    from sklearn.model_selection import GroupKFold
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    stamp,train,target=run.cases(selected=[protocol["times"][0]])[0]
    _,factory,_=run.setup()
    catalog=json.loads((out/"catalog.json").read_text(encoding="utf-8"))
    key,rows=next((k,r) for k,r in catalog.items()
        if any(n["parameters"].get("regressor")=="rf" for n in r)
        and all(n["operation"] in {"fusion.residual","source.train","source.query","quality.station"} for n in r))
    from ecodata.algorithms.dag import Artifact
    from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
    metadata=dict(unit="ug m-3",metric_crs="EPSG:32651",time=stamp,contract=run.CONTRACT["version"])
    query=target.drop(columns=["value","split"])
    result=UnifiedAlgorithmPipeline(factory.task_registry).run_dag(run.restore(rows),
        task_id="taiwan_pm25_station_aod",version="explicit-real-state-scope-probe",external={
            "train":Artifact("training",train,metadata),"query":Artifact("query",query,metadata)})
    states=[a.value for outputs in result["intermediates"].values() for a in outputs.values() if a.kind=="background_state"]
    assert len(states)==1
    state=states[0]
    # The existing RF stores a sorted membership list, not the bootstrap input
    # order. Compare membership; never reorder the actual training frame.
    assert sorted(state["fit_stations"])==sorted(train.station_id.tolist())
    assert set(state["fit_stations"]).isdisjoint(target.station_id)
    roles=pd.read_parquet(run.ANNUAL/"data/station_metadata.parquet")
    test_ids=set(roles.loc[roles.split.eq("test"),"station_id"])
    assert test_ids.isdisjoint(state["fit_stations"])
    folds=[]
    for fit,held in GroupKFold(5).split(train[["aod"]],groups=train.station_id):
        fit_ids=train.station_id.iloc[fit].tolist()
        held_ids=train.station_id.iloc[held].tolist()
        assert set(fit_ids).isdisjoint(held_ids)
        folds.append(dict(fit_stations=fit_ids,heldout_training_stations=held_ids))
    run.save(out/"real_training_state_scope.json",dict(status="PASS_REAL_EXECUTION",time=stamp,key=key,
        state_count=1,fit_stations=state["fit_stations"],query_stations=target.station_id.tolist(),
        actual_training_input_order=train.station_id.tolist(),metadata_station_list_order="sorted membership only",
        test_station_labels_read=False,test_station_geometry_only_checked=True,
        crossfit=state["crossfit"],folds=folds,residual_finite=bool(np.isfinite(state["residual"]).all()),
        query_has_labels=False,trace=result["trace"]))
    print("PASS actual RF state, 5 station-disjoint folds, no held-out labels")


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=run.ROOT/"output/taiwan_composition_search_v3")
    parser.add_argument("--state-only",action="store_true")
    args=parser.parse_args()
    previous=run.OUT
    run.OUT=args.output.resolve()
    if args.state_only:
        state_scope_probe(run.OUT)
        return
    protocol=json.loads((run.OUT/"protocol.json").read_text())
    _,factory,frozen=run.setup()
    assert len(frozen)==34
    assert set(json.loads((previous/"catalog.json").read_text()))==set(json.loads((run.OUT/"catalog.json").read_text()))
    for spec in frozen: factory.task_registry.get(spec["id"])
    smoke=json.loads((run.OUT/"smoke_summary.json").read_text())
    comparisons=[]
    for arm in smoke:
        folder=run.OUT/"smoke"/f"{arm['mode']}_{arm['seed']}"
        details=json.loads((folder/"summary.json").read_text())
        assert details["best"] and not details["optimizer_error"]
        for row in details["objective_records"]:
            if not row["real_execution"]: continue
            key=row["key"]
            new=pd.read_parquet(folder/"executions"/key/"predictions.parquet")
            old=pd.read_parquet(previous/"enumeration"/key/"predictions.parquet")
            np.testing.assert_allclose(new.prediction,old.prediction,rtol=0,atol=1e-10)
            trace=json.loads((folder/"executions"/key/"execution_trace.jsonl").read_text().splitlines()[0])
            assert any(o["metadata"].get("input_quality") for t in trace["trace"] for o in t.get("outputs",{}).values())
            comparisons.append(dict(arm=arm["mode"],key=key,max_difference=float(np.max(abs(new.prediction-old.prediction)))))
    # A real, intentionally selected validation failure, not synthetic data.
    old_failure=next(p for p in (previous/"final_frozen/validation").glob("*/*/result.json")
        if "a2367629" in str(p) and json.loads(p.read_text())["failure_hours"])
    stamp=json.loads(old_failure.read_text())["failures"][0]["time"]
    selected=run.cases(selected=[stamp])
    assert len(selected)==1
    catalog=json.loads((run.OUT/"catalog.json").read_text())
    barnes=next((key,rows) for key,rows in catalog.items() if len(rows)==4 and any(r["operation"]=="station.metpy_barnes" for r in rows))
    idw=next((key,rows) for key,rows in catalog.items() if len(rows)==4 and any(r["operation"]=="fusion.residual" and r["parameters"].get("mode")=="direct" for r in rows))
    evaluator=run.Evaluator(selected,run.OUT/"real_validation_failure_probe")
    cache={}
    first=run.real_graph_feedback(evaluator,run.restore(barnes[1]),barnes[0],cache)
    second=run.real_graph_feedback(evaluator,run.restore(barnes[1]),barnes[0],cache)
    assert first["score"]==1e6 and first["real_execution"] and not first["cache_hit"]
    assert second["score"]==1e6 and second["cache_hit"] and not second["real_execution"]
    assert first["failures"][0]["failure_node"] is not None
    alternative=run.real_graph_feedback(evaluator,run.restore(idw[1]),idw[0],cache)
    assert alternative["coverage"]==1. and alternative["real_execution"]
    # Fault-injection copy proves invalid units stop before interpolation.
    # It is not an allegation that the untouched original data has bad units.
    stamp,train,target=selected[0]
    train=train.copy(); train["unit"]="mg m-3"
    faulty=run.Evaluator([(stamp,train,target)],run.OUT/"fault_injection_unit_copy")
    blocked=faulty.execute(run.restore(idw[1]),idw[0])
    assert blocked["score"]==1e6 and blocked["failure_hours"]==1
    assert "unit" in blocked["failures"][0]["error"].lower()
    assert not any(t.get("operation") in [s["id"] for s in frozen] for t in blocked["failures"][0]["trace"])
    snapshot=run.OUT/"code_snapshot"
    for relative,expected in protocol["code_hashes"].items():
        source=run.ROOT/relative
        assert run.sha(source)==expected
        target_path=snapshot/relative
        target_path.parent.mkdir(parents=True,exist_ok=True)
        if not target_path.exists(): shutil.copy2(source,target_path)
    run.save(run.OUT/"registry_pipeline_acceptance.json",dict(status="PASS_REAL_VALIDATION_ENGINEERING",
        registry_frozen_ids_found=34,executable_task_ids=7,generated_graphs=21,
        requirements=factory.requirements.as_dict(),real_smoke_replays=comparisons,
        real_failure_probe=dict(time=stamp,station_split="validation",first_execution=True,
            second_execution=False,second_cache_hit=True,failure_node=first["failures"][0]["failure_node"],
            reason=first["failures"][0]["error"],alternative_quality_passed=True),
        fault_injection="in-memory unit copy blocked before any interpolation; raw input untouched",
        test_labels_read=False,test_or_model_selection_changed=False,source_hashes_verified=True))
    print("PASS Registry/Pipeline/semantics; real failure cached; alternate quality passed; unit fault blocked")


if __name__=="__main__": main()
