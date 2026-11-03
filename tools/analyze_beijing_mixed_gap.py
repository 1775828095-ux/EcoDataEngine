"""Independent full-cohort analysis, frozen-test followup and clustered intervals."""
import json,sys
from pathlib import Path
import hashlib
import numpy as np,pandas as pd
from prepare_beijing_mixed_gap import OUT,save,sha


def metrics(frame,prediction):
    finite=np.isfinite(prediction);error=prediction-frame.truth.to_numpy();valid_error=error[finite]
    return dict(n=len(frame),coverage=float(finite.mean()),failures=int((~finite).sum()),
        RMSE=float(np.sqrt(np.mean(error**2))) if finite.all() else None,
        MAE=float(abs(error).mean()) if finite.all() else None,Bias=float(error.mean()) if finite.all() else None,
        partial_RMSE_not_ranked=float(np.sqrt(np.mean(valid_error**2))) if finite.any() else None,
        high_RMSE=float(np.sqrt(np.mean(error[frame.high]**2))) if finite[frame.high].all() and frame.high.any() else None)


def grouped(frame,prediction,identity):
    result=[]
    for columns in [["station"],["big","sub"],["station","replica","big","sub"]]:
        for keys,indices in frame.groupby(columns).indices.items():
            if not isinstance(keys,tuple):keys=(keys,)
            part=frame.iloc[indices]
            result.append(dict(identity=identity,grouping="/".join(columns),**dict(zip(columns,keys)),**metrics(part,prediction[indices])))
    return result


def analyze():
    searches=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT/"search").glob("*/summary.json"))]
    assert len(searches)==50
    enumeration=json.loads((OUT/"enumeration_summary.json").read_text(encoding="utf-8"))
    freeze=json.loads((OUT/"selected_graphs_frozen.json").read_text(encoding="utf-8"))
    final=json.loads((OUT/"test_results.json").read_text(encoding="utf-8"))
    fixed=json.loads((OUT/"fixed_validation_results.json").read_text(encoding="utf-8"))
    by_test={r["key"]:r for r in final["results"]};runrows=[];curves=[];events=[]
    for r in searches:
        good=[x for x in r["rows"] if x["status"]=="PASS"]
        first=good[0] if good else None;best=r["best"]
        row=dict(mode=r["mode"],seed=r["seed"],scored=r["scored"],seconds=r["seconds"],cpu_seconds=r["cpu_seconds"],
            first_valid_evaluation=first["evaluation"] if first else None,first_valid_seconds=first["elapsed"] if first else None,
            unique_valid=r["unique_valid"],unique_behaviors=r["unique_behaviors"],best_validation=best["objective"] if best else None,
            test_objective=by_test[best["key"]]["objective"] if best else None,success=bool(best),optimizer_error=bool(r["optimizer_error"]),
            node_cache_hits=r["node_hits"],node_new=r["node_new"],kernel_hits=r["kernel_hits"],**r["counts"])
        if best:
            row.update(validation_point_RMSE=best["RMSE"],validation_MAE=best["MAE"],validation_Bias=best["Bias"],
                test_point_RMSE=by_test[best["key"]].get("RMSE"),test_MAE=by_test[best["key"]].get("MAE"),
                test_Bias=by_test[best["key"]].get("Bias"),test_coverage=by_test[best["key"]]["coverage"],
                test_success=by_test[best["key"]]["status"]=="PASS")
        else:row["test_success"]=False
        row["quality_pass_fraction_of_real"]=r["unique_valid"]/max(1,r["counts"].get("real_executions",0))
        row["valid_fraction_of_scored"]=r["unique_valid"]/max(1,r["scored"])
        row["structural_proposal_valid_fraction"]=1-r["counts"].get("proposal_rejections",0)/max(1,r["counts"].get("proposals",0))
        for column in ["platform_setup_seconds","platform_total_seconds","decision_service_seconds"]:
            if column in r:row[column]=r[column]
        for key in ["callbacks","deferred_callbacks","graph_cache_hits","incomplete_feedback","real_executions",
            "quality_failures","proposals","proposal_rejections","posterior_rejections","bounded_stop"]:row.setdefault(key,0)
        runrows.append(row);best_so_far=None;actual=0
        for x in r["rows"]:
            actual+=int(x["unique_execution"])
            if x["status"]=="PASS":best_so_far=x["objective"] if best_so_far is None else min(best_so_far,x["objective"])
            curves.append(dict(mode=r["mode"],seed=r["seed"],evaluation=x["evaluation"],real_executions=actual,seconds=x["elapsed"],
                platform_seconds=x["elapsed"]+r.get("platform_setup_seconds",0.),best=best_so_far,status=x["status"]))
        for ordinal,event in enumerate(r["events"],1):events.append(dict(mode=r["mode"],seed=r["seed"],callback=ordinal,origin="direct event",**event))
        deferred=r["counts"].get("deferred_callbacks",0)
        recorded_deferred=sum(e["status"]=="DEFERRED_NO_EXECUTION" for e in r["events"])
        missing_deferred=deferred-recorded_deferred
        assert missing_deferred>=0 and len(r["events"])+missing_deferred==r["counts"]["callbacks"],"Unexplained raw callback gap"
        for i in range(missing_deferred):
            events.append(dict(mode=r["mode"],seed=r["seed"],callback=len(r["events"])+i+1,
                status="DEFERRED_NO_EXECUTION",origin="derived from explicit deferred counter; individual timestamp not captured",elapsed=None))
    table=pd.DataFrame(runrows);table.to_csv(OUT/"search_runs.csv",index=False,encoding="utf-8-sig")
    table.groupby("mode").agg({"best_validation":["mean","std","min","max"],"test_objective":["mean","std"],
        "seconds":["mean","sum"],"real_executions":["mean","sum"],"unique_valid":"mean","success":"mean","test_success":"mean", "scored":"mean"}).to_csv(OUT/"search_summary.csv",encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(OUT/"best_so_far.csv",index=False);pd.DataFrame(events).to_csv(OUT/"raw_callbacks.csv",index=False)
    # Common attained budget prefix, never top up failed/early-stopped runs.
    prefix=[]
    for seed,g in table.groupby("seed"):
        k=int(g.scored.min())
        for mode in g["mode"]:
            values=[r for r in curves if r["seed"]==seed and r["mode"]==mode and r["evaluation"]<=k and r["best"] is not None]
            prefix.append(dict(seed=seed,mode=mode,common_scored_prefix=k,best=min(v["best"] for v in values) if values else None))
    pd.DataFrame(prefix).to_csv(OUT/"common_budget_prefix.csv",index=False)
    additional_prefix=[]
    for seed,g in table.groupby("seed"):
        for axis,column in [("real_executions","real_executions"),("seconds","seconds")]:
            cutoff=float(g[column].min())
            for mode in g["mode"]:
                values=[r["best"] for r in curves if r["seed"]==seed and r["mode"]==mode and r[axis]<=cutoff and r["best"] is not None]
                additional_prefix.append(dict(seed=seed,mode=mode,axis=axis,common_cutoff=cutoff,best=min(values) if values else None))
    pd.DataFrame(additional_prefix).to_csv(OUT/"common_execution_time_prefix.csv",index=False)
    equal_error=[]
    threshold=fixed["nearest"]["objective"]
    for r in searches:
        reached=[x for x in r["rows"] if x["status"]=="PASS" and x["objective"]<=threshold]
        first=reached[0] if reached else None
        equal_error.append(dict(mode=r["mode"],seed=r["seed"],threshold=threshold,reached=bool(first),
            scored=first["evaluation"] if first else None,seconds=first["elapsed"] if first else None))
    pd.DataFrame(equal_error).to_csv(OUT/"equal_error_cost.csv",index=False)
    # Read all actual predictions, including failed fixed baselines, same targets.
    groups=[];summary=[];predictions={}
    for year,collection,folder in [(2016,fixed,"fixed_validation"),(2017,by_test,"frozen_test")]:
        truth=pd.read_parquet(OUT/f"data/{year}/private_targets.parquet")
        truth.to_parquet(OUT/f"evaluation_queue_{year}.parquet",index=False)
        for name,r in collection.items():
            key=r["key"]
            with np.load(OUT/folder/key/"prediction.npz") as data:p=data["prediction"].copy()
            assert len(p)==len(truth)
            groups.extend(dict(year=year,**v) for v in grouped(truth,p,name))
            summary.append(dict(year=year,identity=name,status=r["status"],objective=r["objective"],**metrics(truth,p)))
            if year==2017:predictions[key]=p
    pd.DataFrame(groups).to_csv(OUT/"group_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUT/"fixed_and_frozen_metrics.csv",index=False)
    save(OUT/"frozen_mapping_with_results.json",[dict(**m,result=by_test[m["key"]] if m["key"] else None) for m in freeze["mapping"]])
    # Distinct structure vs observed numerical behavior, including dormant fallback.
    behavior={};trigger_rows=[]
    for r in enumeration["results"]:
        if r["status"]!="PASS":continue
        behavior.setdefault(r["prediction_fingerprint"],[]).append(r["key"])
        for node in r["node_trace"]:
            if node.get("operation") in {"prepare.conditional_fallback","prepare.visible_gap_split"}:
                trigger_rows.append(dict(graph=r["key"],operation=node["operation"],**node["outputs"]["field"]["metadata"]))
    pd.DataFrame(trigger_rows).to_csv(OUT/"control_trigger_counts.csv",index=False)
    confirmed=0
    enum_by_key={r["key"]:r for r in enumeration["results"]}
    for keys in behavior.values():
        first=Path(enum_by_key[keys[0]].get("reused_from",str(OUT/"enumeration"/keys[0])))
        with np.load(first/"prediction.npz") as d:representative=d["prediction"].copy()
        for key in keys[1:]:
            folder=Path(enum_by_key[key].get("reused_from",str(OUT/"enumeration"/key)))
            with np.load(folder/"prediction.npz") as d:np.testing.assert_allclose(d["prediction"],representative,atol=1e-8,rtol=0)
            confirmed+=1
    save(OUT/"behavior_audit.json",dict(structures=enumeration["count"],enumeration_complete=enumeration["complete"],
        passing=sum(r["status"]=="PASS" for r in enumeration["results"]),quantized_prediction_groups=len(behavior),
        within_group_pairs_confirmed_atol_1e8=confirmed,numerical_equivalence_not_universal=True,
        distinct_quantization_buckets_not_proven_inequivalent_at_all_tolerances=True))
    # Station/time two-way paired bootstrap. Continuous gaps stay in their start
    # week; all artificial replicas share temporal block weights, not independent.
    truth=pd.read_parquet(OUT/"data/2017/private_targets.parquet")
    anchors=truth.groupby(["case_id","block"]).time.transform("min")
    week=pd.factorize(anchors.dt.floor("7D"),sort=True)[0];g=truth.group_id.to_numpy()
    group_table=pd.read_csv(OUT/"data/2017/groups.csv");station_codes=pd.factorize(group_table.station,sort=True)[0]
    weights=1/(12*3*3*group_table.groupby(["station","replica","big"])["sub"].transform("count").to_numpy())
    ng=len(weights);nw=int(week.max()+1);nstation=12
    count=np.bincount(g*nw+week,minlength=ng*nw).reshape(ng,nw)
    rng=np.random.default_rng(20261020);replicates=1000
    sw=np.array([np.bincount(rng.integers(nstation,size=nstation),minlength=nstation) for _ in range(replicates)])
    tw=np.array([np.bincount(rng.integers(nw,size=nw),minlength=nw) for _ in range(replicates)])
    denominator=count@tw.T;active=denominator>0
    replicate_weights=weights[:,None]*sw[:,station_codes].T*active
    replicate_weights/=replicate_weights.sum(0)
    bootstrap={};point={}
    for key,p in predictions.items():
        if not np.isfinite(p).all():continue
        square=np.bincount(g*nw+week,weights=(p-truth.truth.to_numpy())**2,minlength=ng*nw).reshape(ng,nw)
        mse=np.divide(square@tw.T,denominator,out=np.zeros((ng,replicates)),where=active)
        bootstrap[key]=(np.sqrt(mse)*replicate_weights).sum(0);point[key]=by_test[key]["objective"]
    mappings=freeze["mapping"];policies={};blocked_policies=[]
    for mode in sorted({m["mode"] for m in mappings}):
        requested=[m["key"] for m in mappings if m["mode"]==mode]
        if any(k not in bootstrap for k in requested):
            blocked_policies.append(dict(mode=mode,reason="At least one no-solution or failed frozen output; no successful-subset arm CI"));continue
        keys=requested
        if keys:policies[mode]=(float(np.mean([point[k] for k in keys])),np.mean([bootstrap[k] for k in keys],axis=0),len(keys))
    comparisons=[]
    for a,b in [("H0","nearest"),("C","A"),("C","B"),("C","R"),("C","C-gap"),("R","nearest"),
                ("linear_nearest","nearest"),("C","linear_nearest"),("R","linear_nearest")]:
        if a not in policies or b not in policies:continue
        difference=policies[a][1]-policies[b][1]
        comparisons.append(dict(a=a,b=b,objective_difference=policies[a][0]-policies[b][0],
            ci95_low=float(np.quantile(difference,.025)),ci95_high=float(np.quantile(difference,.975)),
            assumption="exchangeable stations and weekly blocks; pointwise exploratory, no multiplicity correction",
            mask_replicas_not_independent=True))
    pd.DataFrame(comparisons).to_csv(OUT/"paired_environment_intervals.csv",index=False)
    save(OUT/"uncertainty_ineligible_policies.json",blocked_policies)
    paired=[]
    for other in ["A","B","R","C-gap"]:
        a=table[table["mode"].eq("C")].set_index("seed");b=table[table["mode"].eq(other)].set_index("seed")
        for metric in ["best_validation","test_objective","seconds"]:
            difference=a[metric]-b[metric]
            paired.append(dict(a="C",b=other,metric=metric,mean_difference=float(difference.mean()),std=float(difference.std()),
                environmental_independent_repeats=False))
    pd.DataFrame(paired).to_csv(OUT/"paired_seed_differences.csv",index=False)
    save(OUT/"analysis_complete.json",dict(search_runs=len(searches),test_unique_outputs=len(final["results"]),
        bootstrap_replicates=replicates,bootstrap_time_blocks=nw,enumeration_complete=enumeration["complete"],
        optimizer_errors=sum(bool(r["optimizer_error"]) for r in searches),freeze_sha256=sha(OUT/"selected_graphs_frozen.json")))
    print(table.groupby("mode")[["best_validation","test_objective","seconds","scored","unique_valid"]].mean().to_string())


if __name__=="__main__":analyze()
