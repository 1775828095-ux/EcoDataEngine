"""Independent product readback, frozen-policy comparisons and day-level intervals."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tools"))
from run_multisource_joint_experiment import OUT, DATA, GRID, sha, save
from ecodata.algorithms.portrait_rules import parameterized_terms
from ecodata.algorithms.capabilities import describe_algorithm
from ecodata.algorithms.platform_search import platform_registry


def interval(paired):
    grouped=pd.DataFrame(paired).groupby("day").difference.mean().to_numpy()
    rng=np.random.default_rng(20261023)
    draws=rng.choice(grouped,size=(2000,len(grouped)),replace=True).mean(1)
    return dict(mean=float(grouped.mean()),ci95=[float(x) for x in np.quantile(draws,[.025,.975])],
                day_groups=len(grouped),meaning="comparator minus method RMSE; positive favors method; exploratory conditional interval")


def main():
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    selection=json.loads((OUT/"development_selection.json").read_text(encoding="utf-8"))
    assert all(sha(p)==value for p,value in protocol["hashes"].items()),"Frozen code/data changed"
    data=pd.read_parquet(DATA);data.station_id=data.station_id.astype(str)
    registry=platform_registry();cost={s.id:describe_algorithm(s).compute_cost for s in registry.specs()}
    all_runs=[];product_count=0;probe_count=0;failures=[];pools=set();scores=set()
    test=[]
    for stage in ("development","holdout","test"):
        folders=json.loads((OUT/(stage+"_index.json")).read_text(encoding="utf-8"))
        for name in folders:
            folder=Path(name);report=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            isolation=json.loads((folder/"isolation.json").read_text(encoding="utf-8"))
            assert not set(isolation["available"])&set(isolation["held"])
            stamp=pd.Timestamp(f"2019-08-{isolation['day']:02d} 12:30")
            held=data[data.time.eq(stamp)&data.station_id.isin(isolation["held"])].copy()
            assert held.value.notna().all() and len(held)==isolation["held_count"]
            probes=json.loads((folder/"station_probes.json").read_text(encoding="utf-8"))
            probe_count+=probes["graph_fold_attempts"]
            assert set(probes["train_station_ids"])==set(isolation["available"])
            for probe in probes["measurements"].values():
                for fold in probe["folds"]:
                    assert not set(fold["fit_stations"])&set(fold["held_stations"])
                    assert set(fold["fit_stations"]+fold["held_stations"])==set(isolation["available"])
                if probe["metrics"] is not None:
                    error=np.asarray(probe["prediction"])-np.asarray(probes["training_values"])
                    assert abs(np.sqrt(np.mean(error**2))-probe["metrics"]["RMSE"])<1e-10
            rows={r["key"]:r for r in report["rows"]}
            assert len(rows)==11 and len(rows)==report["counts"]["candidate_count"]
            for row in rows.values():
                if row["status"]!="PASS":failures.append(dict(folder=name,row=row));continue
                path=Path(row["product"]);assert sha(path)==row["product_sha256"]
                with xr.open_dataset(path) as ds:
                    assert bool(np.isfinite(ds.pm25).all()) and float(ds.pm25.min())>=0
                    values=ds.pm25.interp(lat=xr.DataArray(held.latitude.to_numpy(),dims="held"),
                                         lon=xr.DataArray(held.longitude.to_numpy(),dims="held")).values
                error=values-held.value.to_numpy()
                assert len(error)==row["denominator"]
                for metric,value in dict(RMSE=np.sqrt(np.mean(error**2)),MAE=np.abs(error).mean(),Bias=error.mean()).items():
                    assert abs(value-row[metric])<1e-9
                product_count+=1
            pools.add(report["candidate_pool"]["fingerprint"])
            scores.add(len({row["matching"]["score_interval"][0] for row in rows.values()}))
            def rank(parameters):
                return min(rows,key=lambda k:(-parameterized_terms(rows[k]["matching"]["terms"],parameters)[1],
                    sum(cost[s] for s in rows[k]["steps"]),len(rows[k]["steps"]),k))
            recorded_parameters=report["portraits"]["model"]["scoring_parameters"]
            assert rank(recorded_parameters)==report["recommended_key"]
            all_runs.append(dict(folder=name,stage=stage,seconds=report["total_service_seconds"],probe_seconds=probes["seconds"]))
            if stage=="test":
                initial=rank({});calibrated=report["recommended_key"]
                choices=dict(fixed_single=selection["best_single"],fixed_combination=selection["best_combination"],
                    strongest_fixed=selection["best_fixed"],initial_score=initial,calibrated_score=calibrated)
                test.append(dict(day=isolation["day"],count=isolation["nominal_count"],seed=isolation["seed"],
                    choices=choices,rows=rows,oracle=min(r["RMSE"] for r in rows.values()),
                    random_expectation=float(np.mean([r["RMSE"] for r in rows.values()]))))
    assert len(pools)==1 and not failures
    methods={}
    for method in test[0]["choices"]:
        values=[t["rows"][t["choices"][method]] for t in test]
        methods[method]=dict(mean_task_RMSE=float(np.mean([v["RMSE"] for v in values])),
            pooled_RMSE=float(np.sqrt(sum(v["RMSE"]**2*v["denominator"] for v in values)/sum(v["denominator"] for v in values))),
            MAE=float(np.mean([v["MAE"] for v in values])),Bias=float(np.mean([v["Bias"] for v in values])),
            mean_oracle_regret=float(np.mean([v["RMSE"]-t["oracle"] for v,t in zip(values,test)])),
            combination_choices=sum("fusion.residual" in v["steps"] for v in values),
            unique_choices=len({t["choices"][method] for t in test}))
    methods["random_expectation"]=dict(mean_task_RMSE=float(np.mean([t["random_expectation"] for t in test])))
    rng=np.random.default_rng(20261023)
    random_means=[np.mean([t["rows"][str(rng.choice(sorted(t["rows"])))]["RMSE"] for t in test]) for _ in range(1000)]
    comparisons={}
    for method,baseline in [("fixed_combination","fixed_single"),("calibrated_score","strongest_fixed"),
                            ("calibrated_score","initial_score"),("calibrated_score","random_expectation")]:
        pairs=[]
        for t in test:
            baseline_error=t["random_expectation"] if baseline=="random_expectation" else t["rows"][t["choices"][baseline]]["RMSE"]
            pairs.append(dict(day=t["day"],difference=baseline_error-t["rows"][t["choices"][method]]["RMSE"]))
        comparisons[method+"_vs_"+baseline]=interval(pairs)
    candidate_results={}
    for key,row in test[0]["rows"].items():
        candidate_results[key]=dict(steps=row["steps"],mean_RMSE=float(np.mean([t["rows"][key]["RMSE"] for t in test])))
    report=dict(status="COMPLETED_EXPLORATORY",tasks=len(all_runs),test_tasks=len(test),real_products_read_back=product_count,
        training_probe_graph_folds=probe_count,failures=failures,pool_fingerprint=next(iter(pools)),
        unique_score_values_per_task=sorted(scores),methods=methods,comparisons=comparisons,
        random_1000_policy_means_interval=[float(x) for x in np.quantile(random_means,[.025,.975])],
        candidates=candidate_results,mean_oracle_RMSE=float(np.mean([t["oracle"] for t in test])),
        task_choices=[dict(day=t["day"],count=t["count"],seed=t["seed"],choices=t["choices"]) for t in test],
        costs=dict(total_public_entry_seconds=sum(r["seconds"] for r in all_runs),
            total_training_probe_seconds=sum(r["probe_seconds"] for r in all_runs),
            scope="Full-pool audit timing including shared-node caches; not standalone policy runtime comparison"),
        limitations=protocol["limitations"],calibration_parameters=selection["calibration"].get("parameters"),
        claim_boundary="Local CV-backed three-table score; not zero-shot data-portrait ranking or proven CMAQ benefit")
    save(OUT/"independent_results.json",report)
    print(json.dumps(dict(methods=methods,comparisons=comparisons,counts=[len(all_runs),product_count,probe_count]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
