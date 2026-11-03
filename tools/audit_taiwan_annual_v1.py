"""Uniform-cohort validation enumeration, frozen selection, then selected test evaluation."""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import r2_score

from taiwan_working_contract import CONFIG, CONTRACT, OUT, frozen_selection, save, sha
from run_taiwan_annual_v1 import METHODS


def scores(frame):
    error = frame.predicted.to_numpy()-frame.observed.to_numpy()
    return dict(n=len(frame), RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(abs(error))),
        Bias=float(error.mean()), R2=float(r2_score(frame.observed,frame.predicted)),
        stations=int(frame.station_id.nunique()), dates=int(pd.to_datetime(frame.time,utc=True).dt.tz_convert("Asia/Taipei").dt.date.nunique()))


def assert_unique_truth(frame):
    if not frame.groupby(["time","station_id"])["observed"].nunique().le(1).all():
        raise ValueError("Different methods have inconsistent target observations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    output = OUT / "evaluation" / args.split
    retry = 0
    while output.exists():
        if (output/"verification.json").exists():
            print("Evaluation already completed; preserve results",flush=True)
            return
        retry += 1
        output = OUT / "evaluation" / f"{args.split}_attempt_{retry}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((OUT/f"{args.split}_run_manifest.json").read_text())
    assert not manifest["failures"] and len(manifest["completed"]) == 24
    predictions, executions, coverage = [], [], []
    for receipt in manifest["completed"]:
        folder = OUT / receipt["output"]
        for name, collection in [("predictions",predictions),("executions",executions),("coverage",coverage)]:
            frame = pd.read_parquet(folder/f"{name}.parquet")
            frame["experiment"] = receipt["experiment"]
            collection.append(frame)
    p, r, c = [pd.concat(values,ignore_index=True) for values in [predictions,executions,coverage]]
    assert p.split.eq(args.split).all()
    assert not p.duplicated(["experiment","method","time","station_id"]).any()
    frozen = frozen_selection() if args.split == "test" else None
    summary, details, readbacks = [], [], []
    selections = {}
    for experiment in METHODS:
        group = p.loc[p.experiment.eq(experiment)].copy()
        methods = [frozen[experiment]["method"]] if frozen else METHODS[experiment]
        wide = group.pivot(index=["time","station_id"],columns="method",values="predicted")
        common = wide.index[np.isfinite(wide[methods]).all(axis=1)]
        assert_unique_truth(group)
        eligible = group.set_index(["time","station_id"]).index.isin(common)
        main_set = group.loc[eligible]
        cov = c.loc[c.experiment.eq(experiment)]
        for method in methods:
            rows = main_set.loc[main_set.method.eq(method)].copy()
            run = r.loc[r.experiment.eq(experiment) & r.method.eq(method)]
            all_rows = group.loc[group.method.eq(method)].sort_values(["time","station_id"])
            result = dict(experiment=experiment,method=method,**scores(rows),
                potential_targets=int(cov.possible.sum()), valid_ground=int(cov.ground_valid.sum()),
                covariate_matched=int(cov.covariate_matched.sum()), qualified_targets=int(cov.qualified.sum()),
                actual_predictions=int(all_rows.predicted.notna().sum()), failures=int(run.status.eq("FAIL").sum()),
                calls=len(run), seconds=float(run.seconds.sum()), projected=int(run.projection_count.sum()),
                input_rejected_hours=int(cov.status.ne("ELIGIBLE").sum()), requested_hours=len(cov))
            result["coverage_of_valid_ground"] = result["actual_predictions"]/result["valid_ground"]
            result["coverage_of_qualified"] = result["actual_predictions"]/result["qualified_targets"]
            result["failure_rate"] = result["failures"]/result["calls"]
            result["quality"] = "WORKING_CONTRACT_SOURCE_WARN" if not result["failures"] else "PARTIAL_EXECUTION_FAILURE"
            summary.append(result)
            for category in ["month","station_id"]:
                for value, part in rows.groupby(category):
                    details.append(dict(experiment=experiment,method=method,group=category,key=str(value),**scores(part)))
            times = pd.to_datetime(all_rows.time,utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
            ds = xr.Dataset({"pm25":("sample",all_rows.predicted.to_numpy()),
                "observed":("sample",all_rows.observed.to_numpy()),
                "before_projection":("sample",all_rows.before_projection.to_numpy()),
                "prediction_valid":("sample",all_rows.predicted.notna().to_numpy().astype("int8")),
                "station_id":("sample",all_rows.station_id.to_numpy(dtype=str)),
                "time_bounds":(("sample","bounds"),np.column_stack([times,times+np.timedelta64(1,"h")]))},
                coords={"time":("sample",times),"longitude":("sample",all_rows.longitude.to_numpy()),"latitude":("sample",all_rows.latitude.to_numpy())},
                attrs=dict(contract_version=CONTRACT["version"],contract_sha256=sha(CONFIG),working_assumptions="true",
                    author_original_definitions_verified="false",crs="EPSG:4326",distance_crs="EPSG:32651",
                    split=args.split,method=method,quality=result["quality"],grid_coverage="NOT_APPLICABLE_POINT_OUTPUT"))
            for variable in ["pm25","observed","before_projection"]:
                ds[variable].attrs["units"] = CONTRACT["ground"]["unit"]
            ds.time.attrs.update(timezone="UTC",bounds="time_bounds",interval="[t,t+1h)")
            for name in ["time","time_bounds"]:
                ds[name].encoding.update(units="hours since 2016-12-31 16:00:00",calendar="proleptic_gregorian")
            file = output/f"{experiment}_{method}.nc"
            ds.to_netcdf(file,engine="h5netcdf")
            with xr.open_dataset(file) as check:
                np.testing.assert_allclose(check.pm25.values,all_rows.predicted,equal_nan=True)
                np.testing.assert_array_equal(check.prediction_valid.values,np.isfinite(check.pm25.values))
                assert check.pm25.attrs["units"] == CONTRACT["ground"]["unit"]
                assert not (check.pm25.values[np.isfinite(check.pm25.values)] < 0).any()
                np.testing.assert_array_equal(check.time_bounds.values[:,1]-check.time_bounds.values[:,0],np.full(len(all_rows),np.timedelta64(1,"h")))
            readbacks.append(dict(file=str(file.relative_to(OUT)),sha256=sha(file),rows=len(all_rows),status="PASS_WITH_SOURCE_WARN"))
        if not frozen:
            candidates = [row for row in summary if row["experiment"] == experiment and row["failures"] == 0]
            if not candidates:
                raise ValueError("No fully executed validation candidate; test remains sealed")
            best = min(candidates,key=lambda row:(row["RMSE"],row["seconds"]))
            if best["dates"] < 100 or best["n"] < 1000:
                raise ValueError("Insufficient full-year common validation evidence; test remains sealed")
            selections[experiment] = best
    pd.DataFrame(summary).to_csv(output/"comparison.csv",index=False)
    pd.DataFrame(details).to_csv(output/"by_month_and_station.csv",index=False)
    c.to_parquet(output/"all_coverage.parquet",index=False)
    r.to_parquet(output/"all_execution_records.parquet",index=False)
    save(output/"verification.json",dict(split=args.split,contract_sha256=sha(CONFIG),readbacks=readbacks,
        real_golem_graph_calls=int(r.graph_executed.sum()), no_test_based_selection=True,
        author_definitions_verified=False, new_algorithms=False))
    if not frozen:
        selection = OUT/"selection_frozen.json"
        if selection.exists():
            raise ValueError("Never overwrite an existing model selection")
        save(selection,dict(contract_sha256=sha(CONFIG),selection_source="validation only",final_test_targets_used=False,
            **selections, multisource_feasible_configurations=4, optimizer_claim="NO_GOLEM_ADVANTAGE_PROVEN_FOUR_CONFIGURATIONS_EXHAUSTIVELY_ENUMERATED"))
        save(OUT/"selection_freeze_receipt.json",dict(sha256=sha(selection),frozen_utc=datetime.now(timezone.utc).isoformat(),
            test_not_evaluated_at_freeze=True))
        save(OUT/"exhaustive_objectives.json", dict(candidates=[row for row in summary if row["experiment"] == "experiment3"],
            objective="reject failed configurations, minimize common-cohort validation RMSE in ug m-3; cost tie-break only",
            cost_unit="measured algorithm seconds; not a hardware-independent complexity estimate", graphs_executed=True,
            feasible_graph_count=4, formal_abc_run=False, reason="Small complete search space exactly enumerated; no evidence of evolutionary search advantage"))
    save(OUT/f"{args.split}_evaluation_receipt.json",dict(output=str(output.relative_to(OUT)),
        comparison_sha256=sha(output/"comparison.csv"), contract_sha256=sha(CONFIG)))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
