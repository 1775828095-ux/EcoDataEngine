"""Read-only numerical diagnosis of frozen five-station results; no fit/tuning."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import xarray as xr
from run_bth_sparse_fusion import SEEDS, METHODS, RF_FEATURES, ALIGNED, xy, idw_weights, sha, save


def correlation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None


def error_decomposition(frame,method):
    """Exact descriptive MSE split into hourly common error and spatial error."""
    error=frame[method]-frame.truth
    common=error.groupby(frame.time).transform("mean")
    spatial=error-common
    mse=float(np.mean(error**2));common_mse=float(np.mean(common**2));spatial_mse=float(np.mean(spatial**2))
    np.testing.assert_allclose(mse,common_mse+spatial_mse,rtol=1e-12,atol=1e-12)
    truth_center=frame.truth-frame.truth.groupby(frame.time).transform("mean")
    predicted_center=frame[method]-frame[method].groupby(frame.time).transform("mean")
    hour=frame.groupby("time")[["truth",method]].mean()
    return dict(n=len(frame),RMSE=np.sqrt(mse),common_hour_error_RMSE=np.sqrt(common_mse),
        spatial_error_RMSE=np.sqrt(spatial_mse),common_hour_error_MSE_fraction=common_mse/mse,
        temporal_mean_correlation=correlation(hour.truth,hour[method]),
        spatial_anomaly_correlation=correlation(truth_center,predicted_center),
        spatial_std_ratio=float(predicted_center.std()/truth_center.std()))


def replay(table,live,state):
    """Replay the already executed IDW, RF_BOTH and static-mean-residual paths."""
    base=state["model"].predict(table[list(state["features"])].to_numpy())
    residual=state["residuals"]
    correction=idw_weights(xy(residual)/1000,xy(table)/1000)@residual.value.to_numpy()
    return {"IDW":idw_weights(xy(live)/1000,xy(table)/1000)@live.value.to_numpy(),
        "RF_BOTH":base,"RF_BOTH_OOF_mean_residual":base+correction}


def diagnose(root):
    root=Path(root);out=root/"diagnosis_20261020"
    if out.exists():raise FileExistsError("Preserve diagnosis evidence; use a new version instead of overwriting")
    manifest=json.loads((root/"reproducibility.json").read_text(encoding="utf-8"))
    data=pd.read_parquet(root/"train40_features.parquet")
    assert data.split.eq("train").all() and data.station_id.nunique()==40
    rows=[];correlations=[];probes=[];static=[];training=[]
    for seed in SEEDS:
        folder=root/"run"/f"n5_seed{seed}"
        for filename in ["predictions.parquet","RF_GEOS.joblib","RF_ERA5.joblib","RF_BOTH.joblib"]:
            path=folder/filename
            assert sha(path)==manifest["artifacts"][str(path.relative_to(root))]
        pred=pd.read_parquet(folder/"predictions.parquet")
        audit=json.loads((folder/"fit_audit.json").read_text(encoding="utf-8"))
        selected=set(audit["available"]);held=set(audit["heldout"])
        assert len(selected)==5 and not selected&held
        assert set(pred.station_id)<=held
        history=data[data.station_id.isin(selected)&data.value.notna()]
        frame=pred.merge(data.drop(columns="value"),on=["time","station_id"],validate="one_to_one")
        for method in METHODS:
            rows.append(dict(seed=seed,method=method,**error_decomposition(frame,method)))
        for feature in ["geos",*RF_FEATURES["RF_ERA5"]]:
            hour=frame.groupby("time")[["truth",feature]].mean()
            y=frame.truth-frame.groupby("time").truth.transform("mean")
            x=frame[feature]-frame.groupby("time")[feature].transform("mean")
            correlations.append(dict(seed=seed,feature=feature,global_correlation=correlation(frame.truth,frame[feature]),
                temporal_mean_correlation=correlation(hour.truth,hour[feature]),spatial_anomaly_correlation=correlation(y,x)))
        for method,features in RF_FEATURES.items():
            state=joblib.load(folder/(method+".joblib"))
            fitted=state["model"].predict(history[features].to_numpy())
            training.append(dict(seed=seed,method=method,train_sites=len(selected),train_rows=len(history),
                in_sample_RMSE=float(np.sqrt(np.mean((fitted-history.value)**2))),
                heldout_RMSE=float(np.sqrt(np.mean((pred[method]-pred.truth)**2))),
                assessment="in-sample diagnostic only, NOT cross-validation accuracy"))
            delta=pred[method+"_OOF_mean_residual_raw"]-pred[method+"_raw"]
            grouped=delta.groupby(pred.station_id)
            maximum=float(grouped.std(ddof=0).max())
            assert maximum<1e-10,"Stored static residual path unexpectedly varies in time"
            static.append(dict(seed=seed,method=method,maximum_same_site_correction_time_std=maximum,
                median_target_RF_error_time_std=float((pred.truth-pred[method]).groupby(pred.station_id).std().median()),
                residual_state_rows=len(state["residuals"]),source_history_rows=len(history)))
            if method!="RF_BOTH":continue
            times=sorted(pred.time.unique())
            for index in np.unique(np.linspace(0,len(times)-1,3).astype(int)):
                time=times[index]
                live=data[data.time.eq(time)&data.station_id.isin(selected)&data.value.notna()].copy()
                target=frame[frame.time.eq(time)].copy()
                before=replay(target,live,state)
                perturbed=live.copy();perturbed["value"]+=1.
                after=replay(target,perturbed,state)
                for name in before:
                    np.testing.assert_allclose(before[name],target[name+"_raw"],atol=1e-10)
                    change=after[name]-before[name]
                    np.testing.assert_allclose(change,1. if name=="IDW" else 0.,atol=1e-10)
                    probes.append(dict(seed=seed,time=str(time),method=name,targets=len(target),
                        intervention="all finite input-station observations +1 ug/m3, frozen weights/state/features",
                        change_min=float(change.min()),change_max=float(change.max()),state_refitted=False))
    out.mkdir()
    details=pd.DataFrame(rows);details.to_csv(out/"error_decomposition_by_seed.csv",index=False)
    details.groupby("method").mean(numeric_only=True).drop(columns="seed").to_csv(out/"error_decomposition_summary.csv")
    fields=pd.DataFrame(correlations);fields.to_csv(out/"feature_correlation_by_seed.csv",index=False)
    fields.groupby("feature").mean(numeric_only=True).drop(columns="seed").to_csv(out/"feature_correlation_summary.csv")
    pd.DataFrame(probes).to_csv(out/"frozen_state_observation_probe.csv",index=False)
    pd.DataFrame(static).to_csv(out/"static_correction_audit.csv",index=False)
    pd.DataFrame(training).to_csv(out/"in_sample_vs_heldout.csv",index=False)
    with xr.open_dataset(ALIGNED,engine="h5netcdf") as grid:
        grid_info={axis:dict(size=len(grid[axis]),step=np.unique(np.round(np.diff(grid[axis].values),8)).tolist()) for axis in ["lat","lon"]}
    assert all(sha(path)==value for path,value in manifest["input_hashes"].items())
    save(out/"diagnosis.json",dict(status="EXECUTED_NO_FIT_NO_TUNING",seeds=SEEDS,count=5,
        targets=int(details[details.method.eq("IDW")].n.sum()),input_grid=grid_info,probe_records=len(probes),
        original_validation_or_test_used=False,model_fit_calls=0,new_algorithms=0,changed_numerical_algorithms=False,
        source_artifact_fingerprints_verified=True,script_sha256=sha(__file__),
        limits=["Feature correlation does not establish an information-theoretic upper bound",
            "Frozen-state perturbation is conditional on the already trained state, not refitting response",
            "In-sample fitting error is optimistic and is NOT OOF validation",
            "WGS84 and GEOS lev72/RH35 limitations unchanged"],
        findings=["RF prediction features exclude current ground PM2.5 observations",
            "OOF time series is collapsed into one mean residual per selected station",
            "Existing dynamic residual method is not called by sparse runner; it requires post-training dates",
            "Cannot quantify how much of the IDW gap would disappear with observation conditioning without another controlled experiment"]))
    print(details.groupby("method")[["RMSE","common_hour_error_RMSE","spatial_error_RMSE","spatial_anomaly_correlation","spatial_std_ratio"]].mean().to_string())
    print(fields.groupby("feature")[["global_correlation","temporal_mean_correlation","spatial_anomaly_correlation"]].mean().to_string())
    print(pd.DataFrame(training).groupby("method")[["in_sample_RMSE","heldout_RMSE"]].mean().to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    diagnose(parser.parse_args().root)
