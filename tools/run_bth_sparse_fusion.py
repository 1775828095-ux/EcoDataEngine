"""Train-40-only, fixed-site sparse spatial CV. Reuse existing RF/affine/IDW."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import ConvexHull,Delaunay
from scipy.spatial.distance import cdist
from ecodata.algorithms.bth_exploratory_dag import xy,FEATURES
from ecodata.algorithms.epa_task import EPATask
from ecodata.algorithms.dag import Artifact,DAGError
from ecodata.algorithms.pm25_alignment import observation_centers
from ecodata.algorithms.quality_aware import fit_affine_source
from ecodata.evaluation.quality_gates import ResultQualityGate
from run_pm25_spatiotemporal_peak_diagnostics import idw_weights

GROUND=Path("D:/codex/数据/京津冀_PM25融合测试_201908/01_地面监测站/ground_pm25_201908_60stations.csv")
METADATA=GROUND.parent/"ground_metadata_status.json"
ALIGNED=ROOT/"output/bth_pm25_fusion_20261020_final/native_grid_covariates_NOT_FUSION.nc"
METHODS=["IDW","GEOS_affine","RF_GEOS","RF_ERA5","RF_BOTH","RF_GEOS_OOF_mean_residual",
         "RF_ERA5_OOF_mean_residual","RF_BOTH_OOF_mean_residual"]
RF_FEATURES={"RF_GEOS":FEATURES["geos"],"RF_ERA5":FEATURES["era5"],"RF_BOTH":FEATURES["both"]}
SEEDS=[11,22,33,44,55,66,77,88,99,110]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,default=str,indent=2,allow_nan=False),encoding="utf-8")


def read_train_only(path):
    rows=[];skipped=0
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"]!="train":skipped+=1;continue
            rows.append(dict(station_id=row["station_id"],stored_time=row["time_as_stored"],
                longitude=float(row["longitude_as_stored"]),latitude=float(row["latitude_as_stored"]),
                value=pd.to_numeric(row["pm25_as_stored"],errors="coerce"),split="train",unit="ug m-3"))
    return pd.DataFrame(rows),skipped


def spatial_order(stations,seed):
    """Distance-weighted sampling without replacement; no PM2.5 input or rerolls."""
    stations=stations.sort_values("station_id").reset_index(drop=True)
    points=xy(stations);rng=np.random.default_rng(seed)
    chosen=[int(rng.choice(ConvexHull(points).vertices))]
    while len(chosen)<20:
        distance=cdist(points,points[chosen]).min(axis=1)**2;distance[chosen]=0
        chosen.append(int(rng.choice(len(points),p=distance/distance.sum())))
    return stations.station_id.iloc[chosen].tolist()


def metrics(truth,prediction):
    truth=np.asarray(truth,float);prediction=np.asarray(prediction,float)
    valid=np.isfinite(prediction);e=prediction[valid]-truth[valid];y=truth[valid]
    sst=float(((y-y.mean())**2).sum()) if len(y) else 0.
    return dict(expected=len(y)+int((~valid).sum()),predicted=len(y),coverage=float(valid.mean()) if len(valid) else 0.,
        failure_rate=float((~valid).mean()) if len(valid) else 1.,
        RMSE=float(np.sqrt(np.mean(e**2))) if len(e) else None,MAE=float(np.mean(np.abs(e))) if len(e) else None,
        Bias=float(e.mean()) if len(e) else None,R2=float(1-(e**2).sum()/sst) if sst else None)


class SparseRFTask(EPATask):
    """Only input assertions differ from EPA; the existing RF fitting method is unchanged."""
    def check(self,table,*,history=False):
        quality=super().check(table,history=history)
        quality.update(status="WARN",coordinate_status="WGS84_USER_ASSUMPTION_NOT_VERIFIED",
            scope="BTH train-only retrospective hourly spatial CV, NOT EPA daily data")
        return quality


def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),ROOT/"src/ecodata/algorithms/epa_task.py",ROOT/"src/ecodata/algorithms/quality_aware.py",
          ROOT/"src/ecodata/algorithms/bth_exploratory_dag.py",ROOT/"tools/run_pm25_spatiotemporal_peak_diagnostics.py"]
    signature={str(p):sha(p) for p in code}
    if (out/"config.json").exists():
        config=json.loads((out/"config.json").read_text(encoding="utf-8"))
        if config["code_hashes"]!=signature:raise DAGError("Changed numerical code requires new output version")
        assert all(sha(p)==s for p,s in config["input_hashes"].items())
        return
    raw,skipped=read_train_only(GROUND)
    stations=raw[["station_id","longitude","latitude"]].drop_duplicates().sort_values("station_id")
    if len(stations)!=40 or stations.station_id.duplicated().any():raise DAGError("Original 40 training IDs/coordinates changed")
    old=json.loads((ROOT/"output/golem_multisource_round2_preparation/preparation_audit.json").read_text(encoding="utf-8"))
    allowed={r["station_id"] for r in old["station"]["per_station"] if r["splits"]==["train"]}
    assert set(stations.station_id)==allowed
    combinations=[];full_area=ConvexHull(xy(stations)).volume
    for seed in SEEDS:
        order=spatial_order(stations,seed)
        for count in [5,10,20]:
            chosen=order[:count]
            area=ConvexHull(xy(stations[stations.station_id.isin(chosen)])).volume/full_area
            combinations.append(dict(seed=seed,count=count,available=chosen,heldout=sorted(allowed-set(chosen)),
                ordered20=order,hull_area_fraction=area))
    assert all(len({tuple(sorted(c["available"])) for c in combinations if c["count"]==n})==10 for n in [5,10,20])
    save(out/"station_combinations.json",combinations)
    config=dict(status="EXPLORATORY_TRAIN40_SPATIAL_CV",counts=[5,10,20],seeds=SEEDS,methods=METHODS,
        input_hashes={str(p):sha(p) for p in [GROUND,METADATA,ALIGNED]},code_hashes=signature,
        raw_nontrain_rows_skipped_before_time_or_target_parsing=skipped,original_validation_or_test_targets_parsed=False,
        coordinate_convention="WGS84 working assumption -> EPSG:32650 distances, unverified original datum",
        GEOS_definition="RH35, original lev72 retained; NOT verified surface; raw source not same concentration definition",
        B_direct_error="NOT_COMPARABLE_RH35_VS_UNSPECIFIED_STATION_BASIS; no direct physical-concentration RMSE",
        RF_config=dict(n_estimators=50,max_depth=6,min_samples_leaf=3,random_state=11,n_jobs=1),
        training="whole-month time series from selected sites only; no heldout site at any date; NOT temporal forecasting",
        residual="Existing EPATask station-group OOF mean training-period residual, static per site; NOT hourly in-sample residual",
        spatial_selection="nested distance-squared weighted random prefix; coordinates only, no retries by quality/error",
        support="contemporaneous finite available-site convex hull; >=3 noncollinear sites; no unsupported extrapolation",
        primary_cohort="all remaining original train sites, finite truth and geometric support, identical across methods",
        density_control="same 20 remaining sites (outside selected20) and supported by selected5 at each hour, all densities",
        uncertainty="1000 paired seed+shared UTC-day resamples, descriptive conditional CI; overlapping groups are not independent studies",
        stable_advantage_rule="positive paired CI lower bound AND >=8/10 winning groups AND full common-cohort prediction coverage",
        bootstrap_seed=20261020,search_runs=0,new_algorithms=0)
    save(out/"config.json",config)
    raw["time"]=observation_centers(raw.stored_time,json.loads(METADATA.read_text(encoding="utf-8")))
    if raw.duplicated(["station_id","time"]).any():raise DAGError("Duplicate station/hour")
    with xr.open_dataset(ALIGNED,engine="h5netcdf") as source:
        times=source.time.values
        base=pd.MultiIndex.from_product([times,stations.station_id],names=["time","station_id"]).to_frame(index=False)
        base=base.merge(stations,on="station_id",validate="many_to_one").merge(raw[["time","station_id","value"]],on=["time","station_id"],how="left",validate="one_to_one")
        for key,var in {"geos":"pm25_rh35_gcc",**{n:n for n in FEATURES["era5"]}}.items():
            field=source[var].squeeze(drop=True).transpose("time","lat","lon")
            sampled=field.interp(lat=xr.DataArray(stations.latitude.to_numpy(),dims="station"),
                lon=xr.DataArray(stations.longitude.to_numpy(),dims="station"))
            base[key]=sampled.transpose("time","station").values.reshape(-1)
    base["split"]="train";base["unit"]="ug m-3"
    base.loc[~np.isfinite(base.value)|base.value.lt(0),"value"]=np.nan
    assert np.isfinite(base[FEATURES["both"]].values).all()
    base.to_parquet(out/"train40_features.parquet",index=False)
    stations.to_csv(out/"stations40.csv",index=False)
    save(out/"prepared_integrity.json",dict(features_sha256=sha(out/"train40_features.parquet"),rows=len(base),
        valid_targets=int(base.value.notna().sum()),only_original_train_ids=True,aligned_grid_reused=True))


def support_mask(source,query):
    if len(source)<3:return np.zeros(len(query),bool)
    points=xy(source)
    if np.linalg.matrix_rank(points-points.mean(0))<2:return np.zeros(len(query),bool)
    return Delaunay(points).find_simplex(xy(query))>=0


def execute_group(out,count,seed,stage="run",smoke_hours=3):
    out=Path(out);folder=out/stage/f"n{count}_seed{seed}"
    if (folder/"result.json").exists():return json.loads((folder/"result.json").read_text(encoding="utf-8"))
    folder.mkdir(parents=True,exist_ok=True);started=perf_counter()
    combo=next(c for c in json.loads((out/"station_combinations.json").read_text(encoding="utf-8")) if c["count"]==count and c["seed"]==seed)
    data=pd.read_parquet(out/"train40_features.parquet")
    history=data[data.station_id.isin(combo["available"])&data.value.notna()].copy()
    assert set(history.station_id)==set(combo["available"]) and not set(history.station_id)&set(combo["heldout"])
    times=sorted(data.time.unique())
    selected_times=times if stage=="run" else [times[i] for i in np.unique(np.linspace(0,len(times)-1,smoke_hours).astype(int))]
    states={};failures=[];costs={m:0. for m in METHODS};fit_records={}
    for method,features in RF_FEATURES.items():
        clock=perf_counter()
        try:
            task=SparseRFTask(None,xr.Dataset(),features=tuple(features),train_end=str(max(times)),sealed_test_stations=tuple(combo["heldout"]))
            state=task.fit_rf({"history":Artifact("bth_history",history)}, {}, {})["state"].value
            assert all(not set(f["fit_stations"])&set(f["held_stations"]) and set(f["fit_stations"]+f["held_stations"])<=set(combo["available"]) for f in state["folds"])
            states[method]=state;joblib.dump(state,folder/(method+".joblib"))
            fit_records[method]=dict(features=features,fit_stations=sorted(history.station_id.unique()),fit_rows=len(history),
                train_start=str(history.time.min()),train_end=str(history.time.max()),folds=state["folds"],
                residual_definition=state["residual_definition"])
        except Exception as exc:failures.append(dict(method=method,stage="fit",error=str(exc)))
        costs[method]+=perf_counter()-clock;costs[method+"_OOF_mean_residual"]=costs[method]
    clock=perf_counter();affine=None
    try:affine=fit_affine_source(history.geos.to_numpy(),history.value.to_numpy())
    except Exception as exc:failures.append(dict(method="GEOS_affine",stage="fit",error=str(exc)))
    costs["GEOS_affine"]=perf_counter()-clock
    predictions=[];eligible=[];quality=[]
    for time,all_sites in data[data.time.isin(selected_times)].groupby("time",sort=True):
        live=all_sites[all_sites.station_id.isin(combo["available"])&all_sites.value.notna()].copy()
        target=all_sites[all_sites.station_id.isin(combo["heldout"])&all_sites.value.notna()].copy()
        mask=support_mask(live,target)
        eligibility=target[["station_id","time"]].copy();eligibility["supported"]=mask;eligibility["available_now"]=len(live)
        eligible.append(eligibility)
        target=target.loc[mask].copy()
        if target.empty:continue
        few=all_sites[all_sites.station_id.isin(combo["ordered20"][:5])&all_sites.value.notna()]
        common=support_mask(few,target)&~target.station_id.isin(combo["ordered20"]).to_numpy()
        frame=target[["station_id","time","value","geos"]].rename(columns={"value":"truth","geos":"GEOS_RH35_feature"})
        frame["common_density"]=common;frame["available_now"]=len(live)
        frame["nearest_available_km"]=cdist(xy(target),xy(live)).min(1)/1000.
        clock=perf_counter();frame["IDW"]=idw_weights(xy(live)/1000.,xy(target)/1000.)@live.value.to_numpy()
        costs["IDW"]+=perf_counter()-clock
        clock=perf_counter();frame["GEOS_affine"]=affine[0]*target.geos.to_numpy()+affine[1] if affine else np.nan
        costs["GEOS_affine"]+=perf_counter()-clock
        for method,features in RF_FEATURES.items():
            clock=perf_counter();base=np.full(len(target),np.nan);corrected=base.copy()
            try:
                if method not in states:raise DAGError("Fitted state unavailable; see fit failure")
                state=states[method]
                base=state["model"].predict(target[features].to_numpy())
                residual=state["residuals"]
                corrected=base+idw_weights(xy(residual)/1000.,xy(target)/1000.)@residual.value.to_numpy()
            except Exception as exc:failures.append(dict(time=str(time),method=method,stage="predict",error=str(exc)))
            frame[method]=base;frame[method+"_OOF_mean_residual"]=corrected
            elapsed=perf_counter()-clock;costs[method]+=elapsed;costs[method+"_OOF_mean_residual"]+=elapsed
        for method in METHODS:
            frame[method+"_raw"]=frame[method];frame[method]=frame[method].clip(lower=0)
        predictions.append(frame)
    result=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=["truth",*METHODS,"common_density"])
    result.to_parquet(folder/"predictions.parquet",index=False)
    pd.concat(eligible,ignore_index=True).to_parquet(folder/"eligibility.parquet",index=False)
    reread=pd.read_parquet(folder/"predictions.parquet")
    np.testing.assert_allclose(reread[METHODS],result[METHODS],equal_nan=True)
    rows=[]
    for cohort in ["all_supported","common_density"]:
        part=result if cohort=="all_supported" else result[result.common_density]
        for method in METHODS:
            values=part[method].to_numpy(float)
            ds=xr.Dataset({"estimate":("sample",values)});ds.estimate.attrs["units"]="ug m-3"
            gate=ResultQualityGate().validate(ds,ds,variable="estimate",nonnegative=True)
            quality.append(dict(method=method,cohort=cohort,numeric_gate=gate,geodetic_status="WORKING_ASSUMPTION",exploratory=True))
            rows.append(dict(count=count,seed=seed,method=method,cohort=cohort,**metrics(part.truth,values),
                seconds=costs[method],negative_projection_count=int(part[method+"_raw"].lt(0).sum()) if len(part) else 0))
    pd.DataFrame(rows).to_csv(folder/"metrics.csv",index=False)
    save(folder/"fit_audit.json",dict(available=combo["available"],heldout=combo["heldout"],records=fit_records,
        affine=affine,original_validation_and_test_used=False,history_label_sha256=hashlib.sha256(history[["station_id","time","value"]].to_csv(index=False).encode()).hexdigest()))
    save(folder/"quality.json",quality);save(folder/"failures.json",failures)
    summary=dict(count=count,seed=seed,stage=stage,common_samples=len(result),common_density_samples=int(result.common_density.sum()),
        geometry_coverage=float(pd.concat(eligible).supported.mean()),seconds=perf_counter()-started,failures=len(failures),
        predictions_sha256=sha(folder/"predictions.parquet"),config_sha256=sha(out/"config.json"))
    save(folder/"result.json",summary);print("DONE",stage,count,seed,round(summary["seconds"],2),len(result),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=["prepare","smoke","run"],required=True);parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args();prepare(args.output)
    if args.stage=="smoke":
        for n in [5,10,20]:execute_group(args.output,n,SEEDS[0],"smoke")
    if args.stage=="run":
        for n in [5,10,20]:
            result=json.loads((args.output/"smoke"/f"n{n}_seed{SEEDS[0]}"/"result.json").read_text(encoding="utf-8"))
            assert result["common_samples"]>0 and result["failures"]==0,"Smoke must pass before batch"
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(execute_group,args.output,n,s) for n in [5,10,20] for s in SEEDS]
            summary=[future.result() for future in as_completed(jobs)]
        save(args.output/"batch_summary.json",summary)


if __name__=="__main__":main()
