"""Frozen exploratory multi-source experiment through the common Profile entry."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"tools"))
from ecodata.algorithms import platform_search as api, attribute_registry as ar
from ecodata.algorithms.profile_matching import fingerprint
from run_bth_sparse_fusion import spatial_order

OUT=ROOT/"output/multisource_joint_20261023_v2"
DATA=ROOT/"output/bth_sparse_fusion_20261020/train40_features.parquet"
GRID=ROOT/"output/bth_pm25_fusion_20261020_final/native_grid_covariates_NOT_FUSION.nc"
VERSION="multisource-probes-v1"
CALIBRATION="multisource-joint-20261023-v2"
POOL=["fusion.idw","station.pykrige_ordinary","fusion.station_rf_background","fusion.station_rf_weather",
      "fusion.station_rf_joint","fusion.gp_residual","fusion.residual","raster_task.nonnegative"]
DATES=dict(development=[2,4,6,8],holdout=[11,13],test=[18,20,22,24,26,28])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding="utf-8")


def register():
    attrs=deepcopy(ar.load()["attributes"])
    for metric in ("RMSE","MAE","abs_bias"):
        attrs.append(dict(id="probe_"+metric,definition="Available-station spatial CV "+metric,
            tables=["data","algorithm","model"],unit="bounded normalized utility",role="scoring",
            scoring_enabled=True,hard_allowed=False,provider="station_probe",property=metric,
            activation="constant_one",evidence="Three fixed GroupKFold partitions of available stations only",
            history=["Empirical local suitability; no independent accuracy or simulator sensitivity guarantee"]))
    attrs.append(dict(attrs[-1],id="runtime_tier",definition="Sum of declared compute tiers, not elapsed seconds",
                      provider="declared_cost",property=None,evidence="Existing algorithm compute tier declarations"))
    if (ar.ROOT/(VERSION+".json")).exists():
        assert ar.load(VERSION)["attributes"]==attrs
    else:api.publish_attribute_registry(VERSION,attrs,evidence="Frozen multi-source exploratory protocol")


def request(stations):
    return dict(task="multisource",variable="pm25",unit="ug/m3",semantic_type="continuous",spatial_semantics="point",
        allow_training=True,target_lat=sorted(stations.latitude.unique()),target_lon=sorted(stations.longitude.unique()),
        working_assumptions=["Station coordinates are treated as WGS84; original datum remains unverified"],
        model_profile=dict(id="cmaq",role="background",field="PM25"),
        model_requirements=dict(priorities={"probe_RMSE":.6,"probe_MAE":.2,"probe_abs_bias":.1,"runtime_tier":.1},
            hard=["nonnegative"],evidence="Declared concentration reconstruction utility and nonnegativity; NOT measured CMAQ sensitivity"),
        scoring=dict(registry_version=VERSION),preferences=dict(weights={"RMSE":1},scales={"RMSE":1}),
        parameters=dict(observation_crs="EPSG:4326",include_station_only_baselines=True,
            gp_length_scale_m=60000.,gp_signal_std=10.,gp_noise_std=2.,
            feature_groups=dict(background=["geos"],weather=["t2m","d2m","u10","v10","sp","tp"],
                                joint=["geos","t2m","d2m","u10","v10","sp","tp"])),
        profile_options=dict(max_steps=4,candidate_limit=128,candidate_pool=dict(algorithms=POOL)))


class StationHoldout:
    def __init__(self,held,role):
        self.held=held.copy()
        self.protocol=dict(kind="reference_error",role="independent_validation",objective="RMSE",variable="pm25",
            unit="ug/m3",denominator=len(held),study_partition=role,
            reference_sha256=fingerprint(held[["station_id","value"]].to_dict("records")))
    def __call__(self,ds):
        if ds.pm25.attrs.get("units")!="ug/m3":raise ValueError("Output unit mismatch")
        values=ds.pm25.interp(lat=xr.DataArray(self.held.latitude.to_numpy(),dims="held"),
                             lon=xr.DataArray(self.held.longitude.to_numpy(),dims="held")).values
        if not np.isfinite(values).all():raise ValueError("Incomplete fixed station cohort")
        error=values-self.held.value.to_numpy()
        return dict(objective=float(np.sqrt(np.mean(error**2))),RMSE=float(np.sqrt(np.mean(error**2))),
            MAE=float(np.abs(error).mean()),Bias=float(error.mean()),denominator=len(error),coverage=1.,accuracy_assessed=True)


def prepare_inputs(frame,grid,available):
    train=frame[frame.station_id.isin(available)&frame.value.notna()].copy()
    held=frame[~frame.station_id.isin(available)&frame.value.notna()].copy()
    if len(train)<8 or len(held)<4:raise ValueError("Insufficient finite available/held stations; no replacement draws")
    train=train[["station_id","longitude","latitude","value","time"]].assign(split="train",unit="ug/m3",variable="PM25")
    lat=sorted(frame.latitude.unique());lon=sorted(frame.longitude.unique())
    template=xr.Dataset({"pm25":(("lat","lon"),np.zeros((len(lat),len(lon))))},
        coords=dict(lat=lat,lon=lon,time=grid.time),attrs=dict(crs="EPSG:4326",source_id="geometry-template",data_role="prediction_template"))
    template.pm25.attrs=dict(units="ug/m3",physical_variable="PM25",spatial_semantics="point",
        physical_definition="placeholder for station-target estimate; zeros are geometry only, never concentration observations")
    cov=grid[["pm25_rh35_gcc","t2m","d2m","u10","v10","sp","tp"]].rename(pm25_rh35_gcc="geos").squeeze("lev",drop=True).load()
    cov.attrs=dict(crs="EPSG:4326",source_id="GEOS_RH35_and_ERA5",data_role="independent_covariates_NOT_same_variable_fusion")
    for field in cov.data_vars.values():field.attrs["data_kind"]="continuous"
    return [api.SearchInput("geometry.nc","primary",template.to_netcdf()),
        api.SearchInput("covariates.nc","auxiliary",cov.to_netcdf()),
        api.SearchInput("available_stations.nc","observations",xr.Dataset.from_dataframe(train.reset_index(drop=True)).to_netcdf())],held,train


def load_data():
    # Reused preparation has already skipped the original validation/test rows.
    data=pd.read_parquet(DATA);data["station_id"]=data.station_id.astype(str)
    assert data.split.eq("train").all() and data.station_id.nunique()==40
    stations=data[["station_id","longitude","latitude"]].drop_duplicates().sort_values("station_id")
    return data,stations


def run(stage,pilot=False):
    register();data,stations=load_data();req=request(stations)
    out=OUT/("pilot_v2" if pilot else "runs")
    if not pilot and not (OUT/"protocol.json").exists():raise ValueError("Freeze protocol first")
    if stage=="test" and not (OUT/"development_selection.json").exists():raise ValueError("Freeze fixed and calibrated policies before test")
    records=[]
    dates=DATES[stage][:1] if pilot else DATES[stage]
    with xr.open_dataset(GRID) as source:
        for day in dates:
            stamp=pd.Timestamp(f"2019-08-{day:02d} 12:30")
            frame=data[data.time.eq(stamp)].copy()
            assert len(frame)==40
            for seed in ([11] if pilot else [11,29]):
                order=spatial_order(stations,seed)
                for count in ([12] if pilot else [12,20]):
                    name=f"{stage}_d{day:02d}_n{count}_s{seed}"
                    folder=out/name
                    if (folder/"summary.json").exists():
                        result=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
                    else:
                        inputs,held,train=prepare_inputs(frame,source.sel(time=stamp),order[:count])
                        actual=deepcopy(req)
                        actual["profile_options"].update(group=f"day{day:02d}",evidence_role="development" if stage!="test" else "evaluation")
                        if stage=="test":actual["scoring"]={"registry_version":VERSION,"calibration_id":CALIBRATION}
                        save(folder/"isolation.json",dict(available=train.station_id.tolist(),held=held.station_id.tolist(),
                            day=day,nominal_count=count,actual_count=len(train),held_count=len(held),seed=seed,
                            fit_and_held_disjoint=not bool(set(train.station_id)&set(held.station_id)),
                            held_values_present_in_upload=False,natural_missing_not_truth=True))
                        clock=perf_counter()
                        result=api.submit_search(inputs,actual,StationHoldout(held,stage),folder,mode="Profile",
                            budget=dict(objective_limit=128,wall_seconds=600),seed=11)
                        print(name,"candidates",result["counts"]["candidate_count"],"failures",result["counts"].get("quality_failures"),
                            "seconds",round(perf_counter()-clock,2),flush=True)
                    if result["stop_reason"]!="CANDIDATE_POOL_COMPLETED":raise ValueError("Incomplete pool: "+str(folder))
                    records.append(str(folder))
    save(OUT/("pilot_index.json" if pilot else stage+"_index.json"),records)


def freeze():
    register();data,stations=load_data()
    assert not (OUT/"protocol.json").exists()
    sources=[DATA,GRID,Path(__file__)]
    sources+=list((ROOT/"src/ecodata/algorithms").glob("*.py"))
    hashes={str(p):sha(p) for p in sources}
    save(OUT/"protocol.json",dict(status="FROZEN_EXPLORATORY_NOT_NEW_BLIND_DATA",dates=DATES,utc_time="12:30",
        seeds=[11,29],available_counts=[12,20],unit="ug/m3 (equivalent spelling of original ug m-3; no value scaling)",
        primary="Withheld finite station PM2.5 RMSE; fixed common cohort per task",
        task="same-hour retrospective spatial reconstruction; NOT future forecasting",
        source_count="ground stations + GEOS predictor + ERA5 meteorological covariates",
        candidate_pool=POOL,max_steps=4,all_complete_graphs_really_executed=True,
        fitting="Each task refits numerical kernels on explicitly available current-hour stations only",
        scoring="Three fixed train-station CV folds; errors normalized by available-station std; same additive joint rule",
        calibration="Only development dates tune multipliers; holdout reports selected policy without choosing parameters",
        parameter_grid={"probe_RMSE.multiplier":[.5,1.,1.5,2.],"probe_abs_bias.multiplier":[.5,1.,1.5,2.],"runtime_tier.multiplier":[0.,.5,1.,1.5]},
        random="Uniform over unique legal graphs; 1000 independently seeded selector repeats; same audited result matrix",
        uncertainty="Paired day-cluster bootstrap 2000 replicates; masks/densities share day; descriptive conditional interval",
        limitations=["All dates belong to previously studied archive; exploratory held-out rerun, not a newly blind benchmark",
            "Original station datum unresolved; WGS84 working assumption retained",
            "GEOS RH35/lev72 used only as predictor; no claim of ground concentration equivalence",
            "CMAQ profile supplies field-level contract; utility weights explicitly declared, not measured simulator sensitivities",
            "CV probes execute candidate pipelines and add cost; no claim of zero-shot portrait inference or superior search efficiency"],
        hashes=hashes,request=request(stations)))


def calibrate():
    dev=json.loads((OUT/"development_index.json").read_text(encoding="utf-8"))
    hold=json.loads((OUT/"holdout_index.json").read_text(encoding="utf-8"))
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    report=api.calibrate_profile_scoring(dev+hold,CALIBRATION,holdout_groups=[f"day{x:02d}" for x in DATES["holdout"]],
        parameter_grid=protocol["parameter_grid"],registry_version=VERSION)
    matrix={}
    for folder in dev:
        result=json.loads((Path(folder)/"summary.json").read_text(encoding="utf-8"))
        for row in result["rows"]:
            if row["status"]!="PASS":raise ValueError("Failed development graph; retain and audit before calibration")
            matrix.setdefault(row["key"],dict(losses=[],steps=row["steps"]))["losses"].append(row["RMSE"])
    expected=len(dev)
    assert all(len(v["losses"])==expected for v in matrix.values())
    def choose(items):return min(items,key=lambda k:(np.mean(matrix[k]["losses"]),k))
    single=[k for k,v in matrix.items() if "fusion.residual" not in v["steps"]]
    combo=[k for k,v in matrix.items() if "fusion.residual" in v["steps"]]
    save(OUT/"development_selection.json",dict(calibration=report,best_single=choose(single),best_combination=choose(combo),
        best_fixed=choose(list(matrix)),candidates={k:dict(steps=v["steps"],mean_RMSE=float(np.mean(v["losses"]))) for k,v in matrix.items()},
        selected_on="development dates only; holdout and test not consulted"))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["freeze","pilot","development","holdout","calibrate","test"])
    args=parser.parse_args()
    if args.stage=="freeze":freeze()
    elif args.stage=="calibrate":calibrate()
    else:run("development" if args.stage=="pilot" else args.stage,pilot=args.stage=="pilot")
