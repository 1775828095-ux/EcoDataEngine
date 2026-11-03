"""Fixed functional run on existing PM2.5; final test stations remain sealed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.schemas import EcoDataBundle
from ecodata.io.readers import _frame_to_dataset

ROOT=Path(__file__).resolve().parents[1]
DATA=Path("D:/codex/数据/京津冀_PM25融合测试_201908")
OUT=ROOT/"output/capability_final_acceptance"


def main():
    samples=pd.read_parquet(ROOT/"output/pm25_background_month_screening_20261012/monthly_station_background_samples.parquet")
    samples=samples[samples.split.isin(["train","validation"])]
    time=samples.time.min()
    rows=samples[samples.time.eq(time)][["station_id","split","observed"]]
    coordinates=pd.read_csv(DATA/"05_对齐后的测试数据/station_split_40_10_10.csv")
    obs=rows.merge(coordinates[["station_id","latitude","longitude"]],on="station_id",validate="one_to_one").rename(columns={"observed":"value"})
    obs=obs.dropna(subset=["value","latitude","longitude"])
    geos=xr.load_dataset(DATA/"02_GEOS_CF模拟背景场/geos_cf_pm25_201908_bth.nc",engine="h5netcdf").pm25_rh35_gcc.sel(time=time).isel(lev=0,drop=True).drop_vars("time")
    merra=xr.load_dataset(DATA/"02b_MERRA2_GMI模拟背景场/merra2_gmi_pm25_201908_bth_ugm3.nc",engine="h5netcdf").pm25.interp(time=time).drop_vars("time")
    sources=[]
    for name,field in (("GEOS_CF_RH35",geos),("MERRA2_GMI_reconstructed",merra)):
        ds=field.to_dataset(name="pm25")
        ds.attrs.update(crs="EPSG:4326",source_id=name,data_kind="continuous")
        ds.pm25.attrs.update(units="ug m-3",physical_variable="PM2.5",data_kind="continuous")
        sources.append(EcoDataBundle(ds))
    caveats=["Exploratory original station coordinates treated approximately as WGS84; datum not independently certified",
        "GEOS-CF RH35 and MERRA2-GMI reconstructed PM2.5 definitions differ; calibration is empirical observation-equivalence, not a physical humidity conversion",
        "MERRA instantaneous values linearly matched to GEOS HH:30 window centre; not proven equal to an hourly average",
        "Inverse residual variance is an empirical reliability proxy, not certified independent source-error variance",
        "Standardized qc_flag=VALID means finite nonnegative numeric values and in-domain coordinates only; instrument QA remains UNKNOWN, not inferred from AQI"]
    records=[]
    def execute(name,context,algorithm):
        result=UnifiedAlgorithmPipeline(create_default_registry()).run(context,AlgorithmPlan(name,(algorithm,),0.,("Fixed functional acceptance, no model selection",)))
        result.data.dataset.attrs["experiment_limitations"]="; ".join(caveats)
        result.data.dataset.to_netcdf(OUT/(name+".nc"),engine="h5netcdf")
        records.append({"algorithm":algorithm,"output":name+".nc","report":result.report})
        return result
    context=AlgorithmContext(tuple(sources),"pm25","continuous",observations=obs,model_constraints={"nonnegative":True})
    quality=execute("quality_aware_real_pm25",context,"fusion.quality_aware")
    reports=quality.report["steps"][0]["report"]["sources"]
    train=obs[obs.split.eq("train")]
    corrected=[]
    for bundle,report in zip(sources,reports):
        field=bundle.dataset.pm25.interp(lat=geos.lat,lon=geos.lon)
        field=field*report["bias_correction"]["slope"]+report["bias_correction"]["intercept"]
        sampled=field.interp(lat=xr.DataArray(train.latitude.values,dims="station"),lon=xr.DataArray(train.longitude.values,dims="station"))
        sigma=float(np.sqrt(np.mean((sampled.values-train.value.values)**2)))
        ds=field.to_dataset(name="pm25")
        ds.attrs.update(crs="EPSG:4326",source_id=bundle.dataset.attrs["source_id"],data_kind="continuous",calibration="train-only empirical observation-equivalent concentration")
        ds.pm25.attrs.update(units="ug m-3",physical_variable="PM2.5",data_kind="continuous")
        ds["uncertainty"]=xr.full_like(ds.pm25,sigma)
        ds.uncertainty.attrs["units"]="ug m-3"
        corrected.append(EcoDataBundle(ds))
    execute("weighted_real_pm25",AlgorithmContext(tuple(corrected),"pm25","continuous",model_constraints={"nonnegative":True}),"fusion.uncertainty_weighted")
    assert np.isfinite(train[["value","latitude","longitude"]]).all().all() and (train.value>=0).all()
    frame=train.assign(time=time,variable="PM2.5",unit="ug m-3",qc_flag="VALID")
    observation_data=_frame_to_dataset(frame)
    observation_data["instrument_qa"]=("observation",np.repeat("UNKNOWN",len(frame)))
    observation_data.attrs["qc_scope"]="Basic numeric/domain screening, not instrument QA certification"
    observations=EcoDataBundle(observation_data)
    execute("residual_real_pm25",AlgorithmContext((corrected[0],observations),"pm25","continuous",observations=train,model_constraints={"nonnegative":True}),"fusion.residual")
    # Explicitly confirm the role guard; no source ranking without observations.
    missing=AlgorithmContext(tuple(sources),"pm25","continuous")
    assert not create_default_registry().get("fusion.quality_aware").compatibility(missing).eligible
    assert reports[0]["bias_correction"]["fit_split"]=="train"
    assert quality.report["steps"][0]["report"]["test_used_for_fitting_or_weighting"] is False
    (OUT/"fusion_execution_manifest.json").write_text(json.dumps({"time_UTC":str(time),"training_stations":int((obs.split=="train").sum()),
        "validation_stations":int((obs.split=="validation").sum()),"final_test_used":False,"caveats":caveats,"records":records},ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"outputs":len(records),"train":len(train),"test_used":False}))


if __name__=="__main__":main()
