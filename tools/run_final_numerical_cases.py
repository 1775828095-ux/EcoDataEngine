"""Reuse existing algorithms with persistent outputs and independent assertions."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import ResultQualityGate
from ecodata.evaluation.metrics import relative_conservation_error

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/capability_final_acceptance"


def main():
    records=[]
    def save(code,name,ds,report):
        ds.to_netcdf(OUT/(name+".nc"),engine="h5netcdf")
        with xr.open_dataset(OUT/(name+".nc"),engine="h5netcdf") as read:
            assert read.sizes==ds.sizes
        records.append({"capabilities":code,"output":name+".nc","report":report})
        (OUT/"numerical_execution_manifest.json").write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    def execute(code,name,context,algorithm):
        result=UnifiedAlgorithmPipeline(create_default_registry()).run(context,AlgorithmPlan(name,(algorithm,),0.,("Fixed acceptance case, no algorithm search",)))
        save(code,name,result.data.dataset,result.report)
        return result
    era=xr.load_dataset(ROOT/"output/algorithm_library_expansion/v4/era5_bth_t2m_reference.nc",engine="h5netcdf")
    d=era.isel(time=0,lat=slice(0,4),lon=slice(0,4),drop=True).rename(t2m="value")
    d.attrs.update(crs="EPSG:4326",spatial_semantics="cell_mean",data_kind="continuous",temporal_semantics="instantaneous")
    d.value.attrs.update(spatial_semantics="cell_mean",data_kind="continuous",temporal_semantics="instantaneous")
    target_lat=tuple(d.lat.coarsen(lat=2).mean().values);target_lon=tuple(d.lon.coarsen(lon=2).mean().values)
    result=execute(["C4","C17"],"regular_conservative_real_era5",AlgorithmContext((EcoDataBundle(d),),"value","continuous",target_lat=target_lat,target_lon=target_lon),"spatial.conservative")
    assert relative_conservation_error(d.value,result.data.dataset.value)<1e-10
    times=era.t2m.isel(time=slice(0,4),lat=0,lon=0,drop=True).to_dataset(name="value")
    times.value.attrs.update(temporal_semantics="instantaneous",data_kind="continuous")
    times.attrs.update(temporal_semantics="instantaneous",data_kind="continuous")
    result=execute(["C9"],"instantaneous_time_real_era5",AlgorithmContext((EcoDataBundle(times),),"value","continuous",parameters={"target_time_frequency":"30min"}),"temporal.semantic_resample")
    assert result.data.dataset.time.min()>=times.time.min() and result.data.dataset.time.max()<=times.time.max()
    np.testing.assert_allclose(result.data.dataset.value.values[1],np.mean(times.value.values[:2]),atol=1e-6)
    edges=pd.date_range("2019-08-01",periods=5,freq="h").values
    interval=xr.Dataset({"value":("time",[10.,20.,30.,40.]),"time_bounds":(("time","bounds"),np.column_stack([edges[:-1],edges[1:]]))},coords={"time":edges[1:]},attrs={"temporal_semantics":"mean","data_kind":"continuous","data_origin":"controlled rate-integration test"})
    interval.value.attrs.update(units="kg s-1",temporal_semantics="mean",data_kind="continuous")
    result=execute(["C11","C17"],"interval_rate_control",AlgorithmContext((EcoDataBundle(interval),),"value","continuous",parameters={"target_time_frequency":"4h"}),"temporal.semantic_resample")
    np.testing.assert_allclose(result.data.dataset.value,[25.])
    assert np.isclose(float(result.data.dataset.value.sum())*14400,float(interval.value.sum())*3600)
    bad=d.copy(deep=True);bad.value.values.flat[0]=-1
    gate=ResultQualityGate().validate(d,bad,variable="value",semantic_type="continuous",nonnegative=True)
    assert gate["status"]=="BLOCK"
    (OUT/"physical_violation_expected_block.json").write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding="utf-8")
    # Existing QP case is a labelled controlled fixture, not independent truth.
    from ecodata.interface.workflow import TrialRequest,run_grid_trial
    source=xr.load_dataset(ROOT/"examples/datasets/p3_pm25_coarse_10km_48h.nc",engine="h5netcdf").isel(time=slice(8,9))
    train=pd.read_csv(ROOT/"examples/datasets/p3_pm25_train_stations.csv",parse_dates=["time"])
    train=train[train.time.eq(pd.Timestamp(source.time.values[0]))]
    result=run_grid_trial(source,TrialRequest(target_spatial_resolution_km=2.5,spatial_method="constrained",variable="value",temporal_semantics="instantaneous",unit="ug m-3",target_model="custom"),training_observations=train)
    np.testing.assert_allclose(result.output.value.coarsen(lat=4,lon=4).mean(),source.value,atol=1e-5)
    save(["C18"],"qp_reaggregation_control",result.output,{"coarse_consistency_max_error":float(abs(result.output.value.coarsen(lat=4,lon=4).mean()-source.value).max()),"data_origin":"existing controlled fixture"})
    # Actual Galileo weights and actual USGS terrain; feature output only.
    with rasterio.open(ROOT/"examples/engineering_case_air_quality/terrain/dem.tif") as dem:
        elevation=dem.read(1).reshape(8,16,8,16).mean((1,3))
        pixel=abs(dem.transform.a)*16
    gy,gx=np.gradient(elevation,pixel,pixel)
    slope=np.rad2deg(np.arctan(np.sqrt(gx*gx+gy*gy)))
    terrain=xr.Dataset({"elevation":(("y","x"),elevation.astype("float32")),"slope":(("y","x"),slope.astype("float32"))},coords={"x":np.arange(8)*pixel,"y":np.arange(8)*pixel},attrs={"crs":"EPSG:3857","source":"USGS DEM derived elevation/slope; SRTM model input group; not asserted sensor equivalence"})
    terrain.elevation.attrs["units"]="m";terrain.slope.attrs["units"]="degree"
    context=AlgorithmContext((EcoDataBundle(terrain),),"galileo_embedding","continuous",parameters={"patch_size":2,"input_resolution_m":pixel,"month":8})
    first=execute(["C23"],"galileo_real_terrain",context,"features.galileo_nano")
    perturbed=terrain.copy(deep=True);perturbed.elevation.values[:]+=100
    second=create_default_registry().get("features.galileo_nano").run(replace(context,sources=(EcoDataBundle(perturbed),)))
    delta=float(abs(first.data.dataset.galileo_embedding-second.data.dataset.galileo_embedding).max())
    assert delta>0
    (OUT/"auxiliary_input_sensitivity.json").write_text(json.dumps({"feature_change_max":delta,"test_DEM_offset_m":100,"counterfactual_only":True,"physical_prediction_claimed":False},indent=2),encoding="utf-8")
    print(json.dumps([{r['output']:r['capabilities']} for r in records]))


if __name__=="__main__":main()
