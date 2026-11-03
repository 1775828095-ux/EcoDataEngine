"""Complete screening, native optimisers and complementary-source isolation."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.models import portrait_policy as policy
from ecodata.algorithms.raster_reconstruction_task import degradation
from test_composition_decision import fixture as small_fixture,submit as small_submit,stage
from test_joint_state_platform import stores


def prepare_case(root,*,start=0,synthetic=True):
    variable="air_temperature"
    if synthetic:
        lat=np.linspace(30,35,8);lon=np.linspace(110,115,8);t=np.arange(9)
        yy,xx=np.meshgrid(lat,lon,indexing="ij")
        values=280+np.sin(xx*2)[None,:,:]+np.cos(yy*2)[None,:,:]+t[:,None,None]*.4
        times=pd.date_range("2013-01-01",periods=9,freq="6h")
    else:
        with xr.open_dataset(root/"data/ncep_xarray/air_temperature.nc") as source:
            part=source.sortby("lat").isel(time=slice(start,start+9),lat=slice(6,18),lon=slice(15,27)).load()
        lat=part.lat.values;lon=part.lon.values;times=pd.DatetimeIndex(part.time.values);values=part.air.values.astype(float)
    fine=xr.Dataset({variable:(("time","lat","lon"),values)},coords=dict(time=times,lat=lat,lon=lon),attrs=dict(crs="EPSG:4326"))
    fine[variable].attrs=dict(units="K",data_kind="continuous",physical_variable=variable,spatial_semantics="cell_mean",temporal_semantics="instantaneous",physical_definition="Controlled area-observation reconstruction of gridded temperature samples")
    coarse=[]
    for time in times:
        ds,_=degradation(fine.sel(time=time,drop=True),len(lat)//2,variable)
        coarse.append(ds.expand_dims(time=[time]))
    low=xr.concat(coarse,dim="time");low.attrs["source_id"]="coarse_fast"
    sparse=fine.isel(time=[0,8]);sparse.attrs["source_id"]="fine_slow"
    truth=fine.isel(time=slice(1,8))[variable]
    req=dict(task="multisource",variable=variable,unit="K",semantic_type="continuous",nonnegative=True,
        spatial_semantics="cell_mean",temporal_semantics="instantaneous",target_lat=lat.tolist(),target_lon=lon.tolist(),target_times=[str(t) for t in times[1:8]],
        parameters=dict(target_time_frequency="6h"),model_profile=dict(id="wrf",role="temperature",field=variable),
        scoring=dict(registry_version="complementary-state-v1"),portrait_policy=dict(version="complementary-wrf-v1",scenario="coarse-and-detail"),
        composition=dict(reconstruction=dict(fine_source="fine_slow",coarse_source="coarse_fast",evidence="Explicit paired-resolution controlled observation design"),
            windows=[dict(id="all",start=str(times[0]),end=str(times[-1]))],
            source_stages={"fine_slow":[dict(id="time",options=[dict(algorithm="temporal.semantic_resample",parameters=dict(target_time_frequency="6h",temporal_interpolation=method)) for method in ("linear","nearest")],evidence="Historical interpolation alternatives")],
                "coarse_fast":[stage("grid",["raster_task.spatial.nearest","raster_task.spatial.bilinear","raster_task.pchip","raster_task.spatial.xarray_cubic"])]},
            fusion_options=[dict(algorithm="fusion.primary_weighted_mean",primary=primary,primary_share=share,evidence="Frozen dominant-source mean controls") for primary in ("fine_slow","coarse_fast") for share in (.6,.8)]+[
                dict(algorithm="fusion.spatiotemporal_detail",detail_strength=alpha,evidence="Observed coarse dynamics and supplied fine-anchor residual alternatives") for alpha in (0.,.5,1.)],
            post_stages=[],search=dict(survivors=2,population=4,generations=12,backend="alns")))
    inputs=[api.SearchInput("fine.nc","primary",sparse.to_netcdf()),api.SearchInput("coarse.nc","background",low.to_netcdf())]
    return inputs,req,truth


def publish():
    attrs=deepcopy(ar.load()["attributes"])
    for metric in ("coarse_error","detail_error"):
        attrs.append(dict(id="st_"+metric,definition="Visible complementary-source "+metric,tables=["data","algorithm","model"],unit="K",
            role="scoring",scoring_enabled=True,hard_allowed=False,provider="state_metric",property=metric,activation="constant_one",
            evidence="Declared observation operator and sparse-anchor nullspace residual; no hidden fine truth",history=["Complementary reconstruction task"]))
    api.publish_attribute_registry("complementary-state-v1",attrs,evidence="Complementary-source engineering measurements")
    issues=[dict(attribute="st_"+m,metric=m,acceptable=0.,bad=2.,importance=1.,evidence="Equal Kelvin-scaled coarse agreement and preservation of supplied fine-anchor detail; not measured WRF sensitivity") for m in ("coarse_error","detail_error")]
    api.publish_model_portrait("complementary-wrf-v1","wrf",[dict(id="coarse-and-detail",task="multisource",role="temperature",field="air_temperature",issues=issues,coupled_constraints=[],evidence="Retrospective controlled air-temperature reconstruction")],evidence="Explicit engineering preferences, not full WRF response")


def test_full_pool_scored_before_any_execution(tmp_path):
    datasets,req=small_fixture()
    req["composition"]["search"].update(screen_pool=1,exact_check=False)
    result=small_submit(datasets,req,tmp_path/"full")
    screened=json.loads((tmp_path/"full/initial_screening.json").read_text())
    assert screened["complete"] and not screened["sampling"]
    assert screened["scored_count"]+screened["rejected_count"]==72
    assert screened["scored_count"]==72
    assert result["screening"]["sampling"] is False


@pytest.mark.parametrize("backend",["basic","alns","pymoo","random"])
def test_native_backend_and_sparse_fine_truth_isolation(tmp_path,backend):
    publish();inputs,req,truth=prepare_case(tmp_path)
    req["composition"]["search"]["backend"]=backend
    result=api.submit_search(inputs,req,api.DatasetReference(truth,req["variable"]),tmp_path/backend,mode="Profile",budget=dict(objective_limit=12,proposal_limit=48,wall_seconds=120),seed=11)
    assert result["best"] is not None,[r.get("error") for r in result["rows"]]
    assert result["screening"]["scored"]==56
    assert result["native_optimizer"]["backend"]==backend
    assert result["counts"]["evaluated"]<=12 and not result["reference_used_for_selection"]
    if backend=="alns":assert sum(map(sum,result["native_optimizer"]["operator_counts"]))>0
    # Perturb only independent labels: selection and products must be identical.
    again=api.submit_search(inputs,req,api.DatasetReference(truth+1000,req["variable"]),tmp_path/(backend+"_different_truth"),mode="Profile",budget=dict(objective_limit=12,proposal_limit=48,wall_seconds=120),seed=11)
    assert result["recommended_key"]==again["recommended_key"]
    assert result["best"]["numeric_fingerprint"]==again["best"]["numeric_fingerprint"]


def test_detail_projection_preserves_actual_coarse_observation(tmp_path):
    from ecodata.algorithms.composition_decision import Composition
    publish();inputs,req,truth=prepare_case(tmp_path)
    parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
    choices=[]
    for axis,slot in zip(engine.axes,engine.slots):
        if slot[2]=="fusion":choices.append(next(i for i,o in enumerate(axis) if o["algorithm"]=="fusion.spatiotemporal_detail" and o["detail_strength"]==1))
        else:choices.append(0)
    out,matching,_,_=engine.execute(choices)
    metrics=engine.fields[0]["reconstruction"].metrics(out)
    assert metrics["coarse_error"]<1e-9
    assert metrics["detail_error"]>=0
