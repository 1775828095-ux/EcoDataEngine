"""Real common-entry execution and train-only measurement; synthetic mechanisms."""
from copy import deepcopy
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ecodata.algorithms import attribute_registry as ar
from ecodata.algorithms import platform_search as api
from test_profile_matching import fixture


def register(version="station-probes"):
    attrs=deepcopy(ar.load()["attributes"])
    for metric in ("RMSE","MAE","abs_bias"):
        attrs.append(dict(id="probe_"+metric, definition="Available-station spatial cross-validation "+metric,
            tables=["data","algorithm","model"],unit="bounded normalized utility",role="scoring",
            scoring_enabled=True,hard_allowed=False,provider="station_probe",property=metric,
            activation="constant_one",evidence="Three fixed station-group folds; no independent labels",
            history=["Measured empirical suitability, not a guarantee"]))
    attrs.append(dict(attrs[-1],id="runtime_tier",provider="declared_cost",property=None,
                      definition="Declared graph cost, not measured time"))
    api.publish_attribute_registry(version,attrs,evidence="Mechanism acceptance")


def case():
    ds,_,req,_=fixture()
    var=req["variable"]
    lat=np.linspace(30,32,9);lon=np.linspace(110,112,9)
    template=xr.Dataset({var:(("lat","lon"),np.zeros((9,9)))},
        coords=dict(lat=lat,lon=lon,time=np.datetime64("2020-01-01")),attrs=dict(crs="EPSG:4326",data_role="prediction_template"))
    template[var].attrs=dict(ds[var].attrs,spatial_semantics="point")
    a,b=np.meshgrid(lon,lat)
    cov=xr.Dataset({"background":(("lat","lon"),10+np.sin(a)+b),
                    "weather":(("lat","lon"),np.cos(2*a)+b)},coords=template.coords,attrs=dict(crs="EPSG:4326"))
    cov.background.attrs["units"]=req["unit"];cov.weather.attrs["units"]="K"
    x,y=np.meshgrid(np.linspace(110.2,111.8,4),np.linspace(30.2,31.8,4))
    obs=pd.DataFrame(dict(station_id=[str(i) for i in range(16)],longitude=x.ravel(),latitude=y.ravel(),
        value=10+2*np.sin(x.ravel())+3*y.ravel(),split="train",unit=req["unit"],
        variable=template[var].attrs["physical_variable"],time=pd.Timestamp("2020-01-01")))
    inputs=[api.SearchInput("template.nc","primary",template.to_netcdf()),
        api.SearchInput("covariates.nc","auxiliary",cov.to_netcdf()),
        api.SearchInput("available.nc","observations",xr.Dataset.from_dataframe(obs).to_netcdf())]
    req.update(task="multisource",allow_training=True,spatial_semantics="point",target_lat=lat.tolist(),target_lon=lon.tolist(),
        parameters=dict(observation_crs="EPSG:4326",include_station_only_baselines=True,
            gp_length_scale_m=60000.,gp_signal_std=10.,gp_noise_std=2.,
            feature_groups=dict(background=["background"],weather=["weather"],joint=["background","weather"])),
        model_requirements=dict(priorities={"probe_RMSE":.8,"runtime_tier":.2},hard=["nonnegative"],evidence="Explicit synthetic preparation requirements"),
        scoring=dict(registry_version="station-probes"),preferences=dict(weights={"RMSE":1},scales={"RMSE":1}),
        profile_options=dict(max_steps=4,candidate_limit=128,candidate_pool=dict(algorithms=[
            "fusion.idw","fusion.station_rf_joint","fusion.gp_residual","fusion.residual","raster_task.nonnegative"])))
    truth=xr.DataArray(10+2*np.sin(a)+3*b,dims=("lat","lon"),coords=template.coords,attrs=template[var].attrs)
    return inputs,req,truth


def test_common_entry_executes_forest_residual_and_measured_score_without_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(ar,"ROOT",tmp_path/"registries");register()
    inputs,req,truth=case()
    result=api.submit_search(inputs,req,None,tmp_path/"blind",mode="Profile",budget=dict(objective_limit=64,wall_seconds=120))
    assert result["counts"]["real_executions"]==0
    assert len(result["recommendations"])>=3
    assert any("fusion.residual" in row["steps"] for row in result["recommendations"])
    assert all(next(t for t in row["matching"]["terms"] if t["dimension"]=="probe_RMSE")["status"]=="TRAIN_STATION_PROBE" for row in result["recommendations"])
    evaluated=api.submit_search(inputs,req,api.DatasetReference(truth,req["variable"]),tmp_path/"evaluation",
        mode="Profile",budget=dict(objective_limit=64,wall_seconds=120))
    assert result["recommended_key"]==evaluated["recommended_key"]
    assert all(r["status"]=="PASS" for r in evaluated["rows"])
    assert all(r["file_readback"]=="PASS" for r in evaluated["rows"])
    assert len({r["matching"]["score_interval"][0] for r in evaluated["rows"]})>1
    probes=evaluated["portraits"]["station_probes"]
    assert all(not set(f["fit_stations"]) & set(f["held_stations"]) for p in probes.values() for f in p["folds"])
    changed=deepcopy(req)
    changed["model_requirements"]["priorities"]={"probe_RMSE":0,"runtime_tier":1}
    cost=api.submit_search(inputs,changed,None,tmp_path/"cost",mode="Profile")
    assert cost["recommendations"][0]["steps"][0]=="fusion.idw"
    assert cost["recommended_key"]!=result["recommended_key"]
    from ecodata.interface.search_page import process_search_upload
    page=process_search_upload(inputs,changed,None,tmp_path/"page",mode="Profile")
    assert page["recommended_key"]==cost["recommended_key"]
    # Change visible data, keeping the task/pool/preferences fixed.
    import io
    with xr.open_dataset(io.BytesIO(inputs[-1].content)) as source:
        constant=source.load()
    constant["value"][:]=25.
    altered=inputs[:-1]+[api.SearchInput("constant.nc","observations",constant.to_netcdf())]
    data_changed=api.submit_search(altered,req,None,tmp_path/"constant",mode="Profile",
        budget=dict(objective_limit=64,wall_seconds=120))
    assert data_changed["recommendations"][0]["steps"][0]=="fusion.idw"
    assert data_changed["recommended_key"]!=result["recommended_key"]
    assert data_changed["portraits"]["data"][0]["content_hash"]!=result["portraits"]["data"][0]["content_hash"]


def test_probe_hard_guarantees_remain_forbidden(tmp_path,monkeypatch):
    monkeypatch.setattr(ar,"ROOT",tmp_path/"registries");register()
    _,req,_=case();req["model_requirements"]["hard"]=["probe_RMSE"]
    with pytest.raises(ValueError,match="hard requirement"):
        api.submit_search(case()[0],req,None,tmp_path/"bad",mode="Profile")


def test_station_only_kriging_does_not_require_unused_covariate_rank():
    from ecodata.algorithms.open_source_stations import PyKrigePlugin
    inputs,req,_=case()
    parsed,_=api.parse_inputs(inputs)
    context,_,_=api.make_context(parsed,req)
    for field in context.auxiliaries[0].data_vars.values():field[:]=0
    assert PyKrigePlugin("ordinary").compatibility(context).eligible
    assert not PyKrigePlugin("universal").compatibility(context).eligible
