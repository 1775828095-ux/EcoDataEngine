"""Public numerical cooperation checks; fixtures are not scientific gains."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms import platform_search as service
from ecodata.algorithms.dag import Artifact, DAGError
from ecodata.algorithms.station_calibration import StationAffineCalibrationPlugin
from test_profile_matching import fixture


def case(pool, *, training=False, targets=None, max_steps=3):
    profile, _, request, _ = fixture()
    variable = request["variable"]
    lat, lon = np.linspace(30,32,5), np.linspace(110,112,5)
    values = 10 + 2*(lat[:,None]-30) + 3*(lon[None,:]-110)
    ds = xr.Dataset({variable: (("lat","lon"), values), "uncertainty": (("lat","lon"), np.ones((5,5)))},
                    coords=dict(lat=lat,lon=lon,time=np.datetime64("2020-01-01")),
                    attrs=dict(crs="EPSG:4326",source_id="first"))
    ds[variable].attrs = dict(profile[variable].attrs, physical_definition="synthetic common field",spatial_semantics="point")
    ds.uncertainty.attrs["units"] = request["unit"]
    second = ds.copy(deep=True); second.attrs["source_id"]="second"
    second[variable].values += 1
    obs = pd.DataFrame(dict(longitude=[110.2,111.8,110.2,111.8,111.],latitude=[30.2,30.2,31.8,31.8,31.],
        split="train",unit=request["unit"],variable=ds[variable].attrs["physical_variable"],time=pd.Timestamp("2020-01-01")))
    obs["value"] = 2*(10+2*(obs.latitude-30)+3*(obs.longitude-110))+3
    table = xr.Dataset.from_dataframe(obs.rename_axis("row"))
    request.update(task="multisource",spatial_semantics="point",allow_training=training,
        target_lat=list(lat if targets is None else targets[0]),target_lon=list(lon if targets is None else targets[1]),
        parameters=dict(observation_crs="EPSG:4326",gp_length_scale_m=30000.,gp_signal_std=5.,gp_noise_std=1.),
        profile_options=dict(candidate_pool=dict(algorithms=pool),max_steps=max_steps,candidate_limit=128))
    inputs=[service.SearchInput("a.nc","primary",ds.to_netcdf()),service.SearchInput("b.nc","background",second.to_netcdf()),
            service.SearchInput("stations.nc","observations",table.to_netcdf())]
    reference=ds[variable].interp(lat=request["target_lat"],lon=request["target_lon"])
    return ds,second,obs,request,inputs,reference


def execute(pool,tmp_path,**kwargs):
    ds,other,obs,request,inputs,reference=case(pool,**kwargs)
    result=service.submit_search(inputs,request,service.DatasetReference(reference,request["variable"]),tmp_path,
        mode="Profile",budget=dict(objective_limit=128,wall_seconds=120))
    assert result["rows"]
    return result,ds,obs,request


def test_per_source_optional_calibration_then_fusion(tmp_path):
    result,ds,obs,request=execute(["bias.station_affine","fusion.uncertainty_weighted"],tmp_path,training=True)
    rows=result["rows"]
    assert len(rows)==4 and all(r["status"]=="PASS" and r["file_readback"]=="PASS" for r in rows)
    assert sorted(r["steps"].count("bias.station_affine") for r in rows)==[0,1,1,2]
    assert len({r["prediction_fingerprint"] for r in rows})>=3
    plugin=StationAffineCalibrationPlugin()
    context,_,_=service.make_context([("primary",ds)],dict(request,task="multisource"))
    calibrated=plugin.run(replace(context,observations=obs))
    np.testing.assert_allclose(calibrated.data.dataset[request["variable"]],2*ds[request["variable"]]+3,atol=1e-10)
    assert calibrated.report["slope"]==pytest.approx(2.)
    # Input uncertainty is transformed, never advertised as total calibrated uncertainty.
    np.testing.assert_allclose(calibrated.data.dataset.uncertainty,2.)


def test_regrid_each_source_then_fusion(tmp_path):
    result,*_=execute(["spatial.bilinear","spatial.nearest","fusion.uncertainty_weighted"],tmp_path,
                     targets=([30.5,31.,31.5],[110.5,111.,111.5]))
    assert all(r["status"]=="PASS" and r["file_readback"]=="PASS" for r in result["rows"])
    assert any(sum(op.startswith("prepare.grid.") for op in [n["operation"] for n in r["graph"]])==2 for r in result["rows"])


@pytest.mark.parametrize("estimator",["fusion.gp_residual","station.pykrige_ordinary"])
def test_fusion_then_replaceable_residual_then_addition(tmp_path,estimator):
    result,_,_,request=execute(["fusion.uncertainty_weighted",estimator,"fusion.residual"],tmp_path)
    rows=[r for r in result["rows"] if any(n["parameters"].get("mode")=="add_background" for n in r["graph"])]
    assert rows and all(r["status"]=="PASS" and r["file_readback"]=="PASS" for r in rows)
    assert any(r["steps"]==["fusion.uncertainty_weighted"] for r in result["rows"])
    for row in rows:
        assert any(n["operation"]==estimator and n["parameters"].get("mode")=="residual_correction" for n in row["graph"])
        with xr.open_dataset(row["product"]) as ds:
            np.testing.assert_allclose(ds[request["variable"]], ds.background_value+ds.residual_correction,atol=1e-10)


def test_alignment_calibration_and_fusion_share_real_branch_outputs(tmp_path):
    axes=(np.linspace(30,32,7),np.linspace(110,112,7))
    result,*_=execute(["spatial.bilinear","bias.station_affine","fusion.uncertainty_weighted"],tmp_path,
                     training=True,targets=axes,max_steps=5)
    rows=[r for r in result["rows"] if "bias.station_affine" in r["steps"]]
    assert rows and all(r["status"]=="PASS" and r["file_readback"]=="PASS" for r in rows)
    assert any(len(r["steps"])==5 for r in rows)


def test_linear_residual_with_explicit_complete_station_support(tmp_path):
    result,*_=execute(["spatial.bilinear","fusion.uncertainty_weighted","station.metpy_linear","fusion.residual"],tmp_path,
                     targets=([30.2,31.8],[110.2,111.8]),max_steps=5)
    corrected=[r for r in result["rows"] if any(n["parameters"].get("mode")=="add_background" for n in r["graph"])]
    assert corrected and all(r["status"]=="PASS" and r["file_readback"]=="PASS" for r in corrected)


def test_reference_free_cooperation_and_hard_model_filter(tmp_path,monkeypatch):
    ds,_,_,request,inputs,_=case(["bias.station_affine","fusion.uncertainty_weighted"],training=True)
    monkeypatch.setattr(StationAffineCalibrationPlugin,"run",lambda *a:pytest.fail("No fit during recommendation"))
    result=service.submit_search(inputs,request,None,tmp_path/"recommend",mode="Profile")
    assert result["counts"]["real_executions"]==0 and len(result["recommendations"])==4
    # A frozen pool removes every unrequested scientific stage, retaining only structural bypasses.
    request["profile_options"]["candidate_pool"]["algorithms"]=["fusion.uncertainty_weighted","raster_task.nonnegative"]
    request["model_requirements"]=dict(hard=["nonnegative"],priorities={},evidence="explicit functional acceptance requirement")
    filtered=service.submit_search(inputs,request,None,tmp_path/"hard",mode="Profile")
    assert filtered["recommendations"]
    assert all("raster_task.nonnegative" in r["steps"] and "bias.station_affine" not in r["steps"] for r in filtered["recommendations"])


def test_fusion_then_nonnegative_projection(tmp_path):
    result,*_=execute(["fusion.uncertainty_weighted","raster_task.nonnegative"],tmp_path)
    assert any("raster_task.nonnegative" in r["steps"] and r["status"]=="PASS" for r in result["rows"])


def test_calibration_not_offered_without_training_authorisation(tmp_path):
    result,*_=execute(["fusion.uncertainty_weighted"],tmp_path)
    assert all("bias.station_affine" not in r["steps"] for r in result["rows"])


def test_page_handler_executes_same_cooperation_as_toolkit(tmp_path):
    from ecodata.interface.search_page import process_search_upload
    ds,other,obs,request,inputs,reference=case(["bias.station_affine","fusion.uncertainty_weighted"],training=True)
    evaluation=service.DatasetReference(reference,request["variable"])
    toolkit=service.submit_search(inputs,request,evaluation,tmp_path/"toolkit",mode="Profile")
    page=process_search_upload(inputs,request,evaluation,tmp_path/"page",mode="Profile")
    assert page["best"]["file_readback"]=="PASS"
    assert page["best"]["prediction_fingerprint"]==toolkit["best"]["prediction_fingerprint"]
    assert {tuple(r["steps"]) for r in page["rows"]}=={tuple(r["steps"]) for r in toolkit["rows"]}


def test_owner_filter_cannot_hide_frozen_enumeration_cap(tmp_path):
    ds,other,obs,request,inputs,reference=case(["fusion.gp_residual","fusion.residual","fusion.uncertainty_weighted"])
    request["profile_options"]["candidate_limit"]=2
    with pytest.raises(ValueError,match="truncated"):
        service.submit_search(inputs,request,None,tmp_path,mode="Profile")


def test_structural_and_runtime_background_owner_checks(tmp_path):
    from ecodata.algorithms.router import AlgorithmSelector
    from golem.core.optimisers.graph import OptGraph
    ds,other,obs,request,inputs,ref=case(["fusion.gp_residual","fusion.residual","fusion.uncertainty_weighted"])
    request["_local_problem_mode"]=True
    context,_,_=service.make_context([("primary",ds),("background",other),("observations",xr.Dataset.from_dataframe(obs.rename_axis("row")))],request)
    registry=service.platform_registry()
    decision=AlgorithmSelector(registry).decide(context,dict(operation="fusion",multi_source=True))
    factory,external,_=service.bind_registry(registry,context,decision,request)
    factory.local_recipe_seed=11;factory.reverse_limit=128
    graphs=[OptGraph(factory.instantiate(recipe)) for _,recipe in factory.recipes("platform_corrected_product")]
    assert graphs
    graph=graphs[0]
    factory.inspect(graph)
    addition=next(n for n in graph.nodes if n.parameters.get("mode")=="add_background")
    bindings=addition.parameters["_port_bindings"]
    correction=addition.nodes_from[bindings["correction"][0]]
    # An independently computed background is illegal, even with identical schema.
    import copy
    addition.nodes_from[bindings["field"][0]]=copy.deepcopy(addition.nodes_from[bindings["field"][0]])
    with pytest.raises(DAGError,match="same background"):
        factory.inspect(graph)
    from ecodata.algorithms.contracts import AlgorithmResult
    from ecodata.core.schemas import EcoDataBundle
    b=AlgorithmResult(EcoDataBundle(ds),"fusion.uncertainty_weighted",{})
    delta=Artifact("platform_residual_correction",b,dict(contract=service.VERSION,background_fingerprint="wrong"))
    with pytest.raises(DAGError,match="another background"):
        factory.adapters["fusion.residual","add_background"].execute(dict(field=Artifact("platform_raw_product",b,dict(contract=service.VERSION)),correction=delta),{}, {})
