"""Three-portrait matching + actual stages + official evolutionary search."""
from copy import deepcopy
import json
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.algorithms import portrait_memory as memory
from ecodata.models import portrait_policy as policy
from test_profile_matching import fixture


@pytest.fixture(autouse=True)
def stores(tmp_path,monkeypatch):
    monkeypatch.setattr(ar,"ROOT",tmp_path/"attributes")
    monkeypatch.setattr(policy,"ROOT",tmp_path/"models")
    monkeypatch.setattr(memory,"ROOT",tmp_path/"memories")


def setup(model="wrf"):
    ds,_,req,reference=fixture(model)
    attrs=deepcopy(ar.load()["attributes"])
    for metric in ("missing_fraction","negative_fraction","range_excess","peak_loss","roughness","grid_gap"):
        attrs.append(dict(id="state_"+metric,definition="Measured field state: "+metric,tables=["data","algorithm","model"],
            unit="fraction or explicitly scoped proxy",role="scoring",scoring_enabled=True,hard_allowed=False,
            provider="state_metric",property=metric,activation="constant_one",evidence="Registered visible-field measurement; not accuracy guarantee",
            history=["Acceptance measurement definition"]))
    api.publish_attribute_registry("state-v1",attrs,evidence="Functional state mechanism")
    req["scoring"]={"registry_version":"state-v1"}
    req["profile_options"]=dict(max_steps=2,candidate_limit=64,
        candidate_pool=dict(algorithms=["spatial.nearest","spatial.bilinear","raster_task.nonnegative"]))
    req["preferences"]=dict(weights={"RMSE":1},scales={"RMSE":1})
    return ds,req,reference


def scenario(req,name,weights):
    return dict(id=name,task=req["task"],role=req["model_profile"]["role"],field=req["model_profile"]["field"],
        issues=[dict(attribute="state_"+metric,metric=metric,acceptable=0.,bad=1.,importance=value,
            evidence="Explicit synthetic task priority; not WRF scientific sensitivity") for metric,value in weights.items()],
        coupled_constraints=[],evidence="Synthetic state matching task")


def submit(ds,req,folder,*,reference=None):
    return api.submit_search([api.SearchInput("input.nc","primary",ds.to_netcdf())],req,
        api.DatasetReference(reference,req["variable"]) if reference is not None else None,folder,mode="Profile",
        budget=dict(objective_limit=64,wall_seconds=120))


def test_model_scenario_changes_real_selection_and_input_changes_state(tmp_path):
    ds,req,ref=setup();ds[req["variable"]].values-=12
    api.publish_model_portrait("model-v1","wrf",[
        scenario(req,"negative",{"negative_fraction":1}),scenario(req,"grid",{"grid_gap":1})],evidence="Declared preparation objectives")
    req["portrait_policy"]={"version":"model-v1","scenario":"negative"}
    cold=submit(ds,req,tmp_path/"cold")
    assert cold["scoring_policy"]=="joint_state_v1"
    assert cold["recommendations"][0]["steps"][-1]=="raster_task.nonnegative"
    first=submit(ds,req,tmp_path/"first",reference=ref)
    assert first["recommended_key"]==cold["recommended_key"]
    assert first["best"]["status"]=="PASS"
    assert any("joint_transition" in a["metadata"] for n in first["best"]["node_trace"] for a in n.get("outputs",{}).values())
    req["portrait_policy"]["scenario"]="grid"
    second=submit(ds,req,tmp_path/"grid")
    assert second["recommendations"][0]["steps"][-1]!="raster_task.nonnegative"
    assert second["recommended_key"]!=cold["recommended_key"]
    positive=ds.copy(deep=True);positive[req["variable"]].values+=20
    req["portrait_policy"]["scenario"]="negative"
    changed=submit(positive,req,tmp_path/"positive")
    assert changed["recommendations"][0]["steps"][-1]!="raster_task.nonnegative"
    assert first["portraits"]["model"]["sensitivity"] is None
    from ecodata.interface.search_page import process_search_upload
    inputs=[api.SearchInput("input.nc","primary",ds.to_netcdf())]
    page=process_search_upload(inputs,req,None,tmp_path/"page",mode="Profile")
    source=tmp_path/"input.nc";ds.to_netcdf(source)
    file=api.submit_file_request(dict(inputs=[dict(path=str(source),role="primary")],request=req,mode="Profile"),tmp_path/"file")
    assert page["recommended_key"]==file["recommended_key"]==cold["recommended_key"]


def test_unknown_side_effect_not_filled_and_finite_pool_evolution_is_auditable(tmp_path):
    ds,req,_=setup()
    api.publish_model_portrait("model-v1","wrf",[scenario(req,"structure",{"roughness":.5,"peak_loss":.5})],evidence="Declared task")
    req["portrait_policy"]={"version":"model-v1","scenario":"structure"}
    req["profile_options"]["search"]=dict(strategy="genetic",population=4,generations=3,proposal_limit=50,exact_check=True)
    result=submit(ds,req,tmp_path/"genetic")
    evolution=result["state_evolution"]
    assert result["evolution_used"] and evolution["optimizer_error"] is None
    assert evolution["proposals"] and evolution["visited"]
    assert any(p["status"]=="ACCEPTED" and p["parent"]!=p["child"] for p in evolution["proposals"])
    assert evolution["certification"]=="EXHAUSTIVE_SCORE_LOWER_BOUND_IN_FROZEN_DOMAIN"
    assert evolution["true_effect_global_optimum"]=="NOT_PROVEN"
    assert result["recommendation_certainty"]=="UNRESOLVED_REQUIRES_PROBE"
    assert all(r["matching"]["score"] is None for r in result["recommendations"])
    assert result["counts"]["real_executions"]==0


def test_development_memory_learns_state_responses_and_rejects_test_records(tmp_path):
    ds,req,ref=setup()
    api.publish_model_portrait("model-v1","wrf",[scenario(req,"structure",{"roughness":1})],evidence="Declared task")
    req["portrait_policy"]={"version":"model-v1","scenario":"structure"}
    folders=[]
    for group,offset in [("a",0),("b",1),("held",2)]:
        source=ds.copy(deep=True);source[req["variable"]].values+=offset
        request=deepcopy(req);request["profile_options"].update(evidence_role="development",group=group)
        folder=tmp_path/group
        report=submit(source,request,folder,reference=ref)
        assert report["stop_reason"]=="CANDIDATE_POOL_COMPLETED"
        folders.append(folder)
    bank=api.learn_portrait_memory(folders,"memory-v1",holdout_groups=["held"],features=["std"],min_groups=2,max_distance=1.)
    assert bank["records"] and all(r["group"]!="held" for r in bank["records"])
    assert bank["transitions"] and all(r["group"]!="held" for r in bank["transitions"])
    req["portrait_policy"]["memory_version"]="memory-v1"
    changed=submit(ds,req,tmp_path/"learned")
    assert all(r["matching"]["score"] is not None for r in changed["recommendations"])
    assert changed["portraits"]["state_memory_status"]["loaded"]
    from ecodata.algorithms.portrait_rules import parameterized_terms
    for item in changed["recommendations"]:
        _,lo,hi=parameterized_terms(item["matching"]["terms"],{})
        assert np.allclose([lo,hi],item["matching"]["score_interval"])
    assert any(isinstance(t["state"]["evidence"],dict) and t["state"]["evidence"]["kind"]=="CONDITIONAL_DEVELOPMENT_RESPONSE"
               for r in changed["recommendations"] for t in r["matching"]["transfer_trace"])
    with pytest.raises(ValueError,match="development"):
        api.learn_portrait_memory([tmp_path/"learned"],"bad",holdout_groups=["held"],features=["std"])


def test_source_inspection_and_coupled_constraints_are_not_invented_preferences(tmp_path):
    source=tmp_path/"model.f90";source.write_text('call nf90_get_att(ncid,varid,"units",units)\n! prefer accurate winds\n',encoding="utf-8")
    clues=api.inspect_model_portrait_sources([source])
    assert len(clues["records"])==1 and not clues["records"][0]["affects_selection"] and clues["sensitivity"] is None
    registry=tmp_path/"Registry.EM_COMMON"
    registry.write_text('state real T ijk dyn 1 - i0rh "T" "temperature" "K"\n',encoding="utf-8")
    decoded=api.inspect_model_portrait_sources([registry])
    assert decoded["structured_facts"][0]["variable"]=="T" and decoded["structured_facts"][0]["unit"]=="K"
    assert decoded["hard_requirements_promoted"]==0
    a=xr.Dataset({"a":("x",[1.,2.]),"b":("x",[2.,3.]),"total":("x",[3.,5.])})
    for v in a.data_vars.values():v.attrs["units"]="ug/m3"
    p=dict(scenario=dict(coupled_constraints=[dict(kind="sum",variables=["a","b"],target="total",atol=1e-8,evidence="Explicit example relation")]))
    assert policy.validate_coupled(a,p)["status"]=="PASS"
    a["total"][:]=0
    assert policy.validate_coupled(a,p)["status"]=="FAIL"


def test_metric_comparability_and_source_renaming_do_not_create_evidence():
    from ecodata.algorithms.portrait_state import measure,scientific_identity
    from ecodata.algorithms.contracts import AlgorithmContext
    from ecodata.core.schemas import EcoDataBundle
    lat=np.linspace(0,1,5);lon=np.linspace(0,1,5)
    ds=xr.Dataset({"v":(("lat","lon"),lat[:,None]+2*lon[None,:])},coords=dict(lat=lat,lon=lon))
    ctx=AlgorithmContext((EcoDataBundle(ds),),"v","continuous")
    high=ds.interp(lat=np.linspace(0,1,11),lon=np.linspace(0,1,11))
    assert np.isclose(measure(ds,ctx)["metrics"]["roughness"],measure(high,ctx)["metrics"]["roughness"])
    cropped=high.sel(lat=slice(.2,.8),lon=slice(.2,.8))
    assert measure(cropped,ctx)["metrics"]["peak_loss"] is None
    renamed=ds.copy();renamed.attrs.update(source_file="renamed.nc",source_sha256="transport-only")
    other=AlgorithmContext((EcoDataBundle(renamed),),"v","continuous")
    assert scientific_identity(ctx)==scientific_identity(other)
