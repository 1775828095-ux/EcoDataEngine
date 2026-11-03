"""Public source/time composition: numerical outcomes, not just graph counts."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.composition_decision import Composition, chains
from ecodata.models import portrait_policy
from test_joint_state_platform import setup, scenario, stores


def stage(name, algorithms, after=()):
    return dict(id=name,after=list(after),options=[dict(algorithm=a) for a in algorithms],evidence="Declared finite algorithm alternatives; not an expected winning route")


def fixture(temporal=False):
    ds, req, ref = setup()
    req["task"] = "multisource"
    portrait_policy.publish("composition-policy","wrf",[
        scenario(req,"negative",{"negative_fraction":1}),
        scenario(req,"grid",{"grid_gap":1})],evidence="Explicit engineering priorities, not inferred WRF sensitivity")
    req["portrait_policy"] = dict(version="composition-policy",scenario="negative")
    variable=req["variable"]
    ds[variable].values = np.arange(16.).reshape(4,4)-12
    a=ds.copy(deep=True);b=ds.copy(deep=True);b[variable].values+=10
    a.attrs["source_id"]="A";b.attrs["source_id"]="B"
    if temporal:
        times=pd.date_range("2020-01-01",periods=4,freq="h")
        a=xr.concat([a.assign_coords(time=t).expand_dims("time") for t in times],dim="time")
        b=xr.concat([b.assign_coords(time=t).expand_dims("time") for t in times],dim="time")
        req["target_times"]=[str(t) for t in times]
        a[variable].attrs["temporal_semantics"]="instantaneous"
        b[variable].attrs["temporal_semantics"]="instantaneous"
    base=[stage("grid",["spatial.nearest","spatial.bilinear"]),stage("clip",[None,"raster_task.nonnegative"])]
    req["composition"]=dict(source_stages={"A":base,"B":deepcopy(base)},post_stages=[],
        fusion_options=[dict(algorithm="fusion.primary_weighted_mean",primary=k,primary_share=.75,evidence="Declared role candidates") for k in ("A","B")],
        search=dict(screen_pool=8,survivors=2,generations=6,population=2,exact_check=True))
    return [a,b],req


def submit(datasets,req,folder,**options):
    inputs=[api.SearchInput(str(i)+".nc","primary",ds.to_netcdf()) for i,ds in enumerate(datasets)]
    return api.submit_search(inputs,req,options.pop("evaluation",None),folder,mode="Profile",budget=dict(objective_limit=300,wall_seconds=180),**options)


def test_exact_counts_orders_roles_and_real_numeric_execution(tmp_path):
    datasets,req=fixture()
    parsed=[("primary",ds) for ds in datasets]
    engine=Composition(parsed,req,api.platform_registry())
    # Each source: 2 grid choices without clipping + 2*2 orders with clipping.
    assert engine.sizes==[6,6,2,1] and engine.count==72
    report=submit(datasets,req,tmp_path/"all")
    assert report["counts"]["exact_count_complete"]
    assert report["counts"]["exact_usable_count"]==72, [r.get("error") for r in report["rows"] if r["status"]=="FAIL"][:3]
    assert report["counts"]["failed_observed"]==0
    assert report["optimizer"]["backend"]=="basic"
    assert report["proposals"] and all(p["parent"]!=p["child"] for p in report["proposals"])
    assert report["best"]["matching"]["score_interval"][0]>0
    with xr.open_dataset(report["best"]["product"]) as actual:assert (actual[req["variable"]]>=0).all()
    # Primary role and same/different source algorithms are all real choices.
    numerics=[]
    for primary in ("A","B"):
        genome=[0,0,next(i for i,o in enumerate(engine.axes[2]) if o["primary"]==primary),0]
        out,_,trace,_=engine.execute(genome)
        numerics.append(out[req["variable"]].values)
        assert next(t for t in trace if "roles" in t)["roles"]["primary"]==primary
    assert not np.allclose(*numerics)
    assert any([s["algorithm"] for s in c]==["raster_task.nonnegative","spatial.bilinear"] for c in engine.axes[0])
    assert any([s["algorithm"] for s in c]==["spatial.bilinear","raster_task.nonnegative"] for c in engine.axes[0])


def test_window_switch_overlap_and_coordination_rejection(tmp_path):
    datasets,req=fixture(True)
    simple=[stage("grid",["spatial.nearest"])]
    req["composition"]["source_stages"]={"A":simple,"B":simple}
    req["composition"]["windows"]=[dict(id="early",start=req["target_times"][0],end=req["target_times"][2]),dict(id="late",start=req["target_times"][1],end=req["target_times"][3])]
    engine=Composition([("primary",ds) for ds in datasets],req,api.platform_registry())
    assert engine.count==4 # independent primary choices in each time window
    genome=[0]*len(engine.axes)
    for i,slot in enumerate(engine.slots):
        if slot[2]=="fusion":genome[i]=next(j for j,o in enumerate(engine.axes[i]) if o["primary"]==("A" if slot[1]=="early" else "B"))
    out,_,trace,checks=engine.execute(genome)
    assert len(out.time)==4
    assert {t["owner"][1] for t in trace}=={"early","late"}
    v=req["variable"]
    np.testing.assert_allclose(out[v].isel(time=1), (out[v].isel(time=0)+out[v].isel(time=3))/2)
    req["composition"]["coordination"]=dict(max_source_disagreement=.1,evidence="Deliberately strict engineering gate")
    bad=submit(datasets,req,tmp_path/"rejected")
    assert bad["best"] is None and bad["counts"]["exact_usable_count"]==0
    assert all("Coordination gate" in r["error"] for r in bad["rows"])


def test_public_page_file_entry_and_reference_is_post_selection_only(tmp_path):
    datasets,req=fixture()
    req["composition"]["source_stages"]={k:[stage("grid",["spatial.nearest"])] for k in ("A","B")}
    req["composition"]["search"]["exact_check"]=True
    from ecodata.interface.search_page import process_search_upload
    inputs=[api.SearchInput(str(i)+".nc","primary",ds.to_netcdf()) for i,ds in enumerate(datasets)]
    called=[]
    class Reference:
        protocol={"kind":"declared_task_metric"}
        def __call__(self,ds):called.append(True);return dict(objective=999)
    page=process_search_upload(inputs,req,Reference(),tmp_path/"page",mode="Profile",budget=dict(objective_limit=2,wall_seconds=60))
    assert called==[True] and not page["reference_used_for_selection"]
    paths=[]
    for i,ds in enumerate(datasets):
        p=tmp_path/(str(i)+".nc");ds.to_netcdf(p);paths.append(dict(path=str(p),role="primary"))
    file=api.submit_file_request(dict(inputs=paths,request=req,mode="Profile",budget=dict(objective_limit=2,wall_seconds=60)),tmp_path/"file")
    assert page["recommended_key"]==file["recommended_key"]


def test_cycles_optional_dedup_and_unsupported_rules_are_blocked(tmp_path):
    registry=api.platform_registry()
    assert chains([stage("a",[None]),stage("b",[None])],registry)==[[]]
    with pytest.raises(ValueError,match="cycle"):
        chains([stage("a",[None],["b"]),stage("b",[None],["a"])],registry)
    datasets,req=fixture(True)
    req["composition"]["windows"]=[dict(id="gap",start=req["target_times"][0],end=req["target_times"][1])]
    with pytest.raises(ValueError,match="uncovered"):Composition([("primary",ds) for ds in datasets],req,registry)


def test_topological_requirement_and_halo_are_enforced(tmp_path):
    datasets,req=fixture(True)
    chain=[stage("grid",["spatial.nearest"]),stage("clip",["raster_task.nonnegative"],["grid"])]
    assert len(chains(chain,api.platform_registry()))==1
    chain[0]["context_hours"]=2
    req["composition"]["source_stages"]={"A":chain,"B":chain}
    req["composition"]["windows"]=[dict(id="all",start=req["target_times"][0],end=req["target_times"][-1],halo_hours=1)]
    with pytest.raises(ValueError,match="halo"):Composition([("primary",ds) for ds in datasets],req,api.platform_registry())


def test_temporal_alignment_and_seam_gate_execute_real_kernels(tmp_path):
    datasets,req=fixture(True)
    v=req["variable"]
    # B supplies every second hour. Its real temporal interpolation stage must
    # provide the missing target timestamps, rather than relabelling records.
    datasets[1]=datasets[1].isel(time=[0,2,3])
    for ds in datasets:ds[v].attrs["temporal_semantics"]="instantaneous"
    req["composition"]["source_stages"]={
        "A":[stage("grid",["spatial.bilinear"])],
        "B":[dict(id="time",after=[],options=[dict(algorithm="temporal.semantic_resample",parameters=dict(target_time_frequency="1h"))],evidence="Declared historical hourly interpolation"),stage("grid",["spatial.bilinear"],["time"])]}
    req["composition"]["windows"]=[dict(id="early",start=req["target_times"][0],end=req["target_times"][1],halo_hours=2),dict(id="late",start=req["target_times"][2],end=req["target_times"][3],halo_hours=2)]
    result=submit(datasets,req,tmp_path/"time")
    assert result["counts"]["exact_usable_count"]==4, [r.get("error") for r in result["rows"]]
    assert any(t.get("algorithm")=="temporal.semantic_resample" for t in result["best"]["trace"])
    req["composition"]["coordination"]=dict(max_seam_jump=0,evidence="Strict continuity test, not universal physical threshold")
    failed=submit(datasets,req,tmp_path/"seam")
    assert failed["counts"]["failed_observed"]>0


def test_joint_fields_share_decision_and_unit_checks(tmp_path):
    datasets,req=fixture(True)
    v=req["variable"]
    from ecodata.algorithms.profile_matching import model_table
    role=next(r for r in model_table("wrf")["roles"] if r["role"]=="humidity")
    field=role["fields"][0]; other=field["name"]
    other_request=deepcopy(req);other_request.pop("composition")
    other_request.update(variable=other,unit=field["unit"],model_profile=dict(id="wrf",role="humidity",field=other))
    for ds in datasets:
        ds[other]=ds[v]*0+50
        ds[other].attrs=dict(units=field["unit"],physical_variable=field.get("standard_name",other),spatial_semantics="cell_mean",temporal_semantics="instantaneous")
    portrait_policy.publish("humidity-composition","wrf",[scenario(other_request,"grid",{"grid_gap":1})],evidence="Second field contract")
    other_request["portrait_policy"]=dict(version="humidity-composition",scenario="grid")
    main=deepcopy(req);main.pop("composition")
    simple={k:[stage("grid",["spatial.nearest"])] for k in ("A","B")}
    req["composition"]["fields"]=[dict(request=main,importance=1,source_stages=simple),dict(request=other_request,importance=2,source_stages=simple)]
    req["composition"]["windows"]=[dict(id="all",start=req["target_times"][0],end=req["target_times"][-1])]
    result=submit(datasets,req,tmp_path/"joint")
    assert result["counts"]["exact_usable_count"]==4
    with xr.open_dataset(result["best"]["product"]) as ds:
        assert {v,other}<=set(ds.data_vars)
        assert ds[v].dims==ds[other].dims
        assert ds[other].attrs["units"]==field["unit"]
    broken=deepcopy(datasets);broken[1][other].attrs["units"]="wrong"
    with pytest.raises(ValueError,match="unit"):submit(broken,req,tmp_path/"units")


def test_interval_support_and_uncertainty_are_not_fabricated(tmp_path):
    datasets,req=fixture(True)
    v=req["variable"]
    simple={k:[stage("grid",["spatial.nearest"])] for k in ("A","B")}
    req["composition"]["source_stages"]=simple
    req["composition"]["windows"]=[dict(id="all",start=req["target_times"][0],end=req["target_times"][-1])]
    for ds in datasets:
        ds[v].attrs["temporal_semantics"]="accumulation"
        times=ds.time.values.astype("datetime64[ns]")
        ds["time_bounds"] = (("time","bounds"),np.column_stack((times-np.timedelta64(1,"h"),times)))
    result=submit(datasets,req,tmp_path/"interval")
    assert result["counts"]["exact_usable_count"]==2
    with xr.open_dataset(result["best"]["product"]) as ds:
        np.testing.assert_array_equal(ds.time_bounds,datasets[0].time_bounds)
    # Different intervals at identical timestamp labels must not be fused.
    datasets[1]["time_bounds"].values[:,0]-=np.timedelta64(1,"h")
    bad=submit(datasets,req,tmp_path/"wrong_interval")
    assert bad["counts"]["exact_usable_count"]==0
    # Coefficient weights are not an uncertainty model.
    req["composition"]["fusion_options"]=[dict(algorithm="fusion.uncertainty_weighted",evidence="No uncertainty supplied",independence_confirmed=True)]
    with pytest.raises(ValueError,match="supplied uncertainty"):submit(datasets,req,tmp_path/"sigma")


def test_early_and_late_join_are_actual_different_topologies(tmp_path):
    datasets,req=fixture()
    req["composition"]["source_stages"]={k:[stage("grid",[None,"spatial.bilinear"])] for k in ("A","B")}
    req["composition"]["post_stages"]=[stage("grid",[None,"spatial.bilinear"])]
    req["composition"]["fusion_options"]=[dict(algorithm="fusion.primary_weighted_mean",primary="A",primary_share=.75,join_grid=join,evidence="Compare actual scale-before-join and join-before-scale graphs") for join in ("target","native_common")]
    result=submit(datasets,req,tmp_path/"join")
    assert result["counts"]["grammar_combinations"]==16
    assert result["counts"]["failed_observed"]>0 # unmatched join supports remain invalid
    valid=[r for r in result["rows"] if r["status"]=="PASS"]
    assert any(not any(p["choice"] for p in r["plan"] if p["branch"] in {"A","B"}) and any(p["choice"] for p in r["plan"] if p["branch"]=="post") for r in valid)
    assert any(all(p["choice"] for p in r["plan"] if p["branch"] in {"A","B"}) and not any(p["choice"] for p in r["plan"] if p["branch"]=="post") for r in valid)


def test_cross_variable_relation_changes_joint_role_selection(tmp_path):
    datasets,req=fixture()
    original=req["variable"]
    # An explicit synthetic decomposition, not a claimed WRF physical law.
    values={"part_a":(2.,6.),"part_b":(3.,9.),"total":(5.,15.)}
    for name,pair in values.items():
        for ds,value in zip(datasets,pair):
            ds[name]=xr.full_like(ds[original],value)
            ds[name].attrs=dict(ds[original].attrs)
    s=scenario(req,"joint",{"grid_gap":1})
    s["coupled_constraints"]=[dict(kind="sum",variables=["part_a","part_b"],target="total",atol=1e-9,evidence="Explicit synthetic decomposition identity for engineering test")]
    portrait_policy.publish("joint-relation","wrf",[s],evidence="Synthetic cross-field coordination test")
    fields=[]
    for name in values:
        child=deepcopy(req);child.pop("composition");child["variable"]=name
        child["portrait_policy"]=dict(version="joint-relation",scenario="joint")
        fields.append(dict(request=child,importance=1,source_stages={k:[stage("grid",["spatial.nearest"])] for k in ("A","B")}))
    req["composition"]["fields"]=fields
    result=submit(datasets,req,tmp_path/"coupled")
    assert result["counts"]["grammar_combinations"]==8
    assert result["counts"]["exact_usable_count"]==2
    assert result["counts"]["failed_observed"]==6
    with xr.open_dataset(result["best"]["product"]) as ds:
        xr.testing.assert_allclose(ds.part_a+ds.part_b,ds.total)


def test_observation_resources_drive_optional_per_source_calibration(tmp_path):
    # Use the pre-existing collaboration fixture and real affine kernel.
    from test_collaboration import case
    ds,other,obs,req,inputs,reference=case([],training=True)
    # Register the same visible-state vocabulary, without changing this case's
    # already reviewed source/observation contracts.
    _,_=fixture()
    portrait_policy.publish("observed-composition","wrf",[scenario(req,"grid",{"grid_gap":1})],evidence="Explicit observed-field engineering task")
    req["scoring"]=dict(registry_version="state-v1")
    req["portrait_policy"]=dict(version="observed-composition",scenario="grid")
    req["composition"]=dict(source_stages={k:[stage("calibrate",[None,"bias.station_affine"])] for k in ("first","second")},post_stages=[],
        fusion_options=[dict(algorithm="fusion.primary_weighted_mean",primary=k,primary_share=.75,evidence="Declared role alternatives with train-only station calibration") for k in ("first","second")],
        search=dict(screen_pool=8,survivors=2,generations=3,population=2,exact_check=True))
    result=api.submit_search(inputs,req,None,tmp_path/"observed",mode="Profile",budget=dict(objective_limit=8,wall_seconds=60))
    assert result["counts"]["exact_usable_count"]==8,[r.get("error") for r in result["rows"]]
    assert any(any(t.get("algorithm")=="bias.station_affine" for t in r["trace"]) for r in result["rows"])
    # Test labels may not enter any of the fitting stages.
    bad=obs.copy();bad["split"]="test"
    blocked=inputs[:2]+[api.SearchInput("hidden.nc","observations",xr.Dataset.from_dataframe(bad.rename_axis("row")).to_netcdf())]
    with pytest.raises(ValueError,match="test labels"):
        api.submit_search(blocked,req,None,tmp_path/"leak",mode="Profile",budget=dict(objective_limit=8,wall_seconds=60))


def test_empty_composition_builds_candidates_from_actual_profiles(tmp_path):
    datasets,req=fixture()
    req["composition"]={}
    engine=Composition([("primary",ds) for ds in datasets],req,api.platform_registry())
    assert engine.count>72
    result=submit(datasets,req,tmp_path/"automatic")
    assert result["best"] is not None
    assert result["counts"]["grammar_combinations"]==engine.count
    assert result["counts"]["exact_usable_count"] is None
    algorithms={step["algorithm"] for axis,slot in zip(engine.axes,engine.slots) if slot[2]!="fusion" for chain in axis for step in chain}
    assert {"spatial.bilinear","raster_task.pchip","spatial.xarray_cubic"}<=algorithms
    small=[ds.isel(lat=slice(0,3),lon=slice(0,3)) for ds in datasets]
    req["target_lat"]=req["target_lat"][:2];req["target_lon"]=req["target_lon"][:2]
    smaller=Composition([("primary",ds) for ds in small],req,api.platform_registry())
    methods={step["algorithm"] for axis,slot in zip(smaller.axes,smaller.slots) if slot[2]!="fusion" for chain in axis for step in chain}
    assert "spatial.xarray_cubic" not in methods
    assert smaller.count<engine.count
