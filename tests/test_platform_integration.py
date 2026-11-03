"""Bounded public integration contracts; synthetic values are boundary tests only."""
from pathlib import Path
import inspect,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np,pandas as pd,pytest,xarray as xr
from ecodata.algorithms import platform_search as service
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.interface.search_page import process_search_upload


def grid(category=False):
    ds=xr.Dataset({"v":(("lat","lon"),np.arange(16).reshape(4,4)%3 if category else np.arange(16.).reshape(4,4))},coords={"lat":np.arange(4.)+30,"lon":np.arange(4.)+110},attrs={"crs":"EPSG:4326"})
    ds.v.attrs.update(units="1",data_kind="categorical" if category else "continuous",spatial_semantics="cell_category" if category else "cell_mean")
    return ds


def request(category=False):
    return dict(task="spatial",variable="v",unit="1",semantic_type="categorical" if category else "continuous",nonnegative=True,target_lat=[30.5,32.5],target_lon=[110.5,112.5])


def prepare(ds=None,req=None):
    ds=ds if ds is not None else grid();req=req or request()
    ctx,_,_=service.make_context([("primary",ds)],req)
    registry=service.platform_registry()
    decision=AlgorithmSelector(registry).decide(ctx,dict(operation="spatial_processing",grid_change=True,coarsening=True))
    factory,external,excluded=service.bind_registry(registry,ctx,decision,req)
    return registry,ctx,decision,factory,external,excluded


def test_registry_reachable_contracts_not_just_id_listing():
    registry,ctx,decision,factory,external,excluded=prepare()
    assert len(registry.specs())==91
    assert all(callable(registry.get(s.id).compatibility) and callable(registry.get(s.id).run) for s in registry.specs())
    bound={key[0] for key in factory.adapters}
    assert {"spatial.nearest","spatial.conservative","spatial.xarray_mean","raster_task.nonnegative"}<=bound
    assert "features.galileo_nano" in excluded
    assert "v5.emiproc_spatial" in excluded
    assert decision.matches["raster_task.nonnegative"]["binding"].startswith("postprocessor")


def test_actual_category_semantics_change_capabilities_and_candidates():
    _,_,a,fa,_,_=prepare()
    _,_,b,fb,_,_=prepare(grid(True),request(True))
    assert "categorical_aggregation" in b.requirements.required
    assert "continuous_spatial_resampling" in a.requirements.required
    assert ("spatial.bilinear","run") in fa.adapters and ("spatial.bilinear","run") not in fb.adapters
    assert ("spatial.xarray_most_common","run") in fb.adapters


@pytest.mark.parametrize("case",["unit","crs","unknown","total","interval"])
def test_rejects_before_numerical_execution(case):
    ds=grid();req=request()
    if case=="unit":req["unit"]="K"
    if case=="crs":ds.attrs["crs"]="not a CRS"
    if case=="unknown":req["semantic_type"]="UNKNOWN"
    if case=="total":req["semantic_type"]="cell_total"
    if case=="interval":req["temporal_semantics"]="accumulation"
    with pytest.raises(Exception):service.make_context([("primary",ds)],req)


def test_no_reference_or_unsupported_hard_constraint_is_explicit(tmp_path):
    with pytest.raises(ValueError,match="cannot rank"):service.submit_search([],request(),None,tmp_path/"none")
    req=dict(request(),constraints={"unimplemented_physics":True})
    with pytest.raises(ValueError,match="hard-constraint consumer"):service.submit_search([],req,service.EngineeringMetric("v"),tmp_path/"hard")
    assert (tmp_path/"hard/failure.json").is_file()
    req=dict(request(),constraints={"unit":"K"})
    with pytest.raises(ValueError,match="Conflicting task/constraint"):service.submit_search([],req,service.EngineeringMetric("v"),tmp_path/"conflict")


def test_feature_mode_and_resource_restrictions():
    registry,ctx,*_=prepare()
    for identifier,req in [("features.galileo_nano",dict(task="multisource",purpose="features")),("neural.fno",request()),("baseline.temporal_hour_median",dict(task="temporal")),("v5.emiproc_spatial",dict(task="model_preprocessing"))]:
        assert service.binding_reason(registry.get(identifier).spec,ctx,req)


def test_spatial_statistics_and_category_compositions_are_not_silent_resampling():
    from dataclasses import replace
    registry,ctx,*_=prepare()
    minimum=registry.get("spatial.xarray_min").spec
    assert service.binding_reason(minimum,ctx,request())
    explicit=replace(ctx,parameters={"spatial_statistic":"min"})
    assert service.binding_reason(minimum,explicit,dict(request(),operation="spatial_statistics")) is None
    output=registry.get(minimum.id).run(explicit).data.dataset
    assert output.v.attrs["spatial_semantics"]=="cell_min" and output.v.attrs["spatial_statistic"]=="min"
    fractions=registry.get("spatial.xarray_fractions").spec
    assert service.binding_reason(fractions,ctx,request(True))
    assert service.binding_reason(fractions,ctx,dict(request(True),output_representation="category_fractions")) is None
    facts=SemanticFeatureExtractor().extract(grid(),"v")
    from ecodata.algorithms.semantic_requirements import RequirementMapper
    assert RequirementMapper().map(facts,{"operation":"extreme_value_correction"}).required==("extreme_value_correction",)
    assert RequirementMapper().map(facts,{"operation":"spatial_statistics"}).required==("continuous_spatial_aggregation",)


def test_independent_reference_never_enters_decision_or_ports():
    registry,ctx,decision,factory,external,_=prepare()
    first=service.DatasetReference(grid().v,"v")
    second=service.DatasetReference(grid().v*999,"v")
    assert ctx.validation_reference is None and set(external)=={"source0"}
    assert first.protocol!=second.protocol
    with pytest.raises(ValueError,match="never final test"):service.DatasetReference(grid().v,"v",role="test")
    ds=grid();ds.v.values[0,0]=np.nan
    with pytest.raises(ValueError,match="fixed reference cohort"):first(ds)
    with pytest.raises(ValueError,match="coordinate queue"):first(grid().assign_coords(lat=grid().lat+1))


def test_page_client_is_the_same_service(monkeypatch,tmp_path):
    calls=[]
    import ecodata.interface.search_page as page
    monkeypatch.setattr(page,"submit_search",lambda *args,**kwargs:calls.append((args,kwargs)) or {"same":True})
    assert page.process_search_upload([],request(),None,tmp_path,seed=11)=={"same":True}
    assert len(calls)==1
    assert "semantic_proposal_weights" not in inspect.getsource(page)
    assert "from tools" not in inspect.getsource(service)


def test_toolkit_file_client_uses_shared_service(monkeypatch,tmp_path):
    data=tmp_path/"input.nc";grid().to_netcdf(data,engine="h5netcdf")
    calls=[]
    monkeypatch.setattr(service,"submit_search",lambda *a,**k:calls.append((a,k)) or {"same":True})
    result=service.submit_file_request(dict(inputs=[dict(path=str(data),role="primary")],request=request(),evaluation={"kind":"declared_task_metric","metric":"coverage"}),tmp_path/"out")
    assert result=={"same":True} and len(calls)==1
    assert calls[0][0][0][0].content==data.read_bytes()


def test_public_constraint_calls_existing_kernel_and_inherits_only_upstream(tmp_path):
    registry,ctx,decision,factory,external,_=prepare()
    from ecodata.algorithms.numerical_search import key_for
    recipes=factory.recipes("validated_field")
    chosen=next(recipe for _,recipe in recipes if recipe[0]=="quality.platform_constrained" and recipe[2][0][2][0]=="raster_task.nonnegative")
    from golem.core.optimisers.graph import OptGraph
    graph=OptGraph(factory.instantiate(chosen))
    evaluator=service.RegistryEvaluator(registry,factory,external,decision,service.EngineeringMetric("v"),tmp_path)
    value=evaluator.execute(graph,key_for(graph,factory))
    assert value["status"]=="PASS" and value["task_acceptance"]["acceptance_status"]=="PASS"
    assert any(n["operation"]=="raster_task.nonnegative" for n in value["node_trace"])
    assert value["file_readback"]=="PASS"


def test_half_hour_aggregation_is_not_hour_only_gap_split():
    ds=xr.Dataset({"v":("time",np.arange(8.)+1)},coords={"time":pd.date_range("2020-01-01 00:30",periods=8,freq="30min")})
    ds.v.attrs.update(units="m",data_kind="continuous",temporal_semantics="accumulation")
    ds["time_bounds"]=(("time","bounds"),np.column_stack([ds.time.values-np.timedelta64(30,"m"),ds.time.values]))
    ds.time.attrs["bounds"]="time_bounds"
    req=dict(task="temporal",variable="v",unit="m",semantic_type="continuous",temporal_semantics="accumulation",parameters={"target_time_frequency":"2h"})
    ctx,_,_=service.make_context([("primary",ds)],req)
    plugin=service.platform_registry().get("temporal.semantic_resample")
    assert plugin.compatibility(ctx).eligible
    output=plugin.run(ctx).data.dataset
    np.testing.assert_allclose(output.v.values,[10,26])
    facts=SemanticFeatureExtractor().extract(ds,"v")
    from ecodata.algorithms.semantic_requirements import RequirementMapper
    required=RequirementMapper().map(facts,dict(operation="temporal_processing",temporal_resolution="2h"))
    assert "temporal_accumulation" in required.required


def test_real_saved_runs_have_readback_and_real_branch_execution():
    root=ROOT/"output/platform_integration_20261021/runs"
    for task in ("spatial","temporal","multisource"):
        path=root/(task+"_toolkit_v2")/"summary.json"
        if not path.is_file():pytest.skip("Separately run fixed real acceptance first")
        result=json.loads(path.read_text(encoding="utf-8"))
        assert result["counts"]["real_executions"]>0
        rows=[r for r in result["rows"] if r["status"]=="PASS"]
        assert rows and all(r["file_readback"]=="PASS" for r in rows)
        if task=="multisource":
            assert any(any(n["operation"]=="fusion.residual" for n in r["graph"]) for r in rows)
            assert all(r["hidden_truth_passed_to_nodes"] is False for r in rows)


@pytest.mark.parametrize("unknown",[False,True])
def test_generic_complete_graph_cannot_borrow_or_assume_required_capability(unknown,tmp_path):
    registry,ctx,decision,factory,external,_=prepare()
    from golem.core.optimisers.graph import OptGraph
    from ecodata.algorithms.numerical_search import key_for
    recipe=next(r for _,r in factory.recipes("validated_field") if r[0]=="quality.platform" and r[2][0][2][0]=="spatial.nearest")
    graph=OptGraph(factory.instantiate(recipe))
    if unknown:decision.matches.pop("spatial.nearest")
    else:decision.matches["spatial.nearest"]["capabilities"]=()
    evaluator=service.RegistryEvaluator(registry,factory,external,decision,service.EngineeringMetric("v"),tmp_path)
    result=evaluator.execute(graph,key_for(graph,factory))
    assert result["status"]=="FAIL" and not result["numerical_execution_attempted"]
    assert result["task_acceptance"]["acceptance_status"]==("UNKNOWN" if unknown else "REJECT")


def test_real_curvilinear_operation_does_not_require_rectilinear_contract():
    folder=ROOT/"output/platform_integration_20261021/runs/curvilinear_contract_fixed"
    if not folder.is_dir():pytest.skip("Separate real DEM public acceptance not run")
    result=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    decision=json.loads((folder/"decision.json").read_text(encoding="utf-8"))
    assert decision["requirements"]["required"]==["curvilinear_regridding"]
    assert result["best"]["file_readback"]=="PASS"
    assert any(n["operation"].startswith("v5.pyresample") for n in result["best"]["graph"])


def test_new_page_preserves_result_on_rerender_without_search(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ecodata.interface.search_page as page
    folder=ROOT/"output/platform_integration_20261021/runs/spatial_page_delivery"
    if not folder.is_dir():pytest.skip("Real page-handler acceptance not run")
    monkeypatch.setattr(page,"process_search_upload",lambda *a,**k:pytest.fail("Rerender must not start search"))
    result=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    app=AppTest.from_string("from ecodata.interface.search_page import render_search_page\nrender_search_page()")
    app.session_state["unified_result"]=(result,str(folder));app.run(timeout=30)
    assert not app.exception and len(app.success)==1
    assert {button.key for button in app.get("download_button")} >= {"unified_download","model_portrait_template_download"}
    app.run(timeout=30)
    assert not app.exception and len(app.success)==1
