"""Behavioural checks of the production decision boundary and real DAG ports."""
from dataclasses import replace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
sys.path.append(str(Path(__file__).resolve().parents[1]/".venv-golem/Lib/site-packages"))
import numpy as np
import xarray as xr
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.algorithms.golem_search import semantic_proposal_weights
from ecodata.algorithms.mixed_gap_dag import setup,TASK
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from test_beijing_mixed_gap import fixture
from run_beijing_mixed_gap import fixed_graphs


def decision(external,**task):
    registry,factory=setup();batch=external["batch"].value;case=batch.cases[0]
    ds=xr.Dataset({"value":("time",case.values)},coords={"time":case.time})
    semantics=SemanticFeatureExtractor().extract_series(batch.cases,variable="value",unit="ug m-3",nonnegative=True)
    context=AlgorithmContext((EcoDataBundle(ds),),"value","continuous")
    target=dict(operation="temporal_missing_recovery",require_complete_coverage=True,prefer_observation_support=True)
    target.update(task)
    return registry,factory,AlgorithmSelector(registry).decide(context,target,semantics=semantics)


def test_support_changes_real_requirements_not_only_logs():
    _,external=fixture();_,f,d=decision(external)
    assert "boundary_temporal_recovery" in d.requirements.required
    case=external["batch"].value.cases[0]
    from ecodata.algorithms.mixed_gap_dag import Batch,SeriesCase,SealedArtifact,readonly,missing_runs
    values=case.values.copy();values[:3]=1;values[-3:]=1
    missing=np.flatnonzero(~np.isfinite(values))
    new=SeriesCase(case.station,case.time,readonly(values),readonly(missing))
    batch=Batch((new,),readonly(missing_runs(values)[missing]),((0,len(missing)),))
    changed=dict(external,batch=SealedArtifact("masked_batch",batch,external["batch"].metadata))
    _,f2,d2=decision(changed)
    assert "boundary_temporal_recovery" not in d2.requirements.required
    assert d.assess_graph(f,fixed_graphs(f)["linear"])["hard_reject"]
    assert not d2.assess_graph(f2,fixed_graphs(f2)["linear"])["hard_reject"]


def test_probabilities_consume_support_and_explicit_soft_switch():
    _,external=fixture();_,f,d=decision(external);graphs=fixed_graphs(f)
    candidates=[graphs["nearest"],graphs["hour_median"]]
    _,on=semantic_proposal_weights(d,f,f.initial(),candidates)
    _,off=semantic_proposal_weights(d,f,f.initial(),candidates,soft=False)
    assert on["probabilities"][0]>on["probabilities"][1]
    np.testing.assert_allclose(off["probabilities"],[.5,.5])
    _,_,unrelated=decision(external,description="untrusted filename H0 expected winner")
    _,again=semantic_proposal_weights(unrelated,f,f.initial(),candidates)
    np.testing.assert_array_equal(on["probabilities"],again["probabilities"])


def test_past_only_changes_match_without_accuracy_labels():
    _,external=fixture();_,_,d=decision(external,time_access="past_only")
    assert "past_only" in d.requirements.hard_constraints
    assert not d.matches["baseline.temporal_linear"]["eligible"]
    assert not d.matches["baseline.temporal_nearest"]["eligible"]
    assert d.matches["baseline.temporal_ffill"]["eligible"]


def test_state_on_other_branch_cannot_satisfy_this_port():
    _,external=fixture();_,f,d=decision(external);graph=fixed_graphs(f)["H0"]
    historical=next(n for n in graph.nodes if n.name=="baseline.temporal_hour_median" and n.parameters["mode"]=="long")
    historical.nodes_from=[p for p in historical.nodes_from if p.name!="source.fitted_history"]
    from golem.core.optimisers.graph import OptGraph
    graph=OptGraph([graph.root_node]);a=d.assess_graph(f,graph)
    specific=next(p for p in a["ports"] if p["node"]==historical.name and p["mode"]=="long")
    assert "temporal_missing_recovery" not in specific["capabilities"]
    assert a["port_gap"]==1


def test_split_masks_do_not_take_whole_graph_capability_union():
    _,external=fixture();_,f,d=decision(external);graph=fixed_graphs(f)["H0"]
    long=next(n for n in graph.nodes if n.parameters.get("mode")=="long")
    long.content["name"]="baseline.temporal_linear"
    long.nodes_from=[n for n in long.nodes_from if n.name=="source.masked_batch"]
    from golem.core.optimisers.graph import OptGraph
    graph=OptGraph([graph.root_node]);a=d.assess_graph(f,graph)
    # Both boundary runs have length 3, so the still-valid short fallback
    # handles them. Internal long runs can use the bilateral method.
    assert not a["hard_reject"]
    split=next(n for n in graph.nodes if n.name=="prepare.visible_gap_split")
    split.parameters["threshold"]=1
    assert d.assess_graph(f,graph)["hard_reject"]


def test_real_two_paths_and_quality_agree_with_output_support():
    _,external=fixture();registry,f,d=decision(external);pipeline=UnifiedAlgorithmPipeline(registry)
    graphs=fixed_graphs(f)
    outputs=[]
    for name in ["nearest","H0"]:
        assert not d.assess_graph(f,graphs[name])["hard_reject"]
        out=pipeline.run_dag(graphs[name],task_id=TASK,external=external,version=TASK)
        outputs.append(out["outputs"]["field"].value[1])
    assert not np.array_equal(*outputs)


def test_missing_contract_is_unknown_not_hard_illegal():
    _,external=fixture();_,f,d=decision(external);graph=fixed_graphs(f)["nearest"]
    f.adapter(graph.root_node).decision_contract={}
    result=d.assess_graph(f,graph)
    assert result["unknown"] and not result["hard_reject"]


def test_hidden_labels_are_absent_and_api_shared():
    import inspect
    from ecodata.interface.raster_upload import process_raster_upload
    from run_beijing_mixed_gap import Evaluator
    assert "AlgorithmSelector(registry).decide" in inspect.getsource(process_raster_upload)
    assert "AlgorithmSelector(self.pipeline.registry).decide" in inspect.getsource(Evaluator.decision)
    _,external=fixture();_,_,d=decision(external)
    text=str(d.as_dict())
    for forbidden in ["H0","2017","expected_winner","private_targets","group_id"]:assert forbidden not in text
    assert d.semantics.temporal.status=="UNKNOWN"
    assert d.semantics.spatial.status=="UNKNOWN"


def test_datetime_storage_unit_does_not_change_hourly_features():
    _,external=fixture();cases=external["batch"].value.cases
    first=cases[0];second=replace(first,time=first.time.astype("datetime64[us]"))
    extractor=SemanticFeatureExtractor()
    a=extractor.extract_series((first,),variable="x")
    b=extractor.extract_series((first,second),variable="x")
    assert a.temporal.value["resolution"]==b.temporal.value["resolution"]=="0 days 01:00:00"
    assert b.observed.value["missing_count"]==2*a.observed.value["missing_count"]


def test_soft_support_uses_position_frequency_not_unique_case_count():
    _,external=fixture();_,f,d=decision(external);graph=fixed_graphs(f)["nearest"]
    rows=d.semantics.observed.value["support_conditions"]
    expected=sum(r["count"] for r in rows if not (r["left"] and r["right"]))/sum(r["count"] for r in rows)
    assert d.assess_graph(f,graph)["support_deficit"]==expected
    heavier=[dict(r,count=r["count"]*10 if r["left"] and r["right"] else r["count"]) for r in rows]
    from ecodata.algorithms.semantic_requirements import EvidenceValue
    semantics=replace(d.semantics,observed=EvidenceValue(dict(d.semantics.observed.value,support_conditions=heavier),("test visible-position frequencies",)))
    alternate=replace(d,semantics=semantics)
    assert alternate.assess_graph(f,graph)["support_deficit"]<expected
    assert not alternate.assess_graph(f,graph)["hard_reject"]
    assert alternate.requirements==d.requirements
