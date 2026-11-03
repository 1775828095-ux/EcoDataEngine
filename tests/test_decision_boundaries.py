"""Bounded contract checks; no search, numerical enumeration or test labels."""
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tools"))
sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
import numpy as np
import pandas as pd
import pytest
from ecodata.algorithms.dag import DAGError
from ecodata.algorithms.mixed_gap_dag import SeriesCase,Batch,SealedArtifact,readonly,missing_runs,TASK
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from run_beijing_mixed_gap import restore
from test_beijing_mixed_gap import fixture
from test_unified_decision import decision

HISTORY=ROOT/"output/beijing_mixed_gap_20261020/unified_platform_v4"


def graph(name):
    # Read three saved graphs, not enumerate the grammar to find them.
    for path in ["mechanism_acceptance/results.json","development/results.json"]:
        for row in json.loads((HISTORY/path).read_text(encoding="utf-8")):
            if row["name"]==name:return restore(row.get("result",row)["graph"])
    raise AssertionError(name)


def require(d,name):
    return replace(d,requirements=replace(d.requirements,required=d.requirements.required+(name,)))


def test_complete_missing_required_capability_rejects_before_execution():
    _,ext=fixture();registry,f,d=decision(ext);d=require(d,"required_but_not_provided")
    a=d.assess_graph(f,graph("nearest"))
    assert a["hard_reject"] and a["acceptance_status"]=="REJECT"
    assert "required_but_not_provided" in a["missing_capabilities"]
    for adapter in f.adapters.values():adapter.execute=lambda *args:pytest.fail("Must not numerically execute rejected graph")
    with pytest.raises(DAGError,match="Task requirements REJECT"):
        UnifiedAlgorithmPipeline(registry).run_dag(graph("nearest"),task_id=TASK,external=ext,version=TASK,decision=d,factory=f)


def test_partial_capability_gap_remains_evolvable_but_not_finally_accepted():
    _,ext=fixture();_,f,d=decision(ext);d=require(d,"required_but_not_provided")
    initial=f.initial();a=d.assess_graph(f,initial)
    assert a["port_gap"] and not a["hard_reject"] and a["acceptance_status"]=="INCOMPLETE"
    with pytest.raises(DAGError,match="INCOMPLETE"):d.require_graph(f,initial)


def test_unknown_contract_is_not_missing_or_satisfied_and_cannot_finally_pass():
    _,ext=fixture();registry,f,d=decision(ext);g=graph("nearest")
    f.adapter(g.root_node).decision_contract={}
    a=d.assess_graph(f,g)
    assert not a["hard_reject"] and not a["missing_capabilities"]
    assert a["unknown_capabilities"] and not a["output_capabilities"] and a["acceptance_status"]=="UNKNOWN"
    with pytest.raises(DAGError,match="Task requirements UNKNOWN"):
        UnifiedAlgorithmPipeline(registry).run_dag(g,task_id=TASK,external=ext,version=TASK,decision=d,factory=f)


def test_absent_matching_evidence_is_unknown_not_definite_inapplicability():
    _,ext=fixture();_,f,d=decision(ext);d.matches.pop("baseline.temporal_nearest")
    a=d.assess_graph(f,graph("nearest"))
    assert not a["hard_reject"] and a["acceptance_status"]=="UNKNOWN"


def test_capability_cannot_be_borrowed_from_unused_fallback():
    _,ext=fixture();_,f,d=decision(ext)
    # Unit-contract marker only: available on backup, absent from the primary.
    name="backup_only_capability";d=require(d,name)
    d.matches["baseline.temporal_nearest"]["capabilities"]+=(name,)
    a=d.assess_graph(f,graph("linear_nearest"))
    assert a["hard_reject"] and name in a["missing_capabilities"]


def internal_only(ext):
    case=ext["batch"].value.cases[0];v=case.values.copy();v[:3]=1;v[-3:]=1
    missing=np.flatnonzero(~np.isfinite(v));case=replace(case,values=readonly(v),missing=readonly(missing))
    batch=Batch((case,),readonly(missing_runs(v)[missing]),((0,len(missing)),))
    return dict(ext,batch=SealedArtifact("masked_batch",batch,ext["batch"].metadata))


def test_unused_fallback_need_not_provide_active_primary_capability():
    _,ext=fixture();ext=internal_only(ext);_,f,d=decision(ext)
    name="primary_capability";d=require(d,name);d.matches["baseline.temporal_linear"]["capabilities"]+=(name,)
    assert d.require_graph(f,graph("linear_nearest"))["acceptance_status"]=="PASS"


def test_unused_select_branch_does_not_remove_required_capability_or_add_unknown():
    _,ext=fixture();_,f,d=decision(ext);g=graph("H0")
    split=next(n for n in g.nodes if n.name=="prepare.visible_gap_split");split.parameters["threshold"]=48
    long=next(n for n in g.nodes if n.parameters.get("mode")=="long")
    f.adapter(long).decision_contract={} # All visible gaps are <=7, so this branch is unused.
    assert d.require_graph(f,g)["acceptance_status"]=="PASS"
    split.parameters["threshold"]=1
    a=d.assess_graph(f,g)
    assert not a["hard_reject"] and a["acceptance_status"]=="UNKNOWN"


def test_soft_objective_and_unknown_soft_evidence_do_not_reject():
    _,ext=fixture();_,f,d=decision(ext)
    d=replace(d,requirements=replace(d.requirements,soft_objectives=("unimplemented_soft_goal",)))
    d.matches["baseline.temporal_nearest"]["output_contract"].pop("observation_evidence")
    a=d.require_graph(f,graph("nearest"))
    assert a["acceptance_status"]=="PASS" and a["support_deficit"] is None


def with_axis(ext,times):
    batch=ext["batch"].value;case=replace(batch.cases[0],time=readonly(times))
    return dict(ext,batch=SealedArtifact("masked_batch",replace(batch,cases=(case,)),ext["batch"].metadata))


@pytest.mark.parametrize("kind",["30min","irregular"])
def test_nonhourly_split_rejected_by_decision_and_actual_control(kind):
    _,ext=fixture();v=ext["batch"].value.cases[0].values
    times=pd.date_range("2015-01-01",periods=len(v),freq="30min" if kind=="30min" else "h").to_numpy(copy=True)
    if kind=="irregular":times[80:]+=np.timedelta64(1,"h")
    ext=with_axis(ext,times);registry,f,d=decision(ext);g=graph("H0")
    a=d.assess_graph(f,g)
    assert a["hard_reject"] and a["incompatible"]
    # Even a legacy caller without a decision cannot bypass the splitter's contract.
    with pytest.raises(DAGError,match="requires regular 1-hour"):
        UnifiedAlgorithmPipeline(registry).run_dag(g,task_id=TASK,external=ext,version=TASK)
    row=d.semantics.observed.value["support_conditions"][0]
    assert row["length_unit"]=="records" and row["gap_record_count"]==row["length"]
    if kind=="30min":assert row["duration_seconds"]==row["gap_record_count"]*1800


@pytest.mark.parametrize("dtype",["datetime64[us]","datetime64[ns]"])
def test_hourly_threshold_boundary_and_storage_unit(dtype):
    _,ext=fixture();ext=with_axis(ext,ext["batch"].value.cases[0].time.astype(dtype));registry,f,d=decision(ext)
    g=graph("H0");result=UnifiedAlgorithmPipeline(registry).run_dag(g,task_id=TASK,external=ext,version=TASK,decision=d,factory=f)
    split_nodes=[row for row in result["trace"] if row["operation"]=="prepare.visible_gap_split"]
    assert split_nodes[0]["outputs"]["field"]["metadata"]["threshold_unit"]=="hours"
    p=result["outputs"]["field"].value[1];missing=ext["batch"].value.cases[0].missing
    np.testing.assert_array_equal(p[(missing>=40)&(missing<46)],missing[(missing>=40)&(missing<46)])
    assert (p[(missing>=80)&(missing<87)]==50).all() # 6 -> short; 7 -> long.
    assert d.semantics.observed.value["regular_hourly"]


@pytest.mark.parametrize("failure",["duplicate","decreasing","missing"])
def test_invalid_time_labels_rejected_by_both_extractor_and_splitter(failure):
    _,ext=fixture();times=ext["batch"].value.cases[0].time.copy()
    if failure=="duplicate":times[4]=times[3]
    elif failure=="decreasing":times[[3,4]]=times[[4,3]]
    else:times[3]=np.datetime64("NaT","ns")
    ext=with_axis(ext,times);case=ext["batch"].value.cases[0]
    with pytest.raises(ValueError):SemanticFeatureExtractor().extract_series((case,),variable="x")
    from ecodata.algorithms.mixed_gap_dag import setup
    registry,_=setup()
    with pytest.raises(DAGError):UnifiedAlgorithmPipeline(registry).run_dag(graph("H0"),task_id=TASK,external=ext,version=TASK)


def test_final_acceptance_arguments_cannot_be_half_supplied():
    _,ext=fixture();registry,f,d=decision(ext)
    with pytest.raises(DAGError,match="both decision and typed factory"):
        UnifiedAlgorithmPipeline(registry).run_dag(graph("nearest"),task_id=TASK,external=ext,version=TASK,decision=d)


def test_beijing_execution_cannot_disable_final_acceptance_via_search_switches():
    # Real public execution method, not a search run. No hidden targets are used
    # because rejection must precede prediction and metric calculation.
    from run_beijing_mixed_gap import Evaluator
    _,ext=fixture();registry,f,d=decision(ext);d=require(d,"mandatory_missing")
    e=Evaluator.__new__(Evaluator);e.pipeline=UnifiedAlgorithmPipeline(registry);e.factory=f;e.external=ext
    e.y=np.empty(len(ext["query"].value));e.node_hits=e.node_new=e.kernel_hits=0
    e.decision=lambda:d
    result=e.execute(graph("nearest"),"unit-contract-rejection",persist=False)
    assert result["status"]=="FAIL" and result["objective"] is None
    assert result["failure_stage"]=="task_requirements" and not result["numerical_execution_attempted"]
    assert result["task_acceptance"]["acceptance_status"]=="REJECT" and e.node_new==0


def test_unknown_cadence_does_not_become_a_proven_failure_or_success():
    _,ext=fixture();_,f,d=decision(ext)
    metadata=dict(d.semantics.observed.value);metadata.pop("regular_hourly")
    d=replace(d,semantics=replace(d.semantics,observed=replace(d.semantics.observed,value=metadata)))
    a=d.assess_graph(f,graph("H0"))
    assert not a["hard_reject"] and a["acceptance_status"]=="UNKNOWN"


def test_legacy_pipeline_without_task_requirements_does_not_claim_them_verified():
    _,ext=fixture();registry,_,_=decision(ext)
    result=UnifiedAlgorithmPipeline(registry).run_dag(graph("nearest"),task_id=TASK,external=ext,version=TASK)
    assert result["task_acceptance"]["acceptance_status"]=="NOT_REQUESTED"
