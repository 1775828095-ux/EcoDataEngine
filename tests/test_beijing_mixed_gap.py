from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tools"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
import numpy as np,pandas as pd,pytest
from ecodata.algorithms.mixed_gap_dag import (SeriesCase,Batch,SealedArtifact,readonly,missing_runs,setup,TASK,predict)
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.genetic.operators import base_mutations
from run_beijing_mixed_gap import fixed_graphs,key_for,conditions


def fixture():
    original=np.arange(200,dtype=float);v=original.copy();v[:3]=np.nan;v[40:46]=np.nan;v[80:87]=np.nan;v[-3:]=np.nan
    missing=np.flatnonzero(~np.isfinite(v));time=pd.date_range("2016-01-01",periods=200,freq="h").to_numpy()
    case=SeriesCase("s",readonly(time),readonly(v),readonly(missing));batch=Batch((case,),readonly(missing_runs(v)[missing]),((0,len(missing)),))
    meta=dict(unit="ug m-3",contract=TASK);states={"s":dict(medians=readonly(np.ones(24)*50),training_end="2014-12-31 23:00")}
    external={"batch":SealedArtifact("masked_batch",batch,meta),"history":SealedArtifact("fitted_history",states,meta),"query":SealedArtifact("query_positions",readonly(np.arange(len(missing))),meta)}
    return original,external


def test_visible_gap_lengths_boundary_and_natural_indistinguishable():
    original,external=fixture();batch=external["batch"].value
    assert set(batch.lengths)=={3,6,7}
    assert batch.lengths[3]==6 and batch.lengths[9]==7
    with pytest.raises(ValueError):batch.cases[0].values[10]=999


def test_h0_is_actual_split_fallback_graph_and_protects_visible():
    original,external=fixture();registry,factory=setup();graph=fixed_graphs(factory)["H0"]
    out=UnifiedAlgorithmPipeline(registry).run_dag(graph,task_id=TASK,external=external,version=TASK)
    raw,p=out["outputs"]["field"].value
    missing=external["batch"].value.cases[0].missing
    assert np.all(p[:3]==50) and np.all(p[-3:]==50)
    np.testing.assert_allclose(p[(missing>=40)&(missing<46)],original[missing[(missing>=40)&(missing<46)]])
    assert np.all(p[(missing>=80)&(missing<87)]==50)
    assert out["outputs"]["field"].metadata["original_visible_protected"]
    fallback=[x for x in out["trace"] if x["operation"]=="prepare.conditional_fallback"][0]
    assert fallback["outputs"]["field"]["metadata"]["backup_used"]==6


def test_training_hour_median_does_not_fit_current_context():
    _,external=fixture();batch=external["batch"].value;states=external["history"].value
    assert np.all(predict(batch,"hour_median",states)==50)
    assert predict(batch,"nearest")[0]==3.
    assert "truth" not in external and "target_labels" not in batch.__dict__


def test_source_and_training_state_change_cache_identity():
    _,external=fixture();a=external["history"];meta=dict(a.metadata)
    b=SealedArtifact(a.kind,{"s":dict(medians=readonly(np.zeros(24)))},meta)
    assert a.fingerprint()!=b.fingerprint()
    assert SealedArtifact("query_positions",readonly([1]),meta).fingerprint()!=SealedArtifact("query_positions",readonly([2]),meta).fingerprint()


def test_two_structures_generated_from_common_grammar_and_real_mutation():
    _,external=fixture();registry,factory=setup();graphs=fixed_graphs(factory)
    assert len(factory.recipes("validated_field"))==949
    initial=factory.initial();assert factory.inspect(initial)["gap"]==1
    import random
    random.seed(11)
    completed=base_mutations.add_separate_parent_node(initial,factory)
    assert factory.inspect(completed,complete=True)["gap"]==0
    for name in ["nearest","H0"]:
        out=UnifiedAlgorithmPipeline(registry).run_dag(graphs[name],task_id=TASK,external=external,version=TASK)
        assert np.isfinite(out["outputs"]["field"].value[1]).all()
    assert key_for(graphs["nearest"],factory)!=key_for(graphs["H0"],factory)


def test_quality_failure_does_not_drop_uncovered_targets():
    _,external=fixture();registry,factory=setup()
    with pytest.raises(ValueError,match="Incomplete target coverage") as error:
        UnifiedAlgorithmPipeline(registry).run_dag(fixed_graphs(factory)["linear"],task_id=TASK,external=external,version=TASK)
    assert np.isfinite(error.value.details["raw_prediction"]).any()
    assert np.isnan(error.value.details["raw_prediction"]).any()


def test_semantic_coverage_uses_input_support_not_h0_or_truth():
    _,external=fixture();_,factory=setup();graphs=fixed_graphs(factory);c=conditions(external["batch"].value)
    assert factory.semantic_risk(graphs["linear"],c)>0
    assert factory.semantic_risk(graphs["nearest"],c)==0
    assert factory.semantic_risk(graphs["H0"],c)==0
    assert factory.semantic_risk(factory.initial(),c)==1


def test_uniform_negative_projection_preserves_raw_output():
    _,external=fixture();registry,_=setup();batch=external["batch"]
    raw=readonly(-np.ones(len(batch.value.lengths)))
    item=SealedArtifact("estimate_direct",raw,dict(batch.metadata))
    adapter=registry.task_adapters(TASK)["quality.mixed_gap","run"]
    output=adapter.execute({"field":item,"batch":batch},{},external)["field"]
    assert np.all(output.value[0]==-1) and np.all(output.value[1]==0)
    assert output.metadata["negative_projection_count"]==len(raw)


def test_reference_is_not_read_by_search_and_cgap_switch_is_explicit():
    import inspect
    from ecodata.algorithms.numerical_search import _run_numerical_search
    code=inspect.getsource(_run_numerical_search)
    assert "H0" not in code and "fixed_graphs" not in code
    assert 'soft=mode=="C"' in code
    assert 'semantic_chain' in code and 'semantic_proposal_weights' in code


def test_overbudget_raw_callbacks_cannot_execute(monkeypatch,tmp_path):
    import run_beijing_mixed_gap as runner
    import ecodata.algorithms.numerical_search as service
    from types import SimpleNamespace
    _,external=fixture();executed=[]
    class SmallEvaluator:
        def __init__(self,*args):
            self.factory=setup()[1];self.external=external;self.node_hits=self.node_new=self.kernel_hits=0
        def execute(self,graph,key):
            executed.append(key)
            return dict(status="PASS",objective=1.,key=key,graph=runner.record(graph),prediction_fingerprint=key)
        def decision(self):
            from ecodata.algorithms.temporal_search import TemporalEvaluator,ReferenceEvaluation
            return TemporalEvaluator(external,ReferenceEvaluation(np.ones(len(external["query"].value))),tmp_path/"metrics",
                dict(operation="temporal_missing_recovery",require_complete_coverage=True)).decision()
    class CallbackProbe:
        def __init__(self,obj,*args):self.obj=obj;self.stop_optimization=SimpleNamespace(add_condition=lambda *a:None)
        def optimise(self,evaluate):
            _,f=setup()
            for _,recipe in f.recipes("validated_field")[:10]:self.obj(OptGraph([f.instantiate(recipe)]))
    monkeypatch.setattr(service,"EvoGraphOptimizer",CallbackProbe)
    (tmp_path/"protocol_frozen.json").write_text("{}")
    result=service.run_numerical_search("A",11,SmallEvaluator(),dict(objective_limit=2,proposal_limit=20,callback_limit=100,wall_seconds=10),tmp_path/"run")
    assert len(executed)==2 and result["scored"]==2
    assert result["counts"]["callbacks"]==10 and result["counts"]["deferred_callbacks"]==8


def test_redundant_mutated_controls_are_rejected_by_shared_factory():
    _,factory=setup();graph=fixed_graphs(factory)["H0"]
    backup=next(n for n in graph.nodes if n.name=="baseline.temporal_hour_median" and n.parameters.get("mode")=="short_backup")
    backup.content["name"]="baseline.temporal_linear"
    # Remove now-unused fit state edge; keep the same valid backup port type.
    backup.nodes_from=[p for p in backup.nodes_from if p.name=="source.masked_batch"]
    from golem.core.optimisers.graph import OptGraph
    graph=OptGraph([graph.root_node])
    with pytest.raises(ValueError,match="Self-fallback"):factory.inspect(graph)


def test_generated_and_mutated_complete_graphs_belong_to_frozen_grammar():
    import random,hashlib
    from copy import deepcopy
    from datetime import timedelta
    from run_beijing_mixed_gap import GraphRequirements,GPAlgorithmParameters,GraphGenerationParams,IdentityAdapter,typed_tree_growth
    registry,f=setup();canonical={hashlib.sha256(f.inspect(OptGraph([f.instantiate(r)]),complete=True)["key"].encode()).hexdigest() for _,r in f.recipes("validated_field")}
    req=GraphRequirements(max_depth=9,start_depth=9,timeout=timedelta(seconds=10),n_jobs=1)
    gen=GraphGenerationParams(adapter=IdentityAdapter(),node_factory=f,random_graph_factory=f.random_graph)
    params=GPAlgorithmParameters();random.seed(11);graph=f.initial()
    for _ in range(10):
        try:
            candidate=typed_tree_growth(deepcopy(graph),req,gen,params)
            status=f.inspect(candidate)
        except ValueError:continue
        if status["complete"]:assert key_for(candidate,f) in canonical
        graph=candidate


def test_mask_replicas_cannot_donate_values_to_each_other():
    _,external=fixture();first=external["batch"].value.cases[0]
    second=SeriesCase(first.station,first.time,readonly(np.ones(len(first.values))*99999),readonly(np.array([],int)))
    batch=external["batch"].value
    combined=Batch((first,second),batch.lengths,((0,len(batch.lengths)),(len(batch.lengths),len(batch.lengths))))
    for method in ["linear","nearest","ffill","hour_median"]:
        np.testing.assert_allclose(predict(batch,method,external["history"].value),predict(combined,method,external["history"].value),equal_nan=True)
