"""Interface fixtures only; research metrics come from the real execution driver."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
from run_taiwan_composition_search import setup, restore, record, fingerprint
from ecodata.algorithms.dag import Artifact, DAGError
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.genetic.operators import base_mutations


def test_fitted_model_cache_key_is_content_not_repr():
    a = LinearRegression().fit([[0],[1]],[0,1])
    b = LinearRegression().fit([[0],[1]],[0,8])
    assert str(a) == str(b)
    assert Artifact("state",a).fingerprint() != Artifact("state",b).fingerprint()
    assert Artifact("state",a).fingerprint() == Artifact("state",a).fingerprint()


def test_shared_factory_generates_real_multi_branch_contracts():
    backend,factory,frozen = setup()
    graphs = [OptGraph([factory.instantiate(recipe)]) for _,recipe in factory.recipes("validated_field")]
    assert len(frozen)==34
    assert len(graphs)>4
    for graph in graphs:
        assert factory.inspect(graph,complete=True)["gap"]==0
        assert fingerprint(restore(record(graph)),factory)==fingerprint(graph,factory)
    assert any(any(n.parameters.get("mode")=="residual" for n in g.nodes) for g in graphs)


def test_official_add_completes_missing_port_deterministically():
    _,factory,_=setup()
    initial=factory.initial()
    assert factory.inspect(initial)["gap"]==1
    graph=base_mutations.add_separate_parent_node(initial,factory)
    assert factory.inspect(graph,complete=True)["gap"]==0


def test_every_raw_prediction_subtree_shares_one_fit_state():
    _,factory,_=setup()
    # Exhaust the finite subtree grammar instead of retrying random mutations.
    for _,recipe in factory.recipes("raw_prediction"):
        assert len(factory.regressors(recipe))<=1
        subtree=OptGraph([factory.instantiate(recipe)])
        assert sum(n.parameters.get("mode")=="fit" for n in subtree.nodes)<=1


def test_heldout_labels_rejected_at_source():
    backend,_,_=setup()
    t=pd.DataFrame(dict(station_id=["x"],longitude=[121.],latitude=[24.],time=[pd.Timestamp("2017-01-01",tz="UTC")],unit=["ug m-3"],aod=[.2],value=[12.]))
    with pytest.raises(DAGError,match="held-out labels"):
        backend.source({},dict(source="query"),{"query":Artifact("query",t,{})})


def test_fit_apply_rejects_wrong_training_state():
    backend,_,_=setup()
    with pytest.raises(DAGError,match="another training"):
        backend.state({"train":Artifact("training",pd.DataFrame({"value":[1]})),
            "state":Artifact("background_state",{"fit_fingerprint":"incorrect"})})


def test_nonfinite_result_is_block_not_reduced_cohort():
    backend,_,_=setup()
    backend.check=lambda inputs: None
    with pytest.raises(DAGError,match="entire common cohort"):
        backend.finish({"prediction":Artifact("raw_prediction",np.array([np.nan])),
            "query":Artifact("query",pd.DataFrame({"x":[1]}))},{},{})


def test_real_budget_curve_counts_cache_callbacks_in_order():
    path=ROOT/"output/taiwan_composition_search_v1/budget_curves.csv"
    if not path.exists():
        pytest.skip("Real search artifact not present in this checkout")
    curves=pd.read_csv(path)
    assert "source_evaluation_index" in curves
    for _,group in curves.groupby(["mode","seed"]):
        assert group.evaluation.tolist()==list(range(1,len(group)+1))
        assert len(group)<=20


def test_registry_task_binding_is_real_and_does_not_add_plugins():
    _,factory,_=setup()
    registry=factory.task_registry
    count=len(registry.specs())
    assert registry.task_adapters("taiwan_pm25_station_aod")==factory.adapters
    assert len(registry.specs())==count
    adapter=next(iter(factory.adapters.values()))
    with pytest.raises(KeyError,match="Unknown algorithm"):
        registry.register_task_adapters("invalid",{("unregistered.fake","run"):adapter},
            algorithm_ids=["unregistered.fake"])


def test_semantics_mapper_drives_real_missing_capability():
    _,factory,_=setup()
    status=factory.inspect(factory.initial())
    assert set(factory.requirements.required)=={"observation_interpolation","nonnegative_validation"}
    assert status["semantic_missing"]==["observation_interpolation"]
    assert status["semantic_trace"]


def test_unknown_regressor_is_rejected_not_silently_linear():
    backend,_,_=setup()
    with pytest.raises(DAGError,match="Unsupported regression parameter"):
        backend.fit({},dict(regressor="invented"),{})


def test_pipeline_rejects_unchecked_output_before_execution():
    from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
    _,factory,_=setup()
    _,recipe=factory.recipes("raw_prediction")[0]
    graph=OptGraph([factory.instantiate(recipe)])
    with pytest.raises(DAGError,match="quality-validated"):
        UnifiedAlgorithmPipeline(factory.task_registry).run_dag(graph,task_id="taiwan_pm25_station_aod",external={},version="fixture")


def test_input_quality_block_precedes_any_fit():
    backend,_,_=setup()
    backend.input_quality=lambda frame: {"status":"BLOCK"}
    stamp=pd.Timestamp("2017-01-01",tz="UTC")
    frame=pd.DataFrame(dict(station_id=[f"s{i}" for i in range(15)],longitude=np.linspace(120,121,15),
        latitude=np.linspace(23,24,15),time=[stamp]*15,unit=["ug m-3"]*15,aod=[.2]*15,value=[12.]*15,split=["train"]*15))
    with pytest.raises(DAGError,match="before algorithm fitting"):
        backend.source({},dict(source="train"),{"train":Artifact("training",frame,
            dict(time=stamp,unit="ug m-3",metric_crs="EPSG:32651",contract="fixture"))})


def test_terminal_failure_feedback_is_cached_without_reexecution():
    from run_taiwan_composition_search import real_graph_feedback
    class FixtureEvaluator:
        calls=0
        def execute(self,graph,key):
            self.calls+=1
            return {"status":"BLOCK_INCOMPLETE_COVERAGE","score":1e6}
    evaluator=FixtureEvaluator()
    _,factory,_=setup()
    graph=factory.initial()
    cache={}
    first=real_graph_feedback(evaluator,graph,"fixture",cache)
    second=real_graph_feedback(evaluator,graph,"fixture",cache)
    assert first["real_execution"] and not second["real_execution"]
    assert second["cache_hit"] and evaluator.calls==1


def test_executed_sum_rejects_cross_model_states_even_without_factory():
    backend,_,_=setup()
    metadata=dict(unit="ug m-3",metric_crs="EPSG:32651",time="fixture",contract="fixture")
    inputs={"background":Artifact("background_prediction",np.array([10.]),dict(metadata,fit_state_fingerprint="RF")),
        "residual":Artifact("residual_prediction",np.array([-2.]),dict(metadata,fit_state_fingerprint="linear",valid_residual_mask=[True]))}
    with pytest.raises(DAGError,match="same fitted state"):
        backend.combine(inputs,{}, {})
    inputs["residual"].metadata["fit_state_fingerprint"]="RF"
    np.testing.assert_allclose(backend.combine(inputs,{}, {})["prediction"].value,[8.])


def test_search_journal_is_immediately_readable_and_preserves_failures(tmp_path):
    import json
    from run_taiwan_composition_search import write_search_event
    path=tmp_path/"events.jsonl"
    write_search_event(path,"proposal_start",proposal=1)
    assert len(path.read_text(encoding="utf-8").splitlines())==1
    write_search_event(path,"proposal_result",valid=False,error="契约不兼容")
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["error"]=="契约不兼容" and rows[0]["event"]=="proposal_start"


def test_strict_registry_never_smuggles_rf_or_idw_as_residual_modes():
    backend,factory,_=setup()
    adapters=backend.adapters()
    assert ("fusion.residual","fit") not in adapters
    assert ("fusion.residual","direct") not in adapters
    assert ("fusion.residual","residual_interpolate") not in adapters
    for _,recipe in factory.recipes("validated_field"):
        graph=OptGraph([factory.instantiate(recipe)])
        assert all(n.parameters.get("regressor") is None for n in graph.nodes)
        assert all(n.parameters.get("estimator_id") in {None,"station.pykrige_ordinary","station.pykrige_universal"} for n in graph.nodes)


def test_three_missing_inputs_are_completed_in_three_official_adds():
    _,factory,_=setup()
    _,recipe=factory.recipes("validated_field")[0]
    graph=OptGraph([factory.instantiate(recipe,omit_root_ports=("train","query","prediction"))])
    gaps=[factory.inspect(graph)["gap"]]
    for _ in range(3):
        graph=base_mutations.add_separate_parent_node(graph,factory)
        gaps.append(factory.inspect(graph)["gap"])
    assert gaps==[3,2,1,0]
