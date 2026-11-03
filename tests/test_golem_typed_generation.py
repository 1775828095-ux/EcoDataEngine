"""Small structural tests; no data scan, fitting, or random retry probes."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / ".venv-golem/Lib/site-packages"))
from golem.core.optimisers.genetic.operators import base_mutations
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from ecodata.algorithms.dag import NumericalDAGExecutor, DAGError
from ecodata.algorithms.golem_generation import TypedGolemFactory, GenerationError, typed_tree_growth
from ecodata.algorithms.pm25_dag import PM25DAGAdapters
sys.path.remove(str(ROOT / ".venv-golem/Lib/site-packages"))


def factory():
    frozen = json.loads((ROOT / "output/capability_final_acceptance/frozen_generic_pool.json").read_text(encoding="utf-8"))
    adapters = PM25DAGAdapters(None, "2019-08-15 00:30", [36.5, 36.6], [114.4, 114.5], exploratory=True).adapters()
    return TypedGolemFactory(adapters, [a["id"] for a in frozen["algorithms"]])


@pytest.mark.parametrize("operation", ["station.pykrige_universal", "fusion.residual"])
def test_official_add_mutation_completes_each_route_once(operation):
    random.seed(11)
    np.random.seed(11)
    f = factory()
    graph = f.initial()
    assert f.inspect(graph)["gap"] == 1
    with pytest.raises(DAGError):
        NumericalDAGExecutor(f.adapters).from_golem(graph)
    f.preferred["analysis_field"] = (operation, "run")
    mutated = base_mutations.add_separate_parent_node(deepcopy(graph), f)
    status = f.inspect(mutated, complete=True)
    assert status["gap"] == 0
    nodes, sink = NumericalDAGExecutor(f.adapters).from_golem(mutated)
    assert operation in {n.operation for n in nodes}
    if operation == "fusion.residual":
        apply = next(n for n in nodes if n.operation == "fusion.quality_aware" and n.parameters["mode"] == "apply")
        parent = next(n for n in nodes if n.id == apply.inputs["state"][0])
        assert parent.operation == "fusion.quality_aware" and parent.parameters["mode"] == "fit"


def test_random_graph_hook_is_one_bounded_construction(monkeypatch):
    import golem.core.optimisers.random_graph_factory as upstream
    def forbidden(*args, **kwargs):
        pytest.fail("Legacy random_graph retry loop must not run")
    monkeypatch.setattr(upstream, "random_graph", forbidden)
    f = factory()
    req = GraphRequirements(timeout=timedelta(seconds=1), max_depth=12)
    partial = f.random_graph(req, max_depth=1)
    assert not f.inspect(partial)["complete"]
    complete = f.random_graph(req, max_depth=12)
    assert f.inspect(complete, complete=True)["complete"]
    assert sum(t["event"] == "random_graph_request" for t in f.trace) == 2


def test_impossible_request_fails_directly_and_catalog_is_finite():
    f = factory()
    with pytest.raises(GenerationError, match="No producer"):
        f.build("nonexistent_physical_contract")
    assert sum(t["event"] == "reject" for t in f.trace) == 1
    assert len(f.complete_keys()) == 5
    assert len(f.available_nodes) == 34


def test_parameter_exchange_keeps_input_contracts():
    f = factory()
    f.preferred["analysis_field"] = ("station.pykrige_universal", "run")
    root, _ = f.build("analysis_field")
    changed = f.exchange_node(root)
    assert changed is not None and not changed.nodes_from
    assert f.adapter(root).inputs == f.adapter(changed).inputs
    assert root.parameters["variogram_model"] != changed.parameters["variogram_model"]


def test_official_single_add_completes_partial_with_shared_depth_contract():
    random.seed(11)
    np.random.seed(11)
    f = factory()
    graph = f.initial()
    req = GraphRequirements(start_depth=12, max_depth=12)
    generation = GraphGenerationParams(node_factory=f, random_graph_factory=f.random_graph)
    mutated = base_mutations.single_add_mutation(graph, req, generation, GPAlgorithmParameters())
    assert f.inspect(mutated, complete=True)["gap"] == 0


def recorded_graph(rows):
    from golem.core.optimisers.graph import OptGraph, OptNode
    nodes = [OptNode({"name": row["operation"], "params": deepcopy(row["parameters"])}) for row in rows]
    for node, row in zip(nodes, rows):
        node.nodes_from = [nodes[index] for index in row["parents"]]
    return OptGraph(nodes)


def test_replay_all_six_b_additions_reach_objective(tmp_path):
    import hashlib
    saved = json.loads((ROOT / "output/golem_generation_fix/small_v2/B/pilot_result.json").read_text(encoding="utf-8"))
    f = factory()
    evidence = []
    for proposal in saved["mutation_records"]:
        if proposal["operator"] != "single_add_mutation":
            continue
        state = f.inspect(recorded_graph(proposal["graph"]))
        key = hashlib.sha256(state["key"].encode()).hexdigest()
        evaluations = [r for r in saved["objective_records"] if r.get("key") == key]
        assert proposal["valid"] and state["gap"] == proposal["gap_after"]
        assert evaluations and all(r["status"] == "INCOMPLETE" for r in evaluations)
        assert all(r["score"] == 1000+state["gap"] for r in evaluations)
        evidence.append({"proposal": proposal["proposal"], "gap_before": proposal["gap_before"],
                         "gap_after": state["gap"], "objective_callbacks": [r["callback"] for r in evaluations]})
    assert len(evidence) == 6
    (tmp_path / "b_replay.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def test_partial_graph_survives_official_mutation_inheritance_and_selection(monkeypatch, tmp_path):
    from golem.core.optimisers.genetic.operators.mutation import Mutation
    from golem.core.optimisers.genetic.operators.selection import Selection
    from golem.core.optimisers.genetic.operators.inheritance import Inheritance
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from golem.core.optimisers.graph import OptGraph
    from golem.core.optimisers.objective import Objective
    f = factory()
    f.preferred["analysis_field"] = ("fusion.residual", "run")
    # Fixed incomplete root: missing both inputs. No stochastic search/probes.
    graph = OptGraph([f.instantiate(f.recipes("validated_field")[0][1], depth=1)])
    assert f.inspect(graph)["gap"] == 2
    monkeypatch.setattr(base_mutations, "shuffle", lambda values: None)
    monkeypatch.setattr(random, "choice", lambda values: values[0])

    def add_parent(graph, **kwargs):
        return base_mutations.add_separate_parent_node(graph, f)

    def posterior(graph, *args, **kwargs):
        f.inspect(graph)  # Same partial-contract acceptance as B's legal().
        return True

    params = GPAlgorithmParameters(pop_size=3, max_pop_size=3, mutation_prob=1.,
        mutation_types=[add_parent], max_num_of_operator_attempts=1)
    generation = GraphGenerationParams(node_factory=f, random_graph_factory=f.random_graph,
                                      rules_for_constraint=[posterior])
    operator = Mutation(params, GraphRequirements(start_depth=12, max_depth=12), generation)
    select = Selection(params)
    inherit = Inheritance(params, select)
    objective = Objective({"structural_gap": lambda g: 1000+f.inspect(g)["gap"]})
    parent = Individual(graph)
    parent.set_evaluation_result(objective(parent.graph))
    evidence = [{"gap": 2, "uid": parent.uid}]
    for expected in (1, 0):
        child = operator(parent)
        assert isinstance(child, Individual) and child.uid != parent.uid
        assert child.parents[0].uid == parent.uid
        assert f.inspect(child.graph)["gap"] == expected
        child.set_evaluation_result(objective(child.graph))
        next_population = inherit([parent], [child])
        assert len(next_population) == 1 and next_population[0].uid == child.uid
        parent = select(next_population, pop_size=1)[0]
        evidence.append({"gap": expected, "uid": parent.uid, "parent_uid": child.parents[0].uid})
    assert f.inspect(parent.graph, complete=True)["complete"]
    NumericalDAGExecutor(f.adapters).from_golem(parent.graph)
    (tmp_path / "continuous_completion.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


@pytest.mark.parametrize("operation,mode", [("prepare.geos", "run"),
    ("fusion.quality_aware", "fit"), ("source.era5", "run")])
def test_tree_growth_passes_actual_subtree_type(monkeypatch, operation, mode):
    import golem.core.optimisers.genetic.operators.base_mutations as upstream
    f = factory()
    f.preferred["analysis_field"] = ("fusion.residual", "run") if mode == "fit" else ("station.pykrige_universal", "run")
    from golem.core.optimisers.graph import OptGraph
    recipe = next(recipe for _, recipe in f.recipes("validated_field")
                  if any(n.name == operation and n.parameters.get("mode", "run") == mode
                         for n in OptGraph([f.instantiate(recipe)]).nodes))
    graph = OptGraph([f.instantiate(recipe)])
    index = next(i for i, n in enumerate(graph.nodes)
                 if n.name == operation and n.parameters.get("mode", "run") == mode)

    def target_first(values):
        values[:] = [index]+[i for i in values if i != index]

    monkeypatch.setattr(upstream, "shuffle", target_first)
    monkeypatch.setattr(random, "shuffle", target_first)
    req = GraphRequirements(start_depth=12, max_depth=12)
    generation = GraphGenerationParams(node_factory=f, random_graph_factory=f.random_graph)
    # Reproduce old hook's wrong final-output replacement, no retries.
    broken = upstream.tree_growth(deepcopy(graph), req, generation, GPAlgorithmParameters(), local_growth=False)
    with pytest.raises(GenerationError, match="contract conflict"):
        f.inspect(broken)
    expected = next(iter(f.adapter(graph.nodes[index]).outputs.values())).kind
    repaired = typed_tree_growth(deepcopy(graph), req, generation, GPAlgorithmParameters(), local_growth=False)
    assert f.inspect(repaired, complete=True)["complete"]
    NumericalDAGExecutor(f.adapters).from_golem(repaired)
    requests = [e for e in f.trace if e["event"] == "subtree_request"]
    assert len(requests) == 1 and requests[0]["output_kind"] == expected
    assert [e for e in f.trace if e["event"] == "random_graph_request"][-1]["output_kind"] == expected
