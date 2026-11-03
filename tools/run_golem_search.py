"""Isolated GOLEM 0.5 runner. Input/output are JSON; no EcoData dependencies."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np

from golem.core.adapter import IdentityAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators import base_mutations
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams


def names(graph):
    return [node.name for node in graph.nodes]


def graph_record(graph):
    nodes = list(graph.nodes)
    return [{"index": i, "algorithm": node.name,
             "parents": [nodes.index(parent) for parent in node.nodes_from]} for i, node in enumerate(nodes)]


def metrics(graph, required, catalog, hard):
    selected = [catalog[name] for name in names(graph) if name in catalog]
    provided = {cap for item in selected for cap in item["capabilities"]}
    missing = set(required) - provided
    violations = 0
    incompatible = [name for name in names(graph) if name not in catalog or not catalog[name].get("eligible", True)]
    violations += len(incompatible)
    if "category_preservation" in hard and any(name == "spatial.bilinear" for name in names(graph)):
        violations += 1
    if "nonnegative_output" in hard and "nonnegative_validation" not in provided:
        violations += 1
    if "interval_conservation" in hard and "temporal_accumulation" not in provided:
        violations += 1
    cost = sum(float(item["compute_cost"]) for item in selected)
    return {"missing": sorted(missing), "physical_violations": violations, "incompatible_nodes": incompatible, "cost": cost,
            "complexity": len(selected), "qualified": not missing and violations == 0}


def run_one(payload, mode, seed):
    random.seed(seed); np.random.seed(seed)
    required = payload["requirements"]["required"]
    hard = payload["requirements"]["hard_constraints"]
    catalog = {item["algorithm_id"]: item for item in payload["catalog"] if item["executable"] and item["capabilities"]}
    available = sorted(catalog)
    if not available:
        raise ValueError("No executable capability modules")
    counters = Counter()
    mutation_trace = []
    evaluated = {}
    started = time.perf_counter()
    first_qualified = None
    budget = payload["budget"].get("max_evaluations")

    def count_proposal(graph):
        semantic_invalid = bool(metrics(graph, required, catalog, hard)["physical_violations"])
        structural_invalid = False
        try:
            has_no_cycle(graph)
            has_no_self_cycled_nodes(graph)
        except ValueError:
            structural_invalid = True
        counters["structural_invalid_mutation_proposals"] += int(structural_invalid)
        counters["invalid_mutation_proposals"] += int(semantic_invalid or structural_invalid)

    def objective(graph):
        nonlocal first_qualified
        counters["objective_calls"] += 1
        if budget is not None and counters["evaluations"] >= budget:
            counters["budget_deferred"] += 1
            return 1e12
        counters["evaluations"] += 1
        value = metrics(graph, required, catalog, hard)
        if value["physical_violations"]:
            counters["invalid_evaluations"] += 1
        if value["qualified"] and first_qualified is None:
            first_qualified = time.perf_counter() - started
        evaluated[graph.descriptive_id] = {"graph": graph_record(graph), "metrics": value}
        return 1000 * (len(value["missing"]) + value["physical_violations"]) + value["cost"] + 0.1 * value["complexity"]

    def semantic_constraint(graph, *args, **kwargs):
        counters["constraint_checks"] += 1
        # Missing capabilities are allowed during evolution and are penalised by the
        # objective. Only combinations that directly violate known semantics are rejected.
        valid = not any(not catalog.get(name, {}).get("eligible", True) for name in names(graph))
        valid = valid and not ("category_preservation" in hard and "spatial.bilinear" in names(graph))
        if not valid:
            counters["invalid_candidates"] += 1
        return valid

    official = [base_mutations.single_add_mutation, base_mutations.single_change_mutation,
                base_mutations.single_drop_mutation, base_mutations.single_edge_mutation]

    def counted(function):
        def mutate(graph, **kwargs):
            counters["mutation_proposals"] += 1
            mutated = function(graph, kwargs["requirements"], kwargs["graph_gen_params"], kwargs["parameters"])
            count_proposal(mutated)
            return mutated
        mutate.__name__ = "counted_" + function.__name__
        return mutate

    def semantic_mutation(graph, **kwargs):
        counters["mutation_proposals"] += 1
        before = names(graph)
        before_structure = graph_record(graph)
        current = metrics(graph, required, catalog, hard)
        gap = set(current["missing"])
        legal, weights = [], []
        for algorithm_id, item in catalog.items():
            if not item.get("eligible", True):
                continue
            trial_names = before + [algorithm_id]
            if "category_preservation" in hard and "spatial.bilinear" in trial_names:
                continue
            match = len(set(item["capabilities"]) & set(required))
            fill = len(set(item["capabilities"]) & gap)
            score = payload["mutation"]["alpha"] * match + payload["mutation"]["beta"] * fill - payload["mutation"]["gamma"] * item["compute_cost"]
            legal.append(algorithm_id); weights.append(math.exp(min(score, 30)))
        if not legal:
            counters["invalid_candidates"] += 1
            return graph
        if random.random() < payload["mutation"].get("exploration", 0.1):
            chosen = random.choice(legal); reason = "random_exploration"
        else:
            chosen = random.choices(legal, weights=weights, k=1)[0]; reason = "capability_gap_weighted"
        roll = random.random()
        operation = "add" if gap or len(graph.nodes) == 1 or roll < 0.55 else "replace"
        if not gap and len(graph.nodes) >= 2 and roll >= 0.85:
            operation = "reconnect"
        if operation == "add" and chosen not in before:
            parent = random.choice(graph.nodes)
            graph.add_node(OptNode(chosen, nodes_from=[parent]))
        elif operation == "reconnect":
            parent, child = random.sample(graph.nodes, 2)
            graph.connect_nodes(parent, child)
        else:
            node = random.choice(graph.nodes)
            node.content = {"name": chosen}
        after = names(graph)
        count_proposal(graph)
        mutation_trace.append({"before": before, "after": after, "operation": operation, "candidate": chosen,
                               "reason": reason, "capability_gap": sorted(gap), "hard_constraints": hard,
                               "before_structure": before_structure, "after_structure": graph_record(graph)})
        return graph

    rules = [has_no_cycle, has_no_self_cycled_nodes]
    if mode in {"B", "C"}:
        rules.append(semantic_constraint)
    mutation_types = [counted(fn) for fn in official] if mode in {"A", "B"} else [semantic_mutation]
    compatible = [name for name in available if catalog[name].get("eligible", True)]
    if not compatible:
        return {"mode": mode, "seed": seed, "status": "INPUT_REQUIREMENT_MISSING", "reasons": "No context-compatible executable nodes"}
    initial_id = min(compatible, key=lambda name: (-len(set(catalog[name]["capabilities"]) & set(required)), catalog[name]["compute_cost"], name))
    initial = [OptGraph([OptNode(initial_id)])]
    requirements = GraphRequirements(num_of_generations=payload["budget"]["generations"], timeout=timedelta(seconds=payload["budget"]["timeout_seconds"]),
                                     max_depth=payload["budget"].get("max_depth", 5), n_jobs=1, show_progress=False)
    params = GPAlgorithmParameters(pop_size=payload["budget"]["population"], max_pop_size=payload["budget"]["population"],
                                   mutation_prob=1.0, crossover_prob=0.0, mutation_types=mutation_types,
                                   crossover_types=[CrossoverTypesEnum.none], max_num_of_operator_attempts=20)
    generation = GraphGenerationParams(adapter=IdentityAdapter(), rules_for_constraint=rules, available_node_types=available)
    golem_objective = Objective({"physics_cost_complexity": objective})
    optimizer = EvoGraphOptimizer(golem_objective, initial, requirements, generation, params)
    if budget is not None:
        optimizer.stop_optimization.add_condition(lambda: counters["evaluations"] >= budget, "Evaluation budget reached")
    started = time.perf_counter()
    found = optimizer.optimise(ObjectiveEvaluate(golem_objective))
    runtime = time.perf_counter() - started
    best = found[0]
    best_metrics = metrics(best, required, catalog, hard)
    qualified = sum(item["metrics"]["qualified"] for item in evaluated.values())
    return {"mode": mode, "seed": seed, "best_graph": names(best), "best_metrics": best_metrics,
            "mutation_proposals": counters["mutation_proposals"], "actual_evaluations": counters["evaluations"],
            "invalid_candidates": counters["invalid_candidates"], "constraint_checks": counters["constraint_checks"],
            "qualified_graphs": qualified, "unique_evaluated_graphs": len(evaluated),
            "invalid_mutation_proposals": counters["invalid_mutation_proposals"],
            "structural_invalid_mutation_proposals": counters["structural_invalid_mutation_proposals"],
            "legal_mutation_ratio": 1 - counters["invalid_mutation_proposals"] / max(1, counters["mutation_proposals"]),
            "invalid_evaluations": counters["invalid_evaluations"], "objective_calls": counters["objective_calls"],
            "budget_deferred": counters["budget_deferred"], "first_qualified_seconds": first_qualified,
            "best_graph_structure": graph_record(best),
            "qualified_graph_records": [value for value in evaluated.values() if value["metrics"]["qualified"]],
            "runtime_seconds": runtime, "semantic_mutation_trace": mutation_trace}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("input"); parser.add_argument("output")
    args = parser.parse_args(); payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = [run_one(payload, mode, seed) for mode in ("A", "B", "C") for seed in payload["seeds"]]
    output = {"backend": "thegolem", "version": "0.5.0", "official_classes": ["EvoGraphOptimizer", "GraphRequirements", "GraphGenerationParams", "GPAlgorithmParameters", "Objective"],
              "results": results}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
