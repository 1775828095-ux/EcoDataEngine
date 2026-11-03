"""Small A/B/C GOLEM execution pilot, not a confirmatory efficacy experiment."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter

from run_pm25_multisource_dag import ROOT, dump, prepare, graph, graph_record, evaluate
import numpy as np
import pandas as pd
from ecodata.algorithms.dag import NumericalDAGExecutor
from ecodata.algorithms.pm25_dag import PM25DAGAdapters
from ecodata.algorithms.registry import create_default_registry
from golem.core.adapter import IdentityAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators import base_mutations
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.core.optimisers.optimizer import GraphGenerationParams


class SharedFactory(DefaultOptNodeFactory):
    """Same finite parameter schema for all groups; official operators unchanged."""
    def __init__(self, available, counters=None):
        super().__init__(available)
        self.counters = counters if counters is not None else Counter()

    def get_node(self, **kwargs):
        self.counters["node_factory_calls_including_random_growth"] += 1
        node = super().get_node(**kwargs)
        if node.name == "station.pykrige_universal":
            node.parameters = {"variogram_model": random.choice(["linear", "spherical", "exponential"])}
        elif node.name == "fusion.residual":
            node.parameters = {"interpolation": random.choice(["linear", "nearest"])}
        elif node.name == "fusion.quality_aware":
            node.parameters = {"mode": random.choice(["fit", "apply"])}
        return node


def legacy_pilot(data, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pilot_summary.json").exists():
        raise FileExistsError("Use a new directory")
    external, validation, sources = prepare(data)
    registry = create_default_registry()
    frozen_path = ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    available = [a["id"] for a in frozen["algorithms"]]
    initial = [graph("external_drift"), graph("calibrated_residual")]
    official = [base_mutations.tree_growth, base_mutations.single_add_mutation, base_mutations.single_change_mutation,
                base_mutations.single_drop_mutation, base_mutations.single_edge_mutation]
    config = {"seed": 11, "maximum_scored_objective_calls": 20, "maximum_seconds_per_group": 60,
        "maximum_mutation_proposals": 3000, "frozen_pool": available,
        "initial_graphs": [graph_record(g) for g in initial], "sources": sources,
        "design_deviation_from_chapter18": "Shared TWO validated complete initial graphs for executor smoke test; not the planned incomplete-prefix capability-gap experiment",
        "capability_gap_efficacy_testable": False, "reason": "Both initial graphs already satisfy the minimal delivery task",
        "objective": "validation RMSE / training-only std + 0.0001 * algorithm node count; invalid=1000000",
        "validation_samples": "Same two UTC slices and geometry-supported stations, no test targets",
        "factory": "SharedFactory, identical in A/B/C; official five mutation functions",
        "C": "At most four official proposals, semantic legality/declared cost weighting; all proposal costs counted"}
    dump(output / "pilot_config.json", config)
    summaries = []
    for mode in ("A", "B", "C"):
        random.seed(11)
        np.random.seed(11)
        folder = output / mode
        folder.mkdir(exist_ok=True)
        contexts = [PM25DAGAdapters(registry, time, np.arange(36.5, 40.6, .1), np.arange(114.4, 118.4, .1), exploratory=True)
                    for time in ("2019-08-15 00:30", "2019-08-15 12:30")]
        # Fresh per-group caches; every group pays its own identical source preparation.
        executors = [NumericalDAGExecutor(a.adapters(), version="pilot-1-"+str(a.center)) for a in contexts]
        norm = float(external["history"].value.value.std(ddof=0))
        counts, proposals, records, cache = Counter(), [], [], {}
        started = perf_counter()

        def key(g):
            return hashlib.sha256(json.dumps(graph_record(g), sort_keys=True).encode()).hexdigest()

        def inspect(g):
            nodes, sink = executors[0].from_golem(g)
            outputs = executors[0].adapter(next(n for n in nodes if n.id == sink)).outputs
            if not any(p.kind == "validated_field" for p in outputs.values()):
                raise ValueError("Missing final quality-validated target port")
            names = {n.operation for n in nodes}
            if not {"source.geos", "source.stations"}.issubset(names):
                raise ValueError("Missing required independent source roles")
            return nodes

        def legal(g, *args, **kwargs):
            counts["posterior_checks"] += 1
            try:
                inspect(g)
                return True
            except Exception:
                counts["posterior_rejections"] += 1
                return False

        def objective(g):
            counts["objective_calls_including_deferred"] += 1
            if len(records) >= 20:
                counts["budget_deferred"] += 1
                return 1e6
            index = len(records)+1
            fingerprint = key(g)
            if fingerprint in cache:
                row = {**cache[fingerprint], "cache_hit": True}
            else:
                beginning = perf_counter()
                row = {"valid": False, "score": 1e6, "cache_hit": False, "graph": graph_record(g), "key": fingerprint}
                try:
                    nodes = inspect(g)
                    counts["numerical_graph_evaluations"] += 1
                    errors = []
                    execution_nodes = cached_nodes = 0
                    for i, (adapter, engine) in enumerate(zip(contexts, executors)):
                        sub = folder / f"evaluation{index:02d}" / f"slice{i}"
                        sub.mkdir(parents=True, exist_ok=True)
                        result = engine.run_golem(g, external=external)
                        evaluate(result, adapter, validation, sub)
                        execution_nodes += sum(not t["cache_hit"] for t in result["trace"])
                        cached_nodes += sum(t["cache_hit"] for t in result["trace"])
                        values = pd.read_csv(sub / "validation_predictions.csv")
                        errors.extend((values.prediction-values.value).tolist())
                    errors = np.asarray(errors)
                    cost = sum(n.operation in available for n in nodes)
                    rmse = float(np.sqrt(np.mean(errors**2)))
                    row.update(valid=True, score=rmse/norm+.0001*cost, RMSE=rmse,
                        MAE=float(np.mean(np.abs(errors))), Bias=float(errors.mean()), count=len(errors),
                        executed_nodes=execution_nodes, cached_nodes=cached_nodes)
                except Exception as exc:
                    row.update(error=f"{type(exc).__name__}: {exc}", failure_trace=getattr(exc, "trace", []))
                row["numerical_seconds"] = perf_counter()-beginning
                cache[fingerprint] = dict(row)
            records.append({**row, "objective_index": index, "elapsed_seconds": perf_counter()-started})
            return row["score"]

        def propose(fn, g, kwargs):
            counts["mutation_proposals"] += 1
            before = key(g)
            try:
                candidate = fn(deepcopy(g), kwargs["requirements"], kwargs["graph_gen_params"], kwargs["parameters"])
            except Exception as exc:
                counts["invalid_proposals"] += 1
                proposals.append({"operator": fn.__name__, "valid_contract": False, "changed": False,
                    "graph": graph_record(g), "error": f"Mutation execution: {type(exc).__name__}: {exc}"})
                raise
            try:
                nodes = inspect(candidate)
                valid = True
                cost = sum(n.operation in available for n in nodes)
            except Exception as exc:
                valid, cost = False, 999
                counts["invalid_proposals"] += 1
                error = f"{type(exc).__name__}: {exc}"
            proposals.append({"operator": fn.__name__, "valid_contract": valid, "key": key(candidate),
                "changed": before != key(candidate), "graph": graph_record(candidate),
                "error": None if valid else error})
            return candidate, valid, cost

        def counted(fn):
            def mutation(g, **kwargs):
                return propose(fn, g, kwargs)[0]
            mutation.__name__ = "counted_"+fn.__name__
            return mutation

        def guided(g, **kwargs):
            # Same proposal family/factory as A and B, not exclusive template replacement.
            candidates = [propose(random.choice(official), g, kwargs) for _ in range(4)]
            feasible = [(candidate, cost) for candidate, valid, cost in candidates if valid]
            if not feasible:
                counts["guided_no_valid_proposal_fallback"] += 1
                return deepcopy(g)
            # Gap is zero on these complete prefixes. Do not fabricate an active gap term.
            weights = [math.exp(1-.25*cost) for _, cost in feasible]
            return random.choices([c for c, _ in feasible], weights=weights)[0]

        requirements = GraphRequirements(num_of_generations=30, timeout=timedelta(seconds=60), max_depth=12,
            n_jobs=1, show_progress=False, early_stopping_iterations=12)
        params = GPAlgorithmParameters(pop_size=4, max_pop_size=4, mutation_prob=1., crossover_prob=0.,
            mutation_types=[counted(fn) for fn in official] if mode in {"A", "B"} else [guided],
            crossover_types=[CrossoverTypesEnum.none], max_num_of_operator_attempts=10)
        generation = GraphGenerationParams(adapter=IdentityAdapter(), node_factory=SharedFactory(available, counts),
            rules_for_constraint=[has_no_cycle, has_no_self_cycled_nodes]+([legal] if mode in {"B", "C"} else []))
        obj = Objective({"real_validation_score": objective})
        optimizer = EvoGraphOptimizer(obj, deepcopy(initial), requirements, generation, params)
        optimizer.stop_optimization.add_condition(lambda: len(records) >= 20 or counts["mutation_proposals"] >= 3000,
            "Shared Objective/proposal budget")
        error = None
        try:
            optimizer.optimise(ObjectiveEvaluate(obj))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        valid = [r for r in records if r["valid"]]
        unique_valid = {r["key"] for r in valid}
        best = min(valid, key=lambda r: r["score"]) if valid else None
        summary = {"group": mode, "seed": 11, "scored_objective_calls": len(records), "counts": dict(counts),
            "valid_objective_records": len(valid), "unique_valid_graphs": len(unique_valid),
            "proposal_valid_ratio": 1-counts["invalid_proposals"]/max(1, counts["mutation_proposals"]),
            "wall_seconds": perf_counter()-started, "algorithm_seconds": sum(r.get("numerical_seconds", 0) for r in cache.values()),
            "best": best, "optimizer_error": error, "capability_gap_active": False,
            "test_metrics_computed": False, "objective_records": records, "mutation_records": proposals,
            "proposal_accounting_note": "verifier counts also include official random-growth population fill; factory calls include those non-mutation proposals"}
        dump(folder / "pilot_result.json", summary)
        summaries.append({k: v for k, v in summary.items() if k not in {"objective_records", "mutation_records"}})
        dump(output / "pilot_summary.json", summaries)
        print(mode, "objective", len(records), "valid", len(valid), "unique", len(unique_valid), "error", error, flush=True)
    dump(output / "formal_search_decision.json", {"run_formal_search": False,
        "reasons": ["Only eight geometry-supported validation records", "Unconfirmed station datum",
                    "Two complete seeded pathways; capability-gap guidance is inactive",
                    "Pilot diagnoses executability/cost only, cannot establish semantic mutation efficacy"]})


def pilot(data, output, *, stage="generation", acceptance=None):
    # Historical pilot/probe definitions above remain readable for old evidence.
    # The active CLI always uses the common bounded typed factories.
    from pm25_typed_pilot import pilot as typed_pilot
    return typed_pilot(data, output, stage=stage, acceptance=acceptance)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("generation", "small"), default="generation")
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args()
    pilot(args.data, args.output, stage=args.stage, acceptance=args.acceptance)
