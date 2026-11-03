"""Bounded typed-graph acceptance and real A/B/C evaluation; no new optimiser."""
from collections import Counter
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter
import traceback

from run_pm25_multisource_dag import ROOT, dump, prepare, graph_record, evaluate, sha
import numpy as np
import pandas as pd
from ecodata.algorithms.dag import NumericalDAGExecutor
from ecodata.algorithms.golem_generation import TypedGolemFactory, typed_tree_growth
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
from golem.core.optimisers.optimizer import GraphGenerationParams


class SearchLimit(RuntimeError):
    """Declared resource stop, never a successful generation result."""


def setup():
    frozen = ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"
    available = [a["id"] for a in json.loads(frozen.read_text(encoding="utf-8"))["algorithms"]]
    registry = create_default_registry()
    contexts = [PM25DAGAdapters(registry, t, np.arange(36.5, 40.6, .1), np.arange(114.4, 118.4, .1), exploratory=True)
                for t in ("2019-08-15 00:30", "2019-08-15 12:30")]
    factory = TypedGolemFactory(contexts[0].adapters(), available)
    version = sha(ROOT / "src/ecodata/algorithms/golem_generation.py")+sha(ROOT / "src/ecodata/algorithms/dag.py")
    engines = [NumericalDAGExecutor(a.adapters(), version=version+str(a.center)) for a in contexts]
    return factory, contexts, engines, {"frozen_sha256": sha(frozen), "factory_code_sha256": version}


def real_result(graph, contexts, engines, external, validation, folder):
    errors, slices = [], []
    for i, (context, engine) in enumerate(zip(contexts, engines)):
        directory = folder / f"slice{i}"
        directory.mkdir(parents=True, exist_ok=True)
        result = engine.run_golem(graph, external=external)
        metrics = evaluate(result, context, validation, directory)
        values = pd.read_csv(directory / "validation_predictions.csv")
        errors.extend((values.prediction-values.value).tolist())
        slices.append({"center_utc": str(context.center), "metrics": metrics,
                       "node_cache_hits": sum(t["cache_hit"] for t in result["trace"]),
                       "executed_nodes": sum(not t["cache_hit"] for t in result["trace"])})
    error = np.asarray(errors)
    return {"RMSE": float(np.sqrt(np.mean(error**2))), "MAE": float(np.mean(np.abs(error))),
            "Bias": float(error.mean()), "count": len(error), "slices": slices}


def generation_acceptance(external, validation, output):
    results = []
    for route in ("station.pykrige_universal", "fusion.residual"):
        random.seed(11)
        np.random.seed(11)
        factory, contexts, engines, manifest = setup()
        initial = factory.initial()
        before = factory.inspect(initial)
        factory.preferred["analysis_field"] = (route, "run")
        folder = output / route
        folder.mkdir(parents=True, exist_ok=True)
        try:
            # Official single_add strategy; parents are derived from live ports.
            graph = base_mutations.add_separate_parent_node(deepcopy(initial), factory)
            after = factory.inspect(graph, complete=True)
            result = real_result(graph, contexts, engines, external, validation, folder)
            row = {"route": route, "passed": True, "generation_attempts": 1,
                "official_operator": "base_mutations.add_separate_parent_node (single_add_mutation strategy)",
                "gap_before": before["gap"], "gap_after": after["gap"],
                "initial_graph": graph_record(initial), "generated_graph": graph_record(graph), **result}
        except Exception as exc:
            row = {"route": route, "passed": False, "generation_attempts": 1,
                "error": str(exc), "call_chain": traceback.format_exc(), "execution_trace": getattr(exc, "trace", [])}
        dump(folder / "generation_trace.json", {"result": row, "factory_events": factory.trace, **manifest})
        results.append(row)
        if not row["passed"]:
            dump(output / "generation_result.json", {"passed": False, "results": results, **manifest})
            raise RuntimeError(f"Generation failed once at {route}: {row['error']}")
    dump(output / "generation_result.json", {"passed": True, "results": results, **manifest})


def small_pilot(external, validation, sources, output, acceptance):
    accepted = json.loads(Path(acceptance).read_text(encoding="utf-8"))
    _, _, _, version = setup()
    if not accepted["passed"] or accepted["factory_code_sha256"] != version["factory_code_sha256"]:
        raise ValueError("Matching real generation acceptance required before search")
    official = [typed_tree_growth, base_mutations.single_add_mutation, base_mutations.single_change_mutation,
                base_mutations.single_drop_mutation, base_mutations.single_edge_mutation]
    dump(output / "pilot_config.json", {"seed": 11, "maximum_unique_real_evaluations": 5,
        "outer_proposal_limit": 50, "wall_seconds_limit": 30, "sources": sources, "acceptance": str(acceptance),
        "initial": "Same quality node, observations branch only, missing field port",
        "factory": "Shared port grammar; no C-only templates or completion",
        "objective": "complete: RMSE/train_std+0.0001*algorithm_nodes; partial:1000+gap; invalid:1000000",
        "cache_policy": "Cold per group; callbacks/cache/partial counts separate from unique real executions", **version})
    summaries = []
    for mode in ("A", "B", "C"):
        random.seed(11)
        np.random.seed(11)
        factory, contexts, engines, _ = setup()
        catalog = factory.complete_keys()
        initial = factory.initial()
        initial_key = factory.inspect(initial)["key"]
        cache, evaluated_complete = {}, set()
        counts, records, proposals = Counter(), [], []
        folder = output / mode
        folder.mkdir(exist_ok=True)
        started = perf_counter()
        norm = float(external["history"].value.value.std(ddof=0))

        def limited():
            return (counts["outer_proposals"] >= 50 or perf_counter()-started >= 30
                    or counts["real_numerical_evaluations"] >= 5 or evaluated_complete >= catalog)

        def legal(graph, *args, **kwargs):
            counts["posterior_checks"] += 1
            try:
                factory.inspect(graph)  # Incomplete but typed graphs remain evolvable.
                return True
            except Exception:
                counts["posterior_rejections"] += 1
                return False

        def objective(graph):
            counts["objective_callbacks"] += 1
            index = len(records)+1
            row = {"callback": index, "real_execution": False, "cache_hit": False}
            try:
                status = factory.inspect(graph)
                key = status["key"]
                row.update(key=hashlib.sha256(key.encode()).hexdigest(), gap=status["gap"], graph=graph_record(graph))
                if not status["complete"]:
                    row.update(status="INCOMPLETE", score=1000+status["gap"])
                elif key in cache:
                    counts["objective_cache_hits"] += 1
                    row.update(cache[key], real_execution=False, cache_hit=True)
                elif counts["real_numerical_evaluations"] >= 5:
                    row.update(status="BUDGET_DEFERRED", score=1e6)
                else:
                    counts["real_numerical_evaluations"] += 1
                    evaluated_complete.add(key)
                    before = perf_counter()
                    try:
                        result = real_result(graph, contexts, engines, external, validation, folder / f"evaluation{index:02d}")
                        counts["successful_real_evaluations"] += 1
                        row.update(result, status="WARN_EXPLORATORY", score=result["RMSE"]/norm+.0001*status["algorithm_nodes"],
                            real_execution=True)
                    except Exception as exc:
                        row.update(status="NUMERICAL_FAIL", score=1e6, real_execution=True,
                            error=str(exc), failure_trace=getattr(exc, "trace", []), call_chain=traceback.format_exc())
                    row["algorithm_seconds"] = perf_counter()-before
                    cache[key] = dict(row)
            except Exception as exc:
                row.update(status="CONTRACT_REJECT", score=1e6, error=str(exc))
            row["elapsed"] = perf_counter()-started
            records.append(row)
            return row["score"]

        def propose(fn, graph, kwargs):
            if limited():
                raise SearchLimit("Shared proposal/wall/real-evaluation limit reached")
            counts["outer_proposals"] += 1
            record = {"operator": fn.__name__, "proposal": counts["outer_proposals"]}
            try:
                record["gap_before"] = factory.inspect(graph)["gap"]
            except Exception:
                record["gap_before"] = None
            try:
                candidate = fn(deepcopy(graph), kwargs["requirements"], kwargs["graph_gen_params"], kwargs["parameters"])
                status = factory.inspect(candidate)
                record.update(valid=True, gap_after=status["gap"], complete=status["complete"],
                    algorithm_nodes=status["algorithm_nodes"], graph=graph_record(candidate))
                proposals.append(record)
                return candidate, status
            except Exception as exc:
                counts["generation_rejections"] += 1
                record.update(valid=False, error=str(exc), call_chain=traceback.format_exc())
                proposals.append(record)
                if 'candidate' in locals():
                    return candidate, None
                raise

        def counted(fn):
            def mutation(graph, **kwargs):
                return propose(fn, graph, kwargs)[0]
            mutation.__name__ = "counted_"+fn.__name__
            return mutation

        def guided(graph, **kwargs):
            if limited():
                raise SearchLimit("Shared limits reached before semantic proposal")
            candidates, weights = [], []
            try:
                before = factory.inspect(graph)["gap"]
            except Exception:
                before = 0
            for _ in range(4):
                if limited():
                    break
                candidate, status = propose(random.choice(official), graph, kwargs)
                if status is not None:
                    candidates.append(candidate)
                    weights.append(math.exp(1+2*(before-status["gap"])-.25*status["algorithm_nodes"]))
            return random.choices(candidates, weights=weights)[0] if candidates else deepcopy(graph)

        requirements = GraphRequirements(num_of_generations=20, timeout=timedelta(seconds=30), start_depth=12, max_depth=12,
            n_jobs=1, show_progress=False, early_stopping_iterations=8)
        params = GPAlgorithmParameters(pop_size=3, max_pop_size=3, mutation_prob=1., crossover_prob=0.,
            mutation_types=[counted(fn) for fn in official] if mode in {"A", "B"} else [guided],
            crossover_types=[CrossoverTypesEnum.none], max_num_of_operator_attempts=3)
        generation = GraphGenerationParams(adapter=IdentityAdapter(), node_factory=factory,
            random_graph_factory=factory.random_graph,
            rules_for_constraint=[has_no_cycle, has_no_self_cycled_nodes]+([legal] if mode in {"B", "C"} else []))
        obj = Objective({"real_validation_score": objective})
        optimizer = EvoGraphOptimizer(obj, [deepcopy(initial)], requirements, generation, params)
        optimizer.stop_optimization.add_condition(limited, "Shared small-pilot limits/exhaustion")
        error = None
        try:
            optimizer.optimise(ObjectiveEvaluate(obj))
        except SearchLimit:
            counts["normal_resource_stop"] += 1
        except Exception as exc:
            error = {"error": str(exc), "call_chain": traceback.format_exc()}
        valid = [row for row in cache.values() if row["status"] == "WARN_EXPLORATORY"]
        exhausted = evaluated_complete >= catalog
        reason = ("FINITE_SPACE_EXHAUSTED" if exhausted else "REAL_EVALUATION_LIMIT" if counts["real_numerical_evaluations"] >= 5
                  else "PROPOSAL_LIMIT" if counts["outer_proposals"] >= 50 else "WALL_LIMIT" if perf_counter()-started >= 30
                  else "OPTIMIZER_EARLY_STOP")
        summary = {"group": mode, "seed": 11, "counts": dict(counts), "unique_legal_evaluated_graphs": len(valid),
            "finite_complete_catalog_size": len(catalog), "exhausted": exhausted, "stop_reason": reason,
            "initial_graph_key": hashlib.sha256(initial_key.encode()).hexdigest(),
            "gap_completing_mutations": sum(p.get("gap_before", 0) is not None and p.get("gap_after") == 0
                and p.get("gap_before", 0) > 0 for p in proposals),
            "wall_seconds": perf_counter()-started, "best": min(valid, key=lambda r: r["score"]) if valid else None,
            "optimizer_error": error, "data_test_targets_used": False, "objective_records": records,
            "mutation_records": proposals, "factory_events": factory.trace}
        dump(folder / "pilot_result.json", summary)
        summaries.append({k: v for k, v in summary.items() if k not in {"objective_records", "mutation_records", "factory_events"}})
        dump(output / "pilot_summary.json", summaries)
        print(mode, "real", counts["real_numerical_evaluations"], "new legal", len(valid), "proposals", counts["outer_proposals"], reason, error, flush=True)
    normal = all(s["unique_legal_evaluated_graphs"] > 0 and s["gap_completing_mutations"] > 0 and s["optimizer_error"] is None for s in summaries)
    dump(output / "formal_search_decision.json", {"generator_small_pilot_passed": normal,
        "run_twenty_evaluations": False, "run_five_seed_experiment": False,
        "reason": ("Only five distinct complete graphs in this task grammar; twenty unique evaluations would repeat the space. Formal efficacy remains unproven."
            if normal else "Small pilot did not meet generation/execution acceptance; inspect each group's exact stop_reason and optimizer_error before further experiments.")})


def pilot(data, output, *, stage="generation", acceptance=None):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "generation_result.json").exists() or (output / "pilot_summary.json").exists():
        raise FileExistsError("Fresh output directory required; evidence is not overwritten")
    external, validation, sources = prepare(data)
    if stage == "generation":
        generation_acceptance(external, validation, output)
    elif acceptance:
        small_pilot(external, validation, sources, output, acceptance)
    else:
        raise ValueError("Small pilot requires --acceptance from successful generation")
