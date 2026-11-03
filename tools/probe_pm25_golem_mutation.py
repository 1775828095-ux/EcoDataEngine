"""Bounded official-mutation executability probe, separate from A/B/C inference."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import random
from time import perf_counter

from run_pm25_dag_pilot import SharedFactory, GraphGenerationParams, GraphRequirements, GPAlgorithmParameters, base_mutations
from run_pm25_multisource_dag import prepare, graph, graph_record, dump, evaluate
import numpy as np
from ecodata.algorithms.dag import NumericalDAGExecutor
from ecodata.algorithms.pm25_dag import PM25DAGAdapters
from ecodata.algorithms.registry import create_default_registry


def run(data, out):
    import json
    from run_pm25_multisource_dag import ROOT
    out.mkdir(parents=True, exist_ok=True)
    if (out / "mutation_probe.json").exists():
        raise FileExistsError("Use a new output directory")
    external, validation, _ = prepare(data)
    frozen = json.loads((ROOT / "output/capability_final_acceptance/frozen_generic_pool.json").read_text(encoding="utf-8"))
    counters = Counter()
    generation = GraphGenerationParams(node_factory=SharedFactory([a["id"] for a in frozen["algorithms"]], counters))
    requirements, params = GraphRequirements(max_depth=12), GPAlgorithmParameters()
    adapter = PM25DAGAdapters(create_default_registry(), "2019-08-15 00:30", np.arange(36.5, 40.6, .1), np.arange(114.4, 118.4, .1), exploratory=True)
    engine = NumericalDAGExecutor(adapter.adapters(), version="official-mutation-probe-1")
    initial = graph("external_drift")
    random.seed(11)
    np.random.seed(11)
    attempts, results, seen = [], [], set()
    start = perf_counter()
    for index in range(2000):
        candidate = base_mutations.single_change_mutation(deepcopy(initial), requirements, generation, params)
        record = {"proposal": index+1, "graph": graph_record(candidate)}
        try:
            engine.from_golem(candidate)
            record["contract"] = "PASS"
            methods = [n.parameters.get("variogram_model") for n in candidate.nodes if n.name == "station.pykrige_universal"]
            method = methods[0] if len(methods) == 1 else None
            if method in {"spherical", "exponential"} and method not in seen:
                seen.add(method)
                folder = out / f"proposal{index+1}_{method}"
                folder.mkdir(exist_ok=True)
                try:
                    result = engine.run_golem(candidate, external=external)
                    metrics = evaluate(result, adapter, validation, folder)
                    results.append({"proposal": index+1, "variogram": method, "status": "WARN_EXPLORATORY", "metrics": metrics,
                                    "graph": graph_record(candidate)})
                except Exception as exc:
                    failure = {"proposal": index+1, "variogram": method, "status": "FAIL", "error": str(exc), "trace": getattr(exc, "trace", [])}
                    dump(folder / "failure.json", failure)
                    results.append(failure)
        except Exception as exc:
            record.update(contract="FAIL", error=str(exc))
        attempts.append(record)
        if len(seen) == 2:
            break
    summary = {"operator": "official base_mutations.single_change_mutation", "seed": 11,
        "proposal_limit": 2000, "proposal_count": len(attempts), "seconds": perf_counter()-start,
        "results": results, "attempts": attempts, "factory_counters": dict(counters),
        "selection_rule": "First compatible spherical/exponential graph for each parameter, irrespective of prediction quality",
        "claim": "Mutated GOLEM graph execution only; parameter variants are not additional algorithm topologies"}
    dump(out / "mutation_probe.json", summary)
    print({"proposals": len(attempts), "results": [{k: v for k, v in r.items() if k not in {"trace", "graph"}} for r in results]}, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
