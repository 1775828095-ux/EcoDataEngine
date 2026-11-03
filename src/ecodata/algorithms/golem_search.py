"""EcoData adapter for the isolated official GOLEM backend."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecodata.algorithms.capabilities import capability_catalog
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.semantic_requirements import CapabilityRequirements


def semantic_proposal_weights(decision, factory, before, candidates, *, soft=True, alpha=2., beta=4.):
    """Shared typed numerical-search guidance, NOT the legacy structural objective.

    Only gap/support evidence changes proposal probabilities; numerical fitness
    still comes exclusively from the real executor's independent evaluator.
    """
    import numpy as np
    current = decision.assess_graph(factory,before)
    assessments = [decision.assess_graph(factory,c) for c in candidates]
    use_soft=soft and "retained_observation_support" in decision.requirements.soft_objectives
    scores=[alpha*(current["gap"]-a["gap"])-
            (beta*a["support_deficit"] if use_soft and a["support_deficit"] is not None else 0.) for a in assessments]
    weights=np.exp(np.clip(scores,-20,20));probabilities=weights/weights.sum() if len(weights) else weights
    return weights.tolist(),dict(before=current,candidates=assessments,weights=weights.tolist(),
        probabilities=probabilities.tolist(),soft_enabled=use_soft,alpha=alpha,beta=beta,
        reason="shared requirements + output-path capabilities + observable support; no accuracy labels")


@dataclass(frozen=True)
class GolemSearchConfig:
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    population: int = 8
    generations: int = 4
    timeout_seconds: int = 30
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.25
    exploration: float = 0.1
    max_evaluations: int | None = None


class GolemCapabilitySearch:
    def __init__(self, registry: AlgorithmRegistry, python: str | None = None):
        self.registry = registry
        root = Path(__file__).resolve().parents[3]
        self.root = root
        self.python = python or os.environ.get("ECODATA_GOLEM_PYTHON") or str(root / ".venv-golem" / "Scripts" / "python.exe")

    def run(self, requirements: CapabilityRequirements, config: GolemSearchConfig = GolemSearchConfig(), *,
            context: AlgorithmContext | None = None, algorithm_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
        interpreter = Path(self.python)
        if not interpreter.is_file():
            raise RuntimeError("GOLEM隔离环境不存在；请安装thegolem==0.5.0并设置ECODATA_GOLEM_PYTHON")
        catalog = []
        for record in capability_catalog(self.registry):
            if algorithm_ids is not None and record.algorithm_id not in algorithm_ids:
                continue
            item = record.as_dict()
            if context is not None:
                plugin = self.registry.get(record.algorithm_id)
                try:
                    compatible = plugin.compatibility(context)
                    reasons = list(compatible.reasons)
                    operation = context.parameters.get("task_operation")
                    if operation == "station_to_grid" and "observation_interpolation" not in record.capabilities:
                        reasons.append("Task requires station-to-grid estimation, not resampling a background grid")
                    item.update(eligible=compatible.eligible and not reasons, incompatibility_reasons=reasons)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    item.update(eligible=False, incompatibility_reasons=[str(exc)])
            catalog.append(item)
        payload = {"requirements": requirements.as_dict(),
                   "catalog": catalog,
                   "seeds": list(config.seeds),
                   "budget": {"population": config.population, "generations": config.generations,
                              "timeout_seconds": config.timeout_seconds, "max_depth": 5,
                              "max_evaluations": config.max_evaluations},
                   "mutation": {"alpha": config.alpha, "beta": config.beta, "gamma": config.gamma,
                                "exploration": config.exploration}}
        with tempfile.TemporaryDirectory(prefix="ecodata_golem_") as directory:
            input_path = Path(directory) / "input.json"; output_path = Path(directory) / "output.json"
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = [str(interpreter), str(self.root / "tools" / "run_golem_search.py"), str(input_path), str(output_path)]
            completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=config.timeout_seconds * len(config.seeds) * 3 + 60)
            if completed.returncode != 0:
                raise RuntimeError(f"GOLEM真实运行失败：\nSTDOUT:\n{completed.stdout[-3000:]}\nSTDERR:\n{completed.stderr[-3000:]}")
            result = json.loads(output_path.read_text(encoding="utf-8"))
        result["command"] = command[:2] + ["<input.json>", "<output.json>"]
        result["backend_log"] = completed.stdout[-10000:]
        result["stderr"] = completed.stderr[-2000:]
        result["candidate_compatibility"] = [{"algorithm_id": item["algorithm_id"],
            "executable": item["executable"], "eligible": item.get("eligible"),
            "reasons": item.get("incompatibility_reasons", [])} for item in catalog]
        return result
