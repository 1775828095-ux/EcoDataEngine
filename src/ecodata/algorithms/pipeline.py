from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext, AlgorithmResult
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.exceptions import ValidationError


class AlgorithmQualityGate:
    def validate(self, context: AlgorithmContext, result: AlgorithmResult) -> dict:
        from ecodata.evaluation.quality_gates import ResultQualityGate
        report = ResultQualityGate().validate(
            context.sources[0].dataset, result.data.dataset, variable=context.variable,
            semantic_type=context.semantic_type, nonnegative=bool(context.model_constraints.get("nonnegative")),
            model_check=context.parameters.get("model_compatibility_check"),
            reference=context.validation_reference, validation_role=context.validation_role,
            expected_output_unit=context.model_constraints.get("output_unit"),
        )
        report["metrics"]["independent_accuracy_available"] = report["accuracy_assessed"]
        report["resolution_claim"] = "输出网格更密不代表真实精度提高"
        return report


class UnifiedAlgorithmPipeline:
    def __init__(self, registry: AlgorithmRegistry):
        self.registry = registry
        self.quality = AlgorithmQualityGate()
        self._dag_executors = {}

    def run_dag(self, graph, *, task_id: str, external: dict, version: str, decision=None, factory=None):
        """Execute registered task ports through the existing numerical engine."""
        from ecodata.algorithms.dag import NumericalDAGExecutor, DAGError
        if (decision is None)!=(factory is None):
            raise DAGError("Task requirement acceptance needs both decision and typed factory")
        acceptance=decision.require_graph(factory,graph) if decision is not None else {"acceptance_status":"NOT_REQUESTED"}
        key = (task_id, version)
        if key not in self._dag_executors:
            self._dag_executors[key] = NumericalDAGExecutor(self.registry.task_adapters(task_id), version=version)
        engine = self._dag_executors[key]
        nodes, sink = engine.from_golem(graph)
        if not any(port.kind == "validated_field" for port in engine.adapter(next(n for n in nodes if n.id == sink)).outputs.values()):
            raise DAGError("Final task port must be quality-validated, not an unchecked prediction")
        result = engine.run(nodes, sink, external=external)
        result["task_id"] = task_id
        result["task_acceptance"] = acceptance
        return result

    def run(self, context: AlgorithmContext, plan: AlgorithmPlan) -> AlgorithmResult:
        started = perf_counter()
        if context.validation_reference is not None and context.validation_role not in {"validation", "independent_validation"}:
            raise ValidationError("精度选型只允许明确 validation 数据；最终 test 继续封存")
        from ecodata.evaluation.quality_gates import RawInputQualityGate
        input_variable = context.variable if context.variable in context.sources[0].dataset else None
        input_quality = context.input_quality or RawInputQualityGate().validate(
            context.sources[0].dataset, variable=input_variable, intended_operation="fusion" if len(context.sources) > 1 else "reconstruction",
            expected_unit=context.model_constraints.get("unit"), requires_spatial=context.target_lat is not None or len(context.sources) > 1,
        )
        if input_quality["status"] == "BLOCK":
            raise ValidationError(f"原始数据质量检查阻止执行：{[c['message'] for c in input_quality['checks'] if c['status'] == 'BLOCK']}")
        current = replace(context, input_quality=input_quality)
        step_reports = []
        final = None
        for algorithm_id in plan.steps:
            plugin = self.registry.get(algorithm_id)
            compatibility = plugin.compatibility(current)
            if not compatibility.eligible:
                raise ValidationError(f"{algorithm_id} 不可执行：{'；'.join(compatibility.reasons)}")
            final = plugin.run(current)
            step_reports.append({"algorithm": algorithm_id, "compatibility": asdict(compatibility), "report": final.report})
            current = replace(current, sources=(final.data,) + current.sources[1:],
                semantic_type=plugin.spec.output_contract.get("semantic_type", current.semantic_type))
        quality = self.quality.validate(replace(context, semantic_type=current.semantic_type), final)
        if quality["status"] == "BLOCK":
            raise ValidationError(f"算法结果质量检查阻止交付：{[c['message'] for c in quality['checks'] if c['status'] == 'BLOCK']}")
        report = {"status": quality["status"], "plan": plan.id, "steps": step_reports,
                  "selection_reasons": list(plan.reasons), "warnings": list(plan.warnings),
                  "quality": quality, "quality_control": {"input": input_quality, "output": quality,
                  "decisions": [{"stage": "input", "decision": input_quality["decision"]}, {"stage": "output", "decision": quality["decision"]}]},
                  "runtime_seconds": perf_counter() - started}
        return AlgorithmResult(final.data, plan.id, report, quality["metrics"])

