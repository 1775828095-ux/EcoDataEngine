from __future__ import annotations

from dataclasses import dataclass

from ecodata.algorithms.contracts import AlgorithmContext, Compatibility
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.scale import TargetScaleAdvisor, spatial_scale_factor


@dataclass(frozen=True)
class AlgorithmPlan:
    id: str
    steps: tuple[str, ...]
    score: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    selected: AlgorithmPlan
    evaluations: tuple[dict, ...]
    basis: str


class AlgorithmSelector:
    def __init__(self, registry: AlgorithmRegistry):
        self.registry = registry

    def screen(self, context: AlgorithmContext) -> dict[str, Compatibility]:
        if context.input_quality and context.input_quality.get("status") == "BLOCK":
            reasons = tuple(item["message"] for item in context.input_quality.get("checks", []) if item.get("status") == "BLOCK")
            return {spec.id: Compatibility(False, reasons or ("原始数据质量检查阻止算法选择",)) for spec in self.registry.specs()}
        return {spec.id: self.registry.get(spec.id).compatibility(context) for spec in self.registry.specs()}

    def recommend(self, context: AlgorithmContext) -> tuple[AlgorithmPlan, ...]:
        if context.input_quality and context.input_quality.get("status") == "BLOCK":
            return ()
        decision = TargetScaleAdvisor().decide(context)
        scale = spatial_scale_factor(context)
        plans = []
        for spec in self.registry.specs():
            result = self.registry.get(spec.id).compatibility(context)
            if not result.eligible or not (spec.scale_range[0] <= scale <= spec.scale_range[1]):
                continue
            score = {"low": 4.0, "medium": 2.0, "high": 0.5, "gpu": 0.0}.get(spec.compute, 0.0)
            reasons = [f"输入语义 {context.semantic_type} 与插件声明相容"]
            if spec.id == decision.recommended_method:
                score += 5
                reasons.append(decision.reason)
            if len(context.sources) > 1 and spec.category == "multi_source_fusion":
                score += 4
                reasons.append("存在多源数据")
            if context.observations is not None and spec.requires_observations:
                score += 3
                reasons.append("存在训练观测约束")
            if spec.requires_training:
                score -= 2
            if context.input_quality and context.input_quality.get("status") == "WARN":
                reasons.append("输入质量存在警告；结果必须经过第二道质量检查")
                if spec.id == "fusion.quality_aware":
                    score += 2
                    reasons.append("存在可靠观测时优先考虑质量感知校正")
            plans.append(AlgorithmPlan(spec.id, (spec.id,), score, tuple(reasons), decision.warnings + result.warnings))
        # Generic composition: align the background first, then residual-fuse observations.
        if context.target_lat is not None and context.target_lon is not None and len(context.sources) >= 2:
            regrid = decision.recommended_method
            residual = "fusion.residual"
            if all(self.registry.get(step).compatibility(context).eligible for step in (regrid, residual)):
                plans.append(AlgorithmPlan(f"{regrid}+{residual}", (regrid, residual), 10.0,
                    ("先统一尺度，再进行同变量观测残差融合",), decision.warnings))
        return tuple(sorted(plans, key=lambda item: (-item.score, item.id)))

    def select(self, context: AlgorithmContext, plans: tuple[AlgorithmPlan, ...] | None = None) -> SelectionResult:
        """Select on explicit validation data, otherwise retain the conservative heuristic order."""
        plans = plans or self.recommend(context)
        if not plans:
            raise ValueError("没有满足输入语义、依赖和物理约束的算法")
        if context.validation_reference is None:
            return SelectionResult(plans[0], (), "没有独立高分辨率真值；按适用性、物理约束和计算成本推荐，不声称精度最优")
        if context.validation_role not in {"validation", "independent_validation"}:
            raise ValueError("算法选择只能使用 validation，最终 test 必须封存")
        from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
        evaluations = []
        for plan in plans:
            try:
                result = UnifiedAlgorithmPipeline(self.registry).run(context, plan)
                metrics = result.metrics
                evaluations.append({"plan": plan.id, "status": "PASS", "RMSE": metrics.get("RMSE"),
                                    "MAE": metrics.get("MAE"), "bias": metrics.get("bias"),
                                    "runtime_seconds": result.report.get("runtime_seconds")})
            except Exception as exc:
                evaluations.append({"plan": plan.id, "status": "FAIL", "reason": str(exc)})
        valid = [item for item in evaluations if item.get("status") == "PASS" and item.get("RMSE") is not None]
        if not valid:
            raise ValueError("候选算法均未通过独立验证和质量门")
        best = min(valid, key=lambda item: (item["RMSE"], item.get("runtime_seconds") or float("inf")))
        return SelectionResult(next(plan for plan in plans if plan.id == best["plan"]), tuple(evaluations),
                               "使用明确 validation 数据比较 RMSE；物理质量门失败的候选已排除，最终 test 未参与")

    def execute_with_fallback(self, context: AlgorithmContext, plans: tuple[AlgorithmPlan, ...] | None = None,
                              *, max_attempts: int = 3):
        """Try each compatible plan at most once; never retry indefinitely."""
        from dataclasses import replace
        from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
        candidates = (plans or self.recommend(context))[:max(1, max_attempts)]
        attempts = []
        for plan in candidates:
            try:
                result = UnifiedAlgorithmPipeline(self.registry).run(context, plan)
                attempts.append({"plan": plan.id, "status": result.report["status"]})
                report = dict(result.report)
                report["fallback_attempts"] = attempts
                return replace(result, report=report)
            except Exception as exc:
                attempts.append({"plan": plan.id, "status": "BLOCK", "reason": str(exc)})
        raise ValueError(f"候选算法均未通过双层质量门；尝试记录：{attempts}")

