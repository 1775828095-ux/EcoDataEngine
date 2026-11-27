"""Shared quality decision and finite-population gate for fixed local trials."""
import numpy as np

from ecodata.algorithms.reliable_selection import aggregate, common_gain, make_policy
from ecodata.algorithms.spatial_fusion_task import blend_fields
from ecodata.algorithms.spatial_selection_protocol import error_metrics
from ecodata.core.exceptions import EcoDataError

from .sampling_risk import (BAYESIAN_CHECKPOINT, BAYESIAN_TARGET, evaluate_bayesian_recipe,
                            evaluate_revision_sample, sample_other_units, stratified_posterior)

FILL = "fusion.series_missing_fallback"
BLEND = "spatial.blend"
ARRAY_ROUTES = {"windowed_grid_recipe", "indexed_table_recipe"}
UNSCORED = {"insufficient_score_support", "insufficient_gap_score_support", "insufficient_fusion_score_support"}
SELF_REFERENCE = frozenset({
    "REDUCED_SCALE_PRODUCT_RECONSTRUCTION_PROXY",
    "MASKED_PRIMARY_GAP_RECONSTRUCTION_PROXY",
    "reduced_scale_grid_proxy",
    "masked_local_table_observations",
    "reconstructed_primary_table_proxy",
})


def require_external_reference(report):
    """Keep input-product reconstruction evidence outside the accuracy decision."""
    refs = {item.get("reference") for item in report.get("sampled_scores", ())}
    if not refs.intersection(SELF_REFERENCE):
        return report
    result = dict(report)
    if "acceptance_sampling" in result:
        result["diagnostic_sampling"] = result.pop("acceptance_sampling")
    result.update(status="pending", basis="external_accuracy_reference_required",
                  reason="现有评分使用输入来源自身的遮点或重建值，只能诊断重建表现。"
                         "正式精度验收需要独立实测或有来源证明的第三方参考。")
    return result


# Adapters below have a real public mutation path; this is not a ranking rule.
SEARCH_CONTINUATION_ROUTES = frozenset({
    "independent_grid_search", "windowed_grid_recipe", "station_grid", "indexed_table_recipe"})
PUBLIC_SEARCH_ROUTES=SEARCH_CONTINUATION_ROUTES | {'pair_processing','observation_gap_search','original_engineering','native_reference_search','local'}


def continuation_steps(task, rejected):
    """Only retain registered preparation, never the rejected fusion/correction."""
    if task.get("route") not in SEARCH_CONTINUATION_ROUTES:
        return None
    if task["route"] == "independent_grid_search":
        return ["spatial.source"]
    steps = [step for step in rejected["steps"] if step not in {BLEND, "station.bias.holdout"}]
    if not steps:
        return None
    return steps


def required(task):
    return bool(task and (task.get("action") == "fusion" or task.get("route") == "independent_grid_search"
                         or task.get('route')=='station_grid'))


def review(scan, task_id, task, steps, *, model=None, independent=False):
    from . import independent_alignment, indexed_table_recipe, station_grid, windowed_grid_recipe
    from .local_search import model_request

    route = (task or {}).get("route")
    if not independent and task and task.get("holdout_lock_error"):
        return dict(status="pending", basis="holdout_overlap_unverified", count=0, checks=[],
                    reason=task["holdout_lock_error"])
    try:
        if route == "independent_grid_search":
            scoring = model_request(dict(task="multisource", variable="precipitation", unit="mm",
                                         semantic_type="continuous"), model, daily_amount=True)
            return independent_alignment.review(scan, task_id, dict(task, scoring_request=scoring), steps,
                                                independent=independent)
        adapters = {"windowed_grid_recipe": windowed_grid_recipe.review,
                    "indexed_table_recipe": indexed_table_recipe.review, "station_grid": station_grid.review}
        if route in adapters:
            scoring = model_request(dict(task="multisource", variable=task["quantity"], unit=task["unit"],
                                         semantic_type="continuous"), model)
            return adapters[route](scan, dict(task, scoring_request=scoring), steps, independent=independent)
    except indexed_table_recipe.IndexRequired as error:
        return dict(status="pending", basis="joint_recipe_index_required", count=0, checks=[], reason=str(error))
    except (EcoDataError, OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError, ImportError) as error:
        return dict(status="pending", basis="joint_grid_holdout_unavailable", count=0, checks=[], reason=str(error))
    label = {"original_engineering": "原生工程端口", "native_reference_search": "原生网格参考",
             "pair_processing": "通用配对处理", "observation_gap_search": "观测缺口搜索"}.get(route)
    reason = (f"{label or '当前处理路线'}尚未接入完整的留出试跑与评分。"
              if task else "候选还缺少可用的本地处理入口。")
    return dict(status="pending", basis="joint_recipe_adapter_unavailable", count=0, checks=[], route=route, reason=reason)


def combine(aligned, variable, steps):
    from .grid_preparation import fill

    output, evidence = fill(aligned, 0, variable)
    if BLEND in steps:
        left, right = (ds[variable] for ds in aligned)
        paired = np.isfinite(left) & np.isfinite(right)
        output[variable] = blend_fields(left, right, .5).where(paired, output[variable])
        output[variable].attrs = dict(left.attrs)
    return output, evidence


def score(candidate, baseline, truth, reference, *, retained, request=None):
    if not np.isfinite(truth).any():
        return None, "insufficient_score_support"
    expected = np.isfinite(truth)
    if not (np.isfinite(candidate) | ~expected).all():
        return None, "recipe_output_missing_on_reference"
    if not (np.isfinite(baseline) | ~expected).all():
        return None, "insufficient_score_support"
    policy = make_policy(request or {}, retained)
    observed = [error_metrics(values, truth) for values in (candidate, baseline)]
    measured = [aggregate([dict(fold_id="group", status="PASS", metrics=metrics)], policy, ["group"])
                for metrics in observed]
    gain = common_gain(*measured, policy)
    tolerance = max(policy["delta"], policy["roundoff_tolerance"])
    result = dict(candidate=observed[0], baseline=observed[1], count=int(expected.sum()), reference=reference,
                  candidate_loss=gain["candidate_loss"], baseline_loss=gain["baseline_loss"],
                  loss_tolerance=tolerance, scoring_rule=policy["version"], metric_weights=policy["weights"])
    failed = gain["gain"] < -tolerance
    return result, "recipe_score_below_single_source" if failed else None


def sample_acceptance(population, unit, check, *, scope=None, excluded=()):
    if scope:
        return evaluate_revision_sample(population, unit, check, scope)
    return evaluate_bayesian_recipe(population, unit, check, excluded=excluded)


def score_cohort(scope, index):
    if not scope:
        return {}
    if scope["mode"] == "census":
        label = "本批全部组"
    elif scope["mode"] == "development":
        label = "前轮已调组" if index in scope["previous"] else "本轮新增调参组"
    else:
        label = "已调组" if index in scope["previous"] or index in scope["development_new"] else "未调组"
    return dict(review_cohort=label)


def reference_positions(size, phase, seed, *, partitions=4):
    """Stable group-specific partitions shared by all candidates."""
    order = np.random.default_rng(seed).permutation(size)
    return np.sort(order[phase::partitions])


def audit(population, unit, trial, *, excluded=(), scope=None):
    if not population or len(excluded) == population:
        return dict(status="pending", basis="no_disjoint_partition", population=0, unit=unit, count=0, checks=[])
    scores = []

    def check(index):
        group, metrics, reason = trial(index)
        if metrics:
            items = metrics if isinstance(metrics, list) else [metrics]
            scores.extend(dict(unit=str(group), **score_cohort(scope, index), **item) for item in items)
        elif not reason:
            reason = "insufficient_score_support"
        if reason:
            return dict(unit=str(group), rule=reason, unscored=reason in UNSCORED)
        return None

    report = sample_acceptance(population, unit, check, scope=scope, excluded=excluded)
    basis = ("joint_recipe_acceptance_passed" if report["status"] == "checked" else
             "recipe_numeric_rejected" if report["status"] == "failed" else
             "recipe_scoring_reference_insufficient" if report["unscored"] else
             "recipe_coverage_budget_insufficient")
    missing = {item["rule"] for item in report.get("unscored_units", ())}
    reasons = {"insufficient_gap_score_support": "缺少可用于模拟缺测的实测参考和单源基线",
               "insufficient_fusion_score_support": "缺少双源同时有效、参考值未进入融合输入的配对时刻",
               "insufficient_score_support": "缺少能在相同位置比较方案与单源基线的参考值"}
    return dict(status=report["status"], basis=basis, count=sum(item["count"] for item in scores),
                checks=["fixed_alignment_gap_fusion_trial"], acceptance_sampling=report, sampled_scores=scores,
                reason="；".join(reasons[code] for code in sorted(missing)) if report["status"] == "pending" else "")


def needs_revision(report):
    audit = report.get("acceptance_sampling", {})
    status = report.get("status", audit.get("status"))
    return bool(audit.get("inspected_indices") and
                (status == "failed" or (status == "pending" and audit.get("unscored"))))


def revision_scope(audit, previous_scope=None):
    from random import Random
    from secrets import randbits

    population = audit["population"]
    previous = set(audit["inspected_indices"])
    if previous_scope:
        previous.update(previous_scope["previous"])
        previous.update(previous_scope["development_new"])
    # Keep enough untouched groups to match the participants whenever possible.
    target = min(population // 2, BAYESIAN_CHECKPOINT // 2)
    count = max(0, target - len(previous))
    new = sample_other_units(population, previous, count, Random(randbits(64)))
    participants = len(previous) + len(new)
    sizes = participants, population - participants
    quota = min(*sizes, BAYESIAN_CHECKPOINT // 2)
    balanced = quota and stratified_posterior(sizes, (quota, quota), (0, 0)) >= BAYESIAN_TARGET
    if not balanced and population > BAYESIAN_CHECKPOINT:
        return None
    return dict(population=population, previous=tuple(sorted(previous)), development_new=tuple(new),
                seed=randbits(64), mode="stratified" if balanced else "census", mask_phase=3)


def revision_feedback(report):
    audit = report["acceptance_sampling"]
    scores = report.get("unit_scores", report.get("sampled_scores", ()))

    def summarize(items):
        deltas = [entry["candidate"]["RMSE"] - entry["baseline"]["RMSE"] for entry in items]
        return dict(candidate_mean_bias=float(np.mean([entry["candidate"]["Bias"] for entry in items])) if items else None,
                    baseline_mean_bias=float(np.mean([entry["baseline"]["Bias"] for entry in items])) if items else None,
                    candidate_rmse_median=float(np.median([entry["candidate"]["RMSE"] for entry in items])) if items else None,
                    baseline_rmse_median=float(np.median([entry["baseline"]["RMSE"] for entry in items])) if items else None,
                    rmse_change_median=float(np.median(deltas)) if deltas else None,
                    rmse_change_worst=max(deltas, default=None))

    cohorts = [dict(item, **summarize([score for score in scores if score.get("review_cohort") == item["label"]]))
               for item in audit.get("strata", ())]
    return dict(population=audit["population"], inspected=audit["inspected"], scored=audit["scored"],
                failed=audit["failed"], posterior_probability=audit["posterior_probability"],
                unscored=audit.get("unscored", 0), development_only=audit.get("development_only", False),
                assessment=report.get("basis"), reason=report.get("reason"),
                cohorts=cohorts, **summarize(scores),
                failure_rules=sorted({item["rule"] for item in audit["failures"]}))


def revision_configuration(option):
    return {key: option[key] for key in ("id", "steps", "parameters", "time_matching", "correction")
            if key in option}


def request_revision(send, task_id, previous, feedback, options, comparisons, development):
    payload = dict(stage="bayesian_revision", task=task_id, previous_recipe=previous["id"],
                   previous_configuration=revision_configuration(previous),
                   bias_evidence=feedback, previous_development_trials=comparisons,
                   current_development=development, allowed_recipes=options)
    by_id = {option["id"]: option for option in options}
    for _ in range(2):
        reply = send(payload)
        recipe = reply.get("recipe")
        if isinstance(recipe, str) and recipe in by_id:
            return dict(by_id[recipe], reason=str(reply.get("reason", ""))[:600])
        payload["reply_issue"] = "上一回复未返回可用方案。请核对 allowed_recipes，只填写其中尚未试跑的编号。"
    return None


def tune_recipe(first, report, options, evaluate, send, progress, recoverable):
    """Send measured failures back and verify each new recipe until one passes."""
    history, comparisons = [], []
    current = first
    scope = None
    tried = {tuple(first["steps"])}
    while needs_revision(report):
        remaining = [option for option in options if tuple(option["steps"]) not in tried]
        if not remaining:
            return current, report, history, "no_new_recipe"
        previous_feedback = revision_feedback(report)
        scope = revision_scope(report["acceptance_sampling"], scope)
        if scope is None:
            return current, report, history, "insufficient_untouched_evidence"
        development = dict(scope, mode="development", mask_phase=1 + len(history) % 2)
        measured = evaluate(current, development)
        before = revision_feedback(measured) if measured.get("acceptance_sampling") else None
        if progress:
            progress(f"第 {len(history)+1} 轮重荐：DeepSeek 正在根据上一轮初筛结果和偏差修改方案。")
        try:
            choice = request_revision(send, first["task"], current, previous_feedback, remaining, comparisons, before)
        except recoverable as error:
            if progress:
                progress(str(error))
            return current, report, history, "recommendation_unavailable"
        if choice is None:
            return current, report, history, "no_valid_new_recipe"
        choice["task"] = first["task"]
        tried.add(tuple(choice["steps"]))
        trial = evaluate(choice, development)
        after = revision_feedback(trial) if trial.get("acceptance_sampling") else None
        if progress:
            progress(f"方案 {choice['id']} 已冻结，正在用原评分机制复跑初筛。")
        report = evaluate(choice, scope)
        evidence = revision_feedback(report) if report.get("acceptance_sampling") else None
        comparison = dict(previous_recipe=current["id"], recipe=choice["id"], before=before, after=after,
                          configuration=revision_configuration(choice), mask_phase=development["mask_phase"],
                          formal_review=evidence)
        comparisons.append(comparison)
        history.append(dict(recipe=choice["id"], previous_formal_review=previous_feedback,
                            development_trial=comparison))
        current = choice
        if report.get("status", report.get("acceptance_sampling", {}).get("status")) == "checked":
            return current, report, history, "passed"
    return current, report, history, "scoring_unavailable" if history else "not_required"
