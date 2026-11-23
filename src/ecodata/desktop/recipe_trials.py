"""Shared quality decision and finite-population gate for fixed local trials."""
import numpy as np

from ecodata.algorithms.reliable_selection import aggregate, common_gain, make_policy
from ecodata.algorithms.spatial_fusion_task import blend_fields
from ecodata.algorithms.spatial_selection_protocol import error_metrics
from ecodata.core.exceptions import EcoDataError

from .sampling_risk import evaluate_recipe, recipe_settings

FILL = "fusion.series_missing_fallback"
BLEND = "spatial.blend"
ARRAY_ROUTES = {"windowed_grid_recipe", "indexed_table_recipe"}


def required(task):
    return bool(task and (task.get("action") == "fusion" or task.get("route") == "independent_grid_search"))


def review(scan, task_id, task, steps, *, model=None, independent=False):
    from . import independent_alignment, indexed_table_recipe, station_grid, windowed_grid_recipe
    from .local_search import model_request

    route = (task or {}).get("route")
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
    return dict(status="pending", basis="joint_recipe_adapter_unavailable", count=0, checks=[], route=route)


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


def audit(population, unit, trial):
    if not population:
        return dict(status="pending", basis="recipe_coverage_budget_insufficient", count=0, checks=[])
    scores = []

    def check(index):
        group, metrics, reason = trial(index)
        if metrics:
            scores.append(dict(unit=str(group), **metrics))
        if reason:
            return dict(unit=str(group), rule=reason, unscored=reason == "insufficient_score_support")
        return None

    report = evaluate_recipe(population, unit, check, **recipe_settings())
    basis = ("joint_recipe_acceptance_passed" if report["status"] == "checked" else
             "recipe_numeric_rejected" if report["status"] == "failed" else
             "recipe_coverage_budget_insufficient")
    return dict(status=report["status"], basis=basis, count=sum(item["count"] for item in scores),
                checks=["fixed_alignment_gap_fusion_trial"], acceptance_sampling=report, sampled_scores=scores)
