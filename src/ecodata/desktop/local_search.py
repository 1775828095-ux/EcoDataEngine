"""Bind desktop fusion to the original spatial mutation and assessment entry."""
from pathlib import Path


def model_request(request, model_id, *, daily_amount=False):
    if not model_id:
        return request
    from ecodata.algorithms.field_mutation import resolve_registered_policy
    from ecodata.algorithms.profile_matching import model_table

    variable = request["variable"].casefold()
    unit = request["unit"]
    matches = []
    for role in model_table(model_id)["roles"]:
        if role["record_only"]:
            continue
        for field in role["fields"]:
            names = {name.casefold() for name in [field["name"], *(field.get("aliases") or [])]}
            daily_binding = (daily_amount and variable == field["name"] == "precipitation"
                             and unit == "mm" and field.get("unit") == "mm/day")
            if variable not in names or (field.get("unit") != unit and not daily_binding):
                continue
            if field.get("data_type", field.get("kind", "continuous")) != request["semantic_type"]:
                continue
            matches.append((dict(id=model_id, role=role["role"], field=field["name"]), daily_binding))
    if len(matches) != 1:
        return request

    selection, daily_binding = matches[0]
    if request.get("application_contract") or request.get("model_profile",selection)!=selection:
        return request
    request = dict(request, model_profile=selection, soft_guidance=True)
    if daily_binding:
        request["model_unit_binding"] = "verified_utc_daily_precipitation_amount"
    request = resolve_registered_policy(request, allow_unmatched=True)
    if request["soft_guidance"]:
        request["spatial_candidate_pool"] = 4
    return request


def submit(inputs, request, folder, *, model_id=None, daily_amount=False, progress=None):
    from ecodata.algorithms.platform_search import submit_search

    request = model_request(request, model_id, daily_amount=daily_amount)
    return submit_search(inputs, request, None, Path(folder), progress=progress,
        budget=dict(objective_limit=130, total_call_limit=180, proposal_limit=100, wall_seconds=45))


def pareto_comparison(rankings):
    """Compare completed candidates on the same selection support."""
    if not rankings:
        return {}

    import numpy as np
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    except ImportError as exc:
        return dict(backend="pymoo", status="DEPENDENCY_UNAVAILABLE", error=str(exc))
    objectives = ["RMSE", "MAE"]
    scores = np.array([[row["metrics"][metric] for metric in objectives] for row in rankings])
    front = NonDominatedSorting().do(scores, only_non_dominated_front=True)
    return dict(backend="pymoo", scope="same_fixed_selection_folds", used_for_selection=False,
                objectives=objectives, candidates=[rankings[int(index)] for index in front])
