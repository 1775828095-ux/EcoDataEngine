"""Bind desktop fusion to the original spatial mutation and assessment entry."""
from pathlib import Path


def model_request(request, model_id, *, daily_amount=False):
    if not model_id or model_id=='data_only':
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
    result = submit_search(inputs, request, None, Path(folder), progress=progress,
        budget=dict(objective_limit=130, total_call_limit=180, proposal_limit=100, wall_seconds=45))
    if progress:
        gate=result.get('post_mutation_acceptance',{})
        progress('变异后独立复检 · '+gate.get('status','UNCONFIRMED')+'；未通过不作为增强方案推荐。')
    return result


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


def run_paired(scan, task_id, task, frame, request, folder, progress, queues, *, source_indices):
    """Common real paired-series mutation and product export, no selection policy."""
    from pathlib import Path
    import json
    import pandas as pd
    from ecodata.algorithms.platform_search import SearchInput, submit_search
    from ecodata.algorithms.paired_station_search import DEFAULT_BUDGET
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if task.get('review_risk_limits'):
        request=dict(request,review_risk_limits=task['review_risk_limits'])
    result=submit_search([SearchInput('paired.csv','paired_observations',frame.to_csv(index=False).encode())],request,None,
        folder/'mutation',budget=dict(DEFAULT_BUDGET,**task.get('mutation_budget',{})),seed=task.get('seed',11),progress=progress)
    files=[];artifacts=[];reports=[]
    native_updates=[]
    for index,station in enumerate(result['stations'],1):
        if (task.get('search_after_rejection') and station.get('product')
                and not station.get('post_mutation_acceptance', {}).get('recommended')
                and station.get('delivery_status') != 'BASELINE_RETAINED'):
            raise ValueError('变异产物未获最终验收，且没有可交付的质量基线')
        product=station.get('product')
        if product:
            path=Path(product);name=f'station-{index}-product.nc';files.append((name,path.read_bytes()))
            import xarray as xr
            with xr.open_dataset(path) as ds:
                artifacts.append(dict(name=name,dimensions=dict(ds.sizes),fields=list(ds.data_vars)))
                if (task.get('route')=='station_grid' and not task.get('wind_components')
                        and {'native_t','native_y','native_x'}<=set(frame)):
                    from .station_grid import native_update
                    matched=frame[frame.station.astype(str)==str(station['station'])].set_index('grid_time')
                    selected=matched.loc[pd.DatetimeIndex(ds.time.values)].reset_index()
                    native_updates.append(native_update(scan,task,selected,ds[request['variable']].values.reshape(-1),
                        station.get('delivery_status'),task_id))
        root=folder/'mutation'/f'station-{index}'/'search'
        candidate=station.get('candidate_product')
        if candidate and candidate!=product:
            files.append((f'station-{index}-exploratory-candidate.nc',Path(candidate).read_bytes()))
        review=folder/'mutation'/f'station-{index}'/'recommendation_review.json'
        if review.exists():files.append((f'station-{index}-recommendation-review.json',review.read_bytes()))
        for leaf in ['graphs.json','mutation_trace.json','recommendation_validation.json','selection_freeze.json','summary.json']:
            if (root/leaf).exists():files.append((f'station-{index}-'+leaf,(root/leaf).read_bytes()))
        reports.append(dict(mutation_search=station,methods=task['recipes'][0],queue_audit=queues,
            unit=task['unit'],matched_records=len(frame),stations=frame.station.nunique()))
    files.append(('paired-mutation-summary.json',json.dumps(result,ensure_ascii=False,indent=2,default=str).encode()))
    if not artifacts:raise ValueError('No paired station mutation product: '+str(result['stations']))
    return dict(files=files,source_formats=[scan['rows'][i]['format'] for i in source_indices],
        target_format='NetCDF',original_report=reports,local=dict(artifacts=artifacts,native_ready=False,
            reason='公共变异服务已执行；产品为未定位序列预测，不推断地理坐标或模型运行。' if request.get('unlocated_series') else '公共变异服务已执行；产品为站点位置的时间预测，不是原生网格写回或模型运行。'),
        mutation_status=result['status'],native_updates=native_updates)
