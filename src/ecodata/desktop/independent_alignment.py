"""Independent precipitation conversion and grid alignment."""
import hashlib
import io

from .scan import assert_current, local_member, read_selected


def check_cf_headers(scan, fields):
    import numpy as np
    import xarray as xr
    from .grid_preparation import daily_contract, grid_candidates

    periods, grids = [], []
    for index, field in fields:
        with local_member(scan["rows"][index]) as path, xr.open_dataset(path, cache=False) as dataset:
            data, bounds, _ = daily_contract(dataset, field)
            if data.sizes["time"] < 2 or bounds.shape != (data.sizes["time"], 2):
                raise ValueError("需要足够的降水时间区间")
            ends = bounds.isel(time=[0, -1]).values
            if not np.issubdtype(ends.dtype, np.datetime64) or np.isnat(ends).any():
                raise ValueError("降水时间边界没有有效日期")
            periods.append((ends[0, 0], ends[-1, 1]))
            grids.append(xr.Dataset(coords={axis: data[axis].load() for axis in ("lat", "lon")}))
    if max(begin for begin, _ in periods) >= min(end for _, end in periods):
        raise ValueError("两份降水资料没有共同 UTC 日期范围")
    grid_candidates(grids, "conservative")


def prepare(scan, task_id, task, folder):
    import numpy as np
    import pandas as pd
    from ecodata.models.engineering.precipitation_adaptation import adapt_products
    from ecodata.models.engineering.precipitation_products import inspect_product
    from ecodata.recognition import recognise_file
    from .grid_preparation import daily_precipitation
    from .fallback_evidence import digest_file
    import xarray as xr

    sources, audits, adapted, conversion = [], [], [], []
    hashes, reserved_from = set(), []
    for index, field in task["fields"]:
        row = scan["rows"][index]
        if task.get("cf_daily"):
            with local_member(row) as path, xr.open_dataset(path, cache=False) as ds:
                source_times = ds.time.values
                reserved_from.append(pd.Timestamp(source_times[-max(1, len(source_times) // 5)]).ceil("D").to_datetime64())
                adapted.append(dict(product=f"source-{len(adapted) + 1}", dataset=daily_precipitation(ds, field)))
                digest = digest_file(path)
            if digest in hashes:
                raise ValueError(f"{task_id}：同一来源副本不能用于双源补缺或融合")
            hashes.add(digest)
            conversion.append(dict(source=row["name"], field=field, source_sha256=digest,
                                   operation="UTC_interval_daily_aggregation"))
            continue
        raw = read_selected(row, limit=32*1024**2)
        ds, _ = recognise_file(row["name"], raw, record_read_source=False)
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            raise ValueError(f"{task_id}：同一来源副本不能用于双源补缺或融合")
        hashes.add(digest)
        if "time" not in ds.coords or not ds.sizes["time"]:
            raise ValueError(f"{task_id}：没有原始时间坐标")
        source_times = ds.time.values
        if len(source_times) > 1:
            reserved_from.append(pd.Timestamp(source_times[-max(1, len(source_times) // 5)]).ceil("D").to_datetime64())
        product = inspect_product(ds)
        if not product or not any(f["field"]==field for f in product["fields"]):
            raise ValueError(f"{task_id}：未能核验降水产品")
        # Use only the time basis established by the prior hash-bound local audit.
        product_field = next(f for f in product["fields"] if f["field"]==field)
        from .discovery import audit_precipitation
        verified = audit_precipitation(scan,row,{"name":field},raw,ds)
        if not verified or verified["time_basis"]!="UTC" or verified["blockers"]:
            raise ValueError(f"{task_id}：日降水区间尚未核验")
        product_field.update(time_basis="UTC",blockers=[])
        sources.append(dict(resource=row["name"],field=field,quantity="precipitation",dataset=ds,
                            source_id=product["source_id"],sha256=digest,product=product["product"]))
        audits.append(dict(resource=row["name"],**product))
    if not task.get("cf_daily"):
        adapted, conversion = adapt_products(sources, audits, folder, [])
    if len(adapted) != 2 or (not task.get("cf_daily") and not all(entry.get("product") for entry in conversion)):
        raise ValueError(f"{task_id}：原版日累计转换未生成两份产品")
    grids = [entry["dataset"] for entry in adapted]
    from pyproj import CRS
    from .grid_preparation import choose_grid, grid_candidates, remap

    start = max(ds.time.values[0] for ds in grids)
    end = min(ds.time.values[-1] for ds in grids)
    if start > end:
        raise ValueError(f"{task_id}：没有共同 UTC 日期范围")
    day = np.timedelta64(1, "D")
    common = np.arange(start.astype("datetime64[D]"), end.astype("datetime64[D]") + day, day)
    reserved = common[common >= max(reserved_from)] if len(reserved_from) == len(grids) else common[:0]
    split = len(reserved) // 2
    locked = [scan["rows"][index].get("precipitation_holdout_days") for index, _ in task["fields"]]
    planned = task.get("locked_holdout")
    if planned is not None or (locked[0] and locked[0] == locked[1]):
        held = np.array(planned["days"] if planned is not None else locked[0], dtype="datetime64[D]")
        original_overlap = np.intersect1d(*(ds.time.values.astype("datetime64[D]") for ds in grids))
        if not np.isin(held, original_overlap).all():
            raise ValueError("预留日期与实际双源日期不一致，请重新扫描")
        reserved = held
        split = 0
        comparison_times = original_overlap[original_overlap < held[0]] if len(held) else held
    else:
        comparison_times = reserved[:split]
    for ds in grids:
        times = ds.time.values
        if not np.array_equal(times, times.astype("datetime64[D]")) or not np.array_equal(
                ds.time_bounds.values, np.column_stack((times, times + day))):
            raise ValueError(f"{task_id}：源时间标签与 UTC 完整日区间不一致")
        if not ds.attrs.get("crs"):
            ds.attrs.update(crs="EPSG:4326", crs_status="WORKING_ASSUMPTION",
                            crs_evidence="Inspected product geographic axes; datum not independently established")
        if CRS.from_user_input(ds.attrs["crs"]) != CRS.from_epsg(4326):
            raise ValueError(f"{task_id}：此处的保守对齐需要一致的 EPSG:4326 网格")
        for axis in ("lat", "lon"):
            steps = np.diff(ds[axis].values)
            if not len(steps) or not np.all(steps > 0) or not np.allclose(steps, steps[0]):
                raise ValueError(f"{task_id}：需要规则经纬网格")
    grids = [ds.reindex(time=common).assign(time_bounds=(("time", "nv"), np.column_stack((common, common + day))))
             for ds in grids]
    primary = max(range(2), key=lambda i: float(np.diff(grids[i].lat)[0] * np.diff(grids[i].lon)[0]))
    target_spec = task.get("target_grid")
    if target_spec:
        from ecodata.spatial.regridding import resolve_target_grid
        target = resolve_target_grid(grids, target_spec)
        alignment = dict(selection_basis="DECLARED_TARGET_GRID", target_grid=target_spec)
    elif reviewed := task.get("verified_recipe_alignment"):
        candidates = dict(grid_candidates(grids, "conservative"))
        target = candidates.get(reviewed["selected_source_grid"] - 1)
        if target is None:
            raise ValueError("已验收的对齐网格无法在原始数据重现")
        alignment = reviewed
    else:
        _, target, alignment = choose_grid(grids, primary, "precipitation", "conservative",
                                          comparison_times=comparison_times)
    aligned = [remap(ds, target, "precipitation", "conservative") for ds in grids]
    return adapted, conversion, grids, aligned, target, alignment, primary, common, reserved[split:]


def indexed_holdout(scan, task, stack, *, independent=False):
    """Establish dates from coordinates and open only selected observation slices."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from ecodata.spatial.regridding import normalize_grid_axes, resolve_target_grid
    from .grid_preparation import choose_grid, daily_precipitation, remap
    sources = []
    for index, field in task["fields"]:
        path = stack.enter_context(local_member(scan["rows"][index]))
        ds = stack.enter_context(xr.open_dataset(path, cache=False))
        ds, _ = normalize_grid_axes(ds)
        if any(ds.sizes[axis] < 2 for axis in ("lat", "lon")):
            raise ValueError("保守对齐需要至少两个网格坐标")
        time = ds.time.values
        bounds = ds[ds.time.attrs.get("bounds", "time_bounds")].transpose("time", ...).values
        if (len(time) < 2 or bounds.shape != (len(time), 2)
                or not np.issubdtype(bounds.dtype, np.datetime64) or not np.issubdtype(time.dtype, np.datetime64)
                or np.isnat(bounds).any() or np.isnat(time).any()
                or (np.diff(time) <= np.timedelta64(0, "s")).any()
                or (time < bounds[:, 0]).any() or (time > bounds[:, 1]).any()
                or (bounds[:, 1] <= bounds[:, 0]).any() or (bounds[1:, 0] < bounds[:-1, 1]).any()):
            raise ValueError("回检需要两份有序、无重叠的日期区间")
        dates = bounds[:, 0].astype("datetime64[D]")
        first_reserved = (dates[0] if independent else
                          pd.Timestamp(time[-max(1, len(time) // 5)]).ceil("D").to_datetime64())
        sources.append((ds, field, dates, first_reserved))
    day = np.timedelta64(1, "D")
    start = max(source[3] for source in sources).astype("datetime64[D]")
    end = min(source[2][-1] for source in sources)
    reserved = np.arange(start, end + day, day)
    split = 0 if task.get("target_grid") else len(reserved) // 2
    planned = task.get("locked_holdout") if not independent else None
    if planned is not None:
        reserved = np.asarray(planned["days"], dtype="datetime64[D]")
        common_days = np.intersect1d(*(source[2] for source in sources))
        calibration_days = common_days[common_days < reserved[0]] if len(reserved) else reserved
        split = 0
    else:
        calibration_days = reserved[:split]
    inputs = [ds for ds, _, _, _ in sources]
    primary = max(range(2), key=lambda i: float(np.diff(inputs[i].lat)[0] * np.diff(inputs[i].lon)[0]))

    def read_day(date):
        daily = []
        for ds, field, dates, _ in sources:
            low, high = np.searchsorted(dates, [date, date + day])
            if low < high:
                converted = daily_precipitation(ds.isel(time=slice(low, high)), field)
            else:
                shape = (1, ds.sizes["lat"], ds.sizes["lon"])
                field_attrs = dict(units="mm", physical_variable="precipitation", data_kind="continuous",
                                   temporal_semantics="accumulation", spatial_semantics="cell_mean")
                converted = xr.Dataset(
                    {"precipitation": (("time", "lat", "lon"), np.full(shape, np.nan), field_attrs)},
                    coords=dict(time=[date], lat=ds.lat, lon=ds.lon), attrs=dict(crs=ds.attrs["crs"]))
            daily.append(converted)
        return daily

    fixed = task.get("target_grid")
    if fixed:
        target = resolve_target_grid(inputs, fixed)
        alignment = dict(selection_basis="DECLARED_TARGET_GRID", target_grid=fixed)
    else:
        n = len(calibration_days)
        comparison = calibration_days[np.linspace(0, n - 1, min(4, n), dtype=int)] if n else []
        calibration = [xr.concat(datasets, dim="time") for datasets in zip(*(read_day(date) for date in comparison))]
        _, target, alignment = choose_grid(calibration or inputs, primary, "precipitation", "conservative",
                                          comparison_times=comparison)

    def read_aligned(date):
        return [remap(ds, target, "precipitation", "conservative") for ds in read_day(date)]

    return reserved[split:], primary, alignment, read_aligned


def freeze_holdout(scan, task):
    import numpy as np
    import xarray as xr
    from .holdout_plan import exposures, time_boundary
    from .sampling_plan import time_dimension

    groups, cutoffs = [], []
    explicit = []
    for index, name in task["fields"]:
        row = scan["rows"][index]
        field = next(item for item in row["fields"] if item["name"] == name)
        explicit.append(row.get("precipitation_holdout_days", ()))
        with local_member(row) as path, xr.open_dataset(path, cache=False) as dataset:
            axis = time_dimension(dataset[name], dataset)
            dates = dataset[axis].values.astype("datetime64[ns]")
            boundary = time_boundary(exposures(scan, index, field), axis)
            if boundary >= len(dates):
                return {"days": []}
            lower = dates[boundary].astype("datetime64[D]")
            if dates[boundary] != lower:
                lower += np.timedelta64(1, "D")
            groups.append(dates.astype("datetime64[D]"))
            cutoffs.append(lower)
    shared = np.intersect1d(*groups)
    shared = shared[shared >= max(cutoffs)]
    if explicit[0] and explicit[0] == explicit[1]:
        shared = np.intersect1d(shared, np.asarray(explicit[0], dtype="datetime64[D]"))
    return {"days": [str(date) for date in shared]}


def review(scan, task_id, task, steps=(), *, independent=False):
    from contextlib import ExitStack
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with ExitStack() as stack:
        coverage = {}
        if not independent and (plan := task.get("locked_holdout")) is not None:
            if freeze_holdout(scan, task) != plan:
                raise ValueError("锁定的 UTC 日回检范围与当前外发记录不一致")
        if task.get("cf_daily"):
            reserved, primary, alignment, read_aligned = indexed_holdout(scan, task, stack, independent=independent)
        else:
            folder = stack.enter_context(TemporaryDirectory())
            adapted, _, _, aligned, _, alignment, primary, common, reserved = prepare(
                scan, task_id, task, Path(folder) / "adapted")
            coverage = dict(common_range_days=len(common), source_day_counts=[
                item["dataset"].sizes["time"] for item in adapted])

            def read_aligned(date):
                return [ds.sel(time=[date]) for ds in aligned]
        if not len(reserved):
            return dict(status="pending", basis="no_shared_holdout_days", count=0, checks=[],
                        population=0, unit="withheld_utc_day", date_coverage=coverage)
        return review_recipe(reserved, read_aligned, primary, alignment, steps,
                             scoring_request=task.get("scoring_request"))


def review_recipe(reserved, read_aligned, primary, alignment, steps, *, scoring_request=None):
    import numpy as np
    from .grid_preparation import trial_recipe
    from .sampling_risk import evaluate_recipe, recipe_settings

    count = filled = unresolved = 0
    scores = []
    def check_day(index):
        nonlocal count, filled, unresolved
        date = reserved[index]
        output, evidence, metrics, failure = trial_recipe(read_aligned(date), primary, steps,
                                                          scoring_request=scoring_request)
        count += int(np.isfinite(output.precipitation).sum())
        filled += evidence["filled"]
        unresolved += evidence["unresolved"]
        if metrics:
            scores.append(dict(day=str(date), **metrics))
        if failure:
            return dict(unit=str(date), rule=failure, unscored=failure == "insufficient_score_support")
        return None

    report = evaluate_recipe(len(reserved), "withheld_utc_day", check_day, **recipe_settings())
    failures = {row["rule"] for row in report["failures"]}
    if report["status"] == "checked" and "spatial.blend" in steps and not scores:
        basis = "recipe_coverage_budget_insufficient"
    elif report["status"] == "checked":
        basis = "joint_recipe_acceptance_passed"
    elif report["status"] == "pending":
        basis = "recipe_scoring_reference_insufficient" if report["unscored"] else "recipe_coverage_budget_insufficient"
    elif failures and failures <= {"recipe_score_below_single_source", "both_sources_missing_on_holdout"}:
        basis = "recipe_numeric_rejected"
    else:
        basis = "joint_recipe_acceptance_rejected"
    return dict(status="pending" if basis == "recipe_coverage_budget_insufficient" else report["status"],
                basis=basis, checks=["fixed_alignment_gap_fusion_trial"], count=count,
                sampled_days=report["inspected"], gap_filling=dict(filled=filled, unresolved=unresolved),
                acceptance_sampling=report, sampled_scores=scores, fusion_score_groups=len(scores), alignment=alignment,
                reason="部分留出 UTC 日缺少可比较的双源网格或评分参考。"
                       if basis == "recipe_scoring_reference_insufficient" else "")


def run(scan, tasks, folder, progress=None, *, model_id=None):
    import numpy as np
    import xarray as xr
    from ecodata.core.exceptions import EcoDataError
    from .grid_preparation import fill
    from .service import bundle
    from .result_summary import ranked_candidates, seed_usage

    assert_current(scan)
    folder.mkdir(parents=True, exist_ok=True)
    files, artifacts, reports = [], [], []
    searches={}
    for number, (task_id, task) in enumerate(tasks, 1):
        if progress:
            progress(f"正在处理 {number}/{len(tasks)} · {task_id}")
        adapted, conversion, grids, aligned, target, alignment, primary, common, _ = prepare(
            scan, task_id, task, folder / f"pair-{number}")
        target_spec = task.get("target_grid")
        has_gaps = any(not np.isfinite(ds.precipitation).all() for ds in aligned)
        fusion_requested = task.get("route") == "independent_grid_search"
        gap_report = None
        fallback_only = not any("spatial.blend" in recipe for recipe in task.get("recipes", ()))
        if fusion_requested and (has_gaps or fallback_only):
            output, gap_report = fill(aligned, primary, "precipitation")
            outputs = (("filled-primary", output),)
        else:
            delivery_inputs = grids if target_spec and fusion_requested else aligned
            outputs = tuple((adapted[i]["product"], delivery_inputs[i]) for i in (1 - primary, primary))
        if progress:
            progress("已完成保守重网格对齐；" + ("已保留原值并用备用源填补可用缺测" if gap_report else "已核对各源有效覆盖"))
        paired = aligned[0].time.where(
            np.isfinite(aligned[0].precipitation).all(("lat", "lon"))
            & np.isfinite(aligned[1].precipitation).all(("lat", "lon")), drop=True)
        can_search = fusion_requested and not fallback_only and (not has_gaps or len(paired) >= 3)
        search = None
        if fusion_requested and not fallback_only and not can_search:
            gap_report["fusion_status"] = "没有足够的双源完整日，保留补位结果"
        if can_search:
            from ecodata.algorithms.platform_search import SearchInput
            from .local_search import pareto_comparison, submit
            from .service import fusion_evidence
            inputs=[]
            search_inputs = (tuple((adapted[i]["product"], aligned[i].sel(time=paired))
                                   for i in (primary, 1 - primary)) if has_gaps else tuple(reversed(outputs)))
            for i,(label,ds) in enumerate(search_inputs):
                field=ds[["precipitation","time_bounds"]].copy(deep=True)
                field["precipitation"]=ds.precipitation.transpose("time","lat","lon")
                field=field.rename({field.time_bounds.dims[1]:"bounds"}) if field.time_bounds.dims[1]!="bounds" else field
                field.attrs.update(source_id=label)
                field.precipitation.attrs.update(physical_variable="precipitation",
                    physical_definition="UTC daily accumulated precipitation on common grid")
                field.time.encoding["units"]="seconds since 1970-01-01"
                field.time_bounds.encoding["units"]="seconds since 1970-01-01"
                inputs.append(SearchInput(f"source-{i}.nc","primary" if i==0 else "background",bytes(field.to_netcdf(engine="scipy"))))
            request=dict(task="multisource",variable="precipitation",unit="mm",
                physical_variable="precipitation",semantic_type="continuous",temporal_semantics="accumulation",
                nonnegative=True,processing_goal="spatial_fusion",initial_algorithms=task["seeds"])
            from ecodata.algorithms.platform_search import with_recommendations
            request=with_recommendations(request,[dict(id=str(i),task=task_id,
                steps=[op for op in recipe if op != "fusion.series_missing_fallback"],
                scope=dict(variable='precipitation'))
                for i,recipe in enumerate(task['recipes']) if "spatial.blend" in recipe])
            request['completed_preparation']=[] if target_spec and not has_gaps else [
                dict(operation='spatial.regrid.conservative', status='PASS', record=alignment)]
            if target_spec and not has_gaps:request['target_grid']=target_spec
            search_folder=folder/f"search-{number}"
            try:
                search=submit(inputs,request,search_folder,model_id=model_id,daily_amount=True,progress=progress)
                if not search.get("product"):
                    raise ValueError("融合寻优未生成通过评分的产物")
                with xr.open_dataset(search["product"]) as ds:
                    selected = ds.load()
                if gap_report is not None:
                    if not all(np.array_equal(output[axis], selected[axis]) for axis in ("lat", "lon")):
                        raise ValueError("融合结果与已选工作网格不一致")
                    evaluated_days = selected.precipitation.sel(time=paired.values)
                    if not np.isfinite(evaluated_days).all():
                        raise ValueError("融合结果在双源完整日仍存在缺测")
            except (EcoDataError, OSError, ValueError, RuntimeError, KeyError) as error:
                if gap_report is None:
                    raise
                gap_report["fusion_status"] = str(error)
                search = None
            if search:
                if gap_report is not None:
                    output.precipitation.loc[dict(time=paired.values)] = evaluated_days
                    output["spatial_search_applied"] = ("time", np.isin(common, paired.values).astype("int8"))
                    gap_report["paired_days_evaluated"] = len(paired)
                else:
                    outputs=(("selected",selected),)
        if search:
            rankings=ranked_candidates(search_folder,search)
            import json
            evidence=dict(evaluation_protocol=search.get('evaluation_protocol'),
                selection_comparison=search.get('selection_comparison'),post_selection_assessment=search.get('post_selection_assessment'),
                evaluation_cost=search.get('evaluation_cost'),total_cost=search.get('total_cost'))
            files.append((f'pair-{number}-fusion-evaluation.json',json.dumps(evidence,ensure_ascii=False,default=str).encode('utf8')))
            for predictions in sorted(search_folder.glob('*-evaluation.npz')):
                files.append((f'pair-{number}-evaluation-{predictions.name}',predictions.read_bytes()))
            searches[task_id]=dict(evidence=fusion_evidence(search),best=search["best"],
                search_folder=str(search_folder),
                recommendation_validation=search.get('recommendation_validation',[]),
                seed_usage=seed_usage(search_folder,search),
                confirmed_initial=search["confirmed_initial"],selected_graph=search["selected_graph"],
                rankings=rankings,pareto_comparison=pareto_comparison(rankings),
                portrait_strategy=search.get("portrait_strategy"),soft_guidance=search["soft_guidance"])
        for label,ds in outputs:
            export=ds.copy(deep=True)
            if "time_bounds" not in export:
                export["time_bounds"] = grids[primary].time_bounds
            from ecodata.spatial.regridding import require_target_grid
            if target_spec:require_target_grid(export,target,"precipitation")
            export["precipitation"]=ds.precipitation.transpose("time","lat","lon")
            export.attrs.update(product=label,time_basis="UTC")
            export.time.attrs["bounds"] = "time_bounds"
            export.time.encoding["units"] = export.time_bounds.encoding["units"] = "seconds since 1970-01-01"
            name=f"pair-{number}-{label}.nc"
            raw=bytes(export.to_netcdf(engine="scipy"))
            with xr.open_dataset(io.BytesIO(raw),engine="scipy") as reopened:
                xr.testing.assert_allclose(export.precipitation,reopened.precipitation)
                if "source_availability" in export:
                    xr.testing.assert_equal(export.source_availability, reopened.source_availability)
            files.append((name,raw))
            artifacts.append(dict(name=name,dimensions=dict(export.sizes),fields=list(export.data_vars),units={"precipitation":"mm"}))
        reports.append(dict(task=task_id, conversion=conversion, alignment=alignment,
                            gap_filling=gap_report, fusion_executed=task_id in searches, utc_dates=len(common)))
    assert_current(scan)
    result=dict(mode="independent_grid_search" if searches else "independent_alignment",source_formats=["NetCDF"],target_format="NetCDF",
                files=files,search=None,original_report=reports,grid_searches=searches,
                local=dict(artifacts=artifacts,native_ready=False,
                           reason=("按核验的 UTC 日界对齐。缺测时保留主源有效值并用备用源补位；两源都缺的位置保持缺失。"
                                   if any(report["gap_filling"] for report in reports) else
                                   "按核验的 UTC 日界和目标网格对齐；网格选择和融合评分的参考依据见处理记录。")))
    bundle(result)
    return result

