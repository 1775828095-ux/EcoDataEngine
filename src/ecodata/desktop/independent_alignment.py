"""Independent precipitation conversion and grid alignment."""
import hashlib
import io

from .scan import assert_current, read_selected


def prepare(scan, task_id, task, folder):
    import numpy as np
    import pandas as pd
    from ecodata.models.engineering.precipitation_adaptation import adapt_products
    from ecodata.models.engineering.precipitation_products import inspect_product
    from ecodata.recognition import recognise_file
    from .grid_preparation import daily_precipitation

    sources, audits, adapted, conversion = [], [], [], []
    hashes, reserved_from = set(), []
    for index, field in task["fields"]:
        row = scan["rows"][index]
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
        if task.get("cf_daily"):
            adapted.append(dict(product=f"source-{len(adapted) + 1}", dataset=daily_precipitation(ds, field)))
            conversion.append(dict(source=row["name"], field=field, source_sha256=digest,
                                   operation="UTC_interval_daily_aggregation"))
            continue
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
    from .grid_preparation import choose_grid, remap

    start = max(ds.time.values[0] for ds in grids)
    end = min(ds.time.values[-1] for ds in grids)
    if start > end:
        raise ValueError(f"{task_id}：没有共同 UTC 日期范围")
    day = np.timedelta64(1, "D")
    common = np.arange(start.astype("datetime64[D]"), end.astype("datetime64[D]") + day, day)
    reserved = common[common >= max(reserved_from)] if len(reserved_from) == len(grids) else common[:0]
    split = len(reserved) // 2
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
    else:
        _, target, alignment = choose_grid(grids, primary, "precipitation", "conservative",
                                          comparison_times=reserved[:split])
    aligned = [remap(ds, target, "precipitation", "conservative") for ds in grids]
    return adapted, conversion, grids, aligned, target, alignment, primary, common, reserved[split:]


def review(scan, task_id, task):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import numpy as np
    from .grid_preparation import fill
    from .sampling_risk import evaluate, settings

    with TemporaryDirectory() as folder:
        _, _, _, aligned, _, alignment, primary, _, reserved = prepare(
            scan, task_id, task, Path(folder) / "adapted")
    if not len(reserved):
        return dict(status="pending", basis="no_disjoint_partition", count=0)
    count = filled = unresolved = 0
    def check_day(index):
        nonlocal count, filled, unresolved
        date = reserved[index]
        output, evidence = fill([ds.sel(time=[date]) for ds in aligned], primary, "precipitation")
        count += int(np.isfinite(output.precipitation).sum())
        filled += evidence["filled"]
        unresolved += evidence["unresolved"]
        if evidence["unresolved"]:
            return dict(unit=str(date), rule="both_sources_missing_on_holdout")
        return None

    report = evaluate(len(reserved), "withheld_utc_day", check_day, **settings())
    return dict(status=report["status"],
                basis="joint_aligned_gap_recheck" if not unresolved else "both_sources_missing_on_holdout",
                checks=["aligned_two_source_holdout", "valid_primary_values_preserved"], count=count,
                sampled_days=report["inspected"], gap_filling=dict(filled=filled, unresolved=unresolved),
                acceptance_sampling=report, alignment=alignment)


def run(scan, tasks, folder, progress=None, *, model_id=None):
    import numpy as np
    import xarray as xr
    from ecodata.core.exceptions import EcoDataError
    from .grid_preparation import fill
    from .service import bundle
    from .result_summary import ranked_candidates

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
            request=with_recommendations(request,[dict(id=str(i),task=task_id,steps=recipe,scope=dict(variable='precipitation'))
                for i,recipe in enumerate(task['recipes'])])
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

