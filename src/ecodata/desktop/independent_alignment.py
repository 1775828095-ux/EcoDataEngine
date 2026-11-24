"""Independent precipitation conversion and grid alignment."""
import hashlib
import io

from .scan import assert_current, read_selected


def run(scan, tasks, folder, progress=None, *, model_id=None):
    import numpy as np
    import xarray as xr
    from ecodata.core.schemas import EcoDataBundle
    from ecodata.models.engineering.precipitation_adaptation import adapt_products
    from ecodata.models.engineering.precipitation_products import inspect_product
    from ecodata.recognition import recognise_file
    from ecodata.spatial.regridding import GridRegridder, RegridConfig
    from .service import bundle
    from .result_summary import ranked_candidates

    assert_current(scan)
    folder.mkdir(parents=True, exist_ok=True)
    files, artifacts, reports = [], [], []
    searches={}
    for number, (task_id, task) in enumerate(tasks, 1):
        if progress:
            progress(f"正在处理 {number}/{len(tasks)} · {task_id}")
        sources, audits = [], []
        for index, field in task["fields"]:
            row = scan["rows"][index]
            raw = read_selected(row, limit=32*1024**2)
            ds, _ = recognise_file(row["name"], raw, record_read_source=False)
            product = inspect_product(ds)
            if not product or not any(f["field"]==field for f in product["fields"]):
                raise ValueError(f"{task_id}：未能核验降水产品")
            digest = hashlib.sha256(raw).hexdigest()
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
        adapted, conversion = adapt_products(sources,audits,folder/f"pair-{number}",[])
        if len(adapted)!=2 or not all(entry.get("product") for entry in conversion):
            raise ValueError(f"{task_id}：原版日累计转换未生成两份产品")
        grids = [entry["dataset"] for entry in adapted]
        common = np.intersect1d(grids[0].time.values,grids[1].time.values)
        if not common.size:
            raise ValueError(f"{task_id}：没有共同 UTC 日期")
        grids = [ds.sel(time=common) for ds in grids]
        # Product inspection above verifies geographic coordinate units. The datum
        # remains an explicit working convention, not a verified source datum.
        for ds in grids:
            if not ds.attrs.get('crs'):
                ds.attrs.update(crs='EPSG:4326',crs_status='WORKING_ASSUMPTION',
                    crs_evidence='Inspected product geographic axes; original datum not independently established')
        for ds in grids:
            for axis in ("lat","lon"):
                steps=np.diff(ds[axis].values)
                if not len(steps) or not np.all(steps>0) or not np.allclose(steps,steps[0]):
                    raise ValueError(f"{task_id}：需要规则经纬网格")
        target_spec=task.get('target_grid')
        if target_spec:
            from ecodata.spatial.regridding import resolve_target_grid,require_target_grid
            target=resolve_target_grid(grids,target_spec)
            # Reference source is selected by native cell size, never errors.
            target_index=max(range(2),key=lambda i:float(np.diff(grids[i].lat)[0]*np.diff(grids[i].lon)[0]))
            outputs=((adapted[1-target_index]['product'],grids[1-target_index]),(adapted[target_index]['product'],grids[target_index]))
            transformed=None
            if task.get('route')!='independent_grid_search':
                mapped=[]
                for label,ds in outputs:
                    item=GridRegridder(RegridConfig('conservative',tuple(target.lat.values),tuple(target.lon.values),'precipitation')).apply(EcoDataBundle(ds))
                    out=item.data.dataset;out['time_bounds']=ds.time_bounds
                    require_target_grid(out,target,'precipitation');mapped.append((label,out))
                outputs=tuple(mapped)
        else:
            steps = [tuple(float(np.diff(ds[a].values)[0]) for a in ("lat","lon")) for ds in grids]
            target_index = next((i for i in (0,1) if all(x>=y for x,y in zip(steps[i],steps[1-i]))), None)
            if target_index is None:
                raise ValueError(f"{task_id}：两轴粗细交叉，须明确目标网格")
            target = grids[target_index]
            source = grids[1-target_index]
            # Keep only coarse cells wholly covered by the source grid.
            for axis in ("lat","lon"):
                axis_index=("lat","lon").index(axis)
                half=steps[target_index][axis_index]/2
                source_half=steps[1-target_index][axis_index]/2
                mask=(target[axis]-half>=float(source[axis][0])-source_half-1e-8)&(target[axis]+half<=float(source[axis][-1])+source_half+1e-8)
                target=target.isel({axis:np.flatnonzero(mask.values)})
            config=RegridConfig("conservative",tuple(target.lat.values),tuple(target.lon.values),"precipitation")
            # Clip to a small halo around the target before checking for missing cells.
            for axis in ("lat","lon"):
                values=target[axis].values
                if len(values)<2:
                    raise ValueError(f"{task_id}：共同区域不足两行两列")
                halo=steps[target_index][("lat","lon").index(axis)]
                source=source.sel({axis:slice(float(values[0])-halo,float(values[-1])+halo)})
            if not all(np.isfinite(ds.precipitation.values).all() for ds in (source,target)):
                raise ValueError(f"{task_id}：目标区域有缺测，不自动填补")
            transformed=GridRegridder(config).apply(EcoDataBundle(source))
            aligned=transformed.data.dataset
            if not np.isfinite(aligned.precipitation.values).all():
                raise ValueError(f"{task_id}：对齐输出存在缺测")
            outputs=((adapted[1-target_index]["product"],aligned),(adapted[target_index]["product"],target))
        if task.get("route")=="independent_grid_search":
            from ecodata.algorithms.platform_search import SearchInput
            from .local_search import pareto_comparison, submit
            from .service import fusion_evidence
            inputs=[]
            for i,(label,ds) in enumerate(reversed(outputs)):
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
            request['completed_preparation']=[] if target_spec else [dict(operation='spatial.regrid.conservative',status='PASS',record=transformed.record.as_dict())]
            if target_spec:request['target_grid']=target_spec
            search_folder=folder/f"search-{number}"
            search=submit(inputs,request,search_folder,model_id=model_id,daily_amount=True,progress=progress)
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
            with xr.open_dataset(search["product"]) as ds:
                outputs=(("selected",ds.load()),)
        for label,ds in outputs:
            export=ds.copy(deep=True) if target_spec else target[["precipitation","time_bounds"]].copy(deep=True)
            if target_spec:require_target_grid(export,target,"precipitation")
            export["precipitation"]=ds.precipitation.transpose("time","lat","lon")
            export.attrs.update(product=label,time_basis="UTC")
            name=f"pair-{number}-{label}.nc"
            raw=bytes(export.to_netcdf(engine="scipy"))
            with xr.open_dataset(io.BytesIO(raw),engine="scipy") as reopened:
                xr.testing.assert_allclose(export.precipitation,reopened.precipitation)
            files.append((name,raw))
            artifacts.append(dict(name=name,dimensions=dict(export.sizes),fields=["precipitation"],units={"precipitation":"mm"}))
        reports.append(dict(task=task_id,conversion=conversion,method=transformed.record.as_dict() if transformed else dict(target_grid=target_spec,alignment="Public spatial task before candidate evaluation"),
                            target_product='explicit_target_grid' if target_spec else adapted[target_index]["product"],utc_dates=len(common)))
    assert_current(scan)
    result=dict(mode="independent_grid_search" if searches else "independent_alignment",source_formats=["NetCDF"],target_format="NetCDF",
                files=files,search=None,original_report=reports,grid_searches=searches,
                local=dict(artifacts=artifacts,native_ready=False,
                           reason=("按共同 UTC 日期和目标网格执行；本次以所有来源变粗后的原细产品重建误差选图，粗主产品误差仅作诊断，独立实测精度仍未验证。"
                                   if searches and any(s['evidence']['reference_basis']=='REDUCED_SCALE_PRODUCT_RECONSTRUCTION' for s in searches.values()) else "按共同 UTC 日期和明确目标细网格执行全部候选；以原始粗参考像元回聚合代理误差选图，未验证细尺度精度。"
                                   if searches and any(t.get("target_grid") for _,t in tasks) else "按共同 UTC 日期和明确目标细网格分别导出对齐产品。"
                                   if any(t.get("target_grid") for _,t in tasks) and not searches else "按共同 UTC 日期和较粗网格运行原版搜索，导出代理验证评分选中的方案。"
                                   if searches else "按共同 UTC 日期积分为日累计量，保守重网格到较粗产品，两来源分别导出。")))
    bundle(result)
    return result

