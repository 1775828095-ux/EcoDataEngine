"""Desktop bridge to the teammate's native composite reference adapter."""
import hashlib
import io
import json

from .scan import assert_current, read_selected


ALIGNMENT = "reference.native_grid_support"


def is_surface_mcip(row, field):
    return (field["name"].casefold()=="lai" and row.get("ioapi_grid_role")=="interior"
            and field.get("dimensions")==["TSTEP","LAY","ROW","COL"]
            and row.get("dimensions",{}).get("LAY")==1
            and row.get("time_spacing_basis")=="ioapi_timestep_and_tflag")


def plan(scan, pair, model_id):
    from pathlib import PurePosixPath

    if model_id != "cmaq":
        return None
    left,right=pair[:2]
    if {left[2]["name"].casefold(),right[2]["name"].casefold()}!={"lai","lai_500m"}:
        return None
    native,satellite=(left,right) if left[2]["name"].casefold()=="lai" else (right,left)
    rows=scan["rows"]
    met,sat=rows[native[0]],rows[satellite[0]]
    if not is_surface_mcip(met,native[2]) or sat.get("native_product")!="MCD15A3H":
        return None
    quality={"Lai_500m_raw","FparLai_QC_raw","FparExtra_QC_raw"}
    if not quality<={f["name"] for f in sat["fields"]}:
        return None
    clock,composite=met.get("time_coverage"),sat.get("time_coverage")
    if not clock or not composite or not composite[0]<=clock[0]<=clock[1]<composite[1]:
        return None
    signature=met.get("ioapi_grid_signature")
    if signature is None:
        return None
    grids=[i for i,row in enumerate(rows) if row.get("ioapi_grid_role")=="interior"
           and row.get("ioapi_grid_signature")==signature
           and {"LAT","LON"}<={f["name"] for f in row["fields"]}
           and {tuple(f.get("dimensions",[])) for f in row["fields"] if f["name"] in {"LAT","LON"}}
               =={("TSTEP","LAY","ROW","COL")}
           and row.get("dimensions",{}).get("TSTEP")==row.get("dimensions",{}).get("LAY")==1]
    local=[i for i in grids if PurePosixPath(rows[i]["name"]).parent==PurePosixPath(met["name"]).parent]
    grids=local or grids
    if len(grids)!=1 or max(met["bytes"],sat["bytes"],rows[grids[0]]["bytes"])>64*1024**2:
        return None
    return dict(action="fusion",route="native_reference_search",native=native[0],
                satellite=satellite[0],grid=grids[0],variable=native[2]["name"])


def run(scan,tasks,folder,progress=None,*,model_id=None):
    import numpy as np
    import xarray as xr
    from ecodata.algorithms.platform_search import SearchInput
    from ecodata.models.native_reference import fusion_inputs, reference_profile
    from .local_search import submit, pareto_comparison
    from .result_summary import ranked_candidates
    from .service import bundle, fusion_evidence

    assert_current(scan)
    files,artifacts,reports,searches=[],[],[],{}
    for number,(task_id,task) in enumerate(tasks,1):
        if progress:
            progress(f"{task_id} · 核验卫星质量、合成窗口和 CMAQ 网格")
        stage=folder/f"reference-{number}"
        stage.mkdir(parents=True)
        datasets={}
        sources={}
        for name in ("satellite","native","grid"):
            row=scan["rows"][task[name]]
            raw=read_selected(row,limit=64*1024**2)
            if len(raw)!=row["bytes"]:
                raise ValueError("工程成员大小已变化，请重新扫描")
            sources[name]=hashlib.sha256(raw).hexdigest()
            with xr.open_dataset(io.BytesIO(raw),decode_times=False) as ds:
                datasets[name]=ds.load()
        sat,met,grid=(datasets[name] for name in ("satellite","native","grid"))
        profile=reference_profile(sat)
        if profile is None:
            raise ValueError("卫星数据缺少原版参考产品契约")
        variable=task["variable"]
        prepared,ys,xs,background,times=fusion_inputs(
            sat,profile,met,grid,variable,sources["satellite"],stage)
        inputs=[SearchInput(f"{role}.nc",role,bytes(ds.to_netcdf(engine="scipy")))
                for role,ds in prepared]
        request=dict(task="multisource",processing_goal="spatial_fusion",variable=variable,
            unit=profile["unit"],physical_variable=profile["quantity"],semantic_type="continuous",
            temporal_semantics="instantaneous",nonnegative=True,allow_training=True,
            initial_algorithms=task["seeds"],mutation_control="random",observation_crs="EPSG:4326")
        if progress:
            progress(f"{task_id} · 原版变异寻优，比较单源与融合")
        search_dir=stage/"search"
        search=submit(inputs,request,search_dir,model_id=model_id,progress=progress)
        if search.get("status")!="PASS" or not search.get("product"):
            raise ValueError("原版搜索未生成可交付产品")
        with xr.open_dataset(search["product"]) as ds:
            values=ds[variable].values[0]
        # Preserve native hourly anomalies using the original constant-increment policy.
        updated=background.copy()
        updated[:,0,ys,xs]+= (values-background[0,0,ys,xs])[None,:]
        if not np.isfinite(updated).all() or (updated<0).any() or (updated>10).any():
            raise ValueError("参考增量导致 LAI 超出有效范围，拒绝导出该产物")
        prepared_field=xr.Dataset(
            {variable:(("time","row","col"),updated[:,0],{"units":profile["unit"],
              "standard_name":profile["quantity"]})},
            coords={"time":times,"lat":(("row","col"),grid.LAT.values.reshape(updated.shape[-2:])),
                    "lon":(("row","col"),grid.LON.values.reshape(updated.shape[-2:]))},
            attrs={"reference_policy":profile["policy"]})
        name=f"reference-{number}-LAI.nc"
        raw=bytes(prepared_field.to_netcdf(engine="scipy"))
        with xr.open_dataset(io.BytesIO(raw),engine="scipy") as reopened:
            xr.testing.assert_allclose(prepared_field,reopened.load())
        files.append((name,raw))
        artifacts.append(dict(name=name,dimensions=dict(prepared_field.sizes),fields=[variable],
                              units={variable:profile["unit"]}))
        support=json.loads((stage/"reference_support.json").read_text(encoding="utf8"))
        reports.append(dict(task=task_id,source_sha256=sources,support=support,
                            optimizer=search["optimizer"],comparison=search.get("selection_comparison")))
        rankings=ranked_candidates(search_dir,search)
        searches[task_id]=dict(evidence=fusion_evidence(search),best=search["best"],
            confirmed_initial=search["confirmed_initial"],selected_graph=search["selected_graph"],
            rankings=rankings,pareto_comparison=pareto_comparison(rankings),
            portrait_strategy=search.get("portrait_strategy"),soft_guidance=search["soft_guidance"])
    assert_current(scan)
    result=dict(mode="native_reference_search",source_formats=["NetCDF"],target_format="NetCDF",
                files=files,search=None,original_report=reports,grid_searches=searches,
                local=dict(artifacts=artifacts,native_ready=False,
                    reason="卫星 QA 与尺度适配后在共同支持域试跑，保留原始逐小时变化。评分以卫星格点代理为参考。"))
    bundle(result)
    return result
