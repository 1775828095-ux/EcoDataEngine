"""Build and validate recommendations for the operations released on page one."""
from copy import deepcopy
from math import isfinite
from secrets import randbelow

from .scan import assert_current, read_selected
from .service import ALGORITHMS
from .pair_processing import REGISTERED_SPACE, SPATIAL_SEEDS

ENGINEERING = {
    "engineering.precipitation_adaptation": "原版降水产品适配：核对统计区间、积分为日累计量并保留原始网格",
    "engineering.port_binding": "原版工程端口匹配：核对目标支持与统计日界",
    "engineering.port_search": "原版端口变异寻优：工程端口和验证条件满足后运行",
}
ALIGNMENT = {
    "spatial.regrid.conservative": "原版保守重网格：按覆盖面积将单元均值或通量对齐到较粗网格",
    "raster_task.spatial.bilinear": "双线性重采样",
    "raster_task.spatial.nearest": "最近邻重采样",
    "temporal.semantic": "原版 SemanticTimeResampler，按时间统计语义重采样",
    "temporal.semantic.nearest": "原版 SemanticTimeResampler，瞬时值按最近时间重采样",
    "temporal.wide_table_interpolation": "原版宽表时间插值，按站点处理并保留 QC 标识",
    REGISTERED_SPACE: "原版坐标转换与真实位置采样",
    "fusion.uncertainty_weighted.nearest": "投影最近邻对齐后按已声明的不确定度作反方差加权",
    "fusion.uncertainty_weighted.bilinear": "投影双线性对齐后按已声明的不确定度作反方差加权",
    "reference.native_grid_support": "原版卫星 QA 筛选、投影转换与目标尺度面积聚合",
}

ALGORITHM_NAMES = {
    "spatial.regrid.conservative": "保守重网格",
    "spatial.blend": "加权融合",
    "spatial.idw": "反距离权重插值（IDW）",
    "spatial.residual_idw": "残差反距离权重融合",
    "raster_task.spatial.bilinear": "双线性重采样",
    "raster_task.spatial.nearest": "最近邻重采样",
    "temporal.semantic": "按时间语义重采样",
    "temporal.semantic.nearest": "时间最近邻重采样",
    "temporal.wide_table_interpolation": "站点表时间线性插值",
    REGISTERED_SPACE: "坐标归一与位置对齐",
    "fusion.uncertainty_weighted.nearest": "最近邻对齐 + 反方差加权",
    "fusion.uncertainty_weighted.bilinear": "双线性对齐 + 反方差加权",
    "reference.native_grid_support": "卫星质量筛选与网格聚合",
    "engineering.precipitation_adaptation": "降水产品适配",
    "engineering.port_binding": "工程端口匹配",
    "engineering.port_search": "工程端口变异寻优",
}

def recipe_name(steps):
    return " + ".join(ALGORITHM_NAMES.get(step,step) for step in steps)


def attach_samples(scan, pairs):
    from ecodata.recognition import recognise_file
    datasets={}
    used_bytes=0
    samples={}
    for pair in pairs:
        for field in pair.get("pair_details",[]):
            if not isinstance(field,dict):
                continue
            fid=field.get("id","")
            field.pop("perturbed_samples",None)
            if fid in samples:
                field["perturbed_samples"]=samples[fid]
                continue
            try:
                source,variable=fid.split(".")
                index=int(source[1:])-1
                offset=int(variable[1:])-1
                if min(index,offset)<0 or fid not in pair["fields"]:
                    raise ValueError("invalid field")
                row=scan["rows"][index]
                name=row["fields"][offset]["name"]
                if index not in datasets:
                    if used_bytes+row["bytes"]>32*1024**2:
                        raise ValueError("sample budget")
                    used_bytes+=row["bytes"]
                    datasets[index]=recognise_file(row["name"],read_selected(row),record_read_source=False)[0]
                values=datasets[index][name]
                if values.dtype.kind not in "iuf":
                    raise ValueError("not numeric")
                # Select a bounded slice, then add an independent decimal to each real sample.
                indexers={dim:0 for dim in values.dims[:-1]}
                if values.dims:
                    indexers[values.dims[-1]]=slice(0,8)
                sample=values.isel(indexers).values.flat
                field["perturbed_samples"]=[round(float(v)+(randbelow(999)+1)/1000,6)
                                            for v in list(sample)[:8] if isfinite(float(v))][:4]
                samples[fid]=field["perturbed_samples"]
            except (ValueError,AttributeError,KeyError,IndexError,TypeError,OSError):
                field["perturbed_samples"]=[]


def make_request(scan, approved, checked, gated, *, local_checks=()):
    assert_current(scan)
    pairs={pair["id"]:pair for pair in approved["candidates"] if isinstance(pair,dict) and isinstance(pair.get("id"),str)}
    documents=approved.get("engineering_documents",[])
    documents=documents if isinstance(documents,list) else []
    tasks=[]
    for result in checked:
        cid,action=result["candidate"],result["action"]
        original=result.get("next_step")=="original_workbench"
        alignment_plan=result.get("next_step")=="alignment_then_fusion"
        task=gated.get((cid,action),{})
        reference=task.get("route")=="native_reference_search"
        advisory=(alignment_plan and not task) or result.get("next_step")=="recommendation_only"
        if result["status"]!="通过" and not original and not advisory and not reference:
            continue
        pair=deepcopy(pairs[cid])
        if reference:
            from .reference_search import ALIGNMENT as REFERENCE_ALIGNMENT
            recipes=[[REFERENCE_ALIGNMENT,method] for method in ("spatial.blend","spatial.residual_idw","spatial.idw")]
        elif advisory:
            recipes=[]
        elif task.get("route")=="pair_processing":
            recipes=([([time] if time else [])+[method,seed]
                      for time in task.get("time_methods",[None]) for method in task["methods"] for seed in SPATIAL_SEEDS]
                     if action=="fusion" else [[method] for method in task["methods"]])
            recipes.extend([method] for method in task.get("weighted_methods",[]))
        elif task.get("route")=="observation_gap_search":
            recipes=[[seed] for seed in ("spatial.idw","spatial.residual_idw")]
        elif task.get("route")=="independent_alignment":
            recipes=[["spatial.regrid.conservative"]]
        elif task.get("route")=="independent_grid_search":
            recipes=[["spatial.regrid.conservative","spatial.blend"]]
        elif original:
            if approved["target_model"]["id"]=="swat_plus":
                recipes=[["engineering.precipitation_adaptation","engineering.port_binding","engineering.port_search"]] if action=="fusion" else [["engineering.precipitation_adaptation","engineering.port_binding"]]
            else:
                recipes=[["engineering.port_binding"]]
        elif action=="fusion":
            recipes=[[algorithm] for algorithm in task["state"]["eligible"]]
        elif action=="align_time":
            recipes=[["temporal.semantic"]]
        else:
            recipes=[[method] for method in task["methods"] if method in ALIGNMENT]
        methods={**ALGORITHMS,**ENGINEERING,**ALIGNMENT}
        options=[dict(id=f"R{number}",name=recipe_name(steps),steps=steps,description=" → ".join(methods[k] for k in steps))
                 for number,steps in enumerate(recipes,1)]
        tasks.append(dict(id=f"{cid}:{action}",candidate=cid,operation=action,context=pair,
                          route=task.get("route","original_engineering" if original else "local"),allowed_recipes=options,
                          engineering_note=result["reason"] if alignment_plan or not advisory else result["block_code"],
                          evaluation_plan=(
                              dict(order=["align_resolution","evaluate_fusion"],
                                   alignment="比较适用的网格对齐算法，核对投影、空间支持、质量和有效覆盖",
                                   validation="仅用实际重叠、统计口径可比的样本，比较单源与融合的误差及偏差",
                                   output="按本地评分决定保留单源或采用融合")
                              if alignment_plan else None),
                          verified_checks=[record for record in local_checks if record["candidate"]==cid],
                          advisory_only=advisory))
    if not tasks:
        raise ValueError("没有通过首页核验的任务")
    request=dict(stage="recommendation",target_model=approved["target_model"],tasks=tasks,
                 engineering_documents=documents,
                 project_context_document_ids=approved.get("project_context_document_ids",[]),
                 engineering_inventory=approved.get("engineering_inventory",[]))
    assert_current(scan)
    return request

def validate(raw, summary):
    entries=raw.get("recommendations")
    if not isinstance(entries,list):
        raise ValueError("DeepSeek 未返回算法推荐列表")
    options={(task["id"],recipe["id"]):recipe for task in summary["tasks"] for recipe in task["allowed_recipes"]}
    advisory_tasks={task["id"]:task for task in summary["tasks"] if task.get("advisory_only")}
    recommendations=[]
    seen=set()
    for entry in entries:
        if not isinstance(entry,dict):
            continue
        key=(entry.get("task"),entry.get("recipe"))
        if not all(isinstance(k,str) for k in key) or key in seen:
            continue
        if key in options:
            seen.add(key)
            recommendations.append(dict(task=key[0],**options[key],reason=str(entry.get("reason",""))[:600]))
        elif key[0] in advisory_tasks and key[1]=="advisory":
            name=str(entry.get("name","")).strip()[:100]
            if not name:
                continue
            plan=advisory_tasks[key[0]].get("evaluation_plan")
            alignment=entry.get("alignment")
            fusion=entry.get("fusion_trial")
            if plan and not all(isinstance(part,str) and part.strip() for part in (alignment,fusion)):
                continue
            seen.add(key)
            description=(f"分辨率对齐 · {alignment.strip()[:240]}\n融合试跑 · {fusion.strip()[:240]}"
                         if plan else "条件性方案 · 本地执行条件待核验")
            recommendations.append(dict(task=key[0],id=key[1],name=name,steps=[],
                description=description,advisory_only=True,alignment_plan=bool(plan),
                reason=str(entry.get("reason",""))[:600]))
    missing={t["id"] for t in summary["tasks"]}-{r["task"] for r in recommendations}
    if missing:
        raise ValueError("以下任务未收到有效推荐，请重试："+", ".join(sorted(missing)))
    return recommendations
