"""Build and validate recommendations for the operations released on page one."""
from copy import deepcopy

from ecodata.core.exceptions import EcoDataError

from .scan import assert_current
from .service import ALGORITHMS
from .pair_processing import REGISTERED_SPACE, SPATIAL_SEEDS

ENGINEERING = {
    "engineering.precipitation_adaptation": "原版降水产品适配：核对统计区间、积分为日累计量并保留原始网格",
    "engineering.port_binding": "原版工程端口匹配：核对目标支持与统计日界",
    "engineering.port_search": "原版端口变异寻优：工程端口和验证条件满足后运行",
}
ALIGNMENT = {
    "spatial.regrid.conservative": "原版保守重网格：按覆盖面积将单元均值或通量对齐到目标网格",
    "fusion.series_missing_fallback": "保留主源观测，仅在缺测处用同区间的对齐备用源补位",
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

from .station_grid import NAMES as STATION_METHODS
from .statistical_preprocessing import METHODS as STATISTICAL_METHODS

PREPARATION = {
    "scale.semantic_stack":"各变量按自身支持域分别对齐，保留单位，不直接融合不同物理量",
    "units.harmonize":"原版物理单位换算",
    "format.normalize":"原版读取器转换为统一结构",
    "format.netcdf":"原版结构转换后导出 NetCDF",
    "format.parquet":"原版结构转换后导出 Parquet",
    "format.hdf5_netcdf":"原版结构转换后导出 HDF5-NetCDF",
    "format.excel":"原版结构转换后导出 Excel",
    "format.geojson":"原版矢量及坐标系转换后导出 GeoJSON",
    **STATION_METHODS,
    **STATISTICAL_METHODS,
}

ALGORITHM_NAMES = {
    "scale.semantic_stack":"跨变量语义尺度对齐",
    "spatial.source": "保留单源",
    **PREPARATION,
    "spatial.regrid.conservative": "保守重网格",
    "spatial.blend": "加权融合",
    "fusion.series_missing_fallback": "备用源逐缺口补位",
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


def is_private_fusion_candidate(row):
    return (row.get("task", "").endswith(":fusion")
            and row.get("final_verified_recommendation") is not True
            and not row.get("search_admission"))


def display_record(row):
    if is_private_fusion_candidate(row):
        return dict(task=row["task"], status=row.get("execution_release", {}).get("status", "pending"))
    return {key: value for key, value in row.items() if key != "local_recipe_options"}


def attach_samples(scan, pairs):
    from .holdout_plan import export_field

    samples={}
    for pair in pairs:
        for field in pair.get("pair_details",[]):
            fid=field.get("id","")
            if fid not in pair["fields"]:
                continue
            if fid not in samples:
                try:
                    source,variable=fid.split(".")
                    index=int(source[1:])-1
                    offset=int(variable[1:])-1
                    if min(index,offset)<0:
                        raise ValueError("invalid field")
                    row=scan["rows"][index]
                    selected=row["fields"][offset]
                    samples[fid]=export_field(scan,index,selected,phase="recommendation",limit=4).get("perturbed_samples",[])[:4]
                except (EcoDataError,OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,RuntimeError,ImportError):
                    samples[fid]=[]
            field["perturbed_samples"]=samples[fid]


def make_request(scan, approved, checked, gated, *, local_checks=(), include_samples=True, sampled_workflow=False):
    from .sampling_plan import recommendation_metadata
    from .discovery import engineering_fusion_targets
    from .recipe_trials import BLEND, FILL
    from .station_grid import BIAS
    assert_current(scan)
    pairs={pair["id"]:pair for pair in approved["candidates"] if isinstance(pair,dict) and isinstance(pair.get("id"),str)}
    documents=approved.get("engineering_documents",[])
    documents=documents if isinstance(documents,list) else []
    required = engineering_fusion_targets(approved)
    covered_requirements = set()
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
        elif task.get("recipe_steps"):
            recipes=task["recipe_steps"]
        elif task.get("route")=="pair_processing":
            recipes=([([time] if time else [])+[method,seed]
                      for time in task.get("time_methods",[None]) for method in task["methods"] for seed in SPATIAL_SEEDS]
                     if action=="fusion" else [[method] for method in task["methods"]])
            recipes.extend([method] for method in task.get("weighted_methods",[]))
            field_rows=[scan["rows"][index] for index,_ in task["fields"]]
            prep=[]
            if result.get("unit_check",{}).get("status")=="convertible":
                prep.append("units.harmonize")
            if len({row["format"] for row in field_rows})>1:
                prep.append("format.normalize")
            recipes=[[*prep,*recipe] for recipe in recipes]
        elif task.get("route")=="observation_gap_search":
            recipes=[[seed] for seed in ("spatial.idw","spatial.residual_idw")]
        elif task.get("route")=="independent_alignment":
            recipes=[["spatial.regrid.conservative"]]
        elif task.get("route")=="independent_grid_search":
            recipes=[["spatial.regrid.conservative","fusion.series_missing_fallback","spatial.blend"],
                     ["spatial.regrid.conservative","fusion.series_missing_fallback"]]
        elif original:
            if approved["target_model"]["id"]=="swat_plus":
                recipes=[["engineering.precipitation_adaptation","engineering.port_binding","engineering.port_search"]] if action=="fusion" else [["engineering.precipitation_adaptation","engineering.port_binding"]]
            else:
                recipes=[["engineering.port_binding"]]
            if action=='fusion':
                # Concrete existing estimators, not workflow verbs pretending
                # to be numerical nodes. Applicability is checked after binding.
                from ecodata.algorithms.spatial_fusion_task import COMPONENTS
                prefix=recipes[0][:-1] if recipes[0][-1]=='engineering.port_search' else recipes[0]
                recipes=[[*prefix,op] for op in ('spatial.blend','spatial.residual_idw','spatial.idw') if op in COMPONENTS]
        elif action=="fusion":
            recipes=[[algorithm] for algorithm in task["state"]["eligible"]]
        elif action=="align_time":
            recipes=[["temporal.semantic"]]
        else:
            recipes=[[method] for method in task["methods"] if method in ALIGNMENT]
        mandatory = action == "fusion" and cid in required
        if mandatory:
            complete = []
            for steps in recipes:
                if FILL not in steps:
                    continue
                split = steps.index(FILL)
                aligned = any(step.startswith(("temporal.", "spatial.regrid.", "raster_task.spatial.", "station.time."))
                              for step in steps[:split])
                if aligned and {BLEND, BIAS}.intersection(steps[split + 1:]):
                    complete.append(steps)
            recipes = complete
            if not recipes:
                continue
            covered_requirements.add(cid)
        methods={**ALGORITHMS,**ENGINEERING,**ALIGNMENT,**PREPARATION}
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
                          verified_checks=[recommendation_metadata(record) for record in local_checks
                                           if record["candidate"] == cid or (
                                               record.get("check") == "numeric_samples"
                                               and record.get("field") == record["candidate"]
                                               and record["field"] in pair["fields"])],
                          advisory_only=advisory, engineering_requires_fusion=mandatory))
    for cid in sorted(required - covered_requirements):
        reasons = [row["reason"] for row in checked if row["candidate"] == cid and row["action"] == "fusion"
                   and row.get("reason") and row["status"] != "通过"]
        note = "；".join(reasons) or "需要核实对应的两份输入、共同覆盖及整套试跑适配器。"
        tasks.append(dict(id=f"{cid}:fusion_requirement", candidate=cid, operation="fusion",
                          context=deepcopy(pairs.get(cid, dict(fields=[]))), allowed_recipes=[],
                          engineering_note=note, advisory_only=True, engineering_fusion_plan=True))
    if not tasks:
        raise ValueError("没有通过首页核验的任务")
    # Samples are added only to the candidate-specific recommendation request.
    if sampled_workflow:
        joint = [task for task in tasks if task["operation"] == "fusion" and task["allowed_recipes"]
                 and not task["advisory_only"]]
        attach_samples(scan, [task["context"] for task in joint])
        if scan.get("on_demand"):
            from .holdout_plan import freeze

            freeze(scan, gated, joint)
        for task in joint:
            task["requires_sample_acceptance"] = True
            sampled = {field["id"] for field in task["context"].get("pair_details", ()) if field.get("perturbed_samples")}
            task["sample_complete"] = bool(sampled) and sampled == set(task["context"]["fields"])
        # A prerequisite is already included in the fixed recipe for this pair.
        covered = {task["candidate"] for task in joint if task["route"] in {
            "indexed_table_recipe", "windowed_grid_recipe", "station_grid", "independent_grid_search"}}
        tasks = [task for task in tasks if task["candidate"] not in covered or task["operation"] not in {
            "align_space", "align_time", "compare_station_grid"}]
    elif include_samples:
        attach_samples(scan,[task["context"] for task in tasks])
    request=dict(stage="recommendation",target_model=approved["target_model"],tasks=recommendation_metadata(tasks),
                 shared_algorithm_catalogue=shared_algorithm_catalogue(),
                 task_requirements=approved.get('task_requirements',{}),
                 engineering_documents=documents,
                 project_context_document_ids=approved.get("project_context_document_ids",[]),
                 engineering_inventory=approved.get("engineering_inventory",[]))
    assert_current(scan)
    return request

def algorithm_registry():
    return {key: ALGORITHM_NAMES.get(key,value) for key,value in
            {**ALGORITHMS,**ENGINEERING,**ALIGNMENT,**PREPARATION}.items()}


def shared_algorithm_catalogue():
    from ecodata.algorithms.platform_search import platform_registry

    catalogue = {
        spec.id: dict(name=ALGORITHM_NAMES.get(spec.id, spec.name), status=spec.status,
                      category=spec.category, input_kinds=list(spec.input_kinds))
        for spec in platform_registry().specs()
    }
    if "fusion.residual" in catalogue:
        catalogue["fusion.residual"]["station_fit_variants"] = ["linear", "rf"]
    return catalogue


def recommend_plan(plan, decisions, *, mock=False, api_key=None, progress=None, on_request=None):
    """One shared online recommendation protocol over the checked recipe catalog."""
    from . import service
    receipts=[]
    try:
        if progress:progress('DeepSeek正在从已核验的具体算法组合中推荐；推荐仍须真实数值验证')
        approved=service.preview(plan)
        if on_request is not None:
            on_request(approved)
        reply=service.send(approved,approved.digest,mock,api_key=api_key,
            on_progress=progress,on_receipt=receipts.append)
        if not reply.get('recommendations'):raise ValueError('线上未返回目录内的可执行组合')
        rows=reply['recommendations']
        for row in rows:row['recommendation_origin']='deepseek' if not mock else 'simulated'
        return rows,dict(status='SIMULATED' if mock else 'COMPLETED',api_receipts=receipts,request_sha256=approved.digest)
    except (ValueError,OSError,TimeoutError) as exc:
        # Recommendation is advisory: existing checked fallback is allowed,
        # but must never be presented as an online algorithm choice.
        rows=from_decisions(plan,decisions)
        for row in rows:
            row['recommendation_origin']='platform_fallback'
            row['reason']='线上具体算法推荐未完成，以下为平台已有候选补充，不是DS已验证方案。'+row['reason']
        if progress:progress('线上算法推荐未完成；保留平台可核验候选：'+str(exc))
        return rows,dict(status='PLATFORM_FALLBACK',error=str(exc),api_receipts=receipts)


def review_holdouts(scan, approved, plan, recommendations, gated):
    from .catalog_decision import local_fields
    from .sampling_plan import recommendation_review
    from .release_gate import assess
    from .sampling_risk import attach
    from . import recipe_trials

    originals, _ = local_fields(scan, approved)
    tasks = {task["id"]: task for task in plan["tasks"]}
    cache = {}
    assert_current(scan)
    for recommendation in recommendations:
        if recommendation.get("id") == "registry_advice":
            continue
        task = tasks.get(recommendation["task"])
        if not task or task.get("engineering_fusion_plan"):
            continue
        refs = [originals[fid] for fid in task["context"]["fields"] if fid in originals]
        execution = gated.get((task["candidate"], task["operation"]))
        missing_input = None
        if task.get("requires_sample_acceptance"):
            recommendation["requires_final_confirmation"] = True
            if not task["sample_complete"]:
                missing_input = "insufficient_sample_coverage"
            elif recommendation.get("recommendation_origin") == "platform_fallback":
                missing_input = "initial_recommendation_unavailable"
        if recipe_trials.required(execution) or task["operation"] == "fusion":
            key = ("joint_holdout", task["id"], tuple(recommendation["steps"]))
            if missing_input and (execution or {}).get('route')=='station_grid':
                # Missing outbound recommendation samples are not proof that a
                # native reader cannot build the disjoint local trial queue.
                recommendation['automatic_recovery']=dict(action='registered_local_trial',
                    original_issue=missing_input,reason='Use actual station/native-grid reader; no fabricated samples')
                missing_input=None
            if missing_input:
                cache[key] = dict(status="pending", basis=missing_input, count=0, checks=[])
            elif key not in cache:
                cache[key] = recipe_trials.review(scan, task["id"], execution, recommendation["steps"],
                                                  model=plan["target_model"]["id"])
            report = [dict(cache[key], field=fid) for _, fid, _ in refs]
        else:
            report = recommendation_review(scan, recommendation, execution, refs, cache)
            for (index, _, field), item in zip(refs, report):
                attach(scan, index, field, item, cache)
        recommendation["local_holdout_review"] = report
        recommendation["execution_release"] = assess(scan, recommendation, execution, refs)
        recommendation["execution_fields"] = [(index, field["name"], fid) for index, fid, field in refs]
        recommendation["local_recipe_options"] = [dict(option) for option in task["allowed_recipes"]]
    continue_rejected_search(recommendations, gated)
    assert_current(scan)


def continue_rejected_search(rows, gated):
    """A failed recipe is not a failed task; admit exploration, never that recipe."""
    from .release_gate import allowed
    from .recipe_trials import continuation_steps

    for task_id in dict.fromkeys(row.get("task") for row in rows):
        if not task_id or any(allowed(row) or row.get("search_admission")
                              for row in rows if row.get("task") == task_id):
            continue
        task = gated.get(tuple(task_id.split(":", 1)), {})
        rejected = [row for row in rows if row.get("task") == task_id
                    and row.get("execution_release", {}).get("status") == "failed"]
        numeric = []
        for row in rejected:
            reports = row.get("local_holdout_review", ())
            failures = [failure for report in reports
                        for failure in report.get("acceptance_sampling", {}).get("failures", ())]
            if (reports and failures and all(report.get("basis") == "recipe_numeric_rejected" for report in reports)
                    and all(failure.get("rule") == "recipe_score_below_single_source"
                            and not failure.get("unscored") for failure in failures)):
                numeric.append(row)
        if not numeric:
            continue
        preparation = continuation_steps(task, numeric[0])
        if preparation is None:
            continue
        admission = dict(protocol="rejected-recipe-search-v1", route=task["route"],
                         rejected_ids=[row["id"] for row in numeric],
                         status="SEARCH_ONLY", reason="numeric_recipe_failure_not_task_incompatibility")
        task["search_after_rejection"] = admission
        rows.append(dict(task=task_id, id="legal_candidate_search", steps=preparation,
                         name="合法候选变异寻优（推荐未通过）",
                         description="原推荐评分失败并保留；从其他合法候选搜索，最终仍须质量检查和独立复核，不保证融合胜出。",
                         reason="允许继续搜索不等于放行原推荐；拒绝推荐不作为搜索种子或偏好。",
                         recommendation_origin="platform_search_continuation", search_admission=admission,
                         execution_release=dict(status="search_only", checks=[], blockers=[]),
                         execution_fields=numeric[0].get("execution_fields", [])))


def supplementary_search_continuation(previous, gated):
    """Continue an already admitted task without reusing exposed confirmation.

    A new DS reply is not numerical evidence. The old failed recipe remains
    rejected, while other legal graphs can still be explored and exported.
    """
    from copy import deepcopy

    rows=[deepcopy(previous)]
    continue_rejected_search(rows,gated)
    if len(rows)==1:
        return None
    row=rows[-1]
    task=gated[tuple(previous['task'].split(':',1))]
    task['mutation_reservation']=dict(status='INSUFFICIENT',
        reason='Supplementary recommendation exposed task material; no new independent confirmation supplied')
    task['mutation_confirmation_times']=[]
    row['supplementary_evidence_exposed']=True
    row['previous_recommendation_review']=deepcopy(previous)
    return row


def finalize_recipes(plan, rows, *, mock, api_key, progress):
    """Ask DeepSeek to confirm only full recipes that passed local acceptance."""
    passed = {}
    for row in rows:
        row.pop("final_verified_recommendation", None)
        row.pop("final_recommendation_origin", None)
        release = row.get("execution_release", {})
        reports = row.get("local_holdout_review", ())
        if release.get("status") == "released" and any(
                report.get("basis") == "joint_recipe_acceptance_passed" for report in reports):
            passed[(row["task"], row["id"])] = row
    if not passed:
        return None
    tasks = []
    for task in plan["tasks"]:
        recipes = []
        for option in task["allowed_recipes"]:
            row = passed.get((task["id"], option["id"]))
            if row:
                audit = row["local_holdout_review"][0]["acceptance_sampling"]
                text = (f"{option['description']}。留出验收 {audit['inspected']} 组，"
                        f"失败 {audit['failed']} 组，允许 {audit['allowed_failures']} 组。")
                recipes.append(dict(option, description=text))
        if recipes:
            tasks.append(dict(id=task["id"], candidate=task["id"], operation=task["operation"], context={},
                              allowed_recipes=recipes, engineering_note="仅从通过本地验收的完整方案中确认"))
    reviewed = dict(stage="recommendation", target_model=plan["target_model"], tasks=tasks,
                    final_verified_recommendation=True)
    selected, receipt = recommend_plan(reviewed, [], mock=mock, api_key=api_key, progress=progress)
    if receipt["status"] == "PLATFORM_FALLBACK":
        return receipt
    for choice in selected:
        row = passed[(choice["task"], choice["id"])]
        row["final_verified_recommendation"] = True
        row["final_recommendation_origin"] = choice["recommendation_origin"]
        row["reason"] = choice["reason"]
        if "explanation" in row:
            row["explanation"]["reason"] = choice["reason"]
    return receipt


def reviewed_standalone_alignment(scan, approved, checked, gated, rows):
    """Offer the separately verified alignment when the joint recipe was rejected."""
    from .discovery import engineering_fusion_targets
    from .release_gate import allowed

    rejected = {row["task"].split(":", 1)[0] for row in rows
                if row["task"].endswith(":fusion") and row.get("execution_release", {}).get("status") == "failed"}
    rejected.difference_update(engineering_fusion_targets(approved))
    independent = [check for check in checked if check["candidate"] in rejected and check["action"] == "align_space"
                   and check["status"] == "通过"
                   and gated.get((check["candidate"], "align_space"), {}).get("route") == "independent_alignment"]
    if not independent:
        return []
    plan = make_request(scan, approved, independent, gated, include_samples=False)
    alternatives = from_decisions(plan, [])
    for row in alternatives:
        row.update(recommendation_origin="platform_fallback", standalone_alignment=True,
                   reason="融合方案未通过留出评分。可以单独运行单位与网格对齐，输出两份对齐后的来源。")
    review_holdouts(scan, approved, plan, alternatives, gated)
    return [row for row in alternatives if allowed(row)]


def from_decisions(summary,decisions):
    from collections import defaultdict

    by_candidate=defaultdict(list)
    for row in decisions:
        by_candidate[row["candidate"]].append(row)
    entries=[]
    for task in summary["tasks"]:
        related=by_candidate[task["candidate"]]
        direct=[item for item in related if item["action"]==task["operation"]]
        preferred=direct or [item for item in related if item["action"] not in {"reject","direct"}]
        ai=next(iter(preferred),{})
        options=task["allowed_recipes"]
        if options:
            groups=[group for item in preferred for group in item.get("algorithm_steps",[])]
            matches=[option for option in options if any(option["steps"]==group for group in groups)]
            if not matches and groups:
                proposed={step for group in groups for step in group if not step.startswith('engineering.')}
                matches=[option for option in options if proposed.intersection(option["steps"])]
            chosen=matches[:4] or options[:1]
            for option in chosen:
                if option in matches and ai.get("reason"):
                    reason=ai["reason"]
                    if not any(option["steps"]==group for group in groups):
                        reason="按 AI 提议匹配原版可执行步骤。"+reason
                else:
                    reason="平台按本地核验结果补充可执行组合。"+task.get("engineering_note","")
                entries.append(dict(task=task["id"],recipe=option["id"],reason=reason))
        else:
            entries.append(dict(task=task["id"],recipe="advisory",name=ai.get("algorithm_name") or "处理方案待核验",
                                reason=ai.get("reason") or task.get("engineering_note",""),
                                alignment="按已核验的时间与空间支持对齐",
                                fusion_trial="条件满足后比较单源和多源的留出评分"))
    return validate({"recommendations":entries},summary)


def explanation(task,reason):
    """Display inspectable input facts separately from the model's suggestion."""
    facts=[];unknown=[]
    for field in task.get('context',{}).get('pair_details',[]):
        source=field.get('id') or field.get('source') or '未绑定来源'
        variable=field.get('variable_name') or field.get('header') or '未知变量'
        facts.append(f"{source} · {variable} · 单位 {field.get('unit') or 'UNKNOWN'}")
        timing=field.get('temporal_semantics')
        if not timing and field.get('time_cell_methods'):
            timing='原始 cell_methods='+','.join(field['time_cell_methods'])
        facts.append(f"  时间：{timing or 'UNKNOWN'}；采样：{field.get('time_resolution_class') or 'UNKNOWN'}")
        if field.get('grid_interval'):
            grid=[]
            for axis,spacing in field['grid_interval'].items():
                if isinstance(spacing,dict):
                    unit={'degrees_east':'度（经度）','degrees_north':'度（纬度）'}.get(spacing.get('unit'),spacing.get('unit') or '单位未知')
                    grid.append(f"{ {'x':'横向','y':'纵向'}.get(axis,axis)}间距 {spacing.get('step','UNKNOWN')} {unit}")
            facts.append('  网格：'+'；'.join(grid))
        if not field.get('unit'):unknown.append(source+' 的单位未确认')
        if not timing:unknown.append(source+' 的时间统计含义需核实')
    return dict(facts=facts,reason=reason,checks=task.get('engineering_note') or '执行前仍需通过公共契约与质量检查',
        unknowns=unknown,boundary='这是适用性建议，不是效果保证；是否保留由同一评价队列上的真实执行决定。')


def explanation_text(row):
    detail=row.get('explanation') or {}
    report = next((item["acceptance_sampling"] for item in row.get("local_holdout_review", ())
                   if "allowed_failures" in item.get("acceptance_sampling", {})), None)
    acceptance = []
    if len(row.get("local_recipe_options", ())) == 1:
        acceptance.append("当前契约下只有一项可推荐配方；这是待验方案，不代表已完成多算法效果选优。")
    if report:
        planned = report["sample_size"]
        acceptance = [f"留出验收总体 {report['population']} 组 · 计划试跑 {planned} 组 · 允许 {report['allowed_failures']} 组失败"
                      if planned else f"留出验收总体 {report['population']} 组 · 预算内没有满足误判风险的抽检设计",
                      f"实际试跑 {report['inspected']} 组 · 不合格 {report['failed']} 组 · 缺评分证据 {report.get('unscored', 0)} 组"]
        if report.get("acceptance_rule") == "zero_failures":
            acceptance.append(f"真实不合格占比达到 {report['acceptable_share']:.0%} 时，"
                              f"计划把全漏掉的概率控制在 {report['false_accept_limit']:.0%} 以内。")
        scores = next((item["sampled_scores"] for item in row.get("local_holdout_review", ())
                       if item.get("sampled_scores")), ())
        if scores:
            groups = {}
            for score in scores:
                groups.setdefault(score.get("assessment"), []).append(score)
            for stage, group in groups.items():
                title = {"gap_fill": "补齐评分", "fusion": "融合评分"}.get(stage, "留出组评分")
                acceptance.append(f"{title}（方案与单源基线使用相同参考）")
                for label, key in (("固定方案", "candidate"), ("单源基线", "baseline")):
                    errors = [score[key]["RMSE"] for score in group]
                    acceptance.append(f"{label} · 各组 RMSE 范围 {min(errors):.4g} 至 {max(errors):.4g}")
            if any("proxy" in str(score.get("reference", "")).lower() for score in scores):
                acceptance.append("此处是产品重建代理初筛，不是独立实测精度检验；未通过只表示该方案未通过此代理标准，不能证明融合无价值。")
        acceptance.append("此处抽检不代替变异后冻结方案的独立分块复核；缺少未暴露的复核数据时不能确认提升。")
    return '\n'.join(['数据依据',*(detail.get('facts') or ['尚无结构化数据依据，不能由推荐文字推断已核实。']),
        '', '推荐理由',row.get('reason') or '未提供', '', '平台检查与前提',detail.get('checks') or '等待契约核验',
        *acceptance,
        '', '仍需核实',*(detail.get('unknowns') or ['未列出的信息不代表全部已确认；以执行检查记录为准。']),
        '',detail.get('boundary') or '推荐不代表精度提升，必须经过真实评价。'])


def validate(raw, summary):
    entries=raw.get("recommendations")
    if not isinstance(entries,list):
        raise ValueError("DeepSeek 未返回算法推荐列表")
    tasks = {task["id"]: task for task in summary["tasks"]}
    options={(task["id"],recipe["id"]):recipe for task in tasks.values() for recipe in task["allowed_recipes"]}
    advisory_tasks={task["id"]:task for task in summary["tasks"] if task.get("advisory_only")}
    catalogue = summary.get("shared_algorithm_catalogue", {})
    recommendations=[]
    seen=set()
    final = summary.get("final_verified_recommendation")
    confirmed = set()
    for entry in entries:
        if not isinstance(entry,dict):
            continue
        key=(entry.get("task"),entry.get("recipe"))
        if not all(isinstance(k,str) for k in key) or key in seen:
            continue
        if tasks.get(key[0], {}).get("engineering_fusion_plan"):
            continue
        if final and (key not in options or key[0] in confirmed):
            continue
        if key in options:
            if tasks[key[0]].get("operation") == "fusion" and key[0] in confirmed:
                continue
            seen.add(key)
            confirmed.add(key[0])
            task=tasks[key[0]]
            quantities={f.get('physical_family') for f in task['context'].get('pair_details',[])}-{None,'UNKNOWN'}
            scope=dict(variable=next(iter(quantities))) if len(quantities)==1 else {}
            reason=str(entry.get('reason',''))[:600]
            recommendations.append(dict(task=key[0],**options[key],scope=scope,reason=reason,explanation=explanation(task,reason)))
        elif key[0] in tasks and key[1] == "registry_advice":
            ids = entry.get("algorithms")
            if not isinstance(ids, list) or not 1 <= len(ids) <= 8:
                continue
            if any(not isinstance(algorithm, str) or algorithm not in catalogue for algorithm in ids):
                continue
            seen.add(key)
            methods = [catalogue[algorithm]["name"] for algorithm in dict.fromkeys(ids)]
            reason = str(entry.get("reason", ""))[:600]
            recommendations.append(dict(task=key[0], id=key[1], name=" → ".join(methods), steps=[],
                                        catalogue_algorithms=list(dict.fromkeys(ids)), advisory_only=True,
                                        description="仓库算法建议 · 供方案参考", reason=reason,
                                        explanation=explanation(tasks[key[0]], reason)))
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
    for task in tasks.values():
        if task.get("engineering_fusion_plan"):
            recommendations.append(dict(task=task["id"], id="advisory", name="工程要求融合 · 方案待核验",
                steps=[], advisory_only=True, alignment_plan=True,
                description="对齐：统一时间窗口，网格用保守重网格，表格按站点和时间键配对。补齐：以对齐的备用源逐缺口补位。"
                            "融合：共同有效范围内比较加权融合与单源基线。",
                reason=task["engineering_note"]))
    missing={t["id"] for t in summary["tasks"]}-{r["task"] for r in recommendations}
    missing.update(task["id"] for task in tasks.values()
                   if (task.get("requires_sample_acceptance") or task.get("engineering_requires_fusion"))
                   and task["id"] not in confirmed)
    if missing:
        raise ValueError("以下任务未收到有效推荐，请重试："+", ".join(sorted(missing)))
    return recommendations
