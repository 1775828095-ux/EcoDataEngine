"""Build and validate recommendations for the operations released on page one."""
from copy import deepcopy
import json

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
    "bias.prior_series_ratio": "历史实测比例订正",
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
            and not row.get("station_baseline")
            and row.get("final_verified_recommendation") is not True
            and not row.get("search_admission"))


def display_record(row):
    if is_private_fusion_candidate(row):
        return dict(task=row["task"], status=row.get("execution_release", {}).get("status", "pending"))
    return {key: value for key, value in row.items() if key != "local_recipe_options"}


def attach_samples(scan, pairs):
    from .holdout_plan import export_field
    from .sample_values import RECOMMENDATION_SAMPLE_LIMIT

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
                    payload=export_field(scan,index,selected,phase="recommendation",limit=RECOMMENDATION_SAMPLE_LIMIT)
                    samples[fid]=(
                        payload.get("perturbed_samples",[]),
                        (payload.get("sampling") or {}).get("basis") or payload.get("basis"),
                    )
                except (EcoDataError,OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,RuntimeError,ImportError):
                    samples[fid]=([], "sample_unavailable")
            field["perturbed_samples"],field["recommendation_sample_basis"]=samples[fid]


def make_request(scan, approved, checked, gated, *, local_checks=(), include_samples=True, sampled_workflow=False,prepared_graph_service=True):
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
        if scan.get("on_demand"):
            from .holdout_plan import reserve_station_fields

            reserve_station_fields(scan, gated, joint)
        attach_samples(scan, [task["context"] for task in joint])
        if scan.get("on_demand"):
            from .holdout_plan import freeze

            freeze(scan, gated, joint)
        for task in joint:
            task["requires_sample_acceptance"] = True
            sample_counts = dict.fromkeys(task["context"]["fields"], 0)
            for field in task["context"].get("pair_details", ()):
                if field.get("id") in sample_counts:
                    sample_counts[field["id"]] = len(field.get("perturbed_samples", ()))
            task["sample_complete"] = bool(sample_counts) and all(sample_counts.values())
            options = task["allowed_recipes"]
            shared_steps = set(options[0]["steps"]).intersection(*(option["steps"] for option in options[1:]))
            task["pretrial_evidence"] = dict(
                source_sample_counts=sample_counts,
                sample_comparison="unpaired_perturbed_examples",
                shared_steps=[step for step in options[0]["steps"] if step in shared_steps],
                distinguishing_steps={option["id"]: [step for step in option["steps"] if step not in shared_steps]
                                      for option in options},
                comparison_baseline="preserve_primary_source",
            )
        # A prerequisite is already included in the fixed recipe for this pair.
        covered = {task["candidate"] for task in joint if task["route"] in {
            "indexed_table_recipe", "windowed_grid_recipe", "station_grid", "independent_grid_search"}}
        tasks = [task for task in tasks if task["candidate"] not in covered or task["operation"] not in {
            "align_space", "align_time", "compare_station_grid"}]
    elif include_samples:
        attach_samples(scan,[task["context"] for task in tasks])
    for task in tasks:
        bound=gated.get((task['candidate'],task['operation']),{})
        from .recipe_trials import PUBLIC_SEARCH_ROUTES, SEARCH_CONTINUATION_ROUTES
        route = task.get("route")
        if (sampled_workflow and task['operation']=='fusion' and route in SEARCH_CONTINUATION_ROUTES
                and task.get('requires_sample_acceptance')):
            # These adapters can score an entire fixed recipe on a locked holdout.
            # An arbitrary Registry graph cannot inherit a different recipe's acceptance.
            task['bayesian_recipe_gate'] = True
            continue
        if prepared_graph_service and task['operation']=='fusion' and route in PUBLIC_SEARCH_ROUTES and not task.get('advisory_only'):
            task['prepared_graph_service']=True
        if route in {'station_grid','indexed_table_recipe'} and task['operation']=='fusion' and not task.get('advisory_only'):
            task['graph_protocol']=dict(version='registered_graph_v1',source_count=1,
                variable=bound.get('quantity'),
                sources=[dict(id='source:0',role='prepared_station_background')],
                state='Chronological training state fitted by the public paired adapter; never a supplied parameter',
                evaluation='Disjoint reference remains local; all proposed graphs require actual execution and final review',
                node_fields=['id','algorithm','parents','parameters'],output='algorithm node id',
                legacy_recipes='Compatibility examples only; not an algorithm whitelist')
        elif route=='independent_grid_search' and task['operation']=='fusion' and not task.get('advisory_only'):
            from ecodata.algorithms.platform_search import platform_registry
            from ecodata.algorithms.spatial_fusion_task import graph_components
            task['graph_protocol']=dict(version='registered_graph_v1',source_count=2,
                variable='precipitation',
                sources=[dict(id='source:0',role='prepared_primary'),dict(id='source:1',role='prepared_background')],
                resources=['observations'],components=graph_components(platform_registry()),
                resource_meaning='observations contains only retained training positions in each local fold',
                evaluation='Same fixed spatial/date folds and final independent review; no reference leaves machine',
                legacy_recipes='Examples only; DS proposes explicit parent edges')
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
    from ecodata.algorithms.registered_graph import catalogue

    entries=catalogue(platform_registry())
    for identifier,entry in entries.items():entry['name']=ALGORITHM_NAMES.get(identifier,entry['name'])
    if "fusion.residual" in entries:
        entries["fusion.residual"]["station_fit_variants"] = ["linear", "rf"]
    return entries


def requested_evidence(scan, plan, requests):
    from .holdout_plan import exposures, export_field
    from .sample_values import RECOMMENDATION_SAMPLE_LIMIT, RECOMMENDATION_SAMPLE_POOL

    tasks = {task["id"]: task for task in plan["tasks"]}
    documents = {doc["id"]: doc for doc in plan.get("engineering_documents", ()) if doc.get("id")}
    results, seen = [], set()
    assert_current(scan)
    for request in requests[:4]:
        if not isinstance(request, dict):
            continue
        task_id, kind, target = (request.get(key) for key in ("task", "kind", "target"))
        if not all(isinstance(item, str) and len(item) <= 80 for item in (task_id, kind, target)):
            continue
        scope = (task_id, kind, target)
        if scope in seen:
            continue
        seen.add(scope)
        task = tasks.get(task_id)
        if not task:
            results.append(dict(status="outside_scope", basis="unknown_task"))
            continue
        base = dict(task=task_id, kind=kind, target=target)
        if kind == "document_excerpt":
            document = documents.get(target, {})
            text = document.get("text")
            if isinstance(text, str) and text.strip():
                results.append(dict(base, status="available", excerpt=text[:1000]))
            else:
                results.append(dict(base, status="unavailable", basis="no_approved_document_text"))
            continue
        fields = task.get("context", {}).get("fields", ())
        if target not in fields or kind not in {"field_metadata", "numeric_samples"}:
            results.append(dict(base, status="outside_scope", basis="not_an_approved_task_field"))
            continue
        try:
            source_id, field_id = target.split(".")
            index, offset = int(source_id[1:])-1, int(field_id[1:])-1
            if min(index, offset) < 0:
                raise ValueError("invalid field")
            row = scan["rows"][index]
            field = row["fields"][offset]
            if kind == "field_metadata":
                keys = ("name", "unit", "type", "standard_name", "long_name", "temporal_semantics")
                metadata = {key: field[key][:160] for key in keys if isinstance(field.get(key), str)}
                results.append(dict(base, status="available", format=row["format"], metadata=metadata,
                                    dimensions=[str(name)[:80] for name in field.get("dimensions", ())[:8]]))
                continue
            # Initial sampling records the whole authorized recommendation partition.
            # A follow-up may use more values from that same partition, never open a new one.
            if "recommendation" in exposures(scan, index, field):
                payload = export_field(scan, index, field, phase="recommendation_followup",
                                       limit=RECOMMENDATION_SAMPLE_POOL)
                extra = payload.get("perturbed_samples", ())[RECOMMENDATION_SAMPLE_LIMIT:]
                if extra:
                    results.append(dict(base, status="available", perturbed_samples=extra))
                    continue
            results.append(dict(base, status="unavailable", basis="no_additional_permitted_samples"))
        except (EcoDataError,OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,RuntimeError,ImportError):
            results.append(dict(base, status="unavailable", basis="local_reader_or_metadata_unavailable"))
    assert_current(scan)
    return results or [dict(status="outside_scope", basis="no_valid_evidence_requests")]


def recommend_plan(plan, decisions, *, mock=False, api_key=None, progress=None, on_request=None, scan=None, gated=None):
    """One shared online recommendation protocol over the checked recipe catalog."""
    from . import service
    station_tasks = [task for task in plan["tasks"] if task.get("route") == "station_grid"
                     and task["operation"] == "fusion" and task.get("requires_sample_acceptance")
                     and gated and (task["candidate"], task["operation"]) in gated]
    if scan is not None and station_tasks:
        from . import station_recommendation

        station_ids = {task["id"] for task in station_tasks}
        execution = {task["id"]: gated[(task["candidate"], task["operation"])] for task in station_tasks}
        rows, receipt = station_recommendation.recommend(
            scan, dict(plan, tasks=station_tasks), execution, mock=mock, api_key=api_key,
            progress=progress, on_request=on_request)
        other_tasks = [task for task in plan["tasks"] if task["id"] not in station_ids]
        if not other_tasks:
            return rows, receipt
        other_rows, other_receipt = recommend_plan(
            dict(plan, tasks=other_tasks), decisions, mock=mock, api_key=api_key, progress=progress,
            on_request=on_request, scan=scan)
        receipt["other_recommendation"] = {key: value for key, value in other_receipt.items() if key != "api_receipts"}
        receipt["api_receipts"].extend(other_receipt["api_receipts"])
        if other_receipt["status"] != "PLATFORM_FALLBACK":
            receipt["status"] = other_receipt["status"]
        return [*rows, *other_rows], receipt
    receipts=[]
    followup = {}
    try:
        if progress:progress('DeepSeek正在从已核验的具体算法组合中推荐；推荐仍须真实数值验证')
        may_request = bool(scan and scan.get("on_demand") and not plan.get("final_verified_recommendation"))
        outbound = dict(plan, can_request_recommendation_evidence=True) if may_request else plan
        approved=service.preview(outbound)
        if on_request is not None:
            on_request(approved)
        reply=service.send(approved,approved.digest,mock,api_key=api_key,
            on_progress=progress,on_receipt=receipts.append)
        if may_request and reply.get("evidence_requests"):
            if progress:progress('DeepSeek 请求进一步证据，正在按任务范围补充已授权资料…')
            evidence = requested_evidence(scan, plan, reply["evidence_requests"])
            next_request = service.preview(dict(plan, recommendation_evidence=evidence))
            followup = dict(evidence_followup_sha256=next_request.digest,
                            evidence_followup=json.loads(next_request.summary)["recommendation_evidence"])
            if on_request is not None:
                on_request(next_request)
            reply=service.send(next_request,next_request.digest,mock,api_key=api_key,
                on_progress=progress,on_receipt=receipts.append)
        if not reply.get('recommendations'):raise ValueError('线上未返回目录内的可执行组合')
        rows=reply['recommendations']
        for row in rows:row['recommendation_origin']='deepseek' if not mock else 'simulated'
        return rows,dict(status='SIMULATED' if mock else 'COMPLETED',api_receipts=receipts,
                         request_sha256=approved.digest, **followup)
    except (ValueError,OSError,TimeoutError) as exc:
        # Recommendation is advisory: existing checked fallback is allowed,
        # but must never be presented as an online algorithm choice.
        rows=from_decisions(plan,decisions)
        for row in rows:
            row['recommendation_origin']='platform_fallback'
            row['reason']='线上具体算法推荐未完成，以下为平台已有候选补充，不是DS已验证方案。'+row['reason']
        if progress:progress('线上算法推荐未完成；保留平台可核验候选：'+str(exc))
        return rows,dict(status='PLATFORM_FALLBACK',error=str(exc),api_receipts=receipts, **followup)


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
        if recommendation.get('autonomous_recommendation') and task.get('prepared_graph_service'):
            recommendation.update(requires_final_confirmation=False,local_holdout_review=[],
                execution_release=dict(status='search_only',checks=[],blockers=[]),
                search_admission=dict(protocol='prepared-registry-search-v2',route=execution.get('route','local'),status='SEARCH_ONLY'),
                execution_fields=[(index,field['name'],fid) for index,fid,field in refs],local_recipe_options=[])
            continue
        if recommendation.get('graph') is not None:
            # Admission to search is not release of this unmeasured recommendation.
            # The public paired adapter owns isolated fitting, graph compatibility,
            # numerical validation and final independent acceptance.
            if task.get('graph_protocol') and (execution or {}).get('route') in {'station_grid','independent_grid_search','indexed_table_recipe'}:
                recommendation.update(requires_final_confirmation=False,
                    local_holdout_review=[],execution_release=dict(status='search_only',checks=[],blockers=[]),
                    search_admission=dict(protocol='registered-graph-search-v1',route=execution['route'],
                        status='SEARCH_ONLY',reason='Explicit graph requires public numerical and final acceptance'),
                    execution_fields=[(index,field['name'],fid) for index,fid,field in refs],
                    local_recipe_options=[])
                continue
            raise ValueError('Explicit Registry graph has no prepared execution binding for this route')
        missing_input = None
        if task.get("requires_sample_acceptance"):
            recommendation["requires_final_confirmation"] = True
            if not task["sample_complete"]:
                missing_input = "insufficient_sample_coverage"
            elif recommendation.get("recommendation_origin") == "platform_fallback":
                missing_input = "initial_recommendation_unavailable"
        if recommendation.get("station_two_stage"):
            report = [dict(recommendation["station_report"], field=fid) for _, fid, _ in refs]
            recommendation["requires_final_confirmation"] = True
        elif recipe_trials.required(execution) or task["operation"] == "fusion":
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
                trial_execution = (dict(execution, bayesian_mask_phase=0)
                                   if execution and task.get("bayesian_recipe_gate") else execution)
                reviewed = recipe_trials.review(scan, task["id"], trial_execution, recommendation["steps"],
                                                model=plan["target_model"]["id"])
                cache[key] = recipe_trials.require_external_reference(reviewed)
            report = [dict(cache[key], field=fid) for _, fid, _ in refs]
        else:
            report = recommendation_review(scan, recommendation, execution, refs, cache)
            for (index, _, field), item in zip(refs, report):
                attach(scan, index, field, item, cache)
        recommendation["local_holdout_review"] = report
        recommendation["execution_release"] = assess(scan, recommendation, execution, refs)
        recommendation["execution_fields"] = [(index, field["name"], fid) for index, fid, field in refs]
        if recommendation.get("station_two_stage"):
            if recommendation["execution_release"]["status"] == "released":
                recommendation["final_verified_recommendation"] = True
                recommendation["final_recommendation_origin"] = recommendation["recommendation_origin"]
                execution["station_two_stage"] = True
                execution["station_baseline"] = bool(recommendation.get("station_baseline"))
        else:
            recommendation["local_recipe_options"] = [dict(option) for option in task["allowed_recipes"]]
    continue_rejected_search(recommendations, gated)
    assert_current(scan)


def continue_rejected_search(rows, gated):
    """A failed recipe is not a failed task; admit exploration, never that recipe."""
    from .release_gate import allowed
    from .recipe_trials import continuation_steps,UNSCORED
    from .sampling_risk import BAYESIAN_RULE

    for task_id in dict.fromkeys(row.get("task") for row in rows):
        if not task_id or any(allowed(row) or row.get("search_admission")
                              for row in rows if row.get("task") == task_id):
            continue
        task = gated.get(tuple(task_id.split(":", 1)), {})
        rejected = [row for row in rows if row.get("task") == task_id
                    and not row.get("station_two_stage")
                    and row.get("execution_release", {}).get("status") in {"failed","pending"}]
        numeric = []
        for row in rejected:
            reports = row.get("local_holdout_review", ())
            if any(report.get("acceptance_sampling", {}).get("acceptance_rule") == BAYESIAN_RULE for report in reports):
                continue
            failures = [failure for report in reports
                        for failure in report.get("acceptance_sampling", {}).get("failures", ())]
            if (row.get('execution_release',{}).get('status')=='failed' and reports and failures and all(report.get("basis") == "recipe_numeric_rejected" for report in reports)
                    and all(failure.get("rule") == "recipe_score_below_single_source"
                            and not failure.get("unscored") for failure in failures)):
                numeric.append(row)
            elif reports and all(report.get('basis')=='recipe_scoring_reference_insufficient' for report in reports):
                unscored=[unit for report in reports for unit in report.get('acceptance_sampling',{}).get('unscored_units',())]
                if unscored and all(unit.get('rule') in UNSCORED for unit in unscored):
                    numeric.append(row)
        if not numeric:
            continue
        preparation = continuation_steps(task, numeric[0])
        if preparation is None:
            continue
        insufficient=any(r.get('execution_release',{}).get('status')=='pending' for r in numeric)
        admission = dict(protocol="rejected-recipe-search-v1", route=task["route"],
                         rejected_ids=[row["id"] for row in numeric],
                         status="SEARCH_ONLY", reason="insufficient_reference_exploration_only" if insufficient else "numeric_recipe_failure_not_task_incompatibility")
        task["search_after_rejection"] = admission
        rows.append(dict(task=task_id, id="legal_candidate_search", steps=preparation,
                         name="合法候选变异寻优（推荐未通过）",
                         description="原推荐未通过或评分证据不足，记录保留；从其他合法候选搜索，真实数值目标和最终复核仍须满足契约，不保证融合胜出。",
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


def retry_bayesian_reviews(scan, approved, plan, rows, gated, *, mock, api_key, progress):
    from . import recipe_trials, service

    tasks = {task["id"]: task for task in plan["tasks"] if task.get("bayesian_recipe_gate")}
    receipts = []

    def send(payload):
        preview = service.preview(payload)
        return service.send(preview, preview.digest, mock, api_key=api_key, on_progress=progress,
                            on_receipt=receipts.append)

    for index, first in enumerate(rows):
        task = tasks.get(first["task"])
        if not task or first.get("station_two_stage"):
            continue
        key = task["candidate"], task["operation"]
        report = next(iter(first.get("local_holdout_review", ())), {})
        if first.get("execution_release", {}).get("status") not in {"failed", "pending"} or not recipe_trials.needs_revision(report):
            continue
        final_row = {}

        def evaluate(option, scope):
            execution = dict(gated[key], bayesian_review_scope=scope)
            if scope["mode"] == "development":
                return recipe_trials.review(scan, task["id"], execution, option["steps"],
                                            model=plan["target_model"]["id"])
            final_row.clear()
            final_row.update(option, recommendation_origin="simulated" if mock else "deepseek")
            review_holdouts(scan, approved, dict(plan, tasks=[task]), [final_row], {key: execution})
            return next(iter(final_row.get("local_holdout_review", ())), {})

        _, _, history, status = recipe_trials.tune_recipe(
            first, report, task["allowed_recipes"], evaluate, send, progress,
            (ValueError, OSError, TimeoutError, RuntimeError))
        candidate = final_row or first
        candidate.update(bayesian_revision_status=status, bayesian_revision_history=history)
        if status == "passed" and candidate["execution_release"]["status"] == "released":
            candidate.update(final_verified_recommendation=True,
                             final_recommendation_origin=candidate["recommendation_origin"])
        rows[index] = candidate
    return receipts


def finalize_recipes(plan, rows, *, mock, api_key, progress):
    """Ask DeepSeek to confirm only full recipes that passed local acceptance."""
    passed = {}
    for row in rows:
        if row.get("station_two_stage") or row.get("bayesian_revision_status") == "passed" and row.get("final_verified_recommendation"):
            continue
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
                from .sampling_risk import recipe_summary
                if audit.get("strata"):
                    evidence = "方案已通过本地正式分层回检。"
                elif "sample_size" not in audit:
                    evidence = f"验收 {audit['inspected']} 组，其中 {audit['failed']} 组不适配。"
                else:
                    evidence = recipe_summary(audit)
                text = f"{option['description']}。{evidence}"
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

    rejected = {row["task"].split(":", 1)[0] for row in rows if row["task"].endswith(":fusion")
                and (row.get("execution_release", {}).get("status") == "failed"
                     or any(report.get("basis") == "external_accuracy_reference_required"
                            for report in row.get("local_holdout_review", ())))}
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
                   reason="融合精度尚未获验收。可以单独运行单位与网格对齐，输出两份对齐后的来源。")
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
    if row.get("station_two_stage"):
        lines = [*detail.get("facts", ()), "", "小样本试跑"]
        names = {item["id"]: item["name"] for item in row.get("proposal_options", ())}
        def show_blocks(assessment):
            for block in assessment.get("range_comparison", {}).get("blocks", ()):
                if block["status"] != "EVALUATED":
                    lines.append(f"  {block['block']} · 缺少配对参考")
                    continue
                label = {"candidate": "改善", "baseline": "退化", "equivalent": "持平"}[block["lower_error"]]
                lines.append(f"  {block['block']} · {block['candidate']['count']} 点，"
                             f"RMSE 变化 {block['RMSE_change']:+.4g}（{label}）")

        for trial in row.get("development_trials", ()):
            label = names.get(trial["recipe"], trial["recipe"])
            if "candidate" in trial:
                label += (f" · {trial['count']} 点，RMSE {trial['candidate']['RMSE']:.4g}，"
                          f"原网格 RMSE {trial['baseline']['RMSE']:.4g}")
            if trial.get("failure_reason"):
                reasons = {"recipe_score_below_single_source": "试跑劣于原网格",
                           "ratio_training_unavailable": "拟合组数据不满足比例订正条件",
                           "recipe_output_missing_on_reference": "留出点处缺少有效的方案输出",
                           "insufficient_score_support": "可比较的实测参考不足"}
                label += " · " + reasons.get(trial["failure_reason"], "试跑评分未完成")
            lines.append(label)
            show_blocks(trial.get("blocked_comparison", {}))
        report = row["station_report"]
        lines.extend(["", "正式初筛"])
        if report.get("sampled_scores"):
            score = report["sampled_scores"][0]
            lines.append(f"独立实测 {report['count']} 点，RMSE {score['candidate']['RMSE']:.4g}，"
                         f"原网格 RMSE {score['baseline']['RMSE']:.4g}。")
        show_blocks(report.get("blocked_comparison", {}))
        audit = report.get("acceptance_sampling", {})
        from .sampling_risk import BAYESIAN_RULE, recipe_summary
        if audit.get("acceptance_rule") == BAYESIAN_RULE:
            lines.append(recipe_summary(audit))
        if report.get("reason"):
            lines.append(report["reason"])
        return "\n".join(lines)
    report = next((item["acceptance_sampling"] for item in row.get("local_holdout_review", ())
                   if "allowed_failures" in item.get("acceptance_sampling", {})), None)
    acceptance = []
    diagnostic = any(item.get("basis") == "external_accuracy_reference_required"
                     for item in row.get("local_holdout_review", ()))
    if diagnostic:
        acceptance.append("已完成输入产品重建诊断。正式精度验收需要独立参考，代理结果不计入整批放行概率。")
    if len(row.get("local_recipe_options", ())) == 1:
        acceptance.append("当前契约下只有一项可推荐配方；这是待验方案，不代表已完成多算法效果选优。")
    if report:
        from .sampling_risk import recipe_summary
        acceptance.append(recipe_summary(report))
        status = row.get("bayesian_revision_status")
        if status == "insufficient_untouched_evidence":
            acceptance.append("剩余未参与前轮反馈的单元不足以支持整批验收，暂不放行。")
        elif status == "no_new_recipe":
            acceptance.append("当前评分契约中没有尚未验证的可执行方案，暂不放行。")
        elif status == "no_valid_new_recipe":
            acceptance.append("DeepSeek 重荐未返回合法的新方案，暂不放行。")
        elif status == "recommendation_unavailable":
            acceptance.append("DeepSeek 重荐请求未完成，暂不放行。")
        if report.get("acceptance_rule") == "zero_failures":
            acceptance.append(f"真实不合格占比达到 {report['acceptable_share']:.0%} 时，"
                              f"计划把全漏掉的概率控制在 {report['false_accept_limit']:.0%} 以内。")
    if report or diagnostic:
        scores = next((item["sampled_scores"] for item in row.get("local_holdout_review", ())
                       if item.get("sampled_scores")), ())
        if scores:
            groups = {}
            for score in scores:
                groups.setdefault(score.get("assessment"), []).append(score)
            for stage, group in groups.items():
                title = {"gap_fill": "补齐评分", "fusion": "融合评分"}.get(stage, "留出组评分")
                if diagnostic:
                    title = "重建诊断 · " + title
                acceptance.append(f"{title}（方案与单源基线使用相同参考）")
                for label, key in (("固定方案", "candidate"), ("单源基线", "baseline")):
                    errors = [score[key]["RMSE"] for score in group]
                    acceptance.append(f"{label} · 各组 RMSE 范围 {min(errors):.4g} 至 {max(errors):.4g}")
            if not diagnostic and any("proxy" in str(score.get("reference", "")).lower() for score in scores):
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
    station_selection = summary.get("station_trial_selection")
    confirmed = set()
    for entry in entries:
        if not isinstance(entry,dict):
            continue
        key=(entry.get("task"),entry.get("recipe"))
        if not all(isinstance(k,str) for k in key) or key in seen:
            continue
        if tasks.get(key[0], {}).get("engineering_fusion_plan"):
            continue
        revision = {}
        if station_selection:
            from .station_recommendation import RETAIN

            task = tasks.get(key[0])
            if not task or key[0] in confirmed:
                continue
            proposed = entry.get("revision")
            if task.get("revision_components") and isinstance(proposed, dict):
                revision = {"revision": {field: proposed[field][:limit]
                            for field, limit in (("time_matching", 100), ("correction", 100), ("reason", 600))
                            if isinstance(proposed.get(field), str)}}
            if key[1] == RETAIN:
                recommendations.append(dict(task=key[0], id=RETAIN, steps=[], reason=str(entry.get("reason", ""))[:600],
                                            **revision))
                confirmed.add(key[0])
                continue
            if not any(score["recipe"] == key[1] and score["status"] == "scored"
                       for score in task["development_trials"]):
                continue
        if final and (key not in options or key[0] in confirmed):
            continue
        if key[0] in tasks and key[1]=='generated_graph' and tasks[key[0]].get('graph_protocol') and not final:
            from ecodata.algorithms.platform_search import platform_registry
            from ecodata.algorithms.registered_graph import validate as validate_graph
            task=tasks[key[0]]
            graph=validate_graph(entry.get('graph'),platform_registry(),task['graph_protocol']['source_count'],
                components=task['graph_protocol'].get('components'),resources=task['graph_protocol'].get('resources',()))
            quantities={f.get('physical_family') for f in task['context'].get('pair_details',[])}-{None,'UNKNOWN'}
            variable=task['graph_protocol'].get('variable')
            if not variable and len(quantities)!=1:raise ValueError('Generated graph needs one confirmed physical variable scope')
            steps=[node['algorithm'] for node in graph['nodes']]
            reason=str(entry.get('reason',''))[:600]
            recommendations.append(dict(task=key[0],id=key[1],graph=graph,steps=steps,
                name=recipe_name(steps) or '保留已满足要求的输入来源',
                description='DS自主构图 · 尚需真实执行与最终复核',scope=dict(variable=variable or next(iter(quantities))),
                reason=reason,explanation=explanation(task,reason)))
            if task.get('prepared_graph_service'):
                recommendations[-1]['autonomous_recommendation']=True
            seen.add(key);confirmed.add(key[0])
        elif key in options:
            if tasks[key[0]].get("operation") == "fusion" and key[0] in confirmed:
                continue
            seen.add(key)
            confirmed.add(key[0])
            task=tasks[key[0]]
            quantities={f.get('physical_family') for f in task['context'].get('pair_details',[])}-{None,'UNKNOWN'}
            scope=dict(variable=next(iter(quantities))) if len(quantities)==1 else {}
            reason=str(entry.get('reason',''))[:600]
            recommendation = dict(task=key[0],**options[key],scope=scope,reason=reason,explanation=explanation(task,reason),
                                  **revision)
            if task.get('prepared_graph_service'):
                recommendation.update(autonomous_recommendation=True,
                    description='确认前处理条件；准备完成后由DS根据真实输入自主构图，经数值验收后再变异，不固定使用此处示例融合算法。')
            assessment = entry.get("pretrial_assessment")
            if (not final and isinstance(assessment, str)
                    and assessment in {"likely_gain", "uncertain", "prefer_baseline"}):
                recommendation["pretrial_assessment"] = dict(outlook=assessment, reason=reason)
            recommendations.append(recommendation)
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
