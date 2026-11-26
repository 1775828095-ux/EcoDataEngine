"""Execute each approved operation through its original scoped backend."""
from .scan import assert_current
from .service import ALGORITHMS, bundle, run_original_workbench, run_station_batch


def run(scan, tasks, folder, progress=None, *, model_id=None,api_key=None):
    from ecodata.core.recovery import session,attach
    with session(api_key=api_key,progress=progress) as recovery:
        result=_run(scan,tasks,folder,progress,model_id=model_id)
        return attach(result,folder,recovery)


def _run(scan, tasks, folder, progress=None, *, model_id=None):
    from ecodata.core.exceptions import EcoDataError
    from ecodata.core.continuation import decision,receipt
    from . import gap_search, independent_alignment, pair_processing, reference_search, station_grid

    assert_current(scan)
    files, artifacts, reports, formats, outputs, notes = [], [], [], [], [], []
    searches, failures = {}, []
    for index,(task_id,task) in enumerate(tasks,1):
        stage = folder/f"group-{index}"
        seeds = list(dict.fromkeys(step for recipe in task["recipes"] for step in recipe if step in ALGORITHMS))
        route = task.get("route")
        if task.get("search_after_rejection"):
            seeds = []
        if progress:
            progress(f"正在处理 {index}/{len(tasks)} · {task_id}")
        try:
            if task.get("search_after_rejection"):
                from .recipe_trials import SEARCH_CONTINUATION_ROUTES
                if route not in SEARCH_CONTINUATION_ROUTES:
                    raise ValueError("推荐失败后的搜索尚未绑定此执行路线")
            runtime_report = []
            if task.get("runtime_fields") and route not in {"windowed_grid_recipe", "indexed_table_recipe"}:
                from .release_gate import runtime
                steps = [step for recipe in task["recipes"] for step in recipe]
                target = (task.get("target_unit") if route == "structure_conversion" and
                          all("units.harmonize" in recipe for recipe in task["recipes"]) else None)
                runtime_report = runtime(scan, {"fields": task["runtime_fields"], "steps": steps,
                                                "segmented_table_target": target}, folder)
            if route == "station_grid":
                result = station_grid.run(scan,task_id,task,stage,model_id,progress)
            elif route == "windowed_grid_recipe":
                from .windowed_grid_recipe import run as run_grid_recipe
                result = run_grid_recipe(scan, task_id, task, stage, model_id, progress)
            elif route == "indexed_table_recipe":
                from .indexed_table_recipe import run as run_table_recipe
                result = run_table_recipe(scan, task_id, task, stage, model_id, progress)
            elif route == "structure_conversion":
                result = pair_processing.run_conversion(scan,task_id,task,stage,model_id,progress)
            elif route == "statistical_preprocessing":
                from .statistical_preprocessing import run as run_statistics
                result=run_statistics(scan,task_id,task,stage,model_id,progress)
            elif route == "cross_variable_alignment":
                result = pair_processing.run_cross_variable(scan,task_id,task,stage,model_id,progress)
            elif route == "pair_processing":
                result = pair_processing.run(scan,task_id,task,stage,model_id,progress)
            elif route == "observation_gap_search":
                result = gap_search.run(scan,task_id,task,stage,model_id,progress)
            elif route in {"independent_grid_search","independent_alignment","native_reference_search"}:
                backend = reference_search if route == "native_reference_search" else independent_alignment
                result = backend.run(scan,[(task_id,dict(task,seeds=seeds))],stage,progress,model_id=model_id)
            elif task.get("action") == "fusion" and "state" in task:
                result = run_station_batch([(task_id,task["state"],seeds)],stage,model_id,progress)
            else:
                raise ValueError("该任务未绑定可执行入口")
        except Exception as exc:
            from .issues import exception
            failures.append(dict(task=task_id,reason=str(exc),issues=exception(exc)))
            continue
        formats.extend(result["source_formats"])
        outputs.append(result["target_format"])
        notes.append(result["local"]["reason"])
        reports.append(dict(task=task_id,backend=route or "station_search",report=result["original_report"],
                            input_runtime_checks=runtime_report))
        for key, value in (result.get("grid_searches") or result.get("batch_searches") or {}).items():
            searches[key] = dict(value,backend=route or "station_search")
        prefix = f"group-{index}-"
        files.extend((prefix+name,raw) for name,raw in result["files"])
        artifacts.extend(dict(row,name=prefix+row["name"]) for row in result["local"]["artifacts"])
    research = None
    research_tasks = scan.get("research_tasks",{}).get(model_id,[])
    if research_tasks:
        from .research_dispatch import run as run_research
        try:
            research = run_research(scan,research_tasks,folder/"research",progress)
        except (EcoDataError,ValueError,OSError,RuntimeError,KeyError) as exc:
            failures.append(dict(task='研究附加步骤',reason=str(exc)))
    original = None
    native = scan.get("native_model_probes",{}).get(model_id,{})
    if native.get("status")=="declared":
        try:
            if progress:
                progress("正在执行已声明的原版模型组件")
            original = run_original_workbench(scan,model_id,folder/"declared-component",progress,
                                              declared_only=True)
        except (EcoDataError, ValueError, OSError, RuntimeError, KeyError) as exc:
            failures.append(dict(task="原版模型组件",reason=str(exc)))
        else:
            formats.extend(original["source_formats"])
            reports.append(dict(task="原版模型组件",report=original["manifest"]))
            if not files:
                original["failed_tasks"] = failures
                prior=original.get('manifest',{}).get('continuation',{})
                original['continuation']=receipt([*prior.get('decisions',[]),
                    *[decision('operation',f['task'],f['reason']) for f in failures]],
                    [*prior.get('completed_steps',[]),'原版模型组件'])
                if research:
                    original["research_archive"] = research["original_archive"]
                    original["research_records"] = research["research_records"]
                return original
    assert_current(scan)
    if not files and research:
        research["failed_tasks"] = failures
        research['continuation']=receipt([decision('operation',f['task'],f['reason']) for f in failures],['研究附加步骤'])
        return research
    continuation=receipt([decision('operation',f['task'],f['reason']) for f in failures],
        [r['task'] for r in reports])
    if not files and not failures:
        continuation=receipt([decision('operation','processing','未找到满足契约的可执行步骤；未制造数据产品')])
    notes.extend(f"{item['task']} 处理失败：{item['reason']}" for item in failures)
    result = dict(mode="selected_operations",files=files,grid_searches=searches,search=None,
        source_formats=list(dict.fromkeys(formats)),target_format=" / ".join(dict.fromkeys(outputs)),
        original_report=reports,failed_tasks=failures,continuation=continuation,
        processing_completed=bool(files),local=dict(artifacts=artifacts,native_ready=False,
            reason=" ".join(dict.fromkeys(notes)),
            model_input=dict(whole_model_missing=notes if model_id and model_id!='data_only' else [])))
    if original:
        result["component_archive"] = original["original_archive"]
        result["archive_members"] = original["archive_members"]
    if research:
        result["research_archive"] = research["original_archive"]
        result["research_records"] = research["research_records"]
    bundle(result)
    return result
