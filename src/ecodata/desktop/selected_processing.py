"""Execute each approved operation through its original scoped backend."""
from .scan import assert_current
from .service import ALGORITHMS, bundle, run_original_workbench, run_station_batch


def run(scan, tasks, folder, progress=None, *, model_id=None):
    from ecodata.core.exceptions import EcoDataError
    from . import gap_search, independent_alignment, pair_processing, reference_search, station_grid

    assert_current(scan)
    files, artifacts, reports, formats, outputs, notes = [], [], [], [], [], []
    searches, failures = {}, []
    for index,(task_id,task) in enumerate(tasks,1):
        stage = folder/f"group-{index}"
        seeds = list(dict.fromkeys(step for recipe in task["recipes"] for step in recipe if step in ALGORITHMS))
        route = task.get("route")
        if progress:
            progress(f"正在处理 {index}/{len(tasks)} · {task_id}")
        try:
            if route == "station_grid":
                result = station_grid.run(scan,task_id,task,stage,model_id,progress)
            elif route == "structure_conversion":
                result = pair_processing.run_conversion(scan,task_id,task,stage,model_id,progress)
            elif route == "statistical_preprocessing":
                from .statistical_preprocessing import run as run_statistics
                result=run_statistics(scan,task_id,task,stage,model_id,progress)
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
        reports.append(dict(task=task_id,backend=route or "station_search",report=result["original_report"]))
        for key, value in (result.get("grid_searches") or result.get("batch_searches") or {}).items():
            searches[key] = dict(value,backend=route or "station_search")
        prefix = f"group-{index}-"
        files.extend((prefix+name,raw) for name,raw in result["files"])
        artifacts.extend(dict(row,name=prefix+row["name"]) for row in result["local"]["artifacts"])
    research = None
    research_tasks = scan.get("research_tasks",{}).get(model_id,[])
    if research_tasks:
        from .research_dispatch import run as run_research
        research = run_research(scan,research_tasks,folder/"research",progress)
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
                if research:
                    original["research_archive"] = research["original_archive"]
                    original["research_records"] = research["research_records"]
                return original
    assert_current(scan)
    if not files and research:
        research["failed_tasks"] = failures
        return research
    if not files:
        raise ValueError("；".join(f"{item['task']}：{item['reason']}" for item in failures))
    notes.extend(f"{item['task']} 处理失败：{item['reason']}" for item in failures)
    result = dict(mode="selected_operations",files=files,grid_searches=searches,search=None,
        source_formats=list(dict.fromkeys(formats)),target_format=" / ".join(dict.fromkeys(outputs)),
        original_report=reports,failed_tasks=failures,local=dict(artifacts=artifacts,native_ready=False,
            reason=" ".join(dict.fromkeys(notes)),
            model_input=dict(whole_model_missing=notes)))
    if original:
        result["component_archive"] = original["original_archive"]
        result["archive_members"] = original["archive_members"]
    if research:
        result["research_archive"] = research["original_archive"]
        result["research_records"] = research["research_records"]
    bundle(result)
    return result
