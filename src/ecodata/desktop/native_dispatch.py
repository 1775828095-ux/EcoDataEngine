"""Bounded original model probes and automatic preparation admission."""
import json
from pathlib import PurePosixPath

from .scan import assert_current

MODELS = {"swat_plus","wrf","cmaq","camx","calpuff"}
TASK_SCHEMA = "ecodata.model-task.v1"
DECLARATIONS = {TASK_SCHEMA,"ecodata.engineering-semantics.v1","ecodata.cmaq.local-project.v1"}
CONTROL_ROLES = {"engineering_configuration","engineering_dependency"}


def inspect(scan, model):
    from ecodata.models.component_service import capabilities
    from ecodata.models.converter_contract import load_converter
    from ecodata.models.native_source_inspection import inspect_sources, probe_sources

    if model not in MODELS:
        return {}
    assert_current(scan)
    cache = scan.setdefault("native_model_probes", {})
    if model in cache:
        return cache[model]
    members, facts, declared = [], [], []
    for row in scan["rows"]:
        name = row["name"]
        artifact = row.get("original_probe_payload",b"")
        members.append((name,artifact))
        facts.append(dict(file=name,facts=row.get("original_native_header"),
                          payload_read="original_probe_payload" in row or row["bytes"]==0))
        if not artifact or not name.casefold().endswith(".json"):
            continue
        try:
            document = json.loads(artifact)
        except (ValueError,UnicodeError):
            continue
        if isinstance(document,dict) and document.get("schema") in DECLARATIONS:
            if document.get("model_id",model)==model and (document["schema"]!="ecodata.cmaq.local-project.v1" or model=="cmaq"):
                declared.append(dict(file=name,schema=document["schema"],operation=document.get("operation")))
    converter = load_converter(model)
    # A registered, byte-exact scientific declaration uses the same admission
    # path as an uploaded engineering-semantics document. It is not a claim
    # that the agent opened the control this round, or a native-model PASS.
    if scan.get('saved_input_context'):
        from ecodata.models.input_context import find
        context=find(scan['path'],model)
        if context:
            for filename,doc in context.get('documents',{}).items():
                if doc.get('schema')=='ecodata.engineering-semantics.v1' and doc.get('model_id')==model and any(
                    entry.get('role')=='model_input' and entry.get('confirmed') is True for entry in doc.get('resources',[])):
                    declared.append(dict(file=filename,schema=doc['schema'],operation=None,
                        origin='hash_bound_saved_input_context',source_sha256=context['source_sha256'],
                        evidence_status=context['status'],scope='已登记声明；实际工程与科学约定仍须处理时验证'))
    present={name for name,_ in members}
    if scan.get('on_demand'):facts.append(dict(file='',partial_inventory=True))
    for name,evidence in scan.get('directory_members',{}).items():
        if name not in present:
            members.append((name,b''));facts.append(dict(file=name,payload_read=False,present_in_directory=True))
    hook = getattr(converter,"inspect_sources",None)
    prefix_hook = getattr(converter,"probe_sources",None)
    try:
        report = hook(members,facts) if hook else inspect_sources(model,members,facts)
        prefixes = [(name,data[:65536]) for name,data in members if data]
        prefix = prefix_hook(prefixes) if prefix_hook else probe_sources(model,prefixes)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        report = dict(claims=[],issues=[dict(message=str(exc))])
        prefix = {}
    claims = report.get("claims",[])
    controls = [c["file"] for c in claims if c.get("role") in CONTROL_ROLES]
    controls.extend(prefix.get("controls",[]))
    if model=="wrf":
        controls.extend(name for name,data in members
                        if PurePosixPath(name).name.casefold().startswith("namelist.input")
                        and b"&wrfvar1" in data.lower() and b"&wrfvar18" in data.lower())
    if model=="swat_plus":
        controls.extend(row["name"] for row in scan["rows"] if row.get("original_sqlite_header"))
    component_tasks = [x for x in declared if x["schema"]==TASK_SCHEMA]
    registered = capabilities(model)
    allowed = {r["task"] for r in registered}
    single = False
    scientific=[r for r in scan['rows'] if any(f.get('unit') and not f.get('axis') for f in r.get('fields',[]))]
    if not (declared or controls) and len(scientific)==1:
        from ecodata.algorithms.profile_matching import model_table
        from ecodata.models.scientific_schema import identity
        target_names = {f["name"] for role in model_table(model)["roles"] if not role["record_only"]
                        for f in role["fields"]}
        single = any((identity(f["name"],f)[0] or f['name']) in target_names and f.get("unit") and not f["axis"]
                     for f in scientific[0]["fields"])
    status = ("declared" if len(component_tasks)==1 and component_tasks[0]["operation"] in allowed
              else "project" if not component_tasks and (declared or controls or single) else "unavailable")
    issues = [*report.get("issues",[]),*prefix.get("issues",[])]
    if component_tasks and status=="unavailable":
        issues.append(dict(message="需唯一且受原版支持的组件任务声明。"))
    result = dict(status=status, declarations=declared, claims=claims,
                  component_candidate=single or status=='declared',
                  issues=issues, component_count=len(registered), controls=list(dict.fromkeys(controls)))
    assert_current(scan)
    cache[model] = result
    return result


def explain(report):
    if not report:
        return []
    lines = []
    for row in report["declarations"]:
        name = PurePosixPath(row["file"]).name
        lines.append(f"本地识别到原版声明 · {name} · {row['operation'] or row['schema']}")
    if report["controls"]:
        lines.append(f"已按目标模型识别 {len(report['controls'])} 份工程控制或关联资源。")
    for issue in report["issues"][:8]:
        lines.append("工程输入 · "+str(issue.get("message","")))
    return lines
