"""Evidence-bearing, immutable scenario preferences over reviewed model fields."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import hashlib
import numpy as np

from ecodata.algorithms.attribute_registry import identifier, digest

ROOT=Path(__file__).resolve().parents[3]/"data/model_portraits"
METRICS={"missing_fraction","negative_fraction","range_excess","peak_loss","roughness","grid_gap","coarse_error","observation_bias","declared_cost","detail_error","expected_error"}


def template(model_id):
    from ecodata.algorithms.profile_matching import model_table
    model=model_table(model_id)
    return dict(model_id=model_id,reviewed_input_contract=model,
        scientific_preferences={"source_preference":None,"spatial_structure_sensitivity":None,
            "error_tolerance":None,"cross_variable_coordination":None,"scenario_specific_weights":None},
        source_evidence_candidates=[],scenarios=[],
        next_step="Review source/doc evidence, declare task scales/preferences, or attach actual model-sensitivity experiments before publishing",
        automatically_inferred_sensitivity=False,full_model_ready=False)


def publish(version, model_id, scenarios, *, evidence, parent_version=None):
    from ecodata.algorithms.profile_matching import model_table
    identifier(version)
    model=model_table(model_id)
    if not evidence or not isinstance(scenarios,list) or not scenarios:
        raise ValueError("Reviewed scenario definitions and evidence required")
    seen=set()
    for scenario in scenarios:
        if set(scenario)!={"id","task","role","field","issues","coupled_constraints","evidence"}:
            raise ValueError("Incomplete model scenario")
        identifier(scenario["id"])
        if scenario["id"] in seen:raise ValueError("Duplicate scenario")
        seen.add(scenario["id"])
        if scenario["task"] not in {"spatial","temporal","multisource"} or not scenario["evidence"]:
            raise ValueError("Declare scenario task and evidence")
        role=next((r for r in model["roles"] if r["role"]==scenario["role"] and not r["record_only"]),None)
        if role is None or not any(f["name"]==scenario["field"] for f in role["fields"]):
            raise ValueError("Scenario field is outside the reviewed model contract")
        ids=set();metrics=set()
        if not isinstance(scenario["issues"],list) or not scenario["issues"]:
            raise ValueError("Explicit task preferences required; never guess simulator sensitivity")
        for item in scenario["issues"]:
            if set(item)!={"attribute","metric","acceptable","bad","importance","evidence"}:
                raise ValueError("Incomplete issue definition")
            identifier(item["attribute"])
            if item["attribute"] in ids or item["metric"] in metrics or item["metric"] not in METRICS:
                raise ValueError("Duplicate issue or unimplemented scientific measurement")
            ids.add(item["attribute"])
            metrics.add(item["metric"])
            values=[item[k] for k in ("acceptable","bad","importance")]
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in values):
                raise ValueError("Finite scales and preferences required")
            if item["bad"]<=item["acceptable"] or item["importance"]<0 or not item["evidence"]:
                raise ValueError("Invalid preference scale or missing evidence")
        if sum(i["importance"] for i in scenario["issues"])<=0:raise ValueError("At least one positive priority required")
        for constraint in scenario["coupled_constraints"]:
            if set(constraint)!={"kind","variables","target","atol","evidence"} or constraint["kind"]!="sum":
                raise ValueError("Only explicitly declared sum-to-field coupled constraints are implemented")
            if not constraint["variables"] or not isinstance(constraint["target"],str) or not constraint["evidence"]:
                raise ValueError("Coupled physical relation requires variables, target field and evidence")
            if not isinstance(constraint["atol"],(int,float)) or not np.isfinite(constraint["atol"]) or constraint["atol"]<0:
                raise ValueError("Invalid coupled-constraint tolerance")
    parent=load(parent_version) if parent_version else None
    if parent and parent["model_id"]!=model_id:raise ValueError("Cannot change a portrait's model lineage")
    snapshot=dict(version=version,model_id=model_id,model_sha256=model["profile_sha256"],scenarios=deepcopy(scenarios),
        evidence=evidence,parent_sha256=parent["sha256"] if parent else None,created_at=datetime.now(timezone.utc).isoformat(),
        sensitivity=None,sensitivity_status="UNKNOWN_UNLESS_INDEPENDENT_MODEL_EXPERIMENTS",native_acceptance="NOT_ASSESSED")
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/(version+".json")).open("x",encoding="utf-8") as stream:
        json.dump(dict(snapshot=snapshot,sha256=digest(snapshot)),stream,ensure_ascii=False,indent=2,allow_nan=False)
    return dict(snapshot,sha256=digest(snapshot))


def load(version):
    value=json.loads((ROOT/(identifier(version)+".json")).read_text(encoding="utf-8"))
    if set(value)!={"snapshot","sha256"} or digest(value["snapshot"])!=value["sha256"]:
        raise ValueError("Model portrait fingerprint mismatch")
    return dict(value["snapshot"],sha256=value["sha256"])


def resolve(request, table, registry):
    settings=request["portrait_policy"]
    if not isinstance(settings,dict) or set(settings)-{"version","scenario","memory_version"} or not {"version","scenario"}<=set(settings):
        raise ValueError("portrait_policy requires version/scenario and optional development memory_version")
    policy=load(settings["version"])
    if policy["model_id"]!=table["model_id"] or policy["model_sha256"]!=table["profile_sha256"]:
        raise ValueError("Model portrait no longer matches the reviewed model")
    scenario=next((s for s in policy["scenarios"] if s["id"]==settings["scenario"]),None)
    selection=request["model_profile"]
    if scenario is None or any(scenario[k]!=selection[k] for k in ("role","field")) or scenario["task"]!=request["task"]:
        raise ValueError("Model portrait scenario/field/task mismatch")
    definitions={a["id"]:a for a in registry["attributes"]}
    for item in scenario["issues"]:
        a=definitions.get(item["attribute"])
        if a is None or a["provider"]!=("declared_cost" if item["metric"]=="declared_cost" else "state_metric") or a["property"]!=(None if item["metric"]=="declared_cost" else item["metric"]):
            raise ValueError("Scenario requires a registered state measurement: "+item["attribute"])
    if request.get("model_requirements",{}).get("priorities"):
        raise ValueError("Choose scenario preferences or explicit priorities; do not silently combine them")
    return dict(policy_version=policy["version"],policy_sha256=policy["sha256"],scenario=deepcopy(scenario),
                memory_version=settings.get("memory_version"),active_issue_count=sum(definitions[i["attribute"]]["scoring_enabled"] and i["importance"]>0 for i in scenario["issues"]),sensitivity=policy["sensitivity"],origin="explicit scoped task preferences; not inferred simulator sensitivity")


def inspect_model_sources(paths):
    """Read local source/config evidence. Never run code or infer sensitivities."""
    records=[];facts=[]
    for path in paths:
        path=Path(path).resolve()
        if path.suffix.lower() not in {".py",".f",".f90",".f95",".c",".h",".txt",".json",".yaml",".yml",".md",".nml",".registry"} and not path.name.lower().startswith("registry"):
            raise ValueError("Unsupported source-evidence format")
        raw=path.read_bytes()
        if len(raw)>4_000_000:raise ValueError("Inspect bounded individual source files")
        if path.suffix.lower()==".json":
            def walk(value):
                if isinstance(value,dict):
                    if isinstance(value.get("name"),str) and isinstance(value.get("unit",value.get("units")),str):
                        facts.append(dict(variable=value["name"],unit=value.get("unit",value.get("units")),
                            path=str(path),sha256=hashlib.sha256(raw).hexdigest(),syntax="literal_json_field",
                            status="CANDIDATE_REQUIRES_REVIEW",affects_selection=False))
                    for child in value.values():walk(child)
                elif isinstance(value,list):
                    for child in value:walk(child)
            walk(json.loads(raw))
        for number,line in enumerate(raw.decode("utf-8",errors="replace").splitlines(),1):
            if line.lstrip().startswith("state ") and path.name.lower().startswith("registry"):
                quoted=re.findall(r'"([^"\n]*)"',line)
                if len(quoted)>=3:
                    facts.append(dict(variable=quoted[-3],description=quoted[-2],unit=quoted[-1],
                        path=str(path),line=number,sha256=hashlib.sha256(raw).hexdigest(),syntax="wrf_registry_state",
                        status="CANDIDATE_REQUIRES_REVIEW",affects_selection=False))
            if re.search(r"units?|standard_name|stagger|memoryorder|nf90_get_var|nf90_get_att|temporal_resolution|spatial_resolution",line,re.I):
                records.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),line=number,text=line[:1200],
                    status="CANDIDATE_REQUIRES_REVIEW",affects_selection=False))
    return dict(records=records,structured_facts=facts,sensitivity=None,hard_requirements_promoted=0,
        scope="Literal source/document clues only; neither proof of scientific preference nor automatic code understanding")


def validate_coupled(dataset, policy):
    checks=[]
    for relation in policy["scenario"]["coupled_constraints"]:
        names=[*relation["variables"],relation["target"]]
        if any(n not in dataset for n in names):
            checks.append(dict(status="FAIL",relation=relation,reason="Required coupled output variable missing"));continue
        import xarray as xr
        try:
            fields=xr.align(*(dataset[n] for n in names),join="exact")
            if any(a.dims!=fields[0].dims for a in fields) or len({a.attrs.get("units") for a in fields})!=1 or not fields[0].attrs.get("units"):
                raise ValueError("Coupled dimensions/units conflict")
            error=np.asarray(sum(fields[:-1])-fields[-1])
            passed=np.isfinite(error).all() and np.abs(error).max()<=relation["atol"]
            checks.append(dict(status="PASS" if passed else "FAIL",relation=relation,max_abs_error=float(np.max(np.abs(error))) if np.isfinite(error).all() else None))
        except (ValueError,TypeError) as exc:
            checks.append(dict(status="FAIL",relation=relation,reason=str(exc)))
    return dict(status="FAIL" if any(c["status"]=="FAIL" for c in checks) else "PASS",checks=checks)
