"""Reproducible inventory only: no downloads, search or numerical batches."""
from pathlib import Path
import csv,hashlib,inspect,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from ecodata.algorithms.platform_search import platform_registry,FAMILIES
from ecodata.algorithms.capabilities import describe_algorithm
from ecodata.algorithms.plugins import CatalogOnlyPlugin
from ecodata.algorithms.numerical_search import save

OUT=ROOT/"output/platform_integration_20261021"
ACCEPTED_RUNS={"spatial_toolkit_semantic_final","spatial_page_semantic_final","temporal_toolkit_delivery","temporal_page_delivery",
               "multisource_toolkit_delivery","multisource_page_delivery","curvilinear_contract_fixed","temporal_instant"}


def inventory():
    registry=platform_registry();executed={};scenario_matches={}
    for folder in sorted((OUT/"runs").glob("*")):
        if folder.name not in ACCEPTED_RUNS:continue
        summary=folder/"summary.json"
        if not summary.is_file():continue
        result=json.loads(summary.read_text(encoding="utf-8"))
        for row in result["rows"]:
            if row["status"]!="PASS":continue
            for node in row["graph"]:
                executed.setdefault(node["operation"],[]).append(str(folder/"executions"/row["key"]/"result.json"))
        decision=folder/"decision.json"
        if decision.is_file():
            for identifier,match in json.loads(decision.read_text(encoding="utf-8"))["matches"].items():
                scenario_matches.setdefault(identifier,{})[folder.name]=dict(eligible=match["eligible"],reasons=match["reasons"])
    rows=[]
    for spec in registry.specs():
        plugin=registry.get(spec.id);record=describe_algorithm(spec)
        source=Path(inspect.getfile(type(plugin)))
        family=[name for name,categories in FAMILIES.items() if spec.category in categories]
        boundary=spec.id=="validation.result"
        state=spec.input_contract.get("requires_fitted_state",False)
        catalog=isinstance(plugin,CatalogOnlyPlugin)
        layer="MODEL_PREPROCESSING" if spec.category=="model_preprocessing" else "RESEARCH_CATALOG" if catalog else "CONSTRAINT_OR_QUALITY" if spec.category in {"validation","raster_constraint"} else "GENERAL_ALGORITHM"
        if catalog:binding="BLOCKED_CATALOG_NO_NUMERICAL_BACKEND"
        elif spec.status!="available":binding="CONDITIONAL_BLOCKED_STATUS"
        elif boundary:binding="SHARED_RESULT_QUALITY_BOUNDARY; existing validator reused, not a free prediction node"
        elif state:binding="HOURLY_STATE_BOUND_ADAPTER; explicit matching legal history required"
        elif family:binding="PUBLIC_REGISTRY_RUN_BINDING" if spec.category!="raster_constraint" else "PUBLIC_POSTPROCESSOR_BINDING"
        else:binding="MISSING_TASK_FAMILY_BINDING"
        conditions=list(spec.preconditions)+list(spec.runtime_requirements)
        if spec.notes:conditions.append(spec.notes)
        conditions.extend(name for flag,name in [(spec.requires_training,"explicit allow_training and legal fitting data"),(spec.requires_weights,"explicit compatible checkpoint consent/resource; no automatic historical reuse"),(spec.requires_observations,"observations/train port required"),(spec.requires_auxiliary,"auxiliary port and units required")] if flag)
        if state:conditions.append("Existing general plugin.run does not accept its fitted state; use public hourly_recovery task")
        if spec.category=="model_preprocessing":conditions.append("Explicit model task/opt-in; native simulator reading NOT verified this round")
        if boundary:conditions.append("Not an independent predictive operator; AlgorithmQualityGate delegates to the same ResultQualityGate")
        if not record.capabilities:conditions.append("No advertised capability: cannot satisfy nonempty required-capability tasks by name")
        evidence=sorted(set(executed.get(spec.id,[])))
        rows.append(dict(id=spec.id,status=spec.status,name=spec.name,scientific_role=spec.category,layer=layer,source_url=spec.source_url,license=spec.license,
            source_code=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_kinds=spec.input_kinds,semantic_types=spec.semantic_types,
            scale_range=spec.scale_range,capabilities=record.capabilities,physical_constraints=spec.physical_constraints,input_contract=spec.input_contract,output_contract=spec.output_contract,
            compatibility_callable=callable(plugin.compatibility),pipeline_callable=callable(plugin.run),public_binding=binding,task_families=family,
            required_observations=spec.requires_observations,required_auxiliary=spec.requires_auxiliary,required_training=spec.requires_training,required_weights=spec.requires_weights,
            unit_time_space_constraints="See actual compatibility source + input_contract + preconditions; no universal unit/CRS support is inferred",conditions=conditions,
            this_round_execution="REAL_PASS" if evidence else "NOT_EXECUTED_THIS_ROUND",this_round_evidence=evidence,
            historical_evidence_declared=spec.verification_evidence,scenario_compatibility=scenario_matches.get(spec.id,{}),
            capability_consumer="AlgorithmSelector.decide -> output-path assess_graph -> semantic_proposal_weights -> Pipeline.run_dag.require_graph",
            state_dependency="Named source/state ports and actual parent artifacts; scientific adapter owns validation; generic layer does not invent fitted state"))
    save(OUT/"algorithm_library_matrix.json",dict(version="platform_search_v2",base_default_count=69,default_public_count=len(rows),
        optional_existing_plugin=dict(id="raster_task.liif_aod",binding="platform_registry(weights=explicit_AOD_checkpoint)",status="CONDITIONAL_NOT_EXECUTED_THIS_ROUND",source="src/ecodata/algorithms/raster_reconstruction_task.py",conditions="Locally trained dimensionless AOD checkpoint, WGS84 2D finite field, verified 1.5..3x; not ERA5/RGB/LAI weights"),algorithms=rows))
    with (OUT/"algorithm_library_matrix.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows)
    from ecodata.models.library import ModelLibrary
    save(OUT/"model_adapter_layer.json",dict(profiles=ModelLibrary().list(),entry="ecodata.models.scenario + existing model page",new_native_validation="NOT_RUN",general_evolutionary_pool=False))
    print("Inventory",len(rows),"default registrations; optional AOD LIIF separate; real executed",sum(bool(r["this_round_evidence"]) for r in rows))
    return rows


if __name__=="__main__":inventory()
