"""Bounded public-entry acceptance; fixtures are NOT scientific benefit evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tests"));sys.path.insert(0,str(ROOT/"tools"))
from ecodata.algorithms import platform_search as api,attribute_registry as ar,portrait_memory as memory
from ecodata.models import portrait_policy as policy
from ecodata.algorithms.profile_matching import MODELS
from test_joint_state_platform import setup,scenario,submit

OUT=ROOT/"output/joint_state_20261024/acceptance_v3"


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding="utf-8")


def main():
    if (OUT/"acceptance.json").exists():raise ValueError("Acceptance is immutable; choose a new version")
    ar.ROOT=OUT/"attributes";policy.ROOT=OUT/"models";memory.ROOT=OUT/"memory"
    ds,req,ref=setup()
    scenarios=[scenario(req,"negative",{"negative_fraction":1}),scenario(req,"grid",{"grid_gap":1}),
               scenario(req,"structure",{"roughness":.7,"range_excess":.3})]
    api.publish_model_portrait("wrf-state-acceptance","wrf",scenarios,evidence="Synthetic task declarations, not inferred WRF sensitivities")
    templates={m:api.model_portrait_template(m) for m in MODELS}
    save(OUT/"seven_model_templates.json",templates)
    assert all(t["scientific_preferences"]["error_tolerance"] is None and not t["full_model_ready"] for t in templates.values())
    req["portrait_policy"]={"version":"wrf-state-acceptance","scenario":"structure"}
    folders=[]
    for group,offset in [("a",0),("b",1),("held",2)]:
        source=ds.copy(deep=True);source[req["variable"]].values+=offset
        request=deepcopy(req);request["profile_options"].update(evidence_role="development",group=group)
        folder=OUT/("development_"+group)
        source.to_netcdf(OUT/("visible_"+group+".nc"))
        submit(source,request,folder,reference=ref)
        folders.append(folder)
    bank=api.learn_portrait_memory(folders,"development-memory",holdout_groups=["held"],features=["std"],min_groups=2,max_distance=1.)
    req["portrait_policy"]["memory_version"]="development-memory"
    req["profile_options"]["search"]=dict(strategy="genetic",population=4,generations=4,proposal_limit=80,exact_check=True)
    req["profile_options"]["evidence_role"]="evaluation"
    learned=submit(ds,req,OUT/"learned_evolution",reference=ref)
    assert learned["state_evolution"]["optimizer_error"] is None
    assert any(p["status"]=="ACCEPTED" for p in learned["state_evolution"]["proposals"])
    assert all(r["matching"]["score"] is not None for r in learned["rows"])
    negative=ds.copy(deep=True);negative[req["variable"]].values-=12
    requests=[]
    for name in ("negative","grid"):
        request=deepcopy(req);request["portrait_policy"]={"version":"wrf-state-acceptance","scenario":name}
        request["profile_options"].pop("search")
        result=submit(negative,request,OUT/("scenario_"+name),reference=ref)
        requests.append(result)
    assert requests[0]["recommended_key"]!=requests[1]["recommended_key"]
    # Real multi-source input and all eleven existing numerical chains; not an efficacy rerun.
    from run_multisource_joint_experiment import load_data,prepare_inputs,spatial_order,GRID,StationHoldout,request as real_request
    data,stations=load_data();stamp=np.datetime64("2019-08-02T12:30")
    with xr.open_dataset(GRID) as grid:
        inputs,held,train=prepare_inputs(data[data.time.eq(stamp)],grid.sel(time=stamp),spatial_order(stations,11)[:12])
    attrs=deepcopy(ar.load("state-v1")["attributes"])
    attrs.append(dict(id="state_cost",definition="Declared graph cost proxy",tables=["data","algorithm","model"],
        unit="bounded declared tier",role="scoring",scoring_enabled=True,hard_allowed=False,provider="declared_cost",
        property=None,activation="constant_one",evidence="Existing compute-tier declarations, not measured seconds",history=["Acceptance cost consumer"]))
    api.publish_attribute_registry("state-cost-v1",attrs,parent_version="state-v1",evidence="Real-input functionality acceptance")
    real_req=real_request(stations)
    real_req["scoring"]={"registry_version":"state-cost-v1"}
    real_req["model_requirements"]={"hard":["nonnegative"]}
    real_scenario=scenario(real_req,"field-preparation",{"grid_gap":.8})
    real_scenario["issues"].append(dict(attribute="state_cost",metric="declared_cost",acceptable=0,bad=1,importance=.2,
        evidence="Explicit engineering cost tradeoff for this acceptance, NOT CMAQ model sensitivity"))
    real_scenario["evidence"]="Grid contract and declared engineering cost; no accuracy preference inferred"
    api.publish_model_portrait("cmaq-state-acceptance","cmaq",[real_scenario],evidence="Existing reviewed CMAQ field contract plus declared task")
    real_req["portrait_policy"]={"version":"cmaq-state-acceptance","scenario":"field-preparation"}
    real_req["profile_options"]["search"]=dict(strategy="genetic",population=6,generations=4,proposal_limit=100,exact_check=True)
    real=api.submit_search(inputs,real_req,StationHoldout(held,"development_acceptance"),OUT/"real_multisource",
        mode="Profile",budget=dict(objective_limit=128,wall_seconds=120))
    assert len(real["rows"])==11 and all(r["status"]=="PASS" and r["file_readback"]=="PASS" for r in real["rows"])
    assert real["state_evolution"]["optimizer_error"] is None
    source_evidence=api.inspect_model_portrait_sources([ROOT/"src/ecodata/models/profiles/wrf.json"])
    save(OUT/"model_source_inspection.json",source_evidence)
    summary=dict(status="PASS",cases=6,development_memory_records=len(bank["records"]),algorithm_response_records=len(bank["transitions"]),
        heldout_groups_excluded=not any(r["group"]=="held" for r in bank["records"]),
        real_multisource_graphs=len(real["rows"]),real_multisource_failure_count=0,
        synthetic_mutations=learned["state_evolution"]["proposals"],real_mutations=real["state_evolution"]["proposals"],
        certification=real["state_evolution"]["certification"],native_model_validation="NOT_ASSESSED",
        scientific_benefit="NOT_ASSESSED; no formal comparison rerun",
        preferences="Scoped declared task preferences, simulator scientific sensitivities remain unknown",
        limitations=["Seven templates preserve unknown sensitivities; local literal-source clues require review",
            "GA does not guarantee global true-effect optimum; exact check certifies bounded scoring domain only",
            "Finite candidate cap remains 128; truncated frozen domains are refused",
            "Cross-variable hard relations require explicitly supplied output variables and physical evidence",
            "Real data retains original unverified WGS84 working assumption; GEOS remains covariate only"])
    save(OUT/"acceptance.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if "mutations" not in k},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
