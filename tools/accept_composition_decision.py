"""Bounded source/time composition acceptance, including existing real products.

Synthetic cases test control flow, not environmental effect superiority. Real
products share station evidence and are NOT independent observational sources.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests")]
from ecodata.algorithms import platform_search as api, attribute_registry as ar
from ecodata.models import portrait_policy as policy
from ecodata.algorithms.composition_decision import Composition
from test_composition_decision import fixture, stage, submit

OUT=ROOT/"output/composition_20261024/acceptance_v5"


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding="utf-8")


def main():
    if (OUT/"acceptance.json").exists():raise ValueError("Choose a new acceptance version; do not overwrite evidence")
    old_ar,old_policy=ar.ROOT,policy.ROOT
    ar.ROOT=OUT/"attributes";policy.ROOT=OUT/"models"
    datasets,req=fixture()
    synthetic=submit(datasets,req,OUT/"synthetic72")
    assert synthetic["counts"]["exact_usable_count"]==72
    # Same grammar, changed model task preference; numerical pipeline remains.
    req["portrait_policy"]["scenario"]="grid"
    changed=submit(datasets,req,OUT/"changed_model72")
    # Changing data changes predicted/observed issue severity and its decisions.
    positive=[ds.copy(deep=True) for ds in datasets]
    for ds in positive:ds[req["variable"]].values+=20
    req["portrait_policy"]["scenario"]="negative"
    changed_data=submit(positive,req,OUT/"changed_data72")
    assert synthetic["recommended_key"]!=changed["recommended_key"]
    assert synthetic["recommended_key"]!=changed_data["recommended_key"]
    # New isolated model/attribute lineage for temporal fixture.
    ar.ROOT=OUT/"temporal_attributes";policy.ROOT=OUT/"temporal_models"
    datasets,req=fixture(True)
    req["composition"]["source_stages"]={k:[stage("grid",["spatial.nearest","spatial.bilinear"])] for k in ("A","B")}
    req["composition"]["windows"]=[dict(id="early",start=req["target_times"][0],end=req["target_times"][2]),dict(id="late",start=req["target_times"][1],end=req["target_times"][3])]
    temporal=submit(datasets,req,OUT/"temporal64")
    assert temporal["counts"]["exact_usable_count"]==64
    req["composition"]["coordination"]=dict(max_source_disagreement=.1,evidence="Deliberate hard-gate failure acceptance")
    rejected=submit(datasets,req,OUT/"rejected64")
    assert rejected["counts"]["exact_usable_count"]==0
    # Publish an explicit opt-in engineering policy, never simulator sensitivity.
    ar.ROOT,policy.ROOT=old_ar,old_policy
    registry_version="composition-engineering-state-v1"
    attrs=deepcopy(ar.load()["attributes"])
    for metric in ("grid_gap","negative_fraction"):
        attrs.append(dict(id="composition_"+metric,definition="Measured visible field "+metric,tables=["data","algorithm","model"],
            unit="fraction",role="scoring",scoring_enabled=True,hard_allowed=False,provider="state_metric",property=metric,
            activation="constant_one",evidence="Grid equality and negative fraction are directly measured; not accuracy claims",history=["Composition engineering acceptance"]))
    if not (ar.ROOT/(registry_version+".json")).exists():
        api.publish_attribute_registry(registry_version,attrs,evidence="Explicit opt-in composition engineering measurements")
    model_version="composition-cmaq-pm25-v1"
    scenario=dict(id="grid-and-sign",task="multisource",role="background",field="PM25",
        issues=[dict(attribute="composition_"+m,metric=m,acceptable=0.,bad=1.,importance=1.,evidence="Explicit PM2.5 grid/sign preparation requirement, not CMAQ sensitivity") for m in ("grid_gap","negative_fraction")],
        coupled_constraints=[],evidence="Declared nonnegative PM2.5 field on exact target grid")
    if not (policy.ROOT/(model_version+".json")).exists():
        api.publish_model_portrait(model_version,"cmaq",[scenario],evidence=scenario["evidence"])
    historical=ROOT/"output/joint_state_20261024/acceptance_v3/real_multisource/summary.json"
    old=json.loads(historical.read_text(encoding="utf-8"));sources=[];lineage=[]
    for identifier,name in [("fusion.idw","IDW_product"),("station.pykrige_ordinary","Kriging_product")]:
        row=next(r for r in old["rows"] if r["steps"][0]==identifier)
        path=Path(row["product"])
        with xr.open_dataset(path) as ds:data=ds[["pm25"]].load()
        data.attrs["source_id"]=name
        data.pm25.attrs["physical_variable"]="PM25"
        data.pm25.attrs["data_kind"]="continuous"
        data.pm25.attrs["spatial_semantics"]="point"
        data.attrs["composition_lineage"]="Derived from shared station evidence; not independent sensors"
        prepared=OUT/"real_inputs"/(name+".nc");prepared.parent.mkdir(parents=True,exist_ok=True);data.to_netcdf(prepared)
        sources.append((name,data,prepared));lineage.append(dict(original=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prepared=str(prepared),canonical_physical_identity="PM25",data_kind="continuous, from the archived task's explicitly declared semantic type"))
    grid=sources[0][1]
    chain=[stage("grid",["spatial.nearest","spatial.bilinear","spatial.xarray_cubic","raster_task.pchip"]),stage("clip",[None,"raster_task.nonnegative"])]
    request=dict(task="multisource",variable="pm25",unit="ug/m3",semantic_type="continuous",nonnegative=True,
        target_lat=((grid.lat.values[:-1]+grid.lat.values[1:])/2).tolist(),target_lon=((grid.lon.values[:-1]+grid.lon.values[1:])/2).tolist(),
        parameters={},model_profile=dict(id="cmaq",role="background",field="PM25"),
        scoring=dict(registry_version=registry_version),portrait_policy=dict(version=model_version,scenario="grid-and-sign"),
        composition=dict(source_stages={k:chain for k,_,_ in sources},post_stages=[],
            fusion_options=[dict(algorithm="fusion.primary_weighted_mean",primary=k,primary_share=.75,evidence="Declared dominant-source alternatives, no independence/uncertainty claim") for k,_,_ in sources],
            search=dict(screen_pool=8,survivors=2,population=2,generations=6,exact_check=True)))
    config=dict(mode="Profile",request=request,inputs=[dict(path=str(p),role="primary") for _,_,p in sources],budget=dict(objective_limit=288,wall_seconds=300),seed=11)
    save(OUT/"real_request.json",config);save(OUT/"real_input_lineage.json",lineage)
    real=api.submit_file_request(config,OUT/"real288")
    assert real["counts"]["grammar_combinations"]==288 and real["counts"]["exact_count_complete"] and real["counts"]["exact_usable_count"]>0
    automatic=deepcopy(config);automatic["request"]["composition"]={}
    automatic["budget"]=dict(objective_limit=32,proposal_limit=50,wall_seconds=180)
    save(OUT/"automatic_real_request.json",automatic)
    automatic_engine=Composition([("primary",ds) for _,ds,_ in sources],automatic["request"],api.platform_registry())
    save(OUT/"automatic_grammar_count.json",dict(grammar_combinations=automatic_engine.count,
        factors=automatic_engine.sizes,exact_usable_count=None,reason="Exact structural count only; automatic enlarged domain not exhaustively executed"))
    automatic_result=api.submit_file_request(automatic,OUT/"automatic_public")
    assert automatic_result["best"] is not None
    registry=api.platform_registry()
    inventory=[s.as_dict() for s in registry.specs()]
    for item in inventory:item["scale_range"]=[float(v) if np.isfinite(v) else None for v in item["scale_range"]]
    save(OUT/"registry_inventory.json",inventory)
    reports=dict(synthetic72=synthetic,changed_model72=changed,changed_data72=changed_data,temporal64=temporal,rejected64=rejected,real288=real,automatic_public=automatic_result)
    save(OUT/"acceptance.json",dict(status="PASS",cases={k:dict(counts=r["counts"],initial_score=r["initial_best_score"],mutation_score=r["mutation_best_score"],mutation_improvement=r["mutation_improvement"],final_score=r["final_best_score"],mutations=len(r["proposals"])) for k,r in reports.items()},
        model_change_changes_decision=synthetic["recommended_key"]!=changed["recommended_key"],data_change_changes_decision=synthetic["recommended_key"]!=changed_data["recommended_key"],
        real_data_scope="Two archived numerical products from shared real station evidence; not independent observations or new efficacy trial",
        scientific_superiority="NOT_ASSESSED",native_model_validation="NOT_ASSESSED"))
    print(json.dumps({k:r["counts"] for k,r in reports.items()},ensure_ascii=False,indent=2))


if __name__=="__main__":main()

