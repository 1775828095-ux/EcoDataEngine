"""Frozen grouped study via the same public service; never read truth in search."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
from time import perf_counter
import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests")]
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.models import portrait_policy as policy
from test_full_screen_spatiotemporal import publish,prepare_case

OUT=ROOT/"output/complementary_20261024/study_v2"
DEV=[0,360,720,1080];VAL=[1200,1340];TEST=[1460,1820,2180,2540]
DEV_SEEDS=[11,29,47];TEST_SEEDS=[11,29,47,71,101]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding="utf-8")


def run(req,inputs,truth,folder,backend,seed,phase,*,exact=False):
    request=deepcopy(req);request["composition"]["search"].update(backend=backend,exact_check=exact)
    if backend=="none":request["composition"]["search"]["generations"]=0
    evaluation=api.DatasetReference(truth,req["variable"])
    evaluation.protocol.update(study_partition=phase,study="complementary-controlled-v1")
    if (folder/"summary.json").exists():return json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    return api.submit_search(inputs,request,evaluation,folder,mode="Profile",seed=seed,
        budget=dict(objective_limit=56 if exact else 12,proposal_limit=48,wall_seconds=180))


def data(start):
    inputs,req,truth=prepare_case(ROOT,start=start,synthetic=False)
    folder=OUT/"cases"/str(start)
    if not (folder/"request.json").exists():
        for item in inputs:(folder/"inputs").mkdir(parents=True,exist_ok=True);(folder/"inputs"/item.filename).write_bytes(item.content)
        truth.to_dataset(name=req["variable"]).to_netcdf(folder/"hidden_reference.nc",engine="h5netcdf")
        config=dict(inputs=[dict(path=str(folder/"inputs"/i.filename),role=i.role) for i in inputs],request=req,mode="Profile",budget=dict(objective_limit=12,proposal_limit=48,wall_seconds=180))
        save(folder/"request.json",config)
        save(folder/"lineage.json",dict(parent=str(ROOT/"data/ncep_xarray/air_temperature.nc"),start_index=start,
            input_sha256={i.filename:hashlib.sha256(i.content).hexdigest() for i in inputs},
            hidden_times=[str(t) for t in truth.time.values],fine_anchor_indices=[0,8],coarse_operator="area overlap applied to source grid samples; controlled support assumption",native_resolution="2.5 degree; only relative fine/coarse",forecast=False))
    return inputs,req,truth


def observed(report):
    best=report.get("best")
    return dict(score=best["matching"]["score_interval"][0] if best else None,
        rmse=report.get("independent_evaluation",{}).get("RMSE") if best else None,
        evaluations=report["counts"]["evaluated"],passes=report["counts"]["usable_observed"],
        seconds=report["total_service_seconds"],key=report.get("recommended_key"),
        kernel_calls=sum(t.get("kernel_calls",0)+int("roles" in t) for r in report["rows"] for t in r.get("trace",[])))


def oracle(start,inputs,req,truth,phase):
    folder=OUT/"oracles"/str(start)
    report=run(req,inputs,truth,folder,"none",11,phase,exact=True)
    errors=[]
    for row in report["rows"]:
        if row["status"]!="PASS":continue
        with xr.open_dataset(row["product"],engine="h5netcdf") as ds:
            error=ds[req["variable"]]-truth
            rmse=float(np.sqrt(np.mean(error.values**2)))
        errors.append(dict(key=row["key"],rmse=rmse,score=row["matching"]["score_interval"][0],plan=row["plan"]))
    save(folder/"independent_matrix.json",dict(rows=errors,search_consumes_this_matrix=False))
    return max(r["score"] for r in errors),errors


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ar.ROOT=OUT/"attributes";policy.ROOT=OUT/"models"
    if not (ar.ROOT/"complementary-state-v1.json").exists():publish()
    results=[];matrices={}
    # Development and validation are completed before testing or test oracle.
    for phase,starts in [("development",DEV),("validation",VAL)]:
        for start in starts:
            inputs,req,truth=data(start)
            for seed in DEV_SEEDS:
                # Rotate order to reduce systematic cold-start timing effects.
                backends=["basic","alns","pymoo","random"]
                offset=DEV_SEEDS.index(seed);backends=backends[offset:]+backends[:offset]
                for backend in backends:
                    folder=OUT/phase/str(start)/backend/str(seed)
                    report=run(req,inputs,truth,folder,backend,seed,phase)
                    results.append(dict(phase=phase,case=start,backend=backend,seed=seed,folder=str(folder),**observed(report)))
            top,matrix=oracle(start,inputs,req,truth,phase);matrices[start]=matrix
            for row in results:
                if row["case"]==start:row["regret"]=top-row["score"] if row["score"] is not None else 1e6
            save(OUT/"progress.json",dict(completed_case=start,phase=phase,runs=len(results)))
            print(json.dumps(dict(phase=phase,case=start,completed_runs=len(results))),flush=True)
    summary={}
    preference={"alns":0,"pymoo":1,"basic":2,"random":3}
    for backend in preference:
        rows=[r for r in results if r["phase"]=="development" and r["backend"]==backend]
        summary[backend]=dict(mean_regret=float(np.mean([r["regret"] for r in rows])),success_rate=float(np.mean([r["score"] is not None for r in rows])),
            mean_rmse=float(np.mean([r["rmse"] for r in rows if r["rmse"] is not None])),median_seconds=float(np.median([r["seconds"] for r in rows])))
    # Quantise only the declared numerical tie tolerance, never a test metric.
    ordered=sorted(preference,key=lambda b:(round(summary[b]["mean_regret"],9),-summary[b]["success_rate"],preference[b]))
    basic_val=np.mean([r["regret"] for r in results if r["phase"]=="validation" and r["backend"]=="basic"])
    selected=next(b for b in ordered if np.mean([r["regret"] for r in results if r["phase"]=="validation" and r["backend"]==b])<=basic_val+1e-9)
    fixed_keys=set.intersection(*(set(r["key"] for r in matrices[s]) for s in DEV))
    fixed=min(fixed_keys,key=lambda key:np.mean([next(r["rmse"] for r in matrices[s] if r["key"]==key) for s in DEV]))
    selection=dict(selected=selected,development=summary,validation_basic_regret=float(basic_val),
        tie_order=preference,fixed_from_development_key=fixed,test_seen=False,
        no_universal_best_claim=True,selection_metric="same portrait score regret, not hidden test RMSE")
    save(OUT/"frozen_selection.json",selection)
    for start in TEST:
        inputs,req,truth=data(start)
        for seed in TEST_SEEDS:
            for backend in list(dict.fromkeys([selected,"basic","random","none"])):
                folder=OUT/"test"/str(start)/backend/str(seed)
                report=run(req,inputs,truth,folder,backend,seed,"test")
                results.append(dict(phase="test",case=start,backend=backend,seed=seed,folder=str(folder),**observed(report)))
        top,matrix=oracle(start,inputs,req,truth,"test");matrices[start]=matrix
        for row in results:
            if row["case"]==start:row["regret"]=top-row["score"] if row["score"] is not None else 1e6
        # Single-source controls also execute through the public entry.
        for source in ["fine_slow","coarse_fast"]:
            control=deepcopy(req);control["composition"]["source_stages"]["fine_slow"][0]["options"]=[dict(algorithm="temporal.semantic_resample",parameters=dict(target_time_frequency="6h",temporal_interpolation="linear"))]
            control["composition"]["source_stages"]["coarse_fast"][0]["options"]=[dict(algorithm="raster_task.spatial.bilinear")]
            control["composition"]["fusion_options"]=[dict(algorithm="prepare.select_source",source=source,evidence="Source-only numerical control, other source not used in output")]
            report=run(control,inputs,truth,OUT/"controls"/str(start)/source,"none",11,"test")
            results.append(dict(phase="control",case=start,backend=source,seed=11,**observed(report)))
        print(json.dumps(dict(phase="test",case=start,completed_runs=len(results))),flush=True)
    save(OUT/"results.json",dict(selection=selection,runs=results,
        fixed_test=[dict(case=s,rmse=next(r["rmse"] for r in matrices[s] if r["key"]==fixed),oracle_rmse=min(r["rmse"] for r in matrices[s])) for s in TEST],
        scope="Controlled downsampling of true reanalysis; not independent sensors or WRF validation"))
    print(json.dumps(selection,ensure_ascii=False),flush=True)


if __name__=="__main__":main()

