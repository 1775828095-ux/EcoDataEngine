"""Independent experiment client of the actual Streamlit upload service.

References are opened only outside the guarded platform execution boundary.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation, distance_transform_edt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from streamlit.runtime.uploaded_file_manager import UploadedFile,UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs
from ecodata.interface.raster_upload import (RasterRequest,process_raster_upload,FEATURE_NAMES,
    objective_weights,route,upload_registry)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)),encoding="utf-8")


GUARD={"active":False,"prefix":"","denied":0,"calls":0}
def audit(event,args):
    if GUARD["active"] and event=="open" and isinstance(args[0],(str,bytes)):
        path=str(Path(args[0]).resolve()).lower()
        if GUARD["prefix"] in path:
            GUARD["denied"]+=1
            raise PermissionError("Hidden reference access forbidden inside upload/selection/execution")


sys.addaudithook(audit)
@contextmanager
def platform_boundary(root):
    GUARD.update(active=True,prefix=str((root/"sealed_reference").resolve()).lower())
    GUARD["calls"]+=1
    try:yield
    finally:GUARD["active"]=False


def uploaded(path):
    return UploadedFile(UploadedFileRec(path.stem,path.name,"application/x-netcdf",path.read_bytes()),FileURLs())


def evaluate(reference,prediction,coarse,operator,request):
    truth=reference.field.values;p=prediction.field.values;delta=p-truth
    finite=np.isfinite(p);assert finite.all(),"Shared full-field cohort cannot silently shrink"
    std=max(float(coarse.field.std()),1e-8)
    ty,tx=np.gradient(truth);py,px=np.gradient(p)
    gradient=float(np.sqrt(np.mean((ty-py)**2+(tx-px)**2)))
    # Discontinuous edges for boundary assessment, not a routing feature or input.
    edge=np.zeros(truth.shape,bool)
    edge[:-1]|=abs(np.diff(truth,axis=0))>.01
    edge[:,:-1]|=abs(np.diff(truth,axis=1))>.01
    band=binary_dilation(edge,iterations=1)
    boundary=float(np.sqrt(np.mean(delta[band]**2))) if band.any() else 0.
    a=operator@p.ravel()-coarse.field.values.ravel()
    amp=float(abs(p.max()-truth.max()))
    true_peak=np.array(np.unravel_index(truth.argmax(),truth.shape))
    pred_peak=np.array(np.unravel_index(p.argmax(),p.shape))
    position=float(np.linalg.norm(true_peak-pred_peak))
    rmse=float(np.sqrt(np.mean(delta**2)));consistency=float(np.sqrt(np.mean(a*a)))
    components=np.array([rmse/std,gradient/std,boundary/std,amp/std,position/np.sqrt(20000),consistency/std])
    # Common edge threshold, evaluation only. Smooth fields' edge score is descriptive.
    pe=np.zeros(p.shape,bool);pe[:-1]|=abs(np.diff(p,axis=0))>.01;pe[:,:-1]|=abs(np.diff(p,axis=1))>.01
    boundary_distance=float((distance_transform_edt(~pe)[edge].mean()+distance_transform_edt(~edge)[pe].mean())/2) if edge.any() and pe.any() else None
    yy,xx=np.indices(truth.shape);local=(yy-true_peak[0])**2+(xx-true_peak[1])**2<=100
    return dict(RMSE=rmse,MAE=float(abs(delta).mean()),Bias=float(delta.mean()),coverage=float(finite.mean()),
        gradient_RMSE=gradient,boundary_band_RMSE=boundary,boundary_distance_pixels=boundary_distance,
        peak_amplitude_error=amp,peak_position_pixels=position,local_gradient_RMSE=float(np.sqrt(np.mean(((ty-py)**2+(tx-px)**2)[local]))),
        coarse_RMSE=consistency,negative_count=int((p<0).sum()),components=components.tolist(),
        task_score=float(components@objective_weights(request)))


def run_case(root,record,model=None):
    key=Path(record["input"]).stem;folder=root/"runs"/key
    evidence=folder/"metrics.json"
    if evidence.exists():return json.loads(evidence.read_text(encoding="utf-8")),True
    request=RasterRequest(**json.loads(Path(record["request"]).read_text(encoding="utf-8")))
    upload=uploaded(Path(record["input"]))
    with platform_boundary(root):
        result=process_raster_upload(upload,request,model,mode="automatic" if model is not None else "benchmark",execute_all=True)
        if model is not None:
            # Same frozen model/features, restricted candidate set; no extra numerical run.
            from types import SimpleNamespace
            plans=[SimpleNamespace(id=p["id"],steps=p["steps"]) for p in result["trace"]["plans"]]
            single,decision=route(result["features"],request,model,plans,upload_registry(),single_only=True)
            result["trace"]["single_selection"]=single;result["trace"]["single_decision"]=decision
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/"trace_and_quality.json",dict(trace=result["trace"],quality=result["reports"]))
    assert not result["trace"]["failures"],result["trace"]["failures"]
    assert len(result["outputs"])==8,"Incomplete candidates cannot be fairly ranked"
    # Only the independent evaluator opens truth, after routing and execution.
    with xr.open_dataset(record["reference"],engine="h5netcdf") as ds:reference=ds.load()
    metrics=[]
    for i,(identifier,output) in enumerate(result["outputs"].items()):
        output_path=folder/f"candidate_{i}.nc";output.to_netcdf(output_path,engine="h5netcdf")
        with xr.open_dataset(output_path,engine="h5netcdf") as opened:
            actual=opened.load();np.testing.assert_allclose(actual.field,output.field)
            np.testing.assert_allclose(actual.lat,reference.lat,atol=1e-10)
            np.testing.assert_allclose(actual.lon,reference.lon,atol=1e-10)
            assert actual.field.attrs["units"]=="1" and actual.sizes==reference.sizes
        quality=result["reports"][identifier]
        metric=evaluate(reference,actual,result["coarse"],result["operator"],request)
        metric.update(plan=identifier,quality=quality,output=str(output_path),output_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest())
        metrics.append(metric)
    item=dict(field_id=record["field_id"],family=record["family"],split=record["split"],coarse_size=record["coarse_size"],
        subtype=record["metadata"].get("identifiability"),features=result["features"],metrics=metrics,
        selected=result["selected"],single=result["trace"].get("single_selection"),
        weights=objective_weights(request).tolist(),elapsed=result["trace"]["runtime_seconds"],new_execution_count=8,
        reference_opened_after_platform_return=True,readback_count=8)
    save(evidence,item);return item,False


def fit(root,items):
    assert all(r["split"]=="development" for r in items)
    plans=[m["plan"] for m in items[0]["metrics"]]
    x=np.array([[r["features"][f] for f in FEATURE_NAMES] for r in items])
    y=np.array([[v for plan in plans for v in next(m["components"] for m in r["metrics"] if m["plan"]==plan)] for r in items])
    scaler=StandardScaler().fit(x);regressor=KNeighborsRegressor(n_neighbors=5,weights="distance").fit(scaler.transform(x),y)
    model=dict(features=FEATURE_NAMES,plans=plans,scaler=scaler,regressor=regressor,indifference_tolerance=.002)
    joblib.dump(model,root/"router.joblib")
    means={p:float(np.mean([next(m["task_score"] for m in r["metrics"] if m["plan"]==p) for r in items])) for p in plans}
    single=[p for p in plans if not p.endswith("consistency")]
    frozen=dict(best_fixed=min(means,key=means.get),best_single=min(single,key=means.get),development_scores=means,
        training_upload_count=len(items),training_field_count=len({r["field_id"] for r in items}),
        no_family_filename_or_test_labels_in_model=True,router_sha256=hashlib.sha256((root/"router.joblib").read_bytes()).hexdigest())
    save(root/"frozen_selection.json",frozen)
    return model,frozen


def analysis(root,items,frozen):
    rows=[]
    for r in items:
        by={m["plan"]:m for m in r["metrics"]}
        choices={p:p for p in by}
        choices.update(automatic=r["selected"],automatic_single=r["single"],fixed_development=frozen["best_fixed"],
            oracle_evaluation_only=min(by,key=lambda p:by[p]["task_score"]))
        for policy,plan in choices.items():
            if plan is None:continue
            m=by[plan]
            row={k:v for k,v in m.items() if k not in {"components","quality"}}
            row.update(policy=policy,selected_plan=plan,field_id=r["field_id"],family=r["family"],split=r["split"],coarse_size=r["coarse_size"],subtype=r["subtype"],upload_seconds=r["elapsed"])
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(root/"all_metrics.csv",index=False,encoding="utf-8-sig")
    test=frame[frame.split.eq("test")]
    metrics=["RMSE","MAE","Bias","gradient_RMSE","boundary_band_RMSE","peak_amplitude_error","peak_position_pixels","coarse_RMSE","task_score","coverage","negative_count"]
    test.groupby(["family","coarse_size","policy"])[metrics].agg(["mean","std","min","max"]).to_csv(root/"test_summary.csv",encoding="utf-8-sig")
    test.groupby(["family","subtype","policy"])[metrics].mean().to_csv(root/"subtype_summary.csv",encoding="utf-8-sig")
    selected=test[test.policy.eq("automatic")]
    selected.groupby(["family","coarse_size","selected_plan"]).size().to_csv(root/"selection_counts.csv",encoding="utf-8-sig")
    comparisons=[];rng=np.random.default_rng(20261021)
    for family,g in test.groupby("family"):
        fixed=g[g.policy.isin([m["plan"] for m in items[0]["metrics"]])]
        best_fixed=fixed.groupby("policy").task_score.mean().idxmin()
        best_single=fixed[~fixed.policy.str.endswith("consistency")].groupby("policy").task_score.mean().idxmin()
        for baseline in ["fixed_development",best_fixed,best_single,"automatic_single","oracle_evaluation_only"]:
            for metric in ["task_score","RMSE","gradient_RMSE","peak_amplitude_error"]:
                left=g[g.policy.eq("automatic")].set_index(["field_id","coarse_size"])[metric]
                right=g[g.policy.eq(baseline)].set_index(["field_id","coarse_size"])[metric]
                # Field clusters retain both resolutions together (12 independent fields).
                difference=(left-right).groupby("field_id").mean().values
                draws=rng.choice(difference,(2000,len(difference)),replace=True).mean(1)
                comparisons.append(dict(family=family,reference=baseline,metric=metric,n_fields=len(difference),
                    mean_paired_difference=float(difference.mean()),ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),
                    posthoc_reference=baseline in [best_fixed,best_single]))
    pd.DataFrame(comparisons).to_csv(root/"paired_intervals.csv",index=False,encoding="utf-8-sig")
    return frame


def main(root):
    records=json.loads((root/"sealed_reference/manifest.json").read_text(encoding="utf-8"))
    log=[];all_items=[]
    # Smoke is a genuine subset of development, never an extra test peek.
    smoke=[r for r in records if r["split"]=="development" and r["index"]<2]
    for r in smoke:
        item,cached=run_case(root,r);log.append(dict(stage="smoke",input=r["input"],cached=cached,new_executions=0 if cached else 8))
        print("SMOKE",r["family"],r["index"],r["coarse_size"],"PASS",flush=True)
    save(root/"smoke_pass.json",dict(cases=len(smoke),all_candidates_readback=True,full_mask=True,truth_guard_active=True))
    for stage in ["development","validation","test"]:
        model=None
        if stage!="development":model=joblib.load(root/"router.joblib")
        stage_items=[]
        for n,r in enumerate([r for r in records if r["split"]==stage]):
            item,cached=run_case(root,r,model);stage_items.append(item);all_items.append(item)
            log.append(dict(stage=stage,input=r["input"],cached=cached,new_executions=0 if cached else 8))
            if n%6==0:print(stage,n+1,"PASS",flush=True)
        if stage=="development":model,frozen=fit(root,stage_items)
        if stage=="validation":
            # This is explicitly separate truth-assisted selection, never used by router.
            assisted={}
            for weight in {tuple(r["weights"]) for r in stage_items}:
                group=[r for r in stage_items if tuple(r["weights"])==weight]
                means={p:float(np.mean([next(m["task_score"] for m in r["metrics"] if m["plan"]==p) for r in group])) for p in model["plans"]}
                assisted[str(weight)]=dict(selected=min(means,key=means.get),scores=means)
            save(root/"validation_assisted_separate.json",assisted)
            save(root/"pretest_freeze.json",dict(router_sha256=hashlib.sha256((root/"router.joblib").read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest(),
                fixed=frozen,test_read_yet=False,validation_not_used_for_router_adjustment=True))
    frozen=json.loads((root/"frozen_selection.json").read_text(encoding="utf-8"))
    frame=analysis(root,all_items,frozen)
    pd.DataFrame(log).to_csv(root/"execution_log.csv",index=False,encoding="utf-8-sig")
    save(root/"reproducibility.json",dict(total_uploads=len(records),independent_fields=len({r["field_id"] for r in records}),
        new_pipeline_executions=sum(x["new_executions"] for x in log),cache_calls=sum(x["cached"] for x in log),
        guarded_platform_calls=GUARD["calls"],forbidden_reference_attempts=GUARD["denied"],
        output_count=sum(len(r["metrics"]) for r in all_items),common_pixels_per_case=10000,
        model_sha256=hashlib.sha256((root/"router.joblib").read_bytes()).hexdigest(),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path("src/ecodata/interface/raster_upload.py"),Path("src/ecodata/algorithms/raster_reconstruction_task.py")]},
        statistical_unit="field, two scales clustered",test_tuning=False))
    print("COMPLETE",len(frame),"metric rows",flush=True)


if __name__=="__main__":
    warnings.filterwarnings("ignore",category=FutureWarning)
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);main(p.parse_args().output)
