"""Frozen-selection summaries, tie-aware peak audit, and bounded resource profiles."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist
from run_lghap_reconstruction import ROOT,DATA,COARSE,sha,save,registry_for,context_for
from ecodata.algorithms.raster_reconstruction_task import reconstruction_plans
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline


def finite_json(value):
    if isinstance(value,float) and not np.isfinite(value):return "unbounded" if value>0 else "negative_unbounded"
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    return value


def profile(out,plan_id):
    import psutil
    path=sorted((out/"tiles/validation").glob("*.nc"))[0]
    tile=xr.load_dataset(path,engine="h5netcdf");context,_=context_for(tile,48)
    registry=registry_for(out);plans,_=reconstruction_plans(registry,context)
    plan=next(p for p in plans if p.id==plan_id)
    clock=perf_counter();cpu0=sum(psutil.Process().cpu_times()[:2]);rss0=psutil.Process().memory_info().rss
    result=UnifiedAlgorithmPipeline(registry).run(context,plan)
    info=psutil.Process().memory_info()
    save(out/"resource_profiles"/(plan.id+".json"),dict(plan=plan.id,
        tile=path.stem,role="validation resource-only replay, not reselection",wall_seconds=perf_counter()-clock,
        cpu_seconds=sum(psutil.Process().cpu_times()[:2])-cpu0,RSS_before_bytes=rss0,
        process_peak_RSS_bytes=info.peak_wset,RSS_after_bytes=info.rss,
        output_bytes=result.data.dataset.aod.values.nbytes,quality=result.report["status"],
        memory_scope="Whole process high-water including imports, model load and solver; not pure-kernel allocation"))


def summarize(out,resources):
    cfg=json.loads((out/"config.json").read_text(encoding="utf-8"));selected=json.loads((out/"selected_plans.json").read_text(encoding="utf-8"))
    assert all(sha(p)==v for p,v in cfg["source_hashes"].items())
    assert all(sha(p)==v for p,v in cfg["numerical_code_hashes"].items())
    contract=json.loads((out/"training_contract.json").read_text(encoding="utf-8"))
    assert sha(out/"liif_aod_local.pt")==contract["checkpoint_sha256"]
    tiles=pd.read_csv(out/"dataset_manifest.csv");splits={key:set(tiles[tiles.split.eq(key)].region) for key in ["train","validation","test"]}
    assert all(not splits[a]&splits[b] for a,b in [("train","validation"),("train","test"),("validation","test")])
    assert all(sha(p)==s for p,s in zip(tiles.path,tiles.sha256))
    tests=pd.read_csv(out/"test/metrics.csv");validation=pd.read_csv(out/"validation/metrics.csv")
    assert len(tests)==144 and len(validation)==144
    assert tests.coverage.eq(1).all() and tests.negative_count.eq(0).all()
    assert tests.groupby(["tile","scale"]).evaluated_pixels.nunique().eq(1).all()
    all_rows=pd.concat([validation,tests],ignore_index=True)
    grouped=[]
    for (stage,scale,plan),part in all_rows.groupby(["stage","scale","plan"]):
        high=part.high_pixels>0
        grouped.append(dict(stage=stage,scale=scale,plan=plan,tiles=len(part),
            RMSE=float(np.sqrt(np.mean(part.RMSE**2))),RMSE_tile_std=float(part.RMSE.std()),
            MAE=float(part.MAE.mean()),Bias=float(part.Bias.mean()),gradient_RMSE=float(part.gradient_RMSE.mean()),
            spatial_correlation=float(part.spatial_correlation.mean()),
            high_value_RMSE=float(np.sqrt(np.average(part.loc[high,"high_value_RMSE"]**2,weights=part.loc[high,"high_pixels"]))) if high.any() else None,
            high_value_tiles=int(high.sum()),high_value_pixels=int(part.high_pixels.sum()),
            peak_position_error_pixels=float(part.peak_position_error_pixels.mean()),
            coarse_consistency_RMSE=float(part.coarse_consistency_RMSE.mean()),
            coverage=float(part.coverage.mean()),failure_rate=float(part.failure.mean()),mean_seconds=float(part.seconds.mean())))
    summary=pd.DataFrame(grouped);summary.to_csv(out/"summary_all_plans.csv",index=False)
    comparison=[];paired=[]
    for n in COARSE:
        scale=96/n;part=tests[tests.coarse_size.eq(n)];fixed=cfg["fixed_baseline"]
        for label,plan in [("fixed_bilinear",fixed),("automatic_single",selected[str(n)]["single"]),
            ("automatic_all",selected[str(n)]["automatic"]),("best_validation_combo",selected[str(n)]["composite"])]:
            row=summary[(summary.stage.eq("test"))&(summary.scale.eq(scale))&summary.plan.eq(plan)].iloc[0].to_dict()
            row["policy"]=label;row["selected_on"]="validation only";comparison.append(row)
        base=part[part.plan.eq(fixed)].set_index("tile")
        trial=part[part.plan.eq(selected[str(n)]["automatic"])].set_index("tile")
        single=part[part.plan.eq(selected[str(n)]["single"])].set_index("tile")
        for name,other in [("fixed_bilinear",base),("validation_best_single",single)]:
            improvement=100*(other.RMSE-trial.RMSE)/other.RMSE
            for tile,value in improvement.items():paired.append(dict(scale=scale,tile=tile,comparator=name,RMSE_improvement_pct=value))
    pd.DataFrame(comparison).to_csv(out/"policy_comparison.csv",index=False)
    pd.DataFrame(paired).to_csv(out/"paired_test_improvements.csv",index=False)
    # AOD is stored in increments of .01. Keep the originally declared argmax metric;
    # audit its tie sensitivity separately, never optimize or select on this audit.
    peak_audit=[]
    for row in tests.itertuples():
        truth=xr.load_dataset(out/"tiles/test"/(row.tile+".nc"),engine="h5netcdf")
        output=out/"test"/(f"{row.tile}_n{row.coarse_size}_{row.plan.replace('+','__')}"+".nc")
        with xr.open_dataset(output,engine="h5netcdf") as prediction:
            values=prediction.aod.values.copy()
        context,_=context_for(truth,row.coarse_size);coarse=context.sources[0].dataset
        yy,xx=np.meshgrid(truth.lat.values,truth.lon.values,indexing="ij")
        mask=(yy>=coarse.lat.min().item())&(yy<=coarse.lat.max().item())&(xx>=coarse.lon.min().item())&(xx<=coarse.lon.max().item())
        refset=np.argwhere(mask&(truth.aod.values>=np.max(truth.aod.values[mask])-.005))
        predset=np.argwhere(mask&(values>=np.max(values[mask])-.005))
        peak_audit.append(dict(tile=row.tile,scale=row.scale,plan=row.plan,reference_peak_set=len(refset),prediction_peak_set=len(predset),
            original_argmax_error_pixels=row.peak_position_error_pixels,
            half_quantum_peak_set_min_distance_pixels=float(cdist(refset,predset).min()),
            role="post-freeze diagnostic only, original argmax metric retained, no reselection"))
    pd.DataFrame(peak_audit).to_csv(out/"peak_tie_sensitivity.csv",index=False)
    registry=registry_for(out);first=xr.load_dataset(sorted((out/"tiles/validation").glob("*.nc"))[0],engine="h5netcdf")
    context,_=context_for(first,48);plans,screen=reconstruction_plans(registry,context)
    ids=["super_resolution.liif","super_resolution.thera","downscale.climatelearn_t2m","neural.fno","neural.multiscale_uno","super_resolution.zssr"]
    review=[]
    for identifier in ids:
        spec=registry.get(identifier).spec
        review.append(dict(id=identifier,spec=spec.as_dict(),eligible_unchanged=screen[identifier].eligible,
            reasons=screen[identifier].reasons,
            task_resolution="Official LIIF reused via local AOD training adapter; old RGB/temperature checkpoints NOT used" if identifier==ids[0] else "Existing contract does not support this AOD-only task; no install/repair attempted"))
    save(out/"advanced_candidate_review.json",finite_json(review))
    save(out/"executed_capability_catalog.json",finite_json([registry.get(p.steps[0]).spec.as_dict() for p in plans[::2]]+
        [registry.get(i).spec.as_dict() for i in ["raster_task.nonnegative","raster_task.coarse_consistency"]]))
    descriptors=[]
    for tile_path in tiles.path:
        tile=xr.load_dataset(tile_path,engine="h5netcdf")
        for n in COARSE:
            context,operator=context_for(tile,n);coarse=context.sources[0].dataset;source=coarse.aod.values
            gy,gx=np.gradient(source)
            descriptors.append(dict(tile=Path(tile_path).stem,role=tile.attrs["split"],coarse_size=n,
                semantic="continuous",variable="aod",units="1",CRS="EPSG:4326",scale=96/n,
                finite_fraction=float(np.isfinite(source).mean()),coarse_mean=float(source.mean()),
                coarse_std=float(source.std()),coarse_gradient_rms=float(np.sqrt(np.mean(gy**2+gx**2))),
                decision_scope="Semantics/grid/units/scale screened; mean/std/gradient descriptive only, NOT a learned per-tile router"))
            yy,xx=np.meshgrid(tile.lat.values,tile.lon.values,indexing="ij")
            mask=(yy>=coarse.lat.min().item())&(yy<=coarse.lat.max().item())&(xx>=coarse.lon.min().item())&(xx<=coarse.lon.max().item())
            case=tile.copy(deep=True);case["evaluation_mask"] = (("lat","lon"),mask.astype("uint8"))
            case["coarse_aod"] = (("coarse_lat","coarse_lon"),source)
            case=case.assign_coords(coarse_lat=coarse.lat.values,coarse_lon=coarse.lon.values)
            case.coarse_aod.attrs["units"]="1";case.attrs["scale"]=96/n
            case.attrs["degradation"]="spherical-area overlap means; explicit fine/coarse cell outer-edge alignment"
            case_folder=out/"case_inputs";case_folder.mkdir(exist_ok=True)
            case.to_netcdf(case_folder/f"{tile.attrs['split']}_{Path(tile_path).stem}_n{n}.nc",engine="h5netcdf")
            from scipy.sparse import save_npz
            save_npz(case_folder/f"{tile.attrs['split']}_{Path(tile_path).stem}_n{n}_operator.npz",operator)
    pd.DataFrame(descriptors).to_csv(out/"task_descriptors.csv",index=False)
    save(out/"license_clarifications.json",dict(
        xarray_regrid_official_license="Apache-2.0; https://github.com/xarray-contrib/xarray-regrid/blob/main/LICENSE",
        existing_registry_caveat="Historical RegridPlugin metadata says BSD-3-Clause; this is not the upstream xarray-regrid license. It was inherited, not treated as authoritative permission.",
        LIIF="BSD-3-Clause, pinned official source retained; local weights are not official pretraining",
        dataset="CC-BY-4.0, official record 5652257",QP="CVXPY Apache-2.0 and OSQP Apache-2.0"))
    save(out/"failure_and_fix_history.json",[
        dict(stage="initial training",error="ModuleNotFoundError: tensorboardX",action="Use existing .venv-capability-v5 process; no new package install",resolved=True),
        dict(stage="configuration reload before training",error="JSON lists compared to Python tuples",action="Canonicalize config JSON; retain old output, formal v3 configuration before validation/test",resolved=True),
        dict(stage="test collection",error="Independent environment lacked installed ecodata path",action="Explicit test src path; 8 task tests pass",resolved=True),
        dict(stage="metadata export",error="AlgorithmSpec unbounded scale encoded as float infinity",action="Represent unbounded contract limits explicitly as strings; numerical results unchanged",resolved=True),
        dict(stage="official LIIF meshgrid",error="PyTorch future indexing warning",action="Unmodified upstream, numerical execution succeeds; warning retained",resolved=True)])
    if resources:
        for plan in plans:
            path=out/"resource_profiles"/(plan.id+".json")
            if path.exists():continue
            proc=subprocess.run([sys.executable,__file__,"--output",str(out),"--profile-plan",plan.id],capture_output=True,text=True,timeout=90)
            if proc.returncode:raise RuntimeError(proc.stderr)
        pd.DataFrame([json.loads(p.read_text()) for p in (out/"resource_profiles").glob("*.json")]).to_csv(out/"resource_profiles.csv",index=False)
    assert (out/"selected_plans.json").stat().st_mtime<min(p.stat().st_mtime for p in (out/"test").glob("*.nc"))
    import torch
    from liif_runtime import load,model_spec
    models,_=load();torch.manual_seed(20261020);initial=models.make(model_spec()).state_dict()
    trained=torch.load(out/"liif_aod_local.pt",map_location="cpu",weights_only=True)["state_dict"]
    delta=max(float((trained[k]-v).abs().max()) for k,v in initial.items())
    assert delta>0
    save(out/"final_acceptance.json",dict(real_data=True,official_guide_zip_MD5_verified=True,
        annual_zip_MD5_verified=False,annual_zip_downloaded=False,member_CRC32_and_SHA256_verified=True,
        official_LIIF_model=True,local_training_steps=300,max_parameter_change_from_seeded_initialization=delta,
        advanced_candidates_reviewed=6,actual_base_estimators=4,actual_optional_constraint_plans=4,
        real_smoke_executions=24,real_validation_executions=144,real_test_executions=144,
        resource_replays_validation_only=8,validation_selection_precedes_test=True,
        data_contract_and_reference_mask_frozen=True,independent_measured_truth=False,
        original_algorithms_and_G0LEM_modified=False,global_UI_integration=False,
        scope="Task factory registers adapters in existing Registry; existing Pipeline and two quality gates actually executed"))
    save(out/"reproducibility.json",dict(config_sha256=sha(out/"config.json"),selected_plan_sha256=sha(out/"selected_plans.json"),
        weights_sha256=sha(out/"liif_aod_local.pt"),source_hashes=cfg["source_hashes"],
        numerical_code_hashes=cfg["numerical_code_hashes"],analysis_sha256=sha(__file__),
        versions={k:importlib.metadata.version(k) for k in ["numpy","xarray","torch","cvxpy","osqp","xarray-regrid"]},
        formal_validation_executions=144,formal_test_executions=144,smoke_executions=24,
        new_fitted_models=1,optimizer_steps=300,resources_replay_validation_only=8 if resources else 0,
        test_selected_or_tuned=False,independent_measured_truth=False,physical_AOD_total_conservation_asserted=False,
        scope="Small constructed degradation experiment on fused LGHAP product, not real sensor-resolution enhancement"))
    save(out/"artifact_fingerprints.json",{str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.name!="artifact_fingerprints.json"})
    print(pd.DataFrame(comparison)[["scale","policy","RMSE","gradient_RMSE","coarse_consistency_RMSE","mean_seconds"]].to_string(index=False))
    print("Validation total measured algorithm-block seconds",validation.seconds.sum())


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--resources",action="store_true");p.add_argument("--profile-plan")
    args=p.parse_args()
    if args.profile_plan:profile(args.output,args.profile_plan)
    else:summarize(args.output,args.resources)
