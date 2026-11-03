"""Read-only independent analysis of saved predictions; never changes selection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd


def main(root):
    rows=[];timings=[];quality=[];assisted=json.loads((root/"validation_assisted_separate.json").read_text(encoding="utf-8"))
    records=json.loads((root/"sealed_reference/manifest.json").read_text(encoding="utf-8"))
    pd.DataFrame([dict(input=r["input"],request=r["request"],sha256=r["input_sha256"],split=r["split"],
        source_rows=r["coarse_size"],source_columns=r["coarse_size"],target_rows=100,target_columns=100,
        unit="1",crs="EPSG:4326",extent="110..111 E, 30..31 N",time="static synthetic field, no observation date",
        valid_input_fraction=1.,reference_is_independent_observation=False,
        aggregation="spherical-area cell overlap",reference_exposed_to_platform=False) for r in records]).to_csv(root/"dataset_manifest.csv",index=False,encoding="utf-8-sig")
    for file in sorted((root/"runs").glob("*/metrics.json")):
        r=json.loads(file.read_text(encoding="utf-8"))
        trace=json.loads(file.with_name("trace_and_quality.json").read_text(encoding="utf-8"))
        for m in r["metrics"]:
            q=m["quality"]
            timings.append(dict(field_id=r["field_id"],family=r["family"],split=r["split"],coarse_size=r["coarse_size"],plan=m["plan"],
                runtime_seconds=q["runtime_seconds"],status=q["status"],accuracy_available=q["quality"]["accuracy_assessed"],
                nonnegative_projection_count=sum(s["report"].get("negative_projection_count",0) for s in q["steps"])))
            quality.append(dict(input=file.parent.name,plan=m["plan"],status=q["status"],coverage=m["coverage"],
                negative_count=m["negative_count"],coarse_RMSE=m["coarse_RMSE"],output_sha256=m["output_sha256"]))
        if r["split"]=="test":
            winner=assisted[str(tuple(r["weights"]))]["selected"]
            m=next(m for m in r["metrics"] if m["plan"]==winner)
            row={k:v for k,v in m.items() if k not in {"quality","components"}}
            row.update(policy="validation_assisted_not_input_only",selected_plan=winner,field_id=r["field_id"],family=r["family"],
                split=r["split"],coarse_size=r["coarse_size"],subtype=r["subtype"],upload_seconds=r["elapsed"])
            rows.append(row)
    frame=pd.concat([pd.read_csv(root/"all_metrics.csv"),pd.DataFrame(rows)],ignore_index=True)
    frame.to_csv(root/"comparison_with_validation_assisted.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(quality).to_csv(root/"quality_readback_index.csv",index=False,encoding="utf-8-sig")
    times=pd.DataFrame(timings);times.to_csv(root/"pipeline_times.csv",index=False,encoding="utf-8-sig")
    times[times.split.eq("test")].groupby("plan").runtime_seconds.agg(["count","mean","median","std","sum"]).to_csv(root/"pipeline_time_summary.csv",encoding="utf-8-sig")
    test=frame[frame.split.eq("test")];base=test[test.policy.str.startswith("raster_task.")]
    all_fixed=base.groupby("policy").task_score.mean().idxmin()
    paired=[];rng=np.random.default_rng(20261022)
    targets=["RMSE","MAE","Bias","gradient_RMSE","boundary_band_RMSE","peak_amplitude_error","peak_position_pixels","coarse_RMSE","task_score"]
    for family in ["A","B","C","ALL"]:
        g=test if family=="ALL" else test[test.family.eq(family)]
        best=base if family=="ALL" else base[base.family.eq(family)]
        strongest=best.groupby("policy").task_score.mean().idxmin()
        single=best[~best.policy.str.endswith("consistency")].groupby("policy").task_score.mean().idxmin()
        for reference in dict.fromkeys(["fixed_development",all_fixed,strongest,single,"automatic_single","validation_assisted_not_input_only","oracle_evaluation_only"]):
            for metric in targets:
                a=g[g.policy.eq("automatic")].set_index(["field_id","coarse_size"])[metric]
                b=g[g.policy.eq(reference)].set_index(["field_id","coarse_size"])[metric]
                diff=(a-b).groupby("field_id").mean()
                if family=="ALL":
                    # Stratified field bootstrap: preserve 12 fields per family.
                    means=[]
                    for label in ["A","B","C"]:
                        ids=g[g.family.eq(label)].field_id.unique();values=diff.loc[ids].values
                        means.append(rng.choice(values,(2000,len(values)),replace=True).mean(1))
                    draw=np.mean(means,axis=0)
                else:draw=rng.choice(diff.values,(2000,len(diff)),replace=True).mean(1)
                paired.append(dict(family=family,reference=reference,metric=metric,n_fields=len(diff),mean_difference=float(diff.mean()),
                    ci95_low=float(np.quantile(draw,.025)),ci95_high=float(np.quantile(draw,.975)),
                    test_truth_selected_reference=reference.startswith("raster_task."),interval="pointwise percentile, no multiplicity adjustment"))
    pd.DataFrame(paired).to_csv(root/"full_paired_intervals.csv",index=False,encoding="utf-8-sig")
    source_files=[Path("src/ecodata/interface/raster_upload.py"),Path("src/ecodata/interface/raster_reconstruction_page.py"),Path("tools/run_three_task_routing.py"),Path("tools/generate_three_task_rasters.py")]
    counts=dict(status_counts=pd.DataFrame(quality).status.value_counts().to_dict(),full_pixel_cohort=10000,
        output_readbacks=len(quality),all_finite_and_nonnegative=all(q["coverage"]==1 and q["negative_count"]==0 for q in quality),
        strongest_global_fixed_posthoc=all_fixed,reference_disclosure="Oracle minimizes frozen task score, not RMSE: its RMSE need not be minimal",
        plateau_warning="Task A maximum occupies an area, argmax position is not meaningful and must not be interpreted as localization accuracy",
        disk_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        execution_seconds_total=float(times.runtime_seconds.sum()),source_files=[str(f) for f in source_files])
    (root/"final_audit.json").write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(counts,ensure_ascii=True,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);main(p.parse_args().output)
