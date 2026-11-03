"""Bounded development-only resource audit; no reference data enters platform."""
import argparse
import json
from pathlib import Path
import sys
from threading import Event,Thread
from time import perf_counter,process_time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import joblib
import pandas as pd
import psutil
from ecodata.interface.raster_upload import process_raster_upload,RasterRequest
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from run_three_task_routing import uploaded,platform_boundary,save


def main(root,deploy_only=False):
    records=json.loads((root/"sealed_reference/manifest.json").read_text(encoding="utf-8"))
    model=joblib.load(root/"router.joblib");process=psutil.Process();rows=[]
    original=UnifiedAlgorithmPipeline.run
    def measured(self,context,plan):
        clock=perf_counter();cpu=process_time();initial=process.memory_info().rss;peaks=[initial];done=Event()
        def poll():
            while not done.wait(.005):peaks.append(process.memory_info().rss)
        thread=Thread(target=poll,daemon=True);thread.start()
        try:result=original(self,context,plan)
        finally:done.set();thread.join()
        rows.append(dict(plan=plan.id,wall_seconds=perf_counter()-clock,cpu_seconds=process_time()-cpu,
            rss_baseline_bytes=initial,rss_sampled_peak_bytes=max(peaks),rss_delta_bytes=max(peaks)-initial,
            status=result.report["status"],sampling_seconds=.005))
        return result
    UnifiedAlgorithmPipeline.run=measured
    for record in [r for r in records if r["split"]=="development" and r["index"]==0]:
        request=RasterRequest(**json.loads(Path(record["request"]).read_text(encoding="utf-8")))
        start=len(rows)
        with platform_boundary(root):result=process_raster_upload(uploaded(Path(record["input"])),request,model,execute_all=not deploy_only)
        for row in rows[start:]:row.update(field_id=record["field_id"],coarse_size=record["coarse_size"],purpose="automatic deployment only" if deploy_only else "development resource measurement, no evaluation truth",
            full_upload_seconds=result["trace"]["runtime_seconds"],selected=result["selected"])
    UnifiedAlgorithmPipeline.run=original
    name="deployment_profile" if deploy_only else "resource_profile"
    pd.DataFrame(rows).to_csv(root/(name+".csv"),index=False,encoding="utf-8-sig")
    save(root/(name+"_summary.json"),dict(executions=len(rows),sampling_limit="5ms sampled process RSS, not exact per-kernel peak; allocator/cache and first-use imports included",
        reference_used=False,experimental_model_or_parameters_changed=False))
    print("PROFILE",len(rows),"actual pipelines")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--deploy-only",action="store_true")
    args=p.parse_args();main(args.output,args.deploy_only)
