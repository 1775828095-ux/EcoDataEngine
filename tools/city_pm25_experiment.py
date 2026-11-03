"""Fixed masks, observed-only scoring and streamed cross-file temporal baselines."""
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_name]="1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_city_hourly_excel import OUT as DATA_OUT,sha,save
OUT=Path(os.environ.get("ECODATA_CITY_RUN",str(DATA_OUT)))
OUT.mkdir(parents=True,exist_ok=True)
from ecodata.algorithms.temporal_reconstruction_dag import reconstruct,METHODS,temporal_setup
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.dag import Artifact


SEEDS=[11,22,33,44,55]


def series_from(frame,start=None,end=None,entities=None):
    start=pd.Timestamp(start) if start is not None else frame.time.min().floor("h")
    end=pd.Timestamp(end) if end is not None else frame.time.max().floor("h")
    index=pd.date_range(start,end,freq="h")
    for entity,g in frame.groupby("entity_id",sort=True):
        if entities is not None and entity not in entities:continue
        g=g.loc[g.reason.eq("VALID") & g.time.between(start,end)].copy()
        g=g.loc[g.time.dt.floor("h").eq(g.time)].drop_duplicates("time",keep=False)
        y=g.set_index("time").value.reindex(index)
        if y.notna().sum()>=48:yield str(entity),y,g.set_index("time").source_row


def masks(y):
    valid=np.flatnonzero(y.notna().to_numpy())
    for seed in SEEDS:
        idx=np.sort(np.random.default_rng(seed).choice(valid,size=int(len(valid)*.2),replace=False))
        yield "random20",seed,idx
    for phase in [0,1]:yield "alternate_phase"+str(phase),phase,valid[valid%2==phase]
    quarters=y.index.to_period("Q")
    available=y.notna().to_numpy()
    for length in [6,12]:
        chosen=[]
        for quarter in quarters.unique():
            positions=np.flatnonzero(quarters==quarter)
            candidates=[i for i in positions if i>0 and i+length<len(y) and quarters[i+length-1]==quarter
                        and available[i-1:i+length+1].all()]
            if candidates:
                # Only timestamps/validity enter selection, never concentrations.
                start=candidates[np.random.default_rng(11).integers(len(candidates))]
                chosen.extend(range(start,start+length))
        if chosen:yield "block"+str(length),11,np.asarray(chosen,int)


def make_cases(frame,config,file_id,pilot=False):
    options=dict(start=config["pilot_start"],end=config["pilot_end"],entities=config["pilot_entities"]) if pilot else {}
    for entity,y,source_rows in series_from(frame,**options):
        for task,seed,positions in masks(y):
            hidden=y.copy();hidden.iloc[positions]=np.nan
            assert hidden.iloc[positions].isna().all() and y.iloc[positions].notna().all()
            threshold=float(hidden.quantile(.9))
            yield dict(file_id=file_id,entity_id=entity,y=y,masked=hidden,positions=positions,
                source_rows=source_rows,task=task,seed=seed,peak_threshold=threshold,
                case_id=f"{file_id}_{entity}_{task}_{seed}")


def metrics(truth,prediction,threshold):
    truth,prediction=np.asarray(truth,float),np.asarray(prediction,float)
    finite=np.isfinite(prediction)&np.isfinite(truth)
    n=len(truth); error=prediction[finite]-truth[finite]
    peak=finite&(truth>=threshold)
    total=float(np.sum((truth[finite]-np.mean(truth[finite]))**2)) if finite.any() else 0.
    return dict(expected=n,predicted=int(finite.sum()),coverage=float(finite.mean()) if n else 0.,
        failure_rate=float((~finite).mean()) if n else 1.,RMSE=float(np.sqrt(np.mean(error**2))) if error.size else None,
        MAE=float(np.mean(abs(error))) if error.size else None,Bias=float(np.mean(error)) if error.size else None,
        R2=float(1-np.sum(error**2)/total) if total>0 else None,
        peak_n=int(peak.sum()),peak_RMSE=float(np.sqrt(np.mean((prediction[peak]-truth[peak])**2))) if peak.any() else None)


def freeze_config():
    if (OUT/"experiment_config.json").exists():
        path=OUT/"experiment_config.json"
        config=json.loads(path.read_text(encoding="utf-8"))
        current={p:sha(ROOT/p) for p in config["code_sha256"]}
        if current!=config["code_sha256"]:
            assert not list((OUT/"pilot").glob("*/metrics.csv")) and not list((OUT/"search").glob("*/*/summary.json")),"Executed protocol changed; use a new output version"
            save(OUT/"pre_execution_config.json",config)
            config["code_sha256"]=current
            config["pre_execution_amendment"]="Targeted tests fixed normalization metadata and query cache context before real experiments"
            save(path,config)
        return config
    audits=[json.loads(p.read_text(encoding="utf-8")) for p in (DATA_OUT/"prepared").glob("*/audit.json")]
    preferred=[a for a in audits if "安徽、福建、江西" in a["filename"]]
    assert preferred,"Preferred file not yet extracted; do not silently select another"
    audit=preferred[0]; frame=pd.read_parquet(DATA_OUT/"prepared"/audit["file_id"]/"observations.parquet")
    best=None
    starts=pd.date_range(frame.time.min().normalize(),frame.time.max()-pd.Timedelta(days=30),freq="MS")
    for start in starts:
        end=start+pd.Timedelta(hours=719)
        counts=frame.loc[frame.reason.eq("VALID")&frame.time.between(start,end)].groupby("entity_id").size().nlargest(10)
        if len(counts)<2:continue
        key=(int(counts.sum()),-start.value)
        if best is None or key>best[0]:best=(key,start,end,sorted(counts.index.astype(str)))
    assert best
    _,start,end,entities=best
    config=dict(version="city-hourly-pm25-v1",pilot_file=audit["file_id"],pilot_filename=audit["filename"],
        pilot_start=str(start),pilot_end=str(end),pilot_entities=entities,
        validation_end=str(start+pd.Timedelta(hours=479)),random_seeds=SEEDS,search_seeds=[11,22,33],
        search_tasks=["random20","alternate_phase0","alternate_phase1"],search_budget=8,proposal_budget=50,
        search_wall_seconds=45,unit="UNCONFIRMED_NATIVE_VALUE",timezone="NAIVE_PRESERVED",
        hourly_mean_definition="UNCONFIRMED: label restoration only, not mean disaggregation",
        spatial_status="NOT_APPLICABLE: unverified datum and city-aggregate support",
        mask_rule="20% finite values; actual timestamp-index parity; one fully observed bracketed block per quarter",
        peak_rule="retained-input 90th percentile, frozen independently for each mask; no hidden target values",
        temporal_mode="retrospective except ffill past-only; not forecasting",
        selection_rule="pilot first20days validation; last10days test masks sealed until graphs frozen",
        validation_context="All validation inputs and fitted statistics strictly stop at validation_end; no last10day observations",
        transfer="freeze pilot graphs, apply unchanged to other files; overlapping city IDs excluded from external claims",
        candidate_scope="4 minimal task-local baseline IDs, existing pandas/SciPy/UCI operations; frozen34 pool untouched",
        source_files={a["file_id"]:dict(path=a["path"],sha256=a["sha256"]) for a in audits},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"src/ecodata/algorithms/temporal_reconstruction_dag.py"]})
    save(OUT/"experiment_config.json",config);return config


def cases_for_pilot(config):
    frame=pd.read_parquet(DATA_OUT/"prepared"/config["pilot_file"]/"observations.parquet")
    return list(make_cases(frame,config,config["pilot_file"],pilot=True))


def execute_graph(graph,case,positions,pipeline):
    target_times=case["y"].index[positions]
    scale=float(case["masked"].dropna().abs().median()) or 1.
    metadata=dict(entity_id=case["entity_id"],file_id=case["file_id"],unit_status="UNCONFIRMED_NATIVE_VALUE",clock_status="NAIVE_PRESERVED",
        native_scale=scale,scale_fit="retained unmasked values only",normalized_unit="1",
        target_index_sha256=hashlib.sha256(str(list(target_times)).encode()).hexdigest())
    result=pipeline.run_dag(graph,task_id="city_pm25_temporal",version="hour-label-v1",
        external={"series":Artifact("masked_series",case["masked"]/scale,metadata),"target_times":target_times})
    return result["outputs"]["field"].value["prediction"]*scale,result["trace"]


def baseline_file(file_id,config,pilot=False,validation_only=False):
    directory=OUT/("pilot" if pilot else "batch")/file_id
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"metrics.csv").exists():return
    frame=pd.read_parquet(DATA_OUT/"prepared"/file_id/"observations.parquet")
    writer=None; records=[]; failures=[]; first=True
    mask_path=directory/"masks.csv"
    registry,factory=temporal_setup();pipeline=UnifiedAlgorithmPipeline(registry)
    selected=json.loads((OUT/"selected_graphs_frozen.json").read_text(encoding="utf-8"))["graphs"] if (OUT/"selected_graphs_frozen.json").exists() and not validation_only else {}
    from run_city_temporal_search import restore
    for case in make_cases(frame,config,file_id,pilot):
        if validation_only:
            case=dict(case,masked=case["masked"].loc[:config["validation_end"]])
            case["peak_threshold"]=float(case["masked"].quantile(.9))
        positions=case["positions"]
        if validation_only:positions=positions[case["y"].index[positions]<=pd.Timestamp(config["validation_end"])]
        if not len(positions):continue
        truth=case["y"].iloc[positions].to_numpy();times=case["y"].index[positions]
        split=np.where(times<=pd.Timestamp(config["validation_end"]),"validation","test") if pilot else np.full(len(times),"external" if file_id!=config["pilot_file"] else "development_extended")
        info={k:case[k] for k in ["file_id","entity_id","task","seed","case_id"]}
        mask=pd.DataFrame(dict(**info,time=times,split=split,source_row=case["source_rows"].reindex(times).to_numpy(),peak_threshold=case["peak_threshold"]))
        mask.to_csv(mask_path,index=False,mode="w" if first else "a",header=first);first=False
        predictions={}; costs={}; trace_records=[]
        for method in METHODS:
            clock=perf_counter();predictions[method]=reconstruct(case["masked"],method).iloc[positions].to_numpy();costs[method]=perf_counter()-clock
        for key,rows in selected.items():
            clock=perf_counter()
            try:predictions["selected_"+key],trace=execute_graph(restore(rows),case,positions,pipeline);trace_records.append(dict(key=key,trace=trace))
            except Exception as exc:
                predictions["selected_"+key]=np.full(len(positions),np.nan)
                failures.append(dict(**info,key=key,error=str(exc),trace=getattr(exc,"trace",[])))
            costs["selected_"+key]=perf_counter()-clock
        common=np.logical_and.reduce([np.isfinite(predictions[m]) for m in METHODS])
        for method,values in predictions.items():
            pred=pd.DataFrame(dict(**info,time=times,split=split,truth=truth,prediction=values,method=method,
                peak=truth>=case["peak_threshold"],common_fixed_baselines=common))
            table=pa.Table.from_pandas(pred,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(directory/"predictions.parquet",table.schema,compression="zstd")
            writer.write_table(table)
            for role in pd.unique(split):
                selected_role=split==role
                for cohort,use in [("all_targets",selected_role),("common_fixed_baselines",selected_role&common)]:
                    if not use.any():continue
                    record=dict(**info,method=method,split=role,cohort=cohort,
                        **metrics(truth[use],values[use],case["peak_threshold"]),seconds=costs[method],
                        unit="native_unconfirmed",quality="WARN_UNIT_CLOCK_UNCONFIRMED" if np.isfinite(values[use]).all() else "BLOCK_INCOMPLETE_COVERAGE")
                    records.append(record)
        with (directory/"selected_traces.jsonl").open("a",encoding="utf-8") as stream:
            for trace in trace_records:stream.write(json.dumps(dict(**info,**trace),default=str,ensure_ascii=False)+"\n")
        # Keep fitted-data caches bounded; no cross-case cache hits disguised as execution.
        pipeline._dag_executors.clear()
    if writer:writer.close()
    pd.DataFrame(records).to_csv(directory/"metrics.csv",index=False)
    save(directory/"failures.json",failures)
    save(directory/"receipt.json",dict(file_id=file_id,validation_only=validation_only,pilot=pilot,
        config_sha256=sha(OUT/"experiment_config.json"),predictions_sha256=sha(directory/"predictions.parquet")))
    print("BASELINE DONE",file_id,len(records),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","pilot","batch"]);a=p.parse_args()
    config=freeze_config()
    if a.stage=="pilot":baseline_file(config["pilot_file"],config,pilot=True,validation_only=True)
    if a.stage=="batch":
        assert (OUT/"selected_graphs_frozen.json").exists()
        for file_id in sorted(config["source_files"]):baseline_file(file_id,config)
