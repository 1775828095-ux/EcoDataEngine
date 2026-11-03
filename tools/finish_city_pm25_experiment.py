"""Post-freeze test masks, streamed summaries and reproducibility checks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import city_pm25_experiment as exp
from city_pm25_experiment import OUT,DATA_OUT,ROOT,save,sha,np,pd,METHODS
from run_city_temporal_search import restore
from ecodata.algorithms.temporal_reconstruction_dag import temporal_setup,reconstruct
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
import pyarrow.parquet as pq


def pilot_test(config):
    folder=OUT/"pilot_final_test";folder.mkdir(exist_ok=True)
    if (folder/"receipt.json").exists():return
    selection_path=OUT/"selected_graphs_frozen.json"
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    registry,_=temporal_setup();pipeline=UnifiedAlgorithmPipeline(registry)
    parts=[];metrics=[];mask=[];failures=[]
    for case in exp.cases_for_pilot(config):
        idx=case["positions"]
        idx=idx[case["y"].index[idx]>pd.Timestamp(config["validation_end"])]
        if not len(idx):continue
        info={k:case[k] for k in ["file_id","entity_id","task","seed","case_id"]}
        truth=case["y"].iloc[idx].to_numpy();times=case["y"].index[idx]
        prediction={};cost={}
        for method in METHODS:
            clock=perf_counter();prediction[method]=reconstruct(case["masked"],method).iloc[idx].to_numpy();cost[method]=perf_counter()-clock
        for key,rows in selection["graphs"].items():
            method="selected_"+key;clock=perf_counter()
            try:prediction[method],_=exp.execute_graph(restore(rows),case,idx,pipeline)
            except Exception as exc:
                prediction[method]=np.full(len(idx),np.nan);failures.append(dict(**info,key=key,error=str(exc)))
            cost[method]=perf_counter()-clock
        common=np.logical_and.reduce([np.isfinite(prediction[m]) for m in METHODS])
        mask.append(pd.DataFrame(dict(**info,time=times,split="test",source_row=case["source_rows"].reindex(times).to_numpy())))
        for method,values in prediction.items():
            parts.append(pd.DataFrame(dict(**info,time=times,split="test",truth=truth,prediction=values,method=method,
                peak=truth>=case["peak_threshold"],common_fixed_baselines=common)))
            for cohort,use in [("all_targets",np.ones(len(idx),bool)),("common_fixed_baselines",common)]:
                if use.any():metrics.append(dict(**info,method=method,split="test",cohort=cohort,
                    **exp.metrics(truth[use],values[use],case["peak_threshold"]),seconds=cost[method],
                    unit="native_unconfirmed",quality="WARN_UNIT_CLOCK_UNCONFIRMED" if np.isfinite(values[use]).all() else "BLOCK_INCOMPLETE_COVERAGE"))
        pipeline._dag_executors.clear()
    pd.concat(parts,ignore_index=True).to_parquet(folder/"predictions.parquet",index=False)
    pd.concat(mask,ignore_index=True).to_csv(folder/"masks.csv",index=False)
    pd.DataFrame(metrics).to_csv(folder/"metrics.csv",index=False)
    save(folder/"failures.json",failures)
    save(folder/"receipt.json",dict(selection_sha256=sha(selection_path),test_read_after_freeze=True,
        no_reselection=True,rows=sum(len(p) for p in parts)))
    print("PILOT TEST DONE",sum(len(p) for p in parts),flush=True)


SUMS=["expected","predicted","sse","absolute","error","truth_sum","truth_square","peak_n","peak_sse"]


def summarized_predictions(folder):
    totals=defaultdict(lambda:np.zeros(len(SUMS),float))
    for batch in pq.ParquetFile(folder/"predictions.parquet").iter_batches(batch_size=200000,
            columns=["task","method","seed","split","truth","prediction","peak","common_fixed_baselines"]):
        data=batch.to_pandas();good=np.isfinite(data.prediction.to_numpy())
        delta=np.where(good,data.prediction-data.truth,0.)
        data=data.assign(expected=1,predicted=good.astype(int),sse=delta**2,absolute=abs(delta),error=delta,
            truth_sum=np.where(good,data.truth,0.),truth_square=np.where(good,data.truth**2,0.),
            peak_n=(good&data.peak).astype(int),peak_sse=np.where(good&data.peak,delta**2,0.))
        for cohort,values in [("all_targets",data),("common_fixed_baselines",data.loc[data.common_fixed_baselines])]:
            for index,row in values.groupby(["task","method","seed","split"])[SUMS].sum().iterrows():
                totals[index+(cohort,)]+=row.to_numpy()
    records=[]
    for key,values in totals.items():
        row=dict(zip(["task","method","seed","split","cohort"],key),**dict(zip(SUMS,values)))
        records.append(row)
    return pd.DataFrame(records)


def finish_scores(frame):
    n=frame.predicted
    frame["RMSE"]=np.sqrt(frame.sse/n.where(n>0))
    frame["MAE"]=frame.absolute/n.where(n>0)
    frame["Bias"]=frame.error/n.where(n>0)
    total=frame.truth_square-frame.truth_sum**2/n.where(n>0)
    frame["R2"]=1-frame.sse/total.where(total>0)
    frame["coverage"]=n/frame.expected
    frame["failure_rate"]=1-frame.coverage
    frame["peak_RMSE"]=np.sqrt(frame.peak_sse/frame.peak_n.where(frame.peak_n>0))
    frame["unit"]="native_unconfirmed_do_not_pool_across_files"
    frame["strict_comparison_eligible"]=frame.coverage.eq(1.)
    return frame


def combine_masks(metric_paths):
    columns=["file_id","entity_id","task","seed","case_id","time","split","source_row","peak_threshold","stage"]
    first=True
    with (OUT/"mask_manifest.csv").open("w",encoding="utf-8-sig",newline="") as output:
        for path in metric_paths:
            for chunk in pd.read_csv(path.with_name("masks.csv"),chunksize=200000):
                chunk["stage"]="pilot_validation" if "pilot" in path.parts else "pilot_test" if "pilot_final_test" in path.parts else "batch"
                # pilot_final_test has no threshold column; keep an explicit NA,
                # never shift its stage into the threshold position.
                chunk.reindex(columns=columns).to_csv(output,index=False,header=first);first=False


def summarize(config):
    folders=list((OUT/"batch").glob("*/metrics.csv"))
    assert len(folders)==len(config["source_files"]),"Batch not finished; do not publish partial regions"
    metric_paths=list((OUT/"pilot").glob("*/metrics.csv"))+folders+[OUT/"pilot_final_test/metrics.csv"]
    names={a["file_id"]:a["filename"] for a in json.loads((DATA_OUT/"dataset_audit.json").read_text(encoding="utf-8"))}
    selection=json.loads((OUT/"selected_graphs_frozen.json").read_text(encoding="utf-8"))
    labels={m:m for m in METHODS};modes={m:("past_only" if m=="ffill" else "retrospective") for m in METHODS}
    graph_map=[]
    for key,rows in selection["graphs"].items():
        operations=[r for r in rows if r["operation"].startswith("baseline.")]
        partial=[r["operation"].removeprefix("baseline.temporal_") for r in operations if r["parameters"].get("mode")=="partial"]
        fallback=[r["operation"].removeprefix("baseline.temporal_") for r in operations if r["parameters"].get("mode")=="fallback"]
        label=partial[0]+" -> missing-only "+fallback[0] if partial else operations[0]["operation"].removeprefix("baseline.temporal_")
        method="selected_"+key;labels[method]=label;modes[method]="past_only" if label=="ffill" else "retrospective"
        graph_map.append(dict(key=key,method=method,label=label,information_mode=modes[method]))
    pd.DataFrame(graph_map).to_csv(OUT/"selected_graph_map.csv",index=False)
    baseline=pd.concat([pd.read_csv(p) for p in metric_paths],ignore_index=True)
    baseline["filename"]=baseline.file_id.map(names)
    baseline["method_label"]=baseline.method.map(labels);baseline["information_mode"]=baseline.method.map(modes)
    baseline.to_csv(OUT/"baseline_results.csv",index=False)
    all_stats=[]
    for path in metric_paths:
        stats=summarized_predictions(path.parent)
        stats["stage"]="pilot_validation" if "pilot" in path.parts else "pilot_test" if "pilot_final_test" in path.parts else "batch"
        stats["file_id"]=config["pilot_file"] if stats.stage.iloc[0].startswith("pilot") else path.parent.name
        all_stats.append(stats)
    values=pd.concat(all_stats,ignore_index=True)
    values["filename"]=values.file_id.map(names)
    values["method_label"]=values.method.map(labels);values["information_mode"]=values.method.map(modes)
    finish_scores(values.copy()).to_csv(OUT/"summary_by_task.csv",index=False)
    group=["file_id","stage","task","method","split","cohort"]
    overall=values.groupby(group,as_index=False)[SUMS].sum()
    overall["filename"]=overall.file_id.map(names)
    overall["method_label"]=overall.method.map(labels);overall["information_mode"]=overall.method.map(modes)
    timing=baseline.groupby(["file_id","task","method","split","cohort"],as_index=False).seconds.sum()
    overall=overall.merge(timing,on=["file_id","task","method","split","cohort"],how="left",validate="many_to_one")
    finish_scores(overall).to_csv(OUT/"summary_by_file.csv",index=False)
    combine_masks(metric_paths)
    runs=json.loads((OUT/"search_summary.json").read_text(encoding="utf-8"))
    summaries=[]
    for r in runs:
        rows=r["records"];good=[x for x in rows if x.get("score",1e12)<1e6]
        summaries.append(dict(mode=r["mode"],seed=r["seed"],task=r["task"],evaluations=len(rows),
            new_executions=r["counts"].get("new_executions",0),cache_hits=r["counts"].get("cache_hits",0),
            proposals=r["counts"].get("proposals",0),rejected=r["counts"].get("proposal_rejections",0),
            posterior_rejections=r["counts"].get("posterior_rejections",0),unique_valid_graphs=r["unique_valid_graphs"],
            first_valid_evaluation=next((i for i,x in enumerate(rows,1) if x in good),None),
            RMSE=r["best"]["RMSE"] if r["best"] else None,key=r["best"]["key"] if r["best"] else None,
            seconds=r["seconds"],optimizer_error=bool(r["optimizer_error"])))
    pd.DataFrame(summaries).to_csv(OUT/"search_summary.csv",index=False)
    prefix=[]
    for task in config["search_tasks"]:
        for seed in config["search_seeds"]:
            group=[r for r in runs if r["task"]==task and r["seed"]==seed]
            budget=min(len(r["records"]) for r in group)
            for r in group:
                valid=[row for row in r["records"][:budget] if row.get("score",1e12)<1e6]
                prefix.append(dict(task=task,seed=seed,mode=r["mode"],common_callback_budget=budget,
                    best_RMSE=min((row["RMSE"] for row in valid),default=None)))
    pd.DataFrame(prefix).to_csv(OUT/"common_search_budget.csv",index=False)
    signatures=defaultdict(list)
    enum=json.loads((OUT/"enumeration_summary.json").read_text(encoding="utf-8"))
    enum_rows=[]
    for row in enum:
        path=OUT/"enumeration"/row["task"]/row["key"]/"predictions.parquet"
        frame=pd.read_parquet(path)
        digest=hashlib.sha256(frame.prediction.round(10).to_numpy().tobytes()).hexdigest()
        signatures[row["task"]+":"+digest].append(row["key"])
        enum_rows.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
    save(OUT/"observed_behavior_groups.json",dict(signatures))
    pd.DataFrame(enum_rows).to_csv(OUT/"enumeration_results.csv",index=False)
    print("SUMMARIES DONE",len(values),flush=True)


def audit(config):
    audits=json.loads((DATA_OUT/"dataset_audit.json").read_text(encoding="utf-8"))
    counts=[];entities=defaultdict(list);samples=[];file_quality=[];entity_labels=defaultdict(list)
    import openpyxl
    for item in audits:
        assert sha(item["path"])==item["sha256"],"Original workbook changed"
        frame=pd.read_parquet(DATA_OUT/"prepared"/item["file_id"]/"observations.parquet")
        file_quality.append(dict(file_id=item["file_id"],longitude_valid_rate=float(frame.longitude.between(-180,180).mean()),
            latitude_valid_rate=float(frame.latitude.between(-90,90).mean()),
            off_hour_records=int((frame.time.notna()&frame.time.ne(frame.time.dt.floor("h"))).sum()),
            source_quality_meaning="AQI grade text, NOT instrument QA",source_license="USER_PROVIDED_NO_PUBLIC_LICENSE_CONFIRMED"))
        for entity,g in frame.groupby("entity_id"):
            entities[str(entity)].append(item["file_id"])
            entity_labels[(str(g.province.iloc[0]),str(g.entity_name.iloc[0]))].append(item["file_id"])
            span=int((g.time.max()-g.time.min())/pd.Timedelta(hours=1))+1
            counts.append(dict(file_id=item["file_id"],entity_id=entity,name=g.entity_name.iloc[0],rows=len(g),
                expected_hourly_slots=span,valid=int(g.reason.eq("VALID").sum()),
                numeric_time_eligible=bool(g.reason.eq("VALID").sum()>=48),
                natural_axis_missing=span-g.time.nunique(),invalid_values=int((~g.reason.eq("VALID")).sum()),
                suspicious_over_1000=int(g.value.gt(1000).sum()),suspicious_rule="diagnostic only; units unknown, NOT deleted",
                coordinate_pairs=len(g[["longitude","latitude"]].drop_duplicates())))
        raw=pd.read_parquet(DATA_OUT/"prepared"/item["file_id"]/"raw_selected.parquet")
        book=openpyxl.load_workbook(item["path"],read_only=True,data_only=True);sheet=book.worksheets[0]
        rows=list(sheet.iter_rows(min_row=1,max_row=33,values_only=True));headers=rows[0]
        for source_row,values in enumerate(rows[1:],2):
            prepared=raw.loc[raw.source_row.eq(source_row)].iloc[0]
            for original,mapped in item["field_mapping"].items():
                value=values[headers.index(original)];actual=prepared[mapped]
                if original=="pubtime":assert pd.Timestamp(value)==actual
                elif original in {"longitude","latitude","pm2_5"}:
                    assert (pd.isna(value) and pd.isna(actual)) or np.isclose(float(value),actual)
                else:assert (value is None and pd.isna(actual)) or str(value)==str(actual)
        maximum=raw.loc[raw.value.idxmax()]
        if maximum.value>10000:
            original=next(sheet.iter_rows(min_row=int(maximum.source_row),max_row=int(maximum.source_row),values_only=True))
            assert float(original[headers.index("pm2_5")])==maximum.value
            save(OUT/"extreme_original_value_check.json",dict(file_id=item["file_id"],source_row=int(maximum.source_row),
                original_pm2_5=original[headers.index("pm2_5")],parsed_value=float(maximum.value),
                reader="independent openpyxl",decision="Source contains the extreme; not a streaming parser error; validity unconfirmed, not deleted"))
        samples.append(dict(file_id=item["file_id"],independent_openpyxl_rows_checked=len(rows)-1,
            first_rows_column_types={h:sorted({type(r[i]).__name__ for r in rows[1:]}) for i,h in enumerate(headers)}))
        book.close()
    pd.DataFrame(counts).to_csv(OUT/"entity_quality.csv",index=False)
    overlap={entity:files for entity,files in entities.items() if len(files)>1}
    save(OUT/"cross_file_entities.json",dict(unique_entities=len(entities),overlaps=overlap,
        province_name_overlaps={str(k):v for k,v in entity_labels.items() if len(set(v))>1},
        exclusion="Any overlapping development entity cannot support external-location generalization claims"))
    save(OUT/"independent_reader_checks.json",samples)
    manifest=pd.read_csv(DATA_OUT/"dataset_manifest.csv")
    dtype_map={r["file_id"]:json.dumps(r["first_rows_column_types"],ensure_ascii=False) for r in samples}
    manifest["independent_sample_column_types"]=manifest.file_id.map(dtype_map)
    manifest["column_type_scope"]="First32 source rows independently inspected; full selected columns parsed and audited"
    manifest=manifest.merge(pd.DataFrame(file_quality),on="file_id",validate="one_to_one")
    eligible=pd.DataFrame(counts).groupby("file_id").numeric_time_eligible.sum()
    manifest["numeric_time_eligible_entities"]=manifest.file_id.map(eligible)
    manifest.to_csv(OUT/"dataset_manifest.csv",index=False,encoding="utf-8-sig")
    tests=ET.parse(OUT/"targeted_tests.xml").getroot()
    test_counts={k:sum(int(t.attrib.get(k,0)) for t in tests.iter("testsuite")) for k in ["tests","failures","errors","skipped"]}
    assert not test_counts["failures"] and not test_counts["errors"]
    assert len(list((OUT/"batch").glob("*/receipt.json")))==len(audits)
    for path in (OUT/"batch").glob("*/receipt.json"):
        receipt=json.loads(path.read_text(encoding="utf-8"))
        assert receipt["config_sha256"]==sha(OUT/"experiment_config.json")
        assert receipt["predictions_sha256"]==sha(path.with_name("predictions.parquet"))
    test_receipt=json.loads((OUT/"pilot_final_test/receipt.json").read_text(encoding="utf-8"))
    assert test_receipt["selection_sha256"]==sha(OUT/"selected_graphs_frozen.json")
    first=exp.cases_for_pilot(config)[0];again=exp.cases_for_pilot(config)[0]
    np.testing.assert_array_equal(first["positions"],again["positions"])
    assert first["masked"].equals(again["masked"])
    first=dict(first,masked=first["masked"].loc[:config["validation_end"]])
    idx=first["positions"];idx=idx[first["y"].index[idx]<=pd.Timestamp(config["validation_end"])]
    selected=json.loads((OUT/"selected_graphs_frozen.json").read_text(encoding="utf-8"))["graphs"]
    graph=restore(next(iter(selected.values())))
    repeated=[]
    for _ in range(2):
        registry,_=temporal_setup()
        predicted,_=exp.execute_graph(graph,first,idx,UnifiedAlgorithmPipeline(registry))
        repeated.append(predicted)
    np.testing.assert_array_equal(repeated[0],repeated[1])
    save(OUT/"pilot_reproducibility_probe.json",dict(real_user_data=True,cold_executions=2,
        mask_repeat_equal=True,prediction_repeat_equal=True,validation_inputs_end=str(first["masked"].index.max()),
        output_sha256=hashlib.sha256(repeated[0].tobytes()).hexdigest(),test_targets_used=False))
    for relative,expected in config["code_sha256"].items():assert sha(ROOT/relative)==expected
    outputs=[p for p in OUT.glob("*.csv")]+[OUT/"selected_graphs_frozen.json",OUT/"experiment_config.json"]
    save(OUT/"reproducibility.json",dict(status="REAL_NATIVE_VALUE_TEMPORAL_EXPERIMENTS_COMPLETE_WITH_METADATA_LIMITS",
        original_files_unchanged=True,downloaded_files=0,raw_files_modified=0,tests=test_counts,
        input_sha256={r["file_id"]:r["sha256"] for r in audits},
        source_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"tools/run_city_temporal_search.py",
            ROOT/"tools/city_pm25_experiment.py",ROOT/"tools/prepare_city_hourly_excel.py",ROOT/"src/ecodata/algorithms/temporal_reconstruction_dag.py"]},
        output_sha256={p.name:sha(p) for p in outputs},frozen_pool_sha256=sha(ROOT/"output/capability_final_acceptance/frozen_generic_pool.json"),
        source_unit_verified=False,source_clock_verified=False,source_coordinate_datum_verified=False,
        spatial_experiment="NOT_APPLICABLE_UNCONFIRMED_DATUM_AND_CITY_SUPPORT",
        scientific_status="Not a confirmed physical-concentration or spatial-downscaling experiment; no search-superiority assumption"))
    print("AUDIT DONE",len(audits),len(entities),test_counts,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["pilot-test","summarize","audit","repair-mask-manifest"]);a=p.parse_args()
    config=exp.freeze_config()
    if a.stage=="repair-mask-manifest":
        save(OUT/"mask_manifest_schema_failure.json",dict(original_sha256=sha(OUT/"mask_manifest.csv"),
            reason="Final-test masks lacked optional peak_threshold; concatenation shifted stage column",
            affected="Derived combined CSV only; source masks, predictions, selection and metrics unchanged",
            evidence="code_snapshot/finish_before_mask_schema_fix.py and original per-stage masks.csv remain available"))
        paths=list((OUT/"pilot").glob("*/metrics.csv"))+list((OUT/"batch").glob("*/metrics.csv"))+[OUT/"pilot_final_test/metrics.csv"]
        combine_masks(paths)
        print("MASK SCHEMA REPAIRED",sha(OUT/"mask_manifest.csv"),flush=True)
    else:{"pilot-test":pilot_test,"summarize":summarize,"audit":audit}[a.stage](config)
