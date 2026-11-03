"""Read-only analysis of frozen selections; never choose by external test error."""
import json
import sys
import csv
from collections import Counter
import xml.etree.ElementTree as ET
from importlib.metadata import version,PackageNotFoundError
import pandas as pd
from city_pm25_experiment import OUT,ROOT,save,sha


if __name__=="__main__":
    config=json.loads((OUT/"experiment_config.json").read_text(encoding="utf-8"))
    selection_path=OUT/"selected_graphs_frozen.json"
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    score=pd.read_csv(OUT/"summary_by_file.csv")
    rows=[]
    for choice in selection["selections"]:
        subset=score.loc[score.stage.eq("batch")&score.task.eq(choice["task"])&
            score.method.eq("selected_"+choice["key"])]
        for _,result in subset.iterrows():
            rows.append(dict(mode=choice["mode"],search_seed=choice["seed"],task=choice["task"],key=choice["key"],
                file_id=result.file_id,filename=result.filename,cohort=result.cohort,RMSE=result.RMSE,MAE=result.MAE,
                Bias=result.Bias,R2=result.R2,coverage=result.coverage,expected=result.expected,
                external_file=result.file_id!=config["pilot_file"],
                strict_numeric_comparison_eligible=result.strict_comparison_eligible,
                selection_sha256=sha(selection_path),refitted_on_external_targets=False))
    pd.DataFrame(rows).to_csv(OUT/"frozen_transfer_by_search.csv",index=False)
    # A common denominator is verified even when one graph returned no prediction.
    keys=["file_id","stage","task","split","cohort"]
    consistency=score.groupby(keys).expected.nunique()
    assert consistency.eq(1).all()
    task=pd.read_csv(OUT/"summary_by_task.csv")
    assert task.groupby(keys+["seed"]).expected.nunique().eq(1).all()
    runs=pd.read_csv(OUT/"search_summary.csv")
    assert runs.evaluations.le(8).all() and runs.proposals.le(50).all()
    assert len(runs)==len(config["search_tasks"])*len(config["search_seeds"])*4
    stage_counts=Counter()
    with (OUT/"mask_manifest.csv").open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.reader(stream);header=next(reader)
        for row in reader:
            assert len(row)==len(header),"Combined mask row schema shifted"
            assert row[-1] in {"pilot_validation","pilot_test","batch"}
            stage_counts[row[-1]]+=1
    expected=int(score.loc[score.method.eq("nearest")&score.cohort.eq("all_targets"),"expected"].sum())
    assert sum(stage_counts.values())==expected
    save(OUT/"mask_schema_validation.json",dict(total_rows=expected,stages=dict(stage_counts),columns=header,all_rows_valid=True))
    basic=pd.read_csv(OUT/"baseline_results.csv")
    basic=basic.loc[basic.method.eq("nearest")&basic.cohort.eq("all_targets")&basic.split.isin(["external","development_extended"])]
    present={(str(r.file_id),str(r.entity_id),r.task,int(r.seed)):int(r.expected) for r in basic.itertuples()}
    entities=pd.read_csv(OUT/"entity_quality.csv",dtype={"entity_id":str})
    cases=[]
    for r in entities.itertuples():
        for task,seed in [("random20",s) for s in config["random_seeds"]]+[("alternate_phase0",0),("alternate_phase1",1),("block6",11),("block12",11)]:
            n=present.get((str(r.file_id),str(r.entity_id),task,seed),0)
            cases.append(dict(file_id=r.file_id,entity_id=r.entity_id,task=task,seed=seed,targets=n,
                status="EXECUTED" if n else "NOT_APPLICABLE",reason="" if n else
                    "Less than48 valid hours" if not r.numeric_time_eligible else "No fully observed bracketed interval or hidden observed target"))
    pd.DataFrame(cases).to_csv(OUT/"mask_applicability.csv",index=False)
    save(OUT/"comparison_integrity.json",dict(equal_target_denominators=True,
        search_runs=len(runs),all_search_budgets_respected=True,
        three_search_seeds_not_independent_environmental_samples=True,
        external_refitting=False,selection_sha256=sha(selection_path),
        unit_caveat="Per-file native units; no national pooled physical RMSE",
        scope="Numerical restoration and exploratory frozen-policy transfer, not new blind confirmatory evidence"))
    repro_path=OUT/"reproducibility.json"
    repro=json.loads(repro_path.read_text(encoding="utf-8"))
    packages={}
    for name in ["numpy","pandas","scipy","openpyxl","pyarrow","lxml","thegolem"]:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]="See existing .venv-golem source path"
    repro.update(runtime=dict(python=sys.version,packages=packages),
        comparison_integrity_sha256=sha(OUT/"comparison_integrity.json"),
        transfer_results_sha256=sha(OUT/"frozen_transfer_by_search.csv"),
        postprocessing_script_sha256=sha(__file__))
    supplementary=ET.parse(OUT/"postprocess_tests.xml").getroot()
    extra={k:sum(int(t.attrib.get(k,0)) for t in supplementary.iter("testsuite")) for k in ["tests","failures","errors","skipped"]}
    assert extra["failures"]==extra["errors"]==0
    repro["base_targeted_tests"]=repro.get("base_targeted_tests",repro["tests"])
    repro["postprocess_tests"]=extra
    repro["tests"]={k:repro["base_targeted_tests"][k]+extra[k] for k in extra}
    repro["output_sha256"]["mask_manifest.csv"]=sha(OUT/"mask_manifest.csv")
    repro["source_code_sha256"]={k.replace("\\", "/"):v for k,v in repro["source_code_sha256"].items()}
    repro["source_code_sha256"]["tools/finish_city_pm25_experiment.py"]=sha(ROOT/"tools/finish_city_pm25_experiment.py")
    save(repro_path,repro)
    print("PASS same evaluation denominators, bounded search and frozen transfer",len(rows))
