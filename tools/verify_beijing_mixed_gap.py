"""Final independent contracts, budget, file and provenance checks."""
import hashlib,json,sys,subprocess,ast,platform
from pathlib import Path
import xml.etree.ElementTree as ET
import importlib.metadata as md
import joblib,numpy as np,pandas as pd
from prepare_beijing_mixed_gap import OUT,ROOT,DATA,save,sha,load
from run_beijing_mixed_gap import Evaluator,fixed_graphs,key_for,restore,conditions


def verify():
    config=json.loads((OUT/"protocol_frozen.json").read_text(encoding="utf-8"));states=joblib.load(OUT/"training_states.joblib")
    assert sha(OUT/"training_states.joblib")==config["training_state_sha256"]
    for file,fingerprint in config["source_hashes"].items():assert sha(ROOT/file)==fingerprint
    freeze=json.loads((OUT/"selected_graphs_frozen.json").read_text(encoding="utf-8"));assert sha(OUT/"protocol_frozen.json")==freeze["protocol_sha256"]
    final=json.loads((OUT/"test_results.json").read_text(encoding="utf-8"));assert sha(OUT/"selected_graphs_frozen.json")==final["freeze_sha256"]
    enumeration=json.loads((OUT/"enumeration_summary.json").read_text(encoding="utf-8"));assert enumeration["complete"] and enumeration["count"]==949
    canonical={r["key"] for r in enumeration["results"]};searches=[json.loads(p.read_text(encoding="utf-8")) for p in (OUT/"search").glob("*/summary.json")]
    assert len(searches)==50
    counters={};factory_requests=[];total_files=0;callback_count=0
    for r in searches:
        assert r["scored"]<=50 and r["counts"].get("proposals",0)<=500 and len(r["rows"])==r["scored"]
        assert len({x["key"] for x in r["rows"]})==len(r["rows"])
        assert not r["optimizer_error"]
        assert len(r["events"])+r["counts"].get("deferred_callbacks",0)==r["counts"]["callbacks"]
        callback_count+=r["counts"]["callbacks"]
        trace=json.loads((OUT/"search"/f"{r['mode']}_{r['seed']}"/"factory_trace.json").read_text(encoding="utf-8"))
        factory_requests.append(dict(mode=r["mode"],seed=r["seed"],outer_mutation_or_random_proposals=r["counts"].get("proposals",0),
            all_typed_factory_constructions=sum(x.get("event")=="random_graph_request" for x in trace),
            note="Factory constructions include initialization and nested subtree generation; not an extra numerical evaluation"))
        for x in r["rows"]:
            if x["unique_execution"]:
                assert x["key"] in canonical
                file=OUT/"search"/f"{r['mode']}_{r['seed']}"/"executions"/x["key"]/"prediction.npz"
                assert sha(file)==x["prediction_file_sha256"];total_files+=1
    pd.DataFrame(factory_requests).to_csv(OUT/"factory_generation_accounting.csv",index=False)
    for r in final["results"]:
        file=OUT/"frozen_test"/r["key"]/"prediction.npz";assert sha(file)==r["prediction_file_sha256"];total_files+=1
    all_prediction_paths=set()
    for result_file in OUT.glob("**/result.json"):
        value=json.loads(result_file.read_text(encoding="utf-8"));file=result_file.with_name("prediction.npz")
        assert sha(file)==value["prediction_file_sha256"];all_prediction_paths.add(str(file))
    for r in enumeration["results"]:
        file=Path(r.get("reused_from",str(OUT/"enumeration"/r["key"])))/"prediction.npz"
        assert sha(file)==r["prediction_file_sha256"];all_prediction_paths.add(str(file))
    # Candidate contracts are the actual executable task ports, not a claim that
    # the old generic single-Series entry supports all these execution modes.
    evaluator=Evaluator(2016,states,OUT/"reproducibility_probe")
    adapters=evaluator.pipeline.registry.task_adapters("beijing_mixed_gap_v1")
    save(OUT/"candidate_contracts.json",dict(pool_version="beijing task-local five-method v1",grammar_revision=2,
        scope="registered named-port DAG adapters; generic TemporalBaseline.run is not the train-state interface",
        source_units="ug m-3 verified against UCI metadata; no unit or clock conversion",sources_are_independent_stations=True,
        no_cross_station_or_mask_donors=True,adapters=[dict(operation=k[0],mode=k[1],inputs={n:p.kind for n,p in a.inputs.items()},
            outputs={n:p.kind for n,p in a.outputs.items()},version=a.version) for k,a in adapters.items()],
        median_training="2013-2014 station/hour; count>=30 else station training median",knn="24 distance-weighted calendar neighbors; 2013-2014 donors only"))
    observed=conditions(evaluator.external["batch"].value);risk=[]
    for r in enumeration["results"]:
        graph=restore(r["graph"]);evaluator.factory.inspect(graph,complete=True)
        risk.append(dict(key=r["key"],actual_status=r["status"],posterior_predicted_risk=evaluator.factory.semantic_risk(graph,observed)))
    pd.DataFrame(risk).to_csv(OUT/"posterior_coverage_audit.csv",index=False)
    graph=fixed_graphs(evaluator.factory)["H0"];key=key_for(graph,evaluator.factory)
    first_file=evaluator.folder/key/"result.json"
    a=json.loads(first_file.read_text(encoding="utf-8")) if first_file.exists() else evaluator.execute(graph,key)
    second=Evaluator(2016,states,OUT/"reproducibility_probe_second");second_file=second.folder/key/"result.json"
    b=json.loads(second_file.read_text(encoding="utf-8")) if second_file.exists() else second.execute(graph,key)
    with np.load(evaluator.folder/key/"prediction.npz") as aa,np.load(second.folder/key/"prediction.npz") as bb:
        np.testing.assert_array_equal(aa["prediction"],bb["prediction"])
    save(OUT/"actual_cold_reproducibility.json",dict(first=a["prediction_fingerprint"],second=b["prediction_fingerprint"],exact=True,
        independent_node_caches=True,year=2016,search_or_test_selection_changed=False))
    tests=set()
    for file in [OUT.parent/"tests_regression.xml",OUT/"tests_final.xml"]:
        tree=ET.parse(file)
        for case in tree.iter("testcase"):
            assert not any(case.find(k) is not None for k in ["failure","error","skipped"])
            tests.add((case.attrib.get("classname"),case.attrib["name"]))
    source_audit=pd.read_csv(OUT/"source_audit.csv")
    for r in source_audit.itertuples():assert sha(r.file)==r.sha256
    versions={name:md.version(name) for name in ["numpy","pandas","scipy","scikit-learn","joblib","xarray"]}
    versions.update(python=platform.python_version(),golem_path=str(__import__("golem").__file__),thegolem=md.version("thegolem"),
        dependency_warning="GOLEM 0.5.0 metadata on Python 3.12 declares NumPy<2 and SciPy<1.13; actual inherited hybrid environment is newer, targeted execution validated, not a clean upstream dependency lock")
    save(OUT/"environment.json",versions)
    resource=[json.loads(line) for line in (OUT/"process_resources.jsonl").read_text().splitlines()]
    save(OUT/"resource_summary.json",dict(samples=len(resource),sampling_seconds=2,scope="whole strict experiment process, not per graph peak",
        max_process_lifetime_peak_bytes=max(x["process_lifetime_peak_bytes"] for x in resource)))
    save(OUT/"final_acceptance.json",dict(searches=50,protocol_and_training_and_freeze_hashes_unchanged=True,complete_executed_graphs_in_same_grammar=True,
        tested_source_files=len(source_audit),prediction_hashes_verified=total_files,canonical_structures=949,
        reused_canonical_enumeration=enumeration.get("reused_executions",0),scored_evaluations=sum(x["scored"] for x in searches),
        raw_callbacks=callback_count,real_search_executions=sum(x["counts"].get("real_executions",0) for x in searches),
        unique_frozen_test_executions=final["executions"],distinct_related_tests_passed=len(tests),test_failures=0,test_skipped=0,
        search_quality_failures_preserved=sum(x["counts"].get("quality_failures",0) for x in searches),
        frozen_fixed_baseline_failures_preserved=sum(r["status"]!="PASS" for r in final["results"]),
        all_current_and_reused_prediction_hashes_verified=len(all_prediction_paths),
        retrospective_exploratory_only=True,third_party_golem_modified=False,baseline_failures_preserved=True,
        no_data_download_or_deletion=True,main_results_directory=str(OUT)))
    print((OUT/"final_acceptance.json").read_text(encoding="utf-8"))


if __name__=="__main__":verify()
