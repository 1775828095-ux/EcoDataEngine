"""Versioned acceptance/calibration/orchestration; all policy is in the platform.

Reuses the historical numerical runner and frozen masks. This file contains no
method support map, preference policy or replacement optimiser.
"""
import os
os.environ.setdefault("ECODATA_BEIJING_MIXED_OUTPUT",str(__import__("pathlib").Path(__file__).resolve().parents[1]/
    "output/beijing_mixed_gap_20261020/unified_platform_v4"))
import argparse,json,shutil,sys,inspect
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import run_beijing_mixed_gap as base
from prepare_beijing_mixed_gap import OUT,ROOT,save,sha
from ecodata.algorithms.golem_search import semantic_proposal_weights
from golem.core.optimisers.graph import OptGraph

OLD=OUT.parent/"strict_grammar_v2"
FILES=["src/ecodata/algorithms/"+n for n in ["router.py","semantic_requirements.py","semantic_rules.json",
    "capabilities.py","contracts.py","registry.py","golem_search.py","golem_generation.py","pipeline.py",
    "mixed_gap_dag.py","temporal_reconstruction_dag.py","dag.py"]]+[
    "src/ecodata/interface/raster_upload.py","tools/run_beijing_mixed_gap.py","tools/run_beijing_unified_decision.py",
    "tools/prepare_beijing_mixed_gap.py"]


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[p for p in (OLD/"data").rglob("*") if p.is_file()]
    files += [OLD/n for n in ["training_states.joblib","training_statistics.csv","training_fit_cost.json","source_audit.csv","H0_sealed.json"]]
    manifest=[]
    for source in files:
        dest=OUT/source.relative_to(OLD);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(source,dest)
        assert sha(source)==sha(dest)
        manifest.append(dict(source=str(source),path=str(dest),sha256=sha(source),bytes=source.stat().st_size,reused_without_rebuilding=True))
    save(OUT/"reused_artifacts.json",manifest)
    # A pre-run snapshot protects old data/results even though code is extended.
    history={str(p.relative_to(OLD)):sha(p) for p in [OLD/"protocol_frozen.json",OLD/"selected_graphs_frozen.json",OLD/"test_results.json"]}
    save(OUT/"history_protected.json",history)


def acceptance():
    tree=ET.parse(OUT/"tests_mechanism.xml")
    assert not tree.findall(".//failure") and not tree.findall(".//error")
    states=base.joblib.load(OUT/"training_states.joblib")
    e=base.Evaluator(2015,states,OUT/"mechanism_acceptance/executions")
    d=e.decision();f=e.factory;graphs=base.fixed_graphs(f,explanatory=True)
    actual=[]
    for name in ["nearest","linear_nearest"]:
        graph=graphs[name];a=d.assess_graph(f,graph)
        assert not a["hard_reject"]
        result=e.execute(graph,base.key_for(graph,f));assert result["status"]=="PASS" and result["file_readback"]=="PASS"
        actual.append(dict(name=name,assessment=a,result=result))
    assert actual[0]["result"]["prediction_fingerprint"]!=actual[1]["result"]["prediction_fingerprint"]
    _,on=semantic_proposal_weights(d,f,f.initial(),[graphs["nearest"],graphs["hour_median"]])
    _,off=semantic_proposal_weights(d,f,f.initial(),[graphs["nearest"],graphs["hour_median"]],soft=False)
    assert on["probabilities"]!=off["probabilities"]
    save(OUT/"mechanism_acceptance/decision.json",d.as_dict())
    save(OUT/"mechanism_acceptance/proposal_control.json",dict(soft_on=on,soft_off=off))
    save(OUT/"mechanism_acceptance/results.json",actual)
    save(OUT/"mechanism_acceptance.json",dict(status="PASS",year=2015,
        tests=len(tree.findall(".//testcase")),test_xml_sha256=sha(OUT/"tests_mechanism.xml"),
        actual_distinct_routes=2,controlled_requirements_and_probabilities="tests/test_unified_decision.py",
        production_entry="src/ecodata/interface/raster_upload.py:process_raster_upload -> AlgorithmSelector.decide",
        experiment_entry="tools/run_beijing_mixed_gap.py:Evaluator.decision -> AlgorithmSelector.decide",
        hidden_truth_received_by_decision=False,accuracy_prediction_claim=False))


def freeze():
    assert json.loads((OUT/"mechanism_acceptance.json").read_text())["status"]=="PASS"
    states=base.joblib.load(OUT/"training_states.joblib")
    if (OUT/"protocol_frozen.json").exists():return
    config=base.calibrate(states) # 2015 only; same original budgets and parameters.
    config.update(version="unified_platform_v4",semantic_chain="platform_v1",mechanism_gate_sha256=sha(OUT/"mechanism_acceptance.json"),
        source_hashes={p:sha(ROOT/p) for p in FILES},
        switches={"A":dict(typed_safety=True,posterior=False,gap_guide=False,soft_guide=False),
            "B":dict(typed_safety=True,posterior=True,gap_guide=False,soft_guide=False),
            "C":dict(typed_safety=True,posterior=True,gap_guide=True,soft_guide=True),
            "C-gap":dict(typed_safety=True,posterior=True,gap_guide=True,soft_guide=False),
            "R":dict(typed_safety=True,posterior=False,gap_guide=False,soft_guide=False,random_complete=True)},
        guide_parameters=dict(alpha=2.,beta=4.,source="original coefficients retained, no 2016/2017 tuning"),
        soft_weighting="all visible missing-input positions; objective remains frozen equal scenario/group J",
        soft_meaning="retained-observation support vs history/endpoint extension, NOT accuracy prediction",
        actual_algorithm_count=5,frozen_platform_pool=34,grammar_count=949,
        algorithm_pool_scope="five existing temporal baseline modes; other frozen components lack compatible temporal ports, not arbitrary exclusions",
        no_legacy_enumeration_or_test_read_by_search=True,
        uncertainty="paired stations and 7-day blocks, 1000 bootstrap draws, pointwise exploratory intervals; seeds not independent environmental samples")
    save(OUT/"protocol_frozen.json",config)
    for name in FILES:
        dest=OUT/"source_executed"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)


def check_sources(config):
    assert all(sha(ROOT/n)==value for n,value in config["source_hashes"].items()),"Changed source requires a new protocol/output version"


def experiment():
    config=json.loads((OUT/"protocol_frozen.json").read_text());check_sources(config)
    states=base.joblib.load(OUT/"training_states.joblib");searches=[]
    for seed in base.SEEDS:
        for mode in ["A","B","C","C-gap","R"]:searches.append(base.search(mode,seed,states,config))
    # Canonical grammar acceptance occurs BEFORE any newly frozen 2017 followup.
    e=base.Evaluator(2016,states,OUT/"enumeration");recipes=e.factory.recipes("validated_field")
    canonical={base.key_for(OptGraph([e.factory.instantiate(r)]),e.factory) for _,r in recipes}
    complete={r["key"] for run in searches for r in run["rows"] if r["unique_execution"]}
    assert len(canonical)==949 and complete<=canonical
    save(OUT/"pre_followup_grammar_acceptance.json",dict(status="PASS",canonical_count=len(canonical),
        complete_evaluated_count=len(complete),outside=sorted(complete-canonical),test_loaded=False,
        at=datetime.now(timezone.utc).isoformat()))
    results=[];clock=perf_counter()
    for _,r in recipes:
        graph=OptGraph([e.factory.instantiate(r)]);key=base.key_for(graph,e.factory)
        file=e.folder/key/"result.json"
        result=json.loads(file.read_text()) if file.exists() else e.execute(graph,key)
        results.append(result)
        if len(results)%100==0:print("UNIFIED_ENUM",len(results),flush=True)
    save(OUT/"enumeration_summary.json",dict(count=len(results),complete=len(results)==949,results=results,seconds=perf_counter()-clock,reused_executions=0))
    # Cross-check posterior evidence with actual execution; do not tune on it.
    contracts=[]
    d=e.decision()
    for r in results:
        assessment=d.assess_graph(e.factory,base.restore(r["graph"]))
        contracts.append(dict(key=r["key"],rejected=assessment["hard_reject"],actual=r["status"]))
    assert all(r["rejected"]==(r["actual"]!="PASS") for r in contracts)
    save(OUT/"posterior_actual_crosscheck.json",dict(status="PASS",count=len(contracts),rows=contracts))
    fixed=base.fixed_graphs(e.factory,explanatory=True)
    selected={};mapping=[]
    for run in searches:
        best=run["best"]
        if best:selected[best["key"]]=best["graph"]
        mapping.append(dict(mode=run["mode"],seed=run["seed"],key=best["key"] if best else None))
    for name,graph in fixed.items():
        key=base.key_for(graph,e.factory);selected[key]=base.record(graph);mapping.append(dict(mode=name,seed=None,key=key))
    frozen=dict(at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/"protocol_frozen.json"),
        grammar_acceptance_sha256=sha(OUT/"pre_followup_grammar_acceptance.json"),graphs=selected,mapping=mapping,
        final_2017_target_scored=False,exposure="historically exposed exploratory followup; no new blind confirmation")
    file=OUT/"selected_graphs_frozen.json"
    if file.exists():assert json.loads(file.read_text())["graphs"]==selected
    else:save(file,frozen)
    e=base.Evaluator(2016,states,OUT/"fixed_validation")
    save(OUT/"fixed_validation_results.json",{name:e.execute(g,base.key_for(g,e.factory)) for name,g in fixed.items()})
    test=base.Evaluator(2017,states,OUT/"frozen_test");outputs=[]
    for key,rows in selected.items():
        file=test.folder/key/"result.json"
        outputs.append(json.loads(file.read_text()) if file.exists() else test.execute(base.restore(rows),key))
    save(OUT/"test_results.json",dict(freeze_sha256=sha(OUT/"selected_graphs_frozen.json"),results=outputs,executions=len(outputs)))
    check_sources(config)
    save(OUT/"execution_complete.json",dict(searches=len(searches),enumeration=len(results),followup_unique=len(outputs),source_hashes_unchanged=True))
    print("UNIFIED_EXECUTION_COMPLETE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","accept","freeze","run"]);args=parser.parse_args()
    if args.stage=="prepare":initialize()
    elif args.stage=="accept":acceptance()
    elif args.stage=="freeze":freeze()
    else:experiment()
