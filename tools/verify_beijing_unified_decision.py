"""Independent evidence audit; no selection policy and no search retries."""
import json,inspect,platform
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np,pandas as pd
import run_beijing_unified_decision as u
from prepare_beijing_mixed_gap import DATA
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor,RequirementMapper
from ecodata.algorithms.router import AlgorithmSelector,SemanticDecision
from ecodata.algorithms.golem_search import GolemCapabilitySearch,semantic_proposal_weights
from ecodata.algorithms.golem_generation import TypedGolemFactory,typed_tree_growth
from ecodata.algorithms.capabilities import coverage
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.interface.raster_upload import process_raster_upload,route


def locate(obj):
    lines,start=inspect.getsourcelines(obj)
    return dict(path=str(Path(inspect.getsourcefile(obj)).resolve().relative_to(u.ROOT)),start=start,end=start+len(lines)-1)


def main():
    config=json.loads((u.OUT/"protocol_frozen.json").read_text(encoding="utf-8"));u.check_sources(config)
    protected=json.loads((u.OUT/"history_protected.json").read_text(encoding="utf-8"))
    for file,value in protected.items():assert u.sha(u.OLD/file)==value
    reused=json.loads((u.OUT/"reused_artifacts.json").read_text(encoding="utf-8"))
    for row in reused:assert u.sha(row["source"])==u.sha(row["path"])==row["sha256"]
    sources=pd.read_csv(u.OUT/"source_audit.csv")
    for row in sources.itertuples():assert u.sha(row.file)==row.sha256
    freeze=json.loads((u.OUT/"selected_graphs_frozen.json").read_text(encoding="utf-8"))
    assert freeze["protocol_sha256"]==u.sha(u.OUT/"protocol_frozen.json")
    final=json.loads((u.OUT/"test_results.json").read_text(encoding="utf-8"))
    assert final["freeze_sha256"]==u.sha(u.OUT/"selected_graphs_frozen.json")
    enum=json.loads((u.OUT/"enumeration_summary.json").read_text(encoding="utf-8"))
    assert enum["complete"] and enum["count"]==949
    keys={r["key"] for r in enum["results"]};searches=[json.loads(p.read_text(encoding="utf-8")) for p in (u.OUT/"search").glob("*/summary.json")]
    assert len(searches)==50
    counters=Counter();factory_rows=[];guide_rows=[]
    for run in searches:
        assert run["semantic_chain"]=="platform_v1" and not run["optimizer_error"]
        assert run["config_sha256"]==u.sha(u.OUT/"protocol_frozen.json")
        assert run["scored"]<=50 and run["counts"]["proposals"]<=500
        assert len({r["key"] for r in run["rows"]})==len(run["rows"])==run["scored"]
        assert len(run["events"])==run["counts"]["callbacks"]
        assert sum(r["status"]=="DEFERRED_NO_EXECUTION" for r in run["events"])==run["counts"].get("deferred_callbacks",0)
        for row in run["rows"]:
            assert "decision_feedback" in row
            if row["unique_execution"]:assert row["key"] in keys
            counters[row["status"]]+=1
        folder=u.OUT/"search"/f"{run['mode']}_{run['seed']}"
        trace=json.loads((folder/"factory_trace.json").read_text(encoding="utf-8"))
        factory_rows.append(dict(mode=run["mode"],seed=run["seed"],
            outer_proposals=run["counts"]["proposals"],
            constructions=sum(r.get("event")=="random_graph_request" for r in trace),
            created=sum(r.get("event")=="create" for r in trace),
            connected=sum(r.get("event")=="connect" for r in trace),
            rejected=sum(r.get("event","").startswith("reject") for r in trace),
            seconds=run["seconds"],platform_setup_seconds=run["platform_setup_seconds"],platform_total_seconds=run["platform_total_seconds"]))
        for row in run["proposals"]:
            if row["status"]!="SHARED_DECISION":continue
            np.testing.assert_allclose(np.sum(row["probabilities"]),1)
            assert row["keys"][row["chosen_index"]] in row["keys"]
            assert row["soft_enabled"]==(run["mode"]=="C")
            guide_rows.append(dict(mode=run["mode"],seed=run["seed"],elapsed=row["elapsed"],
                nonuniform=bool(np.ptp(row["probabilities"])>1e-12),
                minimum_probability=min(row["probabilities"]),maximum_probability=max(row["probabilities"]),
                chosen_key=row["keys"][row["chosen_index"]],soft=row["soft_enabled"]))
    pd.DataFrame(factory_rows).to_csv(u.OUT/"factory_generation_accounting.csv",index=False)
    pd.DataFrame(guide_rows).to_csv(u.OUT/"guidance_consumption.csv",index=False)
    assert any(r["nonuniform"] for r in guide_rows if r["mode"]=="C")
    paths=list(u.OUT.rglob("result.json"));files=0
    for path in paths:
        r=json.loads(path.read_text(encoding="utf-8"));file=path.with_name("prediction.npz")
        if "prediction_file_sha256" not in r:continue
        assert u.sha(file)==r["prediction_file_sha256"]
        with np.load(file) as arrays:
            assert len(arrays["prediction"])==r["denominator"]
            if r["status"]=="PASS":assert np.isfinite(arrays["prediction"]).all() and arrays["prediction"].min()>=0
        files+=1
    states=u.base.joblib.load(u.OUT/"training_states.joblib")
    e=u.base.Evaluator(2016,states,u.OUT/"cold_replay_1");graph=u.base.fixed_graphs(e.factory,explanatory=True)["linear_nearest"]
    key=u.base.key_for(graph,e.factory)
    def cold(evaluator):
        path=evaluator.folder/key/"result.json"
        if path.exists():
            result=json.loads(path.read_text(encoding="utf-8"))
            assert u.sha(path.with_name("prediction.npz"))==result["prediction_file_sha256"]
            return result
        return evaluator.execute(graph,key)
    first=cold(e);assert first["status"]=="PASS"
    other=u.base.Evaluator(2016,states,u.OUT/"cold_replay_2");second=cold(other);assert second["status"]=="PASS"
    assert first["prediction_fingerprint"]==second["prediction_fingerprint"]
    adapters=e.pipeline.registry.task_adapters(u.base.TASK)
    u.save(u.OUT/"candidate_contracts.json",dict(science=[s.as_dict() for s in e.pipeline.registry.specs()],
        modes=[dict(algorithm=k[0],mode=k[1],inputs={n:p.kind for n,p in a.inputs.items()},outputs={n:p.kind for n,p in a.outputs.items()},
            decision_contract=a.decision_contract,version=a.version) for k,a in adapters.items()],
        scope="Actual five existing named-port task kernels; generic single-Series calendar KNN run is not implemented by this registration",
        frozen_34_sha256=u.sha(u.ROOT/"output/capability_final_acceptance/frozen_generic_pool.json")))
    objects=[SemanticFeatureExtractor.extract,SemanticFeatureExtractor.extract_series,RequirementMapper.map,
        AlgorithmSelector.decide,AlgorithmSelector.screen,SemanticDecision.assess_graph,semantic_proposal_weights,
        TypedGolemFactory.recipes,TypedGolemFactory.inspect,typed_tree_growth,UnifiedAlgorithmPipeline.run_dag,
        GolemCapabilitySearch.run,coverage,process_raster_upload,route,u.base.Evaluator.decision,u.base.search]
    audit=[dict(component=o.__qualname__,**locate(o)) for o in objects]
    rules=json.loads((u.ROOT/"src/ecodata/algorithms/semantic_rules.json").read_text(encoding="utf-8"))["rules"]
    u.save(u.OUT/"call_chain_audit.json",dict(source_locations=audit,
        before=dict(structural="SemanticFeatureExtractor -> RequirementMapper -> GolemCapabilitySearch -> run_golem_search:metrics (whole-graph union, no numeric DAG objective)",
            raster="recognise -> existing compat screen -> raster-specific learned precision router -> Pipeline; no RequirementMapper",
            beijing="Typed DAG + real J; local unique support-set risk; no shared mapper/catalog consumer",
            taiwan="Mapper assigned factory.requirements; search guided by typed port gap and node count, not consuming requirements"),
        after=dict(beijing="masked public arrays -> extract_series -> RequirementMapper -> capability_catalog + real plugin compatibility -> assess_graph named outputs/state -> shared weights -> official GOLEM -> Pipeline.run_dag -> independent J",
            raster="same AlgorithmSelector.decide actual first-step eligibility filter; raster-specific pretrained development router retained as independent accuracy mode",
            unchanged_entries=["legacy structural search","Taiwan historical search; not silently promoted to shared new guidance"],
            not_claimed=["generic accuracy prediction from input variance or peak height","full transfer of raster learned precision model to temporal task","all platform entries migrated"]),
        feature_consumers=dict(variable_class="mapper + actual compatibility",unit="known/unknown mapping + runtime unit metadata",nonnegative="rule -> output validation capability + actual quality gate",
            missing_support="one-sided requirement + per-output coverage + weighted soft evidence",gap_length="declared select control predicates, NOT method winner mapping",
            finite_missing_counts="audit; support count denominator only",resolution="preserved metadata; no resampling requested here",timezone="UNKNOWN, not inferred",crs="UNKNOWN, not needed for per-station temporal restoration",
            declared_compute_cost="audit only in this temporal guide; actual runtime reported independently",
            raster_numeric_features="existing raster-only development model, not imported as Beijing precision predictor"),
        rule_consumers=[dict(rule=r["id"],required=r.get("required",[]),hard=r.get("hard",[]),soft=r.get("soft",[]),
            scope="temporal rules consumed by shared graph assessment/guidance; other domains retain explicit older consumers or declaration-only status") for r in rules],
        declaration_only_soft=sorted({s for r in rules for s in r.get("soft",[])}-{"retained_observation_support"}),
        hard_not_consumed_by_temporal_guidance=sorted({s for r in rules for s in r.get("hard",[])}-
            {"masked_target_isolation","nonnegative_output","past_only","no_semantic_conflict"}),
        hard_consumers=dict(masked_target_isolation="case-isolated read-only public values; hidden labels held only in evaluator; source/mask fingerprints unchanged",
            nonnegative_output="actual quality adapter projection and ResultQualityGate all arms",
            complete_temporal_coverage="common final quality; posterior early rejection only B/C/C-gap",
            past_only="controlled matching rejection; present experiment uses both_sides, no claim that legacy entry gained a causal executor",
            other_constraints="not all have consumers in this temporal task; do not claim universality"),
        numeric_objective="independent Evaluator.execute: full frozen cohort J; never replaced by static capability union"))
    tree=ET.parse(u.OUT/"tests_final.xml")
    assert not tree.findall(".//failure") and not tree.findall(".//error")
    resources=[json.loads(line) for line in (u.OUT/"process_resources.jsonl").read_text().splitlines()]
    started=json.loads((u.OUT/"process_start.json").read_text())
    rejected_resources=[r for r in resources if r["pid"]!=started["pid"]]
    resources=[r for r in resources if r["pid"]==started["pid"]]
    u.save(u.OUT/"resource_summary.json",dict(samples=len(resources),sample_interval_seconds=2,
        sampled_max_rss=max(r["rss_bytes"] for r in resources),process_lifetime_peak=max(r["process_lifetime_peak_bytes"] for r in resources),
        last_sample_wall_since_process_start=resources[-1]["unix_time"]-started["created_unix"],
        excluded_other_pid_samples=len(rejected_resources),raw_samples_preserved=True,
        note="CPU subprocess peak; sampling began after first few runs. Lifetime peak reported by OS, not a full-system resource estimate"))
    u.save(u.OUT/"final_acceptance.json",dict(status="PASS_FOR_BOUNDED_TEMPORAL_CHAIN",searches=len(searches),counter=dict(counters),
        actual_prediction_files_read_and_hashed=files,canonical_complete=949,passing=sum(r["status"]=="PASS" for r in enum["results"]),
        tests=len(tree.findall(".//testcase")),failures=0,skipped=len(tree.findall(".//skipped")),
        input_originals_unchanged=12,old_history_hashes_unchanged=True,all_reused_masks_and_states_identical=True,
        frozen_protocol_unchanged=True,cold_actual_replay_exact=True,
        source_hashes=config["source_hashes"],postprocessing_hashes={n:u.sha(u.ROOT/n) for n in ["tools/analyze_beijing_mixed_gap.py",str(Path(__file__).relative_to(u.ROOT)),"tests/test_unified_decision.py"]},
        followup="historically exposed exploratory; not new blind confirmation",accuracy_preference_model="NOT_IMPLEMENTED",
        no_unrelated_full_regression=True))
    print("FINAL_VERIFIED",files,"prediction files",len(searches),"searches")


if __name__=="__main__":main()
