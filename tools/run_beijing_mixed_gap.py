"""Official GOLEM adapter, real numerical DAG objective, frozen exploratory test."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
import argparse
from datetime import datetime,timezone
import json
from time import perf_counter
import joblib,numpy as np
from prepare_beijing_mixed_gap import OUT,ROOT,save,sha,load,read_sources,build
from ecodata.algorithms.mixed_gap_dag import setup,TASK
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.numerical_search import (OptGraph,GraphRequirements,GPAlgorithmParameters,
    GraphGenerationParams,IdentityAdapter,typed_tree_growth)

SEEDS=[11,22,33,44,55,66,77,88,99,110]


# Research client: data loading and fixed evaluation protocol only.
from ecodata.algorithms.numerical_search import record,restore,key_for
from ecodata.algorithms.temporal_search import TemporalEvaluator,ReferenceEvaluation,submit_temporal_search


class Evaluator(TemporalEvaluator):
    def __init__(self,year,states,folder):
        external,truth,weights=load(year,states)
        self.truth=truth;self.weights=weights;self.year=year
        self.y=truth.truth.to_numpy();self.codes=truth.group_id.to_numpy()
        self.group_count=np.bincount(self.codes)
        evaluate=ReferenceEvaluation(self.y,codes=self.codes,weights=weights)
        super().__init__(external,evaluate,folder,dict(operation="temporal_missing_recovery",
            require_complete_coverage=True,prefer_observation_support=True,time_access="both_sides"))


def search_request(external,request,evaluate,config,folder,*,mode="C",seed=11,progress=None):
    """Same production submission service; research has no local search policy."""
    return submit_temporal_search(external,request,evaluate,config,folder,mode=mode,seed=seed,progress=progress)


def fixed_graphs(factory,*,explanatory=False):
    result={};h0=None;linear_nearest=None
    def named(recipe):
        children={n:c for n,_,c in recipe[2]}
        if recipe[0].startswith("baseline.temporal_"):return recipe[0].removeprefix("baseline.temporal_")
        if recipe[0]=="prepare.conditional_fallback":return ("fallback",named(children["primary"]),named(children["backup"]))
        if recipe[0]=="prepare.visible_gap_split":return ("split",dict(recipe[1])["threshold"],named(children["short"]),named(children["long"]))
        if recipe[0]=="quality.mixed_gap":return named(children["field"])
        return recipe[0]
    for _,recipe in factory.recipes("validated_field"):
        name=named(recipe)
        if isinstance(name,str) and name in ["linear","nearest","ffill","hour_median","calendar_knn"]:
            result[name]=OptGraph([factory.instantiate(recipe)])
        if name==("split",6,("fallback","linear","hour_median"),"hour_median"):h0=OptGraph([factory.instantiate(recipe)])
        if name==("fallback","linear","nearest"):linear_nearest=OptGraph([factory.instantiate(recipe)])
    assert len(result)==5 and h0 is not None
    result["H0"]=h0
    if explanatory:
        assert linear_nearest is not None
        result["linear_nearest"]=linear_nearest
    return result


def conditions(batch):
    rows=set()
    for case in batch.cases:
        v=case.values;missing=~np.isfinite(v);begins=np.flatnonzero(missing & ~np.r_[False,missing[:-1]]);ends=np.flatnonzero(missing & ~np.r_[missing[1:],False])+1
        rows.update((int(e-b),bool(b>0),bool(e<len(v))) for b,e in zip(begins,ends))
    a=np.array(sorted(rows));return a[:,0],a[:,1].astype(bool),a[:,2].astype(bool)


def search(mode,seed,states,config):
    folder=OUT/"search"/f"{mode}_{seed}"
    if (folder/"summary.json").exists():
        raise ValueError("Historical run already exists; preserve it and choose a new protocol/output directory")
    external,truth,weights=load(2016,states)
    evaluate=ReferenceEvaluation(truth.truth.to_numpy(),codes=truth.group_id.to_numpy(),weights=weights)
    request=dict(operation="temporal_missing_recovery",require_complete_coverage=True,
        prefer_observation_support=True,time_access="both_sides")
    return search_request(external,request,evaluate,config,folder,mode=mode,seed=seed)


def calibrate(states):
    e=Evaluator(2015,states,OUT/"development");fixed=fixed_graphs(e.factory)
    save(OUT/"H0_graph_sealed.json",record(fixed["H0"]))
    samples=list(fixed.items())
    # Representative generated graph, not selected by observed error.
    samples.append(("grammar_first_split",OptGraph([e.factory.instantiate(next(r for _,r in e.factory.recipes("validated_field") if r[2][0][2][0]=="prepare.visible_gap_split"))])))
    results=[]
    for name,graph in samples:
        result=e.execute(graph,key_for(graph,e.factory));results.append(dict(name=name,**result));print("DEVELOPMENT",name,result["status"],result.get("objective"),result["seconds"],flush=True)
    save(OUT/"development/results.json",results)
    recipes=e.factory.recipes("validated_field");seconds=np.mean([r["seconds"] for r in results])
    # Fixed before 2016 values are scored. Bound enumeration to a declared 900s.
    config=dict(version=TASK,objective_limit=50,proposal_limit=500,callback_limit=1000,wall_seconds=600,
        search_seeds=SEEDS,mask_seeds=[101,202,303],graph_space=len(recipes),enumeration_soft_seconds=900,
        enumeration_estimated_seconds=float(seconds*len(recipes)),enumeration_allowed=bool(seconds*len(recipes)<900),
        calibration="2015 only; defaults retained; no outcome-based budget change",cold_cache_per_run=True,
        no_human_reference_in_search=True,three_major_scenarios_equal=True,calendar_knn="official sklearn existing 24 neighbors distance; calendar only",
        statistic="station and time-block paired bootstrap; masks not independent; pointwise 95% intervals",
        frozen_at=datetime.now(timezone.utc).isoformat(),training_state_sha256=sha(OUT/"training_states.joblib"),
        source_hashes={p:sha(ROOT/p) for p in ["src/ecodata/algorithms/mixed_gap_dag.py","tools/prepare_beijing_mixed_gap.py","tools/run_beijing_mixed_gap.py"]})
    save(OUT/"protocol_frozen.json",config)
    return config


def run():
    states=joblib.load(OUT/"training_states.joblib")
    config=json.loads((OUT/"protocol_frozen.json").read_text(encoding="utf-8")) if (OUT/"protocol_frozen.json").exists() else calibrate(states)
    searches=[]
    for seed in SEEDS:
        for mode in ["A","B","C","C-gap","R"]:searches.append(search(mode,seed,states,config))
    # Independent cold enumeration reference AFTER searches; never exposed to them.
    e=Evaluator(2016,states,OUT/"enumeration");all_results=[];start=perf_counter()
    reference=OUT.parent
    reuse=reference/"source_v1/mixed_gap_dag.py"
    can_reuse=False
    if reuse.exists():
        def kernels(path):
            tree=ast.parse(path.read_text(encoding="utf-8"))
            return [ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"readonly","missing_runs","calendar_features","predict","setup"}]
        can_reuse=(kernels(reuse)==kernels(ROOT/"src/ecodata/algorithms/mixed_gap_dag.py") and
            sha(reference/"data/2016/public_inputs.joblib")==sha(OUT/"data/2016/public_inputs.joblib") and
            sha(reference/"training_states.joblib")==sha(OUT/"training_states.joblib"))
    reuse_count=0
    for _,recipe in e.factory.recipes("validated_field") if config["enumeration_allowed"] else []:
        if perf_counter()-start>config["enumeration_soft_seconds"]:break
        graph=OptGraph([e.factory.instantiate(recipe)]);key=key_for(graph,e.factory);file=e.folder/key/"result.json"
        old=reference/"enumeration"/key/"result.json"
        if file.exists():result=json.loads(file.read_text(encoding="utf-8"))
        elif can_reuse and old.exists():
            result=json.loads(old.read_text(encoding="utf-8"));result=dict(result,reused_from=str(old.parent),numerical_kernel_ast_unchanged=True)
            reuse_count+=1
        else:result=e.execute(graph,key)
        all_results.append(result)
        if len(all_results)%100==0:print("ENUM",len(all_results),flush=True)
    save(OUT/"enumeration_summary.json",dict(count=len(all_results),complete=len(all_results)==config["graph_space"],results=all_results,seconds=perf_counter()-start,reused_executions=reuse_count))
    selected={};mapping=[]
    for r in searches:
        if r["best"]:selected[r["best"]["key"]]=r["best"]["graph"];mapping.append(dict(mode=r["mode"],seed=r["seed"],key=r["best"]["key"]))
        else:mapping.append(dict(mode=r["mode"],seed=r["seed"],key=None))
    fixed=fixed_graphs(e.factory)
    for name,graph in fixed.items():
        key=key_for(graph,e.factory);selected[key]=record(graph);mapping.append(dict(mode=name,seed=None,key=key))
    freeze=dict(frozen_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/"protocol_frozen.json"),graphs=selected,mapping=mapping,
        final_2017_target_scored=False,history_exposure="exploratory refollowup, not blind new test")
    if not (OUT/"selected_graphs_frozen.json").exists():save(OUT/"selected_graphs_frozen.json",freeze)
    else:assert json.loads((OUT/"selected_graphs_frozen.json").read_text(encoding="utf-8"))["graphs"]==selected
    if not (OUT/"data/2017/summary.json").exists():build(read_sources(),2017,states)
    test=Evaluator(2017,states,OUT/"frozen_test");test_results=[]
    for key,rows in selected.items():
        file=test.folder/key/"result.json"
        result=json.loads(file.read_text(encoding="utf-8")) if file.exists() else test.execute(restore(rows),key)
        test_results.append(result)
    save(OUT/"test_results.json",dict(freeze_sha256=sha(OUT/"selected_graphs_frozen.json"),results=test_results,executions=len(test_results)))
    # Fixed validation baselines, same full denominator and exact executor.
    e=Evaluator(2016,states,OUT/"fixed_validation")
    save(OUT/"fixed_validation_results.json",{name:e.execute(graph,key_for(graph,e.factory)) for name,graph in fixed.items()})
    print("ALL_RUNS_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--calibrate-only",action="store_true");args=p.parse_args()
    if args.calibrate_only:calibrate(joblib.load(OUT/"training_states.joblib"))
    else:run()
