"""Bounded 2015 decision/weight replays and three small numerical route checks.

No search, grammar enumeration, fitting, new masks, RMSE evaluation or 2017 data.
"""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tools"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
import joblib,numpy as np,xarray as xr
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.algorithms.mixed_gap_dag import setup,SeriesCase,Batch,SealedArtifact,readonly,TASK
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.golem_search import semantic_proposal_weights
from ecodata.core.schemas import EcoDataBundle
from run_beijing_mixed_gap import restore,record
from golem.core.optimisers.graph import OptGraph

OUT=ROOT/"output/decision_boundary_fixes_20261020"
HISTORY=ROOT/"output/beijing_mixed_gap_20261020/unified_platform_v4"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def previous(name):
    spec=importlib.util.spec_from_file_location("boundary_before_"+name,OUT/"source_before"/(name+".py"))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def decide(registry,cases,extractor,selector):
    c=cases[0];ds=xr.Dataset({"pm25":("time",c.values)},coords={"time":c.time})
    context=AlgorithmContext((EcoDataBundle(ds),),"pm25","continuous")
    semantics=extractor.extract_series(cases,variable="pm25",unit="ug m-3",nonnegative=True)
    return selector(registry).decide(context,dict(operation="temporal_missing_recovery",require_complete_coverage=True,
        prefer_observation_support=True,time_access="both_sides"),semantics=semantics)


def partial(rows):
    g=restore(rows);root=g.root_node
    root.nodes_from=[p for p in root.nodes_from if p.name=="source.masked_batch"]
    return OptGraph([root])


def main():
    OUT.mkdir(exist_ok=True)
    protected=[HISTORY/n for n in ["protocol_frozen.json","selected_graphs_frozen.json","test_results.json",
        "final_acceptance.json","data/2015/public_inputs.joblib","training_states.joblib"]]
    hashes={str(p):sha(p) for p in protected}
    data=joblib.load(HISTORY/"data/2015/public_inputs.joblib");batch=data["batch"]
    states=joblib.load(HISTORY/"training_states.joblib")
    for c in batch.cases:
        for a in [c.time,c.values,c.missing]:a.flags.writeable=False
    oldrouter,oldsem,oldmixed,oldpipeline=(previous(n) for n in ["router","semantic_requirements","mixed_gap_dag","pipeline"])
    oldreg,oldf=oldmixed.setup();reg,f=setup()
    old=decide(oldreg,batch.cases,oldsem.SemanticFeatureExtractor(),oldrouter.AlgorithmSelector)
    new=decide(reg,batch.cases,SemanticFeatureExtractor(),AlgorithmSelector)
    assert old.requirements.as_dict()==new.requirements.as_dict()
    graphs={r["name"]:r.get("result",r)["graph"] for p in ["mechanism_acceptance/results.json","development/results.json"] for r in read(HISTORY/p)}
    results=[]
    for name in ["nearest","linear_nearest","H0","linear","hour_median","incomplete"]:
        g=partial(graphs["nearest"]) if name=="incomplete" else restore(graphs[name])
        before=old.assess_graph(oldf,g);after=new.assess_graph(f,g)
        fields=["gap","port_gap","hard_reject","support_deficit","output_capabilities"]
        results.append(dict(name=name,graph=record(g),before=before,after=after,
            common_fields_equal=all(before[k]==after[k] for k in fields)))
    initial=partial(graphs["nearest"]);options=[restore(graphs[n]) for n in ["nearest","hour_median"]]
    weights=[];saved=read(HISTORY/"mechanism_acceptance/proposal_control.json")
    for soft,key in [(True,"soft_on"),(False,"soft_off")]:
        _,a=semantic_proposal_weights(old,oldf,initial,options,soft=soft)
        _,b=semantic_proposal_weights(new,f,initial,options,soft=soft)
        for field in ["weights","probabilities"]:
            np.testing.assert_allclose(a[field],b[field],rtol=0,atol=0)
            np.testing.assert_allclose(a[field],saved[key][field],rtol=0,atol=0)
        weights.append(dict(soft=soft,before=a,after=b,archived_2015_exact=True))
    # Only three cases, selected by source order and missing support, never values/errors.
    indices=[0]
    boundary=next(i for i,c in enumerate(batch.cases) if not np.isfinite(c.values[[0,-1]]).all())
    if boundary not in indices:indices.append(boundary)
    for i in range(len(batch.cases)):
        if i not in indices:indices.append(i)
        if len(indices)==3:break
    cases=tuple(batch.cases[i] for i in indices);pieces=[batch.lengths[slice(*batch.offsets[i])] for i in indices]
    lengths=readonly(np.concatenate(pieces));ends=np.cumsum([0]+[len(p) for p in pieces]);offsets=tuple(zip(ends[:-1],ends[1:]))
    small=Batch(cases,lengths,offsets);meta=dict(unit="ug m-3",contract=TASK)
    external=dict(batch=SealedArtifact("masked_batch",small,meta),history=SealedArtifact("fitted_history",states,meta),
        query=SealedArtifact("query_positions",readonly(np.arange(len(lengths))),meta))
    small_d=decide(reg,cases,SemanticFeatureExtractor(),AlgorithmSelector)
    old_pipe=oldpipeline.UnifiedAlgorithmPipeline(oldreg);new_pipe=UnifiedAlgorithmPipeline(reg)
    executions=[];arrays={}
    for name in ["nearest","linear_nearest","H0"]:
        g=restore(graphs[name]);a=old_pipe.run_dag(g,task_id=TASK,external=external,version=TASK)
        b=new_pipe.run_dag(g,task_id=TASK,external=external,version=TASK,decision=small_d,factory=f)
        for index,label in enumerate(["raw","projected"]):
            av=a["outputs"]["field"].value[index];bv=b["outputs"]["field"].value[index]
            np.testing.assert_array_equal(av,bv);arrays[name+"_before_"+label]=av;arrays[name+"_after_"+label]=bv
        executions.append(dict(name=name,equal_raw_and_projected=True,positions=len(lengths),
            before_trace=a["trace"],after_trace=b["trace"],task_acceptance=b["task_acceptance"]["acceptance_status"]))
    np.savez_compressed(OUT/"bounded_predictions.npz",**arrays)
    with np.load(OUT/"bounded_predictions.npz") as reopened:
        for name,values in arrays.items():np.testing.assert_array_equal(values,reopened[name])
    def kernels(path):
        return [ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding="utf-8")).body
            if isinstance(n,ast.FunctionDef) and n.name in {"predict","missing_runs","calendar_features","readonly"}]
    assert kernels(OUT/"source_before/mixed_gap_dag.py")==kernels(ROOT/"src/ecodata/algorithms/mixed_gap_dag.py")
    assert all(sha(p)==v for p,v in hashes.items())
    save(OUT/"representatives.json",results);save(OUT/"weights.json",weights);save(OUT/"executions.json",executions)
    save(OUT/"summary.json",dict(scope="six 2015 decision checks, one archived 2015 two-candidate control with soft on/off; three routes before/after on three cases",
        requirements_equal=True,common_assessments_equal=all(r["common_fields_equal"] for r in results),
        weights_and_probabilities_exact=True,numerical_routes=3,numerical_runs=6,case_indices=indices,positions=len(lengths),
        raw_projected_and_split_outputs_equal=True,prediction_file_sha256=sha(OUT/"bounded_predictions.npz"),
        kernels_AST_unchanged=True,protected_history_hashes=hashes,all_protected_files_unchanged=True,
        search_runs=0,enumeration_runs=0,full_numerical_experiments=0,private_targets_read=False,
        NOT_PROVEN="all Beijing proposals/graphs, full experiment identity, other data frequencies supported by this splitter"))
    print("BOUNDED_CHECK_PASS",len(results),"decisions;",len(indices),"2015 cases; 3 old/new routes")


if __name__=="__main__":main()
