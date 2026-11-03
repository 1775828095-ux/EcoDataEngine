"""Two bounded real executions to attribute effective scored-target control use."""
from collections import Counter
import json,joblib,numpy as np
from prepare_beijing_mixed_gap import OUT,save,load
from run_beijing_mixed_gap import setup,fixed_graphs,restore,key_for,TASK,UnifiedAlgorithmPipeline


def main():
    states=joblib.load(OUT/"training_states.joblib");external,_,_=load(2016,states)
    selected=json.loads((OUT/"search/C_11/summary.json").read_text(encoding="utf-8"))["best"]
    registry,factory=setup();graphs={"H0":fixed_graphs(factory)["H0"],"C_seed11":restore(selected["graph"])}
    evidence={}
    for name,graph in graphs.items():
        registry,_=setup();pipeline=UnifiedAlgorithmPipeline(registry)
        result=pipeline.run_dag(graph,task_id=TASK,external=external,version=TASK)
        engine=pipeline._dag_executors[TASK,TASK];nodes,sink=engine.from_golem(graph);by={n.id:n for n in nodes}
        active=np.zeros(len(external["batch"].value.lengths),bool);active[external["query"].value]=True
        controls=[];usage=Counter()
        def walk(key,mask):
            n=by[key]
            if n.operation=="quality.mixed_gap":walk(n.inputs["field"][0],mask)
            elif n.operation=="prepare.visible_gap_split":
                short=external["batch"].value.lengths<=n.parameters["threshold"]
                controls.append(dict(node=key,operation=n.operation,threshold=n.parameters["threshold"],effective_short=int((mask&short).sum()),effective_long=int((mask&~short).sum())))
                walk(n.inputs["short"][0],mask&short);walk(n.inputs["long"][0],mask&~short)
            elif n.operation=="prepare.conditional_fallback":
                primary,port=n.inputs["primary"];missing=~np.isfinite(result["intermediates"][primary][port].value)
                controls.append(dict(node=key,operation=n.operation,effective_fallback_targets=int((mask&missing).sum()),effective_primary_targets=int((mask&~missing).sum())))
                walk(primary,mask&~missing);walk(n.inputs["backup"][0],mask&missing)
            elif n.operation.startswith("baseline.temporal_"):usage[n.operation]+=int(mask.sum())
        walk(sink,active)
        key=key_for(graph,factory)
        baseline_file=(OUT/"fixed_validation" if name=="H0" else OUT/"search/C_11/executions")/key/"prediction.npz"
        with np.load(baseline_file) as old:np.testing.assert_array_equal(old["prediction"],result["outputs"]["field"].value[1])
        evidence[name]=dict(key=key,controls=controls,effective_method_target_counts=dict(usage),target_count=int(active.sum()),
            exact_saved_prediction_reproduced=True,trace=result["trace"])
    save(OUT/"representative_controls.json",dict(additional_validation_executions=2,search_budget_unchanged=True,scope="effective scored-target usage, not eager branch-kernel execution counts",graphs=evidence))
    print(json.dumps({k:{"controls":v["controls"],"usage":v["effective_method_target_counts"]} for k,v in evidence.items()},indent=2))


if __name__=="__main__":main()
