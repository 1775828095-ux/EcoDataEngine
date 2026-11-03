"""Official GOLEM task adapter: actual masked-series execution, not graph scores."""
from collections import Counter
from copy import deepcopy
from datetime import timedelta,datetime,timezone
import hashlib
import json
import random
from time import perf_counter
import traceback

import city_pm25_experiment as experiment
from city_pm25_experiment import OUT,ROOT,save,sha,np,pd
from ecodata.algorithms.temporal_reconstruction_dag import temporal_setup
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from run_taiwan_composition_search import (record,restore,typed_tree_growth,base_mutations,GraphRequirements,
    GPAlgorithmParameters,CrossoverTypesEnum,GraphGenerationParams,IdentityAdapter,has_no_cycle,
    has_no_self_cycled_nodes,Objective,EvoGraphOptimizer,ObjectiveEvaluate)
from golem.core.optimisers.graph import OptGraph


def key_for(graph,factory):return hashlib.sha256(factory.inspect(graph)["key"].encode()).hexdigest()


class TemporalEvaluator:
    def __init__(self,cases,config,folder):
        self.cases=cases;self.config=config;self.folder=folder
        registry,self.factory=temporal_setup();self.pipeline=UnifiedAlgorithmPipeline(registry)

    def execute(self,graph,key):
        clock=perf_counter();predictions=[];failed=[];traces=[]
        for case in self.cases:
            # Seal the entire final ten-day block, including other masks' targets.
            case=dict(case,masked=case["masked"].loc[:self.config["validation_end"]])
            idx=case["positions"]
            idx=idx[case["y"].index[idx]<=pd.Timestamp(self.config["validation_end"])]
            if not len(idx):continue
            try:values,trace=experiment.execute_graph(graph,case,idx,self.pipeline)
            except Exception as exc:
                values=np.full(len(idx),np.nan);trace=getattr(exc,"trace",[])
                failed.append(dict(case=case["case_id"],error=str(exc)))
            traces.append(dict(case=case["case_id"],trace=trace))
            predictions.append(pd.DataFrame(dict(case=case["case_id"],entity_id=case["entity_id"],
                time=case["y"].index[idx],truth=case["y"].iloc[idx].to_numpy(),prediction=values)))
        frame=pd.concat(predictions,ignore_index=True)
        folder=self.folder/key;folder.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(folder/"predictions.parquet",index=False)
        with (folder/"trace.jsonl").open("w",encoding="utf-8") as stream:
            for trace in traces:stream.write(json.dumps(trace,default=str,ensure_ascii=False)+"\n")
        result=experiment.metrics(frame.truth,frame.prediction,float("inf"))
        result.update(key=key,graph=record(graph),seconds=perf_counter()-clock,failures=failed,
            score=result["RMSE"] if result["coverage"]==1. else 1e12,
            quality="WARN_NATIVE_UNIT_AND_CLOCK" if result["coverage"]==1. else "BLOCK_INCOMPLETE_COVERAGE",
            masked_truth_not_in_input=True,test_targets_used=False)
        reread=pd.read_parquet(folder/"predictions.parquet")
        np.testing.assert_allclose(reread.prediction,frame.prediction,equal_nan=True)
        save(folder/"result.json",result);return result


class StopBudget(RuntimeError):pass


def search(mode,seed,task,cases,config):
    folder=OUT/"search"/task/f"{mode}_{seed}";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"summary.json").exists():return json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    random.seed(seed);np.random.seed(seed)
    evaluator=TemporalEvaluator(cases,config,folder/"executions");factory=evaluator.factory
    counts=Counter();records=[];proposals=[];cache={};begun=perf_counter()
    def limited():return len(records)>=config["search_budget"] or counts["proposals"]>=config["proposal_budget"] or perf_counter()-begun>config["search_wall_seconds"]
    def objective(graph):
        counts["objective_calls"]+=1
        if len(records)>=config["search_budget"]:counts["deferred_callbacks"]+=1;return 1e12
        row=dict(evaluation=len(records)+1,real_execution=False,cache_hit=False,graph=record(graph))
        try:
            status=factory.inspect(graph);key=key_for(graph,factory);row.update(key=key,gap=status["gap"])
            if not status["complete"]:row.update(status="INCOMPLETE",score=1e6+status["gap"])
            elif key in cache:row.update(cache[key],cache_hit=True);counts["cache_hits"]+=1
            else:
                result=evaluator.execute(graph,key);cache[key]=result
                row.update(result,real_execution=True);counts["new_executions"]+=1
            row.setdefault("status",row.get("quality","UNKNOWN"))
        except Exception as exc:row.update(status="CONTRACT_REJECT",score=1e12,error=str(exc),call_chain=traceback.format_exc())
        row["elapsed"]=perf_counter()-begun;records.append(row)
        with (folder/"events.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(row,default=str,ensure_ascii=False)+"\n")
        return row["score"]
    def legal(graph,*args,**kwargs):
        try:factory.inspect(graph);return True
        except Exception as exc:counts["posterior_rejections"]+=1;proposals.append(dict(posterior_error=str(exc)));return False
    operators=[typed_tree_growth,base_mutations.single_add_mutation,base_mutations.single_change_mutation,
        base_mutations.single_drop_mutation,base_mutations.single_edge_mutation]
    def propose(fn,graph,kw):
        if limited():raise StopBudget()
        counts["proposals"]+=1
        try:
            candidate=fn(deepcopy(graph),kw["requirements"],kw["graph_gen_params"],kw["parameters"])
            status=factory.inspect(candidate)
            proposals.append(dict(operator=fn.__name__,gap_before=factory.inspect(graph)["gap"],gap_after=status["gap"],key=key_for(candidate,factory)))
            return candidate,status
        except StopBudget:raise
        except Exception as exc:
            counts["proposal_rejections"]+=1;proposals.append(dict(operator=fn.__name__,error=str(exc)));return deepcopy(graph),None
    def wrap(fn):
        def mutate(graph,**kw):return propose(fn,graph,kw)[0]
        return mutate
    def guided(graph,**kw):
        options=[];weights=[];before=factory.inspect(graph)["gap"]
        for _ in range(3):
            if limited():break
            candidate,status=propose(random.choice(operators),graph,kw)
            if status and not (key_for(candidate,factory) in cache and cache[key_for(candidate,factory)]["score"]>=1e12):
                options.append(candidate);weights.append(float(np.exp(2*(before-status["gap"])-.1*status["algorithm_nodes"])))
        return random.choices(options,weights=weights)[0] if options else deepcopy(graph)
    error=None
    requirements=GraphRequirements(num_of_generations=30,timeout=timedelta(seconds=config["search_wall_seconds"]),
        max_depth=6,start_depth=6,n_jobs=1,show_progress=False,early_stopping_iterations=8)
    if mode=="R":
        objective(factory.initial())
        while not limited():counts["proposals"]+=1;objective(factory.random_graph(requirements))
    else:
        parameters=GPAlgorithmParameters(pop_size=4,max_pop_size=4,mutation_prob=1.,crossover_prob=0.,
            mutation_types=[guided] if mode=="C" else [wrap(fn) for fn in operators],
            crossover_types=[CrossoverTypesEnum.none],max_num_of_operator_attempts=3)
        generation=GraphGenerationParams(adapter=IdentityAdapter(),node_factory=factory,random_graph_factory=factory.random_graph,
            rules_for_constraint=[has_no_cycle,has_no_self_cycled_nodes]+([legal] if mode in {"B","C"} else []))
        obj=Objective({"validation_rmse":objective});optimizer=EvoGraphOptimizer(obj,[factory.initial()],requirements,generation,parameters)
        optimizer.stop_optimization.add_condition(limited,"Shared bounded temporal experiment")
        try:optimizer.optimise(ObjectiveEvaluate(obj))
        except StopBudget:counts["bounded_stop"]+=1
        except Exception as exc:error=dict(error=str(exc),call_chain=traceback.format_exc())
    good=[r for r in cache.values() if r["score"]<1e12]
    summary=dict(mode=mode,seed=seed,task=task,counts=dict(counts),records=records,proposals=proposals,
        factory_trace=factory.trace,seconds=perf_counter()-begun,optimizer_error=error,unique_valid_graphs=len(good),
        best=min(good,key=lambda x:x["score"]) if good else None,config_sha256=sha(OUT/"experiment_config.json"))
    save(folder/"summary.json",summary)
    print("SEARCH",task,mode,seed,len(records),len(good),summary["best"]["RMSE"] if good else None,flush=True)
    return summary


if __name__=="__main__":
    config=experiment.freeze_config();allcases=experiment.cases_for_pilot(config);results=[];enumerated=[]
    for task in config["search_tasks"]:
        cases=[c for c in allcases if c["task"]==task]
        for seed in config["search_seeds"]:
            for mode in "ABCR":results.append(search(mode,seed,task,cases,config))
        evaluator=TemporalEvaluator(cases,config,OUT/"enumeration"/task)
        for _,recipe in evaluator.factory.recipes("validated_field"):
            graph=OptGraph([evaluator.factory.instantiate(recipe)]);key=key_for(graph,evaluator.factory)
            result_path=evaluator.folder/key/"result.json"
            result=json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else evaluator.execute(graph,key)
            enumerated.append(dict(task=task,**result))
    save(OUT/"search_summary.json",results)
    save(OUT/"enumeration_summary.json",enumerated)
    selected=OUT/"selected_graphs_frozen.json"
    if not selected.exists():
        chosen=[dict(mode=r["mode"],seed=r["seed"],task=r["task"],key=r["best"]["key"],graph=r["best"]["graph"]) for r in results if r["best"]]
        save(selected,dict(frozen_at=datetime.now(timezone.utc).isoformat(),config_sha256=sha(OUT/"experiment_config.json"),
            selections=chosen,graphs={r["key"]:r["graph"] for r in chosen},test_targets_used=False,
            selection_rule="per-arm/seed validation minimum; enumeration reference never replaces a selected graph"))
    rows=[]
    for r in results:
        for row in r["records"]:rows.append(dict(mode=r["mode"],seed=r["seed"],task=r["task"],
            **{k:v for k,v in row.items() if not isinstance(v,(list,dict))}))
    pd.DataFrame(rows).to_csv(OUT/"search_runs.csv",index=False)
