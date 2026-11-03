"""Common real numerical GOLEM orchestration. Never imports research tools.

GolemCapabilitySearch remains the separate legacy structural backend.
"""
from collections import Counter
from copy import deepcopy
from contextlib import redirect_stdout,redirect_stderr
from datetime import timedelta
import hashlib,json,random,sys,traceback
from pathlib import Path
from time import perf_counter,process_time
from threading import Lock
import numpy as np
# Existing isolated official GOLEM package, not a copied optimiser or install.
from importlib.util import find_spec
if find_spec("golem") is None:
    runtime=Path(__file__).resolve().parents[3]/".venv-golem/Lib/site-packages"
    if not (runtime/"golem").is_dir():
        raise RuntimeError("Official GOLEM runtime unavailable; configure the existing isolated environment")
    sys.path.append(str(runtime))
from ecodata.algorithms.golem_generation import typed_tree_growth
from ecodata.algorithms.golem_search import semantic_proposal_weights
from golem.core.optimisers.graph import OptGraph,OptNode
from golem.core.adapter import IdentityAdapter
from golem.core.dag.verification_rules import has_no_cycle,has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators import base_mutations
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.objective import Objective,ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

def record(graph):
    nodes=list(graph.nodes)
    return [dict(operation=n.name,parameters=dict(n.parameters),parents=[nodes.index(p) for p in n.nodes_from]) for n in nodes]


def restore(rows):
    nodes=[OptNode(dict(name=r["operation"],params=r["parameters"])) for r in rows]
    for n,r in zip(nodes,rows):n.nodes_from=[nodes[i] for i in r["parents"]]
    return OptGraph(nodes)


def key_for(graph,factory):return hashlib.sha256(factory.inspect(graph)["key"].encode()).hexdigest()


class StopBudget(RuntimeError):pass


_SEARCH_LOCK=Lock()


def run_numerical_search(mode,seed,e,config,folder,*,progress=None):
    """Synchronous resource guard: no scheduler, no duplicate concurrent RNG state."""
    if not _SEARCH_LOCK.acquire(blocking=False):raise RuntimeError("A numerical search is already running; retry only after it stops")
    try:return _run_numerical_search(mode,seed,e,config,folder,progress=progress)
    finally:_SEARCH_LOCK.release()


def _run_numerical_search(mode,seed,e,config,folder,*,progress=None):
    platform_start=perf_counter()
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"summary.json").exists():
        raise ValueError("Search directory already contains a run; use the saved result or a new request id")
    if mode not in {"A","B","C","C-gap","R","C-directed","C-local"}:raise ValueError("Unknown search strategy")
    for name in ("objective_limit","proposal_limit","callback_limit","wall_seconds"):
        if not 0 < config[name] <= {"objective_limit":200,"proposal_limit":2000,"callback_limit":5000,"wall_seconds":600}[name]:
            raise ValueError("Invalid bounded budget: "+name)
    random.seed(seed);np.random.seed(seed);f=e.factory
    if mode in {"C-directed","C-local"}:
        for name,default,upper in (("directed_candidates",4,16),("directed_attempts",6,64),("reverse_recipes",64,128)):
            config.setdefault(name,default)
            if not isinstance(config[name],int) or not 0<config[name]<=upper:raise ValueError("Invalid directed bound: "+name)
        config.setdefault("default_exploration",0.25)
        if not 0.1<=config["default_exploration"]<=0.9:raise ValueError("Explicit nonzero default exploration required")
        f.reverse_limit=config["reverse_recipes"];f._memo.clear()
        if mode=="C-local":
            f.local_recipe_seed=seed
            e.decision().bind_problem_ports(f,e.external)
            save(folder/"local_port_contracts.json",dict(version="local_problem_v1",resources=f.problem_resources,
                adapters=[dict(operation=key[0],mode=key[1],inputs={n:p.kind for n,p in a.inputs.items()},outputs={n:p.kind for n,p in a.outputs.items()},decision=a.decision_contract) for key,a in sorted(f.adapters.items())]))
    unified=True
    decision=e.decision()
    if decision:save(folder/"decision.json",decision.as_dict())
    platform_setup_seconds=getattr(e,"setup_seconds",0.)+perf_counter()-platform_start
    counts=Counter();rows=[];events=[];proposals=[];cache={};transitions=[];start=perf_counter();cpu=process_time();generation_seconds=0.;benefit_seconds=0.
    def limited(*,proposal_budget=True):
        return (len(rows)>=config["objective_limit"] or
            (proposal_budget and counts["proposals"]>=config["proposal_limit"]) or
            counts["callbacks"]>=config["callback_limit"] or perf_counter()-start>=config["wall_seconds"])
    def objective(graph):
        counts["callbacks"]+=1
        # Already generated candidates may still use the remaining evaluation
        # budget; reaching the proposal ceiling must not discard that batch.
        if limited(proposal_budget=False):
            counts["deferred_callbacks"]+=1
            if unified:events.append(dict(status="DEFERRED_NO_EXECUTION",elapsed=perf_counter()-start))
            return 1e12
        state=f.inspect(graph);key=key_for(graph,f)
        if key in cache:
            counts["graph_cache_hits"]+=1;events.append(dict(key=key,status="CACHE",elapsed=perf_counter()-start));return cache[key]["fitness"]
        row=dict(key=key,gap=state["gap"],graph=record(graph),evaluation=len(rows)+1,unique_execution=False)
        if decision:row["decision_feedback"]=decision.assess_graph(f,graph)
        if state["gap"]:
            row.update(status="INCOMPLETE",fitness=1e6+state["gap"],objective=None);counts["incomplete_feedback"]+=1
        else:
            result=e.execute(graph,key);attempted=result.get("numerical_execution_attempted",True)
            row.update(result,unique_execution=attempted)
            row["fitness"]=result["objective"] if result["status"]=="PASS" else 1e12
            counts["real_executions"]+=int(attempted);counts["quality_failures"]+=int(attempted and result["status"]!="PASS")
            counts["task_acceptance_failures"]+=int(not attempted)
        row.update(elapsed=perf_counter()-start);cache[key]=row;rows.append(row);events.append(dict(key=key,status=row["status"],evaluation=len(rows),elapsed=row["elapsed"]))
        if progress:progress(dict(evaluation=len(rows),status=row["status"],counts=dict(counts)))
        return row["fitness"]
    def legal(graph,*args,**kwargs):
        try:
            status=f.inspect(graph)
            rejected=decision.assess_graph(f,graph)["hard_reject"]
            if rejected:
                raise ValueError("Posterior coverage contract: actual visible one-sided gaps lack supported output")
            return True
        except Exception as exc:counts["posterior_rejections"]+=1;proposals.append(dict(status="POSTERIOR_REJECT",error=str(exc)));return False
    operators=[typed_tree_growth,base_mutations.single_add_mutation,base_mutations.single_change_mutation,base_mutations.single_drop_mutation,base_mutations.single_edge_mutation]
    def propose(fn,graph,kw):
        if limited():raise StopBudget()
        counts["proposals"]+=1
        try:
            candidate=fn(deepcopy(graph),kw["requirements"],kw["graph_gen_params"],kw["parameters"])
            status=f.inspect(candidate)
            proposals.append(dict(operator=fn.__name__,gap_before=f.inspect(graph)["gap"],gap_after=status["gap"],key=key_for(candidate,f),status="PROPOSED"))
            if unified:proposals[-1]["graph"]=record(candidate)
            return candidate,status
        except StopBudget:raise
        except Exception as exc:
            counts["proposal_rejections"]+=1;proposals.append(dict(operator=fn.__name__,status="REJECT",error=str(exc),
                call_chain=traceback.format_exc() if unified else None));return deepcopy(graph),None
    def wrapper(fn):
        def mutate(graph,**kw):return propose(fn,graph,kw)[0]
        return mutate
    def guided(graph,**kw):
        options=[]
        for _ in range(3):
            if limited():break
            candidate,status=propose(random.choice(operators),graph,kw)
            if status:
                options.append(candidate)
                proposals[-1].update(guide_mode=mode)
        if decision and options:
            weights,trace=semantic_proposal_weights(decision,f,graph,options,soft=mode=="C")
            indices=random.choices(range(len(options)),weights=weights)
            trace.update(status="SHARED_DECISION",guide_mode=mode,chosen_index=indices[0],
                keys=[key_for(c,f) for c in options],elapsed=perf_counter()-start)
            proposals.append(trace)
            return options[indices[0]]
        return deepcopy(graph)
    def problem_directed(graph,**kw):
        nonlocal generation_seconds,benefit_seconds
        if limited():raise StopBudget()
        parent_key=key_for(graph,f)
        if random.random()<config["default_exploration"]:
            counts["default_exploration_calls"]+=1
            candidate,status=propose(random.choice(operators),graph,kw)
            if status:transitions.append(dict(parent=parent_key,child=key_for(candidate,f),action=dict(type="default_mutation",attribution="composite_or_unknown")))
            return candidate
        begin=perf_counter()
        remaining=min(config["directed_attempts"],config["proposal_limit"]-counts["proposals"])
        generator=f.problem_candidates if mode=="C-local" else f.legacy_problem_candidates
        candidates,attempts=generator(decision,graph,limit=config["directed_candidates"],attempt_limit=remaining,recipe_limit=config["reverse_recipes"])
        generation_seconds+=perf_counter()-begin
        counts["directed_generation_calls"]+=1
        for attempt in attempts:
            counts["proposals"]+=1;counts["directed_proposals"]+=1
            if attempt["status"]=="REJECT":counts["proposal_rejections"]+=1
            if attempt["status"]=="DUPLICATE":counts["directed_duplicate_proposals"]+=1
            proposals.append(dict(operator="problem_directed",parent=parent_key,**attempt))
        if not candidates:
            counts["directed_empty"]+=1
            return propose(random.choice(operators),graph,kw)[0] if not limited() else deepcopy(graph)
        options,actions=zip(*candidates)
        weights,trace=semantic_proposal_weights(decision,f,graph,options,soft=True)
        if mode=='C-local':
            from ecodata.algorithms.capability_scoring import empirical_proposal_weights
            begin=perf_counter()
            weights,trace=empirical_proposal_weights(decision,f,options,weights,trace)
            benefit_seconds+=perf_counter()-begin
        chosen=random.choices(range(len(options)),weights=weights)[0]
        candidate=options[chosen];child_key=key_for(candidate,f)
        trace.update(status="PROBLEM_DIRECTED_CHOICE",operator="problem_directed",parent=parent_key,child=child_key,
                     chosen_index=chosen,actions=actions,keys=[key_for(g,f) for g in options],elapsed=perf_counter()-start,
                     parent_graph=record(graph),chosen_graph=record(candidate),problems=decision.requirements.problems)
        proposals.append(trace)
        transitions.append(dict(parent=parent_key,child=child_key,action=actions[chosen]))
        counts["directed_choices"]+=1
        return candidate
    requirements=GraphRequirements(num_of_generations=500,timeout=timedelta(seconds=config["wall_seconds"]),
        max_depth=9,start_depth=9,n_jobs=1,show_progress=False,early_stopping_iterations=20)
    error=None
    try:
        if mode=="R":
            while not limited():
                counts["proposals"]+=1;objective(f.random_graph(requirements))
        else:
            # Local editing can legitimately expose one feasible child at a
            # time. A serial population avoids filling an artificial minimum
            # with duplicates before any of those children is evaluated.
            population_size=1 if mode=="C-local" else min(config["directed_candidates"],max(1,len(f.recipes("validated_field")))) if mode=="C-directed" else 6
            parameters=GPAlgorithmParameters(pop_size=population_size,max_pop_size=population_size,mutation_prob=1.,crossover_prob=0.,
                mutation_types=[problem_directed] if mode in {"C-directed","C-local"} else [guided] if mode in {"C","C-gap"} else [wrapper(fn) for fn in operators],
                crossover_types=[CrossoverTypesEnum.none],max_num_of_operator_attempts=3)
            generation=GraphGenerationParams(adapter=IdentityAdapter(),node_factory=f,random_graph_factory=f.random_graph,
                rules_for_constraint=[has_no_cycle,has_no_self_cycled_nodes]+([legal] if mode in {"B","C","C-gap","C-directed","C-local"} else []))
            obj=Objective({"real_validation_error":objective});optimizer=EvoGraphOptimizer(obj,[f.initial()],requirements,generation,parameters)
            optimizer.stop_optimization.add_condition(limited,"Shared frozen mixed-gap budget")
            with (folder/"optimizer.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):optimizer.optimise(ObjectiveEvaluate(obj))
    except StopBudget:counts["bounded_stop"]+=1
    except Exception as exc:error=dict(error=str(exc),call_chain=traceback.format_exc())
    good=[r for r in rows if r["status"]=="PASS"]
    evaluation_fingerprint=hashlib.sha256(json.dumps(dict(protocol=getattr(e,"evaluation_protocol",{}),public_inputs={k:v.fingerprint() for k,v in getattr(e,"external",{}).items()}),sort_keys=True,default=str).encode()).hexdigest()
    provenance={}
    for t in transitions:provenance.setdefault(t["child"],set()).add((t["parent"],json.dumps(t["action"],sort_keys=True,default=str)))
    for transition in transitions:
        parent,child=(cache.get(transition[key]) for key in ("parent","child"))
        valid=transition["parent"]!=transition["child"] and all(r and r["status"]=="PASS" and r.get("unique_execution") and r.get("numerical_execution_attempted",True) for r in (parent,child))
        transition.update(reward=float(parent["objective"]-child["objective"]) if valid else None,
            reward_status="REAL_PAIRED_GRAPH_DIFFERENCE" if valid else "NOT_COMPUTABLE: parent/child not both newly executed quality-passing distinct graphs",
            evaluation_fingerprint=evaluation_fingerprint,metric=getattr(e,"evaluation_protocol",{}),
            attribution="composite_or_unresolved" if transition["action"]["type"]=="default_mutation" or transition["action"].get("composite") or len(provenance[transition["child"]])>1 else "identified_graph_edit",
            single_action_learning_label=bool(valid and transition["action"]["type"]!="default_mutation" and not transition["action"].get("composite") and len(provenance[transition["child"]])==1),
            state=dict(features=decision.semantics.as_dict(),requirements=decision.requirements.as_dict(),parent_graph=parent["graph"] if parent else None,
                       parent_feedback={k:parent.get(k) for k in ("status","objective","decision_feedback")} if parent else None))
    save(folder/"operation_transitions.json",dict(version="problem_actions_v1",q_model_trained=False,records=transitions))
    compact=[{k:v for k,v in r.items() if k not in {"node_trace","group_RMSE"}} for r in rows]
    summary=dict(mode=mode,seed=seed,counts=dict(counts),scored=len(rows),rows=compact,events=events,proposals=proposals,
        seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,node_hits=e.node_hits,node_new=e.node_new,kernel_hits=e.kernel_hits,
        best=min(good,key=lambda x:x["objective"]) if good else None,unique_valid=len(good),optimizer_error=error,
        platform_setup_seconds=platform_setup_seconds,platform_total_seconds=getattr(e,"setup_seconds",0.)+perf_counter()-platform_start,
        semantic_chain="platform_v1" if unified else "historical_local",decision_service_seconds=decision.seconds if decision else None,
        unique_behaviors=len({r["prediction_fingerprint"] for r in good}),config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    summary.update(candidate_generation_seconds=generation_seconds,problems=list(decision.requirements.problems) if decision else [],
        optional_benefit_inference_seconds=benefit_seconds,optional_benefit_model=getattr(decision,'benefit_manifest',None),
        operation_transitions_file=str(folder/"operation_transitions.json"),q_model_trained=False,
        directed_config={k:config[k] for k in ("directed_candidates","directed_attempts","reverse_recipes","default_exploration") if k in config})
    if mode=="C-local":summary["population_policy"]="bounded serial official GOLEM population=1; not an efficacy comparison with other strategies"
    summary.update(service="NumericalSearchService/v1",initial_graph=record(f.initial()),stop_reason=("budget" if limited() else "optimizer_finished"),candidate_domain=sorted(f.adapters),evaluation_protocol=getattr(e,"evaluation_protocol",{}))
    if summary["best"]:
        save(folder/"selected_graph.json",summary["best"]["graph"])
    save(folder/"factory_trace.json",f.trace);save(folder/"summary.json",summary)
    print("SEARCH",mode,seed,"scored",len(rows),"actual",counts["real_executions"],"best",summary["best"]["objective"] if good else None,flush=True)
    return summary


