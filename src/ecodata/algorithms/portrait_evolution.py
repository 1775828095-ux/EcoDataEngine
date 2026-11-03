"""Official GOLEM multi-objective mutation over a frozen typed plan domain."""
from copy import deepcopy
from contextlib import redirect_stdout,redirect_stderr
from datetime import timedelta
import random
from time import perf_counter
import numpy as np


def search(candidates,factory,decision,options,folder,*,seed,wall_seconds=60):
    from ecodata.algorithms.numerical_search import (_SEARCH_LOCK,save,key_for,typed_tree_growth,EvoGraphOptimizer,
        GPAlgorithmParameters,GraphRequirements,GraphGenerationParams,IdentityAdapter,Objective,ObjectiveEvaluate,
        base_mutations,CrossoverTypesEnum)
    if set(options)-{"strategy","population","generations","proposal_limit","exact_check"} or options.get("strategy")!="genetic":
        raise ValueError("Unknown portrait search configuration")
    population=options.get("population",4);generations=options.get("generations",5);proposal_limit=options.get("proposal_limit",100)
    if any(type(v) is not int or not 1<=v<=maximum for v,maximum in [(population,16),(generations,50),(proposal_limit,2000)]):
        raise ValueError("Bounded genetic parameters required")
    if type(options.get("exact_check",True)) is not bool:raise ValueError("exact_check must be boolean")
    bykey={c["key"]:c for c in candidates}
    if not candidates:return dict(status="NO_FEASIBLE_CANDIDATES",visited=[],pareto=[],evolution_used=False)
    if not _SEARCH_LOCK.acquire(blocking=False):raise RuntimeError("Another GOLEM search is active")
    python_state=random.getstate();numpy_state=np.random.get_state();random.seed(seed);np.random.seed(seed)
    started=perf_counter();events=[];visited={};proposals=[];error=None
    include_cost=not any(t["definition"]["provider"]=="declared_cost" for t in candidates[0]["matching"]["terms"])
    def vector(c):return [*c["matching"]["objective_vector"],*([c["declared_cost"]] if include_cost else [])]
    try:
        def legal(graph,*args,**kwargs):
            try:
                factory.inspect(graph,complete=True);decision.require_graph(factory,graph)
                return key_for(graph,factory) in bykey
            except Exception:return False
        def evaluate(graph):
            key=key_for(graph,factory)
            if key not in bykey:raise ValueError("Mutation outside frozen legal short-plan domain")
            if key not in visited:
                visited[key]=bykey[key]
                events.append(dict(key=key,vector=bykey[key]["matching"]["objective_vector"],
                    score_interval=bykey[key]["matching"]["score_interval"],seconds=perf_counter()-started))
            c=bykey[key]
            return vector(c)
        operators=[typed_tree_growth,base_mutations.single_change_mutation,base_mutations.single_add_mutation,
                   base_mutations.single_drop_mutation,base_mutations.single_edge_mutation]
        def mutate(graph,**kw):
            parent=key_for(graph,factory);operator=random.choice(operators)
            if len(proposals)>=proposal_limit:return deepcopy(graph)
            try:
                child=operator(deepcopy(graph),kw["requirements"],kw["graph_gen_params"],kw["parameters"])
                if not legal(child):raise ValueError("Typed grammar, ownership, hard constraint or frozen-domain rejection")
                key=key_for(child,factory)
                proposals.append(dict(operator=operator.__name__,parent=parent,child=key,status="ACCEPTED" if key!=parent else "NO_CHANGE"))
                return child
            except Exception as exc:
                proposals.append(dict(operator=operator.__name__,parent=parent,status="REJECTED",reason=str(exc)))
                # Explicit immigrants preserve exploration; never masquerade as a valid edit.
                immigrant=random.choice(candidates)
                proposals[-1]["immigrant"]=immigrant["key"]
                return deepcopy(immigrant["graph"])
        size=min(population,len(candidates))
        initial=random.sample(candidates,size)
        requirements=GraphRequirements(num_of_generations=generations,max_depth=12,start_depth=12,n_jobs=1,
            timeout=timedelta(seconds=min(60,wall_seconds)),show_progress=False,early_stopping_iterations=max(3,generations))
        params=GPAlgorithmParameters(multi_objective=True,pop_size=size,max_pop_size=size,
            mutation_prob=1.,crossover_prob=0.,mutation_types=[mutate],crossover_types=[CrossoverTypesEnum.none],
            min_pop_size_with_elitism=1,max_num_of_operator_attempts=3)
        generation=GraphGenerationParams(adapter=IdentityAdapter(),node_factory=factory,
            random_graph_factory=lambda *a,**kw:deepcopy(random.choice(candidates)["graph"]),rules_for_constraint=[legal])
        metrics={"issue_"+str(i):(lambda graph,index=i:float(evaluate(graph)[index])) for i in range(len(vector(candidates[0])))}
        objective=Objective(metrics,is_multi_objective=True)
        optimizer=EvoGraphOptimizer(objective,[deepcopy(c["graph"]) for c in initial],requirements,generation,params)
        optimizer.stop_optimization.add_condition(lambda:len(proposals)>=proposal_limit,"Frozen proposal budget")
        with (folder/"portrait_optimizer.log").open("w",encoding="utf-8") as stream,redirect_stdout(stream),redirect_stderr(stream):
            optimizer.optimise(ObjectiveEvaluate(objective))
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state);_SEARCH_LOCK.release()
    def order(c):return (-c["matching"]["score_interval"][0],c["declared_cost"],len(c["steps"]),c["key"])
    genetic_best=min(visited.values(),key=order)["key"] if visited else None
    exact=options.get("exact_check",True)
    considered=candidates if exact else list(visited.values())
    vectors={c["key"]:np.asarray(vector(c)) for c in considered}
    pareto=[k for k,v in vectors.items() if not any(np.all(w<=v) and np.any(w<v) for j,w in vectors.items() if j!=k)]
    report=dict(status="ERROR" if error else "COMPLETED",evolution_used=True,backend="official GOLEM EvoGraphOptimizer multi-objective",
        initial_keys=[c["key"] for c in initial],visited=list(visited),events=events,proposals=proposals,
        genetic_best=genetic_best,selected=min(considered,key=order)["key"] if considered else None,
        pareto=pareto,optimizer_error=error,seconds=perf_counter()-started,exact_check=exact,
        certification="EXHAUSTIVE_SCORE_LOWER_BOUND_IN_FROZEN_DOMAIN" if exact else "HEURISTIC_SEARCH_ONLY",
        true_effect_global_optimum="NOT_PROVEN",scoring_candidates=len(candidates),genetic_evaluations=len(visited),
        independent_reference_used=False,scientific_executions_during_evolution=0,
        scope="All finite-domain portrait scores available for exact audit; no claimed speed advantage; Pareto objectives are suitability bounds")
    save(folder/"portrait_evolution.json",report)
    return report
