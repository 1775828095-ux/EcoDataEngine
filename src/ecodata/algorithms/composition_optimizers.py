"""Official optional optimisers over the same screened categorical domain.

The callback owns all scientific execution and budget checks. No reference
labels, model-specific scoring or replacement ranking exists in this module.
"""
from dataclasses import dataclass
from importlib.metadata import version
import random
import numpy as np


def random_graph_search(search, request, selection_policy=None):
    """Shared best-parent random mutation loop; task engines own real execution."""
    empty_families = set()
    budget_blocked = False
    while not search.exhausted():
        current = search.best()['key']
        families = ['algorithm', 'parameters', 'structure', 'source'] + (['joint'] if request.get('joint_destroy_repair', True) else [])
        family = str(search.rng.choice(families))
        plan = search.destroy(current, search.rng) if family == 'joint' else None
        strategy = str(search.rng.choice(['random', 'portrait']))
        if selection_policy is not None and not selection_policy['portraits_enabled']:
            strategy = 'random'
        before_calls = len(search.account.events)
        search.propose(current, family, strategy, search.rng, destruction=plan)
        if len(search.account.events) > before_calls:
            search.numerical_rounds += 1
            empty_families.clear()
            budget_blocked = False
        else:
            empty_families.add(family)
            budget_blocked |= getattr(search,'proposal_budget_blocked',False)
        search.stopper.observe(search.numerical_rounds, search.best(), numerical_evaluations=search.novel_executions,
                               empty_coverage_complete=len(empty_families) == len(families))
        if budget_blocked and len(empty_families)==len(families):
            search.stopper.reason='RESOURCE_BUDGET_EXHAUSTED'
    return dict(backend='random', parent_policy='best', shared_driver='composition_optimizers.random_graph_search')


def search_graph(engine, initial, *, seed):
    """The existing ALNS control logic with an open-DAG state adapter.

    Engine owns portrait screening, real execution, immutable graphs and budgets.
    Neither validation values nor algorithm implementations enter ALNS operators.
    """
    from alns import ALNS
    from alns.select import AlphaUCB
    from alns.accept import HillClimbing

    @dataclass
    class State:
        key: str | None
        family: str = ''
        destruction: dict | None = None

        def objective(self):
            row = engine.evaluated.get(self.key)
            return row['metrics']['objective'] if row and row['status']=='ELIGIBLE_OUTPUT' and not row.get('dominated') else 1e100

    solver = ALNS(np.random.default_rng(seed))
    families=['algorithm','parameters','structure','source']
    if hasattr(engine,'destroy') and engine.request.get('joint_destroy_repair',True):families.append('joint')
    for family in families:
        def destroy(state, rng, family=family):
            parent=engine.best()['key'] if hasattr(engine,'best') else state.key
            plan=engine.destroy(parent,rng) if family=='joint' else None
            return State(parent,family,plan)
        solver.add_destroy_operator(destroy, name=family)
    for strategy in ('random', 'portrait'):
        def repair(state, rng, strategy=strategy):
            key = (engine.propose(state.key,state.family,strategy,rng,destruction=state.destruction)
                   if state.destruction is not None else engine.propose(state.key,state.family,strategy,rng))
            # An empty/cached neighbourhood is not a successful equal-fitness
            # child. Returning the parent rewarded no-op operators in AlphaUCB.
            return State(key)
        solver.add_repair_operator(repair, name=strategy)
    # Equal-fitness acceptance permits exploration but is not an improvement
    # signal. Do not train operator preference on identity-heavy graph edits.
    selector = AlphaUCB(scores=[5., 2., 0., 0.], alpha=.1, num_destroy=len(families), num_repair=2)
    def accept(rng,best,current,candidate):
        return engine.accept(current.key,candidate.key) if hasattr(engine,'accept') else HillClimbing()(rng,best,current,candidate)
    solver.iterate(State(initial), selector, accept,
                   lambda rng, best, current: engine.exhausted())
    return dict(backend='alns', package='alns', version=version('alns'),
                native='ALNS + AlphaUCB + HillClimbing; open-DAG adapter',
                operator_counts=selector._times.tolist(), operator_average_rewards=selector._avg_rewards.tolist(),
                reward_basis='Real search-validation objective; same best-parent policy as random; equal-error growth rejected; rewards [5,2,0,0]; no final test access')


def search(backend,engine,seeds,screened,evaluate,ranking,done,*,seed,proposals,population,generations):
    table={r["index"]:r for r in screened}
    limit=len(proposals)+generations*population
    def fitness(index):
        row=evaluate(int(index),"mutation")
        if not row or row['status']!='PASS':return float('inf')
        return round(row['search_validation']['RMSE'],9) if 'search_validation' in row else -round(row["matching"]["score_interval"][0],9)
    def record(parent,genome,operator):
        child=engine.encode(genome)
        edits=[]
        if parent is not None:
            for slot,a,b in zip(engine.slots,engine.decode(parent),genome):
                if a!=b:edits.append(dict(axis=slot,old_choice=a,new_choice=int(b)))
        proposals.append(dict(parent=parent,child=child,operator=operator,edits=edits,
            in_screened_domain=child in table))
        return child
    if backend=="random":
        rng=random.Random(seed);indices=list(table);rng.shuffle(indices)
        for index in indices:
            if done() or len(proposals)>=limit:break
            record(None,engine.decode(index),"random_legal_control");fitness(index)
        return dict(backend=backend)
    if backend=="pymoo":
        from pymoo.core.mixed import MixedVariableGA
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.variable import Choice
        from pymoo.core.population import Population
        from pymoo.optimize import minimize
        names=["g"+str(i) for i in range(len(engine.sizes))]
        class Problem(ElementwiseProblem):
            def __init__(self):super().__init__(vars={n:Choice(options=list(range(s))) for n,s in zip(names,engine.sizes)},n_obj=1)
            def _evaluate(self,x,out,*args,**kwargs):
                genome=tuple(int(x[n]) for n in names);index=engine.encode(genome)
                if done() or len(proposals)>=limit:out["F"]=1e6;return
                if index not in seeds:record(None,genome,"pymoo_mixed_variable_native_mating")
                out["F"]=fitness(index) if index in table else 1e6
        initial=[dict(zip(names,engine.decode(i))) for i in seeds]
        algorithm=MixedVariableGA(pop_size=max(2,len(initial)),sampling=Population.new(X=initial))
        minimize(Problem(),algorithm,termination=("n_gen",max(2,generations+1)),seed=seed,verbose=False)
        return dict(backend=backend,package="pymoo",version=version("pymoo"),native="MixedVariableGA; choice-aware crossover/mutation and duplicate elimination")
    if backend!="alns":raise ValueError("Unsupported native composition backend: "+backend)
    from alns import ALNS
    from alns.select import AlphaUCB
    from alns.accept import HillClimbing
    @dataclass
    class State:
        genome:tuple
        removed:tuple=()
        parent:int|None=None
        def objective(self):
            index=engine.encode(self.genome)
            return fitness(index) if index in table else 1e6
    rng=np.random.default_rng(seed);solver=ALNS(rng)
    mutable=[i for i,n in enumerate(engine.sizes) if n>1]
    if not mutable:return dict(backend=backend,reason="no mutable dimension",package="alns",version=version("alns"))
    for family in ("source","fusion","window","global"):
        def destroy(state,rng,family=family):
            if family=="source":groups=[i for i in mutable if engine.slots[i][2] not in {"fusion","post"}]
            elif family=="fusion":groups=[i for i in mutable if engine.slots[i][2] in {"fusion","post"}]
            elif family=="window":
                window=engine.slots[int(rng.choice(mutable))][:2]
                groups=[i for i in mutable if engine.slots[i][:2]==window]
            else:groups=mutable
            if not groups:groups=mutable
            amount=int(rng.integers(1,len(groups)+1));selected=tuple(int(i) for i in rng.choice(groups,size=amount,replace=False))
            return State(state.genome,selected,engine.encode(state.genome))
        solver.add_destroy_operator(destroy,name=family)
    for strategy in ("random","portrait_guided"):
        def repair(state,rng,strategy=strategy):
            genome=list(state.genome)
            for axis in state.removed:
                choices=[v for v in range(engine.sizes[axis]) if v!=genome[axis]]
                if strategy=="portrait_guided":
                    from ecodata.algorithms.composition_scoring import gain_priority
                    ranked=[]
                    for value in choices:
                        candidate=genome.copy();candidate[axis]=value;index=engine.encode(candidate)
                        if index in table:
                            row=table[index];ranked.append((row["score_interval"][0],*gain_priority(row),-row["scientific_steps"],value))
                    if ranked:
                        best=max(row[:-1] for row in ranked);choices=[row[-1] for row in ranked if row[:-1]==best]
                genome[axis]=int(rng.choice(choices))
            record(state.parent,genome,"alns_"+strategy)
            return State(tuple(genome))
        solver.add_repair_operator(repair,name=strategy)
    selector=AlphaUCB(scores=[5.,2.,1.,0.],alpha=.1,num_destroy=4,num_repair=2)
    def stop(rng,best,current):return done() or len(proposals)>=limit
    solver.iterate(State(engine.decode(seeds[0])),selector,HillClimbing(),stop)
    return dict(backend=backend,package="alns",version=version("alns"),native="ALNS + AlphaUCB + HillClimbing",
        operator_counts=selector._times.tolist(),operator_average_rewards=selector._avg_rewards.tolist(),
        reward_basis="explicit search-validation RMSE when provided, otherwise portrait objective; never independent test error")
