"""Bounded, shared GOLEM graph grammar derived from executable named ports.

No optimiser is implemented here. Official mutations use these domain factories.
Data leaves and boundary operations are not extra algorithm candidates.
"""
from __future__ import annotations

from itertools import product
import json
import random
import hashlib
from copy import deepcopy

import numpy as np

from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.core.dag.graph_utils import distance_to_primary_level, distance_to_root_level

from ecodata.algorithms.dag import DAGError


class GenerationError(DAGError):
    pass


class TypedGolemFactory(DefaultOptNodeFactory):
    PARAMS = {
        "station.pykrige_universal": {"variogram_model": ("linear", "spherical", "exponential")},
        "fusion.residual": {"interpolation": ("linear", "nearest")},
    }
    SOURCES = {"source.geos": "geos", "source.era5": "era5", "source.stations": "stations", "source.history": "history"}

    def __init__(self, adapters, frozen_ids, *, max_nodes=24, max_depth=12):
        super().__init__(frozen_ids)
        self.adapters = {key: adapter for key, adapter in adapters.items()
            if key[0] in frozen_ids or key[0].startswith(("source.", "prepare.", "quality."))}
        self.max_nodes, self.max_depth = max_nodes, max_depth
        self.trace = []
        self.preferred = {}
        self._memo = {}
        self.producers = {}
        for key, adapter in self.adapters.items():
            for output, port in adapter.outputs.items():
                self.producers.setdefault(port.kind, []).append((key, output))
        self.rejected_candidates = sorted(set(frozen_ids)-{key[0] for key in self.adapters})
        self.trace.append({"event": "catalog", "frozen_count": len(frozen_ids),
            "task_ineligible": self.rejected_candidates,
            "reason": "No executable named-port adapter for this task; same exclusion in A/B/C"})

    def adapter(self, node):
        key = (node.name, node.parameters.get("mode", "run"))
        if key not in self.adapters:
            raise GenerationError(f"No task contract for {key}")
        return self.adapters[key]

    def parameters(self, key):
        domains = self.PARAMS.get(key[0], {})
        names = list(domains)
        combinations = product(*(domains[n] for n in names))
        for values in combinations:
            params = dict(zip(names, values))
            if key[1] != "run":
                params["mode"] = key[1]
            if key[0] in self.SOURCES:
                params["source"] = self.SOURCES[key[0]]
            yield tuple(sorted(params.items()))

    def recipes(self, kind, stack=()):
        """Enumerate the finite contract grammar, not hand-authored full graphs."""
        if kind in stack:
            raise GenerationError(f"Cyclic capability request: {stack} -> {kind}")
        if hasattr(self,"local_recipe_seed"):
            return self._fair_recipes(kind,stack)
        bound=getattr(self,"reverse_limit",None)
        if bound and len(stack)>=self.max_depth:return []
        if not stack and kind in self._memo:
            return self._memo[kind]
        result = []
        for key, output in self.producers.get(kind, []):
            adapter = self.adapters[key]
            names = list(adapter.inputs)
            children = [self.recipes(adapter.inputs[name].kind, stack+(kind,)) for name in names]
            for params in self.parameters(key):
                for choices in product(*children):
                    edges = tuple((port, child_output, child) for port, (child_output, child) in zip(names, choices))
                    result.append((output, (key[0], params, edges)))
                    if bound and len(result)>=bound:break
                if bound and len(result)>=bound:break
            if bound and len(result)>=bound:break
        if not result:
            error = f"No producer satisfies output kind {kind}; requested by {stack}"
            self.trace.append({"event": "reject", "request": kind, "error": error})
            raise GenerationError(error)
        if not stack:
            self._memo[kind] = result
        return result

    def _fair_recipes(self,kind,stack):
        """Lazy round-robin producers, canonical seeded order, bounded products.

        No full Cartesian space is materialised. Subclasses still filter the
        returned recipes through their existing state-ownership contracts.
        """
        bound=getattr(self,"reverse_limit",64)
        if len(stack)>=self.max_depth:return []
        focus=getattr(self,"local_recipe_focus",None)
        memo_key=(kind,self.local_recipe_seed,bound,focus)
        if not stack and memo_key in self._memo:return self._memo[memo_key]
        def priority(value):return hashlib.sha256(json.dumps([self.local_recipe_seed,value],sort_keys=True).encode()).hexdigest()
        producers=sorted(self.producers.get(kind,()),key=lambda item:priority(item))
        def possible(key,seen):
            a=self.adapters[key]
            result=set(a.decision_contract.get("capabilities",()))|set(getattr(self,"local_capabilities",{}).get(key[0],()))
            for port in a.inputs.values():
                if port.kind in seen:continue
                for child,_ in self.producers.get(port.kind,()):result.update(possible(child,seen|{port.kind}))
            return result
        if focus and focus[0]==kind:
            needed=set(focus[1])
            producers=[item for item in producers if possible(item[0],{kind})&needed]
        streams=[];rejected=[]
        for key,output in producers:
            adapter=self.adapters[key];names=sorted(adapter.inputs)
            try:children=[self.recipes(adapter.inputs[name].kind,stack+(kind,)) for name in names]
            except GenerationError as exc:rejected.append(dict(producer=key,reason=str(exc)));continue
            if any(not choices for choices in children):continue
            def stream(key=key,output=output,names=names,children=children):
                for params in sorted(self.parameters(key)):
                    for choices in product(*children):
                        yield output,(key[0],params,tuple((port,child_output,child) for port,(child_output,child) in zip(names,choices)))
            streams.append(iter(stream()))
        result=[];seen=set()
        while streams and len(result)<bound:
            active=[]
            for stream in streams:
                if len(result)>=bound:active.append(stream);continue
                try:item=next(stream)
                except StopIteration:continue
                active.append(stream)
                signature=json.dumps(item,sort_keys=True,default=str)
                if signature not in seen:seen.add(signature);result.append(item)
            streams=active
        self.trace.append(dict(event="local_recipe_enumeration",kind=kind,seed=self.local_recipe_seed,
            strategy="capability/typed-port pruning then canonical seeded producer round-robin; lazy bounded child products",focus=focus,limit=bound,count=len(result),
            possibly_unexpanded_streams=len(streams),rejected=rejected,complete_space_claim=False))
        if not result:raise GenerationError("No bounded legal producer for "+kind)
        if not stack:self._memo[memo_key]=result
        return result

    @staticmethod
    def height(recipe):
        return 1+max((TypedGolemFactory.height(child) for _, _, child in recipe[2]), default=0)

    def instantiate(self, recipe, *, omit_root_ports=(), depth=None):
        interned = {}
        depth = self.max_depth if depth is None else max(1, depth)

        def build(item, remaining, root=False):
            operation, params_tuple, edges = item
            identity = (item, min(remaining, self.height(item)), tuple(omit_root_ports) if root else ())
            if identity in interned:
                self.trace.append({"event": "reuse", "operation": operation})
                return interned[identity]
            node = OptNode({"name": operation, "params": dict(params_tuple)})
            interned[identity] = node
            self.trace.append({"event": "create", "operation": operation, "parameters": dict(params_tuple), "depth_remaining": remaining})
            bindings = {}
            for port, output, child in edges:
                if remaining <= 1 or (root and port in omit_root_ports):
                    self.trace.append({"event": "missing_port", "operation": operation, "port": port,
                        "reason": "bounded partial graph"})
                    continue
                parent = build(child, remaining-1)
                if parent not in node.nodes_from:
                    node.nodes_from.append(parent)
                bindings[port] = [node.nodes_from.index(parent), output]
                self.trace.append({"event": "connect", "parent": parent.name, "output": output,
                    "child": operation, "input": port})
            node.parameters["_port_bindings"] = bindings
            return node

        return build(recipe, depth, True)

    def build(self, kind, *, depth=None):
        choices = self.recipes(kind)
        preference = self.preferred.get(kind)
        if preference:
            choices = [(output, recipe) for output, recipe in choices
                if (recipe[0], dict(recipe[1]).get("mode", "run")) == preference]
            if not choices:
                raise GenerationError(f"Requested producer {preference} cannot supply {kind}")
        output, recipe = random.choice(choices)
        node = self.instantiate(recipe, depth=depth)
        return node, output

    def initial(self):
        # A real quality node with a missing field input, not a placeholder result.
        _, recipe = self.recipes("validated_field")[0]
        graph = OptGraph([self.instantiate(recipe, omit_root_ports=("field",))])
        self.inspect(graph)
        return graph

    def inspect(self, graph, *, complete=False):
        # Official separate-parent mutation indexes the new parent, assuming a
        # leaf. Our typed parent may already carry a real dependency closure.
        # Materialise only those existing edges; never invent repair operations.
        indexed, active, walked = set(map(id, graph.nodes)), set(), set()
        def index_dependencies(node):
            if id(node) in active:
                raise GenerationError("Cycle while indexing typed dependencies")
            if id(node) in walked:
                return
            active.add(id(node))
            for parent in node.nodes_from:
                index_dependencies(parent)
                if id(parent) not in indexed:
                    graph.add_node(parent)
                    indexed.update(map(id, graph.nodes))
                    self.trace.append({"event": "index_existing_dependency", "parent": parent.name, "child": node.name})
            active.remove(id(node))
            walked.add(id(node))
        for node in list(graph.nodes):
            index_dependencies(node)
        raw = list(graph.nodes)
        if not raw or len(raw) > self.max_nodes or graph.depth > self.max_depth:
            raise GenerationError("Graph size/depth outside shared bounds")
        missing, signatures, visiting, visited = [], {}, set(), set()
        consumed = set()

        def visit(node):
            if id(node) in visiting:
                raise GenerationError("Cycle in typed graph")
            if id(node) in visited:
                return signatures[id(node)]
            visiting.add(id(node))
            adapter = self.adapter(node)
            for parent in node.nodes_from:
                visit(parent)
            bindings, signature_inputs, used = {}, [], set()
            for name, contract in adapter.inputs.items():
                matches = [(i, output, p) for i, p in enumerate(node.nodes_from)
                    for output, port in self.adapter(p).outputs.items() if port.kind == contract.kind]
                if not matches:
                    missing.append({"node": node.name, "input": name, "kind": contract.kind})
                    signature_inputs.append((name, None))
                elif len(matches) > 1:
                    raise GenerationError(f"Ambiguous {node.name}.{name} expects {contract.kind}")
                else:
                    index, output, parent = matches[0]
                    bindings[name] = [index, output]
                    used.add(id(parent))
                    signature_inputs.append((name, output, signatures[id(parent)]))
            if used != {id(p) for p in node.nodes_from}:
                raise GenerationError(f"Unexpected parent/contract conflict at {node.name}: expected "
                    f"{[(n, p.kind) for n, p in adapter.inputs.items()]}, received "
                    f"{[(p.name, [c.kind for c in self.adapter(p).outputs.values()]) for p in node.nodes_from]}")
            # Binding normalisation only: never add/delete/repair processing nodes.
            if node.parameters.get("_port_bindings") != bindings:
                self.trace.append({"event": "bind_existing_edges", "operation": node.name, "bindings": bindings})
                node.parameters["_port_bindings"] = bindings
            params = {k: v for k, v in node.parameters.items() if k != "_port_bindings"}
            signatures[id(node)] = (node.name, tuple(sorted(params.items())), tuple(signature_inputs))
            consumed.update(used)
            visiting.remove(id(node))
            visited.add(id(node))
            return signatures[id(node)]

        for n in raw:
            visit(n)
        sinks = [n for n in raw if id(n) not in consumed]
        if len(sinks) != 1:
            raise GenerationError("Task requires one connected output graph")
        sink = sinks[0]
        if not any(p.kind == "validated_field" for p in self.adapter(sink).outputs.values()):
            missing.append({"node": sink.name, "input": "task_output", "kind": "validated_field"})
        if complete and missing:
            raise GenerationError(f"Incomplete capability gap: {missing}")
        return {"missing": missing, "gap": len(missing), "complete": not missing,
            "key": json.dumps(signatures[id(sink)], sort_keys=True),
            "algorithm_nodes": sum(n.name in self.available_nodes for n in raw)}

    def get_parent_node(self, node, **kwargs):
        if not kwargs.get("is_primary", False):
            self.trace.append({"event": "reject_request", "caller": "get_parent_node", "reason": "No compatible wrap-all-inputs operator"})
            return None
        adapter = self.adapter(node)
        supplied = {port.kind for p in node.nodes_from for port in self.adapter(p).outputs.values()}
        missing = [port.kind for port in adapter.inputs.values() if port.kind not in supplied]
        if not missing:
            return None
        return self.build(random.choice(missing))[0]

    def exchange_node(self, node):
        old = self.adapter(node)
        signature = tuple((name, port.kind) for name, port in old.inputs.items())
        outputs = tuple((name, port.kind) for name, port in old.outputs.items())
        choices = []
        current = {k: v for k, v in node.parameters.items() if k != "_port_bindings"}
        for key, adapter in self.adapters.items():
            if key[0] not in self.available_nodes:
                continue
            if tuple((n, p.kind) for n, p in adapter.inputs.items()) != signature or tuple((n, p.kind) for n, p in adapter.outputs.items()) != outputs:
                continue
            for parameters in self.parameters(key):
                if key[0] != node.name or dict(parameters) != current:
                    choices.append((key[0], dict(parameters)))
        if not choices:
            return None
        operation, params = random.choice(choices)
        self.trace.append({"event": "exchange", "old": node.name, "new": operation, "parameters": params})
        # Official update_node preserves the old parents; no extra parents are grafted.
        return OptNode({"name": operation, "params": params})

    def get_node(self, **kwargs):
        if kwargs.get("is_primary"):
            operation = random.choice(list(self.SOURCES))
            return OptNode({"name": operation, "params": {"source": self.SOURCES[operation]}})
        self.trace.append({"event": "reject_request", "caller": "get_node", "reason": "Non-leaf insertion lacks a requested input-port kind"})
        return None

    def get_final_node(self):
        self.trace.append({"event": "reject_request", "caller": "get_final_node", "reason": "Task already fixes its output boundary; no untyped extra final node"})
        return None

    def random_graph(self, requirements, max_depth=None, *, output_kind="validated_field"):
        """GOLEM hook: one bounded typed construction, never its 1001 retry loop."""
        depth = min(self.max_depth, max_depth or requirements.max_depth)
        self.trace.append({"event": "random_graph_request", "depth": depth, "attempt_limit": 1,
                           "output_kind": output_kind})
        node, _ = self.build(output_kind, depth=depth)
        result = OptGraph([node])
        status = self.inspect(result)
        self.trace.append({"event": "random_graph_result", "gap": status["gap"]})
        return result

    def complete_keys(self):
        result = set()
        for _, recipe in self.recipes("validated_field"):
            graph = OptGraph([self.instantiate(recipe)])
            result.add(self.inspect(graph, complete=True)["key"])
        return result

    def problem_candidates(self,decision,graph,*,limit=8,attempt_limit=24,recipe_limit=32):
        """One rule-bound problem per edit; preserve unrelated dependency programs."""
        self.inspect(graph)
        if not hasattr(self,"local_recipe_seed"):self.local_recipe_seed=0;self._memo.clear()
        before=decision.assess_graph(self,graph);before_key=self.inspect(graph)["key"]
        resources=getattr(self,"problem_resources",{})
        source_ids={"source."+key:r["source_id"] for key,r in resources.items()}
        def recipe(node):
            return (node.name,tuple(sorted((k,v) for k,v in node.parameters.items() if k!="_port_bindings")),
                tuple((name,out,recipe(node.nodes_from[index])) for name,(index,out) in sorted(node.parameters.get("_port_bindings",{}).items())))
        def identity(r):return json.dumps(r,sort_keys=True,default=str)
        def sources(r):return ({source_ids[r[0]]} if r[0] in source_ids else set()).union(*(sources(child) for _,_,child in r[2]))
        def capabilities(r):
            adapter=self.adapters[r[0],dict(r[1]).get("mode","run")];declaration=adapter.decision_contract
            direct=set(declaration.get("capabilities",()))|set(decision.matches.get(r[0],{}).get("capabilities",()))
            return direct.union(*(capabilities(child) for _,_,child in r[2]))
        def contains(r,old):return identity(r)==identity(old) or any(contains(c,old) for _,_,c in r[2])
        def science(r):
            from collections import Counter
            adapter=self.adapters[r[0],dict(r[1]).get("mode","run")]
            algorithm=adapter.decision_contract.get("algorithm_id") or (r[0] if r[0] in self.available_nodes else None)
            return (Counter({(algorithm,dict(r[1]).get("mode","run")):1}) if algorithm else Counter())+sum((science(c) for _,_,c in r[2]),Counter())
        def instantiate(r,existing):
            key=identity(r)
            if key in existing:return existing[key]
            node=OptNode(dict(name=r[0],params=dict(r[1])));existing[key]=node
            for _,_,parent in r[2]:
                p=instantiate(parent,existing)
                if p not in node.nodes_from:node.nodes_from.append(p)
            return node
        def paths(r):
            return [dict(operation=r[0],source_ids=sorted(sources(r)),signature=identity(r))]+[p for _,_,c in r[2] for p in paths(c)]
        def edge_set(recipes):
            return {json.dumps(dict(child=r[0],child_program=identity(r),input=name,parent=c[0],parent_program=identity(c),output=out),sort_keys=True)
                for r in recipes for name,out,c in r[2]}
        def output_paths(r,tail=()):
            if r[0] in source_ids:return [dict(source_id=source_ids[r[0]],path=[dict(operation=r[0],output="value")]+list(tail))]
            result=[]
            for name,out,c in r[2]:
                result.extend(output_paths(c,(dict(operation=r[0],input=name,parent_output=out),)+tail))
                if len(result)>=32:return result[:32]
            return result
        old_recipes=[recipe(n) for n in graph.nodes]
        attempts=[];result=[];seen={before_key};streams=[]
        def priority(p):return hashlib.sha256(json.dumps([self.local_recipe_seed,p["id"]]).encode()).hexdigest()
        ordered=sorted(decision.requirements.problems,key=priority)
        turn=getattr(self,"local_problem_turn",0);self.local_problem_turn=turn+1
        problems=[]
        for scoped in (True,False):
            group=[p for p in ordered if (p.get("binding",{}).get("scope")=="source_branch")==scoped]
            if group:
                offset=turn%len(group);problems.extend(group[offset:]+group[:offset])
        for problem in problems:
            binding=problem.get("binding")
            if not binding:continue
            wanted=set(problem["capabilities"]);affected=set(binding["source_ids"])
            if not wanted:continue
            locations=[]
            for index,node in enumerate(graph.nodes):
                adapter=self.adapter(node);bound=node.parameters.get("_port_bindings",{})
                for name,port in sorted(adapter.inputs.items()):
                    if binding["scope"]=="source_branch" and port.kind==binding["kind"]:locations.append((index,name,port.kind))
                if binding["scope"]=="task_output" and node in graph.root_nodes():
                    for output,port in adapter.outputs.items():
                        if port.kind==binding["kind"]:locations.append((index,None,port.kind))
            # The source branch need not exist yet: follow the missing output
            # closure. This is explicitly a composite dependency construction.
            if not locations and before["port_gap"]:
                for index,node in enumerate(graph.nodes):
                    for name,port in sorted(self.adapter(node).inputs.items()):
                        if name not in node.parameters.get("_port_bindings",{}):locations.append((index,name,port.kind))
            for index,name,kind in locations:
                self.local_recipe_focus=(kind,tuple(sorted(wanted)))
                try:choices=self.recipes(kind)
                except GenerationError:continue
                eligible=[]
                for _,r in choices:
                    rs=sources(r)
                    if not wanted.intersection(capabilities(r)):continue
                    if not affected.issubset(rs):continue
                    # A local port must not borrow capability from another source.
                    if binding["scope"]=="source_branch" and kind==binding["kind"] and rs!=affected:continue
                    eligible.append(r)
                bound=graph.nodes[index].parameters.get("_port_bindings",{}).get(name) if name is not None else None
                current=recipe(graph.nodes[index].nodes_from[bound[0]]) if bound else recipe(graph.nodes[index]) if name is None else None
                if current is not None:
                    def programs(r):return ({identity(r)} if r[2] else set()).union(*(programs(c) for _,_,c in r[2]))
                    retained=programs(current)
                    buckets={}
                    for r in eligible:buckets.setdefault((identity(r)==identity(current),-len(programs(r)&retained)),[]).append(r)
                    eligible=[]
                    for score in sorted(buckets):
                        group=buckets[score];offset=turn%len(group)
                        eligible.extend(group[offset:]+group[:offset])
                # Seeded recipes are already producer-balanced; no prefix of
                # registration order is taken before capability/port pruning.
                if eligible and current is None:
                    offset=turn%len(eligible);eligible=eligible[offset:]+eligible[:offset]
                streams.append(iter([(problem,index,name,r) for r in eligible[:recipe_limit]]))
        while streams and len(attempts)<attempt_limit and len(result)<limit:
            active=[]
            for stream in streams:
                if len(attempts)>=attempt_limit or len(result)>=limit:active.append(stream);continue
                try:problem,index,name,r=next(stream)
                except StopIteration:continue
                active.append(stream);binding=problem["binding"];affected=set(binding["source_ids"])
                candidate=deepcopy(graph);target=candidate.nodes[index];existing={identity(recipe(n)):n for n in candidate.nodes}
                original_roots=list(candidate.root_nodes())
                old_binding=target.parameters.get("_port_bindings",{}).get(name) if name is not None else None
                old=recipe(target.nodes_from[old_binding[0]]) if old_binding else recipe(target) if name is None else None
                edit="connect" if old is None or before["port_gap"] else "insert" if contains(r,old) else "simplify" if contains(old,r) else "replace"
                if edit=="replace" and old is not None:
                    # Output validators can change kind while the exact
                    # scientific upstream program is retained under a new stage.
                    retained=any(contains(r,c) for _,_,c in old[2]) or any(contains(old,c) for _,_,c in r[2])
                    if retained and science(old)<science(r):edit="insert"
                    elif retained and science(r)<science(old):edit="simplify"
                action=dict(type=edit,problem_ids=[problem["id"]],source_ids=sorted(affected),target_operation=target.name,target_port=name or binding["kind"],
                    expected_capabilities=problem["capabilities"],reason=problem["evidence"],binding=binding,attribution="single_scoped_problem")
                if edit=="simplify":
                    action.update(problem_ids=[],expected_capabilities=[],goal=dict(type="remove_optional_processing",source_ids=sorted(affected),
                        acceptance="all mandatory output and local port requirements still checked"),attribution="simplification_goal_not_a_data_defect")
                try:
                    replacement=instantiate(r,existing)
                    if name is None:
                        roots=[replacement if n is target else n for n in original_roots]
                        for n in candidate.nodes:n.nodes_from=[replacement if p is target else p for p in n.nodes_from]
                    else:
                        if old_binding:target.nodes_from[old_binding[0]]=replacement
                        elif replacement not in target.nodes_from:target.nodes_from.append(replacement)
                        roots=original_roots
                    candidate=OptGraph(roots);status=self.inspect(candidate);assessment=decision.assess_graph(self,candidate)
                    if assessment["hard_reject"]:raise GenerationError("Final task path rejects edit: "+str(assessment["missing_capabilities"]))
                    if status["key"] in seen:
                        attempts.append(dict(status="DUPLICATE",action=action,reason="canonical graph already proposed"));continue
                    new_recipes=[recipe(n) for n in candidate.nodes];new_signatures={identity(n) for n in new_recipes}
                    protected={identity(n) for n in old_recipes if sources(n) and not sources(n)&affected}
                    if not protected<=new_signatures:raise GenerationError("Local edit would change unrelated source ancestry")
                    seen.add(status["key"])
                    old_signatures={identity(n) for n in old_recipes}
                    changed_sources=set().union(*(sources(n) for n in new_recipes if identity(n) not in old_signatures))
                    action.update(parent_graph=before_key,child_graph=status["key"],graph_before_gap=before["gap"],graph_after_gap=assessment["gap"],
                        algorithm=r[0],parameters=dict(r[1]),output_capabilities=assessment["output_capabilities"],
                        added_nodes=[p for n in new_recipes if identity(n) not in old_signatures for p in paths(n)[:1]],
                        removed_nodes=[p for n in old_recipes if identity(n) not in new_signatures for p in paths(n)[:1]],
                        connections=[dict(operation=n.name,bindings=n.parameters.get("_port_bindings",{})) for n in candidate.nodes],
                        affected_output_paths=[p for n in new_recipes if identity(n) not in old_signatures for p in paths(n)[:1]],
                        shared_or_multibranch=len(changed_sources)>1,actual_dependency_sources=sorted(changed_sources),
                        protected_ancestry=sorted(protected),composite=len(new_signatures-old_signatures)>1)
                    action.update(added_edges=[json.loads(e) for e in sorted(edge_set(new_recipes)-edge_set(old_recipes))],
                        removed_edges=[json.loads(e) for e in sorted(edge_set(old_recipes)-edge_set(new_recipes))],
                        dependency_paths=[p for root in candidate.root_nodes() for p in output_paths(recipe(root)) if p["source_id"] in affected],
                        dependency_path_limit=32)
                    if changed_sources-affected and edit!="simplify":action["attribution"]="scoped_problem_with_shared_or_new_dependency_closure"
                    attempts.append(dict(status="CANDIDATE",action=action));result.append((candidate,action))
                except Exception as exc:attempts.append(dict(status="REJECT",action=action,error=str(exc)))
            streams=active
        self.trace.append(dict(event="local_problem_generation",seed=self.local_recipe_seed,attempts=attempts,candidates=len(result),
            remaining_streams=len(streams),limit=limit,attempt_limit=attempt_limit,complete_space_claim=False,
            candidate_order="retain actual existing upstream programs; canonical seeded round-robin ties, rotating bounded streams; no accuracy preference"))
        return result,attempts

    def legacy_problem_candidates(self, decision, graph, *, limit=8, attempt_limit=24, recipe_limit=32):
        """Target actual missing ports/output needs, before any random mutation.

        Uses this factory's existing reverse recipes (including subclass state
        filters). No algorithm-name rules, reference values or accuracy model.
        """
        self.inspect(graph)
        before=decision.assess_graph(self,graph)
        problems=decision.requirements.problems
        wanted=set(before["missing_capabilities"]+before["unknown_capabilities"])
        wanted.update(c for p in problems for c in p["capabilities"])
        before_key=self.inspect(graph)["key"]
        attempts=[];groups={k:[] for k in ("connect","insert","replace","simplify")};seen={before_key}
        def from_recipe(recipe,existing):
            # Intern exact existing dependency programs. Do not substitute a
            # merely same-typed fit state from an unrelated branch.
            identity=recipe_signature(recipe)
            if identity in existing:return existing[identity]
            operation,params,edges=recipe
            node=OptNode(dict(name=operation,params=dict(params)))
            existing[identity]=node
            for _,_,child in edges:
                parent=from_recipe(child,existing)
                if parent not in node.nodes_from:node.nodes_from.append(parent)
            return node
        def recipe_signature(recipe):
            return (recipe[0],recipe[1],tuple(sorted((name,out,recipe_signature(child)) for name,out,child in recipe[2])))
        def normalized_recipe(node):
            return (node.name,tuple(sorted((k,v) for k,v in node.parameters.items() if k!="_port_bindings")),
                tuple((name,out,normalized_recipe(node.nodes_from[index])) for name,(index,out) in sorted(node.parameters.get("_port_bindings",{}).items())))
        def contains(recipe,needle):
            return recipe==needle or any(contains(child,needle) for _,_,child in recipe[2])
        def caps(recipe):
            key=(recipe[0],dict(recipe[1]).get("mode","run"));a=self.adapters[key];d=a.decision_contract
            result=set(d.get("capabilities",()))|set(decision.matches.get(recipe[0],{}).get("capabilities",()))
            for _,_,child in recipe[2]:result.update(caps(child))
            return result  # proposal preselection only; acceptance uses real paths
        recipe_cache={}
        def recipes(kind):
            if kind not in recipe_cache:
                # Existing bounded task grammar, not numerical enumeration.
                choices=self.recipes(kind)
                ranked=sorted(choices,key=lambda item:-len(caps(item[1])&wanted))
                recipe_cache[kind]=ranked[:recipe_limit]
            return recipe_cache[kind]
        def emit(kind,index,port,recipe=None,parent_index=None):
            if len(attempts)>=attempt_limit:return
            candidate=deepcopy(graph);target=candidate.nodes[index]
            existing={normalized_recipe(n):n for n in candidate.nodes}
            action=dict(type=kind,target_node=index,target_operation=target.name,target_port=port,
                problem_ids=[p["id"] for p in problems if set(p["capabilities"])&wanted],
                expected_capabilities=sorted(wanted),reason="observable problem + requested output capability + actual typed dependency")
            try:
                if kind=="connect":
                    parent=from_recipe(recipe,existing)
                    if parent not in target.nodes_from:target.nodes_from.append(parent)
                    action.update(algorithm=recipe[0],parameters=dict(recipe[1]))
                else:
                    replacement=candidate.nodes[parent_index] if kind=="simplify" and parent_index is not None else from_recipe(recipe,existing)
                    if kind=="simplify" and parent_index is not None:action["algorithm"]=replacement.name
                    else:action.update(algorithm=recipe[0],parameters=dict(recipe[1]))
                    roots=[replacement if n is target else n for n in candidate.root_nodes()]
                    for child in candidate.nodes:child.nodes_from=[replacement if p is target else p for p in child.nodes_from]
                    candidate=OptGraph(roots)
                status=self.inspect(candidate)
                assessment=decision.assess_graph(self,candidate)
                if assessment["hard_reject"]:raise GenerationError("Known task-contract violation: "+str(assessment["missing_capabilities"]))
                if status["key"] in seen:
                    attempts.append(dict(status="DUPLICATE",action=action,reason="same graph fingerprint; not a new operation"));return
                seen.add(status["key"])
                action.update(parent_graph=before_key,child_graph=status["key"],graph_before_gap=before["gap"],graph_after_gap=assessment["gap"],
                              output_capabilities=assessment["output_capabilities"],connections=[dict(node=n.name,bindings=n.parameters.get("_port_bindings",{})) for n in candidate.nodes])
                from collections import Counter
                old_nodes=Counter((n.name,n.parameters.get("mode","run")) for n in graph.nodes if n.name in self.available_nodes)
                new_nodes=Counter((n.name,n.parameters.get("mode","run")) for n in candidate.nodes if n.name in self.available_nodes)
                action.update(added_algorithm_nodes=dict((str(k),v) for k,v in (new_nodes-old_nodes).items()),removed_algorithm_nodes=dict((str(k),v) for k,v in (old_nodes-new_nodes).items()),
                              composite=max(sum((new_nodes-old_nodes).values()),sum((old_nodes-new_nodes).values()))>1)
                attempts.append(dict(status="CANDIDATE",action=action));groups[kind].append((candidate,action))
            except Exception as exc:attempts.append(dict(status="REJECT",action=action,error=str(exc)))

        # Round-robin proposals ensure an early producer list cannot starve
        # insertion/simplification. Each proposal is inspected before use.
        requests={k:[] for k in groups}
        def operations(recipe):
            from collections import Counter
            result=Counter({(recipe[0],recipe[1]):1}) if recipe[0] in self.available_nodes else Counter()
            for _,_,child in recipe[2]:result.update(operations(child))
            return result
        def ancestors(recipe):
            return [child for _,_,child in recipe[2]]+[n for _,_,child in recipe[2] for n in ancestors(child)]
        for index,node in enumerate(graph.nodes):
            adapter=self.adapter(node);bound=node.parameters.get("_port_bindings",{})
            for name,port in adapter.inputs.items():
                if name not in bound:
                    for _,recipe in recipes(port.kind):requests["connect"].append((index,name,recipe,None))
            for output,port in adapter.outputs.items():
                for _,recipe in recipes(port.kind):
                    if (recipe[0],dict(recipe[1]).get("mode","run"))==(node.name,node.parameters.get("mode","run")):continue
                    # An inserted processing stage must actually contain the
                    # current upstream program, not return to raw inputs.
                    old=normalized_recipe(node)
                    old_ops,new_ops=operations(old),operations(recipe)
                    retained=any(contains(recipe,child) for child in ancestors(old))
                    kind="insert" if contains(recipe,old) or (retained and old_ops<new_ops) else "simplify" if retained and new_ops<old_ops else "replace"
                    if kind!="simplify" and not (caps(recipe)&wanted):continue
                    requests[kind].append((index,output,recipe,None))
                for parent_index,parent in enumerate(graph.nodes):
                    if parent is node or parent not in node.nodes_from:continue
                    if any(p.kind==port.kind for p in self.adapter(parent).outputs.values()):
                        requests["simplify"].append((index,output,None,parent_index))
        while any(requests.values()) and len(attempts)<attempt_limit and sum(map(len,groups.values()))<limit:
            for kind in groups:
                if requests[kind]:emit(kind,*requests[kind].pop(0))
                if len(attempts)>=attempt_limit or sum(map(len,groups.values()))>=limit:break
        result=[]
        while any(groups.values()) and len(result)<limit:
            for values in groups.values():
                if values and len(result)<limit:result.append(values.pop(0))
        self.trace.append(dict(event="problem_directed_generation",before=before_key,problem_ids=[p["id"] for p in problems],
            candidate_count=len(result),attempts=attempts,limits=dict(candidates=limit,attempts=attempt_limit,recipes=recipe_limit)))
        return result,attempts


def typed_tree_growth(graph, requirements, graph_gen_params, parameters, local_growth=True):
    """EcoData port context for GOLEM's subtree-growth operation, shared by A/B/C.

    Retain upstream node order/depth/leaf choices. Its factory
    hook has no target argument, so supply that missing context here, not by
    changing GOLEM or silently repairing the graph after replacement.
    """
    factory = graph_gen_params.node_factory
    factory.inspect(graph)
    indices = np.arange(len(graph.nodes))
    random.shuffle(indices)
    for index in indices:
        target = graph.nodes[index]
        if local_growth:
            depth = distance_to_primary_level(target)
            primary = not target.nodes_from or (target != graph.root_node and random.randint(0, 1))
        else:
            depth = requirements.max_depth-distance_to_root_level(graph, target)
            primary = distance_to_root_level(graph, target) >= requirements.max_depth and random.randint(0, 1)
        demanded = set()
        for child in graph.nodes:
            for name, (parent_index, output) in child.parameters.get("_port_bindings", {}).items():
                if child.nodes_from[parent_index] is target:
                    demanded.add(factory.adapter(child).inputs[name].kind)
        if not demanded and target is graph.root_node:
            demanded.add("validated_field")
        if len(demanded) != 1:
            raise GenerationError(f"Ambiguous subtree output contract for {target.name}: {sorted(demanded)}")
        kind = demanded.pop()
        factory.trace.append({"event": "subtree_request", "target": target.name, "output_kind": kind,
                              "primary": bool(primary), "depth": depth})
        if primary:
            leaves = [(output, recipe) for output, recipe in factory.recipes(kind) if not recipe[2]]
            if not leaves:
                factory.trace.append({"event": "reject_request", "caller": "typed_tree_growth",
                                      "reason": f"No data leaf supplies {kind}"})
                continue
            replacement = factory.instantiate(random.choice(leaves)[1])
        else:
            replacement = factory.random_graph(requirements, max(1, depth), output_kind=kind).root_node
        # Atomic contract rejection; original graph is unchanged on a bad edge.
        candidate = deepcopy(graph)
        old = candidate.nodes[index]
        roots = [replacement if root is old else root for root in candidate.root_nodes()]
        # Upstream update_subtree deletes every ancestor of the old subtree,
        # including data shared with another DAG branch. Redirect consumers and
        # rebuild only the reachable index; retain those other branches' edges.
        for child in candidate.nodes:
            child.nodes_from = [replacement if parent is old else parent for parent in child.nodes_from]
        candidate = OptGraph(roots)
        factory.inspect(candidate)
        return candidate
    return graph
