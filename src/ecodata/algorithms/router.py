from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from collections import OrderedDict
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np

from ecodata.algorithms.contracts import AlgorithmContext, Compatibility
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.scale import TargetScaleAdvisor, spatial_scale_factor


@dataclass(frozen=True)
class AlgorithmPlan:
    id: str
    steps: tuple[str, ...]
    score: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    selected: AlgorithmPlan
    evaluations: tuple[dict, ...]
    basis: str


class AlgorithmSelector:
    _decision_cache=OrderedDict()
    def __init__(self, registry: AlgorithmRegistry):
        self.registry = registry

    def decide(self, context: AlgorithmContext, target: dict, *, semantics=None, adapter_compatibility=None,use_cache=False):
        """Shared production/research decision boundary. Never receives an evaluator.

        Numeric precision models (e.g. the development-trained raster router)
        remain separate consumers; this service does not predict unknown RMSE.
        """
        from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor, RequirementMapper
        from ecodata.algorithms.capabilities import capability_catalog
        clock = perf_counter()
        semantics = semantics or SemanticFeatureExtractor().extract(context.sources[0].dataset, context.variable)
        cache_key=None
        if use_cache:
            from ecodata.algorithms.dag import Artifact
            from dataclasses import asdict
            rules=Path(__file__).with_name("semantic_rules.json").read_bytes()
            public=dict(sources=[s.dataset for s in context.sources],auxiliaries=context.auxiliaries,observations=context.observations,
                variable=context.variable,semantic_type=context.semantic_type,target_lat=context.target_lat,target_lon=context.target_lon,
                parameters=context.parameters,constraints=context.model_constraints,input_quality=context.input_quality,validation_role=context.validation_role)
            # Independent validation_reference is deliberately not a decision input.
            cache_key=hashlib.sha256((Artifact("decision_inputs",public).fingerprint()+json.dumps(target,sort_keys=True,default=str)+
                json.dumps(semantics.as_dict(),sort_keys=True,default=str)+json.dumps([s.as_dict() for s in self.registry.specs()],sort_keys=True,default=str)+
                json.dumps({k:asdict(v) for k,v in (adapter_compatibility or {}).items()},sort_keys=True)+hashlib.sha256(rules).hexdigest()).encode()).hexdigest()
            if cache_key in self._decision_cache:
                result=deepcopy(self._decision_cache[cache_key]);result.seconds=perf_counter()-clock
                result.cache_info=dict(hit=True,key=cache_key,scope="input+task+resources+rules+live registry; bounded64")
                return result
        requirements = RequirementMapper().map(semantics, target)
        known={c for r in capability_catalog(self.registry) for c in r.capabilities}
        if (set(target.get('required_capabilities',()))|set(target.get('optional_capabilities',())))-known:raise ValueError('Unknown requested capability ID')
        screen = self.screen(context)
        # Trusted registered task modes may expose point/state ports rather
        # than the plugin's gridded convenience API. They must actually validate
        # those inputs, not infer compatibility from an algorithm name.
        if adapter_compatibility:
            for identifier,check in adapter_compatibility.items():
                self.registry.get(identifier)
                screen[identifier]=check
        matches = {}
        for record in capability_catalog(self.registry):
            check = screen[record.algorithm_id]
            contract = dict(record.output_contract)
            reasons = list(check.reasons)
            if "past_only" in requirements.hard_constraints and contract.get("uses_future"):
                reasons.append("past_only task forbids future retained observations")
            if "no_semantic_conflict" in requirements.hard_constraints:
                reasons.append("Conflicting variable semantics require confirmation")
            model_preprocessing=(self.registry.get(record.algorithm_id).spec.category=="model_preprocessing"
                and target.get("operation")=="model_preprocessing" and context.parameters.get("model_preprocessing") is True)
            matches[record.algorithm_id] = dict(eligible=check.eligible and (record.executable or model_preprocessing) and not reasons,
                reasons=reasons, capabilities=record.capabilities, input_contract=record.input_contract,
                output_contract=contract, matched=sorted(set(record.capabilities)&set(requirements.required)),
                cost=record.compute_cost, evidence=record.evidence)
        result=SemanticDecision(semantics, requirements, matches, perf_counter()-clock)
        result.cache_info=dict(hit=False,key=cache_key)
        if cache_key:
            self._decision_cache[cache_key]=deepcopy(result)
            if len(self._decision_cache)>64:self._decision_cache.popitem(last=False)
        return result

    def screen(self, context: AlgorithmContext) -> dict[str, Compatibility]:
        if context.input_quality and context.input_quality.get("status") == "BLOCK":
            reasons = tuple(item["message"] for item in context.input_quality.get("checks", []) if item.get("status") == "BLOCK")
            return {spec.id: Compatibility(False, reasons or ("原始数据质量检查阻止算法选择",)) for spec in self.registry.specs()}
        results = {}
        for spec in self.registry.specs():
            try:
                results[spec.id] = self.registry.get(spec.id).compatibility(context)
            except Exception as exc:
                results[spec.id] = Compatibility(False, (f"Compatibility error {type(exc).__name__}: {exc}",))
        return results

    def recommend(self, context: AlgorithmContext) -> tuple[AlgorithmPlan, ...]:
        if context.input_quality and context.input_quality.get("status") == "BLOCK":
            return ()
        decision = TargetScaleAdvisor().decide(context)
        scale = spatial_scale_factor(context)
        plans = []
        for spec in self.registry.specs():
            result = self.registry.get(spec.id).compatibility(context)
            if not result.eligible or not (spec.scale_range[0] <= scale <= spec.scale_range[1]):
                continue
            score = {"low": 4.0, "medium": 2.0, "high": 0.5, "gpu": 0.0}.get(spec.compute, 0.0)
            reasons = [f"输入语义 {context.semantic_type} 与插件声明相容"]
            if spec.id == decision.recommended_method:
                score += 5
                reasons.append(decision.reason)
            if len(context.sources) > 1 and spec.category == "multi_source_fusion":
                score += 4
                reasons.append("存在多源数据")
            if context.observations is not None and spec.requires_observations:
                score += 3
                reasons.append("存在训练观测约束")
            if spec.requires_training:
                score -= 2
            if context.input_quality and context.input_quality.get("status") == "WARN":
                reasons.append("输入质量存在警告；结果必须经过第二道质量检查")
                if spec.id == "fusion.quality_aware":
                    score += 2
                    reasons.append("存在可靠观测时优先考虑质量感知校正")
            plans.append(AlgorithmPlan(spec.id, (spec.id,), score, tuple(reasons), decision.warnings + result.warnings))
        # Generic composition: align the background first, then residual-fuse observations.
        if context.target_lat is not None and context.target_lon is not None and len(context.sources) >= 2:
            regrid = decision.recommended_method
            residual = "fusion.residual"
            if all(self.registry.get(step).compatibility(context).eligible for step in (regrid, residual)):
                plans.append(AlgorithmPlan(f"{regrid}+{residual}", (regrid, residual), 10.0,
                    ("先统一尺度，再进行同变量观测残差融合",), decision.warnings))
        return tuple(sorted(plans, key=lambda item: (-item.score, item.id)))

    def select(self, context: AlgorithmContext, plans: tuple[AlgorithmPlan, ...] | None = None) -> SelectionResult:
        """Select on explicit validation data, otherwise retain the conservative heuristic order."""
        plans = plans or self.recommend(context)
        if not plans:
            raise ValueError("没有满足输入语义、依赖和物理约束的算法")
        if context.validation_reference is None:
            return SelectionResult(plans[0], (), "没有独立高分辨率真值；按适用性、物理约束和计算成本推荐，不声称精度最优")
        if context.validation_role not in {"validation", "independent_validation"}:
            raise ValueError("算法选择只能使用 validation，最终 test 必须封存")
        from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
        evaluations = []
        for plan in plans:
            try:
                result = UnifiedAlgorithmPipeline(self.registry).run(context, plan)
                metrics = result.metrics
                evaluations.append({"plan": plan.id, "status": "PASS", "RMSE": metrics.get("RMSE"),
                                    "MAE": metrics.get("MAE"), "bias": metrics.get("bias"),
                                    "runtime_seconds": result.report.get("runtime_seconds")})
            except Exception as exc:
                evaluations.append({"plan": plan.id, "status": "FAIL", "reason": str(exc)})
        valid = [item for item in evaluations if item.get("status") == "PASS" and item.get("RMSE") is not None]
        if not valid:
            raise ValueError("候选算法均未通过独立验证和质量门")
        best = min(valid, key=lambda item: (item["RMSE"], item.get("runtime_seconds") or float("inf")))
        return SelectionResult(next(plan for plan in plans if plan.id == best["plan"]), tuple(evaluations),
                               "使用明确 validation 数据比较 RMSE；物理质量门失败的候选已排除，最终 test 未参与")

    def execute_with_fallback(self, context: AlgorithmContext, plans: tuple[AlgorithmPlan, ...] | None = None,
                              *, max_attempts: int = 3):
        """Try each compatible plan at most once; never retry indefinitely."""
        from dataclasses import replace
        from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
        candidates = (plans or self.recommend(context))[:max(1, max_attempts)]
        attempts = []
        for plan in candidates:
            try:
                result = UnifiedAlgorithmPipeline(self.registry).run(context, plan)
                attempts.append({"plan": plan.id, "status": result.report["status"]})
                report = dict(result.report)
                report["fallback_attempts"] = attempts
                return replace(result, report=report)
            except Exception as exc:
                attempts.append({"plan": plan.id, "status": "BLOCK", "reason": str(exc)})
        raise ValueError(f"候选算法均未通过双层质量门；尝试记录：{attempts}")


@dataclass
class SemanticDecision:
    semantics: object
    requirements: object
    matches: dict
    seconds: float

    def as_dict(self):
        from ecodata.algorithms.capability_scoring import optional_capability_matches
        return dict(service="AlgorithmSelector.decide/v1", semantics=self.semantics.as_dict(),
            requirements=self.requirements.as_dict(),matches=self.matches,seconds=self.seconds,
            hidden_targets_received=False,accuracy_preference="Optional empirical evidence only within explicitly loaded development model support",
            capability_matching=optional_capability_matches(self),feature_declarations=getattr(self,'feature_declarations',[]),
            optional_scoring=getattr(self,'benefit_manifest',dict(status='INSUFFICIENT_EVIDENCE',estimated_relative_benefit=None)))

    def bind_problem_ports(self, factory, external):
        """Bind shared rule results to declared data identities and typed ports.

        Task adapters declare resources, never a ranking or preferred algorithm.
        Evaluation references are not accepted by this interface.
        """
        from dataclasses import replace
        from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor,RequirementMapper
        from ecodata.algorithms.dag import Artifact
        import xarray as xr
        resources={}
        for key,artifact in sorted(external.items()):
            declaration=getattr(factory,"resource_contracts",{}).get(key,{})
            value=artifact.value
            dataset=getattr(value,"dataset",value)
            variable=declaration.get("variable",self.semantics.variable)
            identity=declaration.get("source_id","role:"+key+":"+variable)
            scientific=dataset
            if isinstance(dataset,xr.Dataset):
                scientific=dataset.copy(deep=False)
                scientific.attrs={k:v for k,v in dataset.attrs.items() if k in {"crs","temporal_semantics","spatial_semantics"}}
                for name in scientific.variables:
                    scientific[name].attrs={k:v for k,v in dataset[name].attrs.items() if k in {"units","data_kind","physical_variable","physical_definition","temporal_semantics","spatial_semantics","bounds","valid_min","valid_max","_FillValue","scale_factor","add_offset"}}
            resources[key]=dict(source_id=identity,role=declaration.get("role",key),variable=variable,
                input_port=dict(operation="source."+key,output="value",kind=artifact.kind),
                fingerprint=Artifact(artifact.kind,scientific).fingerprint(),declaration=declaration)
        factory.problem_resources=resources
        factory.local_capabilities={identifier:record.get("capabilities",()) for identifier,record in self.matches.items()}
        # The task-level rule's provenance is the set of actual source ports,
        # not the first file's filename or the current node list order.
        primary=[r for r in resources.values() if r["role"] in {"primary","background","observations","batch"}]
        primary=primary or list(resources.values())[:1]
        problems=[]
        for original in self.requirements.problems:
            p=deepcopy(original)
            relevant=list(resources.values()) if p["rule"] in {"multi_source","quality_aware","covariate_prediction"} else primary
            ids=sorted(r["source_id"] for r in relevant)
            p.update(id="problem_"+hashlib.sha256(json.dumps([p["rule"],p["variable"],ids],sort_keys=True).encode()).hexdigest()[:16],
                source_ids=ids,source_ports=[r["input_port"] for r in relevant],
                source_fingerprints={r["source_id"]:r["fingerprint"] for r in relevant},
                binding=dict(kind="validated_field",scope="task_output",source_ids=ids),
                port=dict(kind="validated_field",direction="output",source_ids=ids),
                dependency_path=dict(status="DEFERRED_TO_ACTUAL_GRAPH",from_ports=[r["input_port"] for r in relevant],to_kind="validated_field"))
            p["measure"]=dict(task_rule_facts=p["measure"]["facts"],scope="declared output requirement; per-source measurements belong to source-scoped problems")
            problems.append(p)
        for key,resource in resources.items():
            declaration=resource["declaration"]
            if "required_grid_kind" not in declaration:continue
            value=external[key].value;dataset=getattr(value,"dataset",value)
            semantics=SemanticFeatureExtractor().extract(dataset,resource["variable"])
            target=dict(source_grid_mismatch=declaration["grid_mismatch"],source_id=resource["source_id"],
                port=dict(kind=declaration["required_grid_kind"],direction="input",source_id=resource["source_id"]))
            for p in RequirementMapper().map(semantics,target).problems:
                if p["rule"]!="source_grid_mismatch":continue
                p.update(source_ids=[resource["source_id"]],source_ports=[resource["input_port"]],
                    source_fingerprints={resource["source_id"]:resource["fingerprint"]},
                    binding=dict(kind=declaration["required_grid_kind"],scope="source_branch",source_ids=[resource["source_id"]]),
                    dependency_path=dict(status="DEFERRED_TO_ACTUAL_GRAPH",from_ports=[resource["input_port"]],to_kind=declaration["required_grid_kind"]))
                problems.append(p)
        self.requirements=replace(self.requirements,problems=tuple(sorted(problems,key=lambda p:p["id"])))
        return self

    def assess_graph(self, factory, graph):
        """Evaluate declared semantics on the selected output's ancestor ports.

        Composite contracts describe which input supplies each output. A state
        in another branch cannot satisfy a predictor's required state port.
        Unknown contracts remain UNKNOWN, never an invented hard violation.
        """
        status = factory.inspect(graph)
        rows = self.semantics.observed.value.get("support_conditions",[])
        length = np.array([r.get("gap_record_count",r["length"]) for r in rows])
        size = len(rows); n=max(1,size)
        left = np.array([r["left"] for r in rows],bool) if size else np.zeros(n,bool)
        right = np.array([r["right"] for r in rows],bool) if size else np.zeros(n,bool)
        counts = np.array([r["count"] for r in rows],float) if size else np.ones(n)
        # Per-condition capability proofs: 1 present, 0 definitely absent,
        # -1 unknown. Select/fallback propagate only their effective paths.
        names=set(self.requirements.required)|{"complete_temporal_coverage","boundary_temporal_recovery","fitted_state"}
        for match in self.matches.values():names.update(match.get("capabilities",()))
        for adapter in factory.adapters.values():names.update(adapter.decision_contract.get("capabilities",()))
        names=sorted(names); columns={name:i for i,name in enumerate(names)}
        unknown=[]; ports=[]; memo={}; incompatible=[]
        def support(name):
            result={"both":left&right,"either":left|right,"left":left,"training":np.ones(n,bool),"none":np.zeros(n,bool)}.get(name)
            return np.full(n,-1,dtype=np.int8) if result is None else result.astype(np.int8)
        def advertised(capabilities):
            result=np.zeros((n,len(names)),dtype=np.int8)
            for name in capabilities:result[:,columns[name]]=1
            return result
        def visit(node, output):
            key=(id(node),output)
            if key in memo:return memo[key]
            adapter=factory.adapter(node); declaration=adapter.decision_contract
            bound={name:(node.nodes_from[i],out) for name,(i,out) in node.parameters.get("_port_bindings",{}).items()}
            deps={name:visit(parent,out) for name,(parent,out) in bound.items()}
            kind=declaration.get("operation")
            caps=np.zeros((n,len(names)),dtype=np.int8);coverage=np.zeros(n,dtype=np.int8);evidence=coverage.copy()
            if kind=="source":
                caps=advertised(declaration.get("capabilities",()))
            elif kind in {"compute","fit"}:
                # General numerical task ports: all declared dependencies must
                # be on this output's ancestry, not elsewhere in the graph.
                match=self.matches.get(node.name) if declaration.get("registered",True) else None
                capabilities=(declaration.get("capabilities",()) if kind=="fit" or not declaration.get("registered",True)
                              else (match or {}).get("capabilities",()))
                caps=advertised(capabilities)
                inherited=declaration.get("input")
                if inherited in deps:
                    caps=np.where(caps==1,1,deps[inherited][0])
                if not set(adapter.inputs)<=set(deps):caps.fill(0)
                state_port=declaration.get("state_port")
                if state_port:
                    fitted=deps[state_port][0][:,columns["fitted_state"]] if state_port in deps else np.zeros(n,dtype=np.int8)
                    caps[fitted==0,:]=0;caps[fitted<0,:]=-1
                if declaration.get("registered",True):
                    if match is None:caps.fill(-1)
                    elif not match.get("eligible"):caps.fill(0)
            elif kind=="predict":
                match=self.matches.get(node.name,{})
                contract=match.get("output_contract",{})
                coverage=support(contract.get("support"));evidence=support(contract.get("observation_evidence"))
                state_port=declaration.get("state_port")
                fitted=np.ones(n,dtype=np.int8)
                if match.get("input_contract",{}).get("requires_fitted_state"):
                    fitted=deps[state_port][0][:,columns["fitted_state"]] if state_port in deps else np.zeros(n,dtype=np.int8)
                data_port=declaration.get("data_port")
                caps=advertised(match.get("capabilities",()))
                caps[fitted<0,:]=-1;coverage=np.where(fitted<0,-1,coverage);evidence=np.where(fitted<0,-1,evidence)
                if data_port not in deps or not match.get("eligible",False):fitted=np.zeros(n,dtype=np.int8)
                caps[fitted==0,:]=0;coverage=np.where(fitted==0,0,coverage);evidence=np.where(fitted==0,0,evidence)
                if not match:
                    caps.fill(-1);coverage.fill(-1);evidence.fill(-1)
            elif kind in {"fallback","select"}:
                branch_inputs=declaration["inputs"]
                if all(name in deps for name in branch_inputs):
                    a,b=(deps[name] for name in branch_inputs)
                    if kind=="fallback":
                        coverage=np.where((a[1]==1)|(b[1]==1),1,np.where((a[1]==0)&(b[1]==0),0,-1))
                        caps=np.where((a[1]==1)[:,None],a[0],np.where((a[1]==0)[:,None],b[0],np.where(a[0]==b[0],a[0],-1)))
                        evidence=np.where(a[1]==1,a[2],np.where(a[1]==0,b[2],np.where(a[2]==b[2],a[2],-1)))
                    else:
                        hourly=self.semantics.observed.value.get("regular_hourly")
                        if declaration.get("required_interval_seconds")==3600 and hourly is not True:
                            caps.fill(-1);coverage.fill(-1);evidence.fill(-1)
                            if hourly is False:incompatible.append(dict(node=node.name,reason="Gap splitter requires regular 1-hour labels",interval_ns=self.semantics.observed.value.get("interval_ns")))
                        elif size:
                            mask=length<=node.parameters[declaration["threshold_parameter"]]
                            caps=np.where(mask[:,None],a[0],b[0]);coverage=np.where(mask,a[1],b[1]);evidence=np.where(mask,a[2],b[2])
                        else:caps.fill(-1);coverage.fill(-1);evidence.fill(-1)
            elif kind=="validate":
                field=declaration["input"]
                if field in deps:
                    caps,coverage,evidence=deps[field];caps=caps.copy()
                    for name in declaration.get("capabilities",()):caps[:,columns[name]]=1
            else:
                caps.fill(-1);coverage.fill(-1);evidence.fill(-1)
            caps[:,columns["complete_temporal_coverage"]]=coverage if size else -1
            boundary=left!=right
            caps[:,columns["boundary_temporal_recovery"]]=np.where(boundary,coverage,1) if boundary.any() else -1
            known=bool(np.all(caps>=0) and np.all(coverage>=0))
            if not known:unknown.append(dict(node=node.name,output=output,reason="Some output-path proofs unavailable; not a proof of absence"))
            ports.append(dict(node=node.name,mode=node.parameters.get("mode","run"),output=output,
                capabilities=[name for name,i in columns.items() if np.all(caps[:,i]==1)],
                uncovered_conditions=int((coverage==0).sum()),unknown_conditions=int((coverage<0).sum()),
                uncovered_positions=int(counts[coverage==0].sum()),known=known,
                bindings={n:dict(node=p.name,output=o) for n,(p,o) in bound.items()}))
            memo[key]=(caps,coverage,evidence);return memo[key]
        output=next(iter(factory.adapter(graph.root_node).outputs))
        caps,cov,observed=visit(graph.root_node,output)
        missing=[name for name in self.requirements.required if np.any(caps[:,columns[name]]==0)]
        unproven=[name for name in self.requirements.required if name not in missing and np.any(caps[:,columns[name]]<0)]
        local_acceptance=[]
        for problem in self.requirements.problems:
            binding=problem.get("binding",{})
            if binding.get("scope")!="source_branch" or not problem["must_handle"]:continue
            evidence=[]
            for node in graph.nodes:
                for port,contract in factory.adapter(node).outputs.items():
                    if contract.kind!=binding["kind"]:continue
                    proof=visit(node,port)[0]
                    evidence.append(dict(operation=node.name,output=port,
                        missing=[c for c in problem["capabilities"] if c not in columns or np.any(proof[:,columns[c]]==0)],
                        unknown=[c for c in problem["capabilities"] if c in columns and np.any(proof[:,columns[c]]<0)]))
            local_missing=not evidence or any(item["missing"] for item in evidence)
            local_unknown=any(item["unknown"] for item in evidence)
            label=problem["id"]+"::"+",".join(problem["capabilities"])
            if local_missing:missing.append(label)
            elif local_unknown:unproven.append(label)
            local_acceptance.append(dict(problem_id=problem["id"],source_ids=binding["source_ids"],kind=binding["kind"],evidence=evidence,
                status="MISSING" if local_missing else "UNKNOWN" if local_unknown else "PASS"))
        reject=bool(status["complete"] and "complete_temporal_coverage" in missing)
        rejected_nodes=[n.name for n in graph.nodes if n.name in self.matches and not self.matches[n.name]["eligible"]]
        hard_reject=bool((status["complete"] and missing) or rejected_nodes or incompatible)
        acceptance="REJECT" if hard_reject else "INCOMPLETE" if not status["complete"] else "UNKNOWN" if unproven else "PASS"
        soft=float(counts[observed==0].sum()/counts.sum()) if size and counts.sum() and np.all(observed>=0) else None
        return dict(port_gap=status["gap"],missing_capabilities=missing,unknown_capabilities=unproven,
            gap=status["gap"]+len(missing)+len(unproven),acceptance_status=acceptance,
            hard_reject=hard_reject,rejected_nodes=rejected_nodes,incompatible=incompatible,known=not unproven,unknown=unknown,
            coverage_reject=reject,output_capabilities=[name for name,i in columns.items() if np.all(caps[:,i]==1)],ports=ports,
            support_deficit=soft,soft_objectives=self.requirements.soft_objectives,
            local_acceptance=local_acceptance,
            weighting="all visible missing-input positions; not validation target/scenario weights")

    def require_graph(self, factory, graph):
        """Final task acceptance; independent of search/posterior switches."""
        from ecodata.algorithms.dag import DAGError
        assessment=self.assess_graph(factory,graph)
        if assessment["acceptance_status"]!="PASS":
            raise DAGError(f"Task requirements {assessment['acceptance_status']}: "
                f"missing={assessment['missing_capabilities']}, unknown={assessment['unknown_capabilities']}, "
                f"incompatible={assessment['incompatible']}",details={"task_acceptance":assessment})
        return assessment

