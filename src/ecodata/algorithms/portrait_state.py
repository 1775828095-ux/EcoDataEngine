"""Three-portrait state reduction, with explicit empirical/unknown transfers."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import xarray as xr


METRICS=("missing_fraction","negative_fraction","range_excess","peak_loss","roughness","grid_gap","coarse_error","observation_bias","detail_error","expected_error")


def scientific_identity(context):
    from ecodata.algorithms.dag import Artifact
    transport={"source_file","source_sha256"}
    def clean(dataset):
        copied=dataset.copy(deep=False)
        copied.attrs={k:v for k,v in dataset.attrs.items() if k not in transport}
        return copied
    observations=context.observations
    if observations is not None:
        observations=observations.copy()
        columns=[k for k in ("station_id","time","longitude","latitude") if k in observations]
        if columns:observations=observations.sort_values(columns).reset_index(drop=True)
        observations.attrs={k:v for k,v in observations.attrs.items() if k not in transport}
    return Artifact("scientific_visible_inputs",dict(sources=[clean(s.dataset) for s in context.sources],
        auxiliaries=[clean(s) for s in context.auxiliaries],observations=observations)).fingerprint()


def measure(dataset, context):
    values=np.asarray(dataset[context.variable].values,float)
    original=context.sources[0].dataset
    base=np.asarray(original[context.variable].values,float)
    finite=values[np.isfinite(values)]; initial=base[np.isfinite(base)]
    metrics={name:None for name in METRICS}
    metrics["missing_fraction"]=float(1-np.isfinite(values).mean())
    template=original.attrs.get("data_role")=="prediction_template"
    is_template=dataset.attrs.get("data_role")=="prediction_template"
    features=dict(missing_fraction=metrics["missing_fraction"],size=int(values.size),ndim=values.ndim)
    if context.observations is not None:
        features["available_stations"]=int(len(context.observations))
    if len(finite) and not is_template:
        metrics["negative_fraction"]=float((finite<0).mean())
        features.update(mean=float(finite.mean()),std=float(finite.std()),span=float(np.ptp(finite)))
        if len(initial) and not template:
            scale=max(float(np.ptp(initial)),1e-8)
            metrics["range_excess"]=float(max(0,initial.min()-finite.min(),finite.max()-initial.max())/scale)
            def same_extent(axis):
                if axis not in dataset.coords or axis not in original.coords or dataset[axis].ndim!=1:return False
                a=np.asarray([dataset[axis].values.min(),dataset[axis].values.max()])
                b=np.asarray([original[axis].values.min(),original[axis].values.max()])
                return np.allclose(a,b) if a.dtype.kind in "iuf" and b.dtype.kind in "iuf" else np.array_equal(a,b)
            same_domain=(dataset[context.variable].dims==original[context.variable].dims and
                all(same_extent(axis) for axis in dataset[context.variable].dims))
            if same_domain:metrics["peak_loss"]=float(abs(finite.max()-initial.max())/scale)
        if values.ndim==2 and min(values.shape)>=3 and np.isfinite(values).all():
            axes=dataset[context.variable].dims
            regular=all(axis in dataset.coords and dataset[axis].ndim==1 and
                len(dataset[axis])>2 and np.allclose(np.diff(dataset[axis]),np.diff(dataset[axis])[0]) for axis in axes)
            comparable=all(axis in original.coords and original[axis].ndim==1 and np.ptp(original[axis].values)>0 for axis in axes)
            if regular and comparable:
                normalized=[(dataset[axis].values-float(original[axis].values.min()))/float(np.ptp(original[axis].values)) for axis in axes]
                dy,dx=np.gradient(values,*normalized)
                denominator=max(float(initial.std()) if len(initial) and not template else float(finite.std()),1e-8)
                metrics["roughness"]=float(np.mean(dx*dx+dy*dy)/denominator**2)
                features["gradient_p95"]=float(np.quantile(np.hypot(dx,dy),.95)/denominator)
    if context.target_lat is not None and context.target_lon is not None:
        metrics["grid_gap"]=float(not all(axis in dataset.coords and np.array_equal(dataset[axis],target)
            for axis,target in (("lat",context.target_lat),("lon",context.target_lon))))
    operator=context.parameters.get("degradation_operator");coarse=context.parameters.get("coarse_reference")
    if operator is not None and coarse is not None and operator.shape[1]==values.size and np.isfinite(values).all():
        metrics["coarse_error"]=float(np.sqrt(np.mean((operator@values.ravel()-np.asarray(coarse).ravel())**2)))
    if (not is_template and context.observations is not None
            and context.parameters.get("observation_comparison_confirmed") is True
            and context.parameters.get("observation_comparison_evidence")):
        from ecodata.algorithms.station_calibration import background_stations
        from ecodata.core.schemas import EcoDataBundle
        from ecodata.core.exceptions import SchemaError
        try:
            _,train,prediction=background_stations(replace(context,sources=(EcoDataBundle(dataset),),auxiliaries=()))
            metrics["observation_bias"]=float(abs((prediction-train.value.to_numpy(float)).mean()))
        except (SchemaError,ValueError,KeyError,TypeError,IndexError):
            pass
    return dict(metrics=metrics,features=features,
        evidence="Measured visible field; roughness/peak are structural proxies, not independent accuracy; observation bias may be in-sample",
        field_role=dataset.attrs.get("data_role","physical_field"))


def dataset_of(value):
    if isinstance(value,xr.Dataset):return value
    if isinstance(getattr(value,"dataset",None),xr.Dataset):return value.dataset
    return getattr(getattr(value,"data",None),"dataset",None)


def instrument(factory, context):
    for key,adapter in list(factory.adapters.items()):
        def execute(inputs,params,external,original=adapter.execute,key=key):
            output=original(inputs,params,external)
            before={name:artifact.metadata.get("joint_state") for name,artifact in inputs.items()}
            for artifact in output.values():
                ds=dataset_of(artifact.value)
                if isinstance(ds,xr.Dataset) and context.variable in ds and artifact.kind!="platform_residual_correction":
                    state=measure(ds,context)
                    artifact.metadata=dict(artifact.metadata,joint_state=state,joint_transition=dict(
                        operation=key[0],mode=key[1],parents=before,after=state,
                        parameters={k:v for k,v in params.items() if k not in {"_port_bindings","mode"}},
                        input_fingerprints={name:a.fingerprint() for name,a in inputs.items()},actual_execution=True))
            return output
        factory.adapters[key]=replace(adapter,execute=execute)


def risk(value,issue):
    return float(np.clip((value-issue["acceptable"])/(issue["bad"]-issue["acceptable"]),0,1))


def score(tables, graph, factory, context, memory=None):
    from ecodata.algorithms.portrait_rules import score_graph,parameterized_terms
    from ecodata.algorithms.numerical_search import key_for
    baseline=measure(context.sources[0].dataset,context)
    algorithms={a["id"]:a for a in tables["algorithms"]}
    trace=[];memo={}
    def visit(node):
        if id(node) in memo:return memo[id(node)]
        parents=[visit(p) for p in node.nodes_from]
        adapter=factory.adapter(node);bindings=node.parameters.get("_port_bindings",{})
        def at(port):
            b=bindings.get(port)
            return parents[b[0]] if b else None
        identifier=adapter.decision_contract.get("algorithm_id",node.name)
        state=dict(metrics={name:None for name in METRICS},features={},evidence="UNKNOWN")
        if node.name.startswith("source.source"):
            index=int(node.name[len("source.source"):])
            state=measure(context.sources[index].dataset,context)
        elif (node.name.startswith("quality.") or adapter.decision_contract.get("structural_only")) and len(parents)==1:
            state=deepcopy(parents[0])
        elif identifier in algorithms:
            parent=at("field") or (at("source0") if "source1" not in bindings else None)
            transfer=algorithms[identifier]["matching_properties"]["transfer"]
            if parent and transfer=="clip_nonnegative":
                state["metrics"]["missing_fraction"]=parent["metrics"]["missing_fraction"]
                state["metrics"]["grid_gap"]=parent["metrics"]["grid_gap"]
                state["metrics"]["negative_fraction"]=0. if parent["metrics"]["missing_fraction"] is not None and parent["metrics"]["missing_fraction"]<1 else None
                state["evidence"]="Analytic nonnegative projection; other side effects not guessed"
                if parent["metrics"]["negative_fraction"]==0:
                    state=deepcopy(parent)
                    state["evidence"]="Projection is identity on an already nonnegative finite field"
            elif parent and transfer in {"nearest","bilinear","pchip","cubic"}:
                facts=tables["data"][0]["matching_facts"]
                if facts["finite_fraction"]==1 and facts["interior_rectilinear"]:
                    state["metrics"].update(missing_fraction=0.,grid_gap=0.)
                    if transfer!="cubic":
                        state["metrics"]["range_excess"]=0.
                        state["metrics"]["negative_fraction"]=0. if facts["minimum"]>=0 else None
                    state["evidence"]="Reviewed finite interior resampling contract; peak and roughness remain unknown"
            if parent and memory is not None:
                from ecodata.algorithms.portrait_memory import estimate_transition
                prediction=estimate_transition(memory,node.name,node.parameters.get("mode","run"),node.parameters,parent)
                if prediction["status"]=="SUPPORTED":
                    state.setdefault("metric_bounds",{})
                    for name,bounds in prediction["metrics"].items():
                        if state["metrics"].get(name) is None:
                            state["metrics"][name]=bounds[1]
                            low,high=parent.get("metric_bounds",{}).get(name,[parent["metrics"][name]]*2)
                            delta=prediction["delta_bounds"][name]
                            state["metric_bounds"][name]=[max(0.,low+delta[0]),max(0.,high+delta[1])]
                    state["features"]=prediction["features"]
                    state["evidence"]=dict(kind="CONDITIONAL_DEVELOPMENT_RESPONSE",evidence=prediction)
        trace.append(dict(operation=node.name,mode=node.parameters.get("mode","run"),
            parents=[p.name for p in node.nodes_from],state=deepcopy(state)))
        memo[id(node)]=state
        return state
    roots=list(graph.root_nodes());final=visit(roots[0])
    case=None
    if memory is not None:
        from ecodata.algorithms.portrait_memory import estimate
        case=estimate(memory,key_for(graph,factory),baseline)
        if case["status"]=="SUPPORTED":
            # Empirical whole-chain response covers its own order/branch effects.
            for name,bounds in case["metrics"].items():
                if final["metrics"].get(name) is None:
                    final["metrics"][name]=bounds[1]
                    final.setdefault("metric_bounds",{})[name]=[bounds[0],bounds[2]]
            final["evidence"]="Analytic guarantees plus scoped development whole-chain responses"
    result=score_graph(tables,graph,factory)
    policy=tables["model"]["state_policy"]
    definitions={a["id"]:a for a in tables["model"]["attribute_registry"]["attributes"]}
    terms=[];bounds={}
    total=sum(i["importance"] for i in policy["scenario"]["issues"])
    for issue in policy["scenario"]["issues"]:
        name=issue["metric"]
        if name=="declared_cost":
            cost=sum(algorithms[factory.adapter(n).decision_contract.get("algorithm_id",n.name)]["declared_cost"] for n in graph.nodes
                if factory.adapter(n).decision_contract.get("algorithm_id",n.name) in algorithms)
            before=0.;after=cost/(1+cost)
        else:before=baseline["metrics"][name];after=final["metrics"][name]
        utility=None if before is None or after is None else risk(before,issue)-risk(after,issue)
        bounds[issue["attribute"]]=([risk(before,issue)-1.,risk(before,issue)] if before is not None else
            [-risk(after,issue),1.-risk(after,issue)] if after is not None else [-1.,1.])
        if utility is not None:
            lo=hi=utility
            if name in final.get("metric_bounds",{}):
                low,high=final["metric_bounds"][name]
                lo=risk(before,issue)-risk(high,issue);hi=risk(before,issue)-risk(low,issue)
            bounds[issue["attribute"]]=[lo,hi]
        terms.append(dict(dimension=issue["attribute"],weight=issue["importance"]/total,activation=1.,raw_activation=1.,
            guarantee=None,utility=utility,definition=definitions[issue["attribute"]],
            utility_interval=bounds[issue["attribute"]],
            data_evidence=dict(metric=name,before=before),algorithm_evidence=dict(after=after,transfer_trace=trace,case=case),
            model_evidence=issue["evidence"]))
    terms,_,_=parameterized_terms(terms,tables["model"].get("scoring_parameters",{}))
    lower=upper=0.
    for term in terms:
        lo,hi=bounds[term["dimension"]]
        if term["status"] in {"INACTIVE","DISABLED"}:lo=hi=0.
        else:lo*=term["effective_weight"];hi*=term["effective_weight"]
        term.update(interval=[lo,hi],contribution=lo if lo==hi else None,
                    status=term["status"] if term["status"] in {"INACTIVE","DISABLED"} else "UNKNOWN" if term["utility"] is None else "STATE_REDUCTION")
        lower+=lo;upper+=hi
    result.update(version="joint_state_v1",terms=terms,score_interval=[lower,upper],
        score=lower if lower==upper else None,baseline=baseline,final_state=final,transfer_trace=trace,
        objective_vector=[-t["interval"][0]/t["weight"] if t["weight"] else 0. for t in terms],
        basis="Three portraits: visible initial issues -> conditional algorithm effects -> scenario-weighted terminal reduction; not independent accuracy",
        uncertainty="Conservative support bounds, not calibrated out-of-domain confidence")
    return result
