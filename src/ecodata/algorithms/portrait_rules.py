"""Evidence-limited three-portrait matching, separate from numerical accuracy.

Boolean properties certify a guarantee, not an observed error rate. Unknowns
remain intervals; neither unsupported algorithms nor combinations get free gains.
"""
import numpy as np
from ecodata.algorithms import attribute_registry as attributes


VERSION = "portrait_rules_v3_registered_joint"
DIMENSIONS = attributes.core_ids()


def model_requirements(value, registry=None):
    registry = registry or attributes.load()
    definitions = {a["id"]:a for a in registry["attributes"]}
    if not isinstance(value, dict) or set(value) - {"priorities", "hard", "evidence"}:
        raise ValueError("Unsupported model_requirements field")
    priorities = value.get("priorities", {})
    hard = value.get("hard", [])
    if not isinstance(priorities, dict) or set(priorities) - set(definitions) or any(definitions[k]["role"]!="scoring" for k in priorities):
        raise ValueError("Unsupported model priority; no evidence-backed rule consumer")
    if not isinstance(hard, list) or len(set(hard)) != len(hard) or set(hard) - set(definitions) or any(not definitions[k]["hard_allowed"] for k in hard):
        raise ValueError("Unsupported model hard requirement")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0
           for v in priorities.values()) or (priorities and sum(priorities.values()) <= 0):
        raise ValueError("Model priorities require finite nonnegative weights and a positive total")
    evidence = value.get("evidence")
    if (priorities or hard) and (not isinstance(evidence, str) or not evidence.strip()):
        raise ValueError("Model priorities/hard requirements need explicit task evidence")
    total = sum(priorities.values())
    return dict(priorities={k: v / total for k, v in priorities.items()} if total else {},
                hard=hard, evidence=evidence, origin="explicit task requirements; not inferred simulator sensitivity")


def data_facts(context, index):
    ds = context.sources[index].dataset
    field = ds[context.variable]
    a = np.asarray(field.values, dtype=float)
    valid = np.isfinite(a)
    facts = dict(source_index=index, finite_fraction=float(valid.mean()),
                 negative_fraction=float((a[valid] < 0).mean()) if valid.any() else None,
                 minimum=float(a[valid].min()) if valid.any() else None,
                 maximum=float(a[valid].max()) if valid.any() else None,
                 increasing_axes=False, min_axis_size=None,
                 variation=None, interior_rectilinear=False,
                 evidence="actual visible input; no reference labels")
    if field.dims != ("lat", "lon") or min(a.shape) < 2 or not valid.all():
        return facts
    axes = [np.asarray(ds[x].values, dtype=float) for x in ("lat", "lon")]
    if any(v.ndim != 1 or not np.isfinite(v).all() or not (np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)) for v in axes):
        return facts
    targets = [context.target_lat, context.target_lon]
    facts["increasing_axes"] = all(np.all(np.diff(v)>0) for v in axes)
    facts["min_axis_size"] = min(len(v) for v in axes)
    facts["interior_rectilinear"] = all(t is not None and len(t) > 0 and np.isfinite(t).all()
        and min(t) >= v.min() and max(t) <= v.max() for t, v in zip(targets, axes))
    span = float(np.ptp(a))
    differences = np.concatenate([np.abs(np.diff(a, axis=i)).ravel() for i in (0, 1)])
    facts["variation"] = float(differences.mean() / span) if span else 0.0
    facts["variation_definition"] = "mean absolute adjacent difference / visible range; pixel-index variation, NOT noise or physical gradient"
    return facts


def algorithm_properties(plugin):
    """Whitelist reviewed implementations, never infer guarantees from ID substrings."""
    from ecodata.algorithms.plugins import RegridPlugin
    from ecodata.algorithms.raster_reconstruction_task import RasterBaselineAdapter, NonnegativeRaster, CoarseObservationProjection
    from ecodata.interface.raster_upload import ShapePreservingRaster
    from ecodata.algorithms.open_source_spatial import XarrayAggregationPlugin
    base = plugin.upstream if isinstance(plugin, RasterBaselineAdapter) else plugin
    transfer = "unknown"
    evidence = []
    if isinstance(base, RegridPlugin) and base.method in {"nearest", "bilinear"}:
        transfer = base.method
        evidence = ["src/ecodata/spatial/regridding.py:GridRegridder.apply",
                    "src/ecodata/engines/grid.py:GridEngine.regrid",
                    "nearest selects one source value; bilinear is a continuous convex combination on finite interior rectilinear cells"]
    elif isinstance(base, ShapePreservingRaster):
        transfer = "pchip"
        evidence = ["src/ecodata/interface/raster_upload.py:ShapePreservingRaster.run",
                    "SciPy PchipInterpolator: shape-preserving 1D stages, latitude then longitude; each stage remains in input range on interior intervals",
                    "composition is continuous; no isotropic 2D C1 or unseen-peak guarantee"]
    elif isinstance(base, XarrayAggregationPlugin) and base.method == "cubic":
        transfer = "cubic"
        evidence = ["src/ecodata/algorithms/open_source_spatial.py:XarrayAggregationPlugin.run",
                    "xarray-regrid cubic delegates to xarray.interp(method=cubic); continuous interior spline with at least four samples per axis",
                    "no convex-weight guarantee: range preservation/nonnegativity are not certified"]
    elif isinstance(plugin, NonnegativeRaster):
        transfer = "clip_nonnegative"
        evidence = ["src/ecodata/algorithms/raster_reconstruction_task.py:NonnegativeRaster.run",
                    "max(x,0) guarantees nonnegativity; continuous projection; no assumed accuracy gain"]
    elif isinstance(plugin, CoarseObservationProjection):
        transfer = "coarse_projection"
        evidence = ["src/ecodata/algorithms/raster_reconstruction_task.py:CoarseObservationProjection.run",
                    "QP enforces nonnegative and declared coarse operator consistency; not physical total-mass conservation"]
    return dict(transfer=transfer, evidence=evidence,
                scope="finite 2D interior rectilinear field; PCHIP increasing axes; cubic at least four samples per axis" if transfer in {"nearest", "bilinear", "pchip", "cubic"} else "actual parent output contract",
                accuracy_gain=None, noise_robustness=None, peak_preservation=None)


def score_graph(tables, graph, factory):
    """Consume all three tables and propagate properties on actual graph edges."""
    algorithms = {a["id"]: a for a in tables["algorithms"]}
    data = tables["data"]
    needs = tables["model"]["matching_requirements"]
    registry=tables["model"].get("attribute_registry") or attributes.load()
    definitions={a["id"]:a for a in registry["attributes"]}
    trace = []; memo = {}

    def visit(node):
        if id(node) in memo:
            return memo[id(node)]
        parents = [visit(p) for p in node.nodes_from]
        bindings=node.parameters.get("_port_bindings",{})
        def parent_at(port):
            binding=bindings.get(port)
            return parents[binding[0]] if binding is not None else None
        adapter = factory.adapter(node)
        identifier = adapter.decision_contract.get("algorithm_id", node.name)
        properties = {k: None for k in DIMENSIONS}
        facts = None
        evidence = []
        if node.name.startswith("source.source"):
            index = int(node.name[len("source.source"):])
            facts = data[index]["matching_facts"]
        elif (node.name.startswith("quality.") or adapter.decision_contract.get("structural_only")) and len(parents) == 1:
            properties = dict(parents[0]["properties"]); facts = parents[0]["facts"]
        elif identifier in algorithms:
            rule = algorithms[identifier]["matching_properties"]
            transfer = rule["transfer"]; evidence = rule["evidence"]
            # A multi-input scientific operation cannot inherit a single-source
            # reconstruction guarantee, except its explicitly typed field tail.
            parent = parent_at("source0") if "source0" in bindings and "source1" not in bindings else None
            source_binding=bindings.get("source0")
            direct_source=source_binding is not None and node.nodes_from[source_binding[0]].name.startswith("source.")
            if transfer in {"clip_nonnegative", "coarse_projection"}:
                parent=parent_at("field")
            if parent:
                facts = parent["facts"]
            if transfer in {"nearest", "bilinear", "pchip", "cubic"} and facts and facts["interior_rectilinear"] and facts["finite_fraction"] == 1:
                # Known kernels consume their source field; do not extend this
                # theorem through an unreviewed upstream scientific operation.
                if direct_source:
                    if (transfer != "pchip" or facts["increasing_axes"]) and (transfer != "cubic" or facts["min_axis_size"] >= 4):
                        properties.update(range_preservation=transfer != "cubic",
                                          continuity=transfer != "nearest" or facts["variation"] == 0,
                                          no_mixing=transfer == "nearest",
                                          nonnegative=facts["minimum"] >= 0 if transfer != "cubic" else None,
                                          coarse_consistency=None)
            elif transfer == "clip_nonnegative" and parent:
                properties = dict(parent["properties"])
                properties["nonnegative"] = True
                if facts:
                    if facts["maximum"] is not None and facts["maximum"] < 0:
                        properties["range_preservation"] = False
                    if facts["negative_fraction"]:
                        # A zero projection is not certified to copy source values.
                        properties["no_mixing"] = False
                else:
                    properties["range_preservation"] = None
                    properties["no_mixing"] = None
                properties["coarse_consistency"] = None
            elif transfer == "coarse_projection":
                properties.update(nonnegative=True, coarse_consistency=True)
        state = dict(properties=properties, facts=facts)
        memo[id(node)] = state
        trace.append(dict(operation=node.name, algorithm_id=identifier,
                          parents=[p.name for p in node.nodes_from],
                          input_bindings={k:node.nodes_from[v[0]].name for k,v in bindings.items()},
                          properties=properties, evidence=evidence))
        return state

    roots = list(graph.root_nodes())
    final = visit(roots[0]) if len(roots) == 1 else dict(properties={k: None for k in DIMENSIONS}, facts=None)
    failures = [dict(dimension=k, value=final["properties"].get(definitions[k]["property"]), reason="required guarantee not established")
                for k in needs["hard"] if final["properties"].get(definitions[k]["property"]) is not True]
    # Activation depends on visible data, never a validation/test score. With
    # no reviewed transfer/scope, activation itself stays unknown.
    facts = final["facts"]
    terms = []
    for dimension, weight in needs["priorities"].items():
        if not weight:
            continue
        definition=definitions[dimension]
        activation = (1.0 if definition["activation"]=="constant_one" else facts.get(definition["activation"])) if facts else None
        value = final["properties"].get(definition["property"]) if definition["provider"]=="guarantee" else None
        utility = None
        if definition["provider"]=="declared_cost":
            cost=sum(algorithms[t["algorithm_id"]]["declared_cost"] for t in trace if t["algorithm_id"] in algorithms and t["operation"]!="validation.result")
            utility=-cost/(1+cost); activation=1.0
        if definition["provider"]=="station_probe":
            from ecodata.algorithms.numerical_search import key_for
            probe=tables.get("station_probes",{}).get(key_for(graph,factory))
            activation=1.0
            if probe and probe["metrics"] is not None:
                error=probe["metrics"][definition["property"]]
                utility=-error/(probe["scale"]+error)
        terms.append(dict(dimension=dimension, weight=weight, activation=activation, guarantee=value,
                          model_evidence=needs["evidence"],
                          data_evidence="matching_facts: actual visible input",
                          algorithm_evidence=[t for t in trace if t["evidence"]],
                          definition=definition, raw_activation=activation, utility=utility))
    terms,lower,upper=parameterized_terms(terms,tables["model"].get("scoring_parameters",{}))
    return dict(version=VERSION, hard_status="REJECTED" if failures else "PASS", hard_failures=failures,
                score=None if any(t["status"] == "UNKNOWN" for t in terms) else lower,
                score_interval=[lower, upper], terms=terms, transfer_trace=trace,registry_sha256=registry["sha256"],
                properties=final["properties"],
                basis="structural suitability, NOT expected RMSE or combination gain; unknown interval is not statistical confidence")


def validate_product(dataset, variable, tables):
    """Independent numerical backstop for checkable hard guarantees."""
    a = np.asarray(dataset[variable].values)
    checks = []
    registry=tables["model"].get("attribute_registry") or attributes.load()
    definitions={a["id"]:a for a in registry["attributes"]}
    for attribute_id in tables["model"]["matching_requirements"]["hard"]:
        dimension=definitions[attribute_id]["property"]
        if dimension == "nonnegative":
            ok = bool(np.isfinite(a).all() and (a >= 0).all())
        elif dimension == "range_preservation":
            facts = [d["matching_facts"] for d in tables["data"]]
            lo = min(f["minimum"] for f in facts); hi = max(f["maximum"] for f in facts)
            tolerance = 1e-10 * max(1, abs(lo), abs(hi))
            ok = bool(np.isfinite(a).all() and a.min() >= lo-tolerance and a.max() <= hi+tolerance)
        else:
            checks.append(dict(dimension=dimension, status="CONTRACT_ONLY", reason="guarantee checked on graph; coarse projection checks its own residual"))
            continue
        checks.append(dict(dimension=dimension, status="PASS" if ok else "FAIL"))
    return dict(status="FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS", checks=checks)


def parameterized_terms(terms, parameters):
    """Same bounded rule calculator for inference and development calibration."""
    from copy import deepcopy
    result=[];lower=upper=0.
    for original in terms:
        term=deepcopy(original)
        setting=parameters.get(term["dimension"],{})
        multiplier=setting.get("multiplier",1.)
        threshold=setting.get("activation_threshold",0.)
        activation=term["raw_activation"]
        if activation is not None and activation<threshold:activation=0.
        weight=term["weight"]*multiplier
        if not term["definition"]["scoring_enabled"]:
            lo=hi=0.;status="DISABLED"
        elif weight==0 or activation==0:
            lo=hi=0.;status="INACTIVE"
        elif "utility_interval" in term and activation is not None:
            lo,hi=[weight*activation*v for v in term["utility_interval"]]
            status="UNKNOWN" if term.get("utility") is None else "STATE_REDUCTION"
        elif activation is None or (term["guarantee"] is None and term.get("utility") is None):
            radius=weight*(activation if activation is not None else 1.)
            lo,hi=-radius,radius;status="UNKNOWN"
        else:
            utility=term.get("utility")
            if utility is None:utility=1. if term["guarantee"] else -1.
            lo=hi=weight*activation*utility
            status="ESTABLISHED_GUARANTEE" if term["guarantee"] is True else "NO_GUARANTEE" if term["guarantee"] is False else "TRAIN_STATION_PROBE" if term["definition"]["provider"]=="station_probe" else "STATE_REDUCTION" if term["definition"]["provider"]=="state_metric" else "DECLARED_COST"
        term.update(activation=activation,parameters=dict(multiplier=multiplier,activation_threshold=threshold),
            effective_weight=weight,status=status,contribution=None if status=="UNKNOWN" else lo,interval=[lo,hi])
        lower+=lo;upper+=hi;result.append(term)
    return result,float(lower),float(upper)


def combination_gains(rows, protocol):
    valid = [r for r in rows if r["status"] == "PASS" and r.get("task_loss") is not None]
    singles = [r for r in valid if len(r["steps"]) == 1]
    baseline = min(singles, key=lambda r: r["task_loss"]) if singles else None
    return [dict(key=r["key"], steps=r["steps"], status="OBSERVED" if baseline else "NO_COMPARABLE_SINGLE",
                 baseline_key=baseline["key"] if baseline else None,
                 gain=baseline["task_loss"]-r["task_loss"] if baseline else None,
                 accuracy_claim=protocol["kind"] == "reference_error",
                 scope="same run, data, evaluation protocol and preferences; evaluated feasible singles only; no synergy assumed")
            for r in valid if len(r["steps"]) > 1]
