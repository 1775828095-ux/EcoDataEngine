"""Task-local mixed-gap controls over existing temporal kernels and numerical DAG.

No targets or artificial-mask identities are exposed to predictors or routing.
"""
from dataclasses import dataclass
from hashlib import sha256
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.temporal_reconstruction_dag import reconstruct,TemporalBaseline
from ecodata.algorithms.dag import Artifact,Port,NodeAdapter,DAGError
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.golem_generation import TypedGolemFactory
from ecodata.evaluation.quality_gates import RawInputQualityGate,ResultQualityGate
from ecodata.algorithms.semantic_requirements import require_regular_hourly

METHODS=("linear","nearest","ffill","hour_median","calendar_knn")
IDS=tuple("baseline.temporal_"+m for m in METHODS)
THRESHOLDS=(1,3,6,12,24,48)
TASK="beijing_mixed_gap_v1"


def readonly(a):
    result=np.array(a,copy=True);result.flags.writeable=False;return result


class SealedArtifact(Artifact):
    """Content-addressed immutable task payload; avoid copying a whole year per edge.

    Only constructed after arrays have been made read-only. No third-party DAG
    changes; fitted state is never mutated by these adapters.
    """
    def __init__(self,kind,value,metadata=None):
        super().__init__(kind,value,metadata or {})
        self._identity=super().fingerprint()
    def fingerprint(self):return self._identity
    def __deepcopy__(self,memo):return self


@dataclass(frozen=True)
class SeriesCase:
    station: str
    time: np.ndarray
    values: np.ndarray
    missing: np.ndarray


@dataclass(frozen=True)
class Batch:
    cases: tuple
    lengths: np.ndarray
    offsets: tuple


def missing_runs(values):
    missing=~np.isfinite(values);starts=np.flatnonzero(missing & ~np.r_[False,missing[:-1]])
    ends=np.flatnonzero(missing & ~np.r_[missing[1:],False])+1
    lengths=np.zeros(len(values),int)
    for start,end in zip(starts,ends):lengths[start:end]=end-start
    return lengths


def calendar_features(index):
    return np.c_[np.sin(index.hour*2*np.pi/24),np.cos(index.hour*2*np.pi/24),
        np.sin(index.dayofyear*2*np.pi/366),np.cos(index.dayofyear*2*np.pi/366)]


def predict(batch,method,states=None):
    pieces=[]
    for case in batch.cases:
        times=pd.DatetimeIndex(case.time);series=pd.Series(case.values,index=times)
        if method in {"linear","nearest","ffill"}:
            values=reconstruct(series,method).to_numpy()[case.missing]
        elif method=="hour_median":values=states[case.station]["medians"][times[case.missing].hour]
        elif method=="calendar_knn":values=states[case.station]["knn"].predict(calendar_features(times[case.missing]))
        else:raise DAGError("No such actual temporal kernel")
        pieces.append(values)
    return np.concatenate(pieces)


def setup():
    registry=AlgorithmRegistry()
    for method in METHODS:registry.register(TemporalBaseline(method))
    adapters={};kernel_cache={};port=lambda kind:Port(kind,("unit","contract"))
    meta=dict(unit="ug m-3",contract=TASK)
    def source(name,kind):
        def run(inputs,params,external):
            value=external[name]
            if value.kind!=kind:raise DAGError("Source contract mismatch")
            return {"value":value}
        return run
    adapters["source.masked_batch","run"]=NodeAdapter({}, {"value":port("masked_batch")},source("batch","masked_batch"))
    adapters["source.fitted_history","run"]=NodeAdapter({}, {"value":port("fitted_history")},source("history","fitted_history"))
    def predictor(method,kind):
        def run(inputs,params,external):
            state=inputs["state"].value if "state" in inputs else None
            key=(inputs["batch"].fingerprint(),method,inputs["state"].fingerprint() if "state" in inputs else None)
            hit=key in kernel_cache
            if not hit:kernel_cache[key]=readonly(predict(inputs["batch"].value,method,state))
            return {"field":SealedArtifact(kind,kernel_cache[key],dict(meta,method=method,kernel_cache_hit=hit))}
        return run
    for role in ["direct","short","long"]:
        for method in METHODS:
            inputs={"batch":port("masked_batch")}
            if method in {"hour_median","calendar_knn"}:inputs["state"]=port("fitted_history")
            for suffix,kind in [(role,"estimate_"+role),(role+"_backup","backup_"+role)]+(
                [(role+"_partial","partial_"+role)] if method in {"linear","ffill"} else []):
                adapters["baseline.temporal_"+method,suffix]=NodeAdapter(inputs,{"field":port(kind)},predictor(method,kind),mode=suffix)
        def fallback(inputs,params,external,role=role):
            a,b=inputs["primary"].value,inputs["backup"].value
            if a.shape!=b.shape:raise DAGError("Fallback shape mismatch")
            missing=~np.isfinite(a);out=np.where(missing,b,a)
            return {"field":SealedArtifact("estimate_"+role,readonly(out),dict(meta,fallback_triggered=int(missing.sum()),
                primary_preserved=True,backup_used=int((missing & np.isfinite(b)).sum())))}
        adapters["prepare.conditional_fallback",role]=NodeAdapter({"primary":port("partial_"+role),"backup":port("backup_"+role)},
            {"field":port("estimate_"+role)},fallback,mode=role)
    def split(inputs,params,external):
        require_regular_hourly(inputs["batch"].value.cases)
        lengths=inputs["batch"].value.lengths;threshold=params["threshold"]
        if threshold not in THRESHOLDS:raise DAGError("Threshold outside frozen domain")
        short=lengths<=threshold;a,b=inputs["short"].value,inputs["long"].value
        if a.shape!=lengths.shape or b.shape!=lengths.shape:raise DAGError("Split shape mismatch")
        return {"field":SealedArtifact("estimate_direct",readonly(np.where(short,a,b)),dict(meta,
            threshold=threshold,threshold_unit="hours",record_interval_seconds=3600,
            short_count=int(short.sum()),long_count=int((~short).sum()),routing_source="visible NaN runs only"))}
    adapters["prepare.visible_gap_split","run"]=NodeAdapter({"batch":port("masked_batch"),"short":port("estimate_short"),"long":port("estimate_long")},
        {"field":port("estimate_direct")},split)
    def quality(inputs,params,external):
        raw=inputs["field"].value;batch=inputs["batch"].value;query=external["query"].value
        if len(raw)!=len(batch.lengths):raise DAGError("Only missing-position writes permitted")
        # Original finite values are not writable via this port. Explicit merge
        # verifies protection, including natural missing positions not scored.
        for case,(start,end) in zip(batch.cases,batch.offsets):
            merged=case.values.copy();visible=np.isfinite(case.values);merged[case.missing]=raw[start:end]
            if not np.array_equal(merged[visible],case.values[visible]):raise DAGError("Visible value modified")
        values=raw[query];clipped=np.maximum(values,0)
        if not np.isfinite(clipped).all():raise DAGError(f"Incomplete target coverage: {int(np.isfinite(clipped).sum())}/{len(clipped)}",
            details={"raw_prediction":readonly(values),"projected_prediction":readonly(clipped)})
        ds=xr.Dataset({"pm25":("sample",clipped)});ds.pm25.attrs["units"]="ug m-3"
        gate=ResultQualityGate().validate(ds,ds,variable="pm25",semantic_type="continuous",nonnegative=True,expected_output_unit="ug m-3")
        if gate["status"]=="BLOCK":raise DAGError("Result quality BLOCK")
        return {"field":SealedArtifact("validated_field",(readonly(values),readonly(clipped)),dict(meta,quality=gate["status"],
            negative_projection_count=int((values<0).sum()),negative_projection_max=float(max(0,-values.min())),
            original_visible_protected=True,target_count=len(values)))}
    adapters["quality.mixed_gap","run"]=NodeAdapter({"field":port("estimate_direct"),"batch":port("masked_batch")},
        {"field":port("validated_field")},quality)
    # Execution semantics, consumed by the common platform decision service.
    # No algorithm preferences or gap-length winners are registered here.
    for (operation,mode),adapter in adapters.items():
        if operation=="source.masked_batch":declaration=dict(operation="source",capabilities=["masked_input"])
        elif operation=="source.fitted_history":declaration=dict(operation="source",capabilities=["fitted_state"])
        elif operation in IDS:declaration=dict(operation="predict",data_port="batch",state_port="state")
        elif operation=="prepare.conditional_fallback":declaration=dict(operation="fallback",inputs=["primary","backup"])
        elif operation=="prepare.visible_gap_split":declaration=dict(operation="select",inputs=["short","long"],threshold_parameter="threshold",
            threshold_unit="hours",required_interval_seconds=3600)
        else:declaration=dict(operation="validate",input="field",capabilities=["nonnegative_validation"])
        adapter.decision_contract=declaration
    boundaries=("source.masked_batch","source.fitted_history","prepare.conditional_fallback","prepare.visible_gap_split","quality.mixed_gap")
    registry.register_task_adapters(TASK,adapters,algorithm_ids=IDS,boundary_ids=boundaries)
    return registry,MixedGapFactory(adapters,IDS,max_nodes=18,max_depth=9)


class MixedGapFactory(TypedGolemFactory):
    SOURCES={"source.masked_batch":"batch","source.fitted_history":"history"}
    PARAMS={"prepare.visible_gap_split":{"threshold":THRESHOLDS}}
    SUPPORT={"linear":"both","nearest":"either","ffill":"left","hour_median":"training","calendar_knn":"training"}
    def parameters(self,key):
        for values in super().parameters(key):
            value=dict(values)
            if key[0]=="quality.mixed_gap":value["source"]="query"
            yield tuple(sorted(value.items()))
    def recipes(self,kind,stack=()):
        recipes=super().recipes(kind,stack)
        filtered=[]
        for output,recipe in recipes:
            if recipe[0]=="prepare.conditional_fallback":
                by={name:child for name,_,child in recipe[2]}
                if by["primary"][0]==by["backup"][0]:continue
            if recipe[0]=="prepare.visible_gap_split":
                by={name:child for name,_,child in recipe[2]}
                # Reject structurally identical branch programs, not numerical
                # equivalence inferred from any development/test performance.
                def neutral(r):return (r[0],tuple((k,v) for k,v in r[1] if k!="mode"),tuple((n,neutral(c)) for n,_,c in r[2]))
                if neutral(by["short"])==neutral(by["long"]):continue
            filtered.append((output,recipe))
        return filtered
    def inspect(self,graph,*,complete=False):
        status=super().inspect(graph,complete=complete)
        def neutral(node):
            return (node.name,tuple(sorted((k,v) for k,v in node.parameters.items() if k not in {"mode","_port_bindings"})),
                tuple((name,neutral(node.nodes_from[index])) for name,(index,_) in sorted(node.parameters.get("_port_bindings",{}).items())))
        for node in graph.nodes:
            if node.name=="prepare.visible_gap_split" and node.parameters.get("threshold") not in THRESHOLDS:
                raise DAGError("Illegal threshold")
            parents={name:node.nodes_from[index] for name,(index,_) in node.parameters.get("_port_bindings",{}).items()}
            if node.name=="prepare.conditional_fallback" and {"primary","backup"}<=parents.keys():
                if parents["primary"].name==parents["backup"].name:raise DAGError("Self-fallback is not a meaningful control graph")
            if node.name=="prepare.visible_gap_split" and {"short","long"}<=parents.keys():
                if neutral(parents["short"])==neutral(parents["long"]):raise DAGError("Identical split branches are not meaningful")
        return status

    def semantic_risk(self,graph,conditions):
        """Coverage capabilities on observable gap length/support, never targets."""
        lengths,left,right=conditions;memo={}
        def coverage(node):
            if id(node) in memo:return memo[id(node)]
            parents={k:node.nodes_from[i] for k,(i,_) in node.parameters.get("_port_bindings",{}).items()}
            if node.name.startswith("baseline.temporal_"):
                support=self.SUPPORT[node.name.removeprefix("baseline.temporal_")]
                value={"both":left & right,"either":left | right,"left":left,"training":np.ones(len(lengths),bool)}[support]
            elif node.name=="prepare.conditional_fallback":
                value=coverage(parents["primary"])|coverage(parents["backup"]) if {"primary","backup"}<=parents.keys() else np.zeros(len(lengths),bool)
            elif node.name=="prepare.visible_gap_split":
                value=np.where(lengths<=node.parameters["threshold"],coverage(parents["short"]),coverage(parents["long"])) if {"short","long"}<=parents.keys() else np.zeros(len(lengths),bool)
            elif node.name=="quality.mixed_gap":value=coverage(parents["field"]) if "field" in parents else np.zeros(len(lengths),bool)
            else:value=np.zeros(len(lengths),bool)
            memo[id(node)]=value;return value
        return float(1-coverage(graph.root_node).mean())
