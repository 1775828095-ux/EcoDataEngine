"""Supported hourly PM2.5 task adapter for the common numerical search service."""
from pathlib import Path
from time import perf_counter,process_time
import hashlib,traceback
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.numerical_search import run_numerical_search,record,save
from ecodata.algorithms.mixed_gap_dag import setup,TASK,IDS,MixedGapFactory,readonly,SealedArtifact
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor,require_regular_hourly
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.core.schemas import EcoDataBundle


def validate_external(external):
    if set(external)!={"batch","history","query"}:raise ValueError("Only public batch, explicit history and query ports permitted")
    batch=external["batch"].value
    if not batch.cases:raise ValueError("Empty time series")
    require_regular_hourly(batch.cases)
    offset=0
    from ecodata.algorithms.mixed_gap_dag import missing_runs
    for case,bounds in zip(batch.cases,batch.offsets):
        expected=np.flatnonzero(~np.isfinite(case.values))
        if not np.array_equal(case.missing,expected):raise ValueError("Missing mask differs from actual hidden input")
        if np.isinf(case.values).any() or (case.values[np.isfinite(case.values)]<0).any():raise ValueError("PM2.5 input has illegal values")
        if bounds!=(offset,offset+len(expected)):raise ValueError("Invalid batch offsets")
        if not np.array_equal(batch.lengths[slice(*bounds)],missing_runs(case.values)[expected]):raise ValueError("Gap length metadata mismatch")
        offset+=len(expected)
    q=np.asarray(external["query"].value)
    if not len(q) or q.dtype.kind not in "iu" or len(np.unique(q))!=len(q) or q.min()<0 or q.max()>=offset:
        raise ValueError("Evaluation positions must be unique, actual missing-position indices")
    if len(batch.offsets)!=len(batch.cases) or offset!=len(batch.lengths):raise ValueError("Batch shape mismatch")
    for artifact in external.values():
        if artifact.metadata.get("unit")!="ug m-3" or artifact.metadata.get("contract")!=TASK:
            raise ValueError("Explicit supported unit ug m-3 and task port contract required")


def history_reason(batch,states,method):
    for case in batch.cases:
        state=states.get(case.station)
        if not state:return "No explicit historical training state for entity "+case.station
        end=state.get("training_end")
        try:
            if end is None or pd.Timestamp(end)>=pd.Timestamp(case.time[0]):return "Training end must precede processing input"
        except (ValueError,TypeError):return "Training clock is unknown or incompatible"
        if method=="hour_median":
            if np.shape(state.get("medians"))!=(24,) or not np.isfinite(state["medians"]).all():return "Missing finite 24-hour fitted medians"
        elif not hasattr(state.get("knn"),"predict"):return "Missing fitted calendar KNN state"
    return None


class ReferenceEvaluation:
    """Independent validation only. References never passed to nodes or decisions."""
    def __init__(self,truth,*,role="validation",codes=None,weights=None):
        if role not in {"validation","independent_validation"}:raise ValueError("Search requires validation, not final test")
        self.truth=np.array(truth,float,copy=True)
        if not len(self.truth) or not np.isfinite(self.truth).all() or (self.truth<0).any():raise ValueError("Finite nonnegative validation reference required")
        self.codes=np.zeros(len(self.truth),int) if codes is None else np.array(codes,int)
        if self.codes.shape!=self.truth.shape or self.codes.min()<0:raise ValueError("Invalid evaluation groups")
        self.counts=np.bincount(self.codes)
        self.weights=np.ones(1) if weights is None else np.array(weights,float)
        if len(self.weights)!=len(self.counts) or (self.counts==0).any() or not np.isclose(self.weights.sum(),1):raise ValueError("Invalid fixed evaluation weights")
        self.protocol=dict(kind="reference_error",role=role,objective="fixed weighted group RMSE",reference_sha256=hashlib.sha256(self.truth.tobytes()).hexdigest())
    def __call__(self,prediction):
        if prediction.shape!=self.truth.shape:raise ValueError("Reference/output queue mismatch")
        d=prediction-self.truth
        rmses=np.sqrt(np.bincount(self.codes,weights=d*d)/self.counts)
        return dict(objective=float(rmses@self.weights),RMSE=float(np.sqrt(np.mean(d*d))),MAE=float(np.abs(d).mean()),Bias=float(d.mean()),group_RMSE=rmses.tolist())


class CoverageEvaluation:
    protocol=dict(kind="declared_task_metric",objective="negative finite coverage",accuracy_claim=False)
    def __call__(self,prediction):
        return dict(objective=-float(np.isfinite(prediction).mean()),accuracy_assessed=False)


def submit_temporal_search(external,request,evaluate,config,folder,*,mode="C",seed=11,progress=None):
    """Compatibility client of the all-task public dispatcher."""
    from ecodata.algorithms.platform_search import submit_search
    return submit_search(external,dict(task="hourly_recovery",requirements=request),evaluate,folder,
        budget=config,mode=mode,seed=seed,progress=progress)


def _submit_hourly_service(external,request,evaluate,config,folder,*,mode="C",seed=11,progress=None):
    started=perf_counter()
    folder=Path(folder)
    if (folder/"summary.json").exists() or (folder/"request.json").exists():
        raise ValueError("Immutable request directory already exists; reuse its displayed result or create a new request id")
    try:
        result=_submit_temporal_search(external,request,evaluate,config,folder,mode=mode,seed=seed,progress=progress)
        result["service_total_seconds"]=perf_counter()-started
        save(folder/"summary.json",result)
        return result
    except Exception as exc:
        save(folder/"failure.json",dict(status="BLOCKED",error=str(exc),request=request,
            mode=mode,seed=seed,budget=config,reference_values_logged=False))
        raise


def _submit_temporal_search(external,request,evaluate,config,folder,*,mode="C",seed=11,progress=None):
    if request.get("operation")!="temporal_missing_recovery":raise ValueError("Only hourly PM2.5 temporal recovery currently supported")
    if evaluate is None:raise ValueError("Missing validation reference or explicitly declared task metric; no accuracy search can run")
    if getattr(evaluate,"protocol",{}).get("kind") not in {"reference_error","declared_task_metric"}:
        raise ValueError("Independent evaluator must explicitly declare reference_error or declared_task_metric")
    evaluator=TemporalEvaluator(external,evaluate,Path(folder)/"executions",request)
    if isinstance(evaluate,ReferenceEvaluation) and len(evaluate.truth)!=len(external["query"].value):raise ValueError("Reference queue mismatch")
    save(Path(folder)/"request.json",dict(request=request,budget=config,mode=mode,seed=seed,
        public_ports={k:dict(kind=v.kind,metadata=v.metadata,fingerprint=v.fingerprint()) for k,v in external.items()},
        evaluation=evaluator.evaluation_protocol,history_exclusions=evaluator.excluded,input_audit=evaluator.input_audit,
        effective_public_ports={k:v.fingerprint() for k,v in evaluator.external.items()},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name("numerical_search.py")]}))
    result=run_numerical_search(mode,seed,evaluator,config,folder,progress=progress)
    result["task_setup_seconds"]=evaluator.setup_seconds
    result["evaluation_protocol"]=evaluator.evaluation_protocol
    if result["best"]:
        best_folder=Path(folder)/"executions"/result["best"]["key"]
        with np.load(best_folder/"prediction.npz") as read:full=read["full_missing_prediction"]
        rows=[]
        for number,(case,(start,end)) in enumerate(zip(external["batch"].value.cases,external["batch"].value.offsets)):
            merged=case.values.copy();merged[case.missing]=full[start:end]
            rows.append(pd.DataFrame(dict(case=number,station=case.station,time=case.time,input_value=case.values,
                processed_value=merged,was_missing=~np.isfinite(case.values),unit="ug m-3")))
        product=pd.concat(rows,ignore_index=True);product.to_csv(Path(folder)/"processed.csv",index=False)
        check=pd.read_csv(Path(folder)/"processed.csv")
        np.testing.assert_allclose(check.processed_value,product.processed_value,equal_nan=True)
        result["product"]=dict(path="processed.csv",readback="PASS",rows=len(product),
            finite_coverage=float(np.isfinite(product.processed_value).mean()),
            note="Full processing product; remaining missing values are retained, not invented. Accuracy refers only to declared reference positions.")
    save(Path(folder)/"summary.json",result)
    return result


class TemporalEvaluator:
    def __init__(self,external,evaluate,folder,request):
        started=perf_counter()
        registry,self.factory=setup();self.pipeline=UnifiedAlgorithmPipeline(registry)
        self.external=external;self.evaluate=evaluate;self.folder=Path(folder);self.request=request
        validate_external(external)
        # JSON fitted-state arrays are storage adaptation, never a refit or a
        # change to the official predictor. Do not mutate the supplied state.
        supplied=external["history"]
        converted={entity:(dict(state,medians=readonly(np.asarray(state["medians"],float)))
            if isinstance(state,dict) and isinstance(state.get("medians"),list) else state)
            for entity,state in supplied.value.items()}
        if any(converted[k] is not supplied.value[k] for k in converted):
            self.external=dict(external,history=SealedArtifact("fitted_history",converted,supplied.metadata))
        from ecodata.recognition import inspect_dataset
        from ecodata.evaluation.quality_gates import RawInputQualityGate
        self.input_audit=[]
        for case in external["batch"].value.cases:
            dataset=xr.Dataset({"pm25":("time",case.values)},coords={"time":case.time})
            dataset.pm25.attrs.update(units="ug m-3",data_kind="continuous",valid_min=0)
            quality=RawInputQualityGate().validate(dataset,variable="pm25",requires_spatial=False,expected_unit="ug m-3")
            if quality["status"]=="BLOCK":raise ValueError("Raw input quality BLOCK: "+str(quality["checks"]))
            self.input_audit.append(dict(descriptor=inspect_dataset("parsed_hourly_series",dataset).as_dict(),quality=quality))
        self.excluded={}
        states=self.external["history"].value
        for method in ("hour_median","calendar_knn"):
            reason=history_reason(external["batch"].value,states,method)
            if reason:
                identifier="baseline.temporal_"+method
                self.excluded[identifier]=reason
        if self.excluded:
            self.factory=MixedGapFactory({k:v for k,v in self.factory.adapters.items() if k[0] not in self.excluded},IDS,max_nodes=18,max_depth=9)
        self.node_hits=0;self.node_new=0;self.kernel_hits=0
        self._decision=None
        self.evaluation_protocol=dict(getattr(evaluate,"protocol",{}))
        self.setup_seconds=perf_counter()-started
    def decision(self):
        # Format/metadata adaptation only. Selection policy lives in the shared
        # extractor, rule mapper, Registry contracts and AlgorithmSelector.
        if self._decision is not None:return self._decision
        batch=self.external["batch"].value;case=batch.cases[0]
        ds=xr.Dataset({"pm25":("time",case.values)},coords={"time":case.time})
        ds.pm25.attrs.update(units="ug m-3",data_kind="continuous",valid_min=0)
        semantics=SemanticFeatureExtractor().extract_series(batch.cases,variable="pm25",unit="ug m-3",nonnegative=True)
        declarations=self.request.get('feature_declarations',[])
        if any(d.get('source_id')!='role:batch:pm25' for d in declarations):raise ValueError('Unknown temporal batch declaration identity')
        semantics,target,audit=SemanticFeatureExtractor.apply_declarations(semantics,self.request,declarations,source_id='role:batch:pm25')
        checks=[check for audit in self.input_audit for check in audit["quality"]["checks"]]
        context=AlgorithmContext((EcoDataBundle(ds),),"pm25","continuous",input_quality=dict(status="WARN" if any(a["quality"]["status"]=="WARN" for a in self.input_audit) else "PASS",checks=checks))
        self._decision=AlgorithmSelector(self.pipeline.registry).decide(context,target,semantics=semantics)
        self._decision.feature_declarations=audit
        from ecodata.algorithms.capability_scoring import configure_optional_scoring
        configure_optional_scoring(self._decision,self.request,self.evaluation_protocol,None)
        for identifier,reason in self.excluded.items():
            self._decision.matches[identifier].update(eligible=False,reasons=[reason])
        return self._decision
    def execute(self,graph,key,*,persist=True):
        start=perf_counter();cpu=process_time();trace=[];full_prediction=None
        try:
            assert set(self.external)=={"batch","history","query"}
            # Final task acceptance is shared and unconditional across search
            # arms; only posterior *early* filtering is an experimental switch.
            result=self.pipeline.run_dag(graph,task_id=TASK,external=self.external,version=TASK,
                decision=self.decision(),factory=self.factory)
            trace=result["trace"];artifact=result["outputs"]["field"];raw,prediction=artifact.value
            root=graph.root_node;index,output=root.parameters["_port_bindings"]["field"]
            parent=root.nodes_from[index];identifier="n"+str(list(graph.nodes).index(parent))
            full_prediction=np.maximum(result["intermediates"][identifier][output].value,0)
            metrics=self.evaluate(readonly(prediction))
            if not np.isfinite(metrics["objective"]):raise ValueError("Evaluator returned nonfinite objective")
            info=dict(status="PASS",**metrics,coverage=1.,failure_count=0,quality=artifact.metadata,
                denominator=len(prediction),prediction_fingerprint=hashlib.sha256(np.round(prediction,8).tobytes()).hexdigest(),
                negative_count=artifact.metadata["negative_projection_count"])
            info.update(task_acceptance=result["task_acceptance"],numerical_execution_attempted=True)
        except Exception as exc:
            details=getattr(exc,"details",{})
            prediction=details.get("projected_prediction",np.full(len(self.external["query"].value),np.nan));raw=details.get("raw_prediction",prediction.copy());trace=getattr(exc,"trace",trace)
            finite=np.isfinite(prediction)
            info=dict(status="FAIL",objective=None,coverage=float(finite.mean()),denominator=len(self.external["query"].value),failure_count=int((~finite).sum()),
                error=str(exc),call_chain=traceback.format_exc(),partial_error_not_ranked=True)
            info.update(task_acceptance=details.get("task_acceptance"),
                numerical_execution_attempted="task_acceptance" not in details,
                failure_stage="task_requirements" if "task_acceptance" in details else "execution_or_quality")
        for n in trace:
            self.node_hits+=int(n.get("cache_hit",False));self.node_new+=int(n.get("status")=="PASS" and not n.get("cache_hit",False))
            self.kernel_hits+=sum(bool(v.get("metadata",{}).get("kernel_cache_hit")) for v in n.get("outputs",{}).values())
        info.update(key=key,graph=record(graph),seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,
            nodes_cache_hits=sum(n.get("cache_hit",False) for n in trace),node_trace=trace,hidden_truth_passed_to_nodes=False)
        if persist:
            folder=self.folder/key;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/"prediction.npz",raw=raw,prediction=prediction,
                full_missing_prediction=full_prediction if full_prediction is not None else np.full(len(self.external["batch"].value.lengths),np.nan))
            with np.load(folder/"prediction.npz") as read:np.testing.assert_array_equal(read["prediction"],prediction)
            info["file_readback"]="PASS";info["prediction_file_sha256"]=hashlib.sha256((folder/"prediction.npz").read_bytes()).hexdigest()
            save(folder/"result.json",info)
        return info


