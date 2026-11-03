"""Public station/AOD binding of existing frozen PyKrige/residual DAG kernels."""
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS,Transformer
from ecodata.algorithms.dag import Artifact,Port,NodeAdapter,DAGError
from ecodata.algorithms.frozen_station_dag import FrozenStationFusionDAG,FrozenStationGraphFactory
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.algorithms.contracts import AlgorithmContext,Compatibility
from ecodata.core.schemas import EcoDataBundle
from ecodata.io.structures import table_frame
from ecodata.evaluation.quality_gates import RawInputQualityGate,ResultQualityGate

TASK="platform_station_covariates_v2"


def prepare_station_search(parsed,request):
    tables={role:table_frame(ds).copy() for role,ds in parsed}
    if set(tables)!={"observations","query","covariates"}:raise ValueError("Station/AOD task needs exactly observations, query and covariates inputs")
    train,query,cov=tables["observations"],tables["query"],tables["covariates"]
    if "value" in query or "truth" in query or "split" in query:raise ValueError("Query must not carry any held-out target labels")
    if not {"station_id","longitude","latitude","value","time","unit","split"}<=set(train):raise ValueError("Incomplete training observation schema")
    if not train.split.eq("train").all():raise ValueError("Only training-station labels are permitted")
    if not {"station_id","time","aod"}<=set(cov):raise ValueError("Explicit dimensionless AOD data required")
    if request.get("covariate_unit")!="1" or request.get("unit")!="ug m-3":raise ValueError("AOD is dimensionless, not ground concentration")
    if request.get("crs")!="EPSG:4326" or request.get("metric_crs")!="EPSG:32651":raise ValueError("Existing Taiwan task uses explicit EPSG:4326 working convention and EPSG:32651 metric distances")
    if not request.get("coordinate_evidence") or not request.get("time_semantics"):raise ValueError("Coordinate/time evidence or explicit working convention required")
    for table in tables.values():
        table["time"]=pd.to_datetime(table.time,errors="raise")
        if table.duplicated(["station_id","time"]).any():raise ValueError("Duplicate station/time input")
    if train.time.nunique()!=1 or query.time.nunique()!=1:raise ValueError("This task adapter processes one explicit observation window per request")
    stamp=train.time.iloc[0]
    if not query.time.eq(stamp).all():raise ValueError("Training/query time windows differ")
    meta=dict(unit="ug m-3",metric_crs="EPSG:32651",time=str(stamp),contract=TASK,
        coordinate_evidence=request["coordinate_evidence"],time_semantics=request["time_semantics"])
    external={key:Artifact(kind,table,dict(meta)) for key,kind,table in
              [("observations","observed_station_labels",train),("query","query_positions",query),("covariates","station_covariates",cov)]}
    tx=Transformer.from_crs(request["crs"],request["metric_crs"],always_xy=True)
    def xy(t):return np.column_stack(tx.transform(t.longitude.to_numpy(),t.latitude.to_numpy()))
    def quality(inputs,p,e):
        raw=inputs["prediction"].value
        if not np.isfinite(raw).all():raise DAGError("Nonfinite fixed prediction cohort")
        values=np.maximum(raw,0) if request.get("nonnegative_action")=="project" else raw
        source=xr.Dataset({"pm25":("point",values)},attrs={"crs":request["crs"]});source.pm25.attrs["units"]="ug m-3"
        report=ResultQualityGate().validate(source,source,variable="pm25",semantic_type="continuous",nonnegative=True,expected_output_unit="ug m-3")
        if report["status"]=="BLOCK":raise DAGError("ResultQualityGate BLOCK: "+str(report))
        report.update(negative_projection_count=int((raw<0).sum()) if request.get("nonnegative_action")=="project" else 0)
        return {"prediction":Artifact("prediction",values),"raw_prediction":Artifact("raw_prediction",raw),"quality":Artifact("quality",report)}
    def raw_quality(t):
        ds=xr.Dataset({"pm25":("point",t.value.to_numpy())},coords={"longitude":("point",t.longitude.to_numpy()),"latitude":("point",t.latitude.to_numpy())},attrs={"crs":request["crs"]})
        ds.pm25.attrs["units"]="ug m-3"
        return RawInputQualityGate().validate(ds,variable="pm25",requires_spatial=True,expected_unit="ug m-3")
    backend=FrozenStationFusionDAG(xy=xy,quality=quality,input_quality=raw_quality)
    adapters=backend.adapters()
    # AOD is a genuine second data branch, not a synthetic column or a target.
    adapters.pop(("source.train","run"));adapters.pop(("source.query","run"))
    for key in external:
        def read(i,p,e,key=key):return {"value":e[key]}
        adapters["source."+key,"run"]=NodeAdapter({}, {"value":Port(external[key].kind)},read,decision_contract=dict(operation="source",capabilities=[]))
    def join(role):
        def run(inputs,p,e):
            a=inputs["table"];features=inputs["covariates"].value
            if not features.time.eq(stamp).all():raise DAGError("Covariate time window mismatch")
            table=a.value.merge(features[["station_id","time","aod"]],on=["station_id","time"],how="left",validate="one_to_one")
            # Preserve the existing frozen retrieval contract: finite screened
            # AOD can include small negative retrievals. PM2.5 is the constrained
            # target; do not silently clip or invent an extra AOD threshold.
            if len(table)!=len(a.value) or not np.isfinite(table.aod).all():raise DAGError("Missing/illegal AOD; no cohort deletion")
            value=Artifact("training" if role=="train" else "query",table,dict(meta))
            return backend.source({},dict(source=role),{role:value})
        return run
    adapters["prepare.join_train","run"]=NodeAdapter({"table":Port("observed_station_labels"),"covariates":Port("station_covariates")},{"table":Port("training")},join("train"),decision_contract=dict(operation="compute",registered=False,capabilities=[]))
    adapters["prepare.join_query","run"]=NodeAdapter({"table":Port("query_positions"),"covariates":Port("station_covariates")},{"table":Port("query")},join("query"),decision_contract=dict(operation="compute",registered=False,capabilities=[]))
    # Actual preflight of the same source/geometry contracts used at execution.
    a=join("train")({"table":external["observations"],"covariates":external["covariates"]},{},{})["table"]
    b=join("query")({"table":external["query"],"covariates":external["covariates"]},{},{})["table"]
    backend.check(dict(train=a,query=b))
    original_finish=adapters["quality.station","run"].execute
    def finish(inputs,p,e):
        value=original_finish(inputs,p,e)["field"].value
        q=inputs["query"].value
        ds=xr.Dataset({"pm25":("point",value["prediction"])},coords={"longitude":("point",q.longitude.to_numpy()),"latitude":("point",q.latitude.to_numpy()),"station_id":("point",q.station_id.astype(str).to_numpy())},attrs={"crs":request["crs"],"coordinate_evidence":request["coordinate_evidence"],"time_window":str(stamp),"time_semantics":request["time_semantics"]})
        ds.pm25.attrs.update(units="ug m-3",data_kind="continuous")
        return {"field":Artifact("validated_field",ds,dict(meta,quality=value["quality"]))}
    adapters["quality.station","run"].execute=finish
    registry=create_default_registry()
    ids=sorted({key[0] for key in adapters if key[0] in {s.id for s in registry.specs()}})
    for key,adapter in adapters.items():
        if key[0].startswith(("source.","prepare.")):continue
        if key[0]=="quality.station":adapter.decision_contract=dict(operation="validate",input="prediction",capabilities=["nonnegative_validation"])
        elif key[1]=="fit":adapter.decision_contract=dict(operation="fit",data_port="train",capabilities=["fitted_state"])
        else:adapter.decision_contract=dict(operation="compute",state_port="state" if "state" in adapter.inputs else None,
            input="background" if key[1]=="sum" else "state" if key[1]=="background" else None)
    boundaries=[k[0] for k in adapters if k[0].startswith(("source.","prepare.","quality."))]
    registry.register_task_adapters(TASK,adapters,algorithm_ids=ids,boundary_ids=boundaries)
    factory=FrozenStationGraphFactory(adapters,ids,max_nodes=24,max_depth=9)
    factory.task_id=TASK
    factory.SOURCES={"source.observations":"observations","source.query":"query","source.covariates":"covariates"}
    factory.resource_contracts={key:dict(source_id="role:"+key+":"+("aod" if key=="covariates" else "pm25"),role=key,variable="aod" if key=="covariates" else "pm25") for key in external}
    ds=xr.Dataset({"pm25":("station",train.value.to_numpy())},coords={"longitude":("station",train.longitude.to_numpy()),"latitude":("station",train.latitude.to_numpy())},attrs={"crs":request["crs"]})
    ds.pm25.attrs.update(units="ug m-3",data_kind="continuous",valid_min=0)
    context=AlgorithmContext((EcoDataBundle(ds),),"pm25","continuous")
    semantics=SemanticFeatureExtractor().extract(ds,"pm25",confirmed=dict(nonnegative=True,variable_class="continuous"))
    decision=AlgorithmSelector(registry).decide(context,dict(operation="station_to_points",source_count=1,covariate_estimation=True,source_roles=[role for role,_ in parsed],working_assumptions=request.get("working_assumptions",[])),semantics=semantics,use_cache=True,
        adapter_compatibility={identifier:Compatibility(True,(),("Actual task source/geometry contracts validated; scientific state ownership rechecked during execution",)) for identifier in ids})
    excluded={identifier:match["reasons"] for identifier,match in decision.matches.items() if not match["eligible"]}
    return registry,factory,external,decision,excluded
