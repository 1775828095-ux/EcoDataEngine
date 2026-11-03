"""Scoped, grouped development case memory supplies state effects, never ranks."""
from pathlib import Path
import json
import numpy as np
from ecodata.algorithms.attribute_registry import identifier,digest

ROOT=Path(__file__).resolve().parents[3]/"data/portrait_memories"


def scope(tables):
    model=tables["model"]
    context={k:v for k,v in tables["task_context"].items() if k!="state_policy_context"}
    return json.loads(json.dumps(dict(model={k:model[k] for k in ("model_id","version","selected_role","selected_field")},
        context=context,pool=tables["candidate_pool"]["fingerprint"]),sort_keys=True))


def load(version,tables=None):
    envelope=json.loads((ROOT/(identifier(version)+".json")).read_text(encoding="utf-8"))
    if set(envelope)!={"snapshot","sha256"} or digest(envelope["snapshot"])!=envelope["sha256"]:
        raise ValueError("Case memory fingerprint mismatch")
    result=dict(envelope["snapshot"],sha256=envelope["sha256"])
    if tables is not None and result["scope"]!=scope(tables):raise ValueError("Case memory outside model/field/task/pool support")
    return result


def fit(folders,version,*,holdout_groups,features,min_groups=2,max_distance=1.):
    identifier(version)
    if not features or len(set(features))!=len(features) or len(features)>6 or not 2<=min_groups<=20 or not 0<max_distance<=10:
        raise ValueError("Declare 1..6 measured features, >=2 independent groups and bounded similarity radius")
    records=[];transitions=[];transition_seen=set();expected=None;sources={};seen=set();held=set(holdout_groups)
    if not held:raise ValueError("Separate development holdout groups required")
    from ecodata.algorithms.profile_matching import fingerprint_bytes
    for folder in folders:
        folder=Path(folder);report=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        if report.get("evaluation_protocol",{}).get("study_partition") in {"test","final_test"}:
            raise ValueError("Final test cannot be relabeled as development memory")
        if report.get("evidence_role")!="development" or not report.get("group") or report.get("stop_reason")!="CANDIDATE_POOL_COMPLETED":
            raise ValueError("Only complete grouped development runs may enter memory; never final test")
        tables=report["portraits"];current=scope(tables)
        if expected is not None and current!=expected:raise ValueError("Mixed memory scopes")
        expected=current
        identities=tuple(d.get("scientific_content_hash",d["content_hash"]) for d in tables["data"])
        for identity in identities:
            if identity in sources and sources[identity]!=report["group"]:raise ValueError("Same visible source cannot cross development groups")
            sources[identity]=report["group"]
        for row in report["rows"]:
            if row["matching"].get("version")!="joint_state_v1":raise ValueError("Missing actual joint-state instrumentation")
            baseline=row["matching"]["baseline"]
            if any(not isinstance(baseline["features"].get(k),(int,float)) or not np.isfinite(baseline["features"][k]) for k in features):
                raise ValueError("Unmeasured memory feature; do not fill unknown with zero")
            token=(identities,row["key"])
            if token in seen:continue
            seen.add(token)
            after=None
            if row["status"]=="PASS":
                import xarray as xr
                if row.get("file_readback")!="PASS" or not row.get("numerical_execution_attempted") or fingerprint_bytes(row["product"])!=row.get("product_sha256"):
                    raise ValueError("Memory requires real product readback and correct fingerprint")
                states=[a["metadata"]["joint_state"] for n in row["node_trace"] for a in n.get("outputs",{}).values()
                        if a.get("kind")=="validated_field" and "joint_state" in a.get("metadata",{})]
                if not states:raise ValueError("No final measured state in numerical trace")
                after=states[-1]
                for node in row["node_trace"]:
                    for artifact in node.get("outputs",{}).values():
                        transition=artifact.get("metadata",{}).get("joint_transition")
                        if not transition or transition["operation"].startswith(("source.","prepare.","quality.")):
                            continue
                        parents=[p for p in transition["parents"].values() if p is not None]
                        if len(parents)!=1 or any(k not in parents[0]["features"] for k in features):continue
                        identity=digest(dict(operation=transition["operation"],mode=transition["mode"],
                            parameters=transition["parameters"],inputs=transition["input_fingerprints"]))
                        if identity in transition_seen:continue
                        transition_seen.add(identity)
                        parent=parents[0]
                        transitions.append(dict(group=report["group"],held=report["group"] in held,
                            key=transition_key(transition["operation"],transition["mode"],transition["parameters"]),
                            operation=transition["operation"],mode=transition["mode"],status="PASS",
                            features={k:parent["features"][k] for k in features},all_features=parent["features"],
                            after_features=transition["after"]["features"],before=parent["metrics"],after=transition["after"]["metrics"],
                            summary=str(folder/"summary.json"),fingerprint=identity))
                with xr.open_dataset(row["product"]) as ds:
                    a=np.asarray(ds[report["evaluation_protocol"]["variable"]],float)
                    if abs(1-np.isfinite(a).mean()-after["metrics"]["missing_fraction"])>1e-10:
                        raise ValueError("Measured state differs from saved product")
            records.append(dict(group=report["group"],held=report["group"] in held,key=row["key"],status=row["status"],
                features={k:baseline["features"][k] for k in features},before=baseline["metrics"],
                after=after["metrics"] if after else None,steps=row["steps"],summary=str(folder/"summary.json"),
                summary_sha256=fingerprint_bytes(folder/"summary.json"),product=row.get("product"),product_sha256=row.get("product_sha256")))
    training=[r for r in records if not r["held"]];testing=[r for r in records if r["held"]]
    if len({r["group"] for r in training})<min_groups or {r["group"] for r in testing}!=held:
        raise ValueError("Insufficient independent training/holdout groups")
    ranges={k:[float(min(r["features"][k] for r in training)),float(max(r["features"][k] for r in training))] for k in features}
    snapshot=dict(version=version,scope=expected,features=features,ranges=ranges,min_groups=min_groups,max_distance=max_distance,
        records=training,holdout_records=len(testing),sources=sorted({r["summary"] for r in records}),
        learned="Scoped algorithm and whole-chain metric changes; no task-loss/rank predictor",holdout_predictions=[],
        transitions=[r for r in transitions if not r["held"]])
    for r in testing:
        prediction=estimate(snapshot,r["key"],dict(features=r["features"],metrics=r["before"]))
        snapshot["holdout_predictions"].append(dict(summary=r["summary"],key=r["key"],status=prediction["status"],
            errors={k:prediction["metrics"][k][1]-v for k,v in (r["after"] or {}).items()
                    if v is not None and prediction.get("metrics",{}).get(k) is not None} if prediction["status"]=="SUPPORTED" else {}))
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/(version+".json")).open("x",encoding="utf-8") as stream:
        json.dump(dict(snapshot=snapshot,sha256=digest(snapshot)),stream,ensure_ascii=False,indent=2,allow_nan=False)
    return load(version)


def estimate(memory,key,baseline):
    feature=baseline["features"]
    for name,(low,high) in memory["ranges"].items():
        value=feature.get(name)
        if value is None or not np.isfinite(value) or not low-1e-9<=value<=high+1e-9:
            return dict(status="UNKNOWN",reason="Outside measured feature support: "+name)
    nearest=[]
    for r in memory["records"]:
        if r["key"]!=key:continue
        distance=float(np.sqrt(np.mean([((feature[k]-r["features"][k])/max(memory["ranges"][k][1]-memory["ranges"][k][0],1e-8))**2 for k in memory["features"]])))
        if distance<=memory["max_distance"]:nearest.append((distance,r))
    if len({r["group"] for _,r in nearest})<memory["min_groups"] or any(r["status"]!="PASS" for _,r in nearest):
        return dict(status="UNKNOWN",reason="Insufficient independent supported cases or recorded failure")
    metrics={};delta_bounds={}
    for name,before in baseline["metrics"].items():
        if before is None:continue
        changes=[(distance,r["after"][name]-r["before"][name]) for distance,r in nearest
                 if r["after"].get(name) is not None and r["before"].get(name) is not None]
        if len(changes)!=len(nearest):continue
        weights=np.asarray([1/(1+d) for d,_ in changes]);delta=np.asarray([v for _,v in changes])
        # Bounds describe observed conditional changes, not a statistical CI.
        metrics[name]=[max(0.,before+float(delta.min())),max(0.,before+float(np.average(delta,weights=weights))),max(0.,before+float(delta.max()))]
        delta_bounds[name]=[float(delta.min()),float(delta.max())]
    predicted_features={}
    for name,before in feature.items():
        changes=[(distance,r["after_features"][name]-r["all_features"][name]) for distance,r in nearest
                 if name in r.get("after_features",{}) and name in r.get("all_features",{})]
        if len(changes)==len(nearest):
            predicted_features[name]=float(before+np.average([v for _,v in changes],weights=[1/(1+d) for d,_ in changes]))
    return dict(status="SUPPORTED" if metrics else "UNKNOWN",metrics=metrics,delta_bounds=delta_bounds,features=predicted_features,
        groups=sorted({r["group"] for _,r in nearest}),evidence=[dict(summary=r["summary"],key=r["key"],distance=d) for d,r in nearest],
        uncertainty="Observed conditional effect envelope; no out-of-domain coverage guarantee")


def transition_key(operation,mode,parameters):
    return operation+"/"+mode+"/"+digest({k:v for k,v in parameters.items() if k not in {"_port_bindings","mode"}})


def estimate_transition(memory,operation,mode,parameters,parent):
    key=transition_key(operation,mode,parameters)
    records=[r for r in memory.get("transitions",[]) if r["key"]==key]
    if not records:return dict(status="UNKNOWN",reason="No measured algorithm response in this mode")
    ranges={k:[min(r["features"][k] for r in records),max(r["features"][k] for r in records)] for k in memory["features"]}
    return estimate(dict(memory,records=records,ranges=ranges),key,parent)
