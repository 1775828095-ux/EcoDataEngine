"""Immutable, validated attribute definitions and bounded scoring parameters."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "data/attribute_registries"
BUILTIN = Path(__file__).with_name("portrait_attributes_v1.json")
PARAMETERS = {"multiplier": (0., 2., 1.), "activation_threshold": (0., 1., 0.)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}",value):
        raise ValueError("Invalid immutable attribute/calibration version")
    return value


def validate(snapshot):
    if set(snapshot)!={"schema_version","version","parent_sha256","created_at","evidence","attributes"} or snapshot["schema_version"]!=1:
        raise ValueError("Unsupported attribute registry schema")
    identifier(snapshot["version"])
    if not isinstance(snapshot['created_at'],str):raise ValueError('Attribute registry creation date required')
    datetime.fromisoformat(snapshot['created_at'])
    if not isinstance(snapshot["evidence"],str) or not snapshot["evidence"].strip():
        raise ValueError("Attribute registry requires evidence")
    attributes=snapshot["attributes"]
    if not isinstance(attributes,list) or not 1<=len(attributes)<=64:
        raise ValueError("Attribute registry requires 1..64 definitions")
    seen=set()
    for entry in attributes:
        required={"id","definition","tables","unit","role","scoring_enabled","hard_allowed","provider","property","activation","evidence","history"}
        if not isinstance(entry,dict) or set(entry)!=required:
            raise ValueError("Incomplete or unsupported attribute definition")
        identifier(entry["id"])
        if entry["id"] in seen:raise ValueError("Duplicate attribute ID")
        seen.add(entry["id"])
        if entry["role"] not in {"scoring","diagnostic"} or type(entry["scoring_enabled"]) is not bool or type(entry["hard_allowed"]) is not bool:
            raise ValueError("Invalid attribute role or enable flag")
        if entry["role"]=="diagnostic" and (entry["scoring_enabled"] or entry["hard_allowed"]):
            raise ValueError("Diagnostic attributes cannot silently become scoring/hard consumers")
        if not isinstance(entry["tables"],list) or not entry["tables"] or not set(entry["tables"])<={"data","algorithm","model"}:
            raise ValueError("Attribute must declare its portrait tables")
        for key in ("definition","unit","evidence"):
            if not isinstance(entry[key],str) or not entry[key].strip():raise ValueError("Attribute requires definition, unit and evidence")
        if not isinstance(entry["history"],list) or not entry["history"] or any(not isinstance(v,str) or not v.strip() for v in entry["history"]):
            raise ValueError("Attribute revisions require an audit history")
        if entry["provider"] not in {"guarantee","declared_cost","unavailable","station_probe","state_metric"}:
            raise ValueError("No registered measurement/scoring provider; arbitrary expressions are forbidden")
        if entry["activation"] not in {"variation","negative_fraction","constant_one"}:
            raise ValueError("No registered data activation provider")
        if entry["provider"]=="guarantee":
            if entry["property"] not in core_ids():raise ValueError("No reviewed property transfer consumer")
        elif entry["provider"]=="state_metric":
            from ecodata.algorithms.portrait_state import METRICS
            if entry["property"] not in METRICS or entry["hard_allowed"] or entry["activation"]!="constant_one":
                raise ValueError("State metrics require a registered measurement, never a hard guarantee")
        elif entry["provider"]=="station_probe":
            if entry["property"] not in {"RMSE","MAE","abs_bias"} or entry["hard_allowed"] or entry["activation"]!="constant_one":
                raise ValueError("Station probes expose bounded empirical errors, never hard guarantees")
        elif entry["property"] is not None or entry["hard_allowed"]:
            raise ValueError("Unmeasured/empirical attributes cannot establish hard guarantees")
    # Disabling a soft term cannot erase the canonical hard-rule vocabulary.
    current={entry["id"]:entry for entry in attributes}
    for entry in builtin()["attributes"]:
        value=current.get(entry["id"])
        if value is None or any(value[k]!=entry[k] for k in ("provider","property","activation","hard_allowed","role")):
            raise ValueError("Protected hard attribute cannot be removed or redefined")
    return deepcopy(snapshot)


def builtin():
    return json.loads(BUILTIN.read_text(encoding="utf-8"))


def core_ids():
    return tuple(entry["id"] for entry in builtin()["attributes"])


def load(version=None):
    version=identifier(version or "builtin-v1")
    path=BUILTIN if version=="builtin-v1" else ROOT/(version+".json")
    envelope=json.loads(path.read_text(encoding="utf-8"))
    if version=="builtin-v1":snapshot=envelope
    else:
        if set(envelope)!={"snapshot","sha256"} or digest(envelope["snapshot"])!=envelope["sha256"]:
            raise ValueError("Attribute registry fingerprint mismatch")
        snapshot=envelope["snapshot"]
    snapshot=validate(snapshot)
    if snapshot["version"]!=version:raise ValueError("Attribute registry version mismatch")
    return dict(snapshot,sha256=digest(snapshot))


def publish(version, attributes, *, parent_version="builtin-v1", evidence):
    identifier(version)
    if version=="builtin-v1":raise ValueError("Builtin registry is immutable")
    parent=load(parent_version)
    snapshot=validate(dict(schema_version=1,version=version,parent_sha256=parent["sha256"],created_at=datetime.now(timezone.utc).isoformat(),evidence=evidence,attributes=attributes))
    old={a["id"]:a for a in parent["attributes"]};new={a["id"]:a for a in attributes}
    # Keep tombstones: remove optional attributes from scoring, not provenance.
    if not set(old)<=set(new):raise ValueError("Retain retired attributes with scoring_enabled=false")
    for key,entry in new.items():
        if key in old and entry!=old[key] and (entry["history"]==old[key]["history"] or entry["history"][:len(old[key]["history"])]!=old[key]["history"]):
            raise ValueError("Changed attributes must append a revision reason")
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/(version+".json")
    with path.open("x",encoding="utf-8") as stream:
        json.dump(dict(snapshot=snapshot,sha256=digest(snapshot)),stream,ensure_ascii=False,indent=2,allow_nan=False)
    return load(version)


def parameters(value, registry):
    if not isinstance(value,dict):raise ValueError("Scoring parameters require an object")
    attributes={a["id"]:a for a in registry["attributes"]}
    result={}
    for key,settings in value.items():
        if key not in attributes or attributes[key]["role"]!="scoring":raise ValueError("Unknown scoring parameter attribute")
        if not isinstance(settings,dict) or set(settings)-set(PARAMETERS):raise ValueError("Unknown scoring parameter")
        result[key]={}
        for name,v in settings.items():
            lo,hi,default=PARAMETERS[name]
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or not lo<=v<=hi:
                raise ValueError("Scoring parameter outside reviewed bounds")
            result[key][name]=float(v)
    return result


def configuration(request):
    settings=request.get("scoring",{})
    if not isinstance(settings,dict) or set(settings)-{"registry_version","parameters","calibration_id"}:
        raise ValueError("Unknown joint scoring configuration")
    registry=load(settings.get("registry_version"))
    if settings.get("calibration_id") and settings.get("parameters"):
        raise ValueError("Do not silently override a calibrated parameter snapshot")
    return registry,parameters(settings.get("parameters",{}),registry)
