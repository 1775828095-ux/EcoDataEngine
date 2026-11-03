"""Explainable S(D) -> R mapping for physics-aware algorithm search."""
from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.evaluation.quality_gates import RawInputQualityGate
from ecodata.recognition import inspect_dataset


UNKNOWN = "UNKNOWN"


def series_time_axis(case):
    """Normalize storage units, not timezone or observation-window meaning."""
    if np.asarray(case.time).dtype.kind in "ifu":
        raise ValueError("Time labels require explicit datetimes, not undeclared numeric epochs")
    times = pd.DatetimeIndex(case.time).as_unit("ns")
    if times.hasnans:
        raise ValueError("Missing time labels are not a valid time axis")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Unique increasing time labels required")
    if np.asarray(case.values).ndim != 1 or len(times) != len(case.values):
        raise ValueError("Time labels and one-dimensional values must have equal lengths")
    delta = np.diff(times.asi8)
    step = int(delta[0]) if len(delta) and np.all(delta == delta[0]) else None
    return times, step


def require_regular_hourly(cases):
    """Contract of the existing record-count gap splitter; no new resampler."""
    for index, case in enumerate(cases):
        _, step = series_time_axis(case)
        if step != 3600 * 10**9:
            raise ValueError(f"Gap splitter requires regular 1-hour labels; series {index}: "
                             f"interval_ns={step!r} (None means irregular/insufficient labels)")


@dataclass(frozen=True)
class EvidenceValue:
    value: Any
    evidence: tuple[str, ...]
    status: str = "CONFIRMED"
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataSemantics:
    structure: EvidenceValue
    variable_class: EvidenceValue
    spatial: EvidenceValue
    temporal: EvidenceValue
    unit: EvidenceValue
    physics: EvidenceValue
    quality: EvidenceValue
    variable: str
    observed: EvidenceValue = field(default_factory=lambda: EvidenceValue({}, ("not extracted",), UNKNOWN))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityRequirements:
    required: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    soft_objectives: tuple[str, ...]
    unknowns: tuple[str, ...]
    trace: tuple[dict[str, Any], ...]
    problems: tuple[dict[str, Any], ...] = ()
    optional_capabilities: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticFeatureExtractor:
    _cache = OrderedDict()

    @staticmethod
    def raster_features(dataset, variable, target_size):
        """Observable shape statistics shared with the existing raster router."""
        from scipy.ndimage import maximum_filter
        a=np.asarray(dataset[variable].values);n=a.shape[0]
        if a.ndim!=2 or min(a.shape)<3 or not np.isfinite(a).all():
            return {}
        spread=max(float(a.std()),1e-8);span=max(float(np.ptp(a)),1e-8)
        dy,dx=np.gradient(a);g=np.sqrt(dx**2+dy**2);curvature=np.diff(a,2,0)
        extrema=(a==maximum_filter(a,size=3))&(g>span*1e-8)
        count=max(1,int(g.size*.1));top=np.partition(g.ravel(),-count)[-count:].sum()
        names=('mean','std','range','gradient_mean','gradient_p95','flat_fraction','edge_concentration',
               'curvature_mean','extrema_fraction','max_median_ratio','unique_fraction','scale')
        return dict(zip(names,[float(a.mean()),float(a.std()),float(np.ptp(a)),float(g.mean()*n/spread),
            float(np.quantile(g,.95)*n/spread),float((g<span*1e-8).mean()),float(top/max(g.sum(),1e-12)),
            float(abs(curvature).mean()*n*n/spread),float(extrema.mean()),float((a.max()-np.median(a))/spread),
            float(len(np.unique(np.round(a/span,8)))/a.size),target_size/n]))

    @staticmethod
    def check_declarations(semantics, target, declarations, *, source_id):
        """Declarations confirm public facts, never override observations or choose an algorithm.

        Unknown facts remain UNKNOWN: metadata must be supplied through the
        existing reader declaration protocol before numerical execution.
        """
        allowed={'variable_class':(semantics.variable_class.value,'1'),
            'unit':(semantics.unit.value,'unit_symbol'),
            'temporal_semantics':(semantics.temporal.value.get('semantics',UNKNOWN),'1'),
            'spatial_support':(semantics.spatial.value.get('support',UNKNOWN),'1'),
            'grid_change':(target.get('grid_change',UNKNOWN),'1'),
            'support_conditions':(semantics.observed.value.get('support_conditions',UNKNOWN),'records')}
        allowed.update({k:(v,'1') for k,v in semantics.observed.value.items() if k in {'missing_count','missing_fraction','finite_count','regular_hourly'}})
        audit=[]
        for item in declarations:
            if set(item)-{'feature','value','unit','source_id','status','confirmed','evidence'}:
                raise ValueError('Unsupported feature declaration field; no algorithm or graph declarations')
            if item.get('source_id')!=source_id:continue
            name=item.get('feature')
            if name not in allowed:raise ValueError('Unsupported declared feature: '+str(name))
            if not item.get('confirmed') or not item.get('evidence') or item.get('status') not in {'CONFIRMED','UNKNOWN'}:
                raise ValueError('Feature declaration requires confirmation, status and evidence')
            actual,unit=allowed[name]
            if item.get('unit')!=unit:raise ValueError('Feature declaration unit conflict: '+name)
            status='UNKNOWN' if (isinstance(actual,str) and actual==UNKNOWN) or item['status']=='UNKNOWN' else 'CONFIRMED'
            if status=='CONFIRMED':
                value=item.get('value')
                equal=np.isclose(actual,value,rtol=1e-9,atol=1e-12) if isinstance(actual,(int,float)) and not isinstance(actual,bool) and isinstance(value,(int,float)) else actual==value
                if isinstance(actual,bool) and not isinstance(value,bool):equal=False
                if not equal:raise ValueError('Declared/observed feature conflict: '+name)
            audit.append(dict(**item,observed=actual,verification=status,consumer='shared RequirementMapper; no client policy',
                              reason='Confirmed against actual input' if status=='CONFIRMED' else 'Not established; not promoted to a fact'))
        return audit

    @classmethod
    def apply_declarations(cls,semantics,target,declarations,*,source_id):
        from dataclasses import replace
        audit=cls.check_declarations(semantics,target,declarations,source_id=source_id)
        target=dict(target);observed=dict(semantics.observed.value);changes={}
        for item in audit:
            if item['verification']!='CONFIRMED':continue
            name=item['feature'];value=item['value'];evidence=tuple([item['evidence'],'user-confirmed declaration checked against input'])
            if name=='grid_change':target[name]=value
            elif name in {'variable_class','unit'}:changes[name]=EvidenceValue(value,evidence)
            elif name in {'temporal_semantics','spatial_support'}:
                attr='temporal' if name=='temporal_semantics' else 'spatial';key='semantics' if name=='temporal_semantics' else 'support'
                changes[attr]=EvidenceValue(dict(getattr(semantics,attr).value,**{key:value}),evidence,getattr(semantics,attr).status)
            else:observed[name]=value
        if audit:changes['observed']=EvidenceValue(observed,semantics.observed.evidence+('confirmed declaration input to common mapper',))
        return replace(semantics,**changes),target,audit
    def extract_series(self, cases, *, variable, unit=UNKNOWN, variable_class="continuous",
                       temporal_semantics=UNKNOWN, nonnegative=False):
        """Only visible, isolated series; no target queue, case IDs or labels.

        Counts refer to ALL missing input positions, including natural gaps.
        They are not evaluation strata weights or independent observations.
        """
        from collections import Counter
        conditions = Counter(); finite_count = missing_count = 0; intervals = set(); steps = []
        for case in cases:
            values = np.asarray(case.values); times, step = series_time_axis(case)
            if np.isinf(values).any():
                raise ValueError("Unique ordered finite-or-missing temporal input required")
            steps.append(step)
            intervals.update(np.diff(times.asi8).tolist())
            missing = ~np.isfinite(values)
            finite_count += int((~missing).sum()); missing_count += int(missing.sum())
            starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
            ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])+1
            for start, end in zip(starts, ends):
                conditions[int(end-start), bool(start > 0), bool(end < len(values)), step] += int(end-start)
        resolution = str(pd.Timedelta(next(iter(intervals)), unit="ns")) if len(intervals) == 1 else UNKNOWN
        observed = dict(finite_count=finite_count, missing_count=missing_count,
            support_conditions=[dict(length=n, gap_record_count=n, length_unit="records", left=l, right=r, count=c,
                duration_seconds=n*step/10**9 if step is not None else UNKNOWN)
                for (n,l,r,step),c in sorted(conditions.items(), key=lambda item:str(item[0]))],
            regular_hourly=bool(steps) and all(step == 3600*10**9 for step in steps),
            interval_ns=sorted(intervals), gap_length_unit="records", duration_unit="seconds",
            duration_definition="missing grid slots times regular label interval, NOT an inferred observation averaging window",
            available_at="masked input, before candidate generation", weighting="missing input positions, NOT objective scenario weights")
        return DataSemantics(EvidenceValue({"structure":"time_series"}, ("actual per-entity arrays",)),
            EvidenceValue(variable_class,("declared variable semantics",)),
            EvidenceValue(dict(resolution_km=UNKNOWN,crs=UNKNOWN,support=UNKNOWN),("not required for label restoration",),UNKNOWN),
            EvidenceValue(dict(resolution=resolution,semantics=temporal_semantics,timezone=UNKNOWN),
                          ("actual ordered time labels; no timezone inference",),UNKNOWN if temporal_semantics==UNKNOWN else "CONFIRMED"),
            EvidenceValue(unit,("input contract units",),UNKNOWN if unit==UNKNOWN else "CONFIRMED"),
            EvidenceValue(dict(nonnegative=nonnegative,temporal_semantics=temporal_semantics),("declared physical constraints",)),
            EvidenceValue(dict(status="WARN" if missing_count else "PASS",missing_count=missing_count),("finite/missing input inspection",)),
            variable, EvidenceValue(observed,("visible missing runs and adjacent retained observations",)))

    def extract(self, dataset: xr.Dataset, variable: str, *, confirmed: dict[str, Any] | None = None) -> DataSemantics:
        confirmed = confirmed or {}
        from ecodata.algorithms.dag import Artifact
        key=hashlib.sha256((Artifact("public_input",dataset).fingerprint()+variable+json.dumps(confirmed,sort_keys=True,default=str)).encode()).hexdigest()
        self.cache_info=dict(key=key,hit=key in self._cache)
        if key in self._cache:return deepcopy(self._cache[key])
        info = inspect_dataset("semantic_input", dataset)
        field = dataset[variable]
        detected_class = info.semantic_types.get(variable, UNKNOWN)
        declared_class = confirmed.get("variable_class", field.attrs.get("data_kind"))
        conflicts = ()
        if declared_class and detected_class != UNKNOWN and declared_class != detected_class:
            conflicts = (f"declared={declared_class}, detected={detected_class}",)
        variable_class = declared_class or detected_class or UNKNOWN
        temporal_semantics = confirmed.get("temporal_semantics", field.attrs.get("temporal_semantics", dataset.attrs.get("temporal_semantics", UNKNOWN)))
        spatial_semantics = confirmed.get("spatial_semantics", field.attrs.get("spatial_semantics", dataset.attrs.get("spatial_semantics", UNKNOWN)))
        unit = confirmed.get("unit", field.attrs.get("units", UNKNOWN)) or UNKNOWN
        physics = {
            "nonnegative": bool(confirmed.get("nonnegative", field.attrs.get("valid_min") == 0)),
            "spatial_semantics": spatial_semantics,
            "temporal_semantics": temporal_semantics,
        }
        quality = RawInputQualityGate().validate(dataset, variable=variable, intended_operation="conversion")
        spatial_value = {"resolution_km": confirmed.get("spatial_resolution_km", info.spatial_resolution_km or UNKNOWN),
                         "crs": confirmed.get("crs", info.crs or UNKNOWN),
                         "extent": info.extent or UNKNOWN, "support": spatial_semantics}
        temporal_value = {"resolution": confirmed.get("temporal_resolution", info.temporal_resolution or UNKNOWN), "semantics": temporal_semantics,
                          "coordinate": info.time_coordinate or UNKNOWN}
        numeric=np.asarray(field.values)
        observed=dict(available_at="parsed input before candidate generation",scope="all uploaded input positions, not a hidden reference queue")
        if numeric.dtype.kind in "fiu":
            observed.update(finite_count=int(np.isfinite(numeric).sum()),missing_count=int((~np.isfinite(numeric)).sum()),
                            missing_fraction=float((~np.isfinite(numeric)).mean()),negative_count=int((numeric[np.isfinite(numeric)]<0).sum()))
        result = DataSemantics(
            EvidenceValue({"format": info.format, "structure": info.structure, "source": dataset.attrs.get("source_name", UNKNOWN)},
                          ("reader/recognizer",)),
            EvidenceValue(variable_class, ("confirmed metadata" if declared_class else "recognizer",),
                          "CONFLICT" if conflicts else ("UNKNOWN" if variable_class == UNKNOWN else "CONFIRMED"), conflicts),
            EvidenceValue(spatial_value, ("coordinates and CRS metadata",), "UNKNOWN" if UNKNOWN in spatial_value.values() else "CONFIRMED"),
            EvidenceValue(temporal_value, ("time coordinate and temporal_semantics",), "UNKNOWN" if temporal_semantics == UNKNOWN else "CONFIRMED"),
            EvidenceValue(unit, ("variable units attribute" if unit != UNKNOWN else "no reliable unit metadata",), "UNKNOWN" if unit == UNKNOWN else "CONFIRMED"),
            EvidenceValue(physics, ("confirmed/declared physical metadata",), "UNKNOWN" if spatial_semantics == UNKNOWN or temporal_semantics == UNKNOWN else "CONFIRMED"),
            EvidenceValue({"status": quality["status"], "summary": quality["summary"]}, ("RawInputQualityGate",)), variable,
            EvidenceValue(observed,("actual uploaded field and finite mask; no validation labels",)))
        self._cache[key]=deepcopy(result)
        if len(self._cache)>64:self._cache.popitem(last=False)
        return result

    @staticmethod
    def training_bias(reference,historical,application):
        """Optional diagnosis from explicit training pairs, never validation truth."""
        if not isinstance(reference,xr.DataArray) or not isinstance(historical,xr.DataArray):return dict(status=UNKNOWN,reason="No legal training pair")
        if any(a.attrs.get("split")!="train" for a in (reference,historical)):return dict(status=UNKNOWN,reason="Training role unconfirmed")
        if not reference.attrs.get("units") or reference.attrs.get("units")!=historical.attrs.get("units"):return dict(status=UNKNOWN,reason="Training units differ")
        if any("time" not in a.coords for a in (reference,historical,application)):return dict(status=UNKNOWN,reason="Training/application time definition missing")
        if len(np.intersect1d(reference.time.values,application.time.values)):return dict(status=UNKNOWN,reason="Training/application overlap")
        try:ref,hist=xr.align(reference,historical,join="exact")
        except ValueError:return dict(status=UNKNOWN,reason="Training queues differ")
        d=(hist-ref).values;valid=np.isfinite(d)
        if valid.sum()<3:return dict(status=UNKNOWN,reason="Insufficient finite training pairs")
        return dict(status="CONFIRMED",count=int(valid.sum()),bias=float(d[valid].mean()),units=reference.attrs["units"],
                    scope="explicit training period only; not proof of application bias",evidence="historical_simulation - training_reference")


class RequirementMapper:
    def __init__(self, rule_path: str | None = None):
        path = rule_path or str(files("ecodata.algorithms").joinpath("semantic_rules.json"))
        self.rules = json.loads(open(path, encoding="utf-8").read())["rules"]

    @staticmethod
    def _facts(semantics: DataSemantics, target: dict[str, Any]) -> dict[str, Any]:
        source_res = semantics.spatial.value.get("resolution_km")
        target_res = target.get("spatial_resolution_km")
        scale_change = source_res != UNKNOWN and target_res is not None and not np.isclose(float(source_res), float(target_res))
        source_time = semantics.temporal.value.get("resolution")
        return {
            "variable_class": semantics.variable_class.value,
            "spatial_scale_change": bool(scale_change or target.get("grid_change",False)),
            "target_is_finer": bool(scale_change and float(target_res) < float(source_res)),
            "target_is_coarser": bool(target.get("coarsening",False) or (scale_change and float(target_res) > float(source_res))),
            "station_to_grid": target.get("operation") in {"station_to_grid", "station_to_points"},
            "topographic_auxiliary": bool(target.get("topographic_auxiliary")),
            "domain": target.get("domain", UNKNOWN),
            "temporal_scale_change": bool(target.get("temporal_resolution") and source_time != UNKNOWN and target["temporal_resolution"] != source_time),
            "temporal_semantics": {"mean":"interval_mean","interval_total":"accumulation"}.get(semantics.temporal.value.get("semantics"),semantics.temporal.value.get("semantics")),
            "nonnegative": semantics.physics.value.get("nonnegative", False),
            "multi_source": int(target.get("source_count", 1)) > 1,
            "covariate_estimation": bool(target.get("covariate_estimation",False)),
            "temporal_processing": target.get("operation")=="temporal_processing",
            "spatial_processing": target.get("operation")=="spatial_processing",
            "scientific_operation": target.get("operation"),
            "output_representation": target.get("output_representation"),
            "quality_difference": bool(target.get("quality_difference", False)),
            "target_model": bool(target.get("model")),
            "unit_known": semantics.unit.value != UNKNOWN,
            "crs_known": semantics.spatial.value.get("crs") != UNKNOWN,
            "support_known": semantics.spatial.value.get("support") != UNKNOWN,
            "semantic_conflict": semantics.variable_class.status == "CONFLICT",
            "temporal_recovery": target.get("operation") == "temporal_missing_recovery",
            "require_complete_coverage": bool(target.get("require_complete_coverage",False)),
            "prefer_observation_support": bool(target.get("prefer_observation_support",False)),
            "past_only": target.get("time_access") == "past_only",
            "one_sided_support": any(r["left"] != r["right"] for r in semantics.observed.value.get("support_conditions",[])),
            "missing_input": semantics.observed.value.get("missing_count",0)>0,
            "require_coarse_consistency": bool(target.get("require_coarse_consistency",False)),
            "coarse_observations_available": bool(target.get("coarse_observations_available",False)),
            "training_bias_supported": target.get("training_bias",{}).get("status")=="CONFIRMED",
            "source_grid_mismatch": bool(target.get("source_grid_mismatch",False)),
        }

    @staticmethod
    def _matches(facts: dict[str, Any], conditions: dict[str, Any]) -> bool:
        for key, expected in conditions.items():
            actual = facts.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def map(self, semantics: DataSemantics, target: dict[str, Any]) -> CapabilityRequirements:
        facts = self._facts(semantics, target)
        required, hard, soft, unknowns, trace, problems, optional = set(), set(), set(), set(), [], [], set()
        for rule in self.rules:
            if self._matches(facts, rule["when"]):
                required.update(rule.get("required", [])); hard.update(rule.get("hard", []))
                soft.update(rule.get("soft", [])); unknowns.update(rule.get("unknown", []))
                optional.update(rule.get("optional",[]))
                trace.append({"rule": rule["id"], "because": rule["because"], "facts": {k: facts[k] for k in rule["when"]},
                              "produced": {k: rule.get(k, []) for k in ("required", "hard", "soft", "unknown","optional")}})
                capabilities=sorted(set(rule.get("required",[])+rule.get("optional",[])))
                status="UNKNOWN" if rule.get("unknown") else "WORKING_ASSUMPTION" if target.get("working_assumptions") else "CONFIRMED"
                kind=rule.get("problem_type",rule["id"])
                identity=json.dumps([kind,rule["id"],semantics.variable,target.get("source_id")],sort_keys=True)
                problems.append(dict(id="problem_"+hashlib.sha256(identity.encode()).hexdigest()[:16],type=kind,variable=semantics.variable,
                    source_roles=target.get("source_roles",["primary"]),scope=dict(spatial=semantics.spatial.value,temporal=semantics.temporal.value),
                    measure=dict(facts=trace[-1]["facts"],input_statistics=semantics.observed.value,training_diagnostic=target.get("training_bias") if kind=="training_bias" else None),
                    requirement=target.get("operation"),status=status,must_handle=bool(rule.get("required") or rule.get("hard")),capabilities=capabilities,
                    evidence=list(semantics.observed.evidence)+[rule["because"]],rule=rule["id"],port=target.get("port","task_output.field"),
                    scope_note="Input diagnosis / declared task requirement; not a prediction of algorithm gain"))
        for key in ("unit_known", "crs_known", "support_known"):
            if not facts[key]:
                unknowns.add(key.replace("_known", ""))
        if facts["semantic_conflict"]:
            hard.add("no_semantic_conflict")
        # Explicit user capabilities are hard task requirements, regardless of
        # any optional empirical benefit. The selector validates catalog IDs.
        for capability in target.get('required_capabilities',[]):
            required.add(capability);optional.discard(capability)
            problems.append(dict(id='explicit_'+hashlib.sha256(capability.encode()).hexdigest()[:16],type='explicit_requirement',
                variable=semantics.variable,source_roles=target.get('source_roles',['primary']),scope=dict(spatial=semantics.spatial.value,temporal=semantics.temporal.value),
                measure={},requirement=target.get('operation'),status='CONFIRMED',must_handle=True,capabilities=[capability],
                evidence=['User explicitly requires this capability; benefit cannot waive acceptance'],rule='explicit_requirement',port='task_output.field'))
        optional.update(set(target.get('optional_capabilities',[]))-required)
        return CapabilityRequirements(tuple(sorted(required)), tuple(sorted(hard)), tuple(sorted(soft)), tuple(sorted(unknowns)), tuple(trace),tuple(problems),tuple(sorted(optional)))
