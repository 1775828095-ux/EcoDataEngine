"""Three-portrait screening then bounded mutation over explicit source/time DAGs.

Optional native optimisers. Counts refer to the declared finite grammar, never to
all conceivable scientific workflows. Numerical failures remain in the ledger.
"""
from copy import deepcopy
from dataclasses import replace
from itertools import permutations, product
from math import prod
from pathlib import Path
from time import perf_counter
import hashlib
import json
import random

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.attribute_registry import digest
from ecodata.algorithms.contracts import AlgorithmContext, AlgorithmSpec, AlgorithmResult, Compatibility
from ecodata.algorithms.portrait_state import measure, risk
from ecodata.algorithms.portrait_rules import parameterized_terms


VERSION = "source_time_composition_v1"


class PrimaryWeightedMean:
    spec = AlgorithmSpec("fusion.primary_weighted_mean","Declared primary/support weighted mean","1",
        "multi_source_fusion","available",("aligned_grids",),("continuous",),
        physical_constraints=("same_variable","same_unit","same_crs","same_coordinates"),
        capabilities=("multi_source_fusion","explicit_primary_support_roles"),
        backend="Existing UncertaintyWeightedFusion linear-combination kernel; coefficient scales are not uncertainty",
        input_contract=dict(requires_parameters=["primary_index","primary_share"]),
        output_contract=dict(uncertainty_estimated=False))

    def parameter_proposals(self,context,rng):
        if len(context.sources)<2:return []
        try:xr.align(*(s.dataset[context.variable] for s in context.sources),join='exact')
        except ValueError:return []
        return [dict(primary_index=i,primary_share=float(rng.uniform(.5,1))) for i in range(len(context.sources))]

    def compatibility(self, context):
        index = context.parameters.get("primary_index");share = context.parameters.get("primary_share")
        reasons = []
        if context.semantic_type!="continuous":reasons.append("Primary/support averaging requires continuous variables")
        if len(context.sources)<2 or type(index)!=int or not 0<=index<len(context.sources):
            reasons.append("At least two sources and an explicit primary index required")
        if not isinstance(share,(int,float)) or not .5<share<1:
            reasons.append("A dominant primary share in (.5,1) required")
        return Compatibility(not reasons,tuple(reasons))

    def run(self, context):
        from ecodata.fusion.weighted import UncertaintyWeightedFusion, WeightedFusionConfig
        check = self.compatibility(context)
        if not check.eligible:raise ValueError(str(check.reasons))
        n=len(context.sources);index=context.parameters["primary_index"];share=context.parameters["primary_share"]
        weights=[share if i==index else (1-share)/(n-1) for i in range(n)]
        sources=[]
        for attr in ("spatial_semantics","temporal_semantics"):
            if len({s.dataset[context.variable].attrs.get(attr) for s in context.sources})!=1:
                raise ValueError("Incompatible source semantics: "+attr)
        for source,weight in zip(context.sources,weights):
            ds=source.dataset[[context.variable]].copy()
            ds["_mix_scale"]=xr.full_like(ds[context.variable],1/np.sqrt(weight))
            ds["_mix_scale"].attrs["units"]="declared coefficient scale; not observational uncertainty"
            sources.append(EcoDataBundle(ds))
        result=UncertaintyWeightedFusion(WeightedFusionConfig(context.variable,"_mix_scale")).apply(sources).data
        ds=result.dataset.drop_vars("_mix_scale")
        if any("time_bounds" in s.dataset for s in context.sources):
            if not all("time_bounds" in s.dataset for s in context.sources) or any(not np.array_equal(s.dataset.time_bounds,context.sources[0].dataset.time_bounds) for s in context.sources):
                raise ValueError("Source interval supports differ")
            ds["time_bounds"]=context.sources[0].dataset.time_bounds
        ds.attrs.update(fusion_method="declared_primary_weighted_mean",uncertainty_estimated=False)
        return AlgorithmResult(EcoDataBundle(ds,result.history),self.spec.id,dict(primary_index=index,weights=weights,uncertainty_estimated=False))


def chains(stages, registry):
    """All distinct topological chains, including optional skips; no silent cap."""
    if isinstance(stages, dict):
        if set(stages)!={"plans","evidence"} or not stages["evidence"] or not 1 <= len(stages["plans"]) <= 512:
            raise ValueError("Explicit branch plans need evidence and 1..512 alternatives")
        result={}
        for plan in stages["plans"]:
            ordered=[dict(id=str(i),after=[str(i-1)] if i else [],options=[step],
                          evidence=stages["evidence"]) for i,step in enumerate(plan)]
            for chain in chains(ordered,registry):result[digest(chain)]=chain
        return [result[k] for k in sorted(result)]
    if not isinstance(stages, list) or len(stages) > 6:
        raise ValueError("Each branch needs a list of at most six declared stages")
    names = [s["id"] for s in stages]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate stage id")
    for stage in stages:
        if set(stage)-{"id", "after", "options", "context_hours", "evidence"} or not stage.get("evidence"):
            raise ValueError("Stage needs reviewed evidence and supported fields")
        if set(stage.get("after", []))-set(names) or stage["id"] in stage.get("after", []):
            raise ValueError("Unknown/self stage dependency")
        if not 1 <= len(stage["options"]) <= 8:
            raise ValueError("Stage needs 1..8 options")
        for option in stage["options"]:
            if set(option)-{"algorithm", "parameters"} or "algorithm" not in option:
                raise ValueError("Unsupported stage option")
            if option["algorithm"] is not None and registry.get(option["algorithm"]).spec.status != "available":
                raise ValueError("Unavailable stage algorithm")
    result = {}
    for order in permutations(stages):
        positions = {s["id"]: i for i, s in enumerate(order)}
        if any(positions[a] >= positions[s["id"]] for s in order for a in s.get("after", [])):
            continue
        for choices in product(*(s["options"] for s in order)):
            chain = [dict(algorithm=o["algorithm"], parameters=o.get("parameters", {})) for o in choices if o["algorithm"] is not None]
            result[digest(chain)] = chain
            if len(result) > 10000:
                raise ValueError("More than 10000 branch alternatives; narrow the declared grammar")
    if not result:
        raise ValueError("Stage dependency cycle")
    return [result[k] for k in sorted(result)]


def coordination_limits(value):
    if set(value)-{"max_source_disagreement","max_seam_jump","max_phase_hours","evidence"}:
        raise ValueError("Unknown coordination constraint")
    if any(k!="evidence" for k in value) and not value.get("evidence"):
        raise ValueError("Coordination thresholds need task evidence")
    for k,v in value.items():
        if k!="evidence" and (not isinstance(v,(int,float)) or not np.isfinite(v) or v<0):
            raise ValueError("Finite nonnegative coordination threshold required")
    return value


def score_state(field, baseline, final=None, cost=0):
    model = field["model"]; terms = []
    definitions = {a["id"]:a for a in model["attribute_registry"]["attributes"]}
    issues = model["state_policy"]["scenario"]["issues"]; total = sum(i["importance"] for i in issues)
    for issue in issues:
        metric = issue["metric"]; before = baseline.get(metric); after = None if final is None else final.get(metric)
        if metric == "declared_cost":
            before = 0.; after = cost/(1+cost)
        utility = None if before is None or after is None else risk(before,issue)-risk(after,issue)
        bounds = [utility,utility] if utility is not None else [risk(before,issue)-1,risk(before,issue)] if before is not None else [-risk(after,issue),1-risk(after,issue)] if after is not None else [-1,1]
        terms.append(dict(dimension=issue["attribute"],weight=issue["importance"]/total,activation=1.,raw_activation=1.,
            utility=utility,utility_interval=bounds,guarantee=None,definition=definitions[issue["attribute"]],
            data_evidence=dict(before=before,metric=metric),algorithm_evidence=dict(after=after,
                **({"response_portrait":field["response_portrait"].evidence} if metric=="expected_error" and "response_portrait" in field else {})),model_evidence=issue["evidence"]))
    terms,lo,hi = parameterized_terms(terms,model.get("scoring_parameters",{}))
    return dict(terms=terms,score_interval=[lo,hi])


class Composition:
    def __init__(self, parsed, request, registry):
        from ecodata.algorithms.profile_matching import prepare_model_request, validate_model_context
        self.registry = registry
        self.request = deepcopy(request)
        self.config = request["composition"]
        allowed = {"source_stages", "post_stages", "windows", "fusion_options", "search", "coordination", "fields","reconstruction","response_profile","parallel_scoring"}
        if set(self.config)-allowed:
            raise ValueError("Unknown composition setting")
        self.sources = {}
        self.resources = []
        for role, ds in parsed:
            if role not in {"primary", "background"}:
                if role not in {"auxiliary","observations","training_reference","historical_simulation","state","target"}:
                    raise ValueError("Unsupported composition resource role: "+role)
                self.resources.append((role,ds));continue
            key = ds.attrs.get("source_id")
            if not key or key in self.sources:
                raise ValueError("Composition needs unique stable source_id on every source")
            self.sources[key] = ds
        if not self.sources or (len(self.sources)<2 and not self.resources):
            raise ValueError("Source/time composition requires field sources and a second field or explicit observation/covariate resource")
        self.ids = sorted(self.sources)
        self.axes = []
        self.slots = []
        self.fields = []
        self.coordination = coordination_limits(self.config.get("coordination", {}))
        field_specs = self.config.get("fields", [dict(request={k:v for k,v in request.items() if k != "composition"}, importance=1)])
        if not field_specs or len(field_specs) > 8:
            raise ValueError("Composition needs 1..8 field specifications")
        for spec in field_specs:
            if set(spec)-{"request", "importance", "source_stages", "post_stages", "coordination"}:
                raise ValueError("Unknown field setting")
            req, model = prepare_model_request(deepcopy(spec["request"]))
            if set(req.get("constraints",{})) - {"nonnegative","unit","output_unit"}:
                raise ValueError("No implemented composition consumer for requested hard constraint")
            for key,value in req.get("constraints",{}).items():
                if key in req and req[key]!=value:
                    raise ValueError("Conflicting task/constraint declaration: "+key)
                req[key]=value
            if req.get("output_unit",req["unit"])!=req["unit"]:
                raise ValueError("Composition requires an explicit unit-conversion adapter before changing units")
            if "state_policy" not in model or model["state_policy"].get("memory_version"):
                raise ValueError("Composition requires explicit three-portrait state policy; legacy response banks cannot be silently reused for new graph grammar")
            definitions = {a["id"]:a for a in model["attribute_registry"]["attributes"]}
            if any(definitions[a]["property"] not in {"nonnegative","range_preservation"} for a in model["matching_requirements"]["hard"]):
                raise ValueError("No composition hard consumer for requested guarantee; unsupported physical rules must not silently pass")
            if req["task"] not in {"multisource","temporal"} or req["semantic_type"] != "continuous":
                raise ValueError("Composition accepts continuous multisource or temporal field contracts")
            if req["task"]=="temporal" and len(self.sources)!=1:
                raise ValueError("Single-series temporal composition needs exactly one field source")
            variable = req["variable"]
            if variable in [f["variable"] for f in self.fields]:
                raise ValueError("Duplicate output variable")
            weight = spec.get("importance", 1)
            if not isinstance(weight, (int,float)) or not np.isfinite(weight) or weight <= 0:
                raise ValueError("Positive field importance required")
            contexts = {}
            for key in self.ids:
                ds = self.sources[key]
                if variable not in ds or ds[variable].attrs.get("units") != req["unit"]:
                    raise ValueError("Source variable/unit conflict: " + key)
                if not ds[variable].attrs.get("physical_variable"):
                    raise ValueError("Explicit physical_variable required")
                if ds[variable].dims not in {("lat","lon"), ("time","lat","lon")}:
                    raise ValueError("Composition fields need lat/lon or time/lat/lon dimensions")
                if not ds.attrs.get("crs"):
                    raise ValueError("Composition requires explicit source CRS")
                if CRS.from_user_input(ds.attrs["crs"]) != CRS.from_user_input(self.sources[self.ids[0]].attrs["crs"]):
                    raise ValueError("CRS mismatch: explicit reprojection stage is required before this grammar")
                ctx = AlgorithmContext((EcoDataBundle(ds),), variable, req["semantic_type"],
                    tuple(req["target_lat"]), tuple(req["target_lon"]), parameters=req.get("parameters", {}),
                    model_constraints=dict(req.get("constraints", {}),unit=req["unit"],output_unit=req["unit"]))
                if self.resources:
                    from ecodata.algorithms.platform_search import make_context
                    ctx,_,_=make_context([("primary",ds),*self.resources],req)
                ctx=replace(ctx,parameters=dict(ctx.parameters,composition_allow_training=req.get("allow_training",False)))
                validate_model_context(ctx, model)
                contexts[key] = ctx
            physical = {s[variable].attrs["physical_variable"] for s in self.sources.values()}
            if len(physical) != 1:
                raise ValueError("Different physical variables cannot be fused")
            for attr in ("spatial_semantics","temporal_semantics"):
                meanings={s[variable].attrs.get(attr) for s in self.sources.values()}
                required=attr=="spatial_semantics" or any("time" in s.dims for s in self.sources.values())
                if required and (None in meanings or "UNKNOWN" in meanings or len(meanings)!=1):
                    raise ValueError("Source "+attr+" must be explicit and consistent before composition")
                if req.get(attr) is not None and meanings!={req[attr]}:
                    raise ValueError("Source/request semantic conflict: "+attr)
            field = dict(variable=variable, request=req, model=model, weight=weight, contexts=contexts,
                coordination=coordination_limits(spec.get("coordination",self.coordination)))
            if self.config.get("reconstruction"):
                from ecodata.algorithms.spatiotemporal_detail import Reconstruction
                declaration=self.config["reconstruction"]
                if set(declaration)!={"fine_source","coarse_source","evidence"} or not declaration["evidence"] or {declaration["fine_source"],declaration["coarse_source"]}!=set(self.ids):
                    raise ValueError("Reconstruction needs two explicitly identified complementary sources and evidence")
                field["reconstruction"]=Reconstruction(self.sources[declaration["fine_source"]],self.sources[declaration["coarse_source"]],req)
            self.fields.append(field)
            stages = spec.get("source_stages", self.config.get("source_stages"))
            if stages is None:
                stages={key:self.default_stages(field,key) for key in self.ids}
            if set(stages) != set(self.ids):
                raise ValueError("source_stages must cover every stable source_id")
            for key in self.ids:
                required_halo = 0 if isinstance(stages[key],dict) else max((s.get("context_hours", 0) for s in stages[key]), default=0)
                if not np.isfinite(required_halo) or required_halo < 0:
                    raise ValueError("Invalid stage context requirement")
                for window in self.windows(req):
                    if window.get("halo_hours", 0) < required_halo:
                        raise ValueError("Window halo shorter than declared stage context")
                    self.axis((variable, window["id"], key), chains(stages[key], registry))
            for window in self.windows(req):
                self.axis((variable, window["id"], "fusion"), self.fusions())
                post=spec.get("post_stages", self.config.get("post_stages"))
                if post is None:post=self.default_stages(field,None)
                self.axis((variable, window["id"], "post"), chains(post, registry))
        self.sizes = [len(a) for a in self.axes]
        self.count = prod(self.sizes)
        self.baselines = {f["variable"]: self.baseline(f) for f in self.fields}
        for field in self.fields:
            if "reconstruction" in field:self.baselines[field["variable"]].update(field["reconstruction"].baseline())
            if self.config.get("response_profile"):
                if "reconstruction" not in field:raise ValueError("Response portrait requires declared reconstruction scope")
                from ecodata.algorithms.composition_response import ResponsePortrait
                field["response_portrait"]=ResponsePortrait(self.config["response_profile"],self.sources,field["request"],self.config["reconstruction"])
                self.baselines[field["variable"]]["expected_error"]=1. if field["response_portrait"].supported else None
        self._contract_cache={}
        self._stage_cache={}
        self.parallel_scorer=None

    def prepare_parallel_scoring(self):
        if self.config.get("parallel_scoring") is not None and self.parallel_scorer is None:
            from ecodata.algorithms.composition_scoring import ParallelPortraitScoring
            if self.request.get("task")=="temporal":
                from ecodata.algorithms.daily_reconstruction import DailyPortraitScoring
                self.parallel_scorer=DailyPortraitScoring(self,self.config['parallel_scoring'])
            else:
                self.parallel_scorer=ParallelPortraitScoring(self,self.config['parallel_scoring'])

    def preflight(self,genome):
        """Cheap structural legality, not a claim of numerical success."""
        from ecodata.algorithms.portrait_rules import algorithm_properties
        chosen={slot:self.axes[i][g] for i,(slot,g) in enumerate(zip(self.slots,genome))}
        for field in self.fields:
            req=field["request"];var=field["variable"]
            target=(tuple(req["target_lat"]),tuple(req["target_lon"]))
            for window in self.windows(req):
                grids=[]
                for key in self.ids:
                    chain=chosen[var,window["id"],key]
                    cache=(var,window["id"],key,digest(chain))
                    if cache not in self._contract_cache:
                        ds=self.slice(self.sources[key],window)
                        grid=(tuple(ds.lat.values),tuple(ds.lon.values))
                        times=pd.DatetimeIndex(ds.time.values) if "time" in ds.dims else None
                        error=None
                        for step in chain:
                            plugin=self.registry.get(step["algorithm"]);spec=plugin.spec
                            if spec.requires_training and not req.get("allow_training",False):error="training not permitted";break
                            if spec.requires_observations and field["contexts"][key].observations is None:error="missing observation port";break
                            if algorithm_properties(plugin)["transfer"] in {"nearest","bilinear","pchip","cubic"}:grid=target
                            elif spec.category in {"single_source_scale","raster_reconstruction","learned_reconstruction"}:grid=None
                            if spec.category in {"temporal_scale","temporal_statistics"}:
                                frequency=step["parameters"].get("target_time_frequency",req.get("parameters",{}).get("target_time_frequency"))
                                if times is None or not frequency:error="missing time/frequency contract";break
                                times=pd.date_range(times.min().ceil(frequency),times.max().floor(frequency),freq=frequency)
                        if req.get("target_times") is not None:
                            wanted=pd.DatetimeIndex(req["target_times"])
                            wanted=wanted[(wanted>=pd.Timestamp(window["start"]))&(wanted<=pd.Timestamp(window["end"]))]
                            if times is None or not wanted.isin(times).all():error="branch cannot cover declared target timestamps"
                        self._contract_cache[cache]=(grid,error)
                    grid,error=self._contract_cache[cache]
                    if error:return dict(status="REJECT",reason=error)
                    grids.append(grid)
                option=chosen[var,window["id"],"fusion"]
                known=[g for g in grids if g is not None]
                if known and any(g!=known[0] for g in known):return dict(status="REJECT",reason="fusion input grids differ")
                if option.get("join_grid","target")=="target" and any(g is not None and g!=target for g in grids):return dict(status="REJECT",reason="fusion requires target grid")
                output=known[0] if len(known)==len(grids) else None
                for step in chosen[var,window["id"],"post"]:
                    plugin=self.registry.get(step["algorithm"])
                    if algorithm_properties(plugin)["transfer"] in {"nearest","bilinear","pchip","cubic"}:output=target
                if output is not None and output!=target:return dict(status="REJECT",reason="terminal grid differs from requested target")
        return dict(status="CONTRACT_COMPATIBLE",numerical_success="NOT_YET_ASSESSED")

    def axis(self, slot, alternatives):
        self.slots.append(slot)
        self.axes.append(alternatives)

    def default_stages(self, field, source):
        """A capability/compatibility catalog, not hand-picked complete graphs."""
        from ecodata.algorithms.portrait_rules import algorithm_properties
        ctx=field["contexts"][source or self.ids[0]]
        if "time" in ctx.sources[0].dataset.dims:
            ctx=replace(ctx,sources=(EcoDataBundle(ctx.sources[0].dataset.isel(time=0)),))
        methods=[]
        for spec in self.registry.specs():
            plugin=self.registry.get(spec.id)
            if spec.status!="available" or algorithm_properties(plugin)["transfer"] not in {"nearest","bilinear","pchip","cubic"}:
                continue
            # Prefer the original kernel over its boundary wrapper whenever
            # both provide the same reviewed transfer on this input support.
            if spec.id.startswith("raster_task.spatial."):continue
            if plugin.compatibility(ctx).eligible:methods.append(dict(algorithm=spec.id))
        stages=[]
        if methods:
            stages.append(dict(id="scale",options=[dict(algorithm=None),*methods],evidence="Actual registry transfer portraits and compatibility on visible source support"))
        wants_sign=field["request"].get("nonnegative",False) or any(i["metric"]=="negative_fraction" and i["importance"]>0 for i in field["model"]["state_policy"]["scenario"]["issues"])
        if wants_sign:
            stages.append(dict(id="sign",options=[dict(algorithm=None),dict(algorithm="raster_task.nonnegative")],evidence="Explicit model/task sign requirement; not an inferred universal nonnegative law"))
        if source is not None and field["request"].get("allow_training"):
            plugin=self.registry.get("bias.station_affine")
            if plugin.compatibility(ctx).eligible:
                stages.append(dict(id="calibration",options=[dict(algorithm=None),dict(algorithm=plugin.spec.id)],evidence="Train-only source/observation compatibility from the existing calibration plugin"))
        frequency=ctx.parameters.get("target_time_frequency")
        if source is not None and frequency and "time" in field["contexts"][source].sources[0].dataset.dims:
            stages.append(dict(id="time",options=[dict(algorithm="temporal.semantic_resample",parameters=dict(target_time_frequency=frequency))],evidence="Explicit target frequency and source temporal semantics"))
        return stages

    def windows(self, req):
        windows = self.config.get("windows", [dict(id="all")])
        if not windows or len(windows) > 16 or len({w["id"] for w in windows}) != len(windows):
            raise ValueError("Need 1..16 distinct time windows")
        times = req.get("target_times")
        if times is None:
            if windows != [dict(id="all")] or any("time" in s.dims for s in self.sources.values()):
                raise ValueError("Temporal composition requires explicit target_times and window bounds")
        else:
            times = pd.DatetimeIndex(times)
            if not len(times) or times.has_duplicates or not times.is_monotonic_increasing:
                raise ValueError("target_times must increase strictly")
            if "windows" not in self.config:
                windows=[dict(id="all",start=str(times[0]),end=str(times[-1]))]
            covered = np.zeros(len(times), int)
            for w in windows:
                if set(w)-{"id","start","end","halo_hours"} or not {"start","end"} <= set(w):
                    raise ValueError("Time windows need inclusive start/end")
                if w.get("halo_hours", 0) < 0 or not np.isfinite(w.get("halo_hours", 0)):
                    raise ValueError("Invalid time halo")
                selected = (times >= pd.Timestamp(w["start"])) & (times <= pd.Timestamp(w["end"]))
                if not selected.any():
                    raise ValueError("Window has no target timestamp")
                covered += selected
            if not np.all(covered):
                raise ValueError("Time windows leave uncovered target timestamps")
        return windows

    def fusions(self):
        result = {}
        defaults=[dict(algorithm="prepare.single_background",evidence="Single background with explicit resources")] if len(self.ids)==1 else [
            dict(algorithm="fusion.primary_weighted_mean",primary=key,primary_share=share,join_grid=join,evidence="Declared exploration coefficients and all possible dominant-source roles; not measured reliability")
            for key in self.ids for share in (.6,.8) for join in ("target","native_common")]
        for option in self.config.get("fusion_options",defaults):
            if not option.get("evidence"):
                raise ValueError("Fusion role/weight options require task evidence")
            if option["algorithm"] == "fusion.primary_weighted_mean":
                if set(option)-{"algorithm","primary","primary_share","evidence","join_grid"}:
                    raise ValueError("Unsupported primary fusion option")
                if option["primary"] not in self.ids or not .5 < option["primary_share"] < 1:
                    raise ValueError("Primary must be an input with dominant share in (.5,1)")
                if len(self.ids)<2:raise ValueError("Primary/support averaging needs at least two field sources")
            elif option["algorithm"] == "fusion.uncertainty_weighted":
                if set(option)-{"algorithm","evidence","independence_confirmed","join_grid"} or option.get("independence_confirmed") is not True:
                    raise ValueError("Inverse-variance fusion has no artificial primary role")
                if any("uncertainty" not in ds for ds in self.sources.values()):
                    raise ValueError("Inverse-variance option needs supplied uncertainty on every source")
            elif option["algorithm"] == "prepare.single_background":
                if set(option)-{"algorithm","evidence","join_grid"} or len(self.ids)!=1:
                    raise ValueError("Single-background boundary requires exactly one field source")
            elif option["algorithm"]=="prepare.select_source":
                if set(option)-{"algorithm","source","evidence","join_grid"} or option.get("source") not in self.ids:
                    raise ValueError("Source-only control requires an explicitly selected source")
            elif option["algorithm"]=="fusion.spatiotemporal_detail":
                if set(option)-{"algorithm","detail_strength","evidence","join_grid"} or not self.config.get("reconstruction") or not 0<=option.get("detail_strength",-1)<=1:
                    raise ValueError("Spatiotemporal-detail fusion requires an explicit reconstruction contract and strength in [0,1]")
            else:
                raise ValueError("Unsupported executable fusion contract")
            if option.get("join_grid","target") not in {"target","native_common"}:
                raise ValueError("Fusion join_grid must be target or native_common")
            result[digest(option)] = option
        if not result:
            raise ValueError("No fusion options")
        # Evidence is provenance, not an extra executable parameter.
        canonical = {}
        for option in result.values():
            option=dict(option,join_grid=option.get("join_grid","target"))
            canonical[digest({k:v for k,v in option.items() if k!="evidence"})] = option
        return [canonical[k] for k in sorted(canonical)]

    def decode(self, index):
        if not 0 <= index < self.count:
            raise ValueError("Genome index outside frozen grammar")
        genome = [0]*len(self.sizes)
        for i in range(len(genome)-1,-1,-1):
            genome[i] = index % self.sizes[i]
            index //= self.sizes[i]
        return tuple(genome)

    def encode(self, genome):
        value = 0
        for digit, size in zip(genome, self.sizes):
            value = value*size+digit
        return value

    def plan(self, genome):
        return [dict(field=s[0], window=s[1], branch=s[2], choice=self.axes[i][g]) for i,(s,g) in enumerate(zip(self.slots, genome))]

    def steps(self, genome):
        return sum(int(not self.axes[i][g]["algorithm"].startswith("prepare.")) if s[2]=="fusion" else len(self.axes[i][g]) for i,(s,g) in enumerate(zip(self.slots,genome)))

    def states(self, ds, ctx):
        if ds[ctx.variable].dims==("time","lat","lon") and ds.sizes["lat"]==ds.sizes["lon"]==1:
            return measure(ds,ctx)["metrics"]
        if "time" not in ds[ctx.variable].dims:
            return measure(ds, ctx)["metrics"]
        measured = []
        for t in ds.time.values:
            original = ctx.sources[0].dataset
            if "time" in original.dims:
                if t not in original.time:
                    continue
                original = original.sel(time=t, drop=True)
            measured.append(measure(ds.sel(time=t, drop=True), replace(ctx, sources=(EcoDataBundle(original),)))["metrics"])
        return {k: float(np.mean([m[k] for m in measured])) if measured and all(m[k] is not None for m in measured) else None for k in measure(ds.isel(time=0, drop=True), ctx)["metrics"]}

    def baseline(self, field):
        metrics = [self.states(self.sources[k],field["contexts"][k]) for k in self.ids]
        return {name: float(np.mean([m[name] for m in metrics])) if all(m[name] is not None for m in metrics) else None for name in metrics[0]}

    def slice(self, ds, window):
        if "time" not in ds.dims:
            return ds.copy(deep=True)
        delta = pd.Timedelta(hours=window.get("halo_hours",0))
        output = ds.sel(time=slice(pd.Timestamp(window["start"])-delta, pd.Timestamp(window["end"])+delta))
        if not output.sizes.get("time"):
            raise ValueError("Source has no observations in the required time window")
        return output.copy(deep=True)

    def run_chain(self, ds, chain, ctx, trace, owner):
        from ecodata.algorithms.dag import Artifact
        cache_key=(tuple(owner),digest(chain),Artifact("composition_stage_input",ds).fingerprint())
        if cache_key in self._stage_cache:
            result,records=self._stage_cache[cache_key]
            for item in deepcopy(records):
                item.update(cache_hit=True,original_kernel_calls=item.get("kernel_calls",0),kernel_calls=0);trace.append(item)
            return result.copy(deep=True)
        begin=len(trace)
        for step in chain:
            plugin = self.registry.get(step["algorithm"])
            if plugin.spec.requires_training and not ctx.parameters.get("composition_allow_training"):
                raise ValueError("Training stage requires explicit training permission/resources")
            before = self.states(ds, ctx)
            kernel_calls = 0
            def apply(part):
                nonlocal kernel_calls
                local = replace(ctx, sources=(EcoDataBundle(part),), parameters=dict(ctx.parameters, **step["parameters"]))
                if "time" in part.coords and part.time.ndim==0:
                    stamp=pd.Timestamp(part.time.values)
                    observations=ctx.observations
                    if observations is not None and "time" in observations:
                        observations=observations[pd.to_datetime(observations.time).eq(stamp)].copy()
                    auxiliaries=[]
                    for a in ctx.auxiliaries:
                        if "time" in a.dims:
                            if stamp.to_datetime64() not in a.time:raise ValueError("No covariate support at the branch timestamp")
                            a=a.sel(time=stamp.to_datetime64())
                        auxiliaries.append(a)
                    local=replace(local,observations=observations,auxiliaries=tuple(auxiliaries))
                if plugin.spec.category == "raster_constraint":
                    local = replace(local,target_lat=tuple(part.lat.values),target_lon=tuple(part.lon.values))
                check = plugin.compatibility(local)
                if not check.eligible:
                    raise ValueError(plugin.spec.id+": "+str(check.reasons))
                result = plugin.run(local).data.dataset
                kernel_calls += 1
                for name,coordinate in part.coords.items():
                    if coordinate.ndim==0 and name not in result.coords:result=result.assign_coords({name:coordinate})
                # Uncertainty survives only an actual identity transform, or
                # an algorithm which explicitly produced its own uncertainty.
                identity = part[ctx.variable].dims==result[ctx.variable].dims and part[ctx.variable].shape==result[ctx.variable].shape and np.array_equal(part[ctx.variable],result[ctx.variable],equal_nan=True)
                for name, a in part.data_vars.items():
                    if name not in result and identity and all(d in result.coords and np.array_equal(a[d],result[d]) for d in a.dims):
                        result[name] = a
                result.attrs = dict(part.attrs, **result.attrs)
                return result
            if "time" in ds[ctx.variable].dims and plugin.spec.category not in {"temporal_scale", "temporal_statistics", "irregular_time", "temporal_reconstruction"}:
                original=ds
                ds = xr.concat([apply(ds.sel(time=t,drop=True).assign_coords(time=t)).expand_dims(time=[t]) for t in ds.time.values], dim="time")
                if "time_bounds" in original:ds["time_bounds"]=original.time_bounds
            else:
                ds = apply(ds)
            trace.append(dict(owner=owner,algorithm=plugin.spec.id,parameters=step["parameters"],before=before,after=self.states(ds,ctx),kernel_calls=kernel_calls))
        self._stage_cache[cache_key]=(ds.copy(deep=True),deepcopy(trace[begin:]))
        return ds

    def check_source(self, ds, field, window, *, require_target=True):
        req = field["request"]; variable = field["variable"]
        if ds[variable].attrs.get("units") != req["unit"]:
            raise ValueError("Branch changed the physical unit without a reviewed conversion")
        if ds[variable].attrs.get("physical_variable") != self.sources[self.ids[0]][variable].attrs["physical_variable"]:
            raise ValueError("Branch lost or changed its physical variable identity")
        if "temporal_semantics" in req and ds[variable].attrs.get("temporal_semantics") != req["temporal_semantics"]:
            raise ValueError("Branch temporal semantics conflict")
        for axis, target in (("lat",req["target_lat"]),("lon",req["target_lon"])):
            if axis not in ds.coords or (require_target and not np.array_equal(ds[axis],target)):
                raise ValueError("Branch has not reached the target grid")
        if req.get("target_times") is not None:
            times = pd.DatetimeIndex(req["target_times"])
            times = times[(times >= pd.Timestamp(window["start"])) & (times <= pd.Timestamp(window["end"]))]
            if "time" not in ds.dims or not times.isin(pd.DatetimeIndex(ds.time.values)).all():
                raise ValueError("Branch did not produce required target timestamps")
            ds = ds.sel(time=times)
            if ds[variable].attrs.get("temporal_semantics") in {"mean","interval_mean","accumulation"}:
                if "time_bounds" not in ds:
                    raise ValueError("Interval fields require explicit time_bounds after every processing branch")
                bounds=ds.time_bounds.values.astype("datetime64[ns]")
                if bounds.shape!=(len(ds.time),2) or not (bounds[:,1]>bounds[:,0]).all():
                    raise ValueError("Invalid interval bounds")
        if not np.isfinite(ds[variable]).all():
            raise ValueError("Branch has nonfinite target values; no silent cohort deletion")
        return ds

    def fuse(self, branches, field, option):
        variable = field["variable"]
        if option["algorithm"]=="prepare.select_source":
            return branches[option["source"]],dict(kind="source_only_control",selected_source=option["source"],other_sources_used_in_output=False)
        if option["algorithm"]=="fusion.spatiotemporal_detail":
            declaration=self.config["reconstruction"]
            ctx=replace(field["contexts"][self.ids[0]],sources=tuple(EcoDataBundle(branches[k]) for k in self.ids),
                parameters=dict(reconstruction=field["reconstruction"],fine_index=self.ids.index(declaration["fine_source"]),detail_strength=option["detail_strength"]))
            result=self.registry.get(option["algorithm"]).run(ctx)
            return result.data.dataset,dict(kind=option["algorithm"],fine_source=declaration["fine_source"],coarse_source=declaration["coarse_source"],**result.report)
        if option["algorithm"]=="prepare.single_background":
            return branches[self.ids[0]],dict(kind="single_background_with_explicit_observation_or_covariate_stages",primary=self.ids[0])
        if option["algorithm"] == "fusion.uncertainty_weighted":
            if any("uncertainty" not in ds or ds.uncertainty.attrs.get("units")!=field["request"]["unit"] or not np.isfinite(ds.uncertainty).all() or not (ds.uncertainty>0).all() for ds in branches.values()):
                raise ValueError("No valid propagated uncertainty on the final branch support")
            ctx = replace(field["contexts"][self.ids[0]], sources=tuple(EcoDataBundle(branches[k]) for k in self.ids))
            plugin = self.registry.get(option["algorithm"])
            check = plugin.compatibility(ctx)
            if not check.eligible:
                raise ValueError(str(check.reasons))
            result=plugin.run(ctx).data.dataset
            if "time_bounds" in branches[self.ids[0]]:result["time_bounds"]=branches[self.ids[0]].time_bounds
            return result, dict(kind="inverse_variance",primary=None)
        n = len(branches); primary = option["primary"]; share = option["primary_share"]
        weights = {k: share if k==primary else (1-share)/(n-1) for k in self.ids}
        ctx=replace(field["contexts"][self.ids[0]],sources=tuple(EcoDataBundle(branches[k]) for k in self.ids),
            parameters=dict(primary_index=self.ids.index(primary),primary_share=share))
        result=self.registry.get("fusion.primary_weighted_mean").run(ctx).data.dataset
        if "time_bounds" in branches[self.ids[0]]:result["time_bounds"]=branches[self.ids[0]].time_bounds
        return result, dict(kind="declared_primary_weighted_mean",primary=primary,weights=weights,evidence=option["evidence"])

    def coordination_check(self, branches, field):
        var = field["variable"]
        xr.align(*(ds[var] for ds in branches.values()),join="exact")
        if any("time" in ds.coords and ds.time.ndim==0 for ds in branches.values()):
            if not all("time" in ds.coords and ds.time.ndim==0 for ds in branches.values()) or any(not np.array_equal(ds.time,branches[self.ids[0]].time) for ds in branches.values()):
                raise ValueError("Fusion sources have different or missing observation times")
        for attr in ("spatial_semantics","temporal_semantics"):
            meanings={ds[var].attrs.get(attr) for ds in branches.values()}
            if len(meanings)!=1:
                raise ValueError("Fusion branches have inconsistent "+attr)
        if any("time_bounds" in ds for ds in branches.values()):
            if not all("time_bounds" in ds for ds in branches.values()):
                raise ValueError("Interval support missing on one fusion branch")
            if any(not np.array_equal(ds.time_bounds,branches[self.ids[0]].time_bounds) for ds in branches.values()):
                raise ValueError("Fusion branches refer to different time intervals")
        values = xr.concat([branches[k][var] for k in self.ids],dim="source")
        spread = float((values.max("source")-values.min("source")).max())
        phase = None
        if "time" in values.dims and values.sizes["time"] > 1:
            series = values.mean(["lat","lon"])
            sequences = [series.isel(source=i).values for i in range(len(self.ids))]
            if all(np.count_nonzero(s==s.max())==1 for s in sequences):
                peaks = [pd.Timestamp(series.time.values[int(np.argmax(s))]) for s in sequences]
                phase = (max(peaks)-min(peaks)).total_seconds()/3600
        report = dict(max_source_disagreement=spread,max_phase_hours=phase,
            interpretation="Visible field spread and domain-mean peak-time difference; not dynamical balance")
        report["variable_unit"]=field["request"]["unit"]
        for metric in ("max_source_disagreement","max_phase_hours"):
            if metric in field["coordination"] and (report[metric] is None or report[metric] > field["coordination"][metric]):
                raise ValueError("Coordination gate failed: "+metric+"="+str(report[metric]))
        return report

    def score(self, field, final=None, cost=0):
        return score_state(field, self.baselines[field["variable"]], final, cost)

    def screen(self, genome):
        self.prepare_parallel_scoring()
        if self.parallel_scorer is not None:
            return self.parallel_scorer.score(self.plan(genome))
        chosen = {s:self.axes[i][g] for i,(s,g) in enumerate(zip(self.slots,genome))}
        scores = []; evidence = []
        for f in self.fields:
            all_steps = [step for (v,_,b),c in chosen.items() if v==f["variable"] and b!="fusion" for step in c]
            windows = self.windows(f["request"])
            terminal = {m:None for m in self.baselines[f["variable"]]}
            nonnegative = []; aligned = []
            for w in windows:
                post = chosen[f["variable"],w["id"],"post"]
                nonnegative.append(bool(post and post[-1]["algorithm"]=="raster_task.nonnegative") or all(
                    chosen[f["variable"],w["id"],k] and chosen[f["variable"],w["id"],k][-1]["algorithm"]=="raster_task.nonnegative" for k in self.ids))
                aligned.append(any(s["algorithm"] in {"spatial.nearest","spatial.bilinear","raster_task.spatial.nearest","raster_task.spatial.bilinear"} for s in post) or all(any(s["algorithm"] in {"spatial.nearest","spatial.bilinear","raster_task.spatial.nearest","raster_task.spatial.bilinear"} for s in chosen[f["variable"],w["id"],k]) or
                    all(np.array_equal(self.sources[k][a],f["request"]["target_"+a]) for a in ("lat","lon")) for k in self.ids))
            if all(nonnegative):terminal["negative_fraction"] = 0.
            if all(aligned):terminal["grid_gap"] = 0.
            fusion_steps=sum(not chosen[f["variable"],w["id"],"fusion"]["algorithm"].startswith("prepare.") for w in windows)
            if "response_portrait" in f:
                plan=[dict(field=v,window=w,branch=b,choice=c) for (v,w,b),c in chosen.items() if v==f["variable"]]
                terminal["expected_error"]=f["response_portrait"].estimate(plan)
            matching=self.score(f,final=terminal,cost=len(all_steps)+fusion_steps)
            if "response_portrait" in f:matching["response_evidence"]=f["response_portrait"].evidence
            from ecodata.algorithms.portrait_rules import algorithm_properties
            matching["algorithm_portraits"]={s["algorithm"]:algorithm_properties(self.registry.get(s["algorithm"])) for s in all_steps}
            evidence.append(dict(field=f["variable"],matching=matching))
            scores.append((f["weight"],matching["score_interval"]))
        total = sum(w for w,_ in scores)
        return dict(score_interval=[sum(w*s[i] for w,s in scores)/total for i in (0,1)],fields=evidence)

    def execute(self, genome):
        from ecodata.models.portrait_policy import validate_coupled
        chosen = {s:self.axes[i][g] for i,(s,g) in enumerate(zip(self.slots,genome))}
        trace = []; checks = []; outputs = []; scores = []
        self.last_trace = trace
        self.last_checks = checks
        for field in self.fields:
            variable = field["variable"]; windows = self.windows(field["request"]); products = []
            for window in windows:
                branches = {}
                fusion_option=chosen[variable,window["id"],"fusion"]
                for key in self.ids:
                    ctx = field["contexts"][key]
                    ds = self.run_chain(self.slice(self.sources[key],window),chosen[variable,window["id"],key],ctx,trace,[variable,window["id"],key])
                    branches[key] = self.check_source(ds,field,window,require_target=fusion_option.get("join_grid","target")=="target")
                checks.append(dict(field=variable,window=window["id"],**self.coordination_check(branches,field)))
                fused,roles = self.fuse(branches,field,fusion_option)
                trace.append(dict(owner=[variable,window["id"],"fusion"],roles=roles,parents=self.ids,join_grid=fusion_option.get("join_grid","target")))
                ctx = field["contexts"][self.ids[0]]
                final = self.run_chain(fused,chosen[variable,window["id"],"post"],ctx,trace,[variable,window["id"],"post"])
                checked=self.check_source(final,field,window)
                products.append(checked[[variable,*( ["time_bounds"] if "time_bounds" in checked else [])]])
            if len(products)==1:
                ds = products[0]
            else:
                # Explicit equal-weight overlap blend. Each contributor has
                # passed the same grid/time contract; gaps remain forbidden.
                stacked = xr.concat(products,dim="window",join="outer")
                ds = stacked[[variable]].mean("window",skipna=True,keep_attrs=True)
                ds = ds.sel(time=pd.DatetimeIndex(field["request"]["target_times"]))
                if any("time_bounds" in p for p in products):
                    if not all("time_bounds" in p for p in products):raise ValueError("Interval support disappeared across time windows")
                    bounds=xr.concat([p.time_bounds for p in products],dim="time").sortby("time")
                    for t in np.unique(bounds.time.values):
                        repeated=bounds.sel(time=[t]).values
                        if not np.all(repeated==repeated[0]):raise ValueError("Overlapping windows disagree on interval support")
                    _,first=np.unique(bounds.time.values,return_index=True)
                    ds["time_bounds"]=bounds.isel(time=first).sel(time=ds.time)
            seam = 0.
            if "time" in ds.dims and len(windows)>1:
                boundaries = set()
                times = pd.DatetimeIndex(ds.time.values)
                for w in windows:
                    for edge in (pd.Timestamp(w["start"]),pd.Timestamp(w["end"])):
                        j = int(times.searchsorted(edge))
                        if j>0:boundaries.add(j)
                        if j>=len(times):boundaries.discard(j)
                        if 0<=j<len(times)-1 and times[j]==edge:boundaries.add(j+1)
                seam = max((float(abs(ds[variable].isel(time=j)-ds[variable].isel(time=j-1)).max()) for j in boundaries),default=0.)
            if "max_seam_jump" in field["coordination"] and seam>field["coordination"]["max_seam_jump"]:
                raise ValueError("Coordination gate failed: max_seam_jump="+str(seam))
            checks.append(dict(field=variable,max_seam_jump=seam,overlap_policy="equal-weight blend on explicitly overlapping target timestamps"))
            if field["request"].get("nonnegative") and bool((ds[variable]<0).any()):
                raise ValueError("Final nonnegative hard constraint failed")
            if not np.isfinite(ds[variable]).all():
                raise ValueError("Final finite/coverage gate failed")
            from ecodata.algorithms.portrait_rules import validate_product, data_facts
            facts = [dict(matching_facts=data_facts(field["contexts"][k],0)) for k in self.ids]
            hard = validate_product(ds,variable,dict(model=field["model"],data=facts))
            checks.append(hard)
            if hard["status"] != "PASS":
                raise ValueError("Model hard constraint failed: "+str(hard))
            metrics = [self.states(ds,field["contexts"][k]) for k in self.ids]
            final = {n:float(np.mean([m[n] for m in metrics])) if all(m[n] is not None for m in metrics) else None for n in metrics[0]}
            if "reconstruction" in field:final.update(field["reconstruction"].metrics(ds))
            if "response_portrait" in field:
                plan=[dict(field=v,window=w,branch=b,choice=c) for (v,w,b),c in chosen.items() if v==variable]
                final["expected_error"]=field["response_portrait"].estimate(plan)
            cost = sum(len(c) if b!="fusion" else int(not c["algorithm"].startswith("prepare.")) for (v,_,b),c in chosen.items() if v==variable)
            scores.append((field["weight"],self.score(field,final=final,cost=cost)))
            outputs.append(ds)
        # Exact coordinate merge; no broadcasting across unaligned variables.
        output = xr.merge(outputs,join="exact",compat="no_conflicts")
        for field in self.fields:
            coupled = validate_coupled(output,field["model"]["state_policy"])
            checks.append(coupled)
            if coupled["status"] != "PASS":
                raise ValueError("Cross-variable coordination failed: "+str(coupled))
        total = sum(w for w,_ in scores)
        bounds = [sum(w*s["score_interval"][i] for w,s in scores)/total for i in (0,1)]
        matching=dict(score_interval=bounds,fields=[s for _,s in scores])
        if self.config.get('parallel_scoring') is not None:
            matching=dict(self.screen(genome),actual_state_matching=matching)
        return output,matching,trace,checks


def submit(inputs, request, evaluation, folder, *, budget, seed, progress=None,validation=None):
    from ecodata.algorithms.platform_search import parse_inputs, platform_registry
    from ecodata.algorithms.numerical_search import save
    folder = Path(folder); started = perf_counter(); registry = platform_registry()
    parsed,audit = parse_inputs(inputs)
    engine = Composition(parsed,request,registry)
    if validation is not None:
        from ecodata.algorithms.platform_search import SearchValidation
        if not isinstance(validation,SearchValidation):raise ValueError('Explicit SearchValidation feedback required')
        wanted=pd.DatetimeIndex(request['target_times'])
        labels=pd.DatetimeIndex(validation.reference.time.values)
        if not labels.isin(wanted).all():raise ValueError('Validation timestamps outside target cohort')
        if validation.variable!=request['variable'] or validation.reference.attrs.get('units')!=request['unit']:
            raise ValueError('Validation variable/unit mismatch')
        if evaluation is not None and hasattr(evaluation,'reference') and 'time' in evaluation.reference.coords:
            if labels.isin(pd.DatetimeIndex(evaluation.reference.time.values)).any():raise ValueError('Selection validation overlaps independent evaluation timestamps')
        declaration=engine.config.get('reconstruction')
        if declaration and labels.isin(pd.DatetimeIndex(engine.sources[declaration['fine_source']].time.values)).any():
            raise ValueError('Validation fine labels are already visible source anchors')
    options = engine.config.get("search",{})
    if set(options)-{"screen_pool","survivors","generations","population","exact_check","backend","screen_limit","initialization"}:
        raise ValueError("Unknown composition search option")
    population = options.get("population",4); generations = options.get("generations",4)
    screen_pool = options.get("screen_pool",16); survivors = options.get("survivors",4)
    if any(type(v)!=int or not 1<=v<=10000 for v in (population,screen_pool,survivors)) or not 0<=generations<=100:
        raise ValueError("Invalid composition search budget")
    limit = budget.get("objective_limit",32); seconds = budget.get("wall_seconds",120)
    proposal_limit=budget.get("proposal_limit",50)
    if type(limit)!=int or not 1<=limit<=10000 or not 0<seconds<=3600:
        raise ValueError("Composition budget needs 1..10000 executions and <=3600 seconds")
    if type(proposal_limit)!=int or not 1<=proposal_limit<=100000:
        raise ValueError("Composition needs a finite positive mutation proposal limit")
    exact = options.get("exact_check",False)
    if not isinstance(exact,bool) or (exact and engine.count>limit):
        raise ValueError("Exact audit requires enough budget for the entire declared grammar")
    protocol = dict(version=VERSION,source_ids=engine.ids,grammar_count=engine.count,
        axes=[dict(field=s[0],window=s[1],branch=s[2],alternatives=len(a),choices=a) for s,a in zip(engine.slots,engine.axes)],
        counting="Product of distinct declared branch chains, fusion role/weight assignments and postchains per field/time window; parameter values are distinct, no numeric-output uniqueness claim",
        request=request,inputs=audit,seed=seed,
        portraits=[dict(variable=f["variable"],importance=f["weight"],baseline=engine.baselines[f["variable"]],model=f["model"]) for f in engine.fields],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name("portrait_state.py"),Path(__file__).with_name("portrait_rules.py")]})
    if engine.config.get("response_profile"):
        protocol["response_portrait_snapshot"]=deepcopy(engine.fields[0]["response_portrait"].profile)
    if validation is not None:
        path=folder/'search_validation.nc';path.parent.mkdir(parents=True,exist_ok=True)
        validation.reference.to_dataset(name=validation.variable).to_netcdf(path,engine='h5netcdf')
        protocol['search_validation']=dict(protocol=validation.protocol,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    protocol["visible_input_snapshots"] = []
    for key,original in engine.sources.items():
        path=folder/"inputs"/(digest(key)+".nc");path.parent.mkdir(parents=True,exist_ok=True)
        ds=original.copy(deep=True)
        for obj in [ds,*ds.variables.values()]:
            obj.attrs={k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in obj.attrs.items() if v is not None}
        ds.to_netcdf(path,engine="h5netcdf")
        protocol["visible_input_snapshots"].append(dict(source_id=key,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    protocol["resource_snapshots"]=[]
    for index,(role,original) in enumerate(engine.resources):
        path=folder/"inputs"/(role+"_"+str(index)+".nc")
        original.to_netcdf(path,engine="h5netcdf")
        protocol["resource_snapshots"].append(dict(role=role,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(folder/"request.json",protocol)
    rng = random.Random(seed)
    screen_limit=options.get("screen_limit",100000)
    if type(screen_limit)!=int or screen_limit<engine.count:
        raise ValueError("Full three-portrait screening exceeds explicit screen_limit; narrow the grammar, never randomly truncate")
    screened=[];rejected=[];screen_started=perf_counter()
    for index in range(engine.count):
        genome=engine.decode(index);contract=engine.preflight(genome)
        if contract["status"]=="REJECT":rejected.append(dict(index=index,**contract));continue
        matching=engine.screen(genome)
        screened.append(dict(index=index,scientific_steps=engine.steps(genome),score_interval=matching["score_interval"],
            **({"routes":matching['routes']} if 'routes' in matching else {}),
            **({'common_gain':matching['common_gain']} if 'common_gain' in matching else {})))
    from ecodata.algorithms.composition_scoring import gain_priority
    screened.sort(key=lambda r:(-r["score_interval"][0],*[-v for v in gain_priority(r)],r["scientific_steps"],r["index"]))
    initialization=options.get('initialization','portrait')
    if initialization not in {'portrait','uniform_control'} or (initialization=='uniform_control' and (validation is None or options.get('backend') not in {'random','alns'})):
        raise ValueError('Uniform initialization is an explicit validation random/ALNS ablation control only')
    seeds=[r['index'] for r in screened[:survivors]]
    if initialization=='uniform_control':seeds=rng.sample([r['index'] for r in screened],min(survivors,len(screened)))
    screening=dict(rows=screened,rejected=rejected,grammar_count=engine.count,scored_count=len(screened),
        rejected_count=len(rejected),complete=True,sampling=False,deprecated_screen_pool_ignored=options.get("screen_pool"),
        survivors=seeds,initialization=initialization,reference_used=False,seconds=perf_counter()-screen_started)
    save(folder/"initial_screening.json",screening)
    if engine.parallel_scorer is not None:
        save(folder/'parallel_scoring.json',engine.parallel_scorer.receipt())
    if not screened:raise ValueError("No contract-compatible combination after complete screening")
    rows = {}; proposals = []
    legal_indices={r["index"] for r in screened}
    def evaluate(index,origin):
        if index in rows:return rows[index]
        if origin!="exact_audit" and index not in legal_indices:return None
        if len(rows)>=limit or perf_counter()-started>=seconds:return None
        genome = engine.decode(index); key = digest(engine.plan(genome)); local = folder/"executions"/key
        execution_started=perf_counter()
        try:
            ds,matching,trace,checks = engine.execute(genome)
            local.mkdir(parents=True,exist_ok=True)
            product_path = local/"product.nc"
            for obj in [ds,*ds.variables.values()]:
                obj.attrs = {k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in obj.attrs.items() if v is not None}
            ds.to_netcdf(product_path,engine="h5netcdf")
            with xr.open_dataset(product_path,engine="h5netcdf") as readback:xr.testing.assert_allclose(ds,readback.load())
            from ecodata.algorithms.dag import Artifact
            numeric = ds.copy(deep=False);numeric.attrs = {}
            for name in numeric.variables:numeric[name].attrs = {}
            row = dict(status="PASS",key=key,index=index,origin=origin,plan=engine.plan(genome),matching=matching,
                scientific_steps=engine.steps(genome),
                product=str(product_path),product_sha256=hashlib.sha256(product_path.read_bytes()).hexdigest(),numeric_fingerprint=Artifact("composition_product",numeric).fingerprint(),file_readback="PASS",trace=trace,coordination=checks)
            if validation is not None:row['search_validation']=validation(ds)
        except Exception as exc:
            row = dict(status="FAIL",key=key,index=index,origin=origin,plan=engine.plan(genome),error=str(exc),trace=getattr(engine,"last_trace",[]),coordination=getattr(engine,"last_checks",[]))
        row.update(execution_seconds=perf_counter()-execution_started,completed_seconds=perf_counter()-started)
        rows[index] = row; save(local/"result.json",row)
        if progress:progress(dict(event="composition_execution",index=index,status=row["status"],executions=len(rows)))
        return row
    for index in seeds:evaluate(index,'uniform_control' if initialization=='uniform_control' else "three_portrait_screen")
    def ranking():
        return sorted([r for r in rows.values() if r["status"]=="PASS"],key=lambda r:((round(r['search_validation']['RMSE'],9),) if validation is not None else ())+(-round(r["matching"]["score_interval"][0],9),*[-v for v in gain_priority(r['matching'])],r["scientific_steps"],r["index"]))
    initial = ranking(); initial_score = initial[0]["matching"]["score_interval"][0] if initial else None
    initial_validation=initial[0]['search_validation']['RMSE'] if initial and validation is not None else None
    backend=options.get("backend","basic")
    if backend not in {"basic","alns","pymoo","random","none"}:raise ValueError("Unknown composition optimiser")
    native_optimizer=dict(backend=backend)
    if backend in {"alns","pymoo","random"}:
        from ecodata.algorithms.composition_optimizers import search
        seeds=[r["index"] for r in ranking()] or [r["index"] for r in screened[:survivors]]
        native_optimizer=search(backend,engine,seeds,screened,evaluate,ranking,
            lambda:len(rows)>=limit or len(proposals)>=proposal_limit or perf_counter()-started>=seconds,
            seed=seed,proposals=proposals,population=population,generations=generations)
    for generation in range(generations if backend=="basic" else 0):
        parents = [r["index"] for r in ranking()[:population]] or [r["index"] for r in screened[:survivors]]
        for parent in parents:
            if len(proposals)>=proposal_limit or len(rows)>=limit or perf_counter()-started>=seconds:break
            genome = list(engine.decode(parent))
            mutable = [i for i,n in enumerate(engine.sizes) if n>1]
            if not mutable:break
            size=rng.randint(1,len(mutable)) if rng.random()<.25 else 1
            edits=[]
            for axis in rng.sample(mutable,size):
                old=genome[axis]
                genome[axis]=rng.choice([v for v in range(engine.sizes[axis]) if v!=old])
                edits.append(dict(axis=engine.slots[axis],old_choice=old,new_choice=genome[axis]))
            child = engine.encode(genome)
            proposals.append(dict(generation=generation,parent=parent,child=child,operator="change_source_chain_or_role_or_schedule",edits=edits,
                axis=edits[0]["axis"],old_choice=edits[0]["old_choice"],new_choice=edits[0]["new_choice"],cached=child in rows))
            evaluate(child,"mutation")
        if len(proposals)>=proposal_limit or len(rows)>=limit or perf_counter()-started>=seconds:break
    mutation_ranked = ranking()
    mutation_best = mutation_ranked[0] if mutation_ranked else None
    mutation_score = mutation_best["matching"]["score_interval"][0] if mutation_best else None
    before_exact = len(rows)
    if exact:
        for i in range(engine.count):evaluate(i,"exact_audit")
    ranked = ranking(); complete = len(rows)==engine.count
    best = ranked[0] if ranked else None
    # Independent truth is accessed only after recommendation is frozen.
    independent = None
    if best and evaluation is not None:
        try:
            with xr.open_dataset(best["product"],engine="h5netcdf") as ds:independent=evaluation(ds.load())
        except Exception as exc:
            independent = dict(status="FAIL",error=str(exc),recommendation_already_frozen=True)
    certified = complete and best is not None and best["matching"]["score_interval"][0]>=max((r["matching"]["score_interval"][1] for r in ranked[1:]),default=-float("inf"))
    counts = dict(grammar_combinations=engine.count,evaluated=len(rows),usable_observed=len(ranked),failed_observed=sum(r["status"]=="FAIL" for r in rows.values()),
        exact_usable_count=len(ranked) if complete else None,exact_count_complete=complete,
        distinct_numeric_products_observed=len({r["numeric_fingerprint"] for r in ranked}),
        exact_distinct_numeric_products=len({r["numeric_fingerprint"] for r in ranked}) if complete else None,
        structural_variants="Role/order/window/parameters included; equal numerical products may still exist")
    result = dict(service="PlatformSearch/Composition",version=VERSION,rows=list(rows.values()),best=best,recommended_key=best["key"] if best else None,
        proposals=proposals,initial_best_score=initial_score,final_best_score=best["matching"]["score_interval"][0] if best else None,
        mutation_best_score=mutation_score,mutation_recommended_key=mutation_best["key"] if mutation_best else None,
        mutation_improvement=mutation_score-initial_score if mutation_score is not None and initial_score is not None else None,
        mutation_new_executions=sum(r["origin"]=="mutation" for r in rows.values()),exact_audit_additional_executions=len(rows)-before_exact,
        screening=dict(complete=True,scored=len(screened),structurally_rejected=len(rejected),sampling=False),native_optimizer=native_optimizer,
        selection_score_tolerance=1e-9,
        counts=counts,evolution_used=bool(proposals),optimizer=native_optimizer,independent_evaluation=independent,
        evaluation_protocol=getattr(evaluation,"protocol",{}),reference_used_for_selection=False,
        stop_reason="COMPLETE_GRAMMAR_AUDIT" if complete else "BUDGET_OR_GENERATIONS_COMPLETED",
        certification="EXACT_MEASURED_PORTRAIT_OPTIMUM_IN_DECLARED_DOMAIN" if certified else "NOT_CERTIFIED",
        true_scientific_global_optimum="NOT_PROVEN",native_model_validation="NOT_ASSESSED",total_service_seconds=perf_counter()-started)
    if validation is not None:
        result.update(selection_objective='search_validation_RMSE',initial_validation_rmse=initial_validation,
            mutation_validation_rmse=mutation_best['search_validation']['RMSE'] if mutation_best else None,
            final_validation_rmse=best['search_validation']['RMSE'] if best else None,
            certification='EXACT_VALIDATION_RMSE_OPTIMUM_IN_DECLARED_DOMAIN' if complete and best else 'NOT_CERTIFIED',
            search_validation_used_for_selection=True,independent_test_used_for_selection=False)
    if engine.parallel_scorer is not None:
        result['parallel_scoring']=dict(weights=engine.parallel_scorer.weights,
            probe_graph_executions=engine.parallel_scorer.executions,
            probe_kernel_calls=engine.parallel_scorer.kernel_calls,seconds=engine.parallel_scorer.seconds,
            budget_scope='Visible-mask probe executions are additional to objective_limit; included in total_service_seconds and reported separately',
            unique_score_intervals=len({tuple(r['score_interval']) for r in screened}),
            known_score_fraction=sum(r['score_interval'][0]==r['score_interval'][1] for r in screened)/len(screened))
    save(folder/"composition_counts.json",counts);save(folder/"mutation_trace.json",proposals);save(folder/"summary.json",result)
    return result
