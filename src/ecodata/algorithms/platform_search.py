"""Public upload/toolkit protocol for current random-mutation processing.

Task adapters own scientific ports; this module owns parsing, dispatch and the
common evaluation/execution envelope. No research runner is imported.
"""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
from time import perf_counter
import hashlib,json,traceback
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from ecodata.algorithms.numerical_search import record,save
from ecodata.algorithms.contracts import AlgorithmContext,AlgorithmResult
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.capabilities import capability_catalog,describe_algorithm
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline,AlgorithmQualityGate
from ecodata.algorithms.dag import Artifact,Port,NodeAdapter,DAGError
from ecodata.algorithms.golem_generation import TypedGolemFactory
from ecodata.core.schemas import EcoDataBundle
from ecodata.recognition import recognise_file,inspect_dataset
from ecodata.evaluation.quality_gates import RawInputQualityGate

VERSION="platform_search_v15_measured_validation_random"
DEFAULT_BUDGET=dict(objective_limit=5,proposal_limit=50,callback_limit=100,wall_seconds=30)
CURRENT_INPUT_BUDGET=dict(objective_limit=1800,total_call_limit=1800,proposal_limit=400000,wall_seconds=3600,probe_call_limit=600)


def with_recommendations(request, recommendations):
    """Bind advisory recipes to a scientific quantity, never to a winning score."""
    selected=[]
    for row in recommendations or []:
        if row.get('advisory_only'):continue
        scope=row.get('scope',{})
        if not scope.get('variable'):
            raise ValueError('Search recommendation requires an explicit variable scope')
        if scope['variable']!=request['variable']:continue
        steps=row.get('steps')
        if not isinstance(steps,list) or not 1<=len(steps)<=8 or any(not isinstance(s,str) for s in steps):
            raise ValueError('Invalid recommended recipe')
        selected.append(dict(id=str(row.get('task',''))+':'+str(row.get('id','')),
            steps=steps,reason=str(row.get('reason',''))[:600],scope=scope))
    if len(selected)>16:raise ValueError('At most 16 recommended recipes per request')
    return dict(request,recommended_recipes=selected) if selected else request
FAMILIES={
    "spatial":{"single_source_scale","constrained_downscaling","learned_reconstruction","learned_climate_downscaling","terrain_downscaling","terrain_weather_downscaling","vertical_interpolation","soil_profile","observation_interpolation","raster_reconstruction","spatial_filter","geometric_alignment","spatiotemporal_reconstruction"},
    "temporal":{"temporal_scale","temporal_statistics","irregular_time","temporal_reconstruction","spatiotemporal_reconstruction"},
    "multisource":{"multi_source_fusion","bias_correction","observation_interpolation","multi_source_feature_preparation","multi_source_feature_extraction","spatial_filter","geometric_alignment","spatiotemporal_reconstruction","temporal_reconstruction"},
    "model_preprocessing":{"model_preprocessing"},
    "constraint":{"raster_constraint","validation"},
}


@dataclass(frozen=True)
class SearchInput:
    filename: str
    role: str
    content: bytes
    metadata: dict | None=None


def platform_registry(*,weights=None):
    # These are existing registrations, not new numerical algorithms.
    from ecodata.interface.raster_upload import upload_registry
    from ecodata.algorithms.mixed_gap_dag import setup
    registry=upload_registry()
    temporal,_=setup(build_factory=False)
    for spec in temporal.specs():registry.register(temporal.get(spec.id))
    from ecodata.algorithms.composition_decision import PrimaryWeightedMean
    registry.register(PrimaryWeightedMean())
    from ecodata.algorithms.spatiotemporal_detail import SpatiotemporalDetail
    registry.register(SpatiotemporalDetail())
    from ecodata.algorithms.raster_temporal import RasterTimeInterpolation,VisibleAnchorResidual
    for method in ('linear','nearest'):registry.register(RasterTimeInterpolation(method))
    registry.register(VisibleAnchorResidual())
    from ecodata.algorithms.spatial_median import SpatialMedian
    registry.register(SpatialMedian())
    from ecodata.algorithms.daily_reconstruction import DailyRecovery
    for method in ("linear","nearest","ffill","month_mean"):
        registry.register(DailyRecovery(method))
    from ecodata.algorithms.series_fusion import SeriesMissingFallback,PriorSeriesCalibration,SeriesNonnegative
    registry.register(SeriesMissingFallback())
    registry.register(SeriesNonnegative())
    for method in ('affine','ratio','monthly_ratio'):registry.register(PriorSeriesCalibration(method))
    from ecodata.algorithms.station_grid_fusion import StationGridFusion
    for method in ('affine','ratio','monthly_ratio'):registry.register(StationGridFusion(method))
    if weights:
        from ecodata.algorithms.raster_reconstruction_task import LocalLIIFRaster
        registry.register(LocalLIIFRaster(weights))
    return registry


def parse_inputs(inputs):
    parsed=[];audit=[]
    for item in inputs:
        ds,info=recognise_file(item.filename,item.content)
        meta=item.metadata or {}
        if meta.get("source_id"):
            if ds.attrs.get("source_id") not in {None,meta["source_id"]}:raise ValueError("Conflicting stable source identity")
            ds.attrs["source_id"]=meta["source_id"]
        if "crs" in meta:
            declared=CRS.from_user_input(meta["crs"])
            if ds.attrs.get("crs") and CRS.from_user_input(ds.attrs["crs"])!=declared:raise ValueError("Declared/file CRS conflict")
            ds.attrs.update(crs=declared.to_string(),crs_evidence=meta.get("crs_evidence","user declaration"))
        # Explicit field declarations only. A declaration never replaces a
        # conflicting original unit/semantics silently.
        for name,attrs in meta.get("variables",{}).items():
            if name not in ds:raise ValueError("Declared variable absent: "+name)
            for key,value in attrs.items():
                if key in ds[name].attrs and str(ds[name].attrs[key])!=str(value):raise ValueError(f"Conflicting {name}.{key}")
                ds[name].attrs[key]=value
        parsed.append((item.role,ds))
        audit.append(dict(role=item.role,descriptor=replace(inspect_dataset("parsed_input",ds),format=info.format).as_dict(),
                          source_filename=item.filename,sha256=hashlib.sha256(item.content).hexdigest(),declarations=meta))
    return parsed,audit


class DatasetReference:
    def __init__(self,reference,variable,*,role="validation"):
        if role not in {"validation","independent_validation"}:raise ValueError("Search reference must be validation, never final test")
        self.reference=reference.copy(deep=True);self.variable=variable
        self.mask=np.isfinite(reference.values)
        if not self.mask.any():raise ValueError("No finite reference values; natural missing is not truth")
        self.protocol=dict(kind="reference_error",role=role,objective="RMSE",reference_sha256=hashlib.sha256(reference.values.tobytes()).hexdigest())
        self.protocol.update(variable=variable,unit=reference.attrs.get("units","UNKNOWN"),queue_fingerprint=Artifact("validation_coordinates",xr.Dataset(coords=reference.coords)).fingerprint())
    def __call__(self,ds):
        if self.reference.attrs.get("units") and ds[self.variable].attrs.get("units")!=self.reference.attrs["units"]:raise ValueError("Independent reference/prediction unit mismatch")
        for name,coordinate in self.reference.coords.items():
            if name not in ds[self.variable].coords or not np.array_equal(ds[self.variable][name].values,coordinate.values):raise ValueError("Reference/prediction coordinate queue mismatch: "+name)
        prediction,reference=xr.align(ds[self.variable],self.reference,join="exact")
        if prediction.dims!=reference.dims:prediction=prediction.transpose(*reference.dims)
        p=prediction.values[self.mask];y=reference.values[self.mask]
        if not np.isfinite(p).all():raise ValueError("Incomplete fixed reference cohort; do not rank a successful subset")
        d=p-y
        return dict(objective=float(np.sqrt(np.mean(d*d))),RMSE=float(np.sqrt(np.mean(d*d))),MAE=float(np.abs(d).mean()),Bias=float(d.mean()),denominator=len(d),accuracy_assessed=True)


class SearchValidation(DatasetReference):
    """Explicit search feedback, separate from final post-selection evaluation."""
    def __init__(self,reference,variable):
        if reference.attrs.get('split')!='search_validation':
            raise ValueError('Explicit search_validation labels required; never final test')
        super().__init__(reference,variable)
        self.protocol.update(role='search_validation',used_for_selection=True)

    def __call__(self,ds):
        # A fixed subset of timestamps; no successful-subset scoring.
        return super().__call__(ds.sel(time=self.reference.time))


class EngineeringMetric:
    def __init__(self,variable,metric="coverage",*,operator=None,coarse=None):
        if metric not in {"coverage","coarse_consistency"}:raise ValueError("Unsupported declared engineering metric")
        if metric=="coarse_consistency" and (operator is None or coarse is None):raise ValueError("Coarse metric needs actual observation operator and observed coarse values")
        self.variable=variable;self.metric=metric;self.operator=operator;self.coarse=coarse
        self.protocol=dict(kind="declared_task_metric",objective=metric,accuracy_claim=False)
        if operator is not None:self.protocol.update(operator_fingerprint=Artifact("observation_operator",operator).fingerprint(),observations_sha256=hashlib.sha256(np.asarray(coarse).tobytes()).hexdigest())
    def __call__(self,ds):
        a=ds[self.variable].values
        value=-float(np.isfinite(a).mean()) if self.metric=="coverage" else float(np.sqrt(np.mean((self.operator@a.ravel()-self.coarse)**2)))
        return dict(objective=value,accuracy_assessed=False,engineering_metric=self.metric)


def make_context(parsed,request,*,allow_unaligned_time=False):
    variable=request["variable"];sources=[];aux=[];observations=None;state=None;target_grid=None;training={};prior_resources={}
    for role,ds in parsed:
        if role in {"primary","background"}:sources.append(EcoDataBundle(ds.copy(deep=False)))
        elif role=="auxiliary":aux.append(ds)
        elif role=="observations":
            from ecodata.io.structures import table_frame
            observations=table_frame(ds)
        elif role=="state":state=ds
        elif role=="target":target_grid=ds
        elif role in {"training_reference","historical_simulation"}:
            if variable not in ds:raise ValueError("Training variable absent")
            training[role]=ds[variable]
            if role in prior_resources:raise ValueError('Ambiguous repeated prior resource role: '+role)
            prior_resources[role]=ds
        else:raise ValueError("Role requires the station/covariate task adapter: "+role)
    if not sources:raise ValueError("A primary data source is required")
    unit=request.get("unit")
    if not unit or unit=="UNKNOWN":raise ValueError("UNKNOWN unit requires confirmation before numerical execution")
    semantic=request.get("semantic_type")
    if not semantic or semantic=="UNKNOWN":raise ValueError("Unknown variable semantics")
    if request["task"]=="spatial" and semantic in {"cell_total","integer_count"}:raise ValueError("Cell totals require an explicit area/integral conversion contract; current generic raster resamplers are not total allocators")
    input_variable=request.get("source_variable",variable)
    quality=[]
    for source in sources:
        if input_variable not in source.dataset:raise ValueError("Target/source variable absent")
        a=source.dataset[input_variable]
        if a.attrs.get("units")!=unit:raise ValueError("Input/request unit conflict")
        if semantic in {'continuous','categorical'}:
            declared=a.attrs.get('data_kind')
            if declared not in {None,'','UNKNOWN',semantic}:raise ValueError('Input data kind conflicts with confirmed request semantics')
            if declared in {None,'','UNKNOWN'}:
                a.attrs=dict(a.attrs,data_kind=semantic,data_kind_evidence='Explicit confirmed public request semantic_type; values unchanged')
        has_geometry=any(name in source.dataset.coords for name in ("lat","lon","x","y","longitude","latitude"))
        if request["task"] in {"spatial","multisource"} and has_geometry:
            if not source.dataset.attrs.get("crs"):raise ValueError("UNKNOWN CRS requires confirmation")
            CRS.from_user_input(source.dataset.attrs["crs"])
        for key in ("spatial_semantics","temporal_semantics"):
            if request.get(key) and request[key]!="UNKNOWN" and a.attrs.get(key)!=request[key]:raise ValueError("Confirm file/declared "+key+" before execution")
        report=RawInputQualityGate().validate(source.dataset,variable=input_variable,requires_spatial=request["task"] in {"spatial","multisource"} and has_geometry,expected_unit=unit)
        if report["status"]=="BLOCK":raise ValueError("RawInputQualityGate BLOCK: "+str(report["checks"]))
        quality.append(report)
    if len(sources)>1:
        definitions={s.dataset[input_variable].attrs.get("physical_definition") for s in sources}
        if None in definitions or len(definitions)!=1:raise ValueError("Multiple same-variable sources need confirmed identical physical definitions; covariates use auxiliary role")
        if request.get("_local_problem_mode"):
            supports={s.dataset[input_variable].attrs.get("spatial_semantics","UNKNOWN") for s in sources}
            if len(supports)!=1 or "UNKNOWN" in supports:raise ValueError("Source spatial support differs or is UNKNOWN; no point/area observation conversion has been declared")
        for other in sources[1:]:
            for key in ("crs",):
                if CRS.from_user_input(other.dataset.attrs[key])!=CRS.from_user_input(sources[0].dataset.attrs[key]):raise ValueError("Source CRS mismatch")
            if not allow_unaligned_time and "time" in sources[0].dataset.coords and not np.array_equal(sources[0].dataset.time,other.dataset.time):raise ValueError("Source time windows differ; align explicitly")
    for ds in aux:
        if not ds.attrs.get("crs"):raise ValueError("Auxiliary CRS required")
        if sources[0].dataset.attrs.get("crs") and CRS.from_user_input(ds.attrs["crs"])!=CRS.from_user_input(sources[0].dataset.attrs["crs"]):raise ValueError("Auxiliary CRS mismatch")
        if any(not a.attrs.get("units") for a in ds.data_vars.values()):raise ValueError("Auxiliary variable units must be declared independently of target")
    if observations is not None:
        if "split" not in observations or observations.split.eq("test").any():raise ValueError("Fitting ports may not receive final test labels")
        if "unit" not in observations or not observations.unit.eq(unit).all():raise ValueError("Observation unit mismatch")
    params=dict(request.get("parameters",{}))
    if request.get("_local_problem_mode"):
        params["source_identities"]=[dict(role=role,source_id="role:"+role+":"+str(ds.attrs.get("source_id",request["variable"]))) for role,ds in parsed if role in {"primary","background"}]
    params.update(training)
    if prior_resources:params['prior_resources']=prior_resources
    params['allow_training']=bool(request.get('allow_training',False))
    needs_observation_operator=any(request.get("requirements",{}).get(key,False) for key in ("require_coarse_consistency","try_coarse_consistency"))
    if needs_observation_operator:
        if request["task"]!="spatial" or semantic!="continuous" or request.get("target_lat") is None or request.get("target_lon") is None:raise ValueError("Declared coarse consistency requires a continuous spatial target grid")
        from scipy import sparse
        from ecodata.algorithms.raster_reconstruction_task import edges,overlap_weights
        coarse=sources[0].dataset
        if coarse[input_variable].dims!=("lat","lon") or not np.isfinite(coarse[input_variable]).all():raise ValueError("Original coarse observation operator needs a finite 2D field")
        if coarse[input_variable].attrs.get("spatial_semantics") not in {"cell_mean","flux"}:raise ValueError("Automatic area observation operator needs declared pixel-average/flux support; no conservation meaning is guessed")
        ay=overlap_weights(edges(request["target_lat"]),edges(coarse.lat),True)
        ax=overlap_weights(edges(request["target_lon"]),edges(coarse.lon))
        params.update(degradation_operator=sparse.kron(ay,ax,format="csr"),coarse_reference=coarse)
    if state is not None:params["fitted_state"]=state
    if target_grid is not None:params["target_grid"]=target_grid
    constraints=dict(request.get("constraints",{}),unit=unit,output_unit=request.get("output_unit",unit),nonnegative=bool(request.get("nonnegative",False)))
    context=AlgorithmContext(tuple(sources),variable,semantic,
        tuple(request["target_lat"]) if request.get("target_lat") is not None else None,
        tuple(request["target_lon"]) if request.get("target_lon") is not None else None,
        observations=observations,auxiliaries=tuple(aux),model_constraints=constraints,parameters=params,
        validation_role="validation" if request.get("allow_training") and observations is not None and observations.split.eq("validation").any() else None,
        input_quality=dict(status="WARN" if any(q["status"]=="WARN" for q in quality) else "PASS",checks=[c for q in quality for c in q["checks"]]))
    return context,quality,input_variable


def binding_reason(spec,context,request):
    from ecodata.algorithms.plugins import CatalogOnlyPlugin
    if spec.category not in FAMILIES[request["task"]]:return "Different scientific task/layer"
    statistic=spec.input_contract.get("spatial_statistic")
    if statistic in {"min","max","median"} and context.parameters.get("spatial_statistic")!=statistic:return "Spatial min/max/median summaries require the corresponding declared statistic, not ordinary resampling"
    if request.get("operation")=="spatial_statistics" and statistic!=context.parameters.get("spatial_statistic"):return "Output spatial statistic differs from the requested statistic"
    if statistic=="fractions" and request.get("output_representation")!="category_fractions":return "Category fractions are composition features, not category-code output; declare output_representation"
    if request.get("output_representation")=="category_fractions" and spec.output_contract.get("kind")!="category_fractions":return "Requested category composition, not category-code sampling"
    if request.get("purpose")=="features" and spec.category not in {"multi_source_feature_extraction","multi_source_feature_preparation"}:return "Requested features, not a target-variable fusion prediction"
    if spec.status!="available":return "Status: "+spec.status
    if spec.requires_training and not request.get("allow_training",False):return "Training is not authorised; no generic inference-state API is declared"
    if spec.requires_weights:
        if spec.id=="raster_task.liif_aod":
            if not request.get("weights"):return "Explicit AOD checkpoint required"
        elif not request.get("allow_existing_weights",False):return "Explicit consent to verified existing task weights required; an AOD checkpoint does not authorise other models"
    if request["task"]=="temporal":
        if context.parameters.get("target_time_frequency") and spec.category=="temporal_reconstruction":return "Label restoration is not semantic interval aggregation"
        if request.get("temporal_semantics") in {"mean","interval_mean","accumulation","interval_total"} and spec.category not in {"temporal_scale","temporal_statistics"}:return "No interval-statistic contract for this temporal operator"
    if spec.input_contract.get("requires_fitted_state"):return "Use existing hourly state-bound adapter; this plugin.run does not consume that state"
    if spec.category=="model_preprocessing" and (not request.get("model") or context.parameters.get("model_preprocessing") is not True):return "Explicit model task and preprocessing opt-in required; not a generic evolutionary node"
    if request["task"]=="multisource" and context.auxiliaries and spec.category=="observation_interpolation" and not context.parameters.get("include_station_only_baselines", False):
        if "external_drift_estimation" not in describe_algorithm(spec).capabilities and "auxiliary_variable_estimation" not in describe_algorithm(spec).capabilities:return "This station operator does not use the declared covariate input"
    return None


def bind_registry(registry,context,decision,request):
    adapters={};external={};inputs={};source_count=len(context.sources)
    resources=[("source"+str(i),b) for i,b in enumerate(context.sources)]
    resources += [("auxiliary"+str(i),a) for i,a in enumerate(context.auxiliaries)]
    if context.observations is not None:resources.append(("observations",context.observations))
    if "fitted_state" in context.parameters:resources.append(("state",context.parameters["fitted_state"]))
    for key in ("training_reference","historical_simulation"):
        if key in context.parameters:resources.append((key,context.parameters[key]))
    if isinstance(context.parameters.get("target_grid"),xr.Dataset):resources.append(("target_grid",context.parameters["target_grid"]))
    for key in ("degradation_operator","coarse_reference"):
        if key in context.parameters:resources.append((key,context.parameters[key]))
    for key,value in resources:
        kind=VERSION+"_"+key;inputs[key]=Port(kind,("contract",))
        external[key]=Artifact(kind,value,dict(contract=VERSION))
        def read(_i,p,e,key=key):return {"value":e[key]}
        adapters["source."+key,"run"]=NodeAdapter({}, {"value":inputs[key]},read,decision_contract=dict(operation="source",capabilities=[]))
    resource_contracts={}
    for index,source in enumerate(context.sources):
        key="source"+str(index)
        resource_contracts[key]=dict(context.parameters.get("source_identities",[dict(role="primary",source_id="role:source"+str(i)+":"+context.variable) for i in range(source_count)])[index],variable=context.variable)
    # Local preprocessing uses existing Registry kernels and explicit parent
    # ports. This is not an alternative fusion implementation.
    if request.get("_local_problem_mode") and request["task"]=="multisource" and context.target_lat is not None and context.target_lon is not None:
        for index,source in enumerate(context.sources):
            key="source"+str(index);ds=source.dataset
            if not {"lat","lon"}<=set(ds.coords):raise ValueError("Local grid adaptation requires actual rectilinear coordinates")
            mismatch=not(np.array_equal(ds.lat,context.target_lat) and np.array_equal(ds.lon,context.target_lon))
            ready=VERSION+"_aligned_"+resource_contracts[key]["source_id"]
            resource_contracts[key].update(required_grid_kind=ready,grid_mismatch=mismatch)
            raw=inputs[key]
            inputs[key]=Port(ready,("contract",))
            if not mismatch:
                external[key].kind=ready
                adapters["source."+key,"run"].outputs["value"]=inputs[key]
                continue
            for spec in registry.specs():
                from ecodata.algorithms.plugins import RegridPlugin
                if not isinstance(registry.get(spec.id),RegridPlugin):continue
                local=replace(context,sources=(source,),auxiliaries=(),observations=None)
                check=registry.get(spec.id).compatibility(local)
                if not check.eligible:continue
                operation="prepare.grid."+resource_contracts[key]["source_id"]+"."+spec.id
                def align(values,p,e,identifier=spec.id,key=key,output_kind=inputs[key].kind):
                    upstream=values["source"].value
                    actual=replace(context,sources=(upstream,),auxiliaries=(),observations=None)
                    plugin=registry.get(identifier);check=plugin.compatibility(actual)
                    if not check.eligible:raise DAGError("Local source compatibility: "+str(check.reasons))
                    result=plugin.run(actual);output=result.data.dataset
                    for name,field in upstream.dataset.data_vars.items():
                        if name==context.variable:continue
                        a=np.asarray(field.values)
                        if not np.isfinite(a).all() or not np.all(a==a.flat[0]):raise DAGError("No declared propagation model for nonconstant auxiliary field: "+name)
                        output[name]=xr.full_like(output[context.variable],float(a.flat[0]));output[name].attrs=dict(field.attrs)
                    return {"value":Artifact(output_kind,result.data,dict(contract=VERSION))}
                adapters[operation,"run"]=NodeAdapter({"source":raw},{"value":inputs[key]},align,
                    decision_contract=dict(operation="compute",registered=False,capabilities=list(describe_algorithm(spec).capabilities),algorithm_id=spec.id,
                        source_ids=[resource_contracts[key]["source_id"]],input="source"))
    from ecodata.algorithms.collaboration import source_calibration, residual_stages
    collaboration_ids=source_calibration(registry,context,request,adapters,inputs)
    ids=[];excluded={}
    for spec in registry.specs():
        reason=binding_reason(spec,context,request)
        match=decision.matches[spec.id]
        if reason or not match["eligible"]:
            excluded[spec.id]=[reason] if reason else match["reasons"] or ["Capability or implementation not executable"]
            match.update(eligible=False,reasons=excluded[spec.id]);continue
        # Generic port binding delegates scientific work to the actual plugin.
        # The context is reconstructed from parent artifacts, not hidden data.
        def execute(values,p,e,identifier=spec.id):
            actual=replace(context,sources=tuple(values["source"+str(i)].value for i in range(source_count)),
                auxiliaries=tuple(values["auxiliary"+str(i)].value for i in range(len(context.auxiliaries))),
                observations=values["observations"].value if "observations" in values else None)
            if "state" in values:actual=replace(actual,parameters=dict(actual.parameters,fitted_state=values["state"].value))
            if "target_grid" in values:actual=replace(actual,parameters=dict(actual.parameters,target_grid=values["target_grid"].value))
            for key in ("training_reference","historical_simulation"):
                if key in values:actual=replace(actual,parameters=dict(actual.parameters,**{key:values[key].value}))
            check=registry.get(identifier).compatibility(actual)
            if not check.eligible:raise DAGError("Actual input contract rejected: "+str(check.reasons))
            result=registry.get(identifier).run(actual)
            if actual.sources[0].dataset.attrs.get("data_role")=="prediction_template" and registry.get(identifier).spec.category in {"observation_interpolation","multi_source_fusion"}:
                output=result.data.dataset.copy()
                output.attrs["data_role"]="derived_field"
                result=replace(result,data=EcoDataBundle(output,result.data.history))
            if request["task"]=="multisource" and len(actual.sources)==1:
                source=actual.sources[0].dataset
                if "time" in source.coords and source.time.ndim==0 and "time" not in result.data.dataset.coords:
                    result=replace(result,data=EcoDataBundle(result.data.dataset.assign_coords(time=source.time),result.data.history))
            return {"product":Artifact("platform_raw_product",result,dict(contract=VERSION))}
        adapters[spec.id,"run"]=NodeAdapter(dict(inputs),{"product":Port("platform_raw_product",("contract",))},execute,
            decision_contract=dict(operation="compute"))
        ids.append(spec.id)
    collaboration_ids+=residual_stages(registry,context,request,adapters,inputs)
    for identifier in set(collaboration_ids):
        ids.append(identifier);excluded.pop(identifier,None)
        decision.matches[identifier].update(eligible=True,reasons=[],binding="reviewed collaboration mode; actual parent input checked at execution")
    if not ids:raise ValueError("No compatible executable algorithm: "+json.dumps(excluded,ensure_ascii=False))
    # Existing constraint kernels operate on an upstream result, never as a
    # substitute for spatial reconstruction. Two stage kinds avoid graph cycles.
    field=context.sources[0].dataset.get(context.variable)
    if request["task"] in {"spatial","multisource"} and context.semantic_type=="continuous" and field is not None and field.dims==("lat","lon") and all(field[axis].ndim==1 for axis in ("lat","lon")) and context.target_lat is not None and context.target_lon is not None:
        for spec in registry.specs():
            if spec.category!="raster_constraint":continue
            if spec.status!="available" or spec.input_contract.get("prediction_stage")!="after_reconstruction":continue
            if not set(spec.input_contract.get("requires_parameters",()))<=set(context.parameters):continue
            def constrain(values,p,e,identifier=spec.id):
                upstream=values["field"].value
                params=dict(context.parameters)
                for key in ("degradation_operator","coarse_reference"):
                    if key in values:params[key]=values[key].value
                actual=replace(context,sources=(upstream.data,),parameters=params)
                plugin=registry.get(identifier)
                check=plugin.compatibility(actual)
                if not check.eligible:raise DAGError("Constraint input incompatible: "+str(check.reasons))
                output=plugin.run(actual)
                scalar_coords={name:coord for name,coord in upstream.data.dataset.coords.items()
                               if coord.ndim==0 and name not in output.data.dataset.coords}
                if scalar_coords:
                    output=replace(output,data=EcoDataBundle(output.data.dataset.assign_coords(scalar_coords),output.data.history))
                return {"product":Artifact("platform_constrained_product",output,dict(contract=VERSION))}
            ports={"field":Port("platform_raw_product",("contract",))}
            ports.update({key:inputs[key] for key in ("degradation_operator","coarse_reference") if key in inputs})
            adapters[spec.id,"run"]=NodeAdapter(ports,{"product":Port("platform_constrained_product",("contract",))},constrain,
                decision_contract=dict(operation="compute",input="field",capabilities=list(describe_algorithm(spec).capabilities),registered=False))
            ids.append(spec.id);excluded.pop(spec.id,None)
            decision.matches[spec.id].update(eligible=True,reasons=[],binding="postprocessor; actual upstream compatibility rechecked at execution")
    has_corrected=any(port.kind=="platform_corrected_product" for adapter in adapters.values() for port in adapter.outputs.values())
    if has_corrected:
        for key,adapter in list(adapters.items()):
            if key[0] in ids and key[1]=="run" and registry.get(key[0]).spec.category=="raster_constraint":
                ports=dict(adapter.inputs,field=Port("platform_corrected_product",("contract",)))
                adapters[key[0],"after_residual"]=replace(adapter,inputs=ports)
    def validate(values,p,e):
        result=values["field"].value
        output_semantic=registry.get(result.algorithm_id).spec.output_contract.get("semantic_type",context.semantic_type)
        report=AlgorithmQualityGate().validate(replace(context,semantic_type=output_semantic),result)
        if report["status"]=="BLOCK":raise DAGError("Final quality BLOCK: "+str(report["checks"]))
        for axis,target in (("lat",context.target_lat),("lon",context.target_lon)):
            if target is not None and (axis not in result.data.dataset.coords or not np.array_equal(result.data.dataset[axis],target)):raise DAGError("Output target grid mismatch")
        target_grid=context.parameters.get("target_grid")
        if isinstance(target_grid,xr.Dataset):
            for axis,coordinate in target_grid.coords.items():
                if axis not in result.data.dataset.coords or not np.array_equal(result.data.dataset[axis],coordinate):raise DAGError("Output declared target-coordinate mismatch: "+axis)
        frequency=context.parameters.get("target_time_frequency")
        if request["task"]=="temporal" and frequency:
            ds=context.sources[0].dataset
            if request.get("temporal_semantics")=="instantaneous":
                times=pd.DatetimeIndex(ds.time.values)
                expected=pd.date_range(times.min().ceil(frequency),times.max().floor(frequency),freq=frequency)
            else:
                from ecodata.temporal.semantic import _validated_bounds
                bounds=_validated_bounds(ds)
                expected=pd.date_range(pd.Timestamp(bounds[0,0]),pd.Timestamp(bounds[-1,1]),freq=frequency)[1:]
            if "time" not in result.data.dataset.coords or not np.array_equal(pd.DatetimeIndex(result.data.dataset.time).as_unit("ns"),expected.as_unit("ns")):raise DAGError("Output time grid/interval labels do not satisfy declared frequency")
        return {"field":Artifact("validated_field",result.data.dataset,dict(contract=VERSION,quality=report,algorithm=result.algorithm_id,algorithm_report=result.report))}
    adapters["quality.platform","run"]=NodeAdapter({"field":Port("platform_raw_product",("contract",))},{"field":Port("validated_field",("contract",))},validate,
        decision_contract=dict(operation="validate",input="field",capabilities=["nonnegative_validation"] if context.model_constraints["nonnegative"] else []))
    if has_corrected:
        adapters["quality.platform_corrected","run"]=NodeAdapter({"field":Port("platform_corrected_product",("contract",))},
            {"field":Port("validated_field",("contract",))},validate,
            decision_contract=dict(operation="validate",input="field",capabilities=["nonnegative_validation"] if context.model_constraints["nonnegative"] else []))
    if any(a.outputs.get("product",Port("none")).kind=="platform_constrained_product" for a in adapters.values()):
        adapters["quality.platform_constrained","run"]=NodeAdapter({"field":Port("platform_constrained_product",("contract",))},{"field":Port("validated_field",("contract",))},validate,
            decision_contract=dict(operation="validate",input="field",capabilities=["nonnegative_validation"] if context.model_constraints["nonnegative"] else []))
    registry.register_task_adapters(VERSION,adapters,algorithm_ids=ids,boundary_ids=[k[0] for k in adapters if k[0].startswith(("source.","prepare.","quality."))])
    from ecodata.algorithms.collaboration import CollaborationFactory
    factory=CollaborationFactory(adapters,ids,max_nodes=24,max_depth=12) if has_corrected else TypedGolemFactory(adapters,ids,max_nodes=24,max_depth=9)
    factory.SOURCES={"source."+key:key for key in external}
    factory.resource_contracts=resource_contracts
    return factory,external,excluded


class RegistryEvaluator:
    def __init__(self,registry,factory,external,decision,evaluation,folder,*,task_id=VERSION):
        self.pipeline=UnifiedAlgorithmPipeline(registry);self.factory=factory;self.external=external;self._decision=decision
        self.evaluation=evaluation;self.folder=Path(folder);self.task_id=task_id
        self.node_hits=self.node_new=self.kernel_hits=0;self.evaluation_protocol=evaluation.protocol;self.setup_seconds=0.
    def decision(self):return self._decision
    def execute(self,graph,key):
        started=perf_counter();trace=[];attempted=False;costs={}
        try:
            self._decision.require_graph(self.factory,graph)
            attempted=True
            result=self.pipeline.run_dag(graph,task_id=self.task_id,external=self.external,version=VERSION,decision=self._decision,factory=self.factory)
            costs["execution_and_quality_seconds"]=perf_counter()-started
            trace=result["trace"];artifact=result["outputs"]["field"];ds=artifact.value
            begin=perf_counter();metrics=self.evaluation(ds.copy(deep=True));costs["independent_evaluation_seconds"]=perf_counter()-begin
            if not np.isfinite(metrics["objective"]):raise ValueError("Nonfinite independent objective")
            begin=perf_counter();folder=self.folder/key;folder.mkdir(parents=True,exist_ok=True)
            write=ds.copy(deep=True)
            for target in [write,*write.variables.values()]:
                target.attrs={k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in target.attrs.items() if v is not None}
            write.to_netcdf(folder/"product.nc",engine="h5netcdf")
            with xr.open_dataset(folder/"product.nc",engine="h5netcdf") as check:xr.testing.assert_allclose(check.load(),write)
            costs["output_readback_seconds"]=perf_counter()-begin
            # Behavior identity excludes algorithm/report attributes. Alias
            # registrations producing the same numbers are not new behaviors.
            numeric=ds.copy(deep=False);numeric.attrs={}
            for name in numeric.variables:numeric[name].attrs={}
            info=dict(status="PASS",**metrics,quality=artifact.metadata,task_acceptance=result["task_acceptance"],
                prediction_fingerprint=Artifact("numeric_product",numeric).fingerprint(),product=str(folder/"product.nc"),file_readback="PASS")
        except Exception as exc:
            trace=getattr(exc,"trace",trace)
            info=dict(status="FAIL",objective=None,error=str(exc),call_chain=traceback.format_exc(),
                task_acceptance=getattr(exc,"details",{}).get("task_acceptance"),failure_stage="execution_or_quality" if attempted else "task_requirements")
        self.node_hits+=sum(bool(n.get("cache_hit")) for n in trace);self.node_new+=sum(n.get("status")=="PASS" and not n.get("cache_hit") for n in trace)
        info.update(key=key,graph=record(graph),seconds=perf_counter()-started,node_trace=trace,numerical_execution_attempted=attempted,hidden_truth_passed_to_nodes=False,costs=costs)
        save(self.folder/key/"result.json",info)
        return info


def submit_search(inputs,request,evaluation,folder,*,budget=None,mode="PortraitMutation",seed=11,progress=None,validation=None):
    """Single public service used by pages, toolkit and research clients."""
    started=perf_counter();folder=Path(folder)
    if any((folder/name).exists() for name in ("request.json","summary.json")):raise ValueError("Immutable request directory already exists")
    try:
        if mode=='PortraitMutation':
            request=dict(request,soft_guidance=request.get('soft_guidance',False))
            if type(request['soft_guidance']) is not bool:raise ValueError('soft_guidance must be boolean')
            request.setdefault('spatial_soft_guidance',request['soft_guidance'])
            parsed,audit=parse_inputs(inputs)
            from ecodata.algorithms.spatial_fusion_task import applies,submit as spatial_submit
            if applies(parsed,request):
                if evaluation is not None or validation is not None:
                    raise ValueError('Spatial current-input selection uses only internally held visible locations')
                if {'fixed_graph','composition'}&set(request):
                    raise ValueError('Automatic spatial selection cannot receive a prescribed route')
                replay=folder/'replay_inputs';replay.mkdir(parents=True,exist_ok=False)
                index=[]
                for number,item in enumerate(inputs):
                    name=f'{number}.bin';(replay/name).write_bytes(item.content)
                    index.append(dict(path=name,filename=item.filename,role=item.role,metadata=item.metadata,
                        sha256=hashlib.sha256(item.content).hexdigest()))
                save(replay/'index.json',index)
                return spatial_submit(parsed,request,audit,folder,budget=dict(CURRENT_INPUT_BUDGET,**(budget or {})),seed=seed,progress=progress)
        if request.get('decision_policy')=='current_input':
            if mode!='PortraitMutation' or validation is not None or request.get('mutation_control','random')!='random':
                raise ValueError('Current-input decisions require fresh random search with visible-only internal validation')
            if {'fixed_graph','composition'}&set(request):raise ValueError('Current-input decisions cannot reuse a prescribed graph or workflow pool')
        if mode=="PortraitMutation":
            from ecodata.algorithms.field_mutation import submit
            return submit(inputs,request,evaluation,folder,budget=dict(DEFAULT_BUDGET,**(budget or {})),seed=seed,progress=progress,validation=validation)
        if mode=="AutonomousField":
            from ecodata.algorithms.autonomous_field import submit
            return submit(inputs,request,evaluation,folder,budget=dict(DEFAULT_BUDGET,**(budget or {})),seed=seed,progress=progress,validation=validation)
        if mode=="Autonomous":
            if validation is not None:
                raise ValueError('Autonomous assessment builds its own isolated visible cohorts; do not pass a second selection target')
            from ecodata.algorithms.autonomous_decision import submit
            return submit(inputs,request,evaluation,folder,budget=dict(DEFAULT_BUDGET,**(budget or {})),seed=seed,progress=progress)
        if "composition" in request:
            if mode != "Profile":
                raise ValueError("Source/time composition uses the three-portrait Profile entry")
            from ecodata.algorithms.composition_decision import submit
            return submit(inputs,request,evaluation,folder,budget=dict(DEFAULT_BUDGET,**(budget or {})),seed=seed,progress=progress,validation=validation)
        from ecodata.algorithms.numerical_search import RETIRED_MESSAGE
        raise ValueError(RETIRED_MESSAGE)
    except Exception as exc:
        save(folder/"failure.json",dict(status="BLOCKED",error=str(exc),call_chain=traceback.format_exc(),request=request))
        raise


def submit_file_request(config,folder):
    """Toolkit JSON client. Scientific policy remains exclusively in submit_search."""
    inputs=[SearchInput(Path(item["path"]).name,item["role"],Path(item["path"]).read_bytes(),item.get("metadata")) for item in config["inputs"]]
    if config.get('decision_policy')=='current_input' or config['request'].get('decision_policy')=='current_input':
        if config.get('search_validation') or config.get('evaluation'):raise ValueError('Current-input toolkit requests do not consume external selection/test references')
        return process_current_input(inputs,config['request'],folder,budget=config.get('budget'),seed=config.get('seed',11))
    spec=config.get("evaluation",{});variable=config["request"]["variable"]
    if spec.get("kind")=="reference_error":
        path=Path(spec["path"]);data,_=recognise_file(path.name,path.read_bytes())
        evaluation=DatasetReference(data[variable],variable,role=spec.get("role","validation"))
    elif spec.get("kind")=="declared_task_metric":evaluation=EngineeringMetric(variable,spec.get("metric","coverage"))
    else:evaluation=None
    validation=None
    if config.get('search_validation'):
        path=Path(config['search_validation']['path']);data,_=recognise_file(path.name,path.read_bytes())
        validation=SearchValidation(data[variable],variable)
    return submit_search(inputs,config["request"],evaluation,folder,budget=config.get("budget"),mode=config.get("mode","C-local"),seed=config.get("seed",11),validation=validation)


def process_current_input(inputs,request,folder,*,budget=None,seed=11,progress=None):
    result=_process_current_input(inputs,request,folder,budget=budget,seed=seed,progress=progress)
    from .stability import attach
    if progress:progress('方案已冻结；执行有界无真值稳定性检查（不参与选型）')
    return attach(inputs,request,result,folder,execute_frozen_graph)


def _process_current_input(inputs,request,folder,*,budget=None,seed=11,progress=None):
    """Normal reconstruction entry: new input, new decision, no old graph."""
    if {'fixed_graph','composition'}&set(request):raise ValueError('Declare the target, not a prescribed graph, for current-input search')
    if request.get('semantic_type')!='continuous':raise ValueError('Current-input reconstruction requires a continuous variable and a defined numerical error metric')
    if request.get('mutation_control','random')!='random':raise ValueError('Current-input reconstruction uses the random mutation backend')
    parsed,_=parse_inputs(inputs)
    from ecodata.algorithms.spatial_fusion_task import applies
    if applies(parsed,request):
        return submit_search(inputs,dict(request,decision_policy='current_input'),None,folder,
            budget=dict(CURRENT_INPUT_BUDGET,**(budget or {})),mode='PortraitMutation',seed=seed,progress=progress)
    primary=[ds for role,ds in parsed if role=='primary']
    if len(primary)!=1:raise ValueError('One primary target source required; other sources retain their declared roles')
    ds=primary[0]
    request=dict(request,decision_policy='current_input',mutation_control='random')
    for key,coord in [('target_lat','lat'),('target_lon','lon'),('target_times','time')]:
        if request.get(key) is None:
            if coord not in ds:raise ValueError('Explicit reconstruction target required: '+key)
            request[key]=[str(v) for v in ds[coord].values] if coord=='time' else ds[coord].values.tolist()
    request['parameters']=dict(request.get('parameters',{}),raster_target_times=request['target_times'])
    return submit_search(inputs,request,None,folder,budget=dict(CURRENT_INPUT_BUDGET,**(budget or {})),mode='PortraitMutation',seed=seed,progress=progress)


def evaluate_graph_bank(inputs,request,graphs,folder,*,seed=11,budget=None):
    """Diagnostic only: same folds for every graph, no external selection labels.

    Products for every graph are executed inside this public call.  A bank is
    never substituted for the normal open mutation search.
    """
    if {'evaluation','search_validation','fixed_graph','composition'}&set(request):
        raise ValueError('Diagnostic selection uses visible-only folds, never external labels or prescribed winner')
    parsed,audit=parse_inputs(inputs)
    from ecodata.algorithms.spatial_fusion_task import applies,evaluate_bank
    if applies(parsed,request):
        return evaluate_bank(parsed,request,audit,graphs,folder,seed=seed,budget=dict(CURRENT_INPUT_BUDGET,**(budget or {})))
    from ecodata.algorithms.field_mutation import submit
    return submit(inputs,request,None,folder,budget=dict(CURRENT_INPUT_BUDGET,**(budget or {})),seed=seed,graphs=graphs)


def publish_development_validation(envelope):
    """Register explicitly development-only historical resources with hashes."""
    from ecodata.algorithms.attribute_registry import digest
    from ecodata.algorithms.development_validation import DEFAULT_REGISTRY,_read_resource
    if envelope.get('schema_version')!=1 or digest(envelope.get('registry'))!=envelope.get('sha256'):
        raise ValueError('Invalid development registry fingerprint')
    registry=envelope['registry']
    if registry.get('evidence_role')!='development' or not registry.get('evidence'):
        raise ValueError('Only evidence-bearing development resources can be registered')
    for case in registry['cases']:
        for item in case['sources']:_read_resource(item,DEFAULT_REGISTRY.parent)
    DEFAULT_REGISTRY.parent.mkdir(parents=True,exist_ok=True)
    if DEFAULT_REGISTRY.exists() and json.loads(DEFAULT_REGISTRY.read_text(encoding='utf-8'))!=envelope:
        raise ValueError('Development registry already published; explicit version migration required')
    save(DEFAULT_REGISTRY,envelope)
    return dict(status='PASS',path=str(DEFAULT_REGISTRY),sha256=envelope['sha256'])


def calibrate_reliable_selection(development_runs,calibration_id):
    from ecodata.algorithms.reliable_calibration import calibrate
    return calibrate(development_runs,calibration_id)


def execute_evaluated_candidate(inputs,search_folder,key,folder):
    """User choice among measured graphs, on exactly the same visible inputs."""
    import math
    search_folder=Path(search_folder)
    def load(name):return json.loads((search_folder/name).read_text(encoding='utf8'))
    protocol=load('request.json');summary=load('summary.json')
    row=next((r for r in load('candidate_results.json') if r['key']==key and r['status']=='PASS'),None)
    if row is None or not row.get('fixed_support_sha256') or row['fixed_support_sha256']!=summary['best'].get('fixed_support_sha256'):
        raise ValueError('Candidate has no passed evaluation on the same fixed support')
    if not row['metrics']['count'] or not all(math.isfinite(row['metrics'][m]) for m in ('RMSE','MAE')):
        raise ValueError('Candidate has no finite numerical evaluation')
    actual=[dict(source_filename=i.filename,role=i.role,sha256=hashlib.sha256(i.content).hexdigest(),declarations=i.metadata or {}) for i in inputs]
    expected=[{k:r[k] for k in ('source_filename','role','sha256','declarations')} for r in protocol['inputs']]
    if actual!=expected:raise ValueError('User choice cannot reuse validation scores on changed inputs')
    # This is same-input deployment, not current-input search or transfer to new data.
    request=dict(protocol['request']);request.pop('decision_policy',None)
    return execute_frozen_graph(inputs,request,dict(key=key,records=load('graphs.json')),folder)


def execute_frozen_graph(inputs,request,graph,folder):
    """Deploy a previously selected graph on new visible inputs, without labels.

    This is the same public parser, registered adapters and DAG executor used
    during mutation. It cannot reselect a graph using deployment truth.
    """
    if request.get('decision_policy')=='current_input':
        raise ValueError('Current-input mode must use process_current_input; frozen transfer does not reselect')
    parsed,audit=parse_inputs(inputs)
    from ecodata.algorithms.spatial_fusion_task import applies,execute_graph,COMPONENTS
    if applies(parsed,request) and all(row['operation'] in COMPONENTS for row in graph['records'].values()):
        return execute_graph(parsed,request,audit,graph,folder)
    from ecodata.algorithms.profile_matching import prepare_model_request
    from ecodata.algorithms.autonomous_field import build_context,field_registry
    from ecodata.algorithms.autonomous_decision import Executor,clean
    from ecodata.algorithms.search_accounting import SearchAccount
    request,model=prepare_model_request(request)
    context,audit,quality=build_context(inputs,request)
    from ecodata.algorithms.profile_matching import validate_model_context
    validate_model_context(context,model)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'deployment.json').exists():raise ValueError('Immutable deployment already exists')
    for key,row in graph['records'].items():
        if row['operation']=='source':
            index=int(row['parameters']['source'].removeprefix('source'))
            if key!='s'+str(index) or index>=len(context.sources):raise ValueError('Frozen source binding mismatch')
    account=SearchAccount(10000,3600)
    executor=Executor(context,field_registry(platform_registry(),context),account=account,phase='deploy')
    trace=[]
    try:
        output,trace=executor.execute(graph['records'],graph['key'])
        if not request.get('target_times'):raise ValueError('Explicit deployment target times required')
        from ecodata.algorithms.current_input import validate_output
        output,checks=validate_output(output,context,request)
        path=folder/'product.nc';clean(output).to_netcdf(path,engine='h5netcdf')
        with xr.open_dataset(path) as f:xr.testing.assert_allclose(clean(output),f.load())
    except Exception as exc:
        save(folder/'deployment.json',dict(status='FAILED',product=None,error=str(exc),graph=graph,
            inputs=audit,input_quality=quality,trace=getattr(exc,'trace',trace),cost=account.receipt(),
            reference_used=False,selection_performed=False))
        raise
    finally:save(folder/'total_cost.json',account.receipt())
    receipt=dict(product=str(path),graph=graph,inputs=audit,input_quality=quality,output_quality=checks,
        trace=trace,cost=account.receipt(),reference_used=False,selection_performed=False)
    save(folder/'deployment.json',receipt)
    return receipt


def publish_profile_policy(model_id,published_id,*,learning_order,fallback_policy,validation):
    """Retain immutable legacy diagnostic defaults; joint scoring stays primary."""
    from ecodata.algorithms.profile_matching import PlanRankingService
    return PlanRankingService.publish_policy(model_id,published_id,learning_order=learning_order,
        fallback_policy=fallback_policy,validation=validation)


def train_profile_matching(run_folders, model_id, *, holdout_groups):
    """Publish an immutable diagnostic model and three-portrait observations.

    Explicit groups and frozen holdout are required; do not train from final-test
    products or silently promote context observations into physical guarantees.
    """
    from ecodata.algorithms.profile_matching import PlanRankingService
    return PlanRankingService.train(run_folders, model_id, holdout_groups=holdout_groups)


def publish_attribute_registry(version,attributes,*,parent_version="builtin-v1",evidence):
    """Publish an immutable public scoring vocabulary; preserve hard consumers."""
    from ecodata.algorithms.attribute_registry import publish
    return publish(version,attributes,parent_version=parent_version,evidence=evidence)


def calibrate_profile_scoring(run_folders,calibration_id,*,holdout_groups,parameter_grid,registry_version=None):
    """Fit parameters of the joint rules from real development evidence only."""
    from ecodata.algorithms.scoring_calibration import calibrate
    return calibrate(run_folders,calibration_id,holdout_groups=holdout_groups,
        parameter_grid=parameter_grid,registry_version=registry_version)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Unified EcoData numerical GOLEM: immutable JSON request -> product and decision trace")
    parser.add_argument("config");parser.add_argument("output")
    args=parser.parse_args()
    # Use the canonical module so typed validation resources are not duplicated
    # between __main__ and ecodata.algorithms.platform_search under python -m.
    from ecodata.algorithms.platform_search import submit_file_request as canonical_submit
    result=canonical_submit(json.loads(Path(args.config).read_text(encoding="utf-8")),args.output)
    print(json.dumps(dict(selected=bool(result.get("best")),counts=result["counts"],stop_reason=result["stop_reason"]),ensure_ascii=False))


def publish_model_portrait(version,model_id,scenarios,*,evidence,parent_version=None):
    from ecodata.models.portrait_policy import publish
    return publish(version,model_id,scenarios,evidence=evidence,parent_version=parent_version)


def inspect_model_portrait_sources(paths):
    from ecodata.models.portrait_policy import inspect_model_sources
    return inspect_model_sources(paths)


def model_portrait_template(model_id):
    from ecodata.models.portrait_policy import template
    return template(model_id)


def learn_portrait_memory(folders,version,*,holdout_groups,features,min_groups=2,max_distance=1.):
    from ecodata.algorithms.portrait_memory import fit
    return fit(folders,version,holdout_groups=holdout_groups,features=features,min_groups=min_groups,max_distance=max_distance)
