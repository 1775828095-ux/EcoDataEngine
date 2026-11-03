"""User-facing uploaded-raster workflow. No reference field or task-class input."""
from dataclasses import asdict,dataclass
from hashlib import sha256
from time import perf_counter
import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator
from ecodata.algorithms.contracts import AlgorithmContext,AlgorithmSpec,Compatibility
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.raster_reconstruction_task import (edges,degradation,compatible,result,
    RasterBaselineAdapter,NonnegativeRaster,CoarseObservationProjection,reconstruction_plans)
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import RawInputQualityGate
from ecodata.recognition import recognise_file


@dataclass(frozen=True)
class RasterRequest:
    variable: str="field"
    target_size: int=100
    description: str="Reconstruct a nonnegative continuous field"
    boundary_priority: float=0.
    gradient_priority: float=.3
    peak_priority: float=0.
    consistency_priority: float=.3


class ShapePreservingRaster:
    """Separable official PCHIP: protects observed shape, NOT unknown fine peaks."""
    spec=AlgorithmSpec("raster_task.pchip","Local shape-preserving PCHIP","1","raster_reconstruction","available",
        ("rectilinear_grid",),("continuous",),backend="SciPy PchipInterpolator, latitude then longitude",
        license="BSD-3-Clause",source_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html",
        capabilities=("continuous_spatial_resampling","observed_monotone_shape_preservation"),
        notes="C1 piecewise cubic slopes from observed secants; no invented subcell peak amplitude/location; axis order fixed")
    def compatibility(self,context):return Compatibility(not compatible(context),tuple(compatible(context)))
    def run(self,context):
        source=context.sources[0].dataset;a=source[context.variable].values
        lat=np.clip(context.target_lat,source.lat.min().item(),source.lat.max().item())
        lon=np.clip(context.target_lon,source.lon.min().item(),source.lon.max().item())
        stage=PchipInterpolator(source.lat.values,a,axis=0,extrapolate=False)(lat)
        output=PchipInterpolator(source.lon.values,stage,axis=1,extrapolate=False)(lon)
        return result(context,self.spec.id,output,dict(backend="SciPy PCHIP",axis_order="lat then lon",
            observed_shape_only=True,unobserved_peaks_identifiable=False,boundary="constant nearest edge"))


def upload_registry():
    registry=create_default_registry()
    for identifier in ["spatial.nearest","spatial.bilinear","spatial.xarray_cubic"]:
        registry.register(RasterBaselineAdapter(registry.get(identifier)))
    for plugin in [ShapePreservingRaster(),NonnegativeRaster(),CoarseObservationProjection()]:registry.register(plugin)
    from ecodata.algorithms.raster_reconstruction_task import GridCentreStationAdapter
    for identifier in ["fusion.idw","station.metpy_linear","station.pykrige_ordinary"]:
        registry.register(GridCentreStationAdapter(registry.get(identifier)))
    return registry


FEATURE_NAMES=("mean","std","range","gradient_mean","gradient_p95","flat_fraction","edge_concentration",
    "curvature_mean","extrema_fraction","max_median_ratio","unique_fraction","scale")


def input_features(context):
    from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
    return SemanticFeatureExtractor.raster_features(context.sources[0].dataset,context.variable,len(context.target_lat))


def objective_weights(request):
    # Metric order: normalized RMSE, gradient, boundary-side RMSE, peak amplitude,
    # peak location, coarse observation error. No generator labels or algorithm IDs.
    return np.array([1.,request.gradient_priority,request.boundary_priority,
        request.peak_priority,.5*request.peak_priority,request.consistency_priority])


def route(features,request,model,plans,registry,*,single_only=False):
    vector=np.array([[features[n] for n in model["features"]]])
    estimated=model["regressor"].predict(model["scaler"].transform(vector)).reshape(len(model["plans"]),6)
    scores=estimated@objective_weights(request)
    available={p.id:p for p in plans};options=[]
    for identifier,score,components in zip(model["plans"],scores,estimated):
        if identifier not in available:continue
        plan=available[identifier];composite=plan.steps[-1].endswith("consistency")
        if single_only and composite:continue
        cost=sum({"low":1,"medium":3,"high":8}.get(registry.get(s).spec.compute,8) for s in plan.steps)
        options.append(dict(plan=identifier,predicted_score=float(score),predicted_components=components.tolist(),
            declared_cost_units=cost,composite=composite))
    if not options:raise ValueError("No compatible learned routing candidates")
    minimum=min(o["predicted_score"] for o in options)
    near=[o for o in options if o["predicted_score"]<=minimum+model["indifference_tolerance"]]
    chosen=min(near,key=lambda o:(o["declared_cost_units"],o["predicted_score"],o["plan"]))
    return chosen["plan"],dict(features=features,weights=objective_weights(request).tolist(),candidates=options,
        selected=chosen["plan"],reason="Development-only predicted task loss; within frozen tolerance choose lower declared cost",
        indifference_tolerance=model["indifference_tolerance"],hidden_truth_access=False,
        filename_or_task_class_used=False,normalization_fit_on="development only")


def process_raster_upload(upload,request,model=None,*,mode="automatic",execute_all=False):
    """Same UploadedFile handler is used by Streamlit and the benchmark client."""
    clock=perf_counter();content=upload.getvalue()
    parsed,info=recognise_file(upload.name,content)
    variable=request.variable
    if variable not in parsed:raise ValueError("Select the actual uploaded variable")
    if parsed[variable].dims!=("lat","lon") or parsed.sizes["lat"]!=parsed.sizes["lon"]:
        raise ValueError("Current raster entry requires a square 2D lat/lon continuous raster")
    if not 4<=parsed.sizes["lat"]<=200 or not 4<=request.target_size<=200:raise ValueError("Raster size outside bounded CPU task")
    # Whitelist physical metadata. Do not send filename, arbitrary attrs or descriptor labels to routing.
    source=parsed[[variable]].copy();source.attrs={"crs":info.crs or parsed.attrs.get("crs")}
    source[variable].attrs={k:v for k,v in parsed[variable].attrs.items() if k in {"units","data_kind","spatial_semantics","physical_variable"}}
    unit=source[variable].attrs.get("units")
    if not unit:raise ValueError("Explicit units required, no guessing")
    quality=RawInputQualityGate().validate(source,variable=variable,intended_operation="reconstruction",expected_unit=unit,requires_spatial=True)
    if quality["status"]=="BLOCK":raise ValueError("Raw input quality gate BLOCK: "+str(quality))
    ye,xe=edges(source.lat),edges(source.lon);n=request.target_size
    target=xr.Dataset(coords={"lat":ye[0]+(np.arange(n)+.5)*(ye[-1]-ye[0])/n,
        "lon":xe[0]+(np.arange(n)+.5)*(xe[-1]-xe[0])/n},attrs=dict(source.attrs))
    target[variable]=(("lat","lon"),np.zeros((n,n)));target[variable].attrs=dict(source[variable].attrs)
    reconstructed_coarse,operator=degradation(target,source.sizes["lat"],variable)
    for axis in ["lat","lon"]:np.testing.assert_allclose(reconstructed_coarse[axis],source[axis],atol=1e-10)
    context=AlgorithmContext((EcoDataBundle(source),),variable,"continuous",tuple(target.lat.values),tuple(target.lon.values),
        model_constraints={"unit":unit,"output_unit":unit,"nonnegative":True},input_quality=quality,
        parameters={"coarse_reference":source,"degradation_operator":operator})
    registry=upload_registry();plans,screen=reconstruction_plans(registry,context)
    shared_decision=AlgorithmSelector(registry).decide(context,dict(operation="spatial_reconstruction",
        spatial_resolution_km=float(abs(float(target.lat[1]-target.lat[0]))*111.)))
    # Constraint tails consume the predicted grid, not the uploaded coarse grid;
    # their compatibility remains checked against that real intermediate.
    plans=tuple(p for p in plans if shared_decision.matches[p.steps[0]]["eligible"])
    features=input_features(context);decision=None;selected=None
    if mode in {"automatic","automatic_single"}:
        if model is None:raise ValueError("Frozen development router required; no validation truth fallback")
        selected,decision=route(features,request,model,plans,registry,single_only=mode=="automatic_single")
    elif mode!="benchmark":raise ValueError("Unknown public raster mode")
    pipeline=UnifiedAlgorithmPipeline(registry);outputs={};reports={};failures=[]
    for plan in plans:
        if not execute_all and plan.id!=selected:continue
        try:
            result_value=pipeline.run(context,plan)
            outputs[plan.id]=result_value.data.dataset
            reports[plan.id]=result_value.report
        except Exception as exc:failures.append(dict(plan=plan.id,error=str(exc)))
    if selected and selected not in outputs:raise ValueError("Selected pipeline failed: "+str(failures))
    trace=dict(entry="Streamlit single-source raster upload",input_sha256=sha256(content).hexdigest(),
        filename_for_audit_only=upload.name,input_descriptor=info.as_dict(),task_requirements=asdict(request),
        available_metadata=dict(variable=variable,units=unit,crs=source.attrs["crs"],source_shape=list(source[variable].shape)),
        input_features=features,quality_input=quality,screening={s:dict(eligible=c.eligible,reasons=c.reasons) for s,c in screen.items()},
        plans=[dict(id=p.id,steps=p.steps) for p in plans],decision=decision,shared_decision=shared_decision.as_dict(),mode=mode,
        failures=failures,runtime_seconds=perf_counter()-clock,reference_field_received=False)
    return dict(outputs=outputs,reports=reports,trace=trace,features=features,selected=selected,operator=operator,
        coarse=source,request=request)
