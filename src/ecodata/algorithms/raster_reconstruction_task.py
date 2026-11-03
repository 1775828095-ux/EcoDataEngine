"""Small task adapters over existing Registry kernels and official LIIF.

No AOD mass conservation is asserted. Optional projection enforces only the
declared coarse observation operator, using the existing CVXPY dependency.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import sys
import numpy as np
import xarray as xr
from scipy import sparse
from ecodata.algorithms.contracts import AlgorithmSpec, AlgorithmResult, Compatibility
from ecodata.algorithms.router import AlgorithmPlan, AlgorithmSelector
from ecodata.core.schemas import EcoDataBundle


def edges(centers):
    a=np.asarray(centers,float)
    if len(a)<2 or np.any(np.diff(a)<=0):raise ValueError("Increasing grid centers required")
    return np.r_[a[0]-(a[1]-a[0])/2,(a[1:]+a[:-1])/2,a[-1]+(a[-1]-a[-2])/2]


def overlap_weights(fine_edges,coarse_edges,latitude=False):
    fine=np.sin(np.deg2rad(fine_edges)) if latitude else np.asarray(fine_edges,float)
    coarse=np.sin(np.deg2rad(coarse_edges)) if latitude else np.asarray(coarse_edges,float)
    overlap=np.maximum(0,np.minimum(coarse[1:,None],fine[None,1:])-np.maximum(coarse[:-1,None],fine[None,:-1]))
    weights=overlap/np.diff(coarse)[:,None]
    np.testing.assert_allclose(weights.sum(1),1,atol=1e-10)
    return sparse.csr_matrix(weights)


def degradation(reference,coarse_size,variable="aod"):
    ye,xe=edges(reference.lat),edges(reference.lon)
    cy=np.linspace(ye[0],ye[-1],coarse_size+1);cx=np.linspace(xe[0],xe[-1],coarse_size+1)
    ay=overlap_weights(ye,cy,True);ax=overlap_weights(xe,cx)
    operator=sparse.kron(ay,ax,format="csr")
    a=reference[variable].transpose("lat","lon").values
    if not np.isfinite(a).all():raise ValueError("This small experiment requires a complete reference tile; no target filling")
    coarse=xr.Dataset({variable:(("lat","lon"),(operator@a.ravel()).reshape(coarse_size,coarse_size))},
        coords={"lat":(cy[:-1]+cy[1:])/2,"lon":(cx[:-1]+cx[1:])/2},attrs=dict(reference.attrs))
    coarse[variable].attrs=dict(reference[variable].attrs)
    return coarse,operator


def compatible(context):
    reasons=[]
    if context.semantic_type!="continuous":reasons.append("This task requires a continuous variable")
    if not context.sources or context.variable not in context.sources[0].dataset:reasons.append("Missing declared raster variable")
    else:
        ds=context.sources[0].dataset
        field=ds[context.variable]
        expected=context.model_constraints.get("unit","1")
        if ds.attrs.get("crs")!="EPSG:4326" or field.attrs.get("units")!=expected:reasons.append("Explicit WGS84 and matching declared units required")
        if field.dims!=("lat","lon") or not np.isfinite(field.values).all():reasons.append("Finite 2D lat/lon field required")
    if context.target_lat is None or context.target_lon is None:reasons.append("Target coordinates required")
    return reasons


def result(context,identifier,values,report):
    ds=xr.Dataset({context.variable:(("lat","lon"),values)},coords={"lat":list(context.target_lat),"lon":list(context.target_lon)},attrs=dict(context.sources[0].dataset.attrs))
    ds[context.variable].attrs=dict(context.sources[0].dataset[context.variable].attrs)
    return AlgorithmResult(EcoDataBundle(ds),identifier,report)


class RasterBaselineAdapter:
    def __init__(self,upstream):
        self.upstream=upstream
        from ecodata.algorithms.capabilities import describe_algorithm
        self.spec=replace(upstream.spec,id="raster_task."+upstream.spec.id,category="raster_reconstruction",
            capabilities=describe_algorithm(upstream.spec).capabilities,
            notes="Existing backend, explicit nearest boundary extension; no new numerical interpolation kernel")
    def compatibility(self,context):
        reasons=compatible(context)
        if not reasons:
            check=self.upstream.compatibility(context);reasons.extend(check.reasons if not check.eligible else ())
        return Compatibility(not reasons,tuple(reasons))
    def run(self,context):
        a=self.upstream.run(context).data.dataset[context.variable]
        values=a.values.copy();missing=~np.isfinite(values)
        source=context.sources[0].dataset[context.variable]
        if missing.any():
            fallback=source.interp(lat=list(context.target_lat),lon=list(context.target_lon),method="nearest",kwargs={"fill_value":"extrapolate"})
            values[missing]=fallback.values[missing]
        return result(context,self.spec.id,values,dict(upstream=self.upstream.spec.id,
            boundary_rule="nearest edge extension; primary reference metrics use common interior support",boundary_extended_pixels=int(missing.sum())))


class NonnegativeRaster:
    spec=AlgorithmSpec("raster_task.nonnegative","Nonnegative numeric projection","1","raster_constraint","available",
        ("rectilinear_grid",),("continuous",),physical_constraints=("nonnegative",),capabilities=("nonnegative_validation",),
        input_contract=dict(prediction_stage="after_reconstruction",requires_parameters=[]),backend="NumPy",license="BSD-3-Clause")
    def compatibility(self,context):return Compatibility(not compatible(context),tuple(compatible(context)))
    def run(self,context):
        a=context.sources[0].dataset[context.variable].values
        return result(context,self.spec.id,np.maximum(a,0),dict(negative_projection_count=int((a<0).sum())))


class GridCentreStationAdapter:
    """Explicit centre-value approximation over existing station kernels."""
    def __init__(self,upstream):
        self.upstream=upstream
        self.spec=replace(upstream.spec,id="raster_task."+upstream.spec.id,category="raster_reconstruction",
            input_kinds=("rectilinear_grid",),requires_observations=False,
            capabilities=("continuous_spatial_resampling",),
            notes="Cell averages treated as centre-value proxies, not measured point observations; nearest boundary extension")
    def station_context(self,context):
        import pandas as pd
        source=context.sources[0].dataset
        lon,lat=np.meshgrid(source.lon.values,source.lat.values)
        observations=pd.DataFrame(dict(longitude=lon.ravel(),latitude=lat.ravel(),
            value=source[context.variable].values.ravel(),split="train",unit=source[context.variable].attrs["units"]))
        return replace(context,observations=observations,
            parameters=dict(context.parameters,observation_crs="EPSG:4326"))
    def compatibility(self,context):
        reasons=compatible(context)
        if not reasons:
            source=context.sources[0].dataset
            if np.array_equal(source.lat,context.target_lat) and np.array_equal(source.lon,context.target_lon):
                reasons.append('Finite raster already on target grid; exact centre-value resampling has no new information')
        if not reasons:
            check=self.upstream.compatibility(self.station_context(context))
            if not check.eligible:reasons.extend(check.reasons)
        return Compatibility(not reasons,tuple(reasons))
    def run(self,context):
        prediction=self.upstream.run(self.station_context(context))
        values=prediction.data.dataset[context.variable].values.copy()
        missing=~np.isfinite(values)
        if missing.any():
            source=context.sources[0].dataset[context.variable]
            nearest=source.interp(lat=list(context.target_lat),lon=list(context.target_lon),
                method="nearest",kwargs={"fill_value":"extrapolate"})
            values[missing]=nearest.values[missing]
        return result(context,self.spec.id,values,dict(upstream=self.upstream.spec.id,
            support_approximation="coarse cell averages as centre-value proxies; not true point observations",
            boundary_rule="nearest extension for uncovered queries",boundary_extended_pixels=int(missing.sum())))


class CoarseObservationProjection:
    spec=AlgorithmSpec("raster_task.coarse_consistency","Coarse observation consistency QP","1","raster_constraint","available",
        ("rectilinear_grid",),("continuous",),compute="medium",physical_constraints=("nonnegative","declared_observation_consistency"),
        capabilities=("nonnegative_validation","coarse_scale_consistency"),input_contract=dict(prediction_stage="after_reconstruction",requires_parameters=["degradation_operator","coarse_reference"]),backend="CVXPY/OSQP",license="Apache-2.0",notes="Not a physical AOD total-mass conservation law")
    def compatibility(self,context):
        reasons=compatible(context)
        operator=context.parameters.get("degradation_operator");coarse=context.parameters.get("coarse_reference")
        if operator is None or coarse is None:reasons.append("Missing original coarse observations/operator")
        elif operator.shape!=(coarse[context.variable].size,context.sources[0].dataset[context.variable].size):reasons.append("Observation operator shape mismatch")
        return Compatibility(not reasons,tuple(reasons))
    def run(self,context):
        import cvxpy as cp
        if not self.compatibility(context).eligible:raise ValueError("Incompatible coarse projection")
        a=context.sources[0].dataset[context.variable].values;operator=context.parameters["degradation_operator"]
        coarse=context.parameters["coarse_reference"][context.variable].values.ravel();z=cp.Variable(a.size)
        problem=cp.Problem(cp.Minimize(cp.sum_squares(z-a.ravel())),[z>=0,operator@z==coarse])
        problem.solve(solver="OSQP",eps_abs=1e-7,eps_rel=1e-7,max_iter=10000,polishing=True)
        if problem.status not in {"optimal","optimal_inaccurate"} or z.value is None:raise ValueError("Observation-consistency QP failed: "+problem.status)
        values=np.maximum(z.value,0);error=float(np.max(np.abs(operator@values-coarse)))
        if error>1e-5:raise ValueError("Coarse observation consistency tolerance exceeded")
        return result(context,self.spec.id,values.reshape(a.shape),dict(solver="OSQP",solver_status=problem.status,
            coarse_max_abs_error=error,physical_total_conservation_asserted=False))


class LocalLIIFRaster:
    def __init__(self,checkpoint):
        self.checkpoint=Path(checkpoint);self._runtime=None
        self.spec=AlgorithmSpec("raster_task.liif_aod","Official LIIF, local AOD training","1","raster_reconstruction",
            "available" if self.checkpoint.exists() else "checkpoint_missing",("rectilinear_grid",),("continuous",),
            scale_range=(1.5,3.),requires_weights=True,compute="medium",license="BSD-3-Clause",
            source_url="https://github.com/yinboc/liif",backend="Official LIIF CPU, existing liif_runtime device adapter",
            capabilities=("continuous_spatial_resampling","learning_based_spatial_reconstruction"),
            notes="Locally trained on LGHAP AOD only, not official pretrained environmental weights")
    def compatibility(self,context):
        reasons=compatible(context)
        if context.variable!="aod" or (context.sources and context.sources[0].dataset[context.variable].attrs.get("units")!="1"):
            reasons.append("This local LIIF checkpoint supports dimensionless AOD only")
        if not self.checkpoint.exists():reasons.append("Missing AOD-trained checkpoint; temperature/RGB weights forbidden")
        if context.target_lat and context.sources:
            factor=len(context.target_lat)/context.sources[0].dataset.sizes["lat"]
            if not 1.5<=factor<=3:reasons.append("Outside declared trained scale range")
        return Compatibility(not reasons,tuple(reasons))
    def runtime(self):
        if self._runtime is None:
            import torch
            sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"tools"))
            from liif_runtime import load
            models,utils=load();state=torch.load(self.checkpoint,map_location="cpu",weights_only=True)
            if state["variable"]!="aod" or state["unit"]!="1":raise ValueError("Incompatible model variable/units")
            model=models.make(state["spec"]).cpu();model.load_state_dict(state["state_dict"]);model.eval()
            self._runtime=(model,utils,state)
        return self._runtime
    def run(self,context):
        import torch
        model,utils,state=self.runtime();h,w=len(context.target_lat),len(context.target_lon)
        coord=utils.make_coord((h,w)).unsqueeze(0);cell=torch.ones_like(coord);cell[:,:,0]*=2/h;cell[:,:,1]*=2/w
        a=context.sources[0].dataset.aod.values
        tensor=torch.tensor((a-state["mean"])/state["std"],dtype=torch.float32)[None,None].repeat(1,3,1,1)
        with torch.no_grad():values=model(tensor,coord,cell).reshape(h,w).numpy()*state["std"]+state["mean"]
        return result(context,self.spec.id,values,dict(official_model=True,weight_origin="LOCAL_AOD_TRAINING",checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()))


def reconstruction_plans(registry,context):
    """Use existing selector to screen; compose only compatible regrid -> constraints."""
    selector=AlgorithmSelector(registry);screen=selector.screen(context);plans=[]
    for spec in registry.specs():
        if spec.category!="raster_reconstruction" or not screen[spec.id].eligible:continue
        for tail in ["raster_task.nonnegative","raster_task.coarse_consistency"]:
            # Constraint applicability is evaluated on the actual predicted grid, not on the coarse input.
            template=xr.Dataset({context.variable:(("lat","lon"),np.zeros((len(context.target_lat),len(context.target_lon))))},
                coords={"lat":list(context.target_lat),"lon":list(context.target_lon)},attrs=dict(context.sources[0].dataset.attrs))
            template[context.variable].attrs=dict(context.sources[0].dataset[context.variable].attrs)
            output_context=replace(context,sources=(EcoDataBundle(template),))
            if registry.get(tail).compatibility(output_context).eligible:
                plans.append(AlgorithmPlan(spec.id+"+"+tail,(spec.id,tail),0.,
                    ("Task: continuous dimensionless raster; declared scale and source contract compatible",
                     "Optional operator consistency, not asserted AOD mass conservation")))
    return tuple(plans),screen


def select_raster_validation(records,expected_tiles):
    """Task-level multi-example selector; no labels from test or unequal cohorts."""
    import pandas as pd
    table=pd.DataFrame(records)
    if table.empty or not table.stage.eq("validation").all():raise ValueError("Selection accepts validation records only")
    if not table.coverage.eq(1).all() or table.failure.any():raise ValueError("Incomplete validation predictions cannot rank")
    choices={}
    for n,part in table.groupby("coarse_size"):
        if not part.groupby("plan").tile.nunique().eq(expected_tiles).all():raise ValueError("Unequal validation tile coverage")
        if not part.groupby("tile").evaluated_pixels.nunique().eq(1).all():raise ValueError("Unequal reference mask")
        losses=part.assign(MSE=part.RMSE**2).groupby("plan").agg(MSE=("MSE","mean"),seconds=("seconds","mean"))
        losses=losses.reset_index().sort_values(["MSE","seconds","plan"]).set_index("plan")
        singles=losses.loc[~losses.index.str.endswith("consistency")];combos=losses.loc[losses.index.str.endswith("consistency")]
        choices[str(n)]=dict(automatic=losses.index[0],single=singles.index[0],composite=combos.index[0])
    return choices
