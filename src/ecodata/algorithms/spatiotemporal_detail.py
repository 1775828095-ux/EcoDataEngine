"""Observed coarse dynamics plus sparse fine detail, with an explicit operator.

This is a linear residual/projection baseline, NOT STARFM/ESTARFM. The only
fine information is the supplied sparse series; held-out fine fields never
enter this object. No optical-reflectance assumptions are applied to temperature.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import cho_factor,cho_solve
from scipy import sparse
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility,AlgorithmResult
from ecodata.algorithms.raster_reconstruction_task import edges,overlap_weights
from ecodata.core.schemas import EcoDataBundle


@dataclass
class Reconstruction:
    fine:xr.Dataset
    coarse:xr.Dataset
    request:dict

    def __post_init__(self):
        self.variable=self.request["variable"]
        lat=np.asarray(self.request["target_lat"]);lon=np.asarray(self.request["target_lon"])
        if self.fine[self.variable].attrs.get("spatial_semantics")!="cell_mean" or self.coarse[self.variable].attrs.get("spatial_semantics")!="cell_mean":
            raise ValueError("Observed coarse operator requires declared cell-mean support")
        ay=overlap_weights(edges(lat),edges(self.coarse.lat),True)
        ax=overlap_weights(edges(lon),edges(self.coarse.lon))
        self.operator=sparse.kron(ay,ax,format="csr")
        gram=(self.operator@self.operator.T).toarray()
        self.factor=cho_factor(gram)
        times=pd.DatetimeIndex(self.request["target_times"])
        self.coarse_values=self.coarse[self.variable].sel(time=times).transpose("time","lat","lon")
        self.fine_proxy=self.fine[self.variable].interp(time=times,lat=lat,lon=lon).transpose("time","lat","lon")
        if not np.isfinite(self.fine_proxy).all() or not np.isfinite(self.coarse_values).all():
            raise ValueError("Visible fine anchors/coarse observations do not cover requested domain/time; no extrapolation")
        self.reference_detail=self.nullspace(self.fine_proxy.values)

    def lift(self,values):
        # Right inverse of the declared area-mean observation operator.
        array=np.asarray(values);shape=array.shape
        flat=array.reshape(-1,self.operator.shape[0])
        return np.asarray(self.operator.T@cho_solve(self.factor,flat.T)).T.reshape(*shape[:-2],len(self.request["target_lat"]),len(self.request["target_lon"]))

    def aggregate(self,values):
        a=np.asarray(values);flat=a.reshape(-1,self.operator.shape[1])
        return np.asarray(self.operator@flat.T).T.reshape(*a.shape[:-2],len(self.coarse.lat),len(self.coarse.lon))

    def nullspace(self,values):return np.asarray(values)-self.lift(self.aggregate(values))

    def project(self,values,coarse):return np.asarray(values)+self.lift(np.asarray(coarse)-self.aggregate(values))

    def metrics(self,dataset):
        a=dataset[self.variable].transpose("time","lat","lon").values
        return dict(coarse_error=float(np.sqrt(np.mean((self.aggregate(a)-self.coarse_values.values)**2))),
            detail_error=float(np.sqrt(np.mean((self.nullspace(a)-self.reference_detail)**2))))

    def baseline(self):
        # Before risks are measured from the visible complementary sources.
        coarse_error=float(np.sqrt(np.mean((self.aggregate(self.fine_proxy.values)-self.coarse_values.values)**2)))
        detail_error=float(np.sqrt(np.mean(self.reference_detail**2)))
        return dict(coarse_error=coarse_error,detail_error=detail_error)


class SpatiotemporalDetail:
    spec=AlgorithmSpec("fusion.spatiotemporal_detail","Sparse fine detail and observed coarse dynamics","1",
        "multi_source_fusion","available",("fine_low_frequency","coarse_high_frequency"),("continuous",),
        compute="low",capabilities=("multi_source_fusion","temporal_detail_transfer","declared_observation_consistency"),
        input_contract=dict(source_arity=2,requires_parameters=["reconstruction","fine_index","detail_strength"]),
        output_contract=dict(uncertainty_estimated=False),
        backend="SciPy linear solve and existing area observation/residual kernels; not STARFM")

    def parameter_proposals(self,context,rng):
        r=context.parameters.get('reconstruction')
        if not isinstance(r,Reconstruction) or len(context.sources)!=2:return []
        indices=[i for i,s in enumerate(context.sources) if s.dataset.attrs.get('source_id')==r.fine.attrs.get('source_id')]
        if len(indices)!=1:return []
        # Full observed detail is the unshrunk default; uniform exploration
        # samples the admissible continuous shrinkage domain, without truth.
        return [dict(fine_index=indices[0],detail_strength=float(a)) for a in (1.,rng.uniform(0,1))]

    def compatibility(self,context):
        p=context.parameters
        valid=(len(context.sources)==2 and isinstance(p.get("reconstruction"),Reconstruction) and
            p.get("fine_index") in {0,1} and isinstance(p.get("detail_strength"),(int,float)) and 0<=p["detail_strength"]<=1)
        if not valid:return Compatibility(False,('Explicit visible-source reconstruction contract and two aligned fields required',))
        try:
            fields=[s.dataset[context.variable] for s in context.sources]
            xr.align(*fields,join='exact')
            if any(not np.isfinite(a).all() for a in fields):raise ValueError('Finite parent fields required')
            if not np.array_equal(fields[0].lat,context.target_lat) or not np.array_equal(fields[0].lon,context.target_lon):
                raise ValueError('Reconstruction requires target-grid parents')
            p['reconstruction'].coarse_values.sel(time=fields[0].time)
        except (ValueError,KeyError) as exc:return Compatibility(False,(str(exc),))
        return Compatibility(True,())

    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError("Invalid spatiotemporal-detail contract")
        p=context.parameters;r=p["reconstruction"];index=p["fine_index"]
        fine=context.sources[index].dataset[context.variable]
        coarse=context.sources[1-index].dataset[context.variable]
        xr.align(fine,coarse,join="exact")
        alpha=p["detail_strength"]
        initial=alpha*fine.values+(1-alpha)*coarse.values
        coarse_observed=r.coarse_values.sel(time=fine.time).values
        values=r.project(initial,coarse_observed)
        output=context.sources[index].dataset[[context.variable]].copy()
        output[context.variable].values=values
        output.attrs.update(data_role="spatiotemporal_reconstruction",uncertainty_estimated=False)
        return AlgorithmResult(EcoDataBundle(output),self.spec.id,dict(detail_strength=alpha,
            coarse_constraint="exact declared cell-mean operator",reference_used=False,
            assumptions="fine nullspace detail interpolated from supplied anchors; not subpixel truth recovery proof"))
