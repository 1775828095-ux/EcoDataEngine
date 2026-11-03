"""Visible-only raster time interpolation and cross-sensor anomaly transfer.

These are scientific nodes, not a graph pool. Independent optical sensors do
not share an exact cell-mean observation operator, so no exact conservation
claim or use of the complementary_observation_operator is made here.
"""
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility,AlgorithmResult
from ecodata.core.schemas import EcoDataBundle


def time_field(field,targets,method):
    field=field.transpose('time','lat','lon')
    source=pd.DatetimeIndex(field.time.values);target=pd.DatetimeIndex(targets)
    if source.has_duplicates or not source.is_monotonic_increasing or len(source)<2:
        raise ValueError('At least two increasing unique visible dates required')
    queue=source.union(target).sort_values()
    values=pd.DataFrame(field.values.reshape(len(source),-1),index=source).reindex(queue)
    if method=='linear':values=values.interpolate(method='time',limit_direction='both')
    elif method=='nearest':values=values.interpolate(method='nearest',limit_direction='both').ffill().bfill()
    else:raise ValueError('Unknown raster time interpolation')
    result=values.loc[target].to_numpy().reshape(len(target),field.sizes['lat'],field.sizes['lon'])
    return xr.DataArray(result,dims=field.dims,coords=dict(time=target,lat=field.lat,lon=field.lon),attrs=dict(field.attrs),name=field.name)


def check(context,arity):
    if len(context.sources)!=arity or context.semantic_type!='continuous':raise ValueError('Continuous timed raster parents required')
    targets=context.parameters.get('raster_target_times')
    if not targets:raise ValueError('Explicit raster target dates required')
    for source in context.sources:
        if source.dataset[context.variable].dims!=('time','lat','lon'):raise ValueError('Timed rectilinear raster required')
    return targets


class RasterTimeInterpolation:
    def __init__(self,method):
        self.method=method
        self.spec=AlgorithmSpec('temporal.raster_'+method,'Raster visible-date '+method,'1','temporal_reconstruction',
            'available',('timed_rectilinear_grid',),('continuous',),input_contract=dict(source_arity=1),
            capabilities=('irregular_time_interpolation','temporal_missing_recovery'),license='BSD-3-Clause',
            backend='pandas time/nearest interpolation; explicit nearest available endpoint outside support')
    def compatibility(self,context):
        try:
            targets=check(context,1);field=context.sources[0].dataset[context.variable]
            if len(field.time)<2:raise ValueError('Two dates required')
            return Compatibility(True,())
        except (ValueError,KeyError) as exc:return Compatibility(False,(str(exc),))
    def run(self,context):
        targets=check(context,1);source=context.sources[0].dataset
        out=time_field(source[context.variable],targets,self.method).to_dataset()
        out.attrs=dict(source.attrs)
        return AlgorithmResult(EcoDataBundle(out),self.spec.id,dict(reference_used=False,method=self.method,
            extrapolation='nearest available endpoint',all_missing_trace='remains missing'))


class VisibleAnchorResidual:
    """Coarse observed changes plus residuals from supplied fine anchor dates."""
    def __init__(self):
        self.spec=AlgorithmSpec('fusion.visible_anchor_residual','Visible fine-anchor residual transfer','1','multi_source_fusion',
            'available',('fine_sparse_raster','coarse_dense_raster'),('continuous',),input_contract=dict(source_arity=2,
                requires_parameters=['fine_index','residual_strength']),capabilities=('multi_source_fusion','temporal_detail_transfer'),
            backend='xarray spatial interpolation and pandas visible-anchor residual interpolation; not STARFM',license='BSD-3-Clause')
    def parameter_proposals(self,context,rng):
        if len(context.sources)!=2:return []
        sizes=[s.dataset.sizes['lat']*s.dataset.sizes['lon'] for s in context.sources]
        if sizes[0]==sizes[1]:return []
        return [dict(fine_index=int(np.argmax(sizes)),residual_strength=float(a)) for a in [1.,rng.uniform(0,1)]]
    def compatibility(self,context):
        try:
            check(context,2);p=context.parameters;i=p.get('fine_index')
            if i not in (0,1) or not 0<=p.get('residual_strength',-1)<=1:raise ValueError('Explicit fine branch and bounded residual strength required')
            a=context.sources[i].dataset[context.variable];b=context.sources[1-i].dataset[context.variable]
            if len(a.lat)*len(a.lon)<=len(b.lat)*len(b.lon):raise ValueError('Fine branch must have higher spatial resolution')
            if a.attrs.get('physical_definition')!=b.attrs.get('physical_definition'):raise ValueError('Same physical variable required')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError('Incompatible visible-anchor transfer')
        p=context.parameters;i=p['fine_index'];fine=context.sources[i].dataset[context.variable]
        coarse=context.sources[1-i].dataset[context.variable]
        dates=pd.DatetimeIndex(p['raster_target_times']);all_dates=dates.union(pd.DatetimeIndex(fine.time.values)).sort_values()
        filled=time_field(coarse,all_dates,'linear')
        up=filled.interp(lat=fine.lat,lon=fine.lon,method='linear')
        residual=fine-up.sel(time=fine.time)
        detail=time_field(residual,dates,'linear')
        unsupported=int((~np.isfinite(detail)).sum())
        # No observed fine residual means zero asserted fine detail: retain the
        # independently observed coarse background, not fabricated subpixels.
        output=up.sel(time=dates)+p['residual_strength']*detail.fillna(0.)
        output=output.interp(lat=list(context.target_lat),lon=list(context.target_lon))
        output.attrs=dict(fine.attrs);ds=output.to_dataset(name=context.variable)
        ds.attrs=dict(context.sources[i].dataset.attrs,data_role='cross_sensor_residual_reconstruction')
        return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(reference_used=False,residual_strength=p['residual_strength'],
            coarse_background_fallback_cells=unsupported,
            coarse_consistency='UNKNOWN cross-sensor point-spread and observation-date differences',anchor_dates=[str(t) for t in fine.time.values]))
