"""Prior-trained grid-to-station estimates, then observed-value-preserving fusion.

Sampling a gridded estimate does not turn it into a point observation. The explicit
paired-data model estimates a point value; unmeasured uncertainty stays unknown.
All numeric components already exist in xarray and the series calibration API.
"""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.contracts import AlgorithmSpec,AlgorithmResult,Compatibility
from ecodata.algorithms.quality_aware import fit_affine_source
from ecodata.algorithms.series_fusion import PriorSeriesCalibration,SeriesMissingFallback

GRID_SUPPORTS={'cell_mean','grid_estimate'}

def sample_grid(grid,variable,latitude,longitude,method):
    """Sample a covariate, retaining its declared grid support and provenance."""
    a=grid[variable]
    if a.dims!=('time','lat','lon') or a.attrs.get('spatial_semantics') not in GRID_SUPPORTS:
        raise ValueError('Timed gridded estimates/declared cell means required; no guessed point/grid semantics')
    if grid.attrs.get('geometry_kind')=='extracted_cell_series':
        if method!='extracted':raise ValueError('An extracted pixel is not a new rectilinear grid')
        if not (a.sizes['lat']==a.sizes['lon']==1):raise ValueError('One documented extracted cell required')
        if grid.attrs.get('pixel_selection')!='nearest_native_cell' or not grid.attrs.get('extraction_evidence'):
            raise ValueError('Extracted cell needs documented selection operator')
        query=[grid.attrs.get('extraction_latitude'),grid.attrs.get('extraction_longitude')]
        if None in query or not np.allclose(query,[latitude,longitude],rtol=0,atol=1e-8):
            raise ValueError('Extracted grid query does not match the target station')
        if not np.allclose([float(a.lat[0]),float(a.lon[0])],query,rtol=0,atol=1e-8):
            raise ValueError('Extracted cell coordinates must identify their documented query')
        result=grid.copy(deep=True)
        report=dict(operator='documented nearest-native-cell extraction',query=query,
            evidence=grid.attrs['extraction_evidence'],coordinates_are='query location, not inferred native cell centre')
    else:
        if method not in {'linear','nearest'} or min(a.sizes['lat'],a.sizes['lon'])<2:
            raise ValueError('Linear/nearest sampling needs a real rectilinear grid')
        for axis,value in [('lat',latitude),('lon',longitude)]:
            v=np.asarray(a[axis],float)
            if not np.isfinite(v).all() or not (np.all(np.diff(v)>0) or np.all(np.diff(v)<0)) or not min(v)<=value<=max(v):
                raise ValueError('Station outside known grid centres or invalid geometry; no extrapolation')
        result=grid.interp(lat=[latitude],lon=[longitude],method=method)
        result[variable].attrs=dict(a.attrs)
        report=dict(operator='xarray '+method+' grid covariate sampling',query=[latitude,longitude],extrapolation=False)
    return result,report


def _daily(ds,variable):
    a=ds[variable]
    if a.dims!=('time','lat','lon') or len(ds.time)<2 or not np.all(np.diff(ds.time.values)==np.timedelta64(1,'D')):
        raise ValueError('Explicit regular daily time/lat/lon support required')
    if 'time_bounds' not in ds or ds.time_bounds.shape!=(len(ds.time),2):raise ValueError('Daily interval bounds required')
    bounds=ds.time_bounds.values
    if not np.array_equal(bounds[:,0],ds.time.values) or not np.all(bounds[:,1]-bounds[:,0]==np.timedelta64(1,'D')):
        raise ValueError('Daily accumulation intervals must match labels; no calendar guessing')
    if not CRS.from_user_input(ds.attrs.get('crs','')).equals(CRS.from_epsg(4326)):
        raise ValueError('Reviewed station/grid sampler requires EPSG:4326')
    if np.isinf(a).any() or bool((a<0).any()):raise ValueError('Invalid daily precipitation input')


class StationGridFusion:
    def __init__(self,method):
        if method not in {'affine','ratio','monthly_ratio'}:raise ValueError('Unknown prior point-estimation model')
        self.method=method
        self.spec=AlgorithmSpec('fusion.station_grid_'+method,'Prior grid-to-station '+method+' estimate and missing fallback','1',
            'multi_source_fusion','available',('point_series','gridded_precipitation'),('continuous',),requires_training=True,
            input_contract=dict(source_arity=2),capabilities=('multi_source_fusion','temporal_missing_recovery',
                'station_grid_support_estimation','retained_observation_protection'),
            backend='xarray sampling + existing fit_affine_source/PriorSeriesCalibration + SeriesMissingFallback',
            notes='Empirical point prediction from strictly prior paired data; not exact support equivalence or independent sensor evidence')

    def parameter_proposals(self,context,rng):
        grids=[s.dataset for s in context.sources if s.dataset[context.variable].attrs.get('spatial_semantics') in GRID_SUPPORTS]
        if len(grids)!=1:return []
        return [dict(sampling=m) for m in (['extracted'] if grids[0].attrs.get('geometry_kind')=='extracted_cell_series' else ['nearest','linear'])]

    def inputs(self,context):
        if len(context.sources)!=2 or context.parameters.get('allow_training') is not True:
            raise ValueError('Two sources and explicitly authorized prior fitting required')
        if not context.model_constraints.get('nonnegative'):raise ValueError('Explicit nonnegative precipitation requirement needed')
        variable=context.variable;datasets=[s.dataset for s in context.sources]
        points=[d for d in datasets if d[variable].attrs.get('spatial_semantics')=='point']
        grids=[d for d in datasets if d[variable].attrs.get('spatial_semantics') in GRID_SUPPORTS]
        if len(points)!=1 or len(grids)!=1:raise ValueError('Exactly one point source and one explicitly gridded source required')
        point,grid=points[0],grids[0];resources=context.parameters.get('prior_resources',{})
        if not {'training_reference','historical_simulation'}<=set(resources):raise ValueError('Prior paired station and grid resources required')
        target,history=resources['training_reference'],resources['historical_simulation']
        for d in [point,grid,target,history]:_daily(d,variable)
        if point.sizes['lat']!=1 or point.sizes['lon']!=1 or target.sizes['lat']!=1 or target.sizes['lon']!=1:
            raise ValueError('One real target station required')
        if not point.attrs.get('source_id') or target.attrs.get('source_id')!=point.attrs['source_id']:
            raise ValueError('Historical station identity does not bind target')
        if not grid.attrs.get('source_id') or history.attrs.get('source_id')!=grid.attrs['source_id']:
            raise ValueError('Historical grid identity does not bind this source')
        for name in ['lat','lon']:
            if not np.array_equal(point[name],target[name]) or not np.array_equal(grid[name],history[name]):
                raise ValueError('Historical/current source geometry differs')
        if grid.attrs.get('calibration_signature') or grid.attrs.get('fusion_method') or grid.attrs.get('bias_adjustment_applied'):
            raise ValueError('Already modified grid has no matching prior fitting distribution')
        for key in ['units','physical_variable','physical_definition','temporal_semantics']:
            values=[d[variable].attrs.get(key) for d in [point,grid,target,history]]
            if None in values or len(set(values))!=1:raise ValueError('Unknown/different physical contract: '+key)
        if point[variable].attrs['units']!='mm/day' or point[variable].attrs['physical_variable']!='precipitation_amount' or point[variable].attrs['temporal_semantics']!='accumulation':
            raise ValueError('This prior transfer supports explicitly daily precipitation amounts only')
        if target[variable].attrs.get('spatial_semantics')!='point' or history[variable].attrs.get('spatial_semantics')!=grid[variable].attrs['spatial_semantics']:
            raise ValueError('Prior pair must preserve the station and actual declared grid support')
        # Crop extra background dates by exact labels, never interpolate daily
        # totals or use an unknown date's station target in the prior fit.
        if not pd.DatetimeIndex(point.time.values).isin(grid.time.values).all():
            raise ValueError('Grid must provide explicit records for every target date')
        grid=grid.sel(time=point.time)
        for a,b in [(point,grid),(target,history)]:
            if not np.array_equal(a.time,b.time) or not np.array_equal(a.time_bounds,b.time_bounds):
                raise ValueError('Source daily intervals differ; align using an explicit temporal operator')
        for d in [target,history]:
            if any(v.attrs.get('split') in {'test','final_test','external_test','validation'} for v in [d,*d.variables.values()]):
                raise ValueError('Fitting resources cannot be validation/test labels')
            end=d.attrs.get('training_end')
            if end is None or pd.Timestamp(end)<pd.Timestamp(d.time.values[-1]) or pd.Timestamp(end)>=pd.Timestamp(point.time.values[0]):
                raise ValueError('Fitting period must precede application; no overlapping or misleading training provenance')
        if not bool(point[variable].isnull().any()):raise ValueError('No missing target values; preserve observations without redundant fusion')
        lat,lon=float(point.lat[0]),float(point.lon[0])
        if context.target_lat is None or context.target_lon is None or not np.allclose(context.target_lat,[lat],rtol=0,atol=1e-8) or not np.allclose(context.target_lon,[lon],rtol=0,atol=1e-8):
            raise ValueError('Target station differs from requested support')
        sampling=context.parameters.get('sampling')
        sampled,operator=sample_grid(grid,variable,lat,lon,sampling)
        prior,prior_operator=sample_grid(history,variable,lat,lon,sampling)
        if operator['operator']!=prior_operator['operator'] or operator['query']!=prior_operator['query']:
            raise ValueError('Historical and current sampling operators differ')
        return point,grid,target,prior,sampled,operator

    def compatibility(self,context):
        try:self.inputs(context)
        except (ValueError,KeyError,TypeError,IndexError) as exc:return Compatibility(False,(str(exc),))
        return Compatibility(True,())

    def run(self,context):
        point,grid,target,prior,sampled,operator=self.inputs(context);v=context.variable
        x=prior[v].values[:,0,0];y=target[v].values[:,0,0];valid=np.isfinite(x)&np.isfinite(y)
        if valid.sum()<3:raise ValueError('At least3 finite historical pairs required')
        fields={};months=pd.DatetimeIndex(prior.time.values).month
        if self.method=='affine':
            slope,intercept=fit_affine_source(x[valid],y[valid]);fields=dict(affine_slope=slope,affine_intercept=intercept)
        elif self.method=='ratio':
            if x[valid].sum()<=0:raise ValueError('Historical rainfall ratio is unidentifiable')
            fields=dict(mean_ratio=float(y[valid].sum()/x[valid].sum()))
        else:
            ratios=[]
            for month in range(1,13):
                selected=valid&(months==month)
                if selected.sum()<3 or x[selected].sum()<=0:raise ValueError('Monthly rainfall ratio lacks identifiable training pairs')
                ratios.append(float(y[selected].sum()/x[selected].sum()))
            fields=dict(monthly_ratio=('month',ratios))
        state=xr.Dataset(fields,attrs=dict(training_end=target.attrs['training_end'],calibration_source_id=grid.attrs['source_id']))
        local=replace(context,sources=(EcoDataBundle(sampled),),parameters=dict(context.parameters,fitted_state=state))
        prediction=PriorSeriesCalibration(self.method).run(local).data.dataset
        # Only the fitted predictive model changes the output support claim.
        prediction[v].attrs=dict(point[v].attrs,estimate_kind='prior-trained point estimate, not an observed station value',
            support_mapping='Empirical paired-data prediction; not exact cell-to-point support equivalence')
        result=SeriesMissingFallback().run(replace(context,sources=(EcoDataBundle(point),EcoDataBundle(prediction)),
            parameters=dict(context.parameters,primary_index=0)))
        ds=result.data.dataset;ds.attrs.update(fusion_method='station_grid_prior_'+self.method,
            calibration_signature='station_grid_prior_'+self.method,point_source_id=point.attrs['source_id'],grid_source_id=grid.attrs['source_id'],
            derived_from=json.dumps([point.attrs['source_id'],grid.attrs['source_id']]),
            uncertainty_scope='UNKNOWN representativeness and fitted-parameter uncertainty; no invented uncertainty weights')
        ds[v].attrs['output_kind']='Mixed station observations and grid-informed point estimates; never raw cell values relabelled as observations'
        return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(method=self.method,sampling=operator,
            training_start=str(target.time.values[0]),training_end=target.attrs['training_end'],training_pairs=int(valid.sum()),
            excluded_nonfinite_training_pairs=int((~valid).sum()),coefficients={k:np.asarray(val[1] if isinstance(val,tuple) else val).tolist() for k,val in fields.items()},
            point_source=point.attrs['source_id'],grid_source=grid.attrs['source_id'],reference_used=False,
            stages=['sample gridded covariate','fit prior paired point-estimation model','apply existing prior calibration','existing missing fallback'],
            original_grid_semantics_preserved=True,**result.report))
