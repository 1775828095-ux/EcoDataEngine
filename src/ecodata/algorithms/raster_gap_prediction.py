"""Raster recovery adapters over existing SciPy and shared feature-forest kernels."""
from time import perf_counter
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from sklearn.impute import SimpleImputer
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility
from ecodata.algorithms.feature_forest import fit_station_forest
from ecodata.algorithms.open_source_spatial import finish


def eligible(context,*,allow_complete=False):
    if not context.parameters.get('grid_gap_recovery'):raise ValueError('Explicit grid-gap task required')
    if len(context.sources)!=1 or context.semantic_type!='continuous':raise ValueError('One continuous parent required')
    source=context.sources[0].dataset;field=source[context.variable]
    if field.dims!=('time','lat','lon') or min(field.sizes['lat'],field.sizes['lon'])<2:
        raise ValueError('Timed two-dimensional rectilinear fields required')
    if not allow_complete and not np.isnan(field.values).any():raise ValueError('No missing parent values; recovery would be an identity')
    return source


class GridGapRecovery:
    def __init__(self,method):
        self.method=method
        self.spec=AlgorithmSpec('reconstruction.grid_gap_'+method,'Grid gap recovery: '+method,'1','raster_reconstruction','available',
            ('timed_rectilinear_grid',),('continuous',),input_contract=dict(source_arity=1),
            capabilities=('spatial_missing_recovery',),backend='Existing SciPy griddata; nearest fallback outside hull',license='BSD-3-Clause')
    def compatibility(self,context):
        try:eligible(context,allow_complete=True);return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def run(self,context):
        started=perf_counter();source=eligible(context,allow_complete=True);out=source.copy(deep=True)
        if not np.isnan(source[context.variable].values).any():
            return finish(context,self.spec,out,dict(visible_values_preserved=True,analytic_identity=True,reference_used=False),started)
        from pyproj import CRS,Transformer
        lon,lat=np.meshgrid(source.lon,source.lat)
        crs=CRS(source.attrs.get('crs','EPSG:4326'))
        if crs.is_geographic:
            zone=int((float(source.lon.mean())+180)//6)+1
            projection=Transformer.from_crs(crs,f'EPSG:{32600+zone if float(source.lat.mean())>=0 else 32700+zone}',always_xy=True)
            x,y=projection.transform(lon,lat)
        else:x,y=lon,lat
        points=np.column_stack([x.ravel(),y.ravel()]);fallback=0
        for index in range(len(source.time)):
            values=source[context.variable].values[index].ravel();known=np.isfinite(values);missing=~known
            if not missing.any():continue
            if known.sum()<4:raise ValueError('At least four real visible grid cells required')
            prediction=griddata(points[known],values[known],points[missing],method=self.method)
            outside=~np.isfinite(prediction);fallback+=int(outside.sum())
            if outside.any():prediction[outside]=griddata(points[known],values[known],points[missing][outside],method='nearest')
            values=out[context.variable].values[index].copy().ravel();values[missing]=prediction
            out[context.variable].values[index]=values.reshape(out[context.variable].values[index].shape)
        return finish(context,self.spec,out,dict(visible_values_preserved=True,boundary_nearest_cells=fallback,reference_used=False),started)


class PriorGridForest:
    def __init__(self):
        self.spec=AlgorithmSpec('reconstruction.prior_grid_forest','Prior grid covariate forest','1',
            'raster_reconstruction','available',('timed_rectilinear_grid','prior_training_grid','covariate_grid'),('continuous',),
            requires_training=True,requires_auxiliary=True,input_contract=dict(source_arity=1,requires_parameters=['grid_predictor_features']),
            capabilities=('auxiliary_variable_estimation','spatial_missing_recovery'),compute='medium',
            backend='Existing fit_station_forest numerical kernel; automatic subset proposals from typed covariates',license='BSD-3-Clause')
    def available_features(self,context):
        if len(context.auxiliaries)!=1:return []
        return sorted(n for n,a in context.auxiliaries[0].data_vars.items() if a.attrs.get('feature_role')=='covariate'
            and a.dims==('time','lat','lon') and a.attrs.get('units'))
    def parameter_proposals(self,context,rng):
        names=self.available_features(context)
        if not names:return []
        count=int(rng.integers(1,len(names)+1))
        subset=sorted(rng.choice(names,size=count,replace=False).tolist())
        return [dict(grid_predictor_features=choice) for choice in [names,subset] if choice]
    def inputs(self,context):
        source=eligible(context)
        if len(context.auxiliaries)!=1:raise ValueError('One typed prior-and-application covariate resource required')
        aux=context.auxiliaries[0];names=context.parameters.get('grid_predictor_features')
        if not names or len(set(names))!=len(names) or not set(names)<=set(self.available_features(context)):
            raise ValueError('Prediction features must be a unique subset of observed typed covariates')
        if 'training_target' not in aux or aux.training_target.attrs.get('split')!='train':raise ValueError('Explicit prior training target required')
        end=pd.Timestamp(aux.attrs['training_end']);times=pd.DatetimeIndex(aux.time.values)
        if pd.Timestamp(source.time.values.min())<=end:raise ValueError('Training/application times overlap')
        if any(n not in aux or not aux[n].attrs.get('units') for n in names):raise ValueError('Named covariates with independent units required')
        if not np.array_equal(aux.lat,source.lat) or not np.array_equal(aux.lon,source.lon):raise ValueError('Covariate grid mismatch')
        if np.isfinite(aux.training_target.sel(time=times>end)).any():raise ValueError('Future target values in fitting resource')
        if not pd.DatetimeIndex(source.time.values).isin(times).all():raise ValueError('Covariate time support incomplete')
        return source,aux,names,end
    def compatibility(self,context):
        try:self.inputs(context);return Compatibility(True,())
        except (ValueError,KeyError,TypeError,IndexError) as exc:return Compatibility(False,(str(exc),))
    def run(self,context):
        started=perf_counter();source,aux,names,end=self.inputs(context)
        training=aux.sel(time=slice(None,end));truth=training.training_target.values.ravel();valid=np.isfinite(truth)
        x=np.column_stack([training[n].values.ravel() for n in names])
        if valid.sum()<20:raise ValueError('At least twenty prior labelled grid cells required')
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        x=imputer.fit_transform(x[valid]);pixel_count=source.sizes['lat']*source.sizes['lon']
        ids=np.tile(np.arange(pixel_count),len(training.time))[valid]
        table=pd.DataFrame(x,columns=names);table['value']=truth[valid];table['station_id']=['grid:'+str(i) for i in ids]
        model,_,_=fit_station_forest(table,names,compute_oof=False)
        application=aux.sel(time=source.time);query=np.column_stack([application[n].values.ravel() for n in names])
        prediction=model.predict(imputer.transform(query)).reshape(source[context.variable].shape)
        out=source.copy(deep=True);observed=np.isfinite(out[context.variable]);out[context.variable]=xr.where(observed,out[context.variable],prediction)
        out[context.variable].attrs=dict(source[context.variable].attrs)
        out.attrs['prior_grid_fit_signature']=hashlib.sha256(np.column_stack([x,truth[valid]]).tobytes()).hexdigest()
        return finish(context,self.spec,out,dict(features=names,training_end=str(end),training_grid_samples=int(valid.sum()),
            spatial_groups=int(len(set(ids))),group_kind='grid cells, not stations',reference_used=False,
            imputation='training-only sklearn median; raw covariate gaps retained in source files',visible_values_preserved=True),started)


class DeclaredRangeProjection:
    spec=AlgorithmSpec('constraint.declared_range_projection','Explicit task range projection','1','raster_constraint','available',
        ('rectilinear_grid',),('continuous',),input_contract=dict(source_arity=1),
        capabilities=('physical_range_validation',),backend='NumPy clip using declared task bounds',license='BSD-3-Clause')
    def compatibility(self,context):
        bounds=context.parameters.get('target_valid_range')
        if len(context.sources)!=1 or bounds is None or len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
            return Compatibility(False,('Explicit finite physical range and one parent required',))
        return Compatibility(True,())
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError('Explicit physical range required')
        started=perf_counter();out=context.sources[0].dataset.copy(deep=True);low,high=context.parameters['target_valid_range']
        out[context.variable]=out[context.variable].clip(min=low,max=high)
        return finish(context,self.spec,out,dict(declared_bounds=[low,high],reference_used=False),started)


def plugins():
    return [GridGapRecovery(m) for m in ('nearest','linear','cubic')]+[PriorGridForest(),DeclaredRangeProjection()]
