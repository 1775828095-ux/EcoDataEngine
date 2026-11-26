"""Aligned-series missing fallback and explicitly prior fitted corrections."""
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility,AlgorithmResult
from ecodata.core.schemas import EcoDataBundle


class SeriesMissingFallback:
    spec=AlgorithmSpec('fusion.series_missing_fallback','Preserve primary observations and fill missing from support','1',
        'multi_source_fusion','available',('aligned_series',),('continuous',),
        input_contract=dict(source_arity=2,requires_parameters=['primary_index']),
        capabilities=('missing_fallback','retained_observation_protection'),backend='xarray where; no new interpolation kernel')
    def parameter_proposals(self,context,rng):
        return [dict(primary_index=i) for i in range(2)] if len(context.sources)==2 else []
    def compatibility(self,context):
        if len(context.sources)!=2 or context.parameters.get('primary_index') not in (0,1):
            return Compatibility(False,('Two sources and an explicit primary index required',))
        try:
            arrays=[s.dataset[context.variable] for s in context.sources]
            xr.align(*arrays,join='exact')
            for name in ('units','physical_variable','spatial_semantics','temporal_semantics'):
                values=[a.attrs.get(name) for a in arrays]
                if None in values or len(set(values))!=1:raise ValueError('Unknown or unequal '+name)
        except (KeyError,ValueError) as exc:return Compatibility(False,(str(exc),))
        return Compatibility(True,())
    def run(self,context):
        check=self.compatibility(context)
        if not check.eligible:raise ValueError(str(check.reasons))
        i=context.parameters['primary_index'];variable=context.variable
        source=context.sources[i].dataset;support=context.sources[1-i].dataset
        output=source.copy(deep=True);a=source[variable];b=support[variable]
        output[variable]=xr.where(np.isfinite(a),a,b)
        output[variable].attrs=dict(a.attrs)
        return AlgorithmResult(EcoDataBundle(output),self.spec.id,dict(primary_index=i,
            filled=int((~np.isfinite(a)&np.isfinite(b)).sum()),retained_values_unchanged=True))


class SeriesNonnegative:
    spec=AlgorithmSpec('constraint.series_nonnegative','Explicit nonnegative series projection','1',
        'raster_constraint','available',('time_series',),('continuous',),
        input_contract=dict(source_arity=1,parameter_domains={}),backend='Existing NumPy nonnegative projection on named time axes')
    def compatibility(self,context):
        ok=(len(context.sources)==1 and context.model_constraints.get('nonnegative') and
            context.variable in context.sources[0].dataset and 'time' in context.sources[0].dataset[context.variable].dims)
        return Compatibility(bool(ok),() if ok else ('An explicitly nonnegative timed task required',))
    def run(self,context):
        source=context.sources[0].dataset;output=source.copy(deep=True)
        output[context.variable]=output[context.variable].clip(min=0)
        return AlgorithmResult(EcoDataBundle(output),self.spec.id,dict(projection='Nonnegative; no resampling'))


class PriorSeriesCalibration:
    def __init__(self,method):
        self.method=method
        self.spec=AlgorithmSpec('bias.prior_series_'+method,'Prior series '+method+' calibration','1',
            'bias_correction','available',('daily_series',),('continuous',),
            input_contract=dict(source_arity=1,parameter_domains={}),requires_training=True,
            capabilities=('bias_correction',),backend='Existing fit_affine_source state / NumPy prior multiplicative scaling',
            notes='Prior fit precedes application and binds its source. Authorized observation-state parents may supply exact-epoch coefficients; no target evaluation labels')
    def compatibility(self,context):
        state=context.parameters.get('fitted_state');reasons=[]
        if len(context.sources)!=1:return Compatibility(False,('One source required',))
        ds=context.sources[0].dataset
        if state is None or state.attrs.get('calibration_source_id')!=ds.attrs.get('source_id'):
            reasons.append('Prior state must bind this actual source')
        elif pd.Timestamp(state.attrs['training_end'])>=pd.Timestamp(ds.time.values[0]):
            reasons.append('Prior fitting must precede application')
        if ds.attrs.get('calibration_signature') or ds.attrs.get('bias_adjustment_applied') or ds.attrs.get('fusion_method'):
            reasons.append('Already calibrated distribution has no matching training pair')
        if context.parameters.get('observation_state_update') and self.method=='monthly_ratio':
            reasons.append('Monthly ratio has no concurrent observation-state update contract')
        fields={'affine':['affine_slope','affine_intercept'],'ratio':['mean_ratio'],
                'monthly_ratio':['monthly_ratio']}[self.method]
        if state is None or any(k not in state or not np.isfinite(state[k]).all() for k in fields):
            reasons.append('Finite explicit fitted coefficients required')
        if state is not None:
            for name in fields:
                if name in state and state[name].ndim and self.method!='monthly_ratio':
                    if (state[name].dims!=('time',) or state.attrs.get('state_update_mode')!='contemporaneous_analysis'
                            or not np.array_equal(state.time.values,ds.time.values)):
                        reasons.append('Observation state requires exact actual source epochs')
                    if not context.parameters.get('_observation_state_bound',False):
                        reasons.append('Updated state requires its actual observation-state producer parent')
        return Compatibility(not reasons,tuple(reasons))
    def run(self,context):
        check=self.compatibility(context)
        if not check.eligible:raise ValueError(str(check.reasons))
        state=context.parameters['fitted_state'];source=context.sources[0].dataset
        output=source.copy(deep=True);a=source[context.variable]
        def coefficient(name):
            coefficient=state[name]
            if coefficient.ndim==0:return float(coefficient)
            if coefficient.dims!=('time',) or state.attrs.get('state_update_mode')!='contemporaneous_analysis':
                raise ValueError('Time-indexed calibration requires an explicit observation-state contract')
            if not np.array_equal(coefficient.time.values,source.time.values):raise ValueError('State epochs differ from actual source periods')
            return coefficient
        if self.method=='affine':value=coefficient('affine_slope')*a+coefficient('affine_intercept')
        elif self.method=='ratio':value=coefficient('mean_ratio')*a
        else:
            ratio=xr.DataArray(state.monthly_ratio.values[pd.DatetimeIndex(a.time.values).month-1],dims='time',coords={'time':a.time})
            value=ratio*a
        if context.model_constraints.get('nonnegative'):value=value.clip(min=0)
        output[context.variable]=value;output[context.variable].attrs=dict(a.attrs)
        output.attrs['calibration_signature']='prior_series_'+self.method
        if state.attrs.get('state_update_mode'):
            output.attrs.update(fusion_method='authorized_concurrent_observation_conditioning',state_update_mode=state.attrs['state_update_mode'])
        return AlgorithmResult(EcoDataBundle(output),self.spec.id,dict(fitted_on='Bound prior and authorized exact-period other-station observations' if state.attrs.get('state_update_mode') else 'Explicit prior state only',
            training_end=state.attrs['training_end'],method=self.method,reference_used=False,
            **({'state_update_mode':state.attrs['state_update_mode']} if state.attrs.get('state_update_mode') else {})))
