"""Daily station reconstruction adapters for the shared composition service.

Only visible observations and explicitly prior fitted state enter kernels.
Visible-mask probes are accounted separately; no SWAT response is inferred.
"""
from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd

from ecodata.algorithms.contracts import AlgorithmSpec, AlgorithmResult, Compatibility
from ecodata.algorithms.temporal_kernels import reconstruct
from ecodata.core.schemas import EcoDataBundle


def gap_lengths(values):
    missing=~np.isfinite(values)
    starts=np.flatnonzero(missing & ~np.r_[False,missing[:-1]])
    ends=np.flatnonzero(missing & ~np.r_[missing[1:],False])+1
    result=np.zeros(len(values),int)
    for start,end in zip(starts,ends):result[start:end]=end-start
    return result


def visible_day_mask(values, rng, fraction=.1):
    """Original daily probe mask, reusable by single- and multi-source DAGs."""
    values=np.asarray(values);lengths=gap_lengths(values)
    unique,counts=np.unique(lengths[lengths>0],return_counts=True)
    if not len(unique):unique,counts=np.array([1]),np.array([1])
    wanted=max(12,int(np.isfinite(values).sum()*fraction));mask=np.zeros(len(values),bool)
    for start in rng.permutation(np.arange(1,len(values)-1)):
        size=int(rng.choice(unique,p=counts/counts.sum()));end=start+size
        if end>=len(values) or not np.isfinite(values[start-1:end+1]).all() or mask[max(0,start-1):end+1].any():continue
        mask[start:end]=True
        if mask.sum()>=wanted:break
    return mask,dict(zip(map(str,unique),counts.tolist()))


class DailyRecovery:
    def __init__(self,method):
        self.method=method
        self.spec=AlgorithmSpec("reconstruction.daily_"+method,"Daily "+method,"1",
            "temporal_reconstruction","available",("regular_daily_station",),("continuous",),
            capabilities=("daily_missing_recovery","retained_observation_protection"),
            input_contract=dict(source_arity=1,deterministic=True,state_complete=True,
                parameter_domains=dict(max_gap_days=dict(source='observed_missing_run_lengths'))),
            backend="Existing reconstruct pandas/SciPy kernels; pandas fitted monthly mean",
            notes="Point series only; no spatial downscaling or daily-total inference from subdaily input")

    def compatibility(self,context):
        ds=context.sources[0].dataset;reasons=[]
        if context.variable in ds and any(ds[context.variable].attrs.get(k)!=v for k,v in {
                'units':'mm/day','physical_variable':'precipitation_amount',
                'spatial_semantics':'point','temporal_semantics':'accumulation'}.items()):
            reasons.append("Explicit point daily precipitation amount in mm/day required")
        if context.variable not in ds or ds[context.variable].dims!=("time","lat","lon"):
            reasons.append("Daily adapter needs time/lat/lon station data")
        elif ds.sizes['lat']!=1 or ds.sizes['lon']!=1:
            reasons.append("Only one actual station is supported, not a manufactured spatial field")
        elif len(ds.time)<2 or not np.all(np.diff(ds.time.values)==np.timedelta64(1,'D')):
            reasons.append("Explicit regular daily timestamps required")
        if self.method=='month_mean':
            state=context.parameters.get('fitted_state')
            if state is None or 'month_mean' not in state:
                reasons.append("Prior fitted monthly state required")
            elif pd.Timestamp(state.attrs['training_end'])>=pd.Timestamp(ds.time.values[0]):
                reasons.append("Fitted history must precede reconstruction input")
        return Compatibility(not reasons,tuple(reasons))

    def run(self,context):
        check=self.compatibility(context)
        if not check.eligible:raise ValueError(str(check.reasons))
        ds=context.sources[0].dataset;var=context.variable
        values=ds[var].values[:,0,0].copy();times=pd.DatetimeIndex(ds.time.values)
        finite=np.isfinite(values)
        if np.isinf(values).any() or (values[finite]<0).any():raise ValueError("Illegal visible precipitation")
        cutoff=context.parameters.get('max_gap_days')
        if cutoff is not None and (type(cutoff)!=int or cutoff<1):raise ValueError("Positive integer gap cutoff required")
        if self.method=='month_mean':
            state=context.parameters['fitted_state'];means=np.asarray(state.month_mean)
            if means.shape!=(12,) or not np.isfinite(means).all() or (means<0).any():raise ValueError("Invalid fitted monthly means")
            prediction=means[times.month-1]
        else:prediction=reconstruct(pd.Series(values,index=times),self.method).to_numpy()
        selected=~finite
        if cutoff is not None:selected &= gap_lengths(values)<=cutoff
        values[selected]=prediction[selected]
        output=ds.copy(deep=True);output[var].values[:,0,0]=values
        assert np.array_equal(output[var].values[:,0,0][finite],ds[var].values[:,0,0][finite])
        return AlgorithmResult(EcoDataBundle(output),self.spec.id,dict(
            method=self.method,max_gap_days=cutoff,filled=int(np.isfinite(values).sum()-finite.sum()),
            retained_values_unchanged=True,fitted_on='explicit prior state' if self.method=='month_mean' else 'no fit'))


class DailyPortraitScoring:
    """Same parallel score aggregation, with actual single-series anchor probes."""
    from ecodata.algorithms.composition_scoring import ParallelPortraitScoring as _Shared
    score=_Shared.score
    receipt=_Shared.receipt
    local_key=staticmethod(_Shared.local_key)

    def __init__(self,engine,declaration):
        from ecodata.algorithms.composition_decision import Composition
        started=perf_counter()
        if set(declaration)!={'weights','evidence'} or not declaration['evidence']:
            raise ValueError('Daily parallel scores need explicit weights/evidence')
        weights=declaration['weights']
        if set(weights)!={'data','model','collaboration'} or any(not np.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
            raise ValueError('Invalid daily route weights')
        self.weights={k:float(v)/sum(weights.values()) for k,v in weights.items()}
        if self.weights['collaboration']!=0:
            raise ValueError('Daily collaboration is diagnostic only; no self-relative bonus')
        self.evidence=declaration['evidence'];self.reference=None;self.reference_probes=[]
        self.probes=[];self.responses={};self.executions=0;self.kernel_calls=0
        self.scope='Internal visible-day holdout; three-portrait engineering suitability, not measured SWAT sensitivity'
        self.unsupported=None
        if len(engine.ids)!=1 or len(engine.fields)!=1:raise ValueError('Daily scorer requires a single field/station')
        field=engine.fields[0];req=field['request'];var=field['variable'];key=engine.ids[0]
        state=field['contexts'][key].parameters.get('fitted_state')
        if state is None or 'scale' not in state:raise ValueError('Prior training normalization state required')
        scale=float(state.scale)
        if not np.isfinite(scale) or scale<=0:raise ValueError('Positive training-only scale required')
        local=deepcopy(engine.request);local['composition'].pop('parallel_scoring')
        parsed=[];holds=[]
        rng=np.random.default_rng(704)
        for role,original in [('primary',engine.sources[key]),*engine.resources]:
            ds=original.copy(deep=True)
            if role=='primary':
                values=ds[var].values[:,0,0]
                mask,length_counts=visible_day_mask(values,rng)
                if not mask.any():raise ValueError('No independent visible-day probe cohort')
                truth=values[mask].copy();values[mask]=np.nan
                holds=np.flatnonzero(mask)
                self.probes.append(dict(window='all',withheld_indices=holds.tolist(),truth=truth.tolist(),
                    normalization='prior training precipitation standard deviation',scale=scale,
                    original_gap_lengths=length_counts,
                    hidden_validation_or_test_labels_accessed=False))
            parsed.append((role,ds))
        probe=Composition(parsed,local,engine.registry)
        if probe.count>512:raise ValueError('Narrow explicit daily grammar to <=512 real probes')
        for index in range(probe.count):
            genome=probe.decode(index);plan=probe.plan(genome);self.executions+=1
            try:
                ds,_,trace,checks=probe.execute(genome)
                self.kernel_calls+=sum(t.get('kernel_calls',0) for t in trace)
                predicted=ds[var].values[:,0,0][holds];error=float(np.sqrt(np.mean((predicted-truth)**2)))
                bias=float(abs(np.mean(predicted-truth))/scale)
                state_metrics=engine.states(ds,field['contexts'][key])
                # This registered metric is explicitly an input-volume proxy, not model error.
                state_metrics['expected_error']=bias
                model=engine.score(field,final=state_metrics,cost=probe.steps(genome))
                from ecodata.algorithms.composition_scoring import visible_response
                response=visible_response(error,scale,model['score_interval'],evidence=self.evidence,
                    probe_prediction=predicted.tolist(),normalized_absolute_volume_bias=bias,model_terms=model['terms'],
                    state_transitions=trace,coordination=checks)
            except (ValueError,KeyError,TypeError,IndexError) as exc:response=dict(status='UNKNOWN',reason=str(exc))
            self.responses[('all',self.local_key(plan,'all'))]=response
        self.seconds=perf_counter()-started
