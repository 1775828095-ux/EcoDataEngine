"""Fixed chronological blocks and a one-shot recommendation gate.

Uses the existing measured-fold aggregator. Minimum support is an engineering
screen, not a confidence bound or a claim that neighbouring times are independent.
No references, candidate preferences or scientific kernels are constructed here.
"""
import hashlib
import json
import numpy as np
import pandas as pd

LEGACY_VERSION='blocked_confirmation_v1'
VERSION='blocked_confirmation_v2_aggregate_risk'
LEGACY_POLICY=dict(version=VERSION,blocks=3,min_times_per_block=2,
    objective='Equal-block mean RMSE',confirmation='Disjoint chronological holdout, once after freezing',
    acceptance='Strict aggregate RMSE gain and no block RMSE/MAE degradation in both stages',
    insufficient='Keep baseline; candidate available as explicitly exploratory',
    confidence='No statistical confidence or universal accuracy guarantee')

POLICY=dict(LEGACY_POLICY,version=VERSION,
    acceptance='Aggregate gain in both stages, paired block uncertainty, declared local risk limits',
    insufficient='Quality-valid candidate may be tried; baseline remains default until confirmed',
    resampling=dict(method='paired cluster percentile bootstrap',confidence_level=.95,resamples=4999,seed=20261125),
    local_risk='No default tolerance percent; undeclared local harm limits remain UNKNOWN')
LEGACY_POLICY['version']=LEGACY_VERSION



def time_blocks(reference):
    """Equal elapsed-time windows, not row parity or value-dependent sampling."""
    clock=reference.indexes.get('time')
    if not isinstance(clock,pd.DatetimeIndex):raise ValueError('Explicit datetime coordinates required; numeric labels are not inferred timestamps')
    clock=clock.as_unit('ns')
    if clock.hasnans or clock.has_duplicates or not clock.is_monotonic_increasing:
        raise ValueError('Blocked evaluation requires unique increasing real timestamps')
    if not len(clock):raise ValueError('No evaluation times')
    elapsed=clock.asi8-clock.asi8[0];span=int(elapsed[-1])
    labels=np.minimum((elapsed.astype(object)*POLICY['blocks']//max(1,span)).astype(int),POLICY['blocks']-1)
    finite=np.isfinite(reference.values).reshape(len(clock),-1).any(axis=1)
    rows=[]
    for index in range(POLICY['blocks']):
        selected=np.flatnonzero((labels==index)&finite)
        rows.append(dict(id='time-block-'+str(index),indices=selected.tolist(),times=[str(t) for t in clock[selected]],
            unique_times=len(selected),status='SUFFICIENT_MINIMUM' if len(selected)>=POLICY['min_times_per_block'] else 'INSUFFICIENT'))
    digest=hashlib.sha256(json.dumps(dict(policy=POLICY,blocks=rows),sort_keys=True).encode()).hexdigest()
    return dict(version=VERSION,blocks=rows,fingerprint=digest,
        sufficient=all(r['status']=='SUFFICIENT_MINIMUM' for r in rows),
        support_note='Unique valid times per window, not pixel count; no independence assertion')


def score(ds,reference,variable,plan):
    from ecodata.algorithms.platform_search import DatasetReference
    from ecodata.algorithms.reliable_selection import aggregate
    data=ds.sel(time=reference.time) if 'time' in reference.dims else ds
    evaluator=DatasetReference(reference,variable)
    pooled=evaluator(data)
    rows=[]
    for block in plan['blocks']:
        if 'site_indices' in block:
            mask=np.zeros(reference.shape,bool)
            if 'time' in reference.dims:
                flat=mask.reshape(reference.sizes['time'],-1)
                flat[np.ix_(block['indices'],block['site_indices'])]=True
            else:mask.reshape(-1)[block['site_indices']]=True
            import xarray as xr
            ref=reference.where(xr.DataArray(mask,dims=reference.dims,coords=reference.coords))
            if not np.isfinite(ref.values).any():
                rows.append(dict(fold_id=block['id'],status='NO_REFERENCE',metrics=None));continue
            metrics=DatasetReference(ref,variable)(data)
            rows.append(dict(fold_id=block['id'],status='PASS',metrics=metrics));continue
        if not block['indices']:
            rows.append(dict(fold_id=block['id'],status='NO_REFERENCE',metrics=None));continue
        ref=reference.isel(time=block['indices'])
        metrics=DatasetReference(ref,variable)(data.sel(time=ref.time))
        rows.append(dict(fold_id=block['id'],status='PASS',metrics=metrics))
    # Do not score only successful predictions. Every finite reference above is
    # mandatory, and an absent block remains an explicit evidence limitation.
    result=dict(pooled,block_metrics=rows,block_support=plan,pooled_RMSE=pooled['RMSE'],
        evaluation_definition=dict(variable=variable,unit=reference.attrs.get('units','UNKNOWN')),
        evaluation_fingerprint=hashlib.sha256(json.dumps(evaluator.protocol,sort_keys=True).encode()).hexdigest())
    if all(row['status']=='PASS' for row in rows):
        policy=dict(weights={'RMSE':1.},scales={'RMSE':1.},
            weight_basis='Frozen equal-block RMSE in the declared physical unit',
            scale_basis={'RMSE':'One declared physical unit; no fitted tolerance'},data_portrait={})
        summary=aggregate(rows,policy,[r['fold_id'] for r in rows])
        result.update(objective=summary['objective'],block_loss_stats=summary['loss_stats'],
            objective_basis='Equal-block mean RMSE; RMSE/MAE/Bias fields retain pooled physical metrics')
    else:
        result['objective_basis']='Diagnostic pooled RMSE only: missing temporal block, recommendation forbidden'
    return result


def reference_blocks(reference,axes):
    """Task-selected time/coordinate blocks using the existing spatial splitter."""
    if axes==['time']:return time_blocks(reference)
    from ecodata.algorithms.spatial_selection_protocol import spatial_blocks
    import itertools
    if not {'lat','lon'}<=set(reference.coords):raise ValueError('Spatial review needs explicit coordinate axes or paired locations')
    if reference.lat.dims==('lat',) and reference.lon.dims==('lon',):
        lon,lat=np.meshgrid(reference.lon.values,reference.lat.values)
    elif reference.lat.dims==reference.lon.dims:lat,lon=reference.lat.values,reference.lon.values
    else:raise ValueError('Unsupported review coordinate geometry')
    xy=list(zip(np.asarray(lon).ravel(),np.asarray(lat).ravel()))
    groups=spatial_blocks(pd.DataFrame(dict(longitude=[p[0] for p in xy],latitude=[p[1] for p in xy])))
    temporal=time_blocks(reference)['blocks'] if 'time' in axes else [dict(id='all-times',indices=list(range(reference.sizes.get('time',1))),times=[str(t) for t in reference.coords.get('time',[]).values] if 'time' in reference.coords else [])]
    rows=[];valid=np.isfinite(reference.values).reshape(reference.sizes.get('time',1),-1)
    occupied=np.flatnonzero(valid.any(axis=0));locations=[str(xy[i]) for i in occupied]
    for group,window in itertools.product(sorted(set(groups)),temporal):
        sites=np.flatnonzero(groups==group);present=valid[np.ix_(window['indices'],sites)]
        real=sites[present.any(axis=0)] if present.size else []
        rows.append(dict(id=str(group)+'|'+window['id'],resampling_unit=window['id'] if 'time' in axes else str(group),indices=window['indices'],times=window['times'],
            site_indices=sites.tolist(),unique_locations=len(real),status='SUFFICIENT_MINIMUM' if len(real)>=2 and
            ('time' not in axes or len(window['indices'])>=2) else 'INSUFFICIENT'))
    payload=dict(axes=axes,locations=locations,blocks=rows)
    return dict(payload,fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
        sufficient=len(set(groups))>=2 and all(r['status']=='SUFFICIENT_MINIMUM' for r in rows),
        support_note='Coordinate-axis quantile blocks; no claimed physical decorrelation')


def block_error_change(candidate, baseline):
    """Measured candidate-minus-baseline errors on a shared reference."""
    if candidate.get('count') != baseline.get('count'):
        raise ValueError('Candidate/baseline evaluation denominators differ')
    return dict(rmse_change=candidate['RMSE']-baseline['RMSE'],
                mae_change=candidate['MAE']-baseline['MAE'])


def compare_legacy(candidate,baseline):
    if candidate['block_support']['fingerprint']!=baseline['block_support']['fingerprint']:
        raise ValueError('Candidate/baseline block queues differ')
    if candidate['evaluation_fingerprint']!=baseline['evaluation_fingerprint']:
        raise ValueError('Candidate/baseline reference definitions differ')
    if candidate['denominator']!=baseline['denominator']:
        raise ValueError('Candidate/baseline evaluation denominators differ')
    if len(candidate['block_metrics'])!=len(baseline['block_metrics']):
        raise ValueError('Candidate/baseline block counts differ')
    differences=[]
    for a,b in zip(candidate['block_metrics'],baseline['block_metrics']):
        if a['fold_id']!=b['fold_id'] or a['status']!=b['status']:
            raise ValueError('Candidate/baseline block mismatch')
        if a['metrics'] is None:continue
        differences.append(dict(block=a['fold_id'], **block_error_change(a['metrics'], b['metrics'])))
    tolerance=64*np.finfo(float).eps*max(1.,abs(candidate['objective']),abs(baseline['objective']))
    return dict(sufficient=candidate['block_support']['sufficient'],differences=differences,
        objective_change=candidate['objective']-baseline['objective'],
        stable_gain=bool(differences and candidate['objective']<baseline['objective']-tolerance and
            all(r['rmse_change']<=tolerance and r['mae_change']<=tolerance for r in differences)),
        roundoff_only_tolerance=tolerance)


def review_legacy(selection,selection_baseline,confirmation,confirmation_baseline,*,quality_pass):
    first=compare_legacy(selection,selection_baseline);second=compare_legacy(confirmation,confirmation_baseline)
    if selection['evaluation_definition']!=confirmation['evaluation_definition']:
        raise ValueError('Selection/confirmation variable or unit mismatch')
    axes=selection['block_support'].get('axes',['time'])
    if axes!=confirmation['block_support'].get('axes',['time']):raise ValueError('Review axes mismatch')
    if 'time' in axes:
        selected=[pd.Timestamp(t) for row in selection['block_support']['blocks'] for t in row['times']]
        held=[pd.Timestamp(t) for row in confirmation['block_support']['blocks'] for t in row['times']]
        if not selected or not held or min(held)<=max(selected):
            raise ValueError('Confirmation must be disjoint and strictly after selection times')
    if 'space' in axes:
        a=set(selection['block_support']['locations']);b=set(confirmation['block_support']['locations'])
        if not a or not b or a&b:raise ValueError('Confirmation locations overlap selection locations')
    if not quality_pass:status='QUALITY_FAILED'
    elif not first['sufficient'] or not second['sufficient']:status='EVIDENCE_INSUFFICIENT'
    elif not first['stable_gain']:status='NO_STABLE_SELECTION_GAIN'
    elif not second['stable_gain']:status='CONFIRMATION_FAILED'
    else:status='RECOMMENDED_WITHIN_OBSERVED_WINDOWS'
    confirmation_rule='Disjoint geographic holdout, once after freezing' if axes==['space'] else 'Disjoint locations and later dates, once after freezing' if set(axes)=={'space','time'} else LEGACY_POLICY['confirmation']
    return dict(version=LEGACY_VERSION,status=status,recommended=status=='RECOMMENDED_WITHIN_OBSERVED_WINDOWS',
        selection=first,confirmation=second,confirmation_used_for_search=False,
        confirmation_retries=0,policy=dict(LEGACY_POLICY,confirmation=confirmation_rule),
        warning='These windows are correlated; this is not a calibrated probability of future success')


def compare(candidate,baseline,*,risk_limits=None):
    """Same fixed queues/objective; separate aggregate, uncertainty and local harm."""
    result=compare_legacy(candidate,baseline)
    rows=result['differences'];tol=result['roundoff_only_tolerance']
    groups={}
    support=candidate['block_support'];blocks=support.get('blocks',())
    axes=support.get('axes',['time'])
    for index,row in enumerate(rows):
        # All regions on the same date move together. Never resample pixels.
        block=next((b for b in blocks if b.get('id')==row['block']),None)
        dates=(block or {}).get('times',())
        unit=(block.get('resampling_unit') if block and block.get('resampling_unit') is not None else
              str(dates[0]) if 'time' in axes and 'space' in axes and len(dates)==1 else row['block'])
        groups.setdefault(unit,[]).append(row['rmse_change'])
    sums=np.array([sum(v) for v in groups.values()],float)
    counts=np.array([len(v) for v in groups.values()],float)
    uncertainty=dict(status='INSUFFICIENT',units=len(sums),resampling_unit='date clusters' if set(axes)=={'space','time'} else 'spatial regions' if axes==['space'] else 'evaluation blocks',
        interpretation='Conditional block-resampling interval; block independence and future coverage are not established',
        **POLICY['resampling'])
    if len(sums)>=3:
        from scipy.stats import bootstrap
        sampled=bootstrap((sums,counts),lambda a,b,axis:np.sum(a,axis=axis)/np.sum(b,axis=axis),
            paired=True,vectorized=True,method='percentile',confidence_level=POLICY['resampling']['confidence_level'],
            n_resamples=POLICY['resampling']['resamples'],rng=np.random.default_rng(POLICY['resampling']['seed']))
        low,high=map(float,sampled.confidence_interval)
        uncertainty.update(status='CONDITIONAL_INTERVAL',lower=low,upper=high,
            gain_supported=bool(high < -tol),
            leave_one_unit_out_range=[float(np.min((sums.sum()-sums)/(counts.sum()-counts))),
                                      float(np.max((sums.sum()-sums)/(counts.sum()-counts)))])
    degraded=[r for r in rows if r['rmse_change']>tol or r['mae_change']>tol]
    local=dict(status='NO_OBSERVED_DEGRADATION' if not degraded else 'UNASSESSED_TOLERANCE',
        degraded_blocks=len(degraded),total_blocks=len(rows),
        degraded_fraction=len(degraded)/len(rows) if rows else None,
        worst_rmse_increase=max([r['rmse_change'] for r in rows],default=None),
        worst_mae_increase=max([r['mae_change'] for r in rows],default=None),
        violations=[],basis='Observed changes only; no task harm tolerance inferred')
    if risk_limits is not None:
        unit=candidate.get('evaluation_definition',{}).get('unit')
        if (not risk_limits.get('basis') or unit in {None,'UNKNOWN'} or risk_limits.get('unit')!=unit):
            raise ValueError('Local risk limits require matching physical unit and auditable basis')
        allowed={'max_rmse_increase':'rmse_change','max_mae_increase':'mae_change'}
        limits={key:float(risk_limits[key]) for key in allowed if key in risk_limits}
        if not limits or any(not np.isfinite(v) or v<0 for v in limits.values()):
            raise ValueError('Explicit nonnegative local risk bound required')
        violations=[dict(block=r['block'],limit=key,actual=r[allowed[key]],allowed=value)
                    for r in rows for key,value in limits.items() if r[allowed[key]]>value+tol]
        local.update(status='LIMIT_EXCEEDED' if violations else 'WITHIN_DECLARED_LIMITS',
            violations=violations,limits=limits,unit=unit,basis=risk_limits['basis'])
    result.update(aggregate_gain=bool(rows and result['objective_change'] < -tol),uncertainty=uncertainty,local_risk=local)
    result['stable_gain']=bool(result['aggregate_gain'] and uncertainty.get('gain_supported'))
    return result


def range_comparison(candidate, baseline, *, phase):
    """Descriptive paired ranges, never a post-hoc selection of winning blocks."""
    compare_legacy(candidate, baseline)  # Same queue, references and denominators.
    rows=[]
    for a,b in zip(candidate['block_metrics'],baseline['block_metrics']):
        if a['metrics'] is None:
            rows.append(dict(block=a['fold_id'],status='NO_REFERENCE'));continue
        left,right=a['metrics'],b['metrics']
        delta=block_error_change(left, right)['rmse_change']
        tolerance=64*np.finfo(float).eps*max(1.,abs(left['RMSE']),abs(right['RMSE']))
        rows.append(dict(block=a['fold_id'],status='EVALUATED',phase=phase,
            candidate=left,baseline=right,RMSE_change=delta,
            lower_error='candidate' if delta < -tolerance else 'baseline' if delta > tolerance else 'equivalent'))
    return dict(phase=phase,blocks=rows,used_for_selection=phase=='selection',
        measured_blocks=sum(r['status']=='EVALUATED' for r in rows),
        candidate_better=sum(r.get('lower_error')=='candidate' for r in rows),
        baseline_better=sum(r.get('lower_error')=='baseline' for r in rows),
        equivalent=sum(r.get('lower_error')=='equivalent' for r in rows),
        limitation='Observed ranges only; unobserved dates, regions and unresolved fine scales remain UNKNOWN')


def _trial_summaries(blocks, unit, sufficient):
    support = dict(fingerprint='shared_local_trial_queue', sufficient=sufficient and all(
        row['candidate'] is not None for row in blocks), axes=['time'],
        blocks=[dict(id=row['block']) for row in blocks])
    definition = dict(unit=unit)
    def summary(side):
        rows = [dict(fold_id=row['block'], status='PASS' if row[side] is not None else 'NO_REFERENCE',
                     metrics=row[side]) for row in blocks]
        measured = [row['metrics'] for row in rows if row['metrics'] is not None]
        return dict(block_support=support, evaluation_fingerprint='shared_local_trial_reference',
                    evaluation_definition=definition, denominator=sum(row['count'] for row in measured),
                    objective=float(np.mean([row['RMSE'] for row in measured])) if measured else 0.,
                    block_metrics=rows)
    return summary('candidate'), summary('baseline')


def trial_range(blocks, *, phase):
    """Shared descriptive range without treating blocks as independent draws."""
    candidate, baseline = _trial_summaries(blocks, None, False)
    return range_comparison(candidate, baseline, phase=phase)


def trial_comparison(blocks, *, phase, unit=None, risk_limits=None, sufficient=False):
    """Compare a paired queue already scored on the same reference."""
    if not blocks:
        return dict(status='NO_REFERENCE', phase=phase)
    candidate, baseline = _trial_summaries(blocks, unit, sufficient)
    return dict(version=VERSION, status='EVALUATED' if candidate['denominator'] else 'NO_REFERENCE',
                comparison=compare(candidate, baseline, risk_limits=risk_limits),
                range_comparison=range_comparison(candidate, baseline, phase=phase))


def review(selection,selection_baseline,confirmation,confirmation_baseline,*,quality_pass,
           risk_limits=None,policy_version=None):
    if policy_version==LEGACY_VERSION:
        return review_legacy(selection,selection_baseline,confirmation,confirmation_baseline,quality_pass=quality_pass)
    if policy_version not in {None,VERSION}:raise ValueError('Unknown review policy version')
    # Reuse every existing queue, semantic and disjointness check.
    original=review_legacy(selection,selection_baseline,confirmation,confirmation_baseline,quality_pass=quality_pass)
    first=compare(selection,selection_baseline,risk_limits=risk_limits)
    second=compare(confirmation,confirmation_baseline,risk_limits=risk_limits)
    if not quality_pass:status='QUALITY_FAILED'
    elif not first['aggregate_gain']:status='NO_STABLE_SELECTION_GAIN'
    elif not second['aggregate_gain']:status='CONFIRMATION_FAILED'
    elif any(r['local_risk']['status']=='LIMIT_EXCEEDED' for r in (first,second)):status='LOCAL_RISK_FAILED'
    elif (not first['sufficient'] or not second['sufficient']
          or not first['stable_gain'] or not second['stable_gain']
          or any(r['local_risk']['status']=='UNASSESSED_TOLERANCE' for r in (first,second))):status='TRIAL_AVAILABLE'
    else:status='RECOMMENDED_WITHIN_OBSERVED_WINDOWS'
    recommended=status=='RECOMMENDED_WITHIN_OBSERVED_WINDOWS'
    return dict(original,version=VERSION,status=status,recommended=recommended,
        decision='RECOMMEND' if recommended else 'TRIAL' if status=='TRIAL_AVAILABLE' else 'KEEP_BASELINE',
        candidate_trial_available=status=='TRIAL_AVAILABLE',
        selection=first,confirmation=second,
        range_comparison=dict(selection=range_comparison(selection,selection_baseline,phase='selection'),
            confirmation=range_comparison(confirmation,confirmation_baseline,phase='confirmation')),
        evidence_status='SUFFICIENT_ENGINEERING_SUPPORT' if first['sufficient'] and second['sufficient'] else 'INSUFFICIENT',
        policy=dict(POLICY,confirmation=original['policy']['confirmation']),
        warning='Paired block intervals are conditional diagnostics; not calibrated future guarantees or independent pixel replicates')


def apply_recommendation_isolation(gate,isolation):
    """A missing reserve affects certainty, never rewrites measured deterioration."""
    if isolation and isolation.get('status')!='RESERVED':
        gate.update(recommended=False,recommendation_isolation=isolation,evidence_status='INSUFFICIENT')
        if gate.get('status') in {'RECOMMENDED_WITHIN_OBSERVED_WINDOWS','PRODUCT_RECONSTRUCTION_CONFIRMED','TRIAL_AVAILABLE'}:
            gate.update(status='TRIAL_AVAILABLE',decision='TRIAL',candidate_trial_available=True)
        gate['isolation_reason']='No separate recommendation/mutation reserve; only development evidence'
    return gate


def unavailable(reason,*,basis='UNKNOWN',axes=()):
    """All task families use an explicit unassessed state, never a success score."""
    return dict(version=VERSION,status='EVIDENCE_INSUFFICIENT',recommended=False,
        accuracy_recommended=False,reference_basis=basis,axes=list(axes),reason=reason,
        confirmation_used_for_search=False,confirmation_retries=0)


def spatial_metrics(report,*,reference_key,variable,unit,locations,times,axes,phase):
    """Normalize existing real date/region metrics; no new prediction or split."""
    blocks=report['blocks'];points=sum(b['metrics']['count'] for b in blocks)
    if not points:raise ValueError('No real block samples')
    rmse=float(np.sqrt(sum(b['metrics']['RMSE']**2*b['metrics']['count'] for b in blocks)/points))
    rows=[dict(fold_id=str(b['date'])+'|'+str(b['region']),status='PASS',metrics=dict(b['metrics'])) for b in blocks]
    regions=len({b['region'] for b in blocks});dates=len({b['date'] for b in blocks})
    support=dict(axes=list(axes),locations=sorted(map(str,locations)),
        blocks=[dict(id=r['fold_id'],times=[str(b['date'])],
            resampling_unit=str(b['date']) if 'time' in axes else str(b['region'])) for b,r in zip(blocks,rows)],
        sufficient=regions>=(2 if phase=='selection' else 1) and len(locations)>=2 and
            ('time' not in axes or dates>=3),
        support_note='Spatial selection needs two occupied regions; outer assessment a distinct region and two locations. Time transfer needs three dates per stage; no independence assertion')
    payload=dict(reference=reference_key,locations=support['locations'],times=[str(t) for t in times],variable=variable,unit=unit)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    support['fingerprint']=digest
    return dict(objective=float(np.mean([r['metrics']['RMSE'] for r in rows])),RMSE=rmse,
        MAE=float(sum(b['metrics']['MAE']*b['metrics']['count'] for b in blocks)/points),
        denominator=points,block_metrics=rows,block_support=support,evaluation_fingerprint=digest,
        evaluation_definition=dict(variable=variable,unit=unit))


def review_spatial(search,comparison,assessment,best):
    """Same gate for station, station/grid, multi-grid and auxiliary branches."""
    basis=search.evaluation_plan['mode'];axes=search.review_axes
    baseline=comparison['families']['SINGLE_SOURCE']['best_key']
    schemes=assessment.get('schemes',{})
    if not baseline or not all(schemes.get(k,{}).get('status')=='PASS' for k in ['selected','single_source_baseline']):
        return unavailable('No complete frozen candidate/comparator on separate held blocks',basis=basis,axes=axes)
    source=next(r for r in search.rows if r['key']==baseline)
    outer=assessment.get('review_support')
    if outer is None:return unavailable('Outer block coordinates or times unavailable',basis=basis,axes=axes)
    if search.reconstruction is not None:
        r=search.reconstruction;ids=list(map(str,r['selected']));support=search.evaluation_plan['fixed_support_sha256']
    else:
        ids=list(map(str,search.location_protocol['selection_ids']))
        support='|'.join(f['support_sha256'] for f in search.folds)
    kwargs=dict(variable=search.request['variable'],unit=search.request['unit'],axes=axes)
    a=spatial_metrics(best['blocked_validation'],reference_key=support,locations=ids,times=search.search_times,phase='selection',**kwargs)
    b=spatial_metrics(source['blocked_validation'],reference_key=support,locations=ids,times=search.search_times,phase='selection',**kwargs)
    try:
        c=spatial_metrics(schemes['selected']['blocked_validation'],reference_key=outer['reference_key'],
            locations=outer['locations'],times=outer['times'],phase='confirmation',**kwargs)
        d=spatial_metrics(schemes['single_source_baseline']['blocked_validation'],reference_key=outer['reference_key'],
            locations=outer['locations'],times=outer['times'],phase='confirmation',**kwargs)
        result=review(a,b,c,d,quality_pass=True,risk_limits=search.request.get("review_risk_limits"))
    except (KeyError,ValueError) as exc:
        return unavailable(str(exc),basis=basis,axes=axes)
    proxy=basis!='OBSERVATION_HOLDOUT'
    result.update(reference_basis=basis,axes=axes,accuracy_recommended=result['recommended'] and not proxy,
        claim='Product reconstruction only; not environmental accuracy' if proxy else 'Held observations; upstream statistical independence still unknown',
        baseline_key=baseline,selection_key=best['key'])
    if proxy and result['recommended']:result['status']='PRODUCT_RECONSTRUCTION_CONFIRMED'
    return result


def fold_metrics(responses,folds,*,variable,unit,axes,phase):
    """Adapt real task-matched execution feedback to the same review protocol."""
    from ecodata.algorithms.reliable_selection import aggregate
    locations=set();times=set();references=[];rows=[]
    by_id={r['fold_id']:r for r in responses}
    for fold in folds:
        ref=fold['reference'];valid=np.isfinite(ref.values)
        for t,y,x in np.argwhere(valid):
            times.add(str(pd.Timestamp(ref.time.values[t])))
            locations.add(str((float(ref.lat.values[y]),float(ref.lon.values[x]))))
        references.append(hashlib.sha256(ref.values.tobytes()).hexdigest())
        response=by_id[fold['id']]
        if response['status']!='PASS':raise ValueError('Required measured review case failed')
        metrics=dict(response['metrics']);metrics['denominator']=metrics.get('denominator',metrics.get('count'))
        rows.append(dict(fold_id=fold['id'],status='PASS',metrics=metrics))
    policy=dict(weights={'RMSE':1.},scales={'RMSE':1.},weight_basis='Equal required task cases',
        scale_basis={'RMSE':'Physical output unit'},data_portrait={})
    metrics=aggregate(rows,policy,[f['id'] for f in folds])
    payload=dict(references=references,locations=sorted(locations),times=sorted(times),unit=unit,variable=variable)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    sufficient=len(rows)>=(2 if phase=='selection' else 1) and ('space' not in axes or len(locations)>=2) and ('time' not in axes or len(times)>=2)
    return dict(metrics,denominator=metrics['valid_n'],block_metrics=rows,evaluation_fingerprint=digest,
        evaluation_definition=dict(variable=variable,unit=unit),block_support=dict(fingerprint=digest,axes=axes,
            locations=sorted(locations),blocks=[dict(times=sorted(times))],sufficient=sufficient))


def review_field(search,result,outer,receipt,request):
    """Shared field/multisource recovery gate; no new graph proposal strategy."""
    basis='VISIBLE_PRODUCT_RECONSTRUCTION_PROXY';axes=receipt.get('axes',['time'])
    if outer is None:return unavailable(receipt.get('reason','No isolated matched confirmation'),basis=basis,axes=axes)
    from dataclasses import replace
    from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
    from ecodata.algorithms.platform_search import platform_registry
    from ecodata.algorithms.autonomous_decision import Executor
    best=result.get('best');baseline=result.get('initial_best')
    if not best or not baseline:return unavailable('No frozen complete comparator',basis=basis,axes=axes)
    ctx=build_context(outer['inputs'],outer.get('request',request))[0]
    registry=field_registry(platform_registry(),ctx)
    executor=Executor(replace(ctx,validation_reference=None,validation_role=None),registry,account=search.account,phase='validation_audit')
    reference=outer['validation'].reference
    outer_fold=[dict(id=outer['id'],reference=reference)]
    predictions={};snapshots={}
    try:
        for name,row in [('candidate',best),('baseline',baseline)]:
            snapshots[name]=fold_metrics(row['fold_responses'],search.folds,variable=ctx.variable,unit=request['unit'],axes=axes,phase='selection')
            if row['key'] not in predictions:
                data,trace=executor.execute(search.records,row['key'])
                predictions[row['key']]=dict(fold_id=outer['id'],status='PASS',metrics=measured(data,reference,ctx),trace=trace)
            snapshots[name+'_confirmation']=fold_metrics([predictions[row['key']]],outer_fold,
                variable=ctx.variable,unit=request['unit'],axes=axes,phase='confirmation')
        result_review=review(snapshots['candidate'],snapshots['baseline'],snapshots['candidate_confirmation'],snapshots['baseline_confirmation'],quality_pass=result.get('status')=='PASS',risk_limits=request.get('review_risk_limits'))
        if result.get('status')!='PASS':result_review.update(status='SEARCH_INCOMPLETE',recommended=False,search_status=result.get('status'))
    except (ValueError,RuntimeError,KeyError) as exc:return unavailable(str(exc),basis=basis,axes=axes)
    result_review.update(reference_basis=basis,axes=axes,accuracy_recommended=False,
        claim='Task-matched product/visible-value reconstruction; no independent environmental accuracy claim',
        baseline_key=baseline['key'],selection_key=best['key'],evidence=snapshots,execution=list(predictions.values()))
    if result_review['recommended']:result_review['status']='PRODUCT_RECONSTRUCTION_CONFIRMED'
    return result_review


def scope_report(result, folder):
    """Expose every measured date/region, including negative results, locally."""
    from pathlib import Path
    path=Path(folder)/'candidate_results.json'
    if not path.is_file():return dict(status='NO_BLOCK_RECORDS',accuracy_claim=False)
    values=json.loads(path.read_text(encoding='utf8'))
    rows=list(values.values()) if isinstance(values,dict) else values
    by_key={r['key']:r for r in rows if r.get('status') in {'PASS','ELIGIBLE_OUTPUT'}}
    baseline=(result.get('selection_comparison',{}).get('families',{}).get('SINGLE_SOURCE',{}).get('best_key')
              or (result.get('initial_best') or {}).get('key'))
    candidates=[('selected',(result.get('best') or {}).get('key'))]
    candidates += [('ds_recommendation_'+str(i),key) for i,r in enumerate(result.get('recommendation_validation',[]))
                   if r.get('status')=='PASS' for key in r.get('keys',[])][:2]
    def blocks(row):
        raw=row.get('blocked_validation',{}).get('blocks',[])
        return {str(b['date'])+'|'+str(b['region']):b['metrics'] for b in raw}
    comparisons=[];assessment=result.get('post_selection_assessment',{}).get('schemes',{})
    for name,key in candidates:
        if key not in by_key or baseline not in by_key:continue
        for phase,a,b in [('selection',by_key[key],by_key[baseline]),
                ('confirmation',assessment.get(name,{}),assessment.get('single_source_baseline',{}))]:
            left,right=blocks(a),blocks(b)
            if not left or left.keys()!=right.keys():
                comparisons.append(dict(candidate=name,key=key,phase=phase,status='NOT_COMPARABLE_OR_NOT_EVALUATED'));continue
            ranges=[]
            for block in sorted(left):
                aa,bb=left[block],right[block]
                if aa['count']!=bb['count']:raise ValueError('Scope report cannot compare different sample denominators')
                delta=aa['RMSE']-bb['RMSE'];tol=64*np.finfo(float).eps*max(1.,aa['RMSE'],bb['RMSE'])
                ranges.append(dict(block=block,candidate=aa,baseline=bb,RMSE_change=delta,
                    lower_error='candidate' if delta < -tol else 'baseline' if delta > tol else 'equivalent'))
            comparisons.append(dict(candidate=name,key=key,phase=phase,status='EVALUATED',blocks=ranges,
                candidate_better=sum(r['lower_error']=='candidate' for r in ranges),
                baseline_better=sum(r['lower_error']=='baseline' for r in ranges),
                equivalent=sum(r['lower_error']=='equivalent' for r in ranges)))
    return dict(version='observed_scope_v1',baseline_key=baseline,comparisons=comparisons,
        date_partition=result.get('evaluation_protocol',{}).get('date_partition'),accuracy_claim=False,
        limitation='Observed reference-supported ranges only; no universal or 95% full-population guarantee',
        retained_products=result.get('retained_products',[]))


def finalize(result,request,folder,*,inputs=(),validation=None,evaluation=None):
    """No successful public fusion/search response silently omits review status."""
    from pathlib import Path
    from ecodata.algorithms.numerical_search import save
    from ecodata.algorithms.platform_search import BlockedSearchValidation,BlockedReference,parse_inputs
    gate=result.get('recommendation_review')
    if request.get('review_policy')=='legacy_v1':
        gate=unavailable('Explicit archived evaluation; not accepted as an independent recommendation')
        gate['status']='LEGACY_NOT_REVIEWED'
    if gate is None and result.get('stations'):
        statuses=[r.get('recommendation_review',unavailable('Station review unavailable')) for r in result['stations']]
        gate=dict(version=VERSION,status='ALL_CONFIRMED' if all(r['recommended'] for r in statuses) else 'UNCONFIRMED_SUBTASKS',
            recommended=bool(statuses) and all(r['recommended'] for r in statuses),subtasks=statuses,
            accuracy_recommended=bool(statuses) and all(r.get('accuracy_recommended',r['recommended']) for r in statuses))
        deliveries={r.get('delivery_status','EXPLORATORY_ONLY') for r in result['stations']}
        if len(deliveries)==1:result.setdefault('delivery_status',deliveries.pop())
        else:result.setdefault('delivery_status','TASK_SPECIFIC_DELIVERY')
    if gate is None and isinstance(validation,BlockedSearchValidation) and isinstance(evaluation,BlockedReference) and result.get('product'):
        try:
            import xarray as xr
            sources=[ds for role,ds in parse_inputs(inputs)[0] if role in {'primary','background'}]
            freeze=json.loads((Path(folder)/'selection_freeze.json').read_text(encoding='utf8'))
            baseline_key=freeze.get('source_baseline_key')
            if baseline_key is None:raise ValueError('No compatible complete source comparator was frozen')
            index=int(baseline_key.removeprefix('s'));base=sources[index]
            with xr.open_dataset(result['product']) as ds:output=ds.load()
            if not base.coords.equals(output.coords):raise ValueError('Baseline needs the same complete target support, not a final interpolation')
            source=next((r for r in result.get('source_baselines',[]) if r['key']==baseline_key and r['status']=='ELIGIBLE_OUTPUT'),None)
            if source is None:raise ValueError('No quality-accepted public baseline')
            # Candidate confirmation was evaluated once by the backend, after freeze.
            gate=review(result['best']['metrics'],validation(base),result['independent_evaluation'],evaluation(base),quality_pass=result.get('status')=='PASS',risk_limits=request.get('review_risk_limits'))
            if result.get('status')!='PASS':gate.update(status='SEARCH_INCOMPLETE',recommended=False,search_status=result.get('status'))
            gate.update(reference_basis='EXPLICIT_REFERENCE_ERROR',axes=validation.blocks.get('axes',['time']),
                accuracy_recommended=False,claim='Explicit reference; independent instrument/provenance not inferred')
            if not gate['recommended']:
                candidate=Path(folder)/'exploratory-candidate.nc';candidate.write_bytes(Path(result['product']).read_bytes())
                path=Path(folder)/'review-baseline.nc';base.to_netcdf(path,engine='h5netcdf')
                with xr.open_dataset(path) as check:xr.testing.assert_allclose(base,check.load())
                result.update(candidate_product=str(candidate),product=str(path),delivered_graph=dict(key=baseline_key,records={baseline_key:dict(operation='source',parents=[],parameters=dict(source='source'+str(index)))}),delivered_best=source,delivery_status='BASELINE_RETAINED')
        except (ValueError,KeyError) as exc:gate=unavailable(str(exc),basis='EXPLICIT_REFERENCE_ERROR')
    if gate is None:
        engineering=getattr(evaluation,'protocol',{}).get('kind')=='declared_task_metric'
        gate=unavailable('No isolated task-matched reference and comparator; numerical output is not a recommendation',
            basis='DECLARED_ENGINEERING_METRIC' if engineering else 'UNKNOWN')
        if engineering:gate.update(status='ENGINEERING_ONLY',metrics=result.get('independent_evaluation'),
            claim='Actual declared engineering metric only; no environmental accuracy claim')
    result['recommendation_review']=gate
    result['review_service']='ecodata.evaluation.blocked_confirmation'
    result['review_service_version']='unified_fusion_review_v1'
    result.setdefault('delivery_status','ENHANCEMENT_RECOMMENDED' if gate['recommended'] else 'EXPLORATORY_ONLY')
    if request.get('recommendation_isolation'):
        result['recommendation_isolation']=request['recommendation_isolation']
    result['post_mutation_acceptance']=mutation_acceptance(result,folder)
    result['scope_report']=scope_report(result,folder)
    save(Path(folder)/'scope_report.json',result['scope_report'])
    save(Path(folder)/'post_mutation_acceptance.json',result['post_mutation_acceptance'])
    save(Path(folder)/'recommendation_review.json',gate)
    save(Path(folder)/'summary.json',result)
    return result


def engineering_review(ds,variable):
    """Actual deterministic-fusion engineering checks, never invented truth."""
    values=np.asarray(ds[variable].values);finite=np.isfinite(values)
    uncertainty=np.asarray(ds.uncertainty.values) if 'uncertainty' in ds else None
    uncertainty_ok=uncertainty is not None and bool(np.isfinite(uncertainty[finite]).all() and (uncertainty[finite]>=0).all())
    return dict(version=VERSION,status='ENGINEERING_ONLY' if finite.any() and uncertainty_ok else 'ENGINEERING_FAILED',
        recommended=False,accuracy_recommended=False,reference_basis='DECLARED_UNCERTAINTY_AND_COVERAGE',
        engineering_passed=bool(finite.any() and uncertainty_ok),metrics=dict(coverage=float(finite.mean()),
        denominator=int(values.size),finite_predictions=int(finite.sum()),uncertainty_nonnegative=uncertainty_ok),
        reason='No independent reference; actual coverage/uncertainty checks only, not error reduction',
        confirmation_used_for_search=False,confirmation_retries=0)


def public_summary(gate):
    """Aggregate review facts only; no locations, dates, labels or state paths."""
    if not gate:return None
    result={k:gate[k] for k in ('version','status','decision','candidate_trial_available','evidence_status','recommended','accuracy_recommended','reference_basis','axes','claim','reason') if k in gate}
    for phase in ('selection','confirmation'):
        if isinstance(gate.get(phase),dict):
            result[phase]={k:gate[phase][k] for k in ('sufficient','stable_gain','aggregate_gain','objective_change','uncertainty','local_risk') if k in gate[phase]}
            if 'local_risk' in result[phase]:
                result[phase]['local_risk']={k:v for k,v in result[phase]['local_risk'].items() if k!='violations'}
    return result


def mutation_acceptance(result,folder):
    """Expose the existing frozen-graph review, never a new scoring rule.

    A recommendation trial cannot be inherited by a changed graph. Missing
    outer numerical evidence remains unverified, even with a valid product.
    """
    from pathlib import Path
    gate=result.get('recommendation_review') or {}
    root=Path(folder)
    freeze=root/'selection_freeze.json'
    best=result.get('best') or {}
    delivered=result.get('delivered_best') or best
    graph=result.get('delivered_graph') or result.get('selected_graph') or {}
    frozen=None
    if freeze.is_file():
        frozen=json.loads(freeze.read_text(encoding='utf8'))
    reviewed_key=gate.get('selection_key')
    identity_ok=bool(frozen and frozen.get('key')==best.get('key') and
                     (reviewed_key is None or reviewed_key==best.get('key')))
    subtasks=[row.get('post_mutation_acceptance') for row in result.get('stations',())]
    if subtasks:
        identity_ok=all(row and row.get('frozen_graph_verified') for row in subtasks)
    isolation=result.get('recommendation_isolation')
    if isolation is not None and isolation.get('status')!='RESERVED':identity_ok=False
    confirmed=bool(identity_ok and gate.get('recommended') and not gate.get('confirmation_used_for_search',False))
    return dict(version='frozen_mutation_acceptance_v1',
        status='CONFIRMED_WITHIN_DECLARED_REFERENCE' if confirmed else 'BASELINE_RETAINED' if result.get('delivery_status')=='BASELINE_RETAINED' else 'UNCONFIRMED',
        recommended=confirmed,frozen_graph_verified=identity_ok,
        selection_key=best.get('key'),delivered_key=delivered.get('key') or graph.get('key'),
        final_review_status=gate.get('status','NOT_EVALUATED'),
        freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest() if frozen else None,
        numerical_review=gate,recommendation_isolation=isolation,confirmation_used_for_search=False,confirmation_retries=0,
        probability_guarantee=None,
        independence_scope='Declared search/confirmation partition only; recommendation exposure and upstream independence require separate evidence',
        reason='Existing one-shot numerical review after graph freeze; a pre-mutation trial is not acceptance of the selected graph')
