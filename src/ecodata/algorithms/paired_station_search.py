"""Point/time adapter into the existing public field mutation service.

No optimizer, preference rules or scientific calibration kernels live here.
Observations after the fitting period enter independent evaluators only.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.numerical_search import save

VERSION='paired_station_mutation_v2_blocked_confirmation'
DEFAULT_BUDGET=dict(objective_limit=80,total_call_limit=120,proposal_limit=256,wall_seconds=30)


def correction_coefficients(training):
    offset = float((training.observed - training.background).mean())
    denominator = float(training.background.sum())
    ratio = float(training.observed.sum()) / denominator if denominator else np.nan
    return offset, ratio


def prepare(frame,variable,unit,*,temporal_semantics='instantaneous',unlocated_series=False):
    required={'station','grid_time','observed','background','unit'}
    if not unlocated_series:required.update({'longitude','latitude'})
    if not required<=set(frame):raise ValueError('Missing paired observation columns: '+str(sorted(required-set(frame))))
    if not unit or unit=='UNKNOWN' or not frame.unit.eq(unit).all():raise ValueError('Paired units unknown or inconsistent')
    if frame.station.nunique()!=1:raise ValueError('Prepare one actual station at a time')
    if frame.duplicated(['station','grid_time']).any():raise ValueError('Duplicate paired station/time')
    frame=frame.sort_values('grid_time').reset_index(drop=True).copy()
    clock=pd.DatetimeIndex(frame.grid_time)
    if clock.hasnans or len(clock)<9:raise ValueError('At least nine actual times required; no invented validation data')
    numeric=['observed','background'] + ([] if unlocated_series else ['longitude','latitude'])
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError('Incomplete fixed paired cohort')
    if not unlocated_series and (frame.longitude.nunique()!=1 or frame.latitude.nunique()!=1):raise ValueError('Moving station needs a trajectory contract')
    if temporal_semantics not in {'instantaneous','mean','accumulation'}:
        raise ValueError('Explicit supported paired temporal semantics required')
    if 'temporal_semantics' in frame and not frame.temporal_semantics.eq(temporal_semantics).all():
        raise ValueError('Paired temporal semantics conflict')
    if temporal_semantics!='instantaneous':
        if not {'interval_start','interval_end'}<=set(frame):raise ValueError('Paired interval values require explicit interval_start and interval_end')
        for name in ['interval_start','interval_end']:frame[name]=pd.to_datetime(frame[name],errors='raise')
        starts=pd.DatetimeIndex(frame.interval_start);ends=pd.DatetimeIndex(frame.interval_end)
        widths=ends-starts
        if (starts.hasnans or ends.hasnans or (widths<=pd.Timedelta(0)).any() or len(set(widths))!=1
                or (starts[1:]<ends[:-1]).any() or not (clock.equals(starts) or clock.equals(ends))):
            raise ValueError('Paired intervals must be equal positive non-overlapping windows consistently labelled at start or end')
    cut=len(clock)//2;end=cut+len(clock)//4
    train=frame.iloc[:cut];validation=frame.iloc[cut:end];test=frame.iloc[end:]
    # Source identity excludes validation/test labels, even their hashes.
    identity_columns=['station','grid_time','background'] + ([] if unlocated_series else ['longitude','latitude'])
    identity=hashlib.sha256(frame[identity_columns].to_csv(index=False).encode()).hexdigest()
    if temporal_semantics!='instantaneous':
        identity=hashlib.sha256((identity+temporal_semantics+frame[['interval_start','interval_end']].to_csv(index=False)).encode()).hexdigest()
    source_id='paired-background:'+identity
    coords={} if unlocated_series else dict(lat=[float(frame.latitude.iloc[0])],lon=[float(frame.longitude.iloc[0])])
    attrs=dict(units=unit,physical_variable=variable,physical_definition=variable,
        data_kind='continuous',spatial_semantics='point',temporal_semantics=temporal_semantics)
    def field(part,column):
        dims=('time',) if unlocated_series else ('time','lat','lon')
        values=part[column].to_numpy(float) if unlocated_series else part[column].to_numpy(float)[:,None,None]
        ds=xr.Dataset({variable:(dims,values)},
            coords=dict(time=part.grid_time.to_numpy(),**coords),attrs=dict(crs='EPSG:4326',source_id=source_id,
            support_evidence='Actual station positions; original model sampled by the upstream registered collocation adapter'))
        if unlocated_series:
            ds.attrs.pop('crs',None)
            ds.attrs['support_evidence']='Explicit unlocated series; no geographical coordinates inferred'
        ds[variable].attrs=attrs.copy()
        if temporal_semantics!='instantaneous':
            ds['time_bounds']=(('time','bounds'),np.column_stack([part.interval_start.to_numpy(dtype='datetime64[ns]'),part.interval_end.to_numpy(dtype='datetime64[ns]')]))
        return ds
    application=field(frame.iloc[cut:],'background')
    fit_start=perf_counter()
    # Same existing fixed mean-residual correction expressed as affine state
    # with slope 1; the Registry's prior calibration kernel applies it.
    intercept,ratio=correction_coefficients(train)
    state=xr.Dataset(dict(affine_slope=xr.DataArray(1.),affine_intercept=xr.DataArray(intercept)),
        attrs=dict(calibration_source_id=source_id,training_end=str(clock[cut-1]),
            fitting='Mean training residual; affine slope fixed to 1, not a new regression algorithm',
            station=str(frame.station.iloc[0]),unit=unit))
    # Existing ratio alternative is only available where its training statistic
    # is identifiable. Missing monthly coefficients are never fabricated.
    if np.isfinite(ratio):state['mean_ratio']=xr.DataArray(ratio)
    protocol=dict(version=VERSION,unlocated_series=unlocated_series,temporal_semantics=temporal_semantics,intervals_verified=temporal_semantics!='instantaneous',station=str(frame.station.iloc[0]),unit=unit,
        source_id=source_id,train_times=[str(t) for t in clock[:cut]],
        validation_times=[str(t) for t in clock[cut:end]],test_times=[str(t) for t in clock[end:]],
        fitted_state_sha256=hashlib.sha256(state.to_netcdf()).hexdigest(),fit_seconds=perf_counter()-fit_start,
        validation_labels_in_predictor=False,test_labels_in_predictor=False,
        scope='One-location chronological calibration; not leave-location-out or a new blind test')
    return application,state,field(validation,'observed')[variable],field(test,'observed')[variable],protocol


def submit(inputs,request,folder,*,budget=None,seed=11,progress=None):
    from ecodata.algorithms.platform_search import parse_inputs,SearchInput,SearchValidation,DatasetReference,submit_search,BlockedSearchValidation,BlockedReference
    from ecodata.io.structures import table_frame
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'paired_summary.json').exists():raise ValueError('Immutable paired station request already exists')
    paired=[i for i in inputs if i.role=='paired_observations'];conditioning=[i for i in inputs if i.role=='observations']
    if len(paired)!=1 or len(conditioning)>1 or len(paired)+len(conditioning)!=len(inputs):raise ValueError('One paired target and at most one authorized observations resource required')
    update=request.get('observation_update')
    if bool(conditioning)!=bool(update):raise ValueError('Conditioning resource and explicit observation_update permission must be supplied together')
    parsed,audit=parse_inputs(inputs);frame=table_frame(next(d for r,d in parsed if r=='paired_observations'))
    policy=request.get('evaluation_protocol','blocked_confirmation_v1')
    if policy not in {'blocked_confirmation_v1','legacy_single_v1'}:raise ValueError('Unknown paired evaluation protocol')
    frame['grid_time']=pd.to_datetime(frame.grid_time,errors='raise')
    stations=sorted(frame.station.astype(str).unique())
    if conditioning:
        allowed=table_frame(next(d for r,d in parsed if r=='observations'))
        if 'station_id' in allowed and allowed.station_id.astype(str).isin(stations).any():
            raise ValueError('Conditioning resource contains an evaluated target station; batch resources must exclude every target')
    budget=dict(DEFAULT_BUDGET,**(budget or {}))
    results=[];started=perf_counter()
    for index,station in enumerate(stations):
        remaining=budget['wall_seconds']-(perf_counter()-started)
        local_budget={k:max(1,int(v)//len(stations)) for k,v in budget.items() if k!='wall_seconds'}
        local_budget['wall_seconds']=remaining/max(1,len(stations)-index)
        if remaining<=0 or local_budget['proposal_limit']<4:
            results.append(dict(station=station,status='BUDGET_STOP',error='No remaining paired search allowance'));continue
        try:
            data,state,reference,held,protocol=prepare(frame[frame.station.astype(str)==station],request['variable'],request['unit'],temporal_semantics=request.get('temporal_semantics','instantaneous'),
                unlocated_series=request.get('unlocated_series',False))
            local=folder/f'station-{index+1}';local.mkdir()
            if update:
                if not request.get('background_product'):raise ValueError('Explicit background product required for collocated observation-state update')
                data.attrs['background_product']=request['background_product']
                protocol['observation_update']=dict(update,reference_used=False,forecast_latency_verified=False)
            save(local/'paired_protocol.json',protocol)
            state.to_netcdf(local/'fit_state.nc',engine='h5netcdf')
            reference.attrs['split']='search_validation'
            validation=(BlockedSearchValidation if policy=='blocked_confirmation_v1' else SearchValidation)(reference,request['variable'])
            independent=(BlockedReference if policy=='blocked_confirmation_v1' else DatasetReference)(held,request['variable'],role='independent_validation')
            application=dict(task='temporal',variable=request['variable'],unit=request['unit'],semantic_type='continuous',
                temporal_semantics=request.get('temporal_semantics','instantaneous'),nonnegative=request.get('nonnegative',False),spatial_semantics='point',allow_training=True,soft_guidance=False,
                mutation_control='random',require_mutation_execution=True,search_stopping=dict(enabled=False,evidence='Bounded integration; no calibrated stopping threshold'),initial_proposal_limit=min(64,max(1,local_budget['proposal_limit']//2)),
                target_times=[str(t) for t in data.time.values],
                parameters=dict(reference_support='fixed_finite_reference',require_complete_output=True),
                application_contract=dict(id='paired_station_calibration',version=VERSION,field=request['variable'],
                    unit=request['unit'],semantic_type='continuous',valid_range=None,issues=[],
                    evidence='Explicit paired units and isolated chronological fitting; unlocated series, no spatial claim' if request.get('unlocated_series') else 'Explicit paired units, real nearest-grid station support and isolated chronological fitting'))
            if request.get('recommendation_isolation'):
                application['recommendation_isolation']=request['recommendation_isolation']
            if request.get('review_risk_limits'):
                application['review_risk_limits']=request['review_risk_limits']
            if update:application['parameters']['observation_state_update']=dict(update,excluded_station_ids=stations)
            if policy=='legacy_single_v1':application['review_policy']='legacy_v1'
            compiled=[]
            for recipe in request.get('recommended_recipes',[]):
                if recipe.get('scope',{}).get('variable')!=request['variable']:raise ValueError('Recommendation variable mismatch')
                steps=[]
                for operation in recipe['steps']:
                    if operation in {'units.harmonize','format.normalize','station.time.exact','station.time.nearest','station.grid.nearest'}:continue
                    # This adapter supplies a single background for masked
                    # periods. Observation fallback is not a unary graph node.
                    if request.get('station_two_stage') and operation=='fusion.series_missing_fallback':continue
                    steps.append('bias.prior_series_affine' if operation=='station.bias.holdout' else operation)
                if steps:compiled.append(dict(recipe,steps=steps))
            if compiled:application['recommended_recipes']=compiled
            if request.get('recommended_graphs'):
                application['recommended_graphs']=request['recommended_graphs']
            application_bytes=bytes(data.to_netcdf(engine='h5netcdf'))
            (local/'application_input.nc').write_bytes(application_bytes)
            resources=[SearchInput('background.nc','primary',application_bytes),
                       SearchInput('fit_state.nc','state',bytes(state.to_netcdf(engine='h5netcdf')))]
            if conditioning:
                resource=conditioning[0];resources.append(resource)
                stored=local/('conditioning_input'+Path(resource.filename).suffix)
                stored.write_bytes(resource.content)
                save(local/'conditioning_resource.json',dict(filename=resource.filename,role=resource.role,path=stored.name,
                    metadata=resource.metadata,sha256=hashlib.sha256(resource.content).hexdigest()))
            result=submit_search(resources,application,lambda d:independent(d.sel(time=held.time)),local/'search',
                budget=local_budget,seed=seed+index,progress=progress,validation=validation)
            baseline=independent(data.sel(time=held.time))
            final=result.get('independent_evaluation')
            if policy=='blocked_confirmation_v1' and result.get('best') and final is not None:
                from ecodata.evaluation.blocked_confirmation import review
                selection_baseline=validation(data)
                best=result['best']
                gate=review(best['metrics'],selection_baseline,final,baseline,
                    quality_pass=best['quality']['status']=='PASS',risk_limits=request.get('review_risk_limits'))
                if result['status']!='PASS':
                    gate.update(status='SEARCH_INCOMPLETE',recommended=False,search_status=result['status'])
                isolation=request.get('recommendation_isolation')
                from ecodata.evaluation.blocked_confirmation import apply_recommendation_isolation
                apply_recommendation_isolation(gate,isolation)
                if request.get('reference_basis'):
                    gate.update(reference_basis=request['reference_basis'],accuracy_recommended=False)
                save(local/'recommendation_review.json',gate)
                result.update(recommendation_review=gate,baseline_selection=selection_baseline,
                    candidate_product=result['product'],candidate_frozen_before_confirmation=True,
                    evaluation_protocol=policy,delivery_policy='Baseline unless frozen candidate passes one-shot review; no alternative search on confirmation')
                if not gate['recommended']:
                    # Explicit conservative delivery, not a new selection on
                    # confirmation labels. The searched candidate stays intact.
                    source=next((r for r in result['source_baselines'] if r['key']=='s0' and r['status']=='ELIGIBLE_OUTPUT'),None)
                    if source is None:raise ValueError('No quality-accepted baseline for conservative delivery')
                    path=local/'baseline_product.nc'
                    background=data.assign_attrs(delivery_basis=gate['status'],enhancement_recommended='false')
                    background.to_netcdf(path,engine='h5netcdf')
                    with xr.open_dataset(path) as restored:xr.testing.assert_allclose(background,restored)
                    result.update(product=str(path),delivered_graph=dict(key='s0',records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')))),
                        delivered_best=source,delivery_status='BASELINE_RETAINED',delivered_confirmation=baseline)
                else:
                    result.update(delivered_graph=dict(key=best['key'],records=json.loads((local/'search/graphs.json').read_text(encoding='utf8'))),
                        delivered_best=best,delivery_status='ENHANCEMENT_RECOMMENDED',delivered_confirmation=final)
            if request.get('recommendation_isolation'):
                result['recommendation_isolation']=request['recommendation_isolation']
            result.update(station=station,paired_protocol=protocol,baseline_final=baseline,
                effect_confirmation='NOT_CONFIRMED' if final is None or final['RMSE']>=baseline['RMSE'] else 'LOCAL_FINAL_GAIN_ONLY',
                product_scope='Unlocated time-series predictions; no inferred geography or simulator execution' if request.get('unlocated_series') else 'Predictions at actual station; native grid writeback and simulator execution not performed')
            from ecodata.evaluation.blocked_confirmation import mutation_acceptance
            result['post_mutation_acceptance']=mutation_acceptance(result,local/'search')
            save(local/'search/post_mutation_acceptance.json',result['post_mutation_acceptance'])
            save(local/'result.json',result);results.append(result)
            save(local/'search/summary.json',result)
            if result.get('recommendation_review'):save(local/'search/recommendation_review.json',result['recommendation_review'])
        except (ValueError,KeyError,RuntimeError) as exc:
            results.append(dict(station=station,status='BLOCKED',error=str(exc)))
    result=dict(service='PlatformSearch/PairedStationMutation',stations=results,inputs=audit,
        seconds=perf_counter()-started,budget=budget,test_used_for_selection=False,
        status='PASS' if results and all(r.get('status')=='PASS' for r in results) else 'PARTIAL_OR_BLOCKED')
    save(folder/'paired_summary.json',result)
    return result
