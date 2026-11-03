"""Visible-only raster selection and current-input deployment verification.

No old recipe, external test labels or hand-written workflow pool is consumed.
"""
from dataclasses import replace
import hashlib
import json
import numpy as np
import xarray as xr
from ecodata.algorithms.autonomous_decision import clean,Executor,protect_observed


def _raster_source(inputs,request,context):
    from ecodata.algorithms.platform_search import parse_inputs
    parsed,_=parse_inputs(inputs)
    sources=[(i,role,ds) for i,(role,ds) in enumerate(parsed) if role in {'primary','background'}]
    selected=request.get('protect_observed_source')
    choices=[(n,item) for n,item in enumerate(sources) if (item[2].attrs.get('source_id')==selected if selected else item[1]=='primary')]
    if len(choices)!=1:raise ValueError('Choose one primary reconstruction source or explicit protect_observed_source')
    source_index,(input_index,_,ds)=choices[0];a=ds[context.variable]
    if a.dims!=('time','lat','lon'):raise ValueError('Current raster selection requires time,lat,lon')
    return sources,source_index,input_index,ds,a


def _masked_raster_inputs(inputs,sources,input_index,ds,mask,variable):
    from ecodata.algorithms.platform_search import SearchInput
    adjusted=list(inputs);masked=[];digest=hashlib.sha256(inputs[input_index].content).hexdigest()
    from ecodata.algorithms.task_validation import check_lineage
    check_lineage(inputs,ds)
    for index,role,source in sources:
        same=(index==input_index or hashlib.sha256(inputs[index].content).hexdigest()==digest or
              bool(ds.attrs.get('source_id') and source.attrs.get('source_id')==ds.attrs['source_id']))
        if not same:continue
        if source.attrs.get('crs')!=ds.attrs.get('crs'):raise ValueError('Shared-source alias has incompatible geometry')
        if any(not np.array_equal(source[k],ds[k]) for k in mask.dims):
            raise ValueError('VALIDATION_EVIDENCE_INSUFFICIENT: transformed same-source alias requires replay from masked raw data')
        local=source.copy(deep=True)
        aligned=mask.reindex({name:local[name] for name in mask.dims},fill_value=False)
        local[variable]=local[variable].where(~aligned)
        item=inputs[index];adjusted[index]=SearchInput(item.filename,item.role,clean(local).to_netcdf(),item.metadata)
        masked.append(index)
    return adjusted,masked


def temporal_masks(a,targets,rng):
    """Hold out complete observed dates for a missing-date task, never pixels."""
    times=np.asarray(a.time.values,dtype='datetime64[ns]')
    wanted=np.asarray(targets,dtype='datetime64[ns]')
    valid=np.isfinite(a.values);frames=np.flatnonzero(valid.any(axis=(1,2)))
    missing=~np.isin(wanted,times[frames]);runs=[]
    for index in np.flatnonzero(missing):
        if runs and runs[-1][-1]+1==index:runs[-1].append(int(index))
        else:runs.append([int(index)])
    if not runs:return None
    if len(frames)<4:
        raise ValueError('Insufficient visible dates for whole-date validation: need >=4 for disjoint search/confirmation and two retained anchors')
    longest=max(runs,key=len);requested=len(longest)
    count=min(requested,len(frames)-3)
    interior=frames[1:-1]
    # Match the visible bracketing interval of the requested temporal gap.
    left=max(0,longest[0]-1);right=min(len(wanted)-1,longest[-1]+1)
    duration=int((wanted[right]-wanted[left]).astype('timedelta64[ns]').astype(np.int64))
    windows=[frames[start:start+count] for start in range(1,len(frames)-count)]
    distances=[abs(int((times[frames[start+count]]-times[frames[start-1]]).astype('timedelta64[ns]').astype(np.int64))-duration)
               for start in range(1,len(frames)-count)]
    choices=np.flatnonzero(np.asarray(distances)==min(distances))
    held=windows[int(rng.choice(choices))]
    remaining=np.setdiff1d(interior,held)
    confirmed=int(rng.choice(remaining))
    search=np.zeros(a.shape,bool);confirmation=np.zeros(a.shape,bool)
    search[held]=valid[held];confirmation[confirmed]=valid[confirmed]
    return search,confirmation,dict(design='whole_date_temporal_block',requested_block_dates=requested,
        actual_search_block_dates=count,support_limited=count<requested,
        search_dates=[str(t) for t in times[held]],confirmation_dates=[str(times[confirmed])],
        retained_anchor_dates=[str(t) for t in times[frames] if t not in set(times[np.r_[held,confirmed]])],
        holdout_design='Complete observed frames; adjacent observed-date block matched to requested gap duration; disjoint whole-date confirmation and two retained endpoint anchors',
        limitation='Limited visible dates can reduce block length; no claim of validation coverage of unobserved seasons')


def raster_validation(inputs,request,context):
    from ecodata.algorithms.platform_search import SearchValidation
    sources,source_index,input_index,ds,a=_raster_source(inputs,request,context);variable=context.variable
    seed=request.get('current_input_validation',{}).get('seed',704)
    if type(seed)!=int or seed<0:raise ValueError('Nonnegative visible-mask seed required')
    rng=np.random.default_rng(seed);valid=np.isfinite(a.values);remaining=valid.sum(axis=0)
    search=np.zeros(a.shape,bool);confirmation=np.zeros(a.shape,bool)
    temporal=temporal_masks(a,request.get('target_times',a.time.values),rng)
    point_fraction=float((~valid).mean()) if (~valid).any() else .2
    for t in range(0 if temporal is not None else len(a.time)):
        if (~valid).any() and valid[t].all():continue
        candidates=np.flatnonzero(valid[t]);rng.shuffle(candidates)
        count=min(max(2,int(np.ceil(len(candidates)*point_fraction))),max(0,len(candidates)-1))
        if count==0:continue
        chosen=[]
        for flat in candidates:
            y,x=np.unravel_index(flat,valid[t].shape)
            if remaining[y,x]<(2 if len(a.time)>1 else 1):continue
            chosen.append((y,x));remaining[y,x]-=1
            if len(chosen)>=count:break
        for k,(y,x) in enumerate(chosen):
            (search if k%2==0 else confirmation)[t,y,x]=True
    if temporal is not None:search,confirmation,design=temporal
    else:design=dict(design='point_missing',requested_point_fraction=point_fraction,
        holdout_design='Actual missing fraction on affected frames, minimum two cells for disjoint search/confirmation;20percent diagnostic only if input has no missing values; retain >=1 observation per temporal trace')
    if not search.any() or not confirmation.any():raise ValueError('Insufficient visible support for isolated raster search and confirmation')
    mask=xr.DataArray(search|confirmation,dims=a.dims,coords=a.coords)
    refs=[]
    for values in [search,confirmation]:
        ref=a.where(xr.DataArray(values,dims=a.dims,coords=a.coords)).copy(deep=True)
        ref=ref.isel(time=np.flatnonzero(values.reshape(len(a.time),-1).any(1)))
        ref.attrs=dict(a.attrs,split='search_validation' if values is search else 'internal_confirmation',reference_origin='currently visible observations only')
        refs.append(ref)
    adjusted,masked=_masked_raster_inputs(inputs,sources,input_index,ds,mask,variable)
    protocol=dict(kind='current_visible_raster_search_and_confirmation',mask_seed=seed,
        target_source_index=source_index,target_source_id=ds.attrs.get('source_id'),masked_input_indices=masked,
        search_points=int(search.sum()),confirmation_points=int(confirmation.sum()),
        **design,
        search_confirmation_disjoint=True,confirmation_used_for_selection=False,external_test_used=False,
        natural_missing_used_as_truth=False,unknown_true_error='NOT_ASSESSED',
        input_time_count=int(len(a.time)),finite_observations_per_pixel_min=int(valid.sum(0).min()),
        current_input_sha256=[hashlib.sha256(i.content).hexdigest() for i in inputs])
    return adjusted,SearchValidation(refs[0],variable),protocol,refs[1]


def raster_validation_folds(inputs,request,context):
    """Fixed visible-only folds; temporal LOO never removes a second confirmation frame."""
    from ecodata.algorithms.platform_search import SearchValidation
    sources,source_index,input_index,ds,a=_raster_source(inputs,request,context)
    times=np.asarray(a.time.values,dtype='datetime64[ns]')
    wanted=np.asarray(request.get('target_times',times),dtype='datetime64[ns]')
    if wanted.ndim!=1 or not len(wanted) or np.isnat(wanted).any() or (np.diff(wanted)<=np.timedelta64(0,'ns')).any():
        raise ValueError('Increasing unique finite target dates required for task-matched validation')
    valid=np.isfinite(a.values);frames=np.flatnonzero(valid.any(axis=(1,2)))
    missing=~np.isin(wanted,times[frames]);runs=[]
    for index in np.flatnonzero(missing):
        if runs and runs[-1][-1]+1==index:runs[-1].append(int(index))
        else:runs.append([int(index)])
    goals=[]
    for run in runs:
        start,end=wanted[run[0]],wanted[run[-1]]
        left=times[frames][times[frames]<start];right=times[frames][times[frames]>end]
        bracketed=bool(len(left) and len(right))
        goals.append(dict(start=str(start),end=str(end),missing_dates=len(run),
            missing_span_days=float((end-start)/np.timedelta64(1,'D')),
            left_anchor=str(left[-1]) if len(left) else None,right_anchor=str(right[0]) if len(right) else None,
            bracket_days=float((right[0]-left[-1])/np.timedelta64(1,'D')) if bracketed else None,
            boundary_extrapolation=not bracketed))
    task=dict(visible_effective_dates=len(frames),input_time_count=len(times),requested_missing_dates=int(missing.sum()),
        max_requested_block_dates=max((g['missing_dates'] for g in goals),default=0),
        requested_gaps=goals,visible_dates=[str(t) for t in times[frames]],
        visible_months=sorted({str(t.astype('datetime64[M]')) for t in times[frames]}),
        visible_finite_fraction=float(valid.mean()),natural_missing_used_as_truth=False)

    def fingerprint(changed,design):
        hashes=[hashlib.sha256(item.content).hexdigest() for item in changed]
        value=dict(inputs=[dict(filename=i.filename,role=i.role,metadata=i.metadata,sha256=h) for i,h in zip(changed,hashes)],
            variable=context.variable,semantic_type=context.semantic_type,target_times=[str(t) for t in wanted],
            target_lat=context.target_lat,target_lon=context.target_lon,design=design)
        return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest(),hashes

    if not runs and a.sizes['lat']*a.sizes['lon']>1:
        changed,validation,legacy,confirmation=raster_validation(inputs,request,context)
        identifier,hashes=fingerprint(changed,'point_missing')
        fold_protocol=dict(legacy,fold_id=identifier,input_sha256=hashes)
        fold=dict(id=identifier,inputs=changed,validation=validation,protocol=fold_protocol,confirmation=confirmation)
        return [fold],dict(legacy,kind='current_visible_raster_multifold',fold_count=1,fold_ids=[identifier],
            task_profile=task,confirmation_available=True,independent_confirmation_available=False,
            confirmation_scope='Disjoint visible pixels in the same cohort; not an independent temporal case')

    if len(frames)<3:
        raise ValueError('Insufficient visible dates for temporal cross-validation: need >=3 for one held date and two retained anchors')
    count=request.get('current_input_validation',{}).get('max_folds',3)
    if type(count)!=int or count<1:raise ValueError('Positive integer max_folds required')
    interior=frames[1:-1];count=min(count,len(interior))
    selected=interior[np.linspace(0,len(interior)-1,count).round().astype(int)]
    folds=[]
    for held in selected:
        values=np.zeros(a.shape,bool);values[held]=valid[held]
        mask=xr.DataArray(values,dims=a.dims,coords=a.coords)
        changed,aliases=_masked_raster_inputs(inputs,sources,input_index,ds,mask,context.variable)
        reference=a.isel(time=[int(held)]).copy(deep=True)
        reference.attrs=dict(a.attrs,split='search_validation',reference_origin='currently visible observations only')
        identifier,hashes=fingerprint(changed,'whole_date_leave_one_out')
        position=int(np.flatnonzero(frames==held)[0]);left,right=frames[position-1],frames[position+1]
        retained=valid.sum(axis=0)-valid[held]
        fold_protocol=dict(design='whole_date_leave_one_out',fold_id=identifier,input_sha256=hashes,
            target_source_index=source_index,target_source_id=ds.attrs.get('source_id'),masked_input_indices=aliases,
            held_dates=[str(times[held])],search_points=int(values.sum()),confirmation_points=0,
            retained_effective_dates=len(frames)-1,retained_anchor_dates=[str(t) for t in times[frames] if t!=times[held]],
            bracket_left=str(times[left]),bracket_right=str(times[right]),
            bracket_days=float((times[right]-times[left])/np.timedelta64(1,'D')),
            retained_finite_observations_per_pixel_min=int(retained.min()),
            external_test_used=False,natural_missing_used_as_truth=False,
            confirmation_available=False,independent_confirmation_available=False)
        folds.append(dict(id=identifier,inputs=changed,validation=SearchValidation(reference,context.variable),
            protocol=fold_protocol,confirmation=None))
    protocol=dict(kind='current_visible_raster_multifold',design='whole_date_leave_one_out',
        fold_count=len(folds),fold_ids=[f['id'] for f in folds],target_source_index=source_index,
        target_source_id=ds.attrs.get('source_id'),task_profile=task,input_time_count=len(times),
        folds=[f['protocol'] for f in folds],confirmation_available=False,independent_confirmation_available=False,
        external_test_used=False,natural_missing_used_as_truth=False,
        support_level='LOCAL_CROSS_VALIDATION_ONLY',support_limited=True,
        limitation='LOO folds share observations; no independent confirmation or proof of long-gap/seasonal transfer. Match complete development cases for that evidence.')
    return folds,protocol


def validate_targets(output,context,request):
    """The declared target applies to candidates, not only deployment."""
    field=output[context.variable]
    if field.dims!=('time','lat','lon'):raise ValueError('Complete timed raster deployment required')
    for coordinate,wanted in [('time',request.get('target_times')),('lat',context.target_lat),('lon',context.target_lon)]:
        if wanted is not None:
            if coordinate=='time':wanted=np.asarray(wanted,dtype='datetime64[ns]')
            if not np.array_equal(field[coordinate],wanted):raise ValueError('Deployment target mismatch: '+coordinate)
    if request.get('parameters',{}).get('require_complete_output') and not np.isfinite(field.values).all():
        raise ValueError('Incomplete current-input output; no subset acceptance')


def validate_output(output,context,request):
    """Shared shape/unit/physical gates; no accuracy labels are read here."""
    if request.get('protect_observed_source'):
        source=next(s.dataset for s in context.sources if s.dataset.attrs.get('source_id')==request['protect_observed_source'])
        output=protect_observed(output,source,context.variable)
    elif '_current_primary_index' in request:
        source=context.sources[request['_current_primary_index']].dataset
        if all(np.array_equal(source[k],output[k]) for k in ['lat','lon']):
            output=protect_observed(output,source,context.variable)
    validate_targets(output,context,request)
    field=output[context.variable]
    bounds=context.parameters.get('target_valid_range')
    if bounds is not None and bool(((field<bounds[0])|(field>bounds[1])).any()):raise ValueError('Deployment outside declared range')
    from ecodata.evaluation.quality_gates import ResultQualityGate
    checks=ResultQualityGate().validate(output,output,variable=context.variable,semantic_type=context.semantic_type,
        nonnegative=bool(context.model_constraints.get('nonnegative')))
    if checks['status']=='BLOCK':raise ValueError('Deployment quality gate blocked output')
    return output,checks


def deploy_ranked(search,context,registry,request):
    """Recheck actual inputs; select among search-ranked graphs without truth."""
    search.account.begin_deployment()
    executor=Executor(replace(context,validation_reference=None,validation_role=None),registry,account=search.account,phase='deploy')
    candidates=(search.ranked_rows() if hasattr(search,'ranked_rows') else
                sorted([r for r in search.evaluated.values() if r['status']=='ELIGIBLE_OUTPUT' and not r.get('dominated')],key=search.rank))
    attempts=[]
    for candidate in candidates:
        try:
            data,trace=executor.execute(search.records,candidate['key'])
            data,quality=validate_output(data,context,request)
            attempts.append(dict(key=candidate['key'],status='PASS',quality=quality,trace=trace))
            return candidate,data,dict(status='PASS',attempts=attempts,selection='Original internal-validation order, then current-input contracts and physical quality only',
                external_labels_used=False,automatic_reselection=len(attempts)>1)
        except Exception as exc:
            attempts.append(dict(key=candidate['key'],status='REJECTED',error=str(exc),trace=getattr(exc,'trace',[])))
            if search.account.exhausted('deploy'):break
    return None,None,dict(status='NO_EXECUTABLE_CURRENT_GRAPH',attempts=attempts,external_labels_used=False,automatic_reselection=False)
