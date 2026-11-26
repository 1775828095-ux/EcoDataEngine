"""Task-selected validation over existing masks, executors and area operators.

Reference values only go to the evaluator. Support labels are evidence scope,
not a claim about the unknown gap. No experiment module is imported here.
"""
from hashlib import sha256
import json
import numpy as np
import xarray as xr
from scipy.ndimage import label,find_objects
from ecodata.algorithms.autonomous_decision import clean


VERSION='task_validation_v4_daily_and_raster_support'


def _geometry(y,x,shape,extent):
    h,w=shape;ny,nx=extent
    distances=(y,ny-y-h,x,nx-x-w)
    return tuple(d==0 for d in distances),tuple(distances)


def _calendar_distance(left,right):
    def day(t):
        return float((np.datetime64('2000-'+str(np.datetime64(t,'D'))[5:])-np.datetime64('2000-01-01'))/np.timedelta64(1,'D'))
    distance=abs(day(left)-day(right))
    return min(distance,366-distance)


def _source_presence(sources,index,ds,left,right,variable):
    """Independent-source availability, not an accuracy certificate."""
    result=[]
    for position,_,source in sources:
        if position==index or (ds.attrs.get('source_id') and source.attrs.get('source_id')==ds.attrs['source_id']):continue
        dates=np.asarray(source.time.values,dtype='datetime64[ns]')
        dates=dates[np.isfinite(source[variable].values).any(axis=(1,2))]
        before=np.flatnonzero(dates<=left);after=np.flatnonzero(dates>=right)
        if not len(before) or not len(after):result.append(False);continue
        result.append(True)
    return result


def _ring_profile(values,t,y,x,template,scale):
    from scipy.ndimage import binary_dilation
    nt,h,w=template.shape;ny,nx=values.shape[1:]
    y0,x0=max(0,y-1),max(0,x-1);y1,x1=min(ny,y+h+1),min(nx,x+w+1)
    held=np.zeros((nt,y1-y0,x1-x0),bool);held[:,y-y0:y-y0+h,x-x0:x-x0+w]=template
    ring=binary_dilation(held,structure=np.ones((1,3,3),bool))&~held
    a=values[t:t+nt,y0:y1,x0:x1].copy();a[~ring]=np.nan
    v=a[np.isfinite(a)]
    if len(v)<4:return None
    changes=np.concatenate([np.abs(np.diff(a,axis=i)).ravel() for i in (1,2)])
    changes=changes[np.isfinite(changes)]
    if not len(changes):return None
    return np.array([np.median(v),np.std(v),np.mean(changes)])/scale


def check_lineage(inputs,primary):
    from ecodata.algorithms.platform_search import parse_inputs
    parsed,_=parse_inputs(inputs);root=primary.attrs.get('source_id')
    dependencies={}
    for (_,ds),item in zip(parsed,inputs):
        declared=ds.attrs.get('derived_from',(item.metadata or {}).get('derived_from',[]))
        if isinstance(declared,str):
            try:declared=json.loads(declared)
            except json.JSONDecodeError:declared=[declared]
        if isinstance(declared,str):declared=[declared]
        if not isinstance(declared,(tuple,list)) or any(not isinstance(x,str) for x in declared):
            raise ValueError('Structured source-ID lineage required')
        dependencies[ds.attrs.get('source_id',item.filename)]=set(declared)
    if not root and any(dependencies.values()):
        raise ValueError('VALIDATION_EVIDENCE_INSUFFICIENT: primary source_id required to isolate declared derived inputs')
    affected={root} if root else set()
    while True:
        changed=affected|{key for key,parents in dependencies.items() if parents&affected}
        if changed==affected:break
        affected=changed
    for (role,ds),item in zip(parsed,inputs):
        sid=ds.attrs.get('source_id',item.filename)
        if dependencies.get(sid,set())&affected:
            raise ValueError('VALIDATION_EVIDENCE_INSUFFICIENT: derived input '+sid+' requires replay after masking, not masking its already-computed output')
        if role in {'state','training_reference','historical_simulation'}:
            end=ds.attrs.get('training_end')
            if end is None or np.datetime64(end)>=np.asarray(primary.time.values).min():
                raise ValueError('VALIDATION_EVIDENCE_INSUFFICIENT: fitted resources need independent prior training provenance')


def _identity(changed,request,kind):
    value=dict(inputs=[sha256(i.content).hexdigest() for i in changed],request=request,kind=kind)
    return sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def _fold(inputs,sources,index,ds,mask,request,kind,detail):
    from ecodata.algorithms.current_input import _masked_raster_inputs
    from ecodata.algorithms.platform_search import SearchValidation
    variable=request['variable'];a=ds[variable]
    mask=xr.DataArray(mask,dims=a.dims,coords=a.coords)&np.isfinite(a)
    if not bool(mask.any()):return None
    changed,aliases=_masked_raster_inputs(inputs,sources,index,ds,mask,variable)
    reference=a.where(mask).isel(time=np.flatnonzero(mask.values.reshape(len(a.time),-1).any(1))).copy(deep=True)
    reference.attrs=dict(a.attrs,split='search_validation',reference_origin='visible observations only')
    q=dict(request,parameters=dict(request.get('parameters',{}),reference_support='fixed_finite_reference',require_complete_output=True))
    identifier=_identity(changed,q,kind)
    protocol=dict(design=kind,fold_id=identifier,masked_input_indices=aliases,search_points=int(mask.sum()),
        input_sha256=[sha256(i.content).hexdigest() for i in changed],external_test_used=False,
        confirmation_available=False,independent_confirmation_available=False,**detail)
    return dict(id=identifier,inputs=changed,validation=SearchValidation(reference,variable),request=q,protocol=protocol,confirmation=None)


def _temporal(inputs,sources,index,ds,request,task,limit):
    """Compare actual calendar/anchor support; never call shorter masks matched."""
    a=ds[request['variable']];valid=np.isfinite(a.values);times=np.asarray(a.time.values,dtype='datetime64[ns]')
    frames=np.flatnonzero(valid.any(axis=(1,2)));goals=task.get('requested_gaps',[])
    folds=[];diagnostics=[]
    if len(frames)<3:return folds,dict(status='INSUFFICIENT',reason='Fewer than three effective dates')
    cadence=float(np.median(np.diff(np.asarray(request['target_times'],dtype='datetime64[ns]'))/np.timedelta64(1,'D')))
    visible=a.values[np.isfinite(a.values)];scale=float(np.std(visible)) or 1.
    point=a.sizes['lat']*a.sizes['lon']==1
    cohorts=[dict(mask=np.zeros(a.shape,bool),anchors=set(),blocks=[]) for _ in range(limit)] if point else []
    def anchor_context(left,right):
        rows=[a.values[np.flatnonzero(times==t)[0]] for t in (left,right)]
        finite=[v[np.isfinite(v)] for v in rows]
        if any(len(v)<(1 if point else 4) for v in finite):return None
        features=[x/scale for v in finite for x in (float(np.median(v)),float(np.std(v)))]
        support=_source_presence(sources,index,ds,left,right,request['variable'])
        for position,_,source in sources:
            if position==index or source.attrs.get('source_id')==ds.attrs.get('source_id'):continue
            dates=np.asarray(source.time.values,dtype='datetime64[ns]')
            window=(dates>=left)&(dates<=right)
            values=source[request['variable']].values[window]
            coverage=float(np.isfinite(values).mean()) if values.size else 0.
            features.append(coverage)
        return np.asarray(features),support
    for goal in goals:
        candidates=[]
        if goal['boundary_extrapolation']:
            diagnostics.append(dict(goal=goal,status='INSUFFICIENT',reason='Boundary extrapolation requires forward/backward availability-specific validation'));continue
        actual_context=anchor_context(np.datetime64(goal['left_anchor']),np.datetime64(goal['right_anchor']))
        if actual_context is None:
            diagnostics.append(dict(goal=goal,status='INSUFFICIENT',reason='Insufficient finite endpoint context'));continue
        for first in range(1,len(frames)-1):
            left=times[frames[first-1]]
            bracket_tolerance=min(cadence,.25*goal['bracket_days'])
            span_tolerance=min(cadence,.25*max(cadence,goal['missing_span_days']))
            # Bound right-anchor positions by the same duration tolerance;
            # avoid quadratic enumeration of impossible multi-year blocks.
            low=left+np.timedelta64(int(round((goal['bracket_days']-bracket_tolerance)*86400e9)),'ns')
            high=left+np.timedelta64(int(round((goal['bracket_days']+bracket_tolerance)*86400e9)),'ns')
            lo=max(first+1,int(np.searchsorted(times[frames],low)))
            hi=min(len(frames),int(np.searchsorted(times[frames],high,side='right')))
            for right_position in range(lo,hi):
                last=right_position-1
                held=frames[first:last+1];left,right=times[frames[first-1]],times[frames[last+1]]
                if point and len(held)!=goal['missing_dates']:continue
                span=float((times[held[-1]]-times[held[0]])/np.timedelta64(1,'D'))
                bracket=float((right-left)/np.timedelta64(1,'D'))
                # Calendar distance uses month/day; no target values are inspected.
                target_middle=np.datetime64(goal['start'])+(np.datetime64(goal['end'])-np.datetime64(goal['start']))/2
                candidate_middle=times[held[0]]+(times[held[-1]]-times[held[0]])/2
                seasonal=_calendar_distance(candidate_middle,target_middle)
                if abs(bracket-goal['bracket_days'])>bracket_tolerance or abs(span-goal['missing_span_days'])>span_tolerance or seasonal>max(cadence,goal['bracket_days']):continue
                context=anchor_context(left,right)
                if context is None or context[1]!=actual_context[1]:continue
                distance=float(np.mean(abs(context[0]-actual_context[0])))
                score=abs(bracket-goal['bracket_days'])+abs(span-goal['missing_span_days'])+seasonal/max(1,cadence)
                candidates.append((distance,score,first,last,held,span,bracket))
        candidates.sort(key=lambda c:c[:3]);used=[]
        if point:
            # Each daily fold covers every requested gap's conditions. Pack
            # non-touching blocks rather than making one fold per year/gap.
            for cohort in cohorts:
                for distance,_,first,last,held,span,bracket in candidates:
                    if any(set(held)&set(previous) for previous in used):continue
                    anchors={int(frames[first-1]),int(frames[last+1])}
                    if any(cohort['mask'][k].any() for k in anchors) or set(held)&cohort['anchors']:continue
                    cohort['mask'][held]=valid[held];cohort['anchors']|=anchors
                    cohort['blocks'].append(dict(held_dates=[str(t) for t in times[held]],
                        requested_gap=goal,actual_hidden_span_days=span,actual_bracket_days=bracket,
                        calendar_distance_days=_calendar_distance(times[held[0]]+(times[held[-1]]-times[held[0]])/2,
                            np.datetime64(goal['start'])+(np.datetime64(goal['end'])-np.datetime64(goal['start']))/2),
                        visible_anchor_context_distance=distance,independent_source_presence_matched=True,
                        retained_anchor_dates=[str(times[k]) for k in sorted(anchors)]))
                    used.append(held);break
            diagnostics.append(dict(goal=goal,status='MATCHED' if len(used)==limit else 'INSUFFICIENT',matched_folds=len(used),
                reason=None if len(used)==limit else 'Not every required daily fold has a duration/calendar/anchor/source-matched block'))
            continue
        for distance,_,first,last,held,span,bracket in candidates:
            if any(set(held)&set(previous) for previous in used):continue
            mask=np.zeros(a.shape,bool);mask[held]=valid[held]
            f=_fold(inputs,sources,index,ds,mask,request,'matched_temporal_block',dict(
                held_dates=[str(t) for t in times[held]],retained_effective_dates=len(frames)-len(held),
                actual_hidden_span_days=span,actual_bracket_days=bracket,requested_gap=goal,
                visible_anchor_context_distance=distance,independent_source_presence_matched=True,
                bracket_tolerance_days=bracket_tolerance,span_tolerance_days=span_tolerance,
                tolerance_basis='At most one sampling interval and25-percent duration distortion; engineering support envelope'))
            if f:folds.append(f);used.append(held)
            if len(used)>=limit:break
        diagnostics.append(dict(goal=goal,status='MATCHED' if used else 'INSUFFICIENT',matched_folds=len(used),
            reason=None if used else 'No observed block matches requested duration, anchors and calendar support'))
    if point:
        for cohort in cohorts:
            f=_fold(inputs,sources,index,ds,cohort['mask'],request,'matched_daily_temporal_blocks',dict(
                blocks=cohort['blocks'],retained_effective_dates=int((valid&~cohort['mask']).any(axis=(1,2)).sum()),
                context_basis='Visible endpoint values, source availability, duration and month/day; held values never rank masks',
                all_requested_gaps_matched=len(cohort['blocks'])==len(goals),validation_metric='RMSE'))
            if f:folds.append(f)
    return folds,dict(status='MATCHED' if goals and all(g['status']=='MATCHED' for g in diagnostics) else 'INSUFFICIENT',gaps=diagnostics)


def _blocks(inputs,sources,index,ds,request,limit):
    a=ds[request['variable']];valid=np.isfinite(a.values)
    # Completely absent dates are handled by temporal validation, not giant clouds.
    missing=~valid;missing[~valid.any(axis=(1,2))]=False
    labels,count=label(missing);objects=find_objects(labels)
    templates=[]
    for number,slices in enumerate(objects,1):
        if slices is None:continue
        template=labels[slices]==number
        if template.sum()<2:continue
        kind='spatiotemporal_block' if template.shape[0]>1 else 'spatial_block'
        templates.append((int(template.sum()),kind,template,tuple(s.start for s in slices)))
    templates.sort(key=lambda t:-t[0]);folds=[];diagnostics=[]
    counts={}
    for _,kind,template,origin in templates:
        original_t,original_y,original_x=origin;shape=template.shape
        used=np.zeros(a.shape,bool);found=0
        geometry=_geometry(original_y,original_x,shape[1:],a.shape[1:])
        v=a.values[np.isfinite(a.values)];scale=float(np.std(v)) or 1.
        original_profile=_ring_profile(a.values,*origin,template,scale)
        candidates=[]
        starts=sorted(range(a.shape[0]-shape[0]+1),key=lambda t:abs(t-original_t))
        dates=np.asarray(a.time.values,dtype='datetime64[ns]')
        original_offsets=(dates[original_t:original_t+shape[0]]-dates[original_t])/np.timedelta64(1,'D')
        tolerance=.25*float(np.median(np.diff(dates)/np.timedelta64(1,'D'))) if len(dates)>1 else 0.
        presence=_source_presence(sources,index,ds,dates[original_t],dates[original_t+shape[0]-1],request['variable'])
        for t in starts:
            if counts.get(kind,0)>=limit:break
            offsets=(dates[t:t+shape[0]]-dates[t])/np.timedelta64(1,'D')
            if np.max(abs(offsets-original_offsets))>tolerance:continue
            if _calendar_distance(dates[t],dates[original_t])>float(original_offsets[-1])+4*tolerance:continue
            if _source_presence(sources,index,ds,dates[t],dates[t+shape[0]-1],request['variable'])!=presence:continue
            for y in [original_y]:
                for x in [original_x]:
                    if _geometry(y,x,shape[1:],a.shape[1:])!=geometry:continue
                    sl=(slice(t,t+shape[0]),slice(y,y+shape[1]),slice(x,x+shape[2]))
                    if not valid[sl][template].all() or original_profile is None:continue
                    profile=_ring_profile(a.values,t,y,x,template,scale)
                    if profile is None:continue
                    candidates.append((float(np.mean(abs(profile-original_profile))),abs(t-original_t),t,y,x))
        for distance,_,t,y,x in sorted(candidates):
            sl=(slice(t,t+shape[0]),slice(y,y+shape[1]),slice(x,x+shape[2]))
            if (used[sl]&template).any():continue
            mask=np.zeros(a.shape,bool);mask[sl]=template
            if (valid&~mask).sum()<4:continue
            f=_fold(inputs,sources,index,ds,mask,request,kind,dict(template_shape=list(shape),
                connected_missing_cells=int(template.sum()),placement=[t,y,x],
                duration_offsets_days=((dates[t:t+shape[0]]-dates[t])/np.timedelta64(1,'D')).tolist(),
                boundary_contacts=list(geometry[0]),boundary_distances=list(geometry[1]),visible_ring_distance=distance,
                independent_source_presence_matched=True,calendar_distance_days=_calendar_distance(dates[t],dates[original_t]),
                context_basis='Outside-mask median, spread and adjacent differences normalized by visible standard deviation; nearest context ranking, not a calibrated accuracy bound',
                geometry='Exact per-side boundary distances on the same grid; held values excluded from context descriptors'))
            if f:folds.append(f);used|=mask;found+=1
            counts[kind]=counts.get(kind,0)+1
            if found>=limit or counts[kind]>=limit:break
        diagnostics.append(dict(kind=kind,status='MATCHED' if found else 'INSUFFICIENT',folds=found,
            template_selection='All connected components audited; existing per-kind fold cap retained',
            requested_shape=list(shape),reason=None if found else 'No matching finite block within position/calendar/source support, or per-kind fold cap reached'))
    return folds,diagnostics


def _scale(inputs,sources,index,ds,request,context,account):
    from ecodata.algorithms.raster_reconstruction_task import edges,overlap_weights
    from ecodata.algorithms.platform_search import SearchInput,SearchValidation
    from scipy import sparse
    a=ds[context.variable]
    factor=max(float(np.median(abs(np.diff(a.lat)))/np.median(abs(np.diff(context.target_lat)))),
               float(np.median(abs(np.diff(a.lon)))/np.median(abs(np.diff(context.target_lon)))))
    size=[int(np.floor(a.sizes[c]/factor)) for c in ('lat','lon')]
    if min(size)<2 or a.attrs.get('spatial_semantics')!='cell_mean' or ds.attrs.get('crs')!='EPSG:4326':
        return [],dict(status='INSUFFICIENT',reason='Reduced-scale reference needs >=2 coarse cells per axis, WGS84 and declared cell means',scale_ratio=factor)
    fine_edges=[edges(a[c]) for c in ('lat','lon')]
    coarse_edges=[np.linspace(e[0],e[-1],n+1) for e,n in zip(fine_edges,size)]
    weights=[overlap_weights(fine_edges[0],coarse_edges[0],True),overlap_weights(fine_edges[1],coarse_edges[1])]
    operator=sparse.kron(*weights,format='csr')
    adjusted=[]
    for i,item in enumerate(inputs):
        source=next((source for position,_,source in sources if position==i),None)
        same=source is not None and (i==index or (ds.attrs.get('source_id') and source.attrs.get('source_id')==ds.attrs['source_id']))
        if not same:adjusted.append(item);continue
        if any(not np.array_equal(source[c],ds[c]) for c in a.dims):
            raise ValueError('VALIDATION_EVIDENCE_INSUFFICIENT: scale alias requires raw-source replay')
        token=account.before('validation.area_degradation','validation_setup',{},graph='scale_proxy') if account else None
        try:
            values=source[context.variable].values.reshape(len(a.time),-1)
            known=np.isfinite(values);coverage=np.stack([operator@k.astype(float) for k in known])
            result=np.stack([operator@np.nan_to_num(v,nan=0.) for v in values]);result[coverage<1-1e-10]=np.nan
            degraded=xr.Dataset({context.variable:(('time','lat','lon'),result.reshape(len(a.time),*size))},
                coords=dict(time=a.time,lat=(coarse_edges[0][:-1]+coarse_edges[0][1:])/2,lon=(coarse_edges[1][:-1]+coarse_edges[1][1:])/2),attrs=dict(source.attrs))
            degraded[context.variable].attrs=dict(a.attrs)
        except Exception as exc:
            if account:account.after(token,error=exc)
            raise
        else:
            if account:account.after(token)
        adjusted.append(SearchInput(item.filename,item.role,clean(degraded).to_netcdf(),item.metadata))
    q=dict(request,target_lat=a.lat.values.tolist(),target_lon=a.lon.values.tolist(),
        target_times=[str(t) for t in a.time.values],parameters=dict(request.get('parameters',{}),
            raster_target_times=[str(t) for t in a.time.values],reference_support='fixed_finite_reference',require_complete_output=True))
    ref=a.copy(deep=True);ref.attrs=dict(a.attrs,split='search_validation',reference_origin='native visible observations at reduced-resolution proxy only')
    identifier=_identity(adjusted,q,'reduced_resolution_scale')
    protocol=dict(design='reduced_resolution_scale',fold_id=identifier,scale_ratio=factor,
        input_shape=size,reference_shape=[a.sizes[c] for c in ('lat','lon')],
        operator='Existing spherical area-overlap cell-mean operator; no unknown sensor PSF asserted',
        native_fine_truth_verified=False,external_test_used=False,confidence='PROXY_ONLY',
        input_sha256=[sha256(i.content).hexdigest() for i in adjusted])
    return [dict(id=identifier,inputs=adjusted,request=q,validation=SearchValidation(ref,context.variable),protocol=protocol,confirmation=None)],dict(status='PROXY_ONLY',**protocol)


def plan(inputs,request,context,account=None):
    from ecodata.algorithms.current_input import _raster_source,raster_validation_folds
    sources,source_index,index,ds,a=_raster_source(inputs,request,context)
    check_lineage(inputs,ds)
    target=[context.target_lat,context.target_lon]
    scaling=all(v is not None and len(v)>1 for v in target) and any(
        np.median(abs(np.diff(v)))<np.median(abs(np.diff(a[c])))*(1-1e-6) for c,v in zip(('lat','lon'),target))
    if scaling:
        folds,support=_scale(inputs,sources,index,ds,request,context,account)
        if not folds:raise ValueError('VALIDATION_EVIDENCE_INSUFFICIENT: '+support['reason'])
        protocol=dict(version=VERSION,kind='task_matched_raster',design='reduced_resolution_scale',
            fold_count=len(folds),target_source_index=source_index,input_time_count=len(a.time),task_profile={},
            tasks=['scale_refinement'],support_level='REDUCED_RESOLUTION_PROXY_ONLY',support_limited=True,
            task_support=[support],primary_fold_ids=[f['id'] for f in folds],confirmation_available=False,
            independent_confirmation_available=False,external_test_used=False,unknown_true_error='NOT_ASSESSED')
        requested=np.asarray(request.get('target_times',a.time.values),dtype='datetime64[ns]')
        visible=np.asarray(a.time.values)[np.isfinite(a.values).any(axis=(1,2))]
        if not np.isfinite(a.values).all() or not np.isin(requested,visible).all():
            from dataclasses import replace
            native_request=dict(request,target_lat=a.lat.values.tolist(),target_lon=a.lon.values.tolist())
            native_folds,native=plan(inputs,native_request,replace(context,target_lat=a.lat.values,target_lon=a.lon.values),account)
            folds+=native_folds
            protocol.update(design='composed_scale_and_recovery',tasks=['scale_refinement',*native['tasks']],
                task_profile=native['task_profile'],task_support=[support,*native['task_support']],
                fold_count=len(folds),primary_fold_ids=[f['id'] for f in folds],
                recovery_support=native['support_level'],folds=[f['protocol'] for f in folds])
        protocol['isolation_applied']=len(sources)>1
        return folds,protocol
    try:base,protocol=raster_validation_folds(inputs,request,context)
    except ValueError as exc:
        if str(exc).startswith(('Insufficient visible dates','Insufficient visible support')):
            raise ValueError('VALIDATION_EVIDENCE_INSUFFICIENT: '+str(exc)) from exc
        raise
    limit=request.get('current_input_validation',{}).get('max_folds',3)
    temporal,temporal_support=_temporal(inputs,sources,index,ds,request,protocol['task_profile'],limit)
    blocks,block_support=_blocks(inputs,sources,index,ds,request,limit)
    selected=list({f['id']:f for f in temporal+blocks}.values())
    tasks=(['temporal_gap'] if protocol['task_profile']['requested_gaps'] else [])+[s['kind'] for s in block_support]
    support=([dict(kind='temporal_gap',**temporal_support)] if 'temporal_gap' in tasks else [])+block_support
    if not selected:selected=base
    covered=bool(tasks) and all(s['status']=='MATCHED' for s in support)
    protocol.update(version=VERSION,kind='task_matched_series' if a.sizes['lat']*a.sizes['lon']==1 else 'task_matched_raster',tasks=tasks or ['point_missing'],
        design='task_matched_blocks' if temporal or blocks else protocol['design'],
        fold_count=len(selected),fold_ids=[f['id'] for f in selected],folds=[f['protocol'] for f in selected],
        primary_fold_ids=[f['id'] for f in selected],task_support=support,
        support_level='TASK_MATCHED_INTERNAL' if covered else 'LIMITED_DIAGNOSTIC_ONLY' if tasks else 'POINT_HOLDOUT',
        support_limited=bool(tasks) and not covered,independent_confirmation_available=False,
        confirmation_available=bool(not tasks and selected[0].get('confirmation') is not None),
        confirmation_points=0 if tasks else protocol.get('confirmation_points',0),
        isolation_applied=len(sources)>1,
        objective='Equal required-fold mean of RMSE by default; explicit model metric weights are recorded separately',
        metric_definition='Per-fold RMSE=sqrt(mean(error**2)) on every finite reference cell; frame means are diagnostic only',
        limitation='Internal matched conditions only; no guarantee for unobserved gaps, events or domains')
    return selected,protocol


def outcome(protocol,request,stop_reason,confirmation=None):
    goal=request.get('quality_goal')
    if goal is not None and (not isinstance(goal,dict) or not goal.get('evidence')):
        raise ValueError('Quality goal requires explicit purpose/evidence; never an invented error threshold')
    matched=not protocol.get('support_limited',True)
    return dict(status='EVIDENCE_INSUFFICIENT' if not matched else 'GOAL_NOT_SPECIFIED' if goal is None else 'INDEPENDENT_CONFIRMATION_REQUIRED',
        task_quality_achieved=False,goal=goal,computation_stop_reason=stop_reason,
        explanation='A qualified output or stagnant search does not certify the unknown task error',
        support_level=protocol.get('support_level'),confirmation_available=confirmation is not None)


def reserve_review(folds,inputs,request,context):
    """Reserve a whole existing task-matched case before numerical search.

    Its labels are removed from every selection input and the feature context.
    Incompatible geometries/lineage or fewer than two disjoint cases leave an
    explicit limitation; they never produce a counterfeit confirmation fold.
    """
    from ecodata.algorithms.current_input import _raster_source,_masked_raster_inputs
    sources,_,index,ds,a=_raster_source(inputs,request,context)
    kinds=[f['protocol'].get('design','') for f in folds]
    axes=['space','time'] if any('spatiotemporal' in k for k in kinds) else ['space'] if any(k=='spatial_block' for k in kinds) else ['time']
    if len(folds)<3:return folds,None,None,dict(status='INSUFFICIENT',reason='Need two selection cases and one separate matched case',axes=axes)
    outer=max(folds,key=lambda f:np.asarray(f['validation'].reference.time.values).max()) if 'time' in axes else folds[-1]
    ref=outer['validation'].reference
    mask=np.isfinite(ref).reindex_like(a,fill_value=False)
    # A held date/location is sealed in full, not merely at a few pixels.
    original_mask=mask
    if 'time' in axes:mask=mask|original_mask.any(dim=('lat','lon')).broadcast_like(a)
    if 'space' in axes:mask=mask|original_mask.any(dim='time').broadcast_like(a)
    if any(not np.array_equal(ref[c],a[c]) for c in ('lat','lon')):
        return folds,None,None,dict(status='INSUFFICIENT',reason='Outer/inner native support differs; no unsafe geometry mapping',axes=axes)
    def support(reference):
        valid=np.isfinite(reference.values)
        t=np.any(valid,axis=(1,2));s=np.any(valid,axis=0)
        return set(reference.time.values[t].astype('datetime64[ns]').astype(str)),set(zip(*np.where(s)))
    ot,os=support(ref);remaining=[]
    try:
        profile_inputs,_=_masked_raster_inputs(inputs,sources,index,ds,mask,context.variable)
        from ecodata.algorithms.autonomous_field import build_context
        inner_context=build_context(profile_inputs,request)[0]
        candidates,inner_protocol=plan(profile_inputs,request,inner_context)
        if inner_protocol.get('support_limited') is True:
            return folds,None,None,dict(status='INSUFFICIENT',reason='Task-matched selection support becomes limited after isolating confirmation',axes=axes)
        from ecodata.algorithms.platform_search import parse_inputs
        for fold in candidates:
            ft,fs=support(fold['validation'].reference)
            if ('time' in axes and ft&ot) or ('space' in axes and fs&os):continue
            if 'time' in axes and max(ft)>=min(ot):continue
            parsed,_=parse_inputs(fold['inputs'])
            local_sources=[(i,r,d) for i,(r,d) in enumerate(parsed) if r in {'primary','background'}]
            local=parsed[index][1]
            adjusted,aliases=_masked_raster_inputs(fold['inputs'],local_sources,index,local,mask,context.variable)
            identifier=_identity(adjusted,fold.get('request',request),'outer_isolated_selection')
            remaining.append(dict(fold,id=identifier,inputs=adjusted,protocol=dict(fold['protocol'],fold_id=identifier,outer_labels_removed_from_inputs=True,
                review_masked_aliases=aliases,input_sha256=[sha256(i.content).hexdigest() for i in adjusted])))
        parsed,_=parse_inputs(outer['inputs'])
        outer_sources=[(i,r,d) for i,(r,d) in enumerate(parsed) if r in {'primary','background'}]
        outer_inputs,_=_masked_raster_inputs(outer['inputs'],outer_sources,index,parsed[index][1],mask,context.variable)
        outer=dict(outer,inputs=outer_inputs,id=_identity(outer_inputs,outer.get('request',request),'frozen_outer_review'))
    except ValueError as exc:
        return folds,None,None,dict(status='INSUFFICIENT',reason=str(exc),axes=axes)
    if len(remaining)<2:return folds,None,None,dict(status='INSUFFICIENT',reason='Fewer than two cases remain after support-isolated reservation',axes=axes)
    return remaining,outer,profile_inputs,dict(status='RESERVED',axes=axes,outer_id=outer['id'],
        selection_ids=[f['id'] for f in remaining],reference_scope='Withheld visible values; upstream product independence unknown',
        outer_labels_in_search_inputs=False,outer_labels_in_features=False,
        regenerated_selection_support=inner_protocol.get('support_level','UNKNOWN'))


def scale_consistency(output,source,variable,account):
    """Observed-scale consistency is a diagnostic, never fine-resolution truth."""
    from ecodata.algorithms.raster_reconstruction_task import edges,overlap_weights
    from scipy import sparse
    from ecodata.algorithms.search_accounting import SearchBudgetExceeded
    token=None;failure=None
    try:
        if source.attrs.get('crs')!='EPSG:4326' or source[variable].attrs.get('spatial_semantics')!='cell_mean':
            return dict(status='INSUFFICIENT',reason='Explicit WGS84 cell means required')
        fine=[edges(output[c]) for c in ('lat','lon')];native=[edges(source[c]) for c in ('lat','lon')]
        indices=[np.flatnonzero((e[:-1]>=f[0]-1e-12)&(e[1:]<=f[-1]+1e-12)) for f,e in zip(fine,native)]
        times=np.intersect1d(output.time.values,source.time.values)
        if any(not len(i) for i in indices) or not len(times):
            return dict(status='INSUFFICIENT',reason='No fully covered native cell and exact common observation time')
        token=account.before('validation.area_backprojection','validation_audit',{},graph='deployed_scale_consistency')
        weights=[overlap_weights(f,e[np.r_[i,i[-1]+1]],latitude=(axis==0)) for axis,(f,e,i) in enumerate(zip(fine,native,indices))]
        operator=sparse.kron(*weights,format='csr')
        a=output[variable].sel(time=times).values.reshape(len(times),-1)
        known=np.stack([operator@np.isfinite(row).astype(float) for row in a])
        projected=np.stack([operator@np.nan_to_num(row,nan=0.) for row in a])
        reference=source[variable].sel(time=times).isel(lat=indices[0],lon=indices[1]).values.reshape(len(times),-1)
        mask=np.isfinite(reference)&(known>=1-1e-10)
        if not mask.any():return dict(status='INSUFFICIENT',reason='No finite area-complete native support')
        error=(projected-reference)[mask]
        return dict(status='MEASURED',RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.mean(abs(error))),
            points=int(mask.sum()),common_times=len(times),fine_truth_verified=False,
            interpretation='Spherical area backprojection against native cell means; consistency only, not fine-scale accuracy')
    except (ValueError,AssertionError,SearchBudgetExceeded) as exc:
        failure=exc
        return dict(status='INSUFFICIENT',reason=str(exc),fine_truth_verified=False)
    finally:
        if token is not None:account.after(token,error=failure)
