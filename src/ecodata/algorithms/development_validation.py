"""Published development-only temporal simulations, separate from local CV."""
from pathlib import Path
import hashlib
import json
import numpy as np
import xarray as xr
from ecodata.algorithms.attribute_registry import digest
from ecodata.algorithms.autonomous_decision import clean


DEFAULT_REGISTRY=Path(__file__).resolve().parents[3]/'data/development_validation/published.json'


def _calendar_days(times):
    return np.array([int((np.datetime64('2000-'+str(t.astype('datetime64[D]'))[5:],'D')-
        np.datetime64('2000-01-01','D'))/np.timedelta64(1,'D')) for t in times])


def _ordered_dates(wanted,available):
    """Minimum total calendar distance, chronological and without replacement."""
    n,m=len(wanted),len(available);cost=np.full((n+1,m+1),np.inf);cost[0]=0
    take=np.zeros((n+1,m+1),bool)
    for i in range(1,n+1):
        for j in range(1,m+1):
            selected=cost[i-1,j-1]+abs(int(wanted[i-1])-int(available[j-1]))
            take[i,j]=selected<cost[i,j-1]
            cost[i,j]=selected if take[i,j] else cost[i,j-1]
    indices=[];i,j=n,m
    while i:
        if take[i,j]:indices.append(j-1);i-=1
        j-=1
    return np.array(indices[::-1],int)


def _read_resource(item,base):
    path=Path(item['path']);path=path if path.is_absolute() else base/path
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item.get('sha256'):raise ValueError('Development resource SHA256 mismatch')
    with xr.open_dataset(path) as opened:
        if opened.attrs.get('evidence_role')!='development':raise ValueError('Historical resources must declare evidence_role=development, never test')
        if opened.attrs.get('sensor')!=item.get('sensor'):raise ValueError('Published development sensor differs from file metadata')
        if opened.attrs.get('split') in {'test','final_test','external_test'}:
            raise ValueError('Test resource cannot be used for historical validation')
        for value in opened.data_vars.values():
            if value.attrs.get('evidence_role') in {'test','final_test','external_test'} or value.attrs.get('split') in {'test','final_test','external_test'}:
                raise ValueError('Test-labelled field cannot be used for historical validation')
        return opened.load(),item['sha256']


def applicability(current,historical,wanted,matched,visible,variable):
    """Conservative input-support checks, never a predicted gain or hidden label test."""
    days=_calendar_days(wanted);other=_calendar_days(matched)
    cadence=float(np.median(np.diff(days)))
    checks=[]
    def add(name,ok,actual,limit,basis):
        checks.append(dict(name=name,status='PASS' if ok else 'REJECT',actual=actual,limit=limit,evidence=basis))
    distance=float(abs(days-other).max())
    add('calendar_assignment',distance<=cadence/2,distance,cadence/2,
        'Half the target median spacing avoids representing a date by a different nominal sampling slot')
    spacing_error=float(abs(np.diff(days)-np.diff(other)).max())
    add('sampling_intervals',spacing_error<=cadence,spacing_error,cadence,
        'Reject interval distortion larger than one target median interval; engineering support guard, not learned accuracy')
    if any(c['status']=='REJECT' for c in checks):
        return dict(version='visible_support_v1',status='REJECT',checks=checks,gain_prediction='UNKNOWN')
    for index,(a,b) in enumerate(zip(current,historical)):
        ratios=[float(abs(np.median(np.diff(b[c])))/abs(np.median(np.diff(a[c])))) for c in ('lat','lon')]
        add('source'+str(index)+'_spatial_spacing',all(.8<=r<=1.25 for r in ratios),ratios,[.8,1.25],
            'Frozen 25-percent grid-spacing support envelope; no cross-scale applicability claim')
        av=a[variable].values;bv=b[variable].values
        if index==0:
            av=a[variable].sel(time=wanted[visible]).values
            bv=b[variable].sel(time=matched[visible]).values
        af=av[np.isfinite(av)];bf=bv[np.isfinite(bv)]
        if len(af)<4 or len(bf)<4:
            return dict(version='visible_support_v1',status='UNKNOWN',reason='Insufficient retained support for applicability',checks=checks)
        aq=np.quantile(af,[.1,.9]);bq=np.quantile(bf,[.1,.9])
        overlap=bool(max(aq[0],bq[0])<=min(aq[1],bq[1]))
        add('source'+str(index)+'_visible_distribution',overlap,dict(current=aq.tolist(),history=bq.tolist()),'overlapping central80-percent ranges',
            'Only retained fine anchors and independently observed coarse fields; rejects unsupported field regimes')
        qa=float(np.isfinite(av).mean());qb=float(np.isfinite(bv).mean())
        add('source'+str(index)+'_coverage',qb>=qa/2,qb,qa/2,'Historical support must retain at least half current finite coverage')
    traces=[]
    for ds,dates in zip([current[1],historical[1]],[wanted,matched]):
        field=ds[variable];values=np.nanmedian(field.values,axis=(1,2));valid=np.isfinite(values)
        if valid.sum()<3:return dict(version='visible_support_v1',status='UNKNOWN',reason='Insufficient coarse temporal support',checks=checks)
        native=_calendar_days(np.asarray(field.time.values,dtype='datetime64[ns]'))[valid]
        target=_calendar_days(dates)
        if target.min()<native.min() or target.max()>native.max():
            return dict(version='visible_support_v1',status='UNKNOWN',reason='Coarse history/current does not bracket required dates',checks=checks)
        traces.append(np.interp(target,native,values[valid]))
    centered=[a-a.mean() for a in traces];norms=[np.linalg.norm(a) for a in centered]
    if min(norms)>np.finfo(float).eps:
        cosine=float(np.dot(*centered)/np.prod(norms))
        add('coarse_seasonal_direction',cosine>=0,cosine,0.,'Reject opposing seasonal coarse dynamics; no hidden fine values used')
    else:
        add('coarse_seasonal_direction',max(norms)<=np.finfo(float).eps,norms,'both constant or both identifiable',
            'Do not infer temporal similarity from a constant/nonconstant pair')
    return dict(version='visible_support_v1',status='PASS' if all(c['status']=='PASS' for c in checks) else 'REJECT',
        checks=checks,gain_prediction='UNKNOWN',threshold_role='Frozen engineering applicability guards, not proof of cross-domain accuracy')


def history_folds(request,context,task_profile,registry_path=None):
    """Build matched-count simulations; hidden fine values only reach the scorer."""
    from ecodata.algorithms.platform_search import SearchInput,SearchValidation
    path=Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY
    receipt=dict(status='UNKNOWN',registry=str(path),cases=[],external_test_used=False,
        independent_confirmation_available=False,evidence_role='development',selection_uses_values=True,
        selection_uses_hidden_fine_values=False)
    if not path.is_file():return [],dict(receipt,reason='No development validation registry published')
    envelope=json.loads(path.read_text(encoding='utf-8'))
    if envelope.get('schema_version')!=1 or not isinstance(envelope.get('registry'),dict):
        raise ValueError('Unsupported development validation registry')
    registry=envelope['registry']
    if envelope.get('sha256')!=digest(registry):raise ValueError('Development registry SHA256 mismatch')
    if registry.get('evidence_role')!='development' or not registry.get('evidence'):
        raise ValueError('Development validation registry needs development evidence')
    if not isinstance(registry.get('cases'),list):raise ValueError('Development cases must be a list')
    receipt['registry_sha256']=envelope['sha256'];receipt['evidence']=registry['evidence']
    if len(context.sources)!=2:return [],dict(receipt,reason='Historical simulations currently require two fine/coarse sources')
    index=task_profile.get('target_source_index',0)
    protected=request.get('protect_observed_source')
    if protected:
        matches=[i for i,b in enumerate(context.sources) if b.dataset.attrs.get('source_id')==protected]
        if len(matches)!=1:return [],dict(receipt,reason='Ambiguous primary source for historical matching')
        index=matches[0]
    if index not in (0,1):return [],dict(receipt,reason='Invalid target source index')
    current=[context.sources[index].dataset,context.sources[1-index].dataset];variable=context.variable
    sensors=[d.attrs.get('sensor') for d in current]
    if any(not s or s=='UNKNOWN' for s in sensors):return [],dict(receipt,reason='Current source sensor identity is unknown')
    attrs=[d[variable].attrs for d in current]
    units=attrs[0].get('units');definition=attrs[0].get('physical_definition')
    if units in (None,'UNKNOWN') or definition in (None,'UNKNOWN') or any(a.get('units')!=units or a.get('physical_definition')!=definition for a in attrs):
        return [],dict(receipt,reason='Current variable physical definition or units are unmatched')
    if any(context.parameters.get(k) is not None for k in ('fitted_state','training_reference','historical_simulation')):
        return [],dict(receipt,reason='Source-specific fitted resources require independent historical-fold provenance')
    wanted=np.asarray(request.get('target_times',current[0].time.values),dtype='datetime64[ns]')
    if wanted.ndim!=1 or not len(wanted) or np.isnat(wanted).any() or (np.diff(wanted)<=np.timedelta64(0,'ns')).any():
        raise ValueError('Increasing unique finite target dates required for historical validation')
    years=set(wanted.astype('datetime64[Y]').astype(int)+1970)
    if len(years)!=1:return [],dict(receipt,reason='Single-year development cases cannot represent a multi-year target')
    a=current[0][variable];observed=a.time.values[np.isfinite(a.values).any(axis=(1,2))]
    visible=np.isin(wanted,observed);hidden=np.flatnonzero(~visible)
    if not len(hidden) or visible.sum()<2:return [],dict(receipt,reason='Need missing target dates and at least two visible anchors')
    target_days=_calendar_days(wanted);folds=[];seen=set()
    for case in registry['cases']:
        identifier=case.get('id');year=case.get('year')
        if not isinstance(identifier,str) or not identifier or identifier in seen or type(year)!=int:
            raise ValueError('Unique development case ID and integer year required')
        seen.add(identifier);detail=dict(case=identifier,year=year,status='UNKNOWN')
        if year in years:
            receipt['cases'].append(dict(detail,reason='Current target year excluded to prevent same-year reference leakage'));continue
        items=case.get('sources',[])
        if len(items)!=2 or {i.get('role') for i in items}!={'primary','background'}:
            raise ValueError('Development case requires one primary and one background resource')
        items=sorted(items,key=lambda i:i['role']!='primary')
        if [i.get('sensor') for i in items]!=sensors:
            receipt['cases'].append(dict(detail,reason='Sensor identities do not match current task'));continue
        loaded=[_read_resource(i,path.parent) for i in items];datasets=[i[0] for i in loaded]
        if any(variable not in d or d[variable].dims!=('time','lat','lon') for d in datasets):
            receipt['cases'].append(dict(detail,reason='Historical timed variable is unavailable'));continue
        if any(d[variable].attrs.get('units')!=units or d[variable].attrs.get('physical_definition')!=definition for d in datasets):
            receipt['cases'].append(dict(detail,reason='Historical units or physical definition do not match'));continue
        datasets=[d.sel(time=d.time.dt.year==year) for d in datasets]
        if any(not d.sizes['time'] for d in datasets):
            receipt['cases'].append(dict(detail,reason='Published year lacks fine or coarse observations'));continue
        fine,coarse=datasets
        times=np.asarray(fine.time.values,dtype='datetime64[ns]')
        if np.isnat(times).any() or (np.diff(times)<=np.timedelta64(0,'ns')).any():
            raise ValueError('Historical fine dates must be increasing and unique')
        available=np.flatnonzero(np.isfinite(fine[variable].values).any(axis=(1,2)))
        if len(available)<len(wanted):
            receipt['cases'].append(dict(detail,reason='Insufficient historical dates for matching anchor and hidden counts',available_dates=len(available)));continue
        chosen=available[_ordered_dates(target_days,_calendar_days(times[available]))]
        full=fine.isel(time=chosen);target_times=[str(t) for t in full.time.values]
        support=applicability(current,datasets,wanted,np.asarray(full.time.values),visible,variable)
        if support['status']!='PASS':
            receipt['cases'].append(dict(detail,status='INAPPLICABLE' if support['status']=='REJECT' else 'UNKNOWN',
                reason='Historical input support is not established',applicability=support));continue
        source_id=full.attrs.get('source_id')
        if not source_id:
            receipt['cases'].append(dict(detail,reason='Historical primary source identity is unknown'));continue
        retained=full.isel(time=np.flatnonzero(visible)).copy(deep=True)
        reference=full[variable].isel(time=hidden).copy(deep=True)
        reference.attrs=dict(reference.attrs,split='search_validation',reference_origin='published development simulation')
        q=dict(request,protect_observed_source=source_id,target_lat=full.lat.values.tolist(),target_lon=full.lon.values.tolist(),target_times=target_times)
        q['parameters']=dict(request.get('parameters',{}),raster_target_times=target_times,reference_support='fixed_finite_reference',require_complete_output=True)
        inputs=[SearchInput(identifier+'_fine.nc','primary',clean(retained).to_netcdf()),SearchInput(identifier+'_coarse.nc','background',clean(coarse).to_netcdf())]
        input_hashes=[hashlib.sha256(i.content).hexdigest() for i in inputs]
        scope=dict(case=identifier,registry_sha256=envelope['sha256'],input_sha256=input_hashes,request=q)
        fold_id=hashlib.sha256(json.dumps(scope,sort_keys=True,default=str).encode()).hexdigest()
        differences=(_calendar_days(full.time.values)-target_days).tolist()
        current_intervals=np.diff(target_days).astype(float)
        historical_intervals=np.diff(_calendar_days(full.time.values)).astype(float)
        def spacing(d):
            return {c:float(abs(np.median(np.diff(d[c].values)))) if len(d[c])>1 else None for c in ('lat','lon')}
        protocol=dict(detail,status='MATCHED_DEVELOPMENT_SIMULATION',fold_id=fold_id,evidence_role='development',applicability=support,
            resource_sha256=[v[1] for v in loaded],input_sha256=input_hashes,sensors=sensors,
            retained_dates=int(visible.sum()),hidden_dates=len(hidden),visible_indices=np.flatnonzero(visible).tolist(),
            hidden_indices=hidden.tolist(),target_times=target_times,calendar_day_differences=differences,
            max_abs_calendar_day_distance=max(abs(v) for v in differences),selection_uses_values=True,
            selection_uses_hidden_fine_values=False,
            current_intervals_days=current_intervals.tolist(),historical_intervals_days=historical_intervals.tolist(),
            max_abs_interval_difference_days=float(abs(historical_intervals-current_intervals).max()),
            current_source_spacing=[spacing(d) for d in current],historical_source_spacing=[spacing(d) for d in datasets],
            retained_finite_fraction=float(np.isfinite(retained[variable]).mean()),
            support_level='APPROXIMATE_HISTORICAL_TRANSFER_NOT_CERTIFIED',
            selection='Minimum total calendar distance under chronological assignment; ties prefer earlier dates',
            geometry=dict(lat=full.lat.values.tolist(),lon=full.lon.values.tolist()),
            external_test_used=False,natural_missing_used_as_truth=False,independent_confirmation_available=False,
            evidence_scope='Matched anchor/hidden counts and nearest calendar dates on another development year; temporal or domain differences remain empirical transfer limits')
        folds.append(dict(id=fold_id,inputs=inputs,validation=SearchValidation(reference,variable),request=q,protocol=protocol,confirmation=None))
        receipt['cases'].append(protocol)
    if folds:receipt.update(status='MATCHED_DEVELOPMENT_SIMULATIONS',fold_count=len(folds),fold_ids=[f['id'] for f in folds])
    else:receipt['reason']='No published development case matches current task with sufficient dates'
    return folds,receipt
