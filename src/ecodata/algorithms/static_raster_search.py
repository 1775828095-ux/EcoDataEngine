"""Static rasters -> existing graph service using a declared reduced-scale proxy.

No synthetic timestamp, new kernel, local optimizer or independent accuracy claim.
"""
from pathlib import Path
import numpy as np
import xarray as xr
from ecodata.algorithms.numerical_search import save
from ecodata.algorithms.autonomous_decision import clean,Executor
from ecodata.algorithms.raster_reconstruction_task import edges,overlap_weights


def submit(inputs,request,evaluation,folder,*,budget,seed,progress=None,validation=None):
    from ecodata.algorithms.platform_search import (parse_inputs,make_context,SearchInput,
        RasterSearchValidation,platform_registry)
    from ecodata.algorithms.field_mutation import submit as search
    from ecodata.algorithms.autonomous_field import field_registry
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    parsed,audit=parse_inputs(inputs)
    parsed=[(role,ds.sortby('lat').sortby('lon')) for role,ds in parsed]
    context,quality,_=make_context(parsed,request,allow_unaligned_time=True)
    variable=context.variable;primary=context.sources[0].dataset
    if context.semantic_type!='continuous':raise ValueError('Static proxy requires continuous rasters; categorical averaging is forbidden')
    if min(primary.sizes['lat'],primary.sizes['lon'])<4:raise ValueError('Static spatial validation requires at least four real cells per axis')
    if any(not np.array_equal(s.dataset[axis],primary[axis]) for s in context.sources for axis in ('lat','lon')):
        raise ValueError('Static proxy requires sources aligned by their declared spatial support first')
    for axis in ('lat','lon'):
        target=request.get('target_'+axis)
        if target is not None and not np.array_equal(np.asarray(target),primary[axis].values):
            raise ValueError('Static reduced-scale validation needs a reference on the declared target grid; target is never silently replaced')
    child=dict(request,adapter='static_raster_execution',soft_guidance=False,
        target_lat=primary.lat.values.tolist(),target_lon=primary.lon.values.tolist())
    child.pop('processing_goal',None)
    if not child.get('model_profile') and not child.get('application_contract'):
        child['application_contract']=dict(id='static-raster',version='1',field=variable,unit=request['unit'],
            semantic_type='continuous',valid_range=None,issues=[],evidence='Explicit static raster unit, geometry and physical variable; no observation time inferred')
    reference=None;prepared=[];operators={}
    if validation is None:
        for axis in ('lat','lon'):
            fine=edges(primary[axis].values);coarse=np.r_[fine[::2],fine[-1]] if (len(fine)-1)%2 else fine[::2]
            centres=(coarse[1:]+coarse[:-1])/2
            operators[axis]=(centres,overlap_weights(fine,coarse,axis=='lat').toarray())
        ay=operators['lat'][1];ax=operators['lon'][1]
        for item,(role,ds) in zip(inputs,parsed):
            if role not in {'primary','background'}:raise ValueError('Static proxy auxiliary/state resources need their own compatible degradation contract')
            values=np.asarray(ds[variable].transpose('lat','lon').values,float)
            if not np.isfinite(values).all():raise ValueError('Static reduced-scale proxy needs finite native cells; natural missing values are not reference truth')
            reduced=ay@values@ax.T
            coarse=xr.Dataset({variable:(('lat','lon'),reduced,dict(ds[variable].attrs))},
                coords={axis:operators[axis][0] for axis in ('lat','lon')},attrs=dict(ds.attrs))
            prepared.append(SearchInput(item.filename,item.role,bytes(clean(coarse).to_netcdf(engine='h5netcdf'))))
        reference=primary[variable].copy(deep=True);reference.attrs['split']='search_validation'
        validation=RasterSearchValidation(reference,variable)
        validation.protocol.update(reference_scope='Uploaded native product at reduced scale; NOT independent truth',accuracy_claim=False)
        protocol=dict(kind='static_reduced_resolution_proxy',operator='Existing latitude-area/longitude overlap weights; 2-cell intervals',
            synthetic_timestamp=False,reference_leaves_prediction_input=True,independent_confirmation=False,
            operators={axis:dict(coarse_centres=centres.tolist(),weights=weights.tolist()) for axis,(centres,weights) in operators.items()})
    else:
        prepared=list(inputs)
        validation=RasterSearchValidation(validation.reference,variable)
        protocol=dict(kind='explicit_static_validation',synthetic_timestamp=False,independent_confirmation=False)
    save(folder/'static_protocol.json',dict(protocol,inputs=audit,quality=quality,budget=budget,seed=seed))
    result=search(prepared,child,evaluation,folder/'search',budget=budget,seed=seed,progress=progress,validation=validation)
    result['static_protocol']=protocol
    result['confirmed_initial']=[]
    if result.get('best'):
        frozen=result.get('selected_graph') or result.get('graph')
        if frozen is None:
            record=__import__('json').loads((folder/'search/selection_freeze.json').read_text(encoding='utf8'))
            frozen=dict(key=record['key'],records=record['graph'])
        result['selected_graph']=frozen
        # Transfer is a separate execution; never evaluate the undegraded primary
        # against itself and call that evidence of fusion improvement.
        registry=field_registry(platform_registry(),context)
        try:
            executor=Executor(context,registry)
            candidate,trace=executor.execute(frozen['records'],frozen['key'])
            path=folder/'candidate.nc';clean(candidate).to_netcdf(path,engine='h5netcdf')
            result.update(candidate_product=str(path),candidate_trial_available=True,transfer_trace=trace,
                          transfer_kernel_calls=executor.calls)
        except (ValueError,RuntimeError,KeyError) as exc:
            result['transfer_failure']=str(exc)
        baseline=folder/'product.nc';clean(primary).to_netcdf(baseline,engine='h5netcdf')
        with xr.open_dataset(baseline) as restored:xr.testing.assert_allclose(primary,restored.load())
        result.update(search_status=result['status'],status='PASS',product=str(baseline),delivery_status='BASELINE_RETAINED',
            post_mutation_acceptance=dict(status='PROXY_ONLY_NO_INDEPENDENT_CONFIRMATION',recommended=False,
                reason='Static product reconstruction is not independent environmental accuracy evidence'))
    save(folder/'summary.json',result)
    return result
