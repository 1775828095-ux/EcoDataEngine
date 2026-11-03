"""Open-registry field DAG exploration; labels remain in the evaluator only."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from itertools import product
import json
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.contracts import Compatibility,AlgorithmResult
from ecodata.algorithms.autonomous_decision import Executor,canonical,algorithm_nodes,clean,round_robin,parameter_variants
from ecodata.algorithms.dag import Artifact
from ecodata.algorithms.numerical_search import save


class Framewise:
    """Apply an existing single-frame implementation independently in time."""
    def __init__(self,plugin):
        self.plugin=plugin;self.frame_calls=0;self.invocations=0
        self.spec=replace(plugin.spec,id='framewise.'+plugin.spec.id,
            input_contract=dict(plugin.spec.input_contract,source_arity=1),
            notes=plugin.spec.notes+'; independent-frame adapter, no new scientific kernel')
    def frame(self,context,index):
        return replace(context,sources=(EcoDataBundle(context.sources[0].dataset.isel(time=index,drop=True)),))
    def compatibility(self,context):
        if len(context.sources)!=1 or 'time' not in context.sources[0].dataset.dims:
            return Compatibility(False,('Independent-frame adapter needs one timed source',))
        return self.plugin.compatibility(self.frame(context,0))
    def run(self,context):
        outputs=[];self.invocations+=1
        from ecodata.algorithms.raster_reconstruction_task import RasterBaselineAdapter,NonnegativeRaster
        if isinstance(self.plugin,NonnegativeRaster):
            self.frame_calls+=1;ds=context.sources[0].dataset.copy(deep=True)
            ds[context.variable]=ds[context.variable].clip(min=0)
            return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(batched_frames=True,reference_used=False))
        if isinstance(self.plugin,RasterBaselineAdapter) and self.plugin.upstream.compatibility(context).eligible:
            # The same upstream regridder already supports a leading time axis.
            # Batch its frames, preserving the adapter's nearest-edge rule.
            self.frame_calls+=1
            ds=self.plugin.upstream.run(context).data.dataset
            values=ds[context.variable].values.copy();missing=~np.isfinite(values)
            if missing.any():
                source=context.sources[0].dataset[context.variable]
                nearest=source.interp(lat=list(context.target_lat),lon=list(context.target_lon),
                    method='nearest',kwargs={'fill_value':'extrapolate'})
                values[missing]=nearest.values[missing]
            ds[context.variable].values=values
            return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(upstream=self.plugin.spec.id,
                batched_frames=True,reference_used=False))
        if self.plugin.compatibility(context).eligible:
            self.frame_calls+=1
            result=self.plugin.run(context)
            return AlgorithmResult(result.data,self.spec.id,dict(result.report,batched_frames=True))
        for i,time in enumerate(context.sources[0].dataset.time.values):
            local=self.frame(context,i);check=self.plugin.compatibility(local)
            if not check.eligible:raise ValueError(str(check.reasons))
            self.frame_calls+=1
            outputs.append(self.plugin.run(local).data.dataset.expand_dims(time=[time]))
        return AlgorithmResult(EcoDataBundle(xr.concat(outputs,dim='time')),self.spec.id,
            dict(upstream=self.plugin.spec.id,frames=len(outputs),reference_used=False))


def field_registry(registry,context=None):
    # Discover adapters from the registered implementations, without method IDs.
    if context is not None and any(context.sources[0].dataset.sizes.get(axis,0)<2 for axis in ('lat','lon')):
        return registry  # A point series is not a 2D spatial field.
    for spec in list(registry.specs()):
        if spec.status=='available':
            plugin=registry.get(spec.id)
            if context is not None:
                try:
                    # A natively timed algorithm needs no duplicate framewise
                    # registration. Failures retain the original implementation.
                    if plugin.compatibility(replace(context,sources=(context.sources[0],))).eligible:continue
                except Exception:pass
            registry.register(Framewise(plugin))
    return registry


def build_context(inputs,request):
    from ecodata.algorithms.platform_search import parse_inputs,make_context
    parsed,audit=parse_inputs(inputs);context,quality,_=make_context(parsed,request,allow_unaligned_time=True)
    if request.get('complementary_observation_operator'):
        from ecodata.algorithms.spatiotemporal_detail import Reconstruction
        sources=context.sources
        if len(sources)!=2:raise ValueError('Complementary observation operator requires two declared sources')
        sizes=[s.dataset.sizes['lat']*s.dataset.sizes['lon'] for s in sources]
        fine=int(np.argmax(sizes));coarse=1-fine
        operator=Reconstruction(sources[fine].dataset,sources[coarse].dataset,request)
        context=replace(context,parameters=dict(context.parameters,reconstruction=operator))
    return context,audit,quality


def measured(dataset,reference,context):
    variable=context.variable
    if variable not in dataset:raise ValueError('Output variable absent')
    partial=context.parameters.get('reference_support')=='fixed_finite_reference'
    mask=np.isfinite(reference.values)
    if not mask.any():raise ValueError('No finite reference support')
    if not mask.all() and not partial:raise ValueError('This field protocol needs a complete finite reference; partial-reference protocol not declared')
    field=dataset[variable]
    bounds=context.parameters.get('target_valid_range')
    if bounds is not None:
        if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:raise ValueError('Invalid explicit target range')
        if bool(((field<bounds[0])|(field>bounds[1])).any()):raise ValueError('Output outside declared application physical range')
    if field.dims!=('time','lat','lon'):raise ValueError('Full timed field required')
    if context.parameters.get('require_complete_output') and not np.isfinite(field.values).all():
        raise ValueError('Complete deployment support required; no partial-period scoring')
    if field.attrs.get('units')!=reference.attrs.get('units'):raise ValueError('Output units differ')
    for name in ('physical_variable','spatial_semantics','temporal_semantics'):
        expected=reference.attrs.get(name)
        if expected in (None,'UNKNOWN') or field.attrs.get(name)!=expected:
            raise ValueError('Output physical contract unknown or changed: '+name)
    field=field.sel(time=reference.time,lat=reference.lat,lon=reference.lon)
    if not np.isfinite(field.values[mask]).all():raise ValueError('Incomplete output on fixed reference support: no subset scoring')
    if context.model_constraints.get('nonnegative') and bool((field<0).any()):raise ValueError('Negative output')
    predicted=np.asarray(field.values,dtype=np.float64)
    expected=np.asarray(reference.values,dtype=np.float64)
    error=np.where(mask,predicted-expected,np.nan)
    if not mask.reshape(len(reference.time),-1).any(axis=1).all():raise ValueError('Every reference date needs finite truth')
    per=np.sqrt(np.nanmean(error**2,axis=(1,2)));windows={}
    if context.parameters.get('objective_scope')=='calendar_year_RMSE':
        years=pd.DatetimeIndex(reference.time.values).year
        windows={str(year):float(np.sqrt(np.nanmean(error[years==year]**2))) for year in np.unique(years)}
    gradient=np.diff(predicted,axis=2)-np.diff(expected,axis=2)
    return dict(objective=float(np.mean(list(windows.values()))) if windows else float(np.sqrt(np.nanmean(error**2))),RMSE=float(np.sqrt(np.nanmean(error**2))),
        MAE=float(np.nanmean(abs(error))),Bias=float(np.nanmean(error)),frame_RMSE=per.tolist(),denominator=int(mask.sum()),
        window_RMSE=windows,metric_definition='RMSE=sqrt(mean(error**2)) on all finite reference cells; frame_RMSE is diagnostic only',
        gradient_RMSE=float(np.sqrt(np.nanmean(gradient**2))) if field.sizes['lon']>1 else None)


def structural_distance(ds,context,request):
    if context.variable not in ds:return 4
    return sum([not np.array_equal(ds.lat,context.target_lat),not np.array_equal(ds.lon,context.target_lon),
        not pd.DatetimeIndex(request['target_times']).isin(pd.DatetimeIndex(ds.time.values)).all()])


def submit(inputs,request,evaluation,folder,*,budget,seed=11,validation=None,progress=None):
    from ecodata.algorithms.platform_search import platform_registry,parse_inputs,SearchInput,SearchValidation
    if 'composition' in request:raise ValueError('No hand-authored pool in AutonomousField')
    if request.get('task')!='multisource':raise ValueError('AutonomousField requires explicit multisource task')
    deployment_context=None;internal_holdout=None
    if validation is None:
        parsed,_=parse_inputs(inputs)
        candidates=[(i,ds) for i,(role,ds) in enumerate(parsed) if role in {'primary','background'}]
        fine_index,fine=max(candidates,key=lambda item:item[1].sizes.get('lat',0)*item[1].sizes.get('lon',0))
        if fine.sizes.get('time',0)<3:raise ValueError('Need at least three visible fine anchors or explicit search_validation; accuracy unknown')
        middle=fine.sizes['time']//2
        internal_holdout=fine.isel(time=[middle]).copy(deep=True)
        internal_holdout[request['variable']].attrs['split']='search_validation'
        validation=SearchValidation(internal_holdout[request['variable']],request['variable'])
        search_fine=fine.isel(time=[i for i in range(fine.sizes['time']) if i!=middle])
        deployment_context,_,_=build_context(inputs,request)
        inputs=list(inputs);item=inputs[fine_index]
        inputs[fine_index]=SearchInput(item.filename,item.role,search_fine.to_netcdf(),item.metadata)
    context,audit,quality=build_context(inputs,request);registry=field_registry(platform_registry(),context)
    reference=validation.reference
    objective_name='Mean calendar-year hidden-site RMSE' if context.parameters.get('objective_scope')=='calendar_year_RMSE' else 'RMSE on all finite reference cells'
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'request.json').exists():raise ValueError('Immutable run already exists')
    save(folder/'request.json',dict(request=request,inputs=audit,quality=quality,budget=budget,seed=seed,
        validation=validation.protocol,objective=objective_name,handwritten_pool=False,max_depth=None))
    save(folder/'catalog.json',[s.as_dict() for s in registry.specs()])
    if internal_holdout is not None:clean(internal_holdout).to_netcdf(folder/'internal_validation.nc',engine='h5netcdf')
    records={};states={};frontier=[];seen={};rows=[];attempts=0;calls=0;started=perf_counter()
    engine=Executor(replace(context,validation_reference=None,validation_role=None),registry)
    for i,source in enumerate(context.sources):
        key='s'+str(i);records[key]=dict(operation='source',parents=[],parameters=dict(source='source'+str(i)))
        states[key]=source.dataset;frontier.append(key)
        seen[Artifact('state',source).fingerprint()]=key
    available=[s for s in registry.specs() if s.status=='available']
    executed=0;best=None;best_single=None;stop='RESOURCE_BUDGET_EXHAUSTED';rng=np.random.default_rng(seed)
    tried=set();layers=0
    search_policy=request.get('search_policy','contract_guided')
    if search_policy not in {'contract_guided','random'}:raise ValueError('Unknown exploration policy')
    def exhausted():return executed>=budget['objective_limit'] or attempts>=budget['proposal_limit'] or perf_counter()-started>=budget['wall_seconds']
    def parameters(spec,parents):
        # Task frequencies and native registered defaults are retained. Fusion's
        # bounded numeric domain is sampled by resource policy, not a plan list.
        plugin=registry.get(spec.id)
        if hasattr(plugin,'parameter_proposals'):
            local=replace(context,sources=tuple(EcoDataBundle(states[p]) for p in parents))
            return plugin.parameter_proposals(local,rng)
        if 'parameter_domains' in spec.input_contract:
            return parameter_variants(spec,[states[p] for p in parents],context.variable)
        return [{}]
    ledger=(folder/'attempts.jsonl').open('x',encoding='utf-8')
    try:
        while frontier and not exhausted():
            layers+=1;new=[];keys=list(states)
            distances={k:structural_distance(states[k],context,request) for k in keys}
            node_counts={k:algorithm_nodes(records,k) for k in keys}
            # Every discovered state gets an expansion turn. Coordinate-complete
            # parents first is task-contract guidance, not target-value access.
            frontier.sort(key=lambda k:(distances[k],node_counts[k],k))
            if search_policy=='random':rng.shuffle(frontier)
            producer_order=list(available)
            if search_policy=='random':rng.shuffle(producer_order);rng.shuffle(keys)
            partner_orders={}
            for parent in frontier:
                origin=states[parent].attrs.get('source_id')
                partner_orders[parent]=sorted(keys,key=lambda k:(states[k].attrs.get('source_id')==origin,
                    distances[k],node_counts[k],k)) if search_policy!='random' else list(keys)
            def producer(parent,count,spec):
                arity=spec.input_contract.get('source_arity')
                if arity is not None and arity!=count:return
                for others in product(partner_orders[parent],repeat=count-1):
                    if parent in others:continue
                    yield spec,(parent,*others)
            def proposals_for(parent,count):
                yield from round_robin([producer(parent,count,spec) for spec in producer_order])
            streams=[proposals_for(parent,count) for count in range(1,layers+2) for parent in frontier]
            for spec,parents in round_robin(streams):
                    if exhausted():break
                    for param in parameters(spec,parents):
                        if exhausted():break
                        token=canonical([spec.id,parents,param])
                        if token in tried:continue
                        tried.add(token);attempts+=1
                        record=dict(algorithm=spec.id,parents=list(parents),parameters=param)
                        local=replace(context,sources=tuple(EcoDataBundle(states[p]) for p in parents),
                            parameters=dict(context.parameters,**param),validation_reference=None,validation_role=None)
                        try:
                            check=registry.get(spec.id).compatibility(local)
                            if not check.eligible:
                                record.update(status='INCOMPATIBLE',reasons=list(check.reasons));ledger.write(json.dumps(record,default=str)+'\n');continue
                        except Exception as exc:
                            record.update(status='CONTRACT_ERROR',error=str(exc));ledger.write(json.dumps(record,default=str)+'\n');continue
                        key='g'+str(executed);executed+=1
                        records[key]=dict(operation=spec.id,parents=list(parents),parameters=param)
                        record.update(key=key,algorithm_nodes=algorithm_nodes(records,key))
                        try:
                            output,trace=engine.execute(records,key)
                            record.update(status='INTERMEDIATE_ONLY',trace=trace)
                            # Guard unusable intermediate explosion; no raw
                            # target values or statistics enter this gate.
                            if context.variable not in output or output[context.variable].size==0:raise ValueError('Empty numeric state')
                            fingerprint=Artifact('state',EcoDataBundle(output)).fingerprint()
                            if fingerprint not in seen:
                                seen[fingerprint]=key;states[key]=output;new.append(key)
                            else:record['equivalent_state']=seen[fingerprint]
                            try:
                                record['metrics']=measured(output,reference,context);record['status']='ELIGIBLE_OUTPUT'
                                rank=(record['metrics']['objective'],record['algorithm_nodes'],key)
                                if best is None or rank<(best['metrics']['objective'],best['algorithm_nodes'],best['key']):best=record
                                if record['algorithm_nodes']==1 and (best_single is None or rank<(best_single['metrics']['objective'],1,best_single['key'])):best_single=record
                            except (ValueError,KeyError,IndexError) as exc:record['terminal_reason']=str(exc)
                        except Exception as exc:record.update(status='EXECUTION_FAILED',error=str(exc))
                        rows.append(record);ledger.write(json.dumps(record,default=str)+'\n');ledger.flush()
                        if progress:progress(dict(candidate=key,status=record['status'],attempts=attempts))
            frontier=new
        if not frontier and not exhausted():stop='OBSERVED_FRONTIER_CLOSED_DEFAULT_PARAMETERS_ONLY'
    finally:ledger.close()
    product_path=None;independent_evaluation=None
    deployment_calls=0
    if best:
        deploy=Executor(deployment_context,registry) if deployment_context is not None else engine
        output,_=deploy.execute(records,best['key'])
        deployment_calls=deploy.calls if deployment_context is not None else 0
        protected=request.get('protect_observed_source')
        if protected:
            roots=(deployment_context or context).sources
            matches=[s.dataset for s in roots if s.dataset.attrs.get('source_id')==protected]
            if len(matches)!=1:raise ValueError('Observed-value protection source is unknown or ambiguous')
            original=matches[0][context.variable];xr.align(output[context.variable],original,join='exact')
            output[context.variable]=xr.where(np.isfinite(original),original,output[context.variable])
            output[context.variable].attrs=dict(original.attrs)
            output.attrs.update(protected_observed_source=protected,fusion_method='observations_preserved_reconstruction',
                protected_visible_values=int(np.isfinite(original).sum()))
        product_path=folder/'product.nc'
        clean(output).to_netcdf(product_path,engine='h5netcdf')
        with xr.open_dataset(product_path) as f:xr.testing.assert_allclose(clean(output),f.load())
        if evaluation is not None:independent_evaluation=evaluation(output)
    save(folder/'graphs.json',records)
    wrappers=[registry.get(s.id) for s in available if isinstance(registry.get(s.id),Framewise)]
    frame_calls=sum(p.frame_calls for p in wrappers);wrapper_calls=sum(p.invocations for p in wrappers)
    result=dict(service='PlatformSearch/AutonomousField',status='PASS' if best else 'NO_ELIGIBLE_RESULT',
        registry_count=len(registry.specs()),available_count=len(available),attempts=attempts,executed_candidates=executed,
        kernel_calls=engine.calls+deployment_calls-wrapper_calls+frame_calls,adapter_invocations=engine.calls+deployment_calls,
        frame_kernel_invocations=frame_calls,best=best,best_single=best_single,product=str(product_path) if product_path else None,
        independent_evaluation=independent_evaluation,
        stop_reason=stop,max_depth=None,handwritten_pool=False,complete_space=False,
        certification='NOT_CERTIFIED_CURRENT_BEST_ONLY',source_count=len(context.sources),layers=layers,
        failed_candidates=sum(r['status']=='EXECUTION_FAILED' for r in rows),unique_states=len(states),
        elapsed_seconds=perf_counter()-started,objective=objective_name,test_used_for_selection=False,search_policy=search_policy,
        self_validation_from_visible_anchors=internal_holdout is not None,search_adapter_calls=engine.calls,deployment_adapter_calls=deployment_calls,
        limitations=['Unspecified parameter domains tested at registered defaults; declared fusion domains sampled under budget',
            'Observed-state deduplication is heuristic, not full graph equivalence','No native downstream-model response validated'])
    save(folder/'summary.json',result)
    return result
