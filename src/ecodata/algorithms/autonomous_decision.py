"""Registry-driven DAG exploration with one measured reconstruction objective.

No hand-authored plan pool or maximum depth. Resource exhaustion is not a
global-optimality certificate. Existing plugins and the numerical DAG executor
own all scientific computation.
"""
from dataclasses import replace
from collections import deque
from itertools import product
from pathlib import Path
from time import perf_counter
import hashlib,json

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.dag import Artifact,Port,NodeAdapter,Node,NumericalDAGExecutor
from ecodata.algorithms.daily_reconstruction import gap_lengths
from ecodata.algorithms.numerical_search import save
from ecodata.core.schemas import EcoDataBundle

CONTRACT='autonomous_registry_reconstruction_v1'


def round_robin(streams):
    active=deque(iter(s) for s in streams)
    while active:
        stream=active.popleft()
        try:value=next(stream)
        except StopIteration:continue
        active.append(stream)
        yield value


def algorithm_nodes(records,sink):
    visited=set()
    def visit(key):
        if key in visited:return
        visited.add(key)
        for parent in records[key]['parents']:visit(parent)
    visit(sink)
    return sum(records[key]['operation']!='source' for key in visited)


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def positive_compositions(total,count):
    if count==1:
        if total>=1:yield (total,)
    else:
        for head in range(1,total-count+2):
            for rest in positive_compositions(total-head,count-1):yield (head,*rest)


def clean(dataset):
    ds=dataset.copy(deep=True)
    for obj in [ds,*ds.variables.values()]:
        obj.attrs={k:(str(v) if isinstance(v,(dict,bool)) else v) for k,v in obj.attrs.items() if v is not None}
    return ds


class RecoveryAssessment:
    """Self-supervision over genuinely visible observations, not unknown truth."""
    def __init__(self,context):
        ds=context.sources[0].dataset;var=context.variable
        if ds[var].dims!=('time','lat','lon') or ds.sizes['lat']!=1 or ds.sizes['lon']!=1:
            raise ValueError('Automatic measured assessment currently requires a single timed station; no substitute score for other tasks')
        time=pd.DatetimeIndex(ds.time.values)
        if len(time)<4 or not np.all(np.diff(time.values)==np.timedelta64(1,'D')):
            raise ValueError('Automatic daily assessment needs regular daily observations')
        periods=time.to_period('Y')
        if len(periods.unique())<2:periods=time.to_period('M')
        groups=list(periods.unique())
        if len(groups)<2:
            raise ValueError('Need at least two independent calendar windows; accuracy remains unassessed')
        self.variable=var;self.search=[];self.confirm=[]
        for group in groups:
            part=ds.isel(time=np.flatnonzero(periods==group)).copy(deep=True)
            values=part[var].values[:,0,0];visible=np.isfinite(values)
            if visible.sum()<4:continue
            lengths=gap_lengths(values);observed=lengths[lengths>0]
            choices,counts=np.unique(observed if len(observed) else [1],return_counts=True)
            for seed in (1701,2701,3701):
                rng=np.random.default_rng(seed);mask=np.zeros(len(values),bool)
                for begin in rng.permutation(np.arange(1,len(values)-1)):
                    size=int(rng.choice(choices,p=counts/counts.sum()));end=begin+size
                    if end>=len(values) or not visible[begin-1:end+1].all() or mask[max(0,begin-1):end+1].any():continue
                    mask[begin:end]=True
                    if mask.sum()>=max(1,int(visible.sum()*.1)):break
                if not mask.any():continue
                truth=values[mask].copy();part_masked=part.copy(deep=True)
                part_masked[var].values[mask]=np.nan
                local=replace(context,sources=(EcoDataBundle(part_masked),),validation_reference=None,validation_role=None)
                panel=dict(group=str(group),seed=seed,context=local,mask=mask,truth=truth)
                (self.confirm if group==groups[-1] else self.search).append(panel)
        if not self.search or not self.confirm:
            raise ValueError('No finite search/confirmation references; do not invent accuracy')
        self.protocol=dict(kind='visible_reconstruction',objective='mean_calendar_window_RMSE',
            search_groups=sorted({p['group'] for p in self.search}),
            confirmation_groups=sorted({p['group'] for p in self.confirm}),
            seeds=[1701,2701,3701],cohort='All candidates share frozen masks; only originally finite values are references',
            confirmation_rule='Each confirmation panel no worse and mean strictly better than search-frozen best single algorithm; otherwise retain baseline',
            model_sensitivity='UNKNOWN',unknown_true_error='NOT_ASSESSED')

    def terminal(self,dataset,panel):
        var=self.variable;root=panel['context'].sources[0].dataset
        if var not in dataset:raise ValueError('Output variable absent')
        actual=dataset[var];original=root[var]
        if actual.dims!=original.dims or any(not np.array_equal(actual[c],original[c]) for c in original.dims):
            raise ValueError('Output support differs from the declared daily task')
        if actual.attrs.get('units')!=original.attrs.get('units'):
            raise ValueError('Output unit is unknown or changed without conversion')
        if actual.attrs.get('temporal_semantics')!=original.attrs.get('temporal_semantics'):
            raise ValueError('Daily interval meaning changed')
        output=dataset.copy(deep=True);a=output[var].values
        known=np.isfinite(original.values);a[known]=original.values[known]
        if not np.isfinite(a).all():raise ValueError('Incomplete output; no successful-subset scoring')
        if panel['context'].model_constraints.get('nonnegative') and (a<0).any():
            raise ValueError('Negative output violates explicit task contract')
        return output

    def metrics(self,outputs,panels=None):
        panels=self.search if panels is None else panels;rows=[]
        for ds,panel in zip(outputs,panels):
            product=self.terminal(ds,panel);pred=product[self.variable].values[:,0,0][panel['mask']]
            error=pred-panel['truth'];rows.append(dict(group=panel['group'],seed=panel['seed'],
                RMSE=float(np.sqrt(np.mean(error*error))),MAE=float(abs(error).mean()),Bias=float(error.mean()),
                n=len(error),prediction=pred.tolist()))
        windows=pd.DataFrame(rows).groupby('group').RMSE.mean()
        return dict(objective=float(windows.mean()),RMSE=float(windows.mean()),
            window_RMSE=windows.to_dict(),window_std=float(windows.std(ddof=0)),panels=rows,
            score_basis='Measured equal calendar-window mean RMSE; no portrait weights')


def protect_observed(output,source,variable):
    """Preserve finite visible anchors even when target dates include gaps."""
    for axis in ('lat','lon'):
        if not np.array_equal(source[axis],output[axis]):raise ValueError('Observed protection needs the identical spatial grid')
    if not pd.DatetimeIndex(source.time.values).isin(output.time.values).all():raise ValueError('Visible dates absent from deployment targets')
    visible=source[variable].reindex(time=output.time)
    result=output.copy()
    result[variable]=xr.where(np.isfinite(visible),visible,output[variable])
    result[variable].attrs=dict(source[variable].attrs)
    result.attrs.update(protected_observed_source=source.attrs.get('source_id'),fusion_method='observations_preserved_reconstruction')
    return result


class Executor:
    """One cache per immutable cohort; adapters never receive reference values."""
    def __init__(self,context,registry,*,account=None,phase='search'):
        self.context=context;self.registry=registry;self.adapters={};self.calls=0
        self.account=account;self.phase=phase
        def source(_inputs,params,external):return {'field':external[params['source']]}
        self.adapters['source','run']=NodeAdapter({}, {'field':Port('working_field',('contract',))},source)
        self.engine=NumericalDAGExecutor(self.adapters,version=CONTRACT)

    def execute(self,records,sink):
        self.current_sink=sink
        for row in records.values():
            if row['operation']=='source':continue
            identifier=row['operation'];arity=len(row['parents']);mode='arity_'+str(arity)
            if (identifier,mode) in self.adapters:continue
            def compute(inputs,parameters,external,identifier=identifier,arity=arity):
                context=replace(self.context,sources=tuple(inputs['p'+str(i)].value for i in range(arity)),
                    parameters=dict(self.context.parameters,**{k:v for k,v in parameters.items() if k!='mode'}),
                    validation_reference=None,validation_role=None)
                plugin=self.registry.get(identifier);check=plugin.compatibility(context)
                if not check.eligible:raise ValueError('Actual parent contract: '+str(check.reasons))
                token=self.account.before(identifier,self.phase,parameters,graph=self.current_sink,requires_training=plugin.spec.requires_training) if self.account is not None else None
                self.calls+=1
                try:
                    result=plugin.run(context)
                except Exception as exc:
                    if self.account is not None:self.account.after(token,error=exc)
                    raise
                else:
                    if self.account is not None:self.account.after(token)
                dataset=result.data.dataset
                root=self.context.sources[0].dataset
                if ('time_bounds' in root and 'time_bounds' in dataset and
                        'time' in dataset.coords and dataset.time_bounds.shape==(dataset.sizes['time'],2)):
                    for anchor in (0,1):
                        if np.array_equal(root.time.values,root.time_bounds.values[:,anchor]):
                            labels=dataset.time_bounds.values[:,anchor]
                            if not np.array_equal(dataset.time.values,labels):
                                dataset=dataset.assign_coords(time=labels)
                                result=replace(result,data=EcoDataBundle(dataset,result.data.history),
                                    report=dict(result.report,interval_label_anchor=anchor))
                            break
                if context.variable in dataset:
                    field=dataset[context.variable]
                    if set(field.dims)=={'time','lat','lon'} and field.dims!=('time','lat','lon'):
                        dataset=dataset.copy();dataset[context.variable]=field.transpose('time','lat','lon')
                        result=replace(result,data=EcoDataBundle(dataset,result.data.history),
                            report=dict(result.report,axis_order_normalized='time,lat,lon; values unchanged'))
                return {'field':Artifact('working_field',result.data,dict(contract=CONTRACT,algorithm=identifier,report=result.report))}
            self.adapters[identifier,mode]=NodeAdapter(
                {'p'+str(i):Port('working_field',('contract',)) for i in range(arity)},
                {'field':Port('working_field',('contract',))},compute)
            self.engine.adapters[identifier,mode]=self.adapters[identifier,mode]
        nodes=[];visited=set()
        def visit(key):
            if key in visited:return
            row=records[key]
            for parent in row['parents']:visit(parent)
            params=dict(row['parameters'])
            if row['operation']!='source':params['mode']='arity_'+str(len(row['parents']))
            nodes.append(Node(key,row['operation'],inputs={'p'+str(i):(p,'field') for i,p in enumerate(row['parents'])},parameters=params))
            visited.add(key)
        visit(sink)
        external={'source'+str(i):Artifact('working_field',source,dict(contract=CONTRACT))
                  for i,source in enumerate(self.context.sources)}
        result=self.engine.run(nodes,sink,external=external)
        return result['outputs']['field'].value.dataset,result['trace']


def parameter_variants(spec,parents,variable):
    domains=spec.input_contract.get('parameter_domains',{});choices={}
    for key,value in domains.items():
        if isinstance(value,list):choices[key]=value
        elif isinstance(value,dict) and value.get('source')=='observed_missing_run_lengths':
            lengths=set()
            for ds in parents:
                if variable in ds and ds[variable].dims==('time','lat','lon'):
                    lengths.update(int(x) for x in gap_lengths(ds[variable].values[:,0,0]) if x>0)
            choices[key]=[None,*sorted(lengths)]
        else:raise ValueError('Parameter domain not executable: '+key)
    for values in product(*(choices[k] for k in choices)):
        yield {k:v for k,v in zip(choices,values) if v is not None}


def submit(inputs,request,evaluation,folder,*,budget,seed=11,progress=None):
    from ecodata.algorithms.platform_search import parse_inputs,platform_registry,make_context
    from ecodata.algorithms.profile_matching import prepare_model_request,validate_model_context
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);started=perf_counter()
    if any((folder/n).exists() for n in ('request.json','summary.json')):raise ValueError('Immutable autonomous request already exists')
    if 'composition' in request:raise ValueError('Autonomous mode forbids hand-authored composition pools')
    parsed,audit=parse_inputs(inputs);registry=platform_registry()
    req=dict(request)
    if 'model_profile' in request:req,model=prepare_model_request(request)
    else:model=None
    context,quality,_=make_context(parsed,req)
    if model is not None:validate_model_context(context,model)
    if len(context.sources)!=1:
        raise ValueError('Automatic recovery assessment needs one observed source; no fabricated multisource reference')
    assessment=RecoveryAssessment(context)
    specs=registry.specs();catalog=[dict(spec=s.as_dict(),used_for_screening='actual compatibility only',
        parameter_coverage='DECLARED' if 'parameter_domains' in s.input_contract else 'DEFAULT_ONLY_UNDOCUMENTED_DOMAIN',
        source_arity=s.input_contract.get('source_arity','UNDECLARED_EXPLORED_WITH_RESOURCE_BUDGET')) for s in specs]
    save(folder/'catalog.json',catalog)
    for i,p in enumerate(assessment.search+assessment.confirm):
        p['context'].sources[0].dataset.to_netcdf(folder/('cohort_'+str(i)+'.nc'),engine='h5netcdf')
    save(folder/'request.json',dict(version=CONTRACT,request=request,inputs=audit,quality=quality,model=model,
        protocol=assessment.protocol,budget=budget,algorithm_count=len(specs),handwritten_pool=False,
        max_depth=None,max_algorithms=None,confirmation_not_visible_to_search=True,soft_portrait_score_used=False))
    source_record=dict(operation='source',parents=[],parameters=dict(source='source0'))
    records={'s0':source_record};levels={1:['s0']}
    contexts=[p['context'] for p in assessment.search];executors=[Executor(c,registry) for c in contexts]
    states={'s0':[c.sources[0].dataset for c in contexts]};rows=[];aliases=[];seen={}
    attempts=0;candidate_count=0;depth=2;stop=None;best=None;baseline=None;unresolved_contract=False
    limits=dict(wall_seconds=float(budget.get('wall_seconds',180)),proposals=int(budget.get('proposal_limit',3000)),
        executions=int(budget.get('objective_limit',80)))
    if any(v<=0 for v in limits.values()):raise ValueError('Positive resource budgets required')
    ledger=(folder/'attempts.jsonl').open('x',encoding='utf-8')
    def exhausted():
        return perf_counter()-started>=limits['wall_seconds'] or attempts>=limits['proposals'] or candidate_count>=limits['executions']
    def log(row):ledger.write(json.dumps(row,ensure_ascii=False,default=str)+'\n');ledger.flush()
    available=[s for s in specs if s.status=='available']
    declared=bool(available) and all(isinstance(s.input_contract.get('source_arity'),int) and s.input_contract['source_arity']>=1
        and 'parameter_domains' in s.input_contract and s.input_contract.get('deterministic') is True
        and s.input_contract.get('state_complete') is True for s in available)
    def actions(arity,size):
        def producer(spec):
            declared_arity=spec.input_contract.get('source_arity')
            if isinstance(declared_arity,int) and arity!=declared_arity:return
            for sizes in positive_compositions(size-1,arity):
                if any(s not in levels for s in sizes):continue
                for parents in product(*(list(levels[s]) for s in sizes)):
                    parent_data=[states[key][panel] for key in parents for panel in range(len(contexts))]
                    try:
                        for parameters in parameter_variants(spec,parent_data,context.variable):
                            yield spec,parents,parameters,None
                    except Exception as exc:yield spec,parents,{},str(exc)
        yield from round_robin([producer(spec) for spec in available])
    try:
        while not exhausted():
            if not available:stop='NO_AVAILABLE_ALGORITHMS';break
            for spec,parents,parameters,error in round_robin([actions(arity,depth) for arity in range(1,depth)]):
                if exhausted():break
                if error is not None:
                    unresolved_contract=True
                    attempts+=1;log(dict(algorithm=spec.id,parents=parents,status='DOMAIN_UNKNOWN',error=error));continue
                attempts+=1;local_context=replace(contexts[0],sources=tuple(EcoDataBundle(states[p][0]) for p in parents),parameters=dict(contexts[0].parameters,**parameters))
                try:
                    check=registry.get(spec.id).compatibility(local_context)
                    if not check.eligible:
                        log(dict(algorithm=spec.id,parents=parents,parameters=parameters,status='INCOMPATIBLE',reasons=list(check.reasons)));continue
                except Exception as exc:
                    unresolved_contract=True
                    log(dict(algorithm=spec.id,parents=parents,parameters=parameters,status='CONTRACT_ERROR',error=str(exc)));continue
                candidate_count+=1;key='g'+str(candidate_count)
                records[key]=dict(operation=spec.id,parents=list(parents),parameters=parameters)
                row=dict(key=key,algorithm=spec.id,parents=list(parents),parameters=parameters,tree_nodes=depth,algorithm_nodes=algorithm_nodes(records,key))
                outputs=[];traces=[];begin=perf_counter()
                try:
                    for engine in executors:
                        ds,trace=engine.execute(records,key);outputs.append(ds);traces.append(trace)
                    row['execution_status']='PASS'
                    try:row['metrics']=assessment.metrics(outputs);row['status']='ELIGIBLE_OUTPUT'
                    except Exception as exc:row.update(status='INTERMEDIATE_ONLY',terminal_reason=str(exc))
                    fingerprint=canonical([Artifact('state',EcoDataBundle(ds)).fingerprint() for ds in outputs])
                    equivalent=seen.get(fingerprint)
                    if equivalent is None:
                        seen[fingerprint]=key;states[key]=outputs;levels.setdefault(depth,[]).append(key)
                    else:row['equivalent_state']=equivalent;aliases.append(dict(alias=key,representative=equivalent))
                    if row['status']=='ELIGIBLE_OUTPUT':
                        ranking=(row['metrics']['objective'],row['algorithm_nodes'],key)
                        if best is None or ranking<(best['metrics']['objective'],best['algorithm_nodes'],best['key']):best=row
                        if all(p=='s0' for p in parents) and (baseline is None or ranking<(baseline['metrics']['objective'],baseline['algorithm_nodes'],baseline['key'])):baseline=row
                except Exception as exc:row.update(status='EXECUTION_FAILED',error=str(exc),trace=getattr(exc,'trace',[]))
                row.update(seconds=perf_counter()-begin,panel_traces=traces);rows.append(row);log(row)
                save(folder/'candidates'/(key+'.json'),row)
                if progress:progress(dict(candidate=key,status=row['status'],attempts=attempts))
            if exhausted():stop='RESOURCE_BUDGET_EXHAUSTED';break
            maximum=max(levels)
            if declared and depth>=max(s.input_contract['source_arity'] for s in available)*maximum+1:
                stop='FINITE_STATE_SPACE_CLOSED';break
            depth+=1
    finally:ledger.close()
    global_certificate=bool(stop=='FINITE_STATE_SPACE_CLOSED' and declared and not unresolved_contract)
    chosen=best;confirmation=[];confirmation_calls=0;deployment_calls=0
    if best is not None:
        for label,row in [('empirical_best',best),('best_single',baseline)]:
            if row is None:continue
            outputs=[]
            try:
                for panel in assessment.confirm:
                    engine=Executor(panel['context'],registry)
                    outputs.append(engine.execute(records,row['key'])[0]);confirmation_calls+=engine.calls
                confirmation.append(dict(label=label,key=row['key'],status='PASS',metrics=assessment.metrics(outputs,assessment.confirm)))
            except Exception as exc:confirmation.append(dict(label=label,key=row['key'],status='FAIL',error=str(exc)))
        a=next((r for r in confirmation if r['label']=='empirical_best' and r['status']=='PASS'),None)
        b=next((r for r in confirmation if r['label']=='best_single' and r['status']=='PASS'),None)
        if baseline is not None and b is not None:
            supported=a is not None and all(x['RMSE']<=y['RMSE']+1e-12 for x,y in zip(a['metrics']['panels'],b['metrics']['panels'])) and a['metrics']['objective']<b['metrics']['objective']-1e-12
            if not supported:chosen=baseline
        elif a is None:chosen=None
    output_path=None
    if chosen is not None:
        deployment_engine=Executor(context,registry)
        deployment=deployment_engine.execute(records,chosen['key'])[0];deployment_calls=deployment_engine.calls
        deployment=assessment.terminal(deployment,dict(context=context))
        output_path=folder/'product.nc';clean(deployment).to_netcdf(output_path,engine='h5netcdf')
        with xr.open_dataset(output_path) as read:xr.testing.assert_allclose(clean(deployment),read.load())
        if evaluation is not None:independent=evaluation(deployment)
        else:independent=None
    else:independent=None
    save(folder/'graphs.json',records);save(folder/'aliases.json',aliases)
    selected_graph=[]
    if chosen:
        reached=set()
        def collect(key):
            if key in reached:return
            reached.add(key)
            for parent in records[key]['parents']:collect(parent)
            selected_graph.append(dict(id=key,**records[key]))
        collect(chosen['key'])
    result=dict(service='PlatformSearch/Autonomous',version=CONTRACT,status='PASS' if chosen else 'NO_ELIGIBLE_RESULT',
        registry_algorithms=len(specs),available_registry_algorithms=len(available),attempts=attempts,
        executed_candidates=candidate_count,unique_working_states=len(states),completed_tree_size=depth,
        kernel_calls=sum(e.calls for e in executors)+confirmation_calls+deployment_calls,state_aliases=len(aliases),
        search_kernel_calls=sum(e.calls for e in executors),confirmation_kernel_calls=confirmation_calls,
        deployment_kernel_calls=deployment_calls,scheduler='Complexity-layered round-robin across arities and registered producers',
        failed_candidates=sum(r['status']=='EXECUTION_FAILED' for r in rows),
        unbounded_or_undocumented_contracts=not declared,handwritten_pool=False,max_depth=None,
        stop_reason=stop or 'RESOURCE_BUDGET_EXHAUSTED',complete_space=global_certificate,
        certification='MEASURED_FINITE_STATE_OBJECTIVE_OPTIMUM' if global_certificate else 'NOT_CERTIFIED_CURRENT_BEST_ONLY',
        empirical_best=best,best_single=baseline,selected_key=chosen['key'] if chosen else None,
        confirmation=confirmation,confirmation_used_for_selection=True,independent_test_used_for_selection=False,
        product=str(output_path) if output_path else None,independent_evaluation=independent,
        best=dict(chosen,product=str(output_path),file_readback='PASS',graph=selected_graph) if chosen else None,
        counts=dict(real_executions=candidate_count,unique_states=len(states),proposals=attempts),
        evaluation_protocol=assessment.protocol,
        protocol=assessment.protocol,objective_definition='Same mean-window RMSE for every candidate and empirical-best ranking',
        soft_portrait_score_used=False,true_unknown_error_global_optimum='NOT_PROVEN',
        scope='Open registry/DAG exploration for timed single-source recovery; other assessment tasks require explicit empirical protocol',
        resource_boundary='Budgets checked between kernels; third-party kernel hard isolation not implemented',
        total_seconds=perf_counter()-started)
    save(folder/'summary.json',result)
    return result
