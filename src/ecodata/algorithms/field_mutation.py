"""Three-portrait screening and existing ALNS over immutable executable DAGs."""
from dataclasses import replace
from pathlib import Path
from itertools import permutations
from math import prod,gcd
from time import perf_counter
import json

import numpy as np
import pandas as pd
import xarray as xr
from threadpoolctl import threadpool_limits

from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.autonomous_decision import Executor, canonical, clean, parameter_variants, RecoveryAssessment, round_robin,protect_observed
from ecodata.algorithms.autonomous_field import build_context, field_registry, measured
from ecodata.algorithms.composition_decision import score_state
from ecodata.algorithms.portrait_state import measure
from ecodata.algorithms.numerical_search import save
from ecodata.algorithms.search_accounting import SearchAccount
from ecodata.algorithms.scientific_state import signature
from ecodata.algorithms.dag import Artifact


def resolve_registered_policy(request, *, allow_unmatched=False):
    """Use an unambiguous reviewed task policy, never invent model preferences."""
    from ecodata.models import portrait_policy
    from ecodata.algorithms import attribute_registry
    request=dict(request);selection=request.get('model_profile')
    if request.get('soft_guidance') is False:
        return request
    if selection and not request.get('portrait_policy'):
        matches=[]
        for path in sorted(portrait_policy.ROOT.glob('*.json')):
            policy=portrait_policy.load(path.stem)
            if policy['model_id']!=selection['id']:continue
            for scenario in policy['scenarios']:
                if all(scenario[k]==selection[k] for k in ('role','field')) and scenario['task']==request['task']:
                    matches.append(dict(version=policy['version'],scenario=scenario['id']))
        if len(matches)!=1:
            if allow_unmatched:
                return dict(request, soft_guidance=False)
            raise ValueError('Choose an explicit registered portrait_policy: zero or multiple matching task policies')
        request['portrait_policy']=matches[0]
        request['policy_resolution']='Unique registered model/role/field/task policy; no inferred simulator sensitivity'
    if request.get('portrait_policy') and not request.get('scoring'):
        declaration=request['portrait_policy'];policy=portrait_policy.load(declaration['version'])
        scenario=next(s for s in policy['scenarios'] if s['id']==declaration['scenario'])
        required={s['attribute'] for s in scenario['issues']};versions=[]
        for version in ['builtin-v1',*[p.stem for p in sorted(attribute_registry.ROOT.glob('*.json'))]]:
            registry=attribute_registry.load(version)
            if required<={a['id'] for a in registry['attributes']}:versions.append(version)
        if len(versions)!=1:raise ValueError('Choose explicit scoring.registry_version: zero or multiple compatible attribute registries')
        request['scoring']=dict(registry_version=versions[0])
    return request


def visible_validation(inputs, request):
    """Reuse daily visible-mask cohorts; retain all other sources and prior state."""
    from ecodata.algorithms.platform_search import SearchInput, SearchValidation, parse_inputs
    context, _, _ = build_context(inputs, request)
    if context.sources[0].dataset.sizes.get('lat',1)*context.sources[0].dataset.sizes.get('lon',1)>1:
        from ecodata.algorithms.current_input import raster_validation
        return raster_validation(inputs,request,context)[:3]
    assessment = RecoveryAssessment(context)
    primary = context.sources[0].dataset.copy(deep=True)
    reference = []
    for panel in assessment.search:
        if panel['seed'] != assessment.protocol['seeds'][0]:
            continue
        days = panel['context'].sources[0].dataset.time.values[panel['mask']]
        reference.append(primary[context.variable].sel(time=days).copy(deep=True))
        primary[context.variable].loc[dict(time=days)] = np.nan
    if not reference:
        raise ValueError('No visible validation cohort; cannot invent an accuracy objective')
    labels = xr.concat(reference, dim='time').sortby('time')
    labels.attrs['split'] = 'search_validation'
    parsed, _ = parse_inputs(inputs)
    index = next(i for i, (role, _) in enumerate(parsed) if role in {'primary', 'background'})
    adjusted = list(inputs)
    item = inputs[index]
    adjusted[index] = SearchInput(item.filename, item.role, clean(primary).to_netcdf(), item.metadata)
    protocol=dict(kind='visible_reconstruction_search_only',objective=assessment.protocol['objective'],
        search_groups=assessment.protocol['search_groups'],masking_seed=assessment.protocol['seeds'][0],
        cohort='Only originally finite values; selected first-seed panels removed from all algorithm inputs',
        final_visible_window_excluded_from_labels=True,confirmation_performed=False,
        model_sensitivity='UNKNOWN',unknown_true_error='NOT_ASSESSED')
    return adjusted, SearchValidation(labels, context.variable), protocol


class GraphMutation:
    """Finite resource search, not a finite hand-written workflow grammar."""
    def __init__(self, context, registry, request, model, reference, folder, budget, seed, progress=None,account=None,folds=None,selection_policy=None,profile_context=None,evaluator=None):
        self.context=context; self.registry=registry; self.request=request; self.reference=reference
        self.evaluator=evaluator
        self.folder=Path(folder); self.budget=budget; self.rng=np.random.default_rng(seed); self.progress=progress
        self.started=perf_counter(); self.attempts=0; self.executions=0; self.iterations=0
        self.empty_neighbourhoods=0;self.prefix_evaluations=0
        self.proposal_budget_blocked=False
        self._node_lists={};self._candidate_cache={};self.duplicate_proposals=0;self.novel_executions=0
        self.recommendation_receipts=[];self.recommended_parents=[]
        self.records={}; self.states={}; self.evaluated={}; self.edits={}; self.transitions=[];self.bank_keys=None
        self.account=account or SearchAccount(budget['objective_limit'],budget['wall_seconds'],probe_calls=budget.get('probe_call_limit'),total_calls=budget.get('total_call_limit'))
        self.executor=Executor(replace(context,validation_reference=None,validation_role=None), registry,account=self.account)
        self.selection_policy=selection_policy;self.folds=folds or []
        self.soft_guidance=request.get('soft_guidance',True)
        if type(self.soft_guidance) is not bool:raise ValueError('soft_guidance must be boolean')
        self.promotion_baseline=None;self.promotion_support=False;self.baseline_transfer_known=None
        self.fold_outputs={};self.fold_executors=[self.executor]
        for fold in self.folds[1:]:
            self.fold_executors.append(Executor(replace(fold['context'],validation_reference=None,validation_role=None),fold['registry'],account=self.account,phase='search_fold_'+fold['id'][:12]))
        self.available=[s for s in registry.specs() if s.status=='available' and s.category!='validation']
        self.field=dict(variable=context.variable,model=model)
        self.baseline=measure(context.sources[0].dataset,context)['metrics']
        finite_reference=np.asarray(reference.values,dtype=float)
        finite_reference=finite_reference[np.isfinite(finite_reference)]
        self.objective_resolution=np.finfo(float).eps*max(1,len(finite_reference))*max(1.,float(np.max(abs(finite_reference))))
        from ecodata.algorithms.search_stopping import SearchStopper,policy
        declaration=policy(request) if request.get('_deployment_reexecution') else request.get('search_stopping',dict(enabled=False))
        if not request.get('search_stopping') and declaration.get('calibration_scope_match') is False:
            declaration=dict(enabled=False,evidence='Published stopping thresholds are outside this task scope; resource limits remain active')
        self.stopper=SearchStopper(declaration,self.objective_resolution,unit='numerical_mutation_round')
        self.numerical_rounds=0
        resources={}
        for name in ('fitted_state','training_reference','historical_simulation'):
            value=context.parameters.get(name)
            if isinstance(value,xr.DataArray):value=value.to_dataset(name=value.name or name)
            if value is not None:resources[name]=Artifact('prior_resource',value).fingerprint()
        for role,dataset in context.parameters.get('prior_resources',{}).items():
            resources['prior:'+role]=Artifact('prior_resource',dataset).fingerprint()
        if context.parameters.get('observation_state_update'):
            resources['conditioning_observations']=Artifact('conditioning_observations',context.observations).fingerprint()
            resources['observation_update_policy']=canonical(context.parameters['observation_state_update'])
        self.state_namespace=canonical(resources)
        self.roots=[];self.representatives={};self.state_keys={};self.deduplicated=0
        for i, source in enumerate(context.sources):
            key='s'+str(i); self.roots.append(key)
            self.records[key]=dict(operation='source',parents=[],parameters=dict(source='source'+str(i)))
            self.states[key]=source.dataset
            for number,fold in enumerate(self.folds):self.fold_outputs[number,key]=fold['context'].sources[i].dataset
            state=self.scientific_key(key,source.dataset);self.state_keys[key]=state
            self.representatives.setdefault(state,key)
        self.screen_log=(self.folder/'portrait_screening.jsonl').open('x',encoding='utf-8')
        self.execution_log=(self.folder/'attempts.jsonl').open('x',encoding='utf-8')
        from ecodata.algorithms.dag_response import DAGResponse
        self.response=DAGResponse(self)
        from ecodata.algorithms.candidate_portraits import CandidatePortraits
        self.candidate_portraits=CandidatePortraits(self,profile_context or context) if selection_policy is not None else None

    def deployment_reserve(self,nodes):
        if self.request.get('_deployment_reexecution') and self.account.total_limit is not None:
            calls=max(1,int(nodes))+1
            estimate=self.account.estimated_cost([self.records[n]['operation'] for n in self.nodes(self.best()['key']) if self.records[n]['operation']!='source']) if self.best() else 0.
            seconds=min(self.account.seconds/2,max(self.account.seconds*.05,2*estimate if np.isfinite(estimate) else 0.))
            self.account.protect_deployment(calls,seconds)

    def exhausted(self):
        return (bool(self.stopper.reason) or self.account.exhausted() or self.attempts>=self.budget['proposal_limit']
                or not self.account.can_search(max(1,len(self.folds)),self.account.reserve)
                or self.iterations>=self.budget['proposal_limit'] or perf_counter()-self.started>=self.budget['wall_seconds'])

    def node_count(self,key):
        return sum(self.records[k]['operation']!='source' for k in self.nodes(key))

    def rank(self,row):
        return (round(row['metrics']['objective']/self.objective_resolution),row['algorithm_nodes'],row['key'])

    def ranked_rows(self,rows=None):
        rows=list(self.evaluated.values()) if rows is None else rows
        if self.bank_keys is not None:rows=[r for r in rows if r['key'] in self.bank_keys]
        if self.selection_policy is not None:
            from ecodata.algorithms.reliable_selection import protected_order
            return protected_order(rows,self.selection_policy,self.promotion_baseline,support_sufficient=self.promotion_support)
        return sorted([r for r in rows if r['status']=='ELIGIBLE_OUTPUT' and not r.get('dominated')],key=self.rank)

    def accept(self,parent,child):
        a=self.evaluated.get(parent);b=self.evaluated.get(child)
        if self.selection_policy is not None:
            chosen=self.best()
            return bool(a and b and chosen and chosen['key']==child and child!=parent)
        return bool(a and b and b['status']=='ELIGIBLE_OUTPUT' and not b.get('dominated') and self.rank(b)<self.rank(a))

    def active_states(self):
        return list(dict.fromkeys(self.representatives.values()))

    def retain_state(self,key,dataset):
        self.states[key]=dataset;state=self.scientific_key(key,dataset);self.state_keys[key]=state
        old=self.representatives.get(state)
        if old is None or self.node_count(key)<self.node_count(old):self.representatives[state]=key
        return self.representatives[state]

    def scientific_key(self,key,dataset):
        if not self.folds:return canonical([self.state_namespace,signature(dataset)])
        # Equivalence on one mask cannot merge different responses on another.
        values=[signature(self.fold_outputs[i,key]) for i in range(len(self.folds))]
        return canonical([self.state_namespace,[f['id'] for f in self.folds],values])

    def numerical_calls(self):return sum(executor.calls for executor in self.fold_executors)

    def destroy(self,sink,rng):
        """Actually detach the current processing subgraph, leaving typed roots."""
        removed=[k for k in self.nodes(sink) if self.records[k]['operation']!='source']
        return dict(parent=sink,cut=sink,removed_nodes=removed,boundary=list(self.roots),
            partial_graph={**{k:self.records[k] for k in self.roots},
                           '__repair_hole__':dict(operation='REPAIR_REQUIRED',parents=[],parameters={})})

    def origin(self,key):
        return {self.records[k]['parameters']['source'] for k in self.nodes(key) if self.records[k]['operation']=='source'}

    def downstream_contract(self,sink):
        """Only reviewed source/calibration transfers; unknowns are not rejected."""
        from ecodata.algorithms.daily_reconstruction import DailyRecovery
        from ecodata.algorithms.series_fusion import PriorSeriesCalibration,SeriesNonnegative,SeriesMissingFallback
        from ecodata.algorithms.open_source_bias import XsdbaPlugin
        from ecodata.algorithms.composition_decision import PrimaryWeightedMean
        from ecodata.algorithms.station_grid_fusion import StationGridFusion
        flags={}
        for key in self.nodes(sink):
            row=self.records[key]
            if row['operation']=='source':
                ds=self.states[key];flags[key]=dict(source=ds.attrs.get('source_id'),
                    corrected=bool(ds.attrs.get('calibration_signature') or ds.attrs.get('bias_adjustment_applied')),
                    fused=bool(ds.attrs.get('fusion_method')));continue
            plugin=self.registry.get(row['operation']);parents=[flags[p] for p in row['parents']]
            if isinstance(plugin,(PriorSeriesCalibration,XsdbaPlugin)):
                parent=parents[0]
                state=self.context.parameters.get('fitted_state')
                expected=(state.attrs.get('calibration_source_id') if isinstance(plugin,PriorSeriesCalibration) and state is not None
                          else self.context.parameters.get('calibration_source_id'))
                if parent.get('corrected') or parent.get('fused'):return 'Downstream calibration has no unmodified training distribution'
                if parent.get('source') is not None and expected is not None and parent['source']!=expected:return 'Downstream calibration source/state mismatch'
                flags[key]=dict(parent,corrected=True)
            elif isinstance(plugin,(DailyRecovery,SeriesNonnegative)):
                flags[key]=dict(parents[0])
            elif isinstance(plugin,StationGridFusion):flags[key]=dict(source=None,corrected=True,fused=True)
            elif isinstance(plugin,PrimaryWeightedMean):flags[key]=dict(source=None,corrected=False,fused=True)
            elif isinstance(plugin,SeriesMissingFallback):
                # Mixed fills do not retain a raw-source training distribution.
                flags[key]=dict(source=None,corrected=False,fused=True)
            else:flags[key]=dict(source=None,corrected=None,fused=None)
        return None

    def nodes(self, sink):
        if sink in self._node_lists:return self._node_lists[sink]
        found=[]; seen=set()
        def visit(key):
            if key in seen: return
            seen.add(key)
            for parent in self.records[key]['parents']: visit(parent)
            found.append(key)
        visit(sink)
        self._node_lists[sink]=found
        return found

    def intern(self, operation, parents, parameters):
        if not self.request.get('_diagnostic_bank'):
            parents=[self.representatives.get(self.state_keys.get(p),p) for p in parents]
        row=dict(operation=operation, parents=list(parents), parameters=parameters)
        key='m'+canonical(row)
        self.records.setdefault(key,row)
        return key

    def replace_node(self, sink, target, replacement):
        visited={}
        def rebuild(key):
            if key==target:return replacement
            if key not in visited:
                row=self.records[key];parents=[rebuild(p) for p in row['parents']]
                visited[key]=self.intern(row['operation'],parents,row['parameters']) if parents!=row['parents'] else key
            return visited[key]
        return rebuild(sink)

    def options(self, spec, parents):
        local=replace(self.context,sources=tuple(EcoDataBundle(self.states[p]) for p in parents))
        plugin=self.registry.get(spec.id)
        if hasattr(plugin,'parameter_proposals'): return plugin.parameter_proposals(local,self.rng)
        if 'parameter_domains' in spec.input_contract:
            return parameter_variants(spec,[self.states[p] for p in parents],self.context.variable)
        return [{}]

    def matching(self, key, actual=None, *, allow_probe=True):
        if not self.soft_guidance:
            return dict(status='DISABLED',score_interval=[0.,0.],reference_used=False,
                        basis='Soft portrait scoring disabled; hard compatibility and measured validation remain active')
        response=self.response.score(key) if allow_probe else self.response.responses.get(key)
        if response is not None and response.get('status') not in {'MEASURED_VISIBLE_MASK','MEASURED_TASK_FOLDS'}:response=None
        if response is not None:
            return dict(response,reference_used=False,response_reference='internal visible-mask cohort only',
                        basis='Three portraits with measured multi-source full-DAG response; not simulator sensitivity')
        # Unexecuted outputs have no invented missingness/accuracy response.
        # An explicitly reviewed projection is the only analytic new transfer.
        final=None
        if actual is not None:
            final=measure(actual,self.context)['metrics']
        elif key in self.states:
            final=measure(self.states[key],self.context)['metrics']
        else:
            from ecodata.algorithms.series_fusion import SeriesNonnegative
            row=self.records[key]
            if isinstance(self.registry.get(row['operation']),SeriesNonnegative):
                parent=row['parents'][0]
                if parent in self.states:
                    before=measure(self.states[parent],self.context)['metrics']
                    final={name:None for name in before}
                    final.update(negative_fraction=0., missing_fraction=before['missing_fraction'],grid_gap=before['grid_gap'])
        result=score_state(self.field,self.baseline,final)
        if self.response.enabled:
            lo,hi=result['score_interval'];model_bounds=[float(np.clip((lo+1)/2,0,1)),float(np.clip((hi+1)/2,0,1))]
            routes=dict(data=[0.,1.],model=model_bounds,collaboration=[0.,1.])
            result.update(score_interval=[sum(self.response.weights[r]*routes[r][i] for r in ('data','model')) for i in (0,1)],
                          routes=routes,weights=self.response.weights,response_status='UNKNOWN',
                          response_reason=self.response.unsupported or ('Cached prefix uses existing response only' if not allow_probe else 'Response call quota exhausted or probe failed'))
        result.update(reference_used=False, accuracy_response='UNKNOWN',
                      basis='Existing three-portrait state-risk formula; unknown responses remain intervals')
        return result

    def candidate(self, sink, target, spec, parents, parameters, operator):
        if self.exhausted(): return None
        self.attempts+=1
        cache_key=(sink,target,spec.id,tuple(parents),canonical(parameters))
        if cache_key in self._candidate_cache:
            self.duplicate_proposals+=1;cached=self._candidate_cache[cache_key]
            return dict(cached,operator=operator) if cached and cached['child'] not in self.evaluated and cached['child']!=sink else None
        self._candidate_cache[cache_key]=None
        record=dict(parent=sink,target=target,algorithm=spec.id,parents=list(parents),parameters=parameters,operator=operator)
        try:
            local=replace(self.context,sources=tuple(EcoDataBundle(self.states[p]) for p in parents),
                          parameters=dict(self.context.parameters,**parameters),validation_reference=None,validation_role=None)
            check=self.registry.get(spec.id).compatibility(local)
            if not check.eligible: raise ValueError(str(check.reasons))
            from ecodata.algorithms.series_fusion import SeriesNonnegative
            from ecodata.algorithms.daily_reconstruction import DailyRecovery
            from ecodata.algorithms.plugins import TemporalSemanticPlugin
            if isinstance(self.registry.get(spec.id),TemporalSemanticPlugin) and len(parents)==1:
                ds=self.states[parents[0]];frequency=local.parameters.get('target_time_frequency')
                try:duration=pd.Timedelta(frequency).to_timedelta64() if frequency else None
                except (ValueError,TypeError):duration=None
                if ('time_bounds' in ds and duration is not None and self.request.get('target_times') and
                    np.array_equal(ds.time.values,pd.DatetimeIndex(self.request['target_times']).values) and
                    np.all(ds.time_bounds.values[:,1]-ds.time_bounds.values[:,0]==duration) and
                    ds[self.context.variable].attrs.get('temporal_semantics') in {'mean','accumulation'}):
                    record['status']='TASK_TIME_ALREADY_ALIGNED'
                    self.screen_log.write(json.dumps(record,default=str)+'\n')
                    return None
            if isinstance(self.registry.get(spec.id),DailyRecovery) and np.isfinite(self.states[parents[0]][self.context.variable]).all():
                record['status']='ANALYTIC_IDENTITY'
                self.screen_log.write(json.dumps(record,default=str)+'\n')
                return None
            from ecodata.algorithms.raster_gap_prediction import DeclaredRangeProjection
            from ecodata.algorithms.raster_gap_prediction import GridGapRecovery
            from ecodata.algorithms.raster_temporal import RasterTimeInterpolation
            if isinstance(self.registry.get(spec.id),GridGapRecovery) and np.isfinite(self.states[parents[0]][self.context.variable]).all():
                record['status']='ANALYTIC_IDENTITY'
                self.screen_log.write(json.dumps(record,default=str)+'\n')
                return None
            if isinstance(self.registry.get(spec.id),RasterTimeInterpolation):
                ds=self.states[parents[0]]
                if np.isfinite(ds[self.context.variable]).all() and np.array_equal(ds.time,pd.DatetimeIndex(local.parameters['raster_target_times']).values):
                    record['status']='ANALYTIC_IDENTITY'
                    self.screen_log.write(json.dumps(record,default=str)+'\n')
                    return None
            if isinstance(self.registry.get(spec.id),DeclaredRangeProjection):
                low,high=local.parameters['target_valid_range'];values=self.states[parents[0]][self.context.variable].values
                if not ((values<low)|(values>high)).any():
                    record['status']='ANALYTIC_IDENTITY'
                    self.screen_log.write(json.dumps(record,default=str)+'\n')
                    return None
            if isinstance(self.registry.get(spec.id),SeriesNonnegative):
                values=np.asarray(self.states[parents[0]][self.context.variable])
                if not (values[np.isfinite(values)]<0).any():
                    record['status']='ANALYTIC_IDENTITY'
                    self.screen_log.write(json.dumps(record,default=str)+'\n')
                    return None
            node=self.intern(spec.id,parents,parameters)
            child=self.replace_node(sink,target,node) if target else node
            reason=self.downstream_contract(child)
            if reason:raise ValueError(reason)
            record.update(child=child,status='COMPATIBLE_LOCAL_EDIT')
            if child==sink or child in self.evaluated:
                record['status']='UNCHANGED_OR_CACHED'
            else:
                record['matching']=self.matching(child)
                self.edits[child]=record
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            record.update(status='INCOMPATIBLE',error=str(exc))
        self.screen_log.write(json.dumps(record,default=str)+'\n')
        if record['status']=='COMPATIBLE_LOCAL_EDIT':self._candidate_cache[cache_key]=dict(record)
        return record if record['status']=='COMPATIBLE_LOCAL_EDIT' else None

    def assess_terminal(self,row,output):
        try:
            if self.selection_policy is not None:
                from ecodata.algorithms.reliable_selection import aggregate
                from ecodata.algorithms.current_input import validate_targets
                from ecodata.evaluation.quality_gates import ResultQualityGate
                responses=[]
                for i,fold in enumerate(self.folds):
                    data=self.fold_outputs[i,row['key']];ctx=fold['context']
                    validate_targets(data,ctx,fold.get('request',self.request))
                    metrics=measured(data,fold['reference'],ctx)
                    quality=ResultQualityGate().validate(data,data,variable=ctx.variable,semantic_type=ctx.semantic_type,
                        nonnegative=bool(ctx.model_constraints.get('nonnegative')))
                    if quality['status']=='BLOCK':raise ValueError('Required fold quality gate blocked output: '+fold['id'])
                    responses.append(dict(fold_id=fold['id'],status='PASS',metrics=metrics))
                row.update(metrics=aggregate(responses,self.selection_policy,[f['id'] for f in self.folds]),
                    fold_responses=responses,quality=quality,status='ELIGIBLE_OUTPUT')
                return
            if self.request.get('_deployment_reexecution'):
                from ecodata.algorithms.current_input import validate_targets
                validate_targets(output,self.context,self.request)
            row['metrics']=self.evaluator(output) if self.evaluator is not None else measured(output,self.reference,self.context)
            from ecodata.evaluation.quality_gates import ResultQualityGate
            quality=ResultQualityGate().validate(output,output,variable=self.context.variable,
                 semantic_type=self.context.semantic_type,nonnegative=bool(self.context.model_constraints.get('nonnegative')))
            if quality['status']=='BLOCK':raise ValueError('Existing result quality gate blocked output')
            row.update(quality=quality,status='ELIGIBLE_OUTPUT')
        except (ValueError,KeyError,IndexError) as exc:row['terminal_reason']=str(exc)

    def assess_prefixes(self,sink):
        for key in self.nodes(sink)[:-1]:
            if key in self.evaluated or key not in self.states:continue
            started=perf_counter();output=self.states[key]
            representative=self.representatives.get(self.state_keys[key],key)
            row=dict(key=key,origin='executed_prefix',parent=sink,status='INTERMEDIATE_ONLY',
                scientific_state=self.state_keys[key],equivalent_to=representative if representative!=key else None,
                dominated=representative!=key and self.evaluated.get(representative,{}).get('status')=='ELIGIBLE_OUTPUT',
                trace=[],matching=self.matching(key,output,allow_probe=False),algorithm_nodes=self.node_count(key),
                total_new_calls=0,adapter_calls_total=self.numerical_calls(),execution_seconds=0.,
                completed_seconds=perf_counter()-self.started)
            self.assess_terminal(row,output)
            if self.selection_policy is not None and row['status']=='ELIGIBLE_OUTPUT':
                self.response.record(row);row['matching']=self.matching(key,output,allow_probe=False)
            row['assessment_seconds']=perf_counter()-started
            row['estimated_cost']=self.account.estimated_cost([self.records[n]['operation'] for n in self.nodes(key) if self.records[n]['operation']!='source'])
            self.evaluated[key]=row;self.executions+=1;self.prefix_evaluations+=1
            self.execution_log.write(json.dumps(row,default=str)+'\n');self.execution_log.flush()

    def can_evaluate(self,key):
        if self.account.total_limit is not None and self.request.get('_deployment_reexecution'):
            best=self.best();nodes=self.node_count(key);reserve=max(nodes,best['algorithm_nodes'] if best else 1)+1
            return self.account.can_search(nodes*max(1,len(self.folds)),reserve)
        return True

    def execute(self, key, origin, parent=None):
        if key in self.evaluated: return self.evaluated[key]
        if not self.can_evaluate(key):return None
        if self.account.total_limit is not None and self.request.get('_deployment_reexecution'):
            best=self.best();self.deployment_reserve(max(self.node_count(key),best['algorithm_nodes'] if best else 1))
        if self.exhausted() and not (origin=='simplification' and not self.account.exhausted()): return None
        self.executions+=1; started=perf_counter();before_calls=len(self.account.events);row=dict(key=key,origin=origin,parent=parent)
        try:
            output,trace=self.executor.execute(self.records,key)
            if self.folds:
                for i,executor in enumerate(self.fold_executors):
                    for node in self.nodes(key):
                        if (i,node) not in self.fold_outputs:
                            self.fold_outputs[i,node]=executor.execute(self.records,node)[0]
            # Cache executed ancestor states without extra unreported probing.
            for node in self.nodes(key):
                if node not in self.states:
                    ds,_=self.executor.execute(self.records,node)
                    self.retain_state(node,ds)
            representative=self.retain_state(key,output)
            row.update(scientific_state=self.state_keys[key],equivalent_to=representative if representative!=key else None,
                       dominated=representative!=key and not self.request.get('_diagnostic_bank'))
            self.deduplicated+=representative!=key
            row.update(status='INTERMEDIATE_ONLY',trace=trace,matching=self.matching(key,output))
            self.assess_terminal(row,output)
            if self.selection_policy is not None and row['status']=='ELIGIBLE_OUTPUT':
                self.response.record(row);row['matching']=self.matching(key,output,allow_probe=False)
        except Exception as exc:
            row.update(status='BUDGET_STOP' if getattr(exc,'details',{}).get('budget_exhausted') else 'EXECUTION_FAILED',error=str(exc),trace=getattr(exc,'trace',[]))
        row.update(algorithm_nodes=sum(self.records[n]['operation']!='source' for n in self.nodes(key)),
                   execution_seconds=perf_counter()-started,completed_seconds=perf_counter()-self.started,
                   adapter_calls_total=self.numerical_calls(),total_new_calls=len(self.account.events)-before_calls)
        row['estimated_cost']=self.account.estimated_cost([self.records[n]['operation'] for n in self.nodes(key) if self.records[n]['operation']!='source'])
        self.evaluated[key]=row
        alias=None
        if row['status']=='ELIGIBLE_OUTPUT' and row.get('dominated') and representative not in self.evaluated:
            before_alias=len(self.account.events)
            alias=dict(row,key=representative,origin='scientific_state_alias',parent=key,dominated=False,
                equivalent_to=None,alias_from=key,algorithm_nodes=self.node_count(representative),
                matching=self.matching(representative),trace=[],execution_seconds=0.)
            alias['estimated_cost']=self.account.estimated_cost([self.records[n]['operation'] for n in self.nodes(representative) if self.records[n]['operation']!='source'])
            alias['total_new_calls']=len(self.account.events)-before_alias
            self.evaluated[representative]=alias;self.executions+=1
            row['promoted_representative']=representative
        if origin in {'mutation','mutation_probe'} and row['total_new_calls']>0:self.novel_executions+=1
        if self.request.get('_deployment_reexecution') and self.account.deployment_limit is None and self.best() is not None:
            remaining=self.account.limit-len(self.account.events)
            eligible=sorted([r for r in self.evaluated.values() if r['status']=='ELIGIBLE_OUTPUT' and not r.get('dominated')
                and self.node_count(r['key'])<=remaining],key=self.rank)
            self.account.reserve=min(remaining,sum(self.node_count(r['key']) for r in eligible[:2]))
        self.execution_log.write(json.dumps(row,default=str)+'\n'); self.execution_log.flush()
        if alias is not None:self.execution_log.write(json.dumps(alias,default=str)+'\n');self.execution_log.flush()
        self.assess_prefixes(key)
        if self.best() is not None:self.deployment_reserve(self.best()['algorithm_nodes'])
        if self.progress: self.progress(dict(event='mutation_execution',candidate=key,status=row['status']))
        return row

    def simplify(self):
        if self.selection_policy is None or not self.selection_policy['parsimony_enabled'] or self.best() is None:
            return dict(enabled=False)
        before=self.best()['key'];trials=[]
        for node in reversed(self.nodes(before)):
            if self.account.exhausted():break
            current=self.best()['key']
            if node not in self.nodes(current) or len(self.records[node]['parents'])!=1:continue
            child=self.replace_node(current,node,self.records[node]['parents'][0])
            if self.downstream_contract(child):continue
            result=self.execute(child,'simplification',current)
            trials.append(dict(removed=node,candidate=child,status=result.get('status') if result else 'BUDGET_STOP'))
        return dict(enabled=True,before=before,after=self.best()['key'],trials=trials,
            selection='Same required folds and frozen practical tolerance; every new call charged')

    def recommendation_parent(self):
        return self.recommended_parents.pop(0) if self.recommended_parents else None

    def recommendation_mutation_family(self,key):
        # This is a graph operation, not a task-to-algorithm preference.
        return 'structure'

    def initialize_recommendations(self):
        for recipe in self.request.get('recommended_recipes',[]):
            receipt=dict(recipe=recipe,status='REJECTED',keys=[])
            try:
                if recipe.get('scope',{}).get('variable')!=self.context.variable:
                    raise ValueError('Recommended recipe variable scope mismatch')
                for root in self.roots:
                    current=root
                    for operation in recipe['steps']:
                        spec=self.registry.get(operation).spec
                        if spec.input_contract.get('source_arity')!=1:
                            raise ValueError('Recommended step requires explicit multi-parent binding')
                        options=list(self.options(spec,[current]))
                        if not options:raise ValueError('No applicable recommended parameter binding')
                        key=self.intern(spec.id,[current],options[0])
                        cached=self.evaluated.get(key)
                        if cached and cached['status']=='ELIGIBLE_OUTPUT':
                            receipt['cache_hits']=receipt.get('cache_hits',0)+1;current=key
                            continue
                        trial=self.candidate(None,None,spec,[current],options[0],'verified_recommendation')
                        if not trial:raise ValueError('Recommended step is incompatible, redundant or unavailable')
                        current=trial['child'];row=self.execute(current,'confirmed_recommendation')
                        if not row or row['status']!='ELIGIBLE_OUTPUT':raise ValueError('Recommended numerical execution failed')
                    receipt['keys'].append(current)
                    if current not in self.recommended_parents:self.recommended_parents.append(current)
                receipt['status']='PASS'
            except (ValueError,KeyError,TypeError) as exc:receipt['error']=str(exc)
            self.recommendation_receipts.append(receipt)
        save(self.folder/'recommendation_validation.json',self.recommendation_receipts)

    def initialize(self):
        screened=[]
        for root in self.roots:
            if not self.exhausted():self.execute(root,'source_baseline')
        self.initialize_recommendations()
        initial_limit=self.request.get('initial_proposal_limit')
        if initial_limit is not None and (type(initial_limit)!=int or not 1<=initial_limit<self.budget['proposal_limit']):
            raise ValueError('Initial proposal limit must reserve mutation attempts')
        available=list(self.available)
        if initial_limit is not None:self.rng.shuffle(available)
        # Score every actual root-neighbour, never a random sixteen-plan pool.
        for spec in available:
            if initial_limit is not None and self.attempts>=initial_limit:break
            arity=spec.input_contract.get('source_arity')
            for count in ([arity] if arity is not None else range(1,len(self.roots)+1)):
                if not count or count>len(self.roots): continue
                for parents in permutations(self.roots,count):
                    try:
                        for param in self.options(spec,parents):
                            row=self.candidate(None,None,spec,parents,param,'initial_registry_neighbour')
                            if row: screened.append(row)
                            if self.exhausted() or (initial_limit is not None and self.attempts>=initial_limit): break
                    except (ValueError,KeyError,TypeError): pass
                    if self.exhausted() or (initial_limit is not None and self.attempts>=initial_limit): break
        # All compatible seeds are retained; real evaluations count in the same budget.
        use_prior=self.soft_guidance and self.candidate_portraits is not None and self.selection_policy['portraits_enabled']
        for row in screened:
            if use_prior:row['candidate_prior']=self.candidate_portraits.score(row['child'])
        screened.sort(key=lambda r:(*self.candidate_portraits.order_key(r['child']),r['child']) if use_prior else (r['child'],))
        save(self.folder/'initial_screening.json',dict(rows=screened,reference_used=False,
             complete_root_scan=not self.exhausted() and (initial_limit is None or self.attempts<initial_limit),global_space_complete=False,
             random_truncation=initial_limit is not None,initial_proposal_limit=initial_limit,
             ordering='seeded_shuffle' if initial_limit is not None else 'registry_order'))
        for key in self.roots:
            if not self.exhausted():self.execute(key,'three_portrait_initialization')
        for row in screened:
            if self.exhausted(): break
            self.execute(row['child'],'three_portrait_initialization')
        self.response.freeze_common()
        return self.best()

    def best(self):
        rows=[r for r in self.evaluated.values() if r['status']=='ELIGIBLE_OUTPUT' and not r.get('dominated')
              and (not self.request.get('_deployment_reexecution') or self.account.deployment_limit is not None or self.account.limit-len(self.account.events)>=r['algorithm_nodes'])]
        return self.ranked_rows(rows)[0] if rows else None

    def neighbourhood(self, sink, family, destruction=None):
        if family=='joint':
            destruction=destruction or self.destroy(sink,self.rng)
            groups=[]
            for root in destruction['boundary']:
                choices=[k for k in self.active_states() if self.origin(k)==self.origin(root)]
                self.rng.shuffle(choices);groups.append(choices)
            specs=list(self.available);self.rng.shuffle(specs)
            # Random bijection of the Cartesian product; never materialise N^sources tuples.
            total=prod(map(len,groups));start=int(self.rng.integers(total)) if total else 0
            stride=int(self.rng.integers(1,max(2,total)))
            while total and gcd(stride,total)!=1:stride=stride%total+1
            def combinations():
                for n in range(total):
                    if self.exhausted():return
                    index=(start+n*stride)%total;parents=[]
                    for group in reversed(groups):index,digit=divmod(index,len(group));parents.append(group[digit])
                    yield tuple(reversed(parents))
            def repairs(spec):
                if spec.input_contract.get('source_arity',len(groups))!=len(groups):return
                for parents in combinations():
                    try:
                        for parameters in self.options(spec,parents):
                            row=self.candidate(sink,destruction['cut'],spec,parents,parameters,'joint_destroy_repair')
                            if row:row['destruction']=destruction
                            yield row
                            if self.exhausted():return
                    except (ValueError,KeyError,TypeError):pass
            yield from round_robin([repairs(s) for s in specs])
            return
        nodes=[k for k in self.nodes(sink) if self.records[k]['operation']!='source']
        self.rng.shuffle(nodes)
        for key in nodes:
            row=self.records[key]; parents=row['parents']
            if not all(p in self.states for p in parents): continue
            if family in {'algorithm','parameters'}:
                specs=list(self.available) if family=='algorithm' else [self.registry.get(row['operation']).spec]
                self.rng.shuffle(specs)
                for spec in specs:
                    if spec.input_contract.get('source_arity',len(parents))!=len(parents): continue
                    try:
                        for param in self.options(spec,parents):
                            yield self.candidate(sink,key,spec,parents,param,'replace_'+family)
                            if self.exhausted(): return
                    except (ValueError,KeyError,TypeError): pass
            if family=='source':
                spec=self.registry.get(row['operation']).spec
                for i in range(len(parents)):
                    partners=self.active_states();self.rng.shuffle(partners)
                    for other in partners:
                        changed=parents.copy();changed[i]=other
                        if len(set(changed))!=len(changed): continue
                        yield self.candidate(sink,key,spec,changed,row['parameters'],'rewire_parent')
                        if self.exhausted(): return
            if family=='structure' and len(parents)==1:
                child=self.replace_node(sink,key,parents[0]); self.attempts+=1
                if child!=sink and child not in self.evaluated:
                    record=dict(parent=sink,child=child,operator='delete_unary',target=key,
                                status='COMPATIBLE_LOCAL_EDIT',matching=self.matching(child))
                    self.edits[child]=record;self.screen_log.write(json.dumps(record,default=str)+'\n');yield record
                inner=self.records[parents[0]]
                if inner['operation']!='source' and len(inner['parents'])==1:
                    # Reverse adjacent unary operations; actual downstream contracts still execute.
                    spec=self.registry.get(row['operation']).spec
                    trial=self.candidate(None,None,spec,inner['parents'],row['parameters'],'swap_order_first')
                    if trial:
                        first=trial['child']
                        # New parent numeric state is unknown until execution; don't invent compatibility.
                        second=self.intern(inner['operation'],[first],inner['parameters'])
                        child=self.replace_node(sink,key,second)
                        record=dict(parent=sink,child=child,operator='swap_adjacent_unary',target=key,
                                    status='COMPATIBLE_LOCAL_EDIT',matching=self.matching(child))
                        self.edits[child]=record;yield record
        if family!='structure': return
        # Insertion at any edge, including the output; multi-parent insertions may
        # combine any executed source branch. No fixed maximum graph depth.
        targets=list(dict.fromkeys([sink,*nodes,*self.roots]));self.rng.shuffle(targets)
        specs=list(self.available);self.rng.shuffle(specs)
        def insertions(target,spec):
            if target in self.states:
                arity=spec.input_contract.get('source_arity')
                for count in ([arity] if arity is not None else range(1,len(self.active_states())+1)):
                    if not count or count>len(self.active_states()): continue
                    partners=[p for p in self.active_states() if p!=target];self.rng.shuffle(partners)
                    for others in permutations(partners,count-1):
                        if self.exhausted():return
                        parents=(target,*others)
                        try:
                            proposed=False
                            for param in self.options(spec,parents):
                                proposed=True
                                yield self.candidate(sink,target,spec,parents,param,'insert_node')
                                if self.exhausted(): return
                            if not proposed:yield None
                        except (ValueError,KeyError,TypeError):yield None
        # Reuse the public explorer's fair stream scheduler: an undeclared-arity
        # plugin must not consume the whole proposal budget before other plugins.
        yield from round_robin([insertions(target,spec) for target in targets for spec in specs])

    def propose(self, sink, family, strategy, rng,destruction=None):
        self.iterations+=1
        self.proposal_budget_blocked=False
        candidates={}
        begin=self.attempts
        allowance=max(1,(self.budget['proposal_limit']-begin)//max(1,self.budget['objective_limit']-(self.executions-self.prefix_evaluations)))
        for row in self.neighbourhood(sink,family,destruction):
            if row and row['child']!=sink and row['child'] not in self.evaluated:
                candidates[row['child']]=row
            if self.exhausted() or self.attempts-begin>=allowance: break
        if not candidates or self.exhausted():
            self.empty_neighbourhoods+=1
            return None
        before=self.evaluated[sink]['metrics']['objective']
        parent_is_best=bool(self.best() and sink==self.best()['key'])
        choices=[]
        for row in candidates.values():
            if self.can_evaluate(row['child']):choices.append(row)
            else:self.proposal_budget_blocked=True
        if not choices:return None
        if self.soft_guidance and strategy=='portrait' and self.selection_policy is not None and self.selection_policy['portraits_enabled']:
            ranks=[self.candidate_portraits.order_key(r['child']) for r in choices]
            best_rank=min(ranks);preferred=[r for r,rank in zip(choices,ranks) if rank==best_rank]
            chosen=preferred[int(rng.integers(len(preferred)))]
            chosen=dict(chosen,candidate_prior=self.candidate_portraits.score(chosen['child']))
        else:
            if self.soft_guidance and strategy=='portrait' and self.response.enabled:
                best=max(r['matching']['score_interval'][0] for r in choices)
                choices=[r for r in choices if r['matching']['score_interval'][0]==best]
            chosen=choices[int(rng.integers(len(choices)))]
        child=chosen['child']
        result=self.execute(child,'mutation',sink)
        proposed_child=child
        if result and result.get('promoted_representative'):
            child=result['promoted_representative'];result=self.evaluated[child]
        after=result.get('metrics',{}).get('objective') if result else None
        self.transitions.append(dict(parent=sink,child=child,family=family,strategy=strategy,edit=chosen,
             iteration=self.iterations,
             proposed_child=proposed_child,
             screened_alternatives=len(candidates),before_objective=before,after_objective=after,
             neighbourhood_proposals=self.attempts-begin,neighbourhood_complete=False,
             parent_is_best_at_proposal=parent_is_best,
             accepted=self.accept(sink,child),actual_execution=result is not None,
             execution_status=result['status'] if result else 'BUDGET_STOP'))
        return child if result else None

    def close(self):
        save(self.folder/'response_scoring.json',self.response.receipt())
        save(self.folder/'total_cost.json',self.account.receipt())
        self.screen_log.close();self.execution_log.close()


@threadpool_limits.wrap(limits=1)
def submit(inputs,request,evaluation,folder,*,budget,seed=11,validation=None,progress=None,graphs=None):
    from ecodata.algorithms.platform_search import platform_registry
    from ecodata.algorithms.profile_matching import prepare_model_request, portraits
    from ecodata.algorithms.composition_optimizers import search_graph
    request=dict(request,soft_guidance=request.get('soft_guidance',False))
    if request.get('review_policy')!='legacy_v1' and graphs is None and budget['proposal_limit']>1:
        request.setdefault('initial_proposal_limit',max(1,min(64,int(budget['proposal_limit'])//3)))
    if type(request['soft_guidance']) is not bool:raise ValueError('soft_guidance must be boolean')
    if not request['soft_guidance']:
        request=dict(request,response_scoring=dict(request.get('response_scoring',{}),enabled=False))
    for name in ('objective_limit','proposal_limit','wall_seconds'):
        if not isinstance(budget.get(name),(int,float)) or budget[name]<=0:
            raise ValueError('Positive mutation resource budget required: '+name)
    account=SearchAccount(budget['objective_limit'],budget['wall_seconds'],probe_calls=budget.get('probe_call_limit'),total_calls=budget.get('total_call_limit'))
    if request.get('target_times') is not None:
        request=dict(request,parameters=dict(request.get('parameters',{}),raster_target_times=request['target_times']))
    request,model=prepare_model_request(resolve_registered_policy(request))
    if 'composition' in request:raise ValueError('PortraitMutation builds graphs from registered interfaces, not a hand-authored composition pool')
    if request['soft_guidance'] and not model.get('state_policy'): raise ValueError('PortraitMutation needs a registered evidence-bearing portrait_policy')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    original_context,original_audit,original_quality=build_context(inputs,request)
    self_protocol=None;confirmation_reference=None;fold_specs=None;selection_policy=None
    review_fold=None;review_receipt=dict(status='INSUFFICIENT',reason='No separate task-matched confirmation supplied')
    review_profile_context=original_context
    if validation is None:
        from ecodata.algorithms.task_validation import plan
        try:fold_specs,self_protocol=plan(inputs,request,original_context,account)
        except ValueError as exc:
            if not str(exc).startswith('VALIDATION_EVIDENCE_INSUFFICIENT:'):raise
            result=dict(status='VALIDATION_EVIDENCE_INSUFFICIENT',product=None,
                reason=str(exc),test_used_for_selection=False,total_cost=account.receipt(),
                task_acceptance=dict(status='EVIDENCE_INSUFFICIENT',task_quality_achieved=False))
            save(folder/'request.json',dict(request=request,inputs=original_audit,quality=original_quality,budget=budget,seed=seed))
            save(folder/'summary.json',result)
            return result
        if request.get('review_policy')!='legacy_v1':
            from ecodata.algorithms.task_validation import reserve_review
            fold_specs,review_fold,profile_inputs,review_receipt=reserve_review(fold_specs,inputs,request,original_context)
            if profile_inputs is not None:review_profile_context=build_context(profile_inputs,request)[0]
            self_protocol.update(review_reservation=review_receipt,fold_count=len(fold_specs),
                primary_fold_ids=[f['id'] for f in fold_specs])
        inputs=fold_specs[0]['inputs'];validation=fold_specs[0]['validation'];confirmation_reference=fold_specs[0]['confirmation']
        request=dict(request,_current_primary_index=self_protocol['target_source_index'],
            parameters=dict(request.get('parameters',{}),reference_support='fixed_finite_reference',require_complete_output=True))
    request=dict(request,_deployment_reexecution=self_protocol is not None,_diagnostic_bank=graphs is not None)
    if self_protocol is not None:
        account.configure_deployment(budget.get('deployment_limit',budget['objective_limit']),budget.get('deployment_seconds',budget['wall_seconds']))
    first_request=fold_specs[0].get('request',request) if fold_specs else request
    context,audit,quality=build_context(inputs,first_request)
    from ecodata.algorithms.profile_matching import validate_model_context
    validate_model_context(context,model)
    if set(model['matching_requirements']['hard'])-{'nonnegative'}:
        raise ValueError('Open-DAG mutation has no hard guarantee consumer for this model requirement')
    registry=field_registry(platform_registry(),context)
    folds=[]
    if fold_specs is not None:
        from ecodata.algorithms.reliable_selection import make_policy
        history=[];history_receipt=dict(status='NOT_REQUIRED_LOCAL_MATCH_OR_OTHER_TASK')
        if request.get('reliable_selection',{}).get('mode')=='fold_only':history_receipt=dict(status='DISABLED_FOLD_ONLY_ABLATION')
        if review_fold is not None:history_receipt=dict(status='NOT_MIXED_WITH_ISOLATED_CURRENT_INPUT_REVIEW',reason='External history must independently prove no shared outer labels')
        if review_fold is None and request.get('reliable_selection',{}).get('mode')!='fold_only' and self_protocol.get('tasks')==['temporal_gap'] and self_protocol.get('support_limited'):
            from ecodata.algorithms.development_validation import history_folds
            history,history_receipt=history_folds(request,original_context,self_protocol.get('task_profile',{}))
            fold_specs+=history
        preferences=model.get('application_preferences',{}).get('weights')
        selection_policy=make_policy(request,context.sources[self_protocol['target_source_index']].dataset[context.variable].values,model_weights=preferences)
        if not request['soft_guidance']:selection_policy['portraits_enabled']=False
        for i,fold in enumerate(fold_specs):
            fold_request=fold.get('request',request)
            ctx=context if i==0 else build_context(fold['inputs'],fold_request)[0]
            folds.append(dict(id=fold['id'],context=ctx,reference=fold['validation'].reference,
                registry=registry if i==0 else field_registry(platform_registry(),ctx),request=fold_request))
        selection_policy['objective_fold_ids']=[f['id'] for f in history] if history else self_protocol.get('primary_fold_ids',[f['id'] for f in folds])
        selection_policy['data_portrait']['task_profile']=self_protocol.get('task_profile',{})
        selection_policy['sha256']=canonical(selection_policy)
        self_protocol=dict(self_protocol,selection_policy=selection_policy,history_validation=history_receipt,
            all_required_fold_ids=[f['id'] for f in folds],objective_fold_ids=selection_policy['objective_fold_ids'],
            evidence_scope='Applicable development simulations are primary, local diagnostics separate' if history else self_protocol.get('support_level','Local diagnostic support only'))
    save(folder/'request.json',dict(request=request,inputs=audit,quality=quality,budget=budget,seed=seed,
         validation=validation.protocol,self_validation=self_protocol,handwritten_pool=False,max_depth=None))
    save(folder/'portraits.json',portraits(context,registry,model))
    save(folder/'catalog.json',[s.as_dict() for s in registry.specs()])
    if self_protocol is not None:
        clean(context.sources[0].dataset).to_netcdf(folder/'self_validation_input.nc',engine='h5netcdf')
        clean(validation.reference.to_dataset(name=context.variable)).to_netcdf(folder/'self_validation_labels.nc',engine='h5netcdf')
        if confirmation_reference is not None:
            clean(confirmation_reference.to_dataset(name=context.variable)).to_netcdf(folder/'self_confirmation_labels.nc',engine='h5netcdf')
            roots=folder/'self_validation_roots';roots.mkdir()
            for index,source in enumerate(context.sources):
                clean(source.dataset).to_netcdf(roots/f'source{index}.nc',engine='h5netcdf')
    if fold_specs is not None:
        for i,fold in enumerate(folds):
            local=folder/'validation_folds'/str(i);local.mkdir(parents=True)
            clean(fold['reference'].to_dataset(name=context.variable)).to_netcdf(local/'labels.nc',engine='h5netcdf')
            for j,source in enumerate(fold['context'].sources):clean(source.dataset).to_netcdf(local/f'source{j}.nc',engine='h5netcdf')
            save(local/'protocol.json',fold_specs[i]['protocol'])
    from ecodata.algorithms.platform_search import BlockedSearchValidation
    search=GraphMutation(context,registry,request,model,validation.reference,folder,budget,seed,progress,account,folds=folds,selection_policy=selection_policy,profile_context=review_profile_context,
        evaluator=validation if isinstance(validation,BlockedSearchValidation) else None)
    try:
        backend=request.get('mutation_control','random')
        if backend not in {'alns','random','none','fixed'}:raise ValueError('Unknown explicit mutation backend; no fallback')
        bank_keys={}
        if graphs is not None:
            if not isinstance(graphs,dict) or not graphs:raise ValueError('Nonempty diagnostic graph bank required')
            search.bank_keys=set()
            for name,graph in graphs.items():
                rebound={}
                def bind(key):
                    if key in rebound:return rebound[key]
                    row=graph['records'][key]
                    if row['operation']=='source':
                        index=int(row['parameters']['source'].removeprefix('source'))
                        if row!=search.records.get('s'+str(index)):raise ValueError('Diagnostic bank source binding differs')
                        rebound[key]='s'+str(index)
                    else:
                        registry.get(row['operation'])
                        rebound[key]=search.intern(row['operation'],[bind(p) for p in row['parents']],row['parameters'])
                    return rebound[key]
                key=bind(graph['key']);bank_keys[name]=key
                search.bank_keys.add(key)
                search.execute(key,'diagnostic_bank')
            initial=search.best();backend='bank';search.response.freeze_common()
        elif backend=='fixed':
            fixed=request.get('fixed_graph')
            if not isinstance(fixed,dict) or not {'records','key'}<=set(fixed):raise ValueError('Explicit frozen graph required for fixed control')
            for key,row in fixed['records'].items():
                if row['operation']=='source':
                    if key not in search.records or row!=search.records[key]:raise ValueError('Fixed source binding differs from actual inputs')
                else:registry.get(row['operation'])
            search.records.update(fixed['records'])
            search.execute(fixed['key'],'fixed_control');initial=search.best()
        else:initial=search.initialize()
        if initial and self_protocol is not None and selection_policy is not None:
            # Freeze the best measured initial result; simplicity only breaks
            # numerical ties through the shared ranking rule.
            initial=search.ranked_rows()[0];search.promotion_baseline=initial
            search.promotion_support=not self_protocol.get('support_limited',True)
        initial_key=initial['key'] if initial else None
        optimizer=dict(backend=backend)
        checkpoint=None
        if initial and self_protocol is not None:
            from ecodata.algorithms.current_input import deploy_ranked
            row,data,receipt=deploy_ranked(search,original_context,registry,request)
            if row is not None:
                checkpoint=(row,data,receipt)
                clean(data).to_netcdf(folder/'checkpoint.nc',engine='h5netcdf')
                save(folder/'checkpoint.json',dict(key=row['key'],deployment=receipt))
        search.stopper.start(initial)
        if initial and not search.exhausted() and graphs is None:
            if backend=='alns':optimizer=search_graph(search,initial_key,seed=seed)
            elif backend=='random':
                from ecodata.algorithms.composition_optimizers import random_graph_search
                optimizer=random_graph_search(search,request,selection_policy or dict(portraits_enabled=search.soft_guidance))
        simplification=search.simplify() if graphs is None and backend not in {'none','fixed'} else dict(enabled=False)
        best=search.best();validation_best=best;product_path=None;deployment=None;confirmation=None
        if best:
            if self_protocol is not None:
                from ecodata.algorithms.current_input import deploy_ranked
                best,output,deployment=deploy_ranked(search,original_context,registry,request)
                if best is None and checkpoint is not None:
                    best,output,previous=checkpoint
                    deployment=dict(deployment,status='PASS',checkpoint_fallback=True,checkpoint_key=best['key'],checkpoint_deployment=previous)
            else:
                output,_=search.executor.execute(search.records,best['key'])
                protected=request.get('protect_observed_source')
                if protected:
                    source=next(s.dataset for s in original_context.sources if s.dataset.attrs.get('source_id')==protected)
                    output=protect_observed(output,source,context.variable)
        if best:
            baselines=[r for r in search.evaluated.values() if r['origin']=='source_baseline' and r['status']=='ELIGIBLE_OUTPUT']
            source_baseline=min(baselines,key=search.rank) if baselines else None
            save(folder/'selection_freeze.json',dict(key=best['key'],baseline_key=initial_key,
                source_baseline_key=source_baseline['key'] if source_baseline else None,
                graph=search.records,confirmation_read=False,review_reservation=review_receipt))
            if confirmation_reference is not None:
                held_output,_=search.executor.execute(search.records,best['key'])
                confirmation=dict(metrics=measured(held_output,confirmation_reference,context),used_for_selection=False,
                    source='Disjoint visible holdout, kept absent from every selection root; evaluated before observation protection')
            product_path=folder/'product.nc';clean(output).to_netcdf(product_path,engine='h5netcdf')
            with xr.open_dataset(product_path) as f:xr.testing.assert_allclose(clean(output),f.load())
        scale_audit=None
        if best and self_protocol is not None and self_protocol.get('tasks')==['scale_refinement']:
            from ecodata.algorithms.task_validation import scale_consistency
            scale_audit=scale_consistency(output,original_context.sources[self_protocol['target_source_index']].dataset,context.variable,account)
        bank_rows=[]
        if graphs is not None:
            from ecodata.algorithms.current_input import validate_output
            account.begin_deployment()
            full_executor=Executor(replace(original_context,validation_reference=None,validation_role=None),registry,account=account,phase='deploy')
            for name,key in bank_keys.items():
                if Path(name).name!=name or name in {'.','..'}:raise ValueError('Safe diagnostic bank identifiers required')
                measured_row=search.evaluated.get(key,{})
                row=dict(bank_id=name,key=key,validation_status=measured_row.get('status','NOT_EVALUATED'),
                    metrics=measured_row.get('metrics'),algorithm_nodes=search.node_count(key),selection_eligible=measured_row.get('status')=='ELIGIBLE_OUTPUT')
                try:
                    data,trace=full_executor.execute(search.records,key)
                    data,checks=validate_output(data,original_context,request)
                    local=folder/'bank_products'/name;local.mkdir(parents=True)
                    product=local/'product.nc';clean(data).to_netcdf(product,engine='h5netcdf')
                    row.update(status='PASS',product=str(product),quality=checks,trace=trace)
                except Exception as exc:row.update(status='FAILED',error=str(exc),trace=getattr(exc,'trace',[]))
                bank_rows.append(row)
        save(folder/'graphs.json',search.records);save(folder/'mutation_trace.json',search.transitions)
        if isinstance(validation,BlockedSearchValidation):
            save(folder/'selection_freeze.json',dict(key=best['key'] if best else None,
                source_baseline_key=source_baseline['key'] if best and source_baseline else None,
                graph=search.records,validation_protocol=validation.protocol,
                reference_used_for_selection='Search validation only; confirmation not read yet'))
        result=dict(service='PlatformSearch/PortraitMutation',status='PASS' if best else ('NO_EXECUTABLE_CURRENT_GRAPH' if deployment is not None else 'NO_ELIGIBLE_RESULT'),
          best=best,initial_best=initial,product=str(product_path) if product_path else None,
          source_baselines=[r for r in search.evaluated.values() if r['origin']=='source_baseline'],
          bank_rows=bank_rows,selected_bank_id=next((name for name,key in bank_keys.items() if best and key==best['key']),None),
          reliable_selection=selection_policy,
          promotion_guard=dict(baseline_key=search.promotion_baseline['key'],support_sufficient=search.promotion_support,
              baseline_transfer_known=search.baseline_transfer_known,
              required_folds=len(search.promotion_baseline['metrics']['fold_losses']),
              rule='Rank complete comparable measured fold loss; simplicity only within global equivalence tolerance. Limited support affects claims, not search or replacement') if search.promotion_baseline else None,
          scale_consistency=scale_audit,
          simplification=simplification,
          validation_best=validation_best,current_input_deployment=deployment,confirmation=confirmation,
          final_validation_best_delivered=bool(best and validation_best and best['key']==validation_best['key']),
          candidate_prior=dict(version='candidate_portraits_v1',validation_loss_used=False,scope='Visible-profile priority before execution; random exploration retained') if selection_policy and search.soft_guidance else None,
          soft_guidance_enabled=search.soft_guidance,
          self_validation_from_current_input=self_protocol is not None,self_validation_protocol=self_protocol,
          objective_tie_resolution=search.objective_resolution,
          numerical_comparison='Required-fold measured task loss, global practical-equivalence simplicity and stable graph ID; wall-time never decides ties' if selection_policy else 'float64 roundoff buckets, then graph simplicity and stable graph ID; wall-time cost never decides ties',
          executed_candidates=search.executions,prefix_terminal_assessments=search.prefix_evaluations,attempts=search.attempts,kernel_calls=search.numerical_calls(),
          adapter_invocations=search.numerical_calls(),search_policy=('three_portrait_then_' if search.soft_guidance else 'hard_contracts_then_')+backend,optimizer=optimizer,
          mutation_actual_new_executions=search.novel_executions,
          mutation_new_executions=sum(r['origin'] in {'mutation','mutation_probe'} for r in search.evaluated.values()),
          evolution_used=bool(search.transitions),elapsed_seconds=perf_counter()-search.started,
          empty_neighbourhoods=search.empty_neighbourhoods,
          iterations=search.iterations,
          failed_candidates=sum(r['status']=='EXECUTION_FAILED' for r in search.evaluated.values()),
          certification='NOT_CERTIFIED_CURRENT_BEST_ONLY',test_used_for_selection=False,
          stop_reason='DIAGNOSTIC_BANK_COMPLETE' if backend=='bank' else 'FIXED_GRAPH_EXECUTED' if backend=='fixed' else 'PORTRAIT_INITIALIZATION_ONLY' if backend=='none' else search.stopper.reason or 'RESOURCE_BUDGET_EXHAUSTED',evaluation_protocol=validation.protocol,
          search_stopping=search.stopper.receipt(),duplicate_proposals_skipped=search.duplicate_proposals,
          checkpoint_available=checkpoint is not None,
          self_validation_from_visible=self_protocol is not None,registry_count=len(registry.specs()),
          available_count=len(search.available),deployment_adapter_calls=account.counts['deploy'],
          total_cost=account.receipt(),scientific_state_deduplications=search.deduplicated,
          recommendation_validation=search.recommendation_receipts,
          parent_policy='verified_recommendations_then_best' if search.recommendation_receipts else 'best',response_score_intervals=len({tuple(r.get('score_interval',[])) for r in search.response.responses.values() if r['status'] in {'MEASURED_VISIBLE_MASK','MEASURED_TASK_FOLDS'}}),
          independent_evaluation=evaluation(output) if best and evaluation is not None else None)
        if self_protocol is not None:
            from ecodata.algorithms.task_validation import outcome
            result['task_acceptance']=outcome(self_protocol,request,result['stop_reason'],confirmation)
        if best and backend not in {'none','fixed','bank'} and not result['evolution_used'] and not search.stopper.reason:
            result['status']='BUDGET_EXHAUSTED_BEFORE_MUTATION'
        if request.get('require_mutation_execution') and search.novel_executions==0:
            result['status']='MUTATION_NOT_EXECUTED'
            result['reason']='No genuinely new mutation evaluation completed; source/cached output is not a mutation acceptance'
        if self_protocol is not None:
            from ecodata.evaluation.blocked_confirmation import review_field
            gate=review_field(search,result,review_fold,review_receipt,request)
            result['recommendation_review']=gate
            result['delivery_status']='ENHANCEMENT_RECOMMENDED' if gate['recommended'] else 'EXPLORATORY_ONLY'
            result['delivered_graph']=dict(key=best['key'],records={k:search.records[k] for k in search.nodes(best['key'])}) if best else None
            result['delivered_best']=best
            if best and initial and not gate['recommended'] and initial['key']!=best['key']:
                candidate=folder/'exploratory-candidate.nc';candidate.write_bytes(product_path.read_bytes())
                result['candidate_product']=str(candidate)
                try:
                    from ecodata.algorithms.current_input import validate_output
                    fallback_executor=Executor(replace(original_context,validation_reference=None,validation_role=None),registry,account=account,phase='deploy')
                    data,trace=fallback_executor.execute(search.records,initial['key'])
                    data,checks=validate_output(data,original_context,request)
                    clean(data).to_netcdf(product_path,engine='h5netcdf')
                    with xr.open_dataset(product_path) as f:xr.testing.assert_allclose(clean(data),f.load())
                    result.update(delivered_graph=dict(key=initial['key'],records={k:search.records[k] for k in search.nodes(initial['key'])}),delivered_best=initial,delivery_status='BASELINE_RETAINED',
                        final_validation_best_delivered=False,delivery_trace=trace,delivery_quality=checks)
                except (ValueError,RuntimeError) as exc:
                    product_path.write_bytes(candidate.read_bytes())
                    gate['delivery_error']=str(exc)
            elif best and initial and not gate['recommended']:result['delivery_status']='BASELINE_RETAINED'
            save(folder/'recommendation_review.json',gate)
            result['total_cost']=account.receipt()
        save(folder/'summary.json',result)
        return result
    finally:search.close()
