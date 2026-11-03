"""Measured multi-source visible-mask response for three-portrait matching."""
from dataclasses import replace
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.daily_reconstruction import visible_day_mask
from ecodata.algorithms.autonomous_decision import Executor
from ecodata.algorithms.composition_decision import score_state
from ecodata.algorithms.composition_scoring import visible_response
from ecodata.algorithms.portrait_state import measure


class DAGResponse:
    def __init__(self, owner):
        self.owner=owner;self.context=owner.context;self.responses={};self.common=None
        self.unsupported=None;self.executor=None;self.protocol={}
        declaration=owner.request.get('response_scoring',{})
        self.enabled=declaration.get('enabled',True)
        self.weights=declaration.get('weights',dict(data=.5,model=.5,collaboration=0.))
        self.task_folds=getattr(owner,'selection_policy',None) is not None
        self.measured_rows={}
        if self.task_folds:
            self.enabled=owner.selection_policy['portraits_enabled']
            self.protocol=dict(kind='shared_task_fold_response',fold_ids=[f['id'] for f in owner.folds],
                loss_policy=owner.selection_policy,additional_response_pipeline_calls=0,
                evidence='Data determines held dates and scale; model defines measured-loss weights; algorithm response is complete graph execution')
            return
        if set(self.weights)!={'data','model','collaboration'} or any(not np.isfinite(w) or w<0 for w in self.weights.values()) or sum(self.weights.values())<=0:
            raise ValueError('Three explicit finite route weights required')
        if self.weights['collaboration']!=0:raise ValueError('Collaboration is a common-reference diagnostic, not a self-relative reward')
        total=sum(self.weights.values());self.weights={k:v/total for k,v in self.weights.items()}
        if not self.enabled:self.unsupported='Explicit no-response ablation';return
        root=self.context.sources[0].dataset;var=self.context.variable
        if root[var].dims!=('time','lat','lon') or root.sizes['lat']!=1 or root.sizes['lon']!=1 or len(root.time)<30:
            self.unsupported='Visible-day response requires a daily point series with sufficient support';return
        period=declaration.get('period')
        years=sorted(set(pd.DatetimeIndex(root.time.values).year))
        if period is None:
            last=years[max(0,len(years)//3-1)]
            period=[str(pd.Timestamp(root.time.values[0])),str(pd.Timestamp(year=last,month=12,day=31,hour=23))]
        sources=[]
        for bundle in self.context.sources:
            ds=bundle.dataset.sel(time=slice(*period)).copy(deep=True)
            if not ds.sizes.get('time'):self.unsupported='No source support in response period';return
            sources.append(ds)
        values=sources[0][var].values[:,0,0];mask,lengths=visible_day_mask(values,np.random.default_rng(declaration.get('seed',704)))
        if not mask.any():self.unsupported='No finite, bracketed visible-mask response cohort';return
        self.reference=sources[0][var].isel(time=np.flatnonzero(mask)).copy(deep=True)
        sources[0][var].values[mask]=np.nan
        state=self.context.parameters.get('fitted_state')
        if state is not None and 'scale' in state:
            scale=float(state.scale);basis='prior fitted training standard deviation'
        else:
            retained=sources[0][var].values;scale=float(np.nanstd(retained));basis='retained visible probe inputs only; not withheld values'
        if not np.isfinite(scale) or scale<=0:self.unsupported='No positive observed normalization scale';return
        self.scale=scale
        self.context=replace(self.context,sources=tuple(EcoDataBundle(s) for s in sources),validation_reference=None,validation_role=None)
        self.executor=Executor(self.context,owner.registry,account=owner.account,phase='probe')
        self.protocol=dict(period=period,mask_times=[str(t) for t in self.reference.time.values],
            mask_values=self.reference.values.ravel().tolist(),gap_lengths=lengths,scale=scale,scale_basis=basis,
            weights=self.weights,weights_evidence=declaration.get('evidence','Reuse existing daily protocol data=.5/model=.5/collaboration=0; explicit experimental priorities, not learned model sensitivities'),
            all_derived_branches_reexecuted_from_masked_roots=True,final_test_labels_used=False,
            legacy_expected_error_slot='measured normalized absolute volume bias, not native simulator error')

    def score(self,key):
        if self.task_folds:return self.responses.get(key) if self.enabled else None
        if key in self.responses:
            cached=self.responses[key]
            return cached if cached['status']=='MEASURED_VISIBLE_MASK' else None
        if self.executor is None:return None
        if self.owner.account.exhausted('probe'):return None
        try:
            ds,trace=self.executor.execute(self.owner.records,key)
            var=self.context.variable;pred=ds[var].sel(time=self.reference.time).values.ravel();truth=self.reference.values.ravel()
            if not np.isfinite(pred).all():raise ValueError('Incomplete fixed response cohort')
            if self.context.model_constraints.get('nonnegative') and np.any(pred<0):raise ValueError('Negative probe output')
            years=pd.DatetimeIndex(self.reference.time.values).year
            error=float(np.mean([np.sqrt(np.mean((pred[years==y]-truth[years==y])**2)) for y in np.unique(years)]))
            bias=float(abs(np.mean(pred-truth))/self.scale)
            measured=measure(ds,self.context)['metrics'];measured['expected_error']=bias
            matching=score_state(self.owner.field,self.owner.baseline,measured)
            response=visible_response(error,self.scale,matching['score_interval'],evidence=self.protocol['weights_evidence'],
                normalized_absolute_volume_bias=bias,model_terms=matching['terms'],trace=trace,
                predicted=pred.tolist(),simulator_loss=None)
            joint=[sum(self.weights[r]*response['routes'][r][i] for r in ('data','model')) for i in (0,1)]
            response.update(score_interval=joint,weights=self.weights)
        except Exception as exc:
            response=dict(status='UNKNOWN',reason=str(exc),trace=getattr(exc,'trace',[]))
        self.responses[key]=response
        return response if response['status']=='MEASURED_VISIBLE_MASK' else None

    def record(self,row):
        """Reuse complete multi-fold execution, never invent a raster gain."""
        if not self.task_folds or row.get('status')!='ELIGIBLE_OUTPUT':return
        metrics=row['metrics'];self.measured_rows[row['key']]=metrics
        value=1/(1+metrics['objective'])
        response=dict(status='MEASURED_TASK_FOLDS',score_interval=[value,value],
            task_loss=metrics['objective'],anchor_rmse=metrics.get('RMSE'),
            measured_portraits=metrics['portrait_response'],fold_losses=metrics['fold_losses'],
            data_evidence=self.protocol,model_evidence=self.owner.selection_policy['weight_basis'],
            gain_status='UNKNOWN_UNTIL_COMMON_BASELINE',additional_response_pipeline_calls=0)
        if self.common is not None:
            from ecodata.algorithms.reliable_selection import common_gain
            response['common_gain']=common_gain(metrics,self.measured_rows[self.common['key']],self.owner.selection_policy)
            response['gain_status']='MEASURED_COMMON_BASELINE'
        self.responses[row['key']]=response

    def freeze_common(self):
        if self.task_folds:
            if not self.enabled:return
            declared=self.owner.selection_policy.get('common_reference_graph')
            if declared:
                from ecodata.algorithms.reliable_selection import aggregate
                from ecodata.algorithms.autonomous_field import measured
                from ecodata.algorithms.current_input import validate_targets
                try:
                    rows=[]
                    for fold,executor in zip(self.owner.folds,self.owner.fold_executors):
                        data,_=executor.execute(declared['records'],declared['key'])
                        validate_targets(data,fold['context'],fold['request'])
                        metrics=measured(data,fold['reference'],fold['context'])
                        rows.append(dict(fold_id=fold['id'],status='PASS',metrics=metrics))
                    metrics=aggregate(rows,self.owner.selection_policy,[f['id'] for f in self.owner.folds])
                    self.common=dict(key=declared['key'],bank_id=declared['bank_id'],scope=declared['evidence'],
                        injected_into_candidate_archive=False)
                    self.measured_rows[self.common['key']]=metrics
                    for row in self.owner.evaluated.values():self.record(row)
                except Exception as exc:
                    self.protocol['common_reference_status']=dict(status='UNKNOWN',reason=str(exc))
                return
            candidates=[r for r in self.owner.evaluated.values() if r['status']=='ELIGIBLE_OUTPUT' and r['key'] in self.measured_rows]
            if candidates:
                chosen=min(candidates,key=lambda r:(r['metrics']['objective'],r['algorithm_nodes'],r['key']))
                self.common=dict(key=chosen['key'],rmse=chosen['metrics'].get('RMSE'),scope='Best measured common initialization graph on identical required folds; frozen before mutation')
                for row in candidates:self.record(row)
            return
        choices=[(r['anchor_rmse'],key) for key,r in self.responses.items()
                 if r['status']=='MEASURED_VISIBLE_MASK' and self.owner.node_count(key)<=1]
        if choices:
            error,key=min(choices);self.common=dict(key=key,rmse=error,scope='Best measured initial single/source on the same visible-mask cohort; frozen before mutation')

    def receipt(self):
        return dict(version='dag_visible_response_v1',enabled=self.enabled,unsupported=self.unsupported,
                    protocol=self.protocol,common_reference=self.common,responses=self.responses,
                    no_global_grammar_enumeration=True)
