"""Three parallel portrait routes from real, visible-only masked-anchor probes.

This is task-local empirical state response, not a learned ranking surrogate.
An entire local chain is executed so that order and nonlinear side effects are
measured together. Unobserved cross-window interactions are never inferred.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ecodata.algorithms.attribute_registry import digest


ROUTES = ("data", "model", "collaboration")


def visible_response(error, scale, model_interval, *, evidence, **details):
    """Shared measured response normalization; not a learned ranking model."""
    if not np.isfinite(error) or error<0 or not np.isfinite(scale) or scale<=0:
        raise ValueError('Finite error and positive prior scale required')
    lo,hi=model_interval
    return dict(status='MEASURED_VISIBLE_MASK',routes=dict(data=[1/(1+error/scale)]*2,
        model=[float(np.clip((lo+1)/2,0,1)),float(np.clip((hi+1)/2,0,1))],collaboration=[0.,1.]),
        anchor_rmse=float(error),normalization_scale=float(scale),evidence=evidence,**details)


class ParallelPortraitScoring:
    def __init__(self, engine, declaration):
        if set(declaration)-{"weights", "evidence", "common_reference"} or not declaration.get("evidence"):
            raise ValueError("Parallel portrait scoring needs weights and their evidence")
        weights = declaration.get("weights")
        if not isinstance(weights, dict) or set(weights)!=set(ROUTES):
            raise ValueError("Exactly three parallel route weights required")
        if any(type(v) not in (int, float) or not np.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
            raise ValueError("Finite nonnegative route weights with positive sum required")
        self.weights={k:float(v)/sum(weights.values()) for k,v in weights.items()}
        self.evidence=declaration["evidence"]
        self.reference=None;self.reference_probes=[]
        if declaration.get('common_reference'):
            spec=declaration['common_reference'];raw=Path(spec['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Common reference digest mismatch')
            self.reference=json.loads(raw)
            if self.reference.get('training_partition')!='development' or self.reference.get('test_used') is not False:
                raise ValueError('Common reference must be frozen from development only')
            if self.reference['scope']!=reference_scope(engine):raise ValueError('Common reference scope mismatch')
            if weights['collaboration']!=0 or weights['data']+weights['model']<=0:
                raise ValueError('Common-reference collaboration is a secondary gain gate, not an additive bonus')
            self.reference_evidence=dict(sha256=spec['sha256'],training_groups=self.reference['training_groups'])
        self.responses={};self.probes=[];started=perf_counter()
        self.executions=0;self.kernel_calls=0
        self.scope="Masked visible interior fine anchors; local full-chain response, not hidden-target accuracy or measured model response"
        if len(engine.fields)!=1 or not engine.config.get("reconstruction") or engine.resources:
            self.unsupported="Requires one complementary field without fitted external resources"
        else:
            self.unsupported=None
            for window in engine.windows(engine.fields[0]["request"]):
                self._window(engine,window)
        self.seconds=perf_counter()-started

    def _reference_window(self,engine,req,parsed,window,truth):
        from ecodata.algorithms.composition_decision import Composition
        for name,plan in self.reference['plans'].items():
            local=deepcopy(req)
            def chosen(branch):
                return next(p['choice'] for p in plan if p['window']==window['id'] and p['branch']==branch)
            def stages(chain):
                return [dict(id='frozen_'+str(i),after=['frozen_'+str(i-1)] if i else [],options=[step],
                    evidence='Development-frozen common reference order') for i,step in enumerate(chain)]
            local['composition']['source_stages']={k:stages(chosen(k)) for k in engine.ids}
            local['composition']['post_stages']=stages(chosen('post'))
            local['composition']['fusion_options']=[chosen('fusion')]
            probe=Composition(parsed,local,engine.registry);self.executions+=1
            try:
                ds,_,trace,checks=probe.execute(probe.decode(0))
                self.kernel_calls+=sum(t.get('kernel_calls',0) for t in trace)
                values=ds[req['variable']].transpose(*truth.dims).values
                error=float(np.sqrt(np.mean((values-truth.values)**2)))
                self.reference_probes.append(dict(window=window['id'],name=name,status='PASS',
                    rmse=error,values=values.tolist(),samples=int(values.size),trace=trace,checks=checks))
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                self.reference_probes.append(dict(window=window['id'],name=name,status='UNKNOWN',reason=str(exc)))

    def _window(self,engine,window):
        from ecodata.algorithms.composition_decision import Composition
        req=deepcopy(engine.request);req['composition'].pop('parallel_scoring',None)
        req['composition'].pop('response_profile',None)
        if 'fields' in req['composition']:
            self.probes.append(dict(window=window['id'],status='UNKNOWN',reason='Nested field request not supported by anchor probe'))
            return
        declaration=engine.config['reconstruction'];fine_id=declaration['fine_source']
        var=req['variable'];fine=engine.slice(engine.sources[fine_id],window)
        times=pd.DatetimeIndex(fine.time.values)
        hold=times[(times>pd.Timestamp(window['start'])) & (times<pd.Timestamp(window['end']))]
        retained=fine.sel(time=~fine.time.isin(hold.values))
        if not len(hold) or len(retained.time)<2 or hold.min()<pd.Timestamp(retained.time.values[0]) or hold.max()>pd.Timestamp(retained.time.values[-1]):
            self.probes.append(dict(window=window['id'],status='UNKNOWN',reason='No bracketing visible interior anchor; no extrapolation or invented response'))
            return
        truth=fine[var].sel(time=hold)
        if not np.isfinite(truth).all():
            self.probes.append(dict(window=window['id'],status='UNKNOWN',reason='Nonfinite probe cohort'))
            return
        scale=float(truth.std())
        if scale<=np.finfo(float).eps:
            self.probes.append(dict(window=window['id'],status='UNKNOWN',reason='Constant probe field has no data-derived normalization scale'))
            return
        req['target_times']=[str(t) for t in hold]
        req['composition']['windows']=[deepcopy(window)]
        parsed=[]
        for key in engine.ids:
            ds=retained if key==fine_id else engine.slice(engine.sources[key],window)
            parsed.append(('primary' if key==fine_id else 'background',ds))
        probe=Composition(parsed,req,engine.registry)
        if self.executions+probe.count>512:
            raise ValueError('Visible-anchor screening exceeds 512 real local probes; narrow the grammar explicitly')
        for i in range(probe.count):
            genome=probe.decode(i);plan=probe.plan(genome);key=self.local_key(plan,window['id'])
            if probe.preflight(genome)['status']=='REJECT':
                self.responses[(window['id'],key)]=dict(status='UNKNOWN',reason='Probe contract rejected')
                continue
            self.executions+=1
            try:
                ds,matching,trace,checks=probe.execute(genome)
                self.kernel_calls+=sum(t.get('kernel_calls',0) for t in trace)
                pred=ds[var].transpose(*truth.dims).values
                error=float(np.sqrt(np.mean((pred-truth.values)**2)))
                # Constituent controls use exactly the candidate's own chains,
                # with no fusion or post stage. Their errors measure actual
                # local complementarity, not a sum of library capability tags.
                branch_errors={}
                for branch in probe.ids:
                    chain=next(p['choice'] for p in plan if p['branch']==branch)
                    records=[];field=probe.fields[0]
                    part=probe.run_chain(probe.slice(probe.sources[branch],window),chain,field['contexts'][branch],records,[var,window['id'],branch])
                    part=probe.check_source(part,field,window)
                    self.kernel_calls+=sum(t.get('kernel_calls',0) for t in records)
                    branch_errors[branch]=float(np.sqrt(np.mean((part[var].transpose(*truth.dims).values-truth.values)**2)))
                baseline=min(branch_errors.values())
                gain=0. if baseline+error==0 else (baseline-error)/(baseline+error)
                lo,hi=matching['score_interval']
                response=dict(status='MEASURED_VISIBLE_MASK',routes=dict(
                    data=[1/(1+error/scale)]*2,
                    model=[float(np.clip((lo+1)/2,0,1)),float(np.clip((hi+1)/2,0,1))],
                    collaboration=[(1+gain)/2]*2),
                    anchor_rmse=error,normalization_scale=scale,constituent_rmse=branch_errors,
                    model_terms=matching['fields'][0]['terms'],
                    state_transitions=trace,coordination=checks,
                    product=dict(dims=list(truth.dims),values=pred.tolist()),
                    observation_metrics=probe.fields[0]['reconstruction'].metrics(ds))
            except (ValueError,KeyError,IndexError,TypeError) as exc:
                response=dict(status='UNKNOWN',reason='Real probe failed: '+str(exc))
            self.responses[(window['id'],key)]=response
        self.probes.append(dict(window=window['id'],status='MEASURED_VISIBLE_MASK',
            withheld_times=[str(t) for t in hold],retained_times=[str(t) for t in retained.time.values],
            fine_source=fine_id,truth=truth.values.tolist(),dims=list(truth.dims),
            candidate_local_graphs=probe.count,
            evidence='Truth removed from all probe execution sources and reconstruction state; coarse observations remain available'))
        if self.reference is not None:self._reference_window(engine,req,parsed,window,truth)

    @staticmethod
    def local_key(plan,window):
        return digest([dict(field=p['field'],branch=p['branch'],choice=p['choice']) for p in plan if p['window']==window])

    def score(self,plan):
        windows=sorted({p['window'] for p in plan});bounds={};states=[]
        for window in windows:
            response=self.responses.get((window,self.local_key(plan,window)))
            states.append(dict(window=window,response_key=self.local_key(plan,window),status=response['status'] if response else 'UNKNOWN'))
        for route in ROUTES:
            values=[]
            for window in windows:
                response=self.responses.get((window,self.local_key(plan,window)),{})
                values.append(response.get('routes',{}).get(route,[0.,1.]))
            bounds[route]=np.mean(values,axis=0).tolist()
        joint=[sum(self.weights[r]*bounds[r][i] for r in ROUTES) for i in (0,1)]
        common=None
        if self.reference is not None:
            measured=[self.responses.get((w,self.local_key(plan,w)),{}) for w in windows]
            references={}
            for name in self.reference['plans']:
                rs=[r for r in self.reference_probes if r['name']==name and r['status']=='PASS']
                if {r['window'] for r in rs}==set(windows):
                    references[name]=float(np.sqrt(np.average([r['rmse']**2 for r in rs],weights=[r['samples'] for r in rs])))
            common=dict(status='UNKNOWN',delta_rmse_K=None,stable_gain=False)
            if len(references)==len(self.reference['plans']) and all(r.get('status')=='MEASURED_VISIBLE_MASK' for r in measured):
                name=min(references,key=lambda k:(references[k],k));baseline=references[name]
                error=float(np.sqrt(np.average([r['anchor_rmse']**2 for r in measured],weights=[np.asarray(r['product']['values']).size for r in measured])))
                deltas=[next(x['rmse'] for x in self.reference_probes if x['name']==name and x['window']==w)-r['anchor_rmse'] for w,r in zip(windows,measured)]
                gain=0. if baseline+error==0 else (baseline-error)/(baseline+error)
                bounds['collaboration']=[(1+gain)/2]*2
                common=dict(status='MEASURED_VISIBLE_MASK',reference=name,reference_rmse=baseline,
                    candidate_rmse=error,delta_rmse_K=baseline-error,paired_window_delta=deltas,
                    stable_gain=all(d>0 for d in deltas),
                    scope='Positive improvement in every observed window only; not a statistical confidence or test guarantee')
            else:bounds['collaboration']=[0.,1.]
            # C never compensates a lower primary data/model score.
            joint=[sum(self.weights[r]*bounds[r][i] for r in ('data','model')) for i in (0,1)]
        return dict(score_interval=joint,routes=bounds,weights=self.weights,evidence=self.evidence,
            **({'common_gain':common} if common is not None else {}),
            state_responses=states,scope=self.scope,
            unknown_interactions=['Cross-window response and hidden-target error are not measured by local anchor probes'])

    def receipt(self):
        return dict(version='three_parallel_visible_anchor_v1',scope=self.scope,
            weights=self.weights,evidence=self.evidence,unsupported=getattr(self,'unsupported',None),
            probe_graph_executions=self.executions,probe_kernel_calls=self.kernel_calls,seconds=self.seconds,
            probes=self.probes,
            common_reference=getattr(self,'reference_evidence',None),reference_probes=self.reference_probes,
            responses=[dict(window=w,key=k,**v) for (w,k),v in self.responses.items()])


def reference_scope(engine):
    field=engine.fields[0]
    return dict(variable=field['variable'],unit=field['request']['unit'],
        sources={k:dict(shape=list(ds[field['variable']].shape),
            hours=(np.diff(pd.DatetimeIndex(ds.time.values)).astype('timedelta64[s]').astype(float)/3600).tolist()) for k,ds in engine.sources.items()},
        grammar=digest(dict(slots=engine.slots,axes=engine.axes)))


def gain_priority(row):
    gain=row.get('common_gain') or {}
    return (int(gain.get('stable_gain',False)),float(gain.get('delta_rmse_K') or 0.))
