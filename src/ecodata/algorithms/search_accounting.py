"""One resource ledger shared by probes, search, failures and deployment."""
from time import perf_counter
from collections import Counter, defaultdict
import numpy as np


class SearchBudgetExceeded(RuntimeError):
    details = {'budget_exhausted': True}


class SearchAccount:
    def __init__(self, calls, seconds, *, probe_calls=None,total_calls=None):
        if total_calls is not None and (type(total_calls)!=int or total_calls<1):raise ValueError('Positive integer total-call limit required')
        self.total_limit=total_calls
        self.limit=int(calls);self.seconds=float(seconds);self.started=perf_counter()
        self.probe_limit=int(probe_calls if probe_calls is not None else calls//3)
        self.events=[];self.pending={};self.counts=Counter();self.timings=defaultdict(list)
        self.reserve=0
        self.deployment_time_reserve=0.
        self.deployment_limit=None;self.deployment_seconds=None;self.deployment_started=None
        self.reservations=[]

    def protect_deployment(self,calls,seconds):
        self.reserve=int(calls);self.deployment_time_reserve=float(seconds)
        self.reservations.append(dict(at_used=len(self.events),calls=self.reserve,seconds=self.deployment_time_reserve))

    def can_search(self,calls,reserve):
        return self.total_limit is None or len(self.events)+int(calls)+int(reserve)<=self.total_limit

    def configure_deployment(self,calls,seconds):
        if calls<1 or seconds<=0:raise ValueError('Positive deployment allowance required')
        self.deployment_limit=int(calls);self.deployment_seconds=float(seconds)

    def begin_deployment(self):
        if self.deployment_limit is not None:self.deployment_started=perf_counter()

    def exhausted(self, phase='search'):
        if self.total_limit is not None and len(self.events)>=self.total_limit-(0 if phase in {'deploy','validation_audit'} else self.reserve):return True
        if self.deployment_limit is not None:
            if phase in {'deploy','validation_audit'}:
                return (self.counts['deploy']>=self.deployment_limit or
                    (self.deployment_started is not None and perf_counter()-self.deployment_started>=self.deployment_seconds))
            return (len(self.events)-self.counts['deploy']>=self.limit or perf_counter()-self.started>=self.seconds-self.deployment_time_reserve
                    or (phase=='probe' and self.counts['probe']>=self.probe_limit))
        usable=self.limit if phase=='deploy' else self.limit-self.reserve
        seconds=self.seconds if phase=='deploy' else self.seconds-self.deployment_time_reserve
        return (len(self.events)>=usable or perf_counter()-self.started>=seconds
                or (phase=='probe' and self.counts['probe']>=self.probe_limit))

    def before(self, algorithm, phase, parameters,*,graph=None,requires_training=False):
        if self.exhausted(phase):raise SearchBudgetExceeded('Shared numerical-call or service-time budget exhausted: '+phase)
        event=dict(algorithm=algorithm,phase=phase,graph=graph,parameters={k:v for k,v in parameters.items() if k!='mode'},
                   status='STARTED',fit_adjust=algorithm.startswith('bias.xsdba_'),requires_training=bool(requires_training))
        self.events.append(event);self.counts[phase]+=1
        token=len(self.events)-1;self.pending[token]=perf_counter()
        return token

    def after(self, token, *, error=None):
        event=self.events[token];event['seconds']=perf_counter()-self.pending.pop(token)
        event['status']='FAIL' if error else 'PASS'
        if error:event['error']=str(error)
        else:self.timings[event['algorithm']].append(event['seconds'])

    def estimated_cost(self, operations):
        if any(op not in self.timings for op in operations):return float('inf')
        return float(sum(np.median(self.timings[op]) for op in operations))

    def receipt(self):
        return dict(limit=self.total_limit if self.total_limit is not None else self.limit+(self.deployment_limit or 0),total_call_limit=self.total_limit,search_limit=self.limit,
             deployment_limit=self.deployment_limit,deployment_seconds=self.deployment_seconds,
             used=len(self.events),probe_limit=self.probe_limit,
             deployment_time_reserve_seconds=self.deployment_time_reserve,
             deployment_call_reserve=self.reserve,reservation_history=self.reservations,
             phases=dict(self.counts),service_seconds=perf_counter()-self.started,
             fit_adjust_calls=sum(e['fit_adjust'] for e in self.events),
             training_capable_calls=sum(e['requires_training'] for e in self.events),
             failed_calls=sum(e['status']=='FAIL' for e in self.events),events=self.events,
             cache_rule='Cached node traversal has no ledger event; all new calls, including failed and probe calls, count',
             time_rule='Search wall time includes proposal/portrait/quality overhead; configured deployment has its own protected deadline; all calls remain in the shared total limit; an in-flight kernel may overrun before returning')
