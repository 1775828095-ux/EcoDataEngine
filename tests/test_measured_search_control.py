"""Regression evidence for usable validation continuing real random search."""
import json
import numpy as np
import pytest
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.reliable_selection import make_policy,aggregate,protected_order
from test_current_input import setup


def test_budget_reserves_a_complete_fold_round_before_any_new_proposal():
    from types import SimpleNamespace
    from time import perf_counter
    from ecodata.algorithms.field_mutation import GraphMutation
    from ecodata.algorithms.search_accounting import SearchAccount
    account=SearchAccount(10,60,total_calls=10);account.protect_deployment(4,1)
    for _ in range(5):account.after(account.before('method','search',{}))
    assert not account.exhausted()
    search=GraphMutation.__new__(GraphMutation)
    search.account=account;search.stopper=SimpleNamespace(reason=None)
    search.attempts=search.iterations=0;search.started=perf_counter()
    search.budget=dict(proposal_limit=100,wall_seconds=60)
    search.folds=[{},{}]
    assert search.exhausted()  # Cannot spend the final call on only half a fold round.
    search.folds=[{}]
    assert not search.exhausted()


def test_random_driver_closes_after_unaffordable_families_without_losing_best():
    from types import SimpleNamespace
    from ecodata.algorithms.composition_optimizers import random_graph_search
    class Choice:
        index=0
        def choice(self,values):
            if len(values)==2:return 'random'
            value=values[self.index%len(values)];self.index+=1;return value
    class Stopper:
        reason=None
        def observe(self,*args,**kwargs):pass
    class Search:
        rng=Choice();stopper=Stopper();account=SimpleNamespace(events=[])
        numerical_rounds=novel_executions=0;proposal_budget_blocked=False;calls=0
        def best(self):return dict(key='retained_best',metrics=dict(objective=.1))
        def exhausted(self):return self.stopper.reason is not None or self.calls>10
        def destroy(self,*args):return {}
        def propose(self,*args,**kwargs):self.calls+=1;self.proposal_budget_blocked=True
    s=Search();random_graph_search(s,{},dict(portraits_enabled=False))
    assert s.calls==5 and s.stopper.reason=='RESOURCE_BUDGET_EXHAUSTED'
    assert s.best()['key']=='retained_best' and not s.account.events


@pytest.mark.parametrize('bad',['incomplete','different_folds','different_loss','failed','dominated'])
def test_invalid_or_incomparable_candidate_never_displaces_incumbent(bad):
    p=make_policy({},[0.,1.])
    def row(key,v):
        m=aggregate([dict(fold_id='a',status='PASS',metrics=dict(RMSE=v))],p,['a'])
        return dict(key=key,metrics=m,status='ELIGIBLE_OUTPUT',algorithm_nodes=1)
    base=row('base',1.);trial=row('trial',.1)
    if bad=='incomplete':trial['metrics']['complete_required_folds']=False
    elif bad=='different_folds':trial['metrics']['fold_ids']=['other']
    elif bad=='different_loss':trial['metrics']['loss_signature']='other'
    elif bad=='failed':trial['status']='EXECUTION_FAILED'
    else:trial['dominated']=True
    assert protected_order([trial,base],p,base,support_sufficient=False)[0]['key']=='base'


def test_public_limited_support_continues_and_initial_is_measured_best(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8)
    # An always-absent boundary component deliberately makes support limited.
    fine.ndvi.values[:,0,:2]=np.nan
    fine.ndvi.values[4,3:5,3:5]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    q.update(soft_guidance=False,search_stopping=dict(enabled=False))
    result=api.process_current_input(inputs,q,tmp_path/'public',seed=11,
        budget=dict(objective_limit=240,total_call_limit=260,proposal_limit=2000,wall_seconds=60))
    assert result['status']=='PASS' and result['self_validation_protocol']['support_limited']
    assert result['task_acceptance']['status']=='EVIDENCE_INSUFFICIENT'
    assert result['stop_reason']!='INSUFFICIENT_MATCHED_VALIDATION_FOR_PROMOTION'
    transitions=json.loads((tmp_path/'public/mutation_trace.json').read_text(encoding='utf-8'))
    assert len(transitions)>1 and sum(t['actual_execution'] for t in transitions)>1
    assert all(t['strategy']=='random' for t in transitions)
    attempts=[json.loads(s) for s in (tmp_path/'public/attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    initial=[]
    for r in attempts:
        if r['origin']=='mutation':break
        if r['status']=='ELIGIBLE_OUTPUT' and not r.get('dominated'):initial.append(r)
    tolerance=max(result['reliable_selection']['roundoff_tolerance'],result['reliable_selection']['delta'])
    assert result['initial_best']['metrics']['objective']<=min(r['metrics']['objective'] for r in initial)+tolerance
    final=result['best']['metrics']
    assert final['objective']<=result['initial_best']['metrics']['objective']+tolerance
    assert final['complete_required_folds'] and result['final_validation_best_delivered']


def test_zero_valid_folds_do_not_start_random_search(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=2)
    q['target_times']=[str(fine.time.values[0]),str(fine.time.values[0]+np.timedelta64(4,'D')),str(fine.time.values[1])]
    result=api.process_current_input(inputs,q,tmp_path/'invalid',seed=11)
    assert result['status']=='VALIDATION_EVIDENCE_INSUFFICIENT' and result['product'] is None
    assert not (tmp_path/'invalid/mutation_trace.json').exists()
