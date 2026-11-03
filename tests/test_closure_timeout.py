"""Simulate an overrun after a real mutation; final execution is still public-service work."""
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.field_mutation import GraphMutation
from test_current_input import setup


def test_search_deadline_overrun_does_not_consume_final_execution(monkeypatch,tmp_path):
    _,_,inputs,request=setup(monkeypatch,tmp_path)
    request['search_stopping']=dict(enabled=False)
    propose=GraphMutation.propose
    def expire_after_real_step(self,*args,**kwargs):
        result=propose(self,*args,**kwargs)
        # A cached/empty neighbourhood is not the real numerical mutation this
        # test promises to expire after.
        if self.transitions and self.transitions[-1]['actual_execution']:
            self.account.started-=1000
        return result
    monkeypatch.setattr(GraphMutation,'propose',expire_after_real_step)
    result=api.process_current_input(inputs,request,tmp_path/'run',seed=11,
        budget=dict(objective_limit=180,proposal_limit=12000,wall_seconds=60))
    assert result['status']=='PASS' and result['evolution_used']
    # The evidence-support gate may stop in the same iteration as the deadline;
    # neither reason is allowed to consume the fresh final-execution allowance.
    assert result['stop_reason'] in {'RESOURCE_BUDGET_EXHAUSTED','INSUFFICIENT_MATCHED_VALIDATION_FOR_PROMOTION'}
    assert result['current_input_deployment']['status']=='PASS'
    assert not result['current_input_deployment'].get('checkpoint_fallback',False)
    attempts=[a for a in result['current_input_deployment']['attempts'] if a['status']=='PASS']
    assert attempts and any(not t.get('cache_hit',False) for t in attempts[0]['trace'])
    # A qualified source-only graph legitimately has no registered numerical
    # node; do not manufacture a ledger call for copying its existing values.
    assert result['total_cost']['phases'].get('deploy',0)>0 or all(t['operation']=='source' for t in attempts[0]['trace'])
    with xr.open_dataset(result['product']) as output:assert output.ndvi.notnull().all()
