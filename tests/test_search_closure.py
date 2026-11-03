import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.search_stopping import SearchStopper
from ecodata.algorithms.search_accounting import SearchAccount
from test_current_input import setup
from test_field_mutation import engine


def declaration(**changes):
    return dict(enabled=True,min_evaluations=2,patience_evaluations=3,no_novel_iterations=4,min_relative_gain=.01,**changes)


def row(error):return dict(metrics=dict(objective=error))


def test_stagnation_counts_real_evaluations_not_cached_aliases():
    stop=SearchStopper(declaration(),1e-12);stop.start(row(1.))
    stop.observe(1,row(.995));stop.observe(2,row(.989))
    assert stop.stale==0  # accumulated improvement exceeds threshold
    stop.observe(3,row(.988));stop.observe(3,row(.988))
    assert stop.stale==1
    stop.observe(4,row(.987));stop.observe(5,row(.986))
    assert stop.reason=='NO_MATERIAL_VALIDATION_GAIN'


def test_no_novel_and_explicit_quality_target_stop():
    stop=SearchStopper(declaration(),1e-12);stop.start(row(1.))
    for _ in range(4):stop.observe(0,row(1.))
    assert stop.reason=='NO_NEW_NUMERICAL_EVALUATIONS'
    stop=SearchStopper(dict(declaration(),target_error=.1),1e-12);stop.start(row(.05))
    assert stop.reason=='VALIDATION_TARGET_REACHED'
    with pytest.raises(ValueError):SearchStopper(dict(declaration(),patience_evaluations=0),1e-12)


def test_deployment_has_fresh_deadline_and_all_calls_are_counted(monkeypatch):
    from ecodata.algorithms import search_accounting as module
    clock=[0.];monkeypatch.setattr(module,'perf_counter',lambda:clock[0])
    account=SearchAccount(1,1);account.configure_deployment(2,3)
    token=account.before('method','search',{});account.after(token)
    clock[0]=10.;assert account.exhausted('search')
    account.begin_deployment();assert not account.exhausted('deploy')
    token=account.before('method','deploy',{});account.after(token)
    assert account.receipt()['used']==2 and account.receipt()['limit']==3
    clock[0]=14.;assert account.exhausted('deploy')


def test_repeated_edit_skips_compatibility_and_preserves_unselected_candidate(tmp_path,monkeypatch):
    search=engine(tmp_path)
    try:
        spec=search.registry.get('fusion.series_missing_fallback').spec
        plugin=search.registry.get(spec.id);original=plugin.compatibility;calls=[]
        def counted(ctx):calls.append(1);return original(ctx)
        monkeypatch.setattr(plugin,'compatibility',counted)
        first=search.candidate(None,None,spec,['s0','s1'],dict(primary_index=0),'test')
        second=search.candidate(None,None,spec,['s0','s1'],dict(primary_index=0),'test')
        assert first and first['child']==second['child'] and len(calls)==1
        assert search.duplicate_proposals==1
        search.execute(first['child'],'mutation')
        assert search.candidate(None,None,spec,['s0','s1'],dict(primary_index=0),'test') is None
    finally:search.close()


def test_rank_is_independent_of_observed_runtime(tmp_path):
    search=engine(tmp_path)
    try:
        a=dict(key='a',metrics=dict(objective=1.),algorithm_nodes=2,estimated_cost=100.)
        b=dict(key='b',metrics=dict(objective=1.),algorithm_nodes=2,estimated_cost=.001)
        assert search.rank(a)<search.rank(b)
    finally:search.close()


def test_current_public_service_repeatable_with_early_stop(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path)
    q['search_stopping']=declaration()
    budget=dict(objective_limit=160,probe_call_limit=0,proposal_limit=12000,wall_seconds=60)
    a=api.process_current_input(inputs,q,tmp_path/'a',budget=budget,seed=29)
    from ecodata.interface.search_page import process_search_upload
    b=process_search_upload(inputs,dict(q,decision_policy='current_input'),None,tmp_path/'b',budget=budget,seed=29)
    assert a['status']==b['status']=='PASS'
    assert a['best']['key']==b['best']['key'] and a['stop_reason']==b['stop_reason']
    assert a['search_stopping']['new_numerical_evaluations']==b['search_stopping']['new_numerical_evaluations']
    assert a['checkpoint_available'] and a['total_cost']['deployment_limit']==160
    with xr.open_dataset(a['product']) as x,xr.open_dataset(b['product']) as y:xr.testing.assert_equal(x.load(),y.load())


def test_final_deployment_failure_returns_real_checkpoint(monkeypatch,tmp_path):
    from ecodata.algorithms import current_input
    _,_,inputs,q=setup(monkeypatch,tmp_path);q['search_stopping']=declaration()
    original=current_input.deploy_ranked;calls=[]
    def fail_final(*args):
        calls.append(1)
        if len(calls)>1:return None,None,dict(status='NO_EXECUTABLE_CURRENT_GRAPH',attempts=[dict(status='REJECTED',error='simulated in-flight timeout')])
        return original(*args)
    monkeypatch.setattr(current_input,'deploy_ranked',fail_final)
    result=api.process_current_input(inputs,q,tmp_path/'run',budget=dict(objective_limit=140,probe_call_limit=0,proposal_limit=8000,wall_seconds=60),seed=11)
    assert result['status']=='PASS' and result['current_input_deployment']['checkpoint_fallback']
    with xr.open_dataset(result['product']) as f,xr.open_dataset(tmp_path/'run/checkpoint.nc') as c:xr.testing.assert_equal(f.load(),c.load())
