"""Controlled functional tests, not environmental accuracy evidence."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms.platform_search import BlockedReference,BlockedSearchValidation,SearchValidation
from ecodata.evaluation.blocked_confirmation import time_blocks,review


def data(n=12):
    a=xr.DataArray(np.ones((n,1,1))*10,dims=('time','lat','lon'),
        coords=dict(time=pd.date_range('2000-01-01',periods=n,freq='h'),lat=[1.],lon=[2.]),
        attrs=dict(units='K',split='search_validation'),name='t')
    return a,a.to_dataset().copy(deep=True)


def held_scores(ref,*datasets):
    clock=ref.time+np.timedelta64(1,'D')
    evaluator=BlockedReference(ref.assign_coords(time=clock),'t')
    return [evaluator(ds.assign_coords(time=clock)) for ds in datasets]


def test_block_objective_actually_differs_from_pooled_error():
    ref,ds=data(8);evaluator=BlockedSearchValidation(ref,'t')
    ds.t.values[-3:]+=3
    metrics=evaluator(ds)
    assert metrics['objective']==pytest.approx(1)
    assert metrics['RMSE']==pytest.approx(np.sqrt(27/8))
    assert metrics['block_support']['sufficient']
    assert SearchValidation(ref,'t')(ds)['objective']==metrics['RMSE']


def test_duplicate_or_bad_time_rejected_and_us_ns_same_windows():
    ref,_=data()
    us=ref.assign_coords(time=ref.time.values.astype('datetime64[us]'))
    assert time_blocks(ref)==time_blocks(us)
    with pytest.raises(ValueError,match='unique'):time_blocks(ref.isel(time=[0,0,1]))
    with pytest.raises(ValueError,match='increasing'):time_blocks(ref.isel(time=slice(None,None,-1)))
    with pytest.raises(ValueError,match='Explicit datetime'):time_blocks(ref.assign_coords(time=np.arange(len(ref.time))))


def test_irregular_times_use_elapsed_windows_not_assumed_hourly_rows():
    ref,_=data(9)
    times=pd.Timestamp('2000-01-01')+pd.to_timedelta([0,1,2,20,21,22,40,41,42],unit='h')
    plan=time_blocks(ref.assign_coords(time=times))
    assert [r['indices'] for r in plan['blocks']]==[[0,1,2],[3,4,5],[6,7,8]]


def test_every_prediction_required_and_no_successful_subset():
    ref,ds=data();evaluator=BlockedSearchValidation(ref,'t')
    ds.t.values[4]=np.nan
    with pytest.raises(ValueError,match='Incomplete'):evaluator(ds)


def test_local_degradation_is_disclosed_not_an_automatic_rejection():
    ref,ds=data();baseline=ds.copy(deep=True);baseline.t.values[:]+=2
    candidate=ds.copy(deep=True);candidate.t.values[-4:]+=3
    evaluator=BlockedSearchValidation(ref,'t')
    a,b=evaluator(candidate),evaluator(baseline)
    assert a['RMSE']<b['RMSE']
    c,d=held_scores(ref,candidate,baseline)
    decision=review(a,b,c,d,quality_pass=True)
    assert decision['status']=='TRIAL_AVAILABLE' and not decision['recommended']
    assert decision['selection']['aggregate_gain']
    assert decision['selection']['local_risk']['degraded_blocks']==1
    old=review(a,b,c,d,quality_pass=True,policy_version='blocked_confirmation_v1')
    assert old['status']=='NO_STABLE_SELECTION_GAIN'


def test_one_shot_confirmation_failure_and_stable_pass():
    ref,ds=data();baseline=ds.copy(deep=True);baseline.t.values[:]+=2
    evaluator=BlockedSearchValidation(ref,'t')
    a,b=evaluator(ds),evaluator(baseline)
    bad=ds.copy(deep=True);bad.t.values[:]+=3
    c,d=held_scores(ref,bad,baseline)
    failed=review(a,b,c,d,quality_pass=True)
    assert failed['status']=='CONFIRMATION_FAILED' and failed['confirmation_retries']==0
    assert not failed['confirmation_used_for_search']
    c,d=held_scores(ref,ds,baseline)
    good=review(a,b,c,d,quality_pass=True)
    assert good['recommended']
    assert review(a,b,c,d,quality_pass=False)['status']=='QUALITY_FAILED'


def test_two_times_cannot_be_promoted_to_sufficient_evidence():
    ref,ds=data(2);evaluator=BlockedSearchValidation(ref,'t')
    base=ds.copy(deep=True);base.t.values[:]+=2
    a,b=evaluator(ds),evaluator(base)
    c,d=held_scores(ref,ds,base)
    decision=review(a,b,c,d,quality_pass=True)
    assert decision['status']=='TRIAL_AVAILABLE'
    assert decision['evidence_status']=='INSUFFICIENT'
    assert not decision['recommended']


def test_baseline_queue_mismatch_rejected():
    a,ds=data();metrics=BlockedSearchValidation(a,'t')(ds)
    changed,other=data(15);comparison=BlockedSearchValidation(changed,'t')(other)
    with pytest.raises(ValueError,match='queues'):review(metrics,comparison,metrics,metrics,quality_pass=True)


def test_reusing_selection_for_confirmation_is_rejected():
    ref,ds=data();metrics=BlockedSearchValidation(ref,'t')(ds)
    with pytest.raises(ValueError,match='disjoint'):review(metrics,metrics,metrics,metrics,quality_pass=True)


def test_blocked_objective_changes_actual_executor_ranking(tmp_path):
    from ecodata.algorithms.contracts import AlgorithmSpec,AlgorithmResult,Compatibility
    from ecodata.algorithms.registry import AlgorithmRegistry
    from ecodata.algorithms.field_mutation import GraphMutation
    from ecodata.core.schemas import EcoDataBundle
    from test_series_fusion import data as series,context
    class Curve:
        def __init__(self,name,values):
            self.values=values
            self.spec=AlgorithmSpec(name,name,'1','test','available',('time',),('continuous',),
                input_contract=dict(source_arity=1,parameter_domains={},deterministic=True,state_complete=True))
        def compatibility(self,context):return Compatibility(True,())
        def run(self,context):
            assert context.validation_reference is None
            ds=context.sources[0].dataset.copy(deep=True)
            ds.precipitation.values[:,0,0]=self.values
            return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(functional_fixture=True))
    source=series([0]*8);reference=series([10]*8).precipitation
    reference.attrs['split']='search_validation'
    registry=AlgorithmRegistry()
    for plugin in [Curve('test.a',[10]*5+[13]*3),Curve('test.b',[11.5]*8)]:registry.register(plugin)
    for name,evaluator,expected in [('blocked',BlockedSearchValidation(reference,'precipitation'),'test.a'),('legacy',None,'test.b')]:
        folder=tmp_path/name;folder.mkdir()
        engine=GraphMutation(context(source),registry,dict(soft_guidance=False,response_scoring=dict(enabled=False)),{},
            reference,folder,dict(objective_limit=10,proposal_limit=30,wall_seconds=10),11,evaluator=evaluator)
        try:
            for operation in ['test.a','test.b']:
                key=engine.intern(operation,['s0'],{});assert engine.execute(key,'functional_check')['status']=='ELIGIBLE_OUTPUT'
            assert engine.records[engine.best()['key']]['operation']==expected
        finally:engine.close()
