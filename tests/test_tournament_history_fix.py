import numpy as np
import pytest
from ecodata.algorithms.search_stopping import SearchStopper
from ecodata.algorithms.search_accounting import SearchAccount,SearchBudgetExceeded
from ecodata.algorithms.development_validation import applicability
from test_development_validation import dataset


def test_tournament_batch_counts_one_round_but_keeps_all_numerical_cost():
    config=dict(enabled=True,min_evaluations=1,patience_evaluations=3,no_novel_iterations=5,min_relative_gain=.01)
    a=SearchStopper(config,1e-12,unit='numerical_mutation_round')
    b=SearchStopper(config,1e-12,unit='numerical_mutation_round')
    best=dict(metrics=dict(objective=1.))
    a.start(best);b.start(best)
    for round_number in (1,2):
        a.observe(round_number,best,numerical_evaluations=round_number*3)
        b.observe(round_number,best,numerical_evaluations=round_number)
        assert a.reason is b.reason is None
    a.observe(3,best,numerical_evaluations=9);b.observe(3,best,numerical_evaluations=3)
    assert a.reason==b.reason=='NO_MATERIAL_VALIDATION_GAIN'
    assert a.receipt()['new_numerical_evaluations']==9
    assert a.receipt()['completed_stopping_units']==3


def test_total_budget_includes_probe_failure_and_deployment():
    account=SearchAccount(100,30,probe_calls=100,total_calls=3);account.configure_deployment(100,30)
    for phase in ('search','probe','deploy'):
        token=account.before('test',phase,{})
        account.after(token,error=ValueError('fixture failure') if phase=='probe' else None)
    for phase in ('search','probe','deploy'):
        with pytest.raises(SearchBudgetExceeded):account.before('test',phase,{})
    assert account.receipt()['used']==account.receipt()['limit']==3
    assert account.receipt()['failed_calls']==1


def support():
    fine=dataset(2024,count=10,step=16);coarse=dataset(2024,count=20,step=8,sensor='terra-modis',n=2)
    visible=np.array([True,False,True,False,True,False,True,False,False,True])
    current=[fine.isel(time=np.flatnonzero(visible)),coarse]
    history=[dataset(2020,count=10,step=16),dataset(2020,count=20,step=8,sensor='terra-modis',n=2)]
    return current,history,fine.time.values,history[0].time.values,visible


@pytest.mark.parametrize('kind',['calendar','grid','distribution','season'])
def test_unsupported_history_is_rejected(kind):
    current,history,wanted,matched,visible=support()
    if kind=='calendar':matched=matched+np.timedelta64(20,'D')
    if kind=='grid':history[0]=history[0].assign_coords(lat=history[0].lat*2)
    if kind=='distribution':history[0].ndvi.values[:]=.9
    if kind=='season':
        current[1].ndvi.values[:]=np.linspace(.2,.5,20)[:,None,None]
        history[1].ndvi.values[:]=np.linspace(.5,.2,20)[:,None,None]
    result=applicability(current,history,wanted,matched,visible,'ndvi')
    assert result['status']!='PASS'


def test_applicability_does_not_use_hidden_fine_truth():
    args=support();a=applicability(*args,'ndvi')
    args[1][0].ndvi.values[~args[4]]=1000.
    b=applicability(*args,'ndvi')
    assert a==b and a['status']=='PASS' and a['gain_prediction']=='UNKNOWN'
