"""Explicit synthetic boundary tests, not environmental evidence."""
import hashlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.evaluation.blocked_confirmation import score,review,compare,apply_recommendation_isolation


def metrics(errors,start="2000-01-01"):
    clock=pd.date_range(start,periods=len(errors)*3,freq="h")
    ref=xr.DataArray(np.zeros(len(clock)),dims="time",coords=dict(time=clock),attrs=dict(units="K"),name="t")
    prediction=xr.Dataset(dict(t=("time",np.repeat(errors,3))),coords=dict(time=clock))
    prediction.t.attrs["units"]="K"
    blocks=[dict(id=str(i),indices=list(range(3*i,3*i+3)),times=[str(t) for t in clock[3*i:3*i+3]]) for i in range(len(errors))]
    plan=dict(blocks=blocks,axes=["time"],sufficient=True,fingerprint=hashlib.sha256(str(clock).encode()).hexdigest())
    return score(prediction,ref,"t",plan)


def quartet(errors):
    return metrics(errors),metrics([2]*len(errors)),metrics(errors,"2000-02-01"),metrics([2]*len(errors),"2000-02-01")


def test_small_local_degradation_can_pass_with_explicit_task_risk_bound():
    a,b,c,d=quartet([1]*9+[2.1])
    risk=dict(unit="K",max_rmse_increase=.2,max_mae_increase=.2,basis="Synthetic task explicit allowed error; not a recommended real threshold")
    result=review(a,b,c,d,quality_pass=True,risk_limits=risk)
    assert result['recommended'] and result['selection']['local_risk']['degraded_blocks']==1
    assert result['confirmation']['uncertainty']['upper']<0
    assert review(a,b,c,d,quality_pass=True)['status']=='TRIAL_AVAILABLE'
    assert review(a,b,c,d,quality_pass=True,policy_version='blocked_confirmation_v1')['status']=='NO_STABLE_SELECTION_GAIN'


def test_declared_severe_local_harm_is_rejected_despite_average_gain():
    a,b,c,d=quartet([0]*9+[3])
    result=review(a,b,c,d,quality_pass=True,risk_limits=dict(unit="K",max_rmse_increase=.2,basis="Explicit synthetic task constraint"))
    assert result['status']=='LOCAL_RISK_FAILED'
    assert result['selection']['aggregate_gain']
    assert not result['recommended']


@pytest.mark.parametrize('risk',[dict(unit='mm',max_rmse_increase=1,basis='wrong unit'),dict(unit='K',max_rmse_increase=1),dict(unit='K',max_rmse_increase=-1,basis='invalid')])
def test_risk_bounds_need_evidence_unit_and_nonnegative_value(risk):
    with pytest.raises(ValueError):review(*quartet([1]*10),quality_pass=True,risk_limits=risk)


def test_overall_deterioration_not_rescued_by_good_blocks():
    assert review(*quartet([1]*8+[10]*2),quality_pass=True)['status']=='NO_STABLE_SELECTION_GAIN'
    assert review(*quartet([1]*10),quality_pass=False)['status']=='QUALITY_FAILED'


def test_isolation_limit_does_not_hide_real_confirmation_failure():
    a,b,_,d=quartet([1]*10)
    gate=review(a,b,metrics([3]*10,'2000-02-01'),d,quality_pass=True)
    apply_recommendation_isolation(gate,dict(status='INSUFFICIENT'))
    assert gate['status']=='CONFIRMATION_FAILED' and gate['evidence_status']=='INSUFFICIENT'


def test_unknown_local_tolerance_is_not_zero_or_a_guessed_percent():
    result=review(*quartet([1]*9+[2.1]),quality_pass=True)
    assert result['decision']=='TRIAL'
    assert result['selection']['local_risk']['status']=='UNASSESSED_TOLERANCE'
    assert 'limits' not in result['selection']['local_risk']


def test_spatial_dates_do_not_multiply_independent_region_units():
    from ecodata.evaluation.blocked_confirmation import spatial_metrics
    def snapshot(error):
        return dict(blocks=[dict(date=str(date),region=region,metrics=dict(RMSE=error,MAE=error,Bias=error,count=10))
                            for date in pd.date_range("2020-01-01",periods=8) for region in range(4)])
    def normal(report):
        return spatial_metrics(report,reference_key="same",variable="t",unit="K",locations=["a","b"],
                               times=pd.date_range("2020-01-01",periods=8),axes=["space"],phase="selection")
    comparison=compare(normal(snapshot(1)),normal(snapshot(2)))
    assert comparison['uncertainty']['units']==4
    assert comparison['local_risk']['total_blocks']==32


def test_public_explanation_excludes_private_block_identifiers():
    import json
    from ecodata.evaluation.blocked_confirmation import public_summary
    gate=review(*quartet([0]*9+[3]),quality_pass=True,
        risk_limits=dict(unit="K",max_rmse_increase=0,basis="Synthetic task"))
    gate['selection']['local_risk']['violations'][0]['block']='private-reference-block'
    assert 'private-reference-block' not in json.dumps(public_summary(gate))
