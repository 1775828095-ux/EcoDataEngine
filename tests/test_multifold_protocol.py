import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.current_input import raster_validation,raster_validation_folds
from ecodata.algorithms.autonomous_field import build_context
from ecodata.algorithms.profile_matching import prepare_model_request
from test_current_input import setup


def context(inputs,request):
    request,_=prepare_model_request(request)
    return request,build_context(inputs,request)[0]


def test_five_visible_dates_produce_three_loo_folds_with_four_retained(monkeypatch,tmp_path):
    original,coarse,inputs,q=setup(monkeypatch,tmp_path,t=10)
    fine=original.isel(time=[0,2,4,6,9]).copy(deep=True)
    fine.ndvi.values[2,0,0]=np.nan
    untouched=fine.copy(deep=True)
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    inputs.append(api.SearchInput('alias.nc','background',fine.to_netcdf()))
    request,ctx=context(inputs,q)
    folds,protocol=raster_validation_folds(inputs,request,ctx)
    assert len(folds)==protocol['fold_count']==3
    assert protocol['design']=='whole_date_leave_one_out'
    assert protocol['task_profile']['visible_effective_dates']==5
    assert protocol['task_profile']['requested_missing_dates']==5
    assert not protocol['confirmation_available'] and not protocol['independent_confirmation_available']
    assert protocol['support_level']=='LOCAL_CROSS_VALIDATION_ONLY' and protocol['support_limited']
    held=[]
    for fold in folds:
        assert fold['confirmation'] is None
        ref=fold['validation'].reference
        assert len(ref.time)==1 and ref.attrs['split']=='search_validation'
        held.extend(ref.time.values.tolist())
        parsed,_=api.parse_inputs(fold['inputs'])
        mask=np.isin(fine.time.values,ref.time.values)
        assert np.isnan(parsed[0][1].ndvi.values[mask]).all()
        assert np.isnan(parsed[2][1].ndvi.values[mask]).all()
        np.testing.assert_array_equal(parsed[0][1].ndvi.values[~mask],fine.ndvi.values[~mask])
        np.testing.assert_array_equal(parsed[1][1].ndvi.values,coarse.ndvi.values)
        assert int(np.isfinite(parsed[0][1].ndvi).any(('lat','lon')).sum())==4
        np.testing.assert_array_equal(ref.values,fine.ndvi.sel(time=ref.time).values)
        assert fold['protocol']['retained_effective_dates']==4
        assert fold['protocol']['masked_input_indices']==[0,2]
        assert len(fold['protocol']['input_sha256'])==3 and len(fold['id'])==64
        assert not fold['protocol']['external_test_used']
    assert len(set(held))==3
    xr.testing.assert_equal(fine,untouched)


def test_folds_repeat_and_cache_identity_includes_input_and_goal(monkeypatch,tmp_path):
    original,_,inputs,q=setup(monkeypatch,tmp_path,t=10)
    fine=original.isel(time=[0,2,4,6,9])
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    request,ctx=context(inputs,q)
    first,p1=raster_validation_folds(inputs,request,ctx)
    second,p2=raster_validation_folds(inputs,request,ctx)
    assert p1==p2 and len({f['id'] for f in first})==3
    for a,b in zip(first,second):
        assert a['id']==b['id']
        xr.testing.assert_equal(a['validation'].reference,b['validation'].reference)
    q2=dict(request,target_times=[*request['target_times'],str(original.time.values[-1]+np.timedelta64(1,'D'))])
    third,_=raster_validation_folds(inputs,q2,ctx)
    assert [f['id'] for f in first]!=[f['id'] for f in third]


@pytest.mark.parametrize('value',[0,-1,True,1.2,'3'])
def test_fold_limit_must_be_a_positive_integer(monkeypatch,tmp_path,value):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=10)
    inputs[0]=api.SearchInput('fine.nc','primary',fine.isel(time=[0,1,2,4,5,6,7,8,9]).to_netcdf())
    q['current_input_validation']=dict(max_folds=value)
    request,ctx=context(inputs,q)
    with pytest.raises(ValueError,match='Positive integer max_folds'):
        raster_validation_folds(inputs,request,ctx)


def test_additional_folds_are_explicit_and_chosen_without_label_ranking(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=10)
    observed=fine.isel(time=[0,1,2,4,5,6,7,8,9])
    inputs[0]=api.SearchInput('fine.nc','primary',observed.to_netcdf())
    request,ctx=context(inputs,q)
    folds,_=raster_validation_folds(inputs,request,ctx)
    assert len(folds)==3
    request=dict(request,current_input_validation=dict(max_folds=6))
    more,_=raster_validation_folds(inputs,request,ctx)
    assert len(more)==6
    assert all(f['protocol']['retained_effective_dates']==8 for f in more)
    held={str(f['validation'].reference.time.values[0]) for f in more}
    assert str(observed.time.values[0]) not in held and str(observed.time.values[-1]) not in held


def test_at_least_two_effective_anchor_dates_and_no_natural_missing_labels(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=6)
    observed=fine.isel(time=[0,2,5]).copy(deep=True)
    inputs[0]=api.SearchInput('fine.nc','primary',observed.to_netcdf())
    request,ctx=context(inputs,q)
    folds,_=raster_validation_folds(inputs,request,ctx)
    assert len(folds)==1 and folds[0]['protocol']['retained_effective_dates']==2
    observed.ndvi.values[1]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',observed.to_netcdf())
    request,ctx=context(inputs,q)
    with pytest.raises(ValueError,match='need >=3'):
        raster_validation_folds(inputs,request,ctx)


def test_point_tasks_reuse_legacy_labels_and_confirmation(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path,t=6)
    request,ctx=context(inputs,q)
    changed,validation,legacy,confirmation=raster_validation(inputs,request,ctx)
    folds,protocol=raster_validation_folds(inputs,request,ctx)
    assert len(folds)==1 and protocol['confirmation_available']
    assert protocol['design']=='point_missing' and not protocol['independent_confirmation_available']
    xr.testing.assert_equal(folds[0]['validation'].reference,validation.reference)
    xr.testing.assert_equal(folds[0]['confirmation'],confirmation)
    first,_=api.parse_inputs(folds[0]['inputs']);second,_=api.parse_inputs(changed)
    for (_,a),(_,b) in zip(first,second):xr.testing.assert_equal(a,b)


def test_full_missing_input_frames_are_recognised_as_temporal_targets(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=5)
    fine.ndvi.values[2]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    request,ctx=context(inputs,q)
    folds,protocol=raster_validation_folds(inputs,request,ctx)
    assert protocol['design']=='whole_date_leave_one_out'
    assert protocol['task_profile']['requested_missing_dates']==1
    assert len(folds)==2
    assert all(str(fine.time.values[2]) not in f['protocol']['held_dates'] for f in folds)
