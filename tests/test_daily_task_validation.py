from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,measured
from ecodata.algorithms.task_validation import plan
from ecodata.algorithms.profile_matching import prepare_model_request
from test_series_fusion import data,context


def daily_fixture():
    time=pd.date_range('2000-01-01','2001-12-31')
    a=data(np.where(time.month==7,10.,0.),times=time)
    mask=((time.month==1)&(time.day>=15)&(time.day<=17))|((time.month==7)&(time.day>=15)&(time.day<=28))
    a.precipitation.values[mask]=np.nan
    q=dict(task='multisource',variable='precipitation',unit='mm/day',semantic_type='continuous',nonnegative=True,
        target_lat=[11.6],target_lon=[37.36],target_times=[str(t) for t in time],parameters={},
        model_profile=dict(id='swat_plus',role='weather',field='precipitation'),
        portrait_policy=dict(version='robit-era5-multisource-swatplus-v1',scenario='daily-recovery'),
        scoring=dict(registry_version='robit-daily-state-v1'))
    return a,[api.SearchInput('primary.nc','primary',a.to_netcdf())],q


def test_daily_blocks_match_both_seasons_and_lengths_without_hidden_context():
    a,inputs,q=daily_fixture()
    inputs.append(api.SearchInput('alias.nc','background',a.to_netcdf()))
    folds,p=plan(inputs,q,build_context(inputs,q)[0])
    assert p['kind']=='task_matched_series' and not p['support_limited'] and len(folds)==3
    assert p['tasks']==['temporal_gap'] and p['isolation_applied']
    masks=[]
    for fold in folds:
        detail=fold['protocol'];assert detail['all_requested_gaps_matched']
        assert {b['requested_gap']['missing_dates'] for b in detail['blocks']}=={3,14}
        assert {pd.Timestamp(b['requested_gap']['start']).month for b in detail['blocks']}=={1,7}
        assert all(len(b['held_dates'])==b['requested_gap']['missing_dates'] for b in detail['blocks'])
        changed,_=api.parse_inputs(fold['inputs']);root=changed[0][1]
        xr.testing.assert_equal(root,changed[1][1])
        for block in detail['blocks']:
            assert block['calendar_distance_days']<=block['requested_gap']['bracket_days']
            assert np.isfinite(root.precipitation.sel(time=block['retained_anchor_dates'])).all()
        ref=fold['validation'].reference
        assert np.isnan(root.precipitation.sel(time=ref.time)).all()
        assert np.isfinite(a.precipitation.sel(time=ref.time)).all()
        masks.append(tuple(ref.time.values))
    assert len(set(masks))==3


@pytest.mark.parametrize('kind',['complete','single','long','boundary'])
def test_all_station_shapes_use_task_planner_or_honest_limited_support(kind):
    a,_,q=daily_fixture();a.precipitation.values[:]=1.
    if kind=='single':a.precipitation.values[200]=np.nan
    if kind=='long':a.precipitation.values[185:199]=np.nan
    if kind=='boundary':a.precipitation.values[:14]=np.nan
    items=[api.SearchInput('daily.nc','primary',a.to_netcdf())]
    folds,p=plan(items,q,build_context(items,q)[0])
    assert folds and p['kind']=='task_matched_series'
    assert p['support_limited']==(kind=='boundary')
    if kind=='boundary':assert p['task_support'][0]['gaps'][0]['status']=='INSUFFICIENT'


def test_default_metric_is_true_rmse_and_can_reverse_mae_ranking():
    ref=data([0,0]).precipitation
    a=data([0,4]);b=data([2.5,2.5])
    first=measured(a,ref,context(a));second=measured(b,ref,context(b))
    assert first['MAE']<second['MAE']
    assert first['objective']==first['RMSE']==pytest.approx(np.sqrt(8))
    assert first['objective']>second['objective']


def test_short_initial_baseline_cannot_discard_lower_rmse_for_transfer_whitelist(tmp_path):
    a,items,q=daily_fixture()
    means=np.zeros(12);means[6]=10.
    state=xr.Dataset(dict(month_mean=('month',means)),coords=dict(month=np.arange(1,13)),
        attrs=dict(training_end='1999-12-31',split='training'))
    items.append(api.SearchInput('prior.nc','state',state.to_netcdf()))
    result=api.process_current_input(items,q,tmp_path/'run',seed=11,
        budget=dict(objective_limit=180,total_call_limit=180,probe_call_limit=0,proposal_limit=1500,wall_seconds=60))
    assert result['status']=='PASS'
    assert result['initial_best']['metrics']['RMSE']==pytest.approx(0.)
    control=api.submit_search(items,dict(q,mutation_control='none'),None,tmp_path/'none',mode='PortraitMutation',seed=11,
        budget=dict(objective_limit=180,total_call_limit=180,probe_call_limit=0,proposal_limit=1500,wall_seconds=60))
    assert control['status']=='PASS'
    assert control['initial_best']['key']==result['initial_best']['key']


@pytest.mark.parametrize('alias',['exact','derived','state_overlap'])
def test_daily_input_isolation_uses_common_lineage_guard(alias,tmp_path):
    a,items,q=daily_fixture()
    if alias=='exact':
        items.append(api.SearchInput('alias.nc','background',a.to_netcdf()))
        folds,_=plan(items,q,build_context(items,q)[0])
        for f in folds:assert f['protocol']['masked_input_indices']==[0,1]
        return
    if alias=='derived':
        b=a.copy(deep=True);b.attrs.update(source_id='smoothed',derived_from=json.dumps(['station']))
        items.append(api.SearchInput('derived.nc','background',b.to_netcdf()))
    else:
        state=xr.Dataset(dict(month_mean=('month',np.ones(12))),coords=dict(month=np.arange(1,13)),attrs=dict(training_end='2000-07-01'))
        items.append(api.SearchInput('state.nc','state',state.to_netcdf()))
    result=api.process_current_input(items,q,tmp_path/'blocked')
    assert result['status']=='VALIDATION_EVIDENCE_INSUFFICIENT' and result['product'] is None
    assert result['total_cost']['used']==0


@pytest.mark.parametrize('mode',['random','none','page','toolkit'])
def test_daily_public_routes_share_folds_metrics_and_protection(mode,tmp_path):
    from ecodata.interface.search_page import process_search_upload
    a,items,q=daily_fixture()
    budget=dict(objective_limit=180,total_call_limit=180,probe_call_limit=0,proposal_limit=1500,wall_seconds=60)
    if mode=='none':
        result=api.submit_search(items,dict(q,mutation_control='none'),None,tmp_path/'run',mode='PortraitMutation',budget=budget,seed=11)
    elif mode=='page':result=process_search_upload(items,dict(q,decision_policy='current_input'),None,tmp_path/'run',budget=budget,seed=11)
    elif mode=='toolkit':
        path=tmp_path/'primary.nc';path.write_bytes(items[0].content)
        result=api.submit_file_request(dict(inputs=[dict(path=str(path),role='primary')],request=q,
            decision_policy='current_input',budget=budget,seed=11),tmp_path/'run')
    else:result=api.process_current_input(items,q,tmp_path/'run',budget=budget,seed=11)
    assert result['status']=='PASS'
    effective=json.loads((tmp_path/'run/request.json').read_text(encoding='utf-8'))['request']
    assert effective['parameters']['raster_target_times']==q['target_times']
    assert result['self_validation_protocol']['kind']=='task_matched_series'
    assert result['best']['metrics']['complete_required_folds'] and result['best']['metrics']['fold_count']==3
    assert result['reliable_selection']['weights']=={'RMSE':1.}
    assert result['promotion_guard']['support_sufficient']
    for response in result['best']['fold_responses']:
        m=response['metrics'];assert m['objective']==pytest.approx(m['RMSE'])
    with xr.open_dataset(result['product']) as ds:
        known=np.isfinite(a.precipitation.values)
        np.testing.assert_array_equal(ds.precipitation.values[known],a.precipitation.values[known])
        assert np.isfinite(ds.precipitation).all() and (ds.precipitation>=0).all()
