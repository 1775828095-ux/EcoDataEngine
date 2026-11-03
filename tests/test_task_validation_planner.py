import json
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.task_validation import plan,outcome
from ecodata.algorithms.autonomous_field import build_context
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.search_accounting import SearchAccount
from test_current_input import setup


def context(inputs,q):
    request,_=prepare_model_request(q);return request,build_context(inputs,request)[0]


def test_dense_long_gap_uses_matching_blocks_not_one_date(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=24)
    original=fine.copy(deep=True);fine.ndvi.values[10:13]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    request,ctx=context(inputs,q);folds,protocol=plan(inputs,request,ctx)
    assert protocol['tasks']==['temporal_gap'] and not protocol['support_limited']
    assert all(f['protocol']['design']=='matched_temporal_block' for f in folds)
    for f in folds:
        assert len(f['validation'].reference.time)==3
        assert f['protocol']['actual_hidden_span_days']==16
        parsed,_=api.parse_inputs(f['inputs']);ds=parsed[0][1]
        assert np.isnan(ds.ndvi.sel(time=f['validation'].reference.time)).all()
        np.testing.assert_array_equal(parsed[1][1].ndvi.values,original.ndvi.values+.015)


def test_unsupported_long_gap_is_not_labeled_matched(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=14)
    observed=fine.isel(time=[0,1,12,13]);inputs[0]=api.SearchInput('fine.nc','primary',observed.to_netcdf())
    request,ctx=context(inputs,q);folds,protocol=plan(inputs,request,ctx)
    assert protocol['support_limited'] and protocol['support_level']=='LIMITED_DIAGNOSTIC_ONLY'
    assert protocol['task_support'][0]['status']=='INSUFFICIENT'
    assert all(f['protocol']['design']=='whole_date_leave_one_out' for f in folds)
    assert outcome(protocol,{},'NO_MATERIAL_VALIDATION_GAIN')['task_quality_achieved'] is False


@pytest.mark.parametrize('joint',[False,True])
def test_actual_block_shape_and_time_support_drive_validation(monkeypatch,tmp_path,joint):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8)
    fine.ndvi.values[2:4 if joint else 3,2:4,2:5]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    request,ctx=context(inputs,q);folds,p=plan(inputs,request,ctx)
    kind='spatiotemporal_block' if joint else 'spatial_block'
    assert p['tasks']==[kind] and p['support_level']=='TASK_MATCHED_INTERNAL'
    assert 1<=len(folds)<=3
    for f in folds:
        assert f['protocol']['design']==kind
        assert f['protocol']['search_points']==(12 if joint else 6)
        assert f['protocol']['template_shape']==([2,2,3] if joint else [1,2,3])
        assert f['protocol']['duration_offsets_days']==([0.,8.] if joint else [0.])
        assert f['protocol']['placement'][1:]==[2,2]


def test_transitive_derived_source_cannot_leak_a_smoothed_target(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=6)
    derived=fine.copy(deep=True);derived.attrs.update(source_id='smoothed',derived_from=json.dumps(['fine']))
    inputs.append(api.SearchInput('smooth.nc','background',derived.to_netcdf()))
    result=api.process_current_input(inputs,q,tmp_path/'blocked',seed=11)
    assert result['status']=='VALIDATION_EVIDENCE_INSUFFICIENT' and result['product'] is None
    assert 'replay' in result['reason']
    assert result['total_cost']['used']==0


def test_scale_proxy_reuses_area_operator_without_native_labels_in_roots(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=4)
    # A single native source with a requested2x finer grid; retain only known labels in evaluator.
    inputs=inputs[:1];q['target_lat']=np.linspace(float(fine.lat[0])-.000075,float(fine.lat[-1])+.000075,16).tolist()
    q['target_lon']=np.linspace(float(fine.lon[0])-.000075,float(fine.lon[-1])+.000075,16).tolist()
    request,ctx=context(inputs,q);account=SearchAccount(100,60,total_calls=100)
    folds,p=plan(inputs,request,ctx,account)
    assert p['tasks']==['scale_refinement'] and p['support_level']=='REDUCED_RESOLUTION_PROXY_ONLY'
    assert p['support_limited'] and len(folds)==1
    parsed,_=api.parse_inputs(folds[0]['inputs']);assert parsed[0][1].sizes['lat']==4
    assert folds[0]['validation'].reference.shape==fine.ndvi.shape
    assert account.receipt()['phases']=={'validation_setup':1}
    graph=dict(records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        a=dict(operation='framewise.raster_task.spatial.bilinear',parents=['s0'],parameters={})),key='a')
    result=api.evaluate_graph_bank(inputs,q,dict(linear=graph),tmp_path/'scale',budget=dict(objective_limit=100,wall_seconds=60))
    assert result['status']=='PASS' and result['best']['metrics']['complete_required_folds']
    assert result['task_acceptance']['status']=='EVIDENCE_INSUFFICIENT'
    assert result['scale_consistency']['status']=='MEASURED'
    assert not result['scale_consistency']['fine_truth_verified']
    with xr.open_dataset(result['product']) as ds:assert ds.ndvi.shape==(4,16,16)


def test_stopping_and_missing_quality_goal_are_separate():
    p=dict(support_level='TASK_MATCHED_INTERNAL',support_limited=False)
    assert outcome(p,{},'NO_MATERIAL_VALIDATION_GAIN')['status']=='GOAL_NOT_SPECIFIED'
    assert not outcome(p,{},'RESOURCE_BUDGET_EXHAUSTED')['task_quality_achieved']
    with pytest.raises(ValueError,match='evidence'):outcome(p,dict(quality_goal=dict(RMSE=.1)),'X')


@pytest.mark.parametrize('kind',['spatial_block','spatiotemporal_block','temporal_gap'])
def test_public_entry_executes_every_task_fold_and_masks_aliases(monkeypatch,tmp_path,kind):
    fine,coarse,inputs,q=setup(monkeypatch,tmp_path,t=16)
    if kind=='temporal_gap':fine.ndvi.values[7:10]=np.nan
    else:fine.ndvi.values[6:8 if kind=='spatiotemporal_block' else 7,2:4,2:5]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    inputs.append(api.SearchInput('alias.nc','background',fine.to_netcdf()))
    graph=dict(records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        a=dict(operation='temporal.raster_linear' if kind=='temporal_gap' else 'reconstruction.grid_gap_linear',parents=['s0'],parameters={})),key='a')
    result=api.evaluate_graph_bank(inputs,q,dict(simple=graph),tmp_path/'public',budget=dict(objective_limit=200,wall_seconds=60))
    assert result['status']=='PASS' and result['best']['metrics']['complete_required_folds']
    p=result['self_validation_protocol'];assert p['tasks']==[kind] and not p['support_limited']
    assert p['history_validation']['status']=='NOT_REQUIRED_LOCAL_MATCH_OR_OTHER_TASK'
    assert 1<=len(p['all_required_fold_ids'])<=3
    assert len(p['all_required_fold_ids'])==p['fold_count']
    for folder in (tmp_path/'public/validation_folds').iterdir():
        with xr.open_dataset(folder/'source0.nc') as a,xr.open_dataset(folder/'source2.nc') as b:
            xr.testing.assert_equal(a.load(),b.load())
        with xr.open_dataset(folder/'source1.nc') as a:np.testing.assert_array_equal(a.ndvi.values,coarse.ndvi.values)


def test_transformed_alias_requires_replay_instead_of_hiding_only_its_output(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8)
    inputs.append(api.SearchInput('resampled.nc','background',fine.isel(lat=slice(None,None,2)).to_netcdf()))
    result=api.process_current_input(inputs,q,tmp_path/'blocked')
    assert result['status']=='VALIDATION_EVIDENCE_INSUFFICIENT' and result['product'] is None
    assert 'replay' in result['reason']


def test_too_few_temporal_anchors_returns_evidence_status_not_fake_accuracy(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=2)
    q['target_times']=[str(fine.time.values[0]),str(fine.time.values[0]+np.timedelta64(4,'D')),str(fine.time.values[1])]
    result=api.process_current_input(inputs,q,tmp_path/'insufficient')
    assert result['status']=='VALIDATION_EVIDENCE_INSUFFICIENT' and result['product'] is None
    assert result['total_cost']['used']==0
    assert result['task_acceptance']['task_quality_achieved'] is False


def test_missing_primary_lineage_id_does_not_bypass_derived_source_isolation(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=6)
    fine.attrs.pop('source_id');inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    derived=fine.copy(deep=True);derived.attrs.update(source_id='smooth',derived_from=json.dumps(['fine.nc']))
    inputs.append(api.SearchInput('smooth.nc','background',derived.to_netcdf()))
    result=api.process_current_input(inputs,q,tmp_path/'missing_id')
    assert result['status']=='VALIDATION_EVIDENCE_INSUFFICIENT' and result['product'] is None
    assert 'source_id' in result['reason'] and result['total_cost']['used']==0
