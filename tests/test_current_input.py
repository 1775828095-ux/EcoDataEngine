from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.current_input import raster_validation,deploy_ranked
from ecodata.algorithms.autonomous_field import build_context
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.search_accounting import SearchAccount
from test_advanced_complements import field
from test_raster_gap_application import registry,contract


def setup(monkeypatch,tmp_path,t=3):
    registry(monkeypatch,tmp_path)
    fine=field(n=8,t=t);coarse=field(n=8,t=t,source='coarse');coarse.ndvi.values+=.015
    definition=contract();definition.update(field='ndvi',valid_range=[-1.,1.]);definition['issues']=definition['issues'][:1]
    request=dict(task='multisource',variable='ndvi',unit='1',semantic_type='continuous',
        target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),target_times=[str(v) for v in fine.time.values],
        parameters=dict(raster_target_times=[str(v) for v in fine.time.values],grid_gap_recovery=True),
        application_contract=definition,scoring=dict(registry_version='grid-test-v1'),response_scoring=dict(enabled=False))
    inputs=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('coarse.nc','background',coarse.to_netcdf())]
    return fine,coarse,inputs,request


def test_visible_raster_masks_disjoint_confirmation_and_aliases(monkeypatch,tmp_path):
    fine,coarse,inputs,q=setup(monkeypatch,tmp_path,t=6);fine.ndvi.values[0,0,0]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf());inputs.append(api.SearchInput('copy.nc','background',fine.to_netcdf()))
    request,_=prepare_model_request(q);ctx,_,_=build_context(inputs,request)
    changed,validation,protocol,confirmation=raster_validation(inputs,request,ctx)
    parsed,_=api.parse_inputs(changed)
    search_mask=np.isfinite(validation.reference.reindex(time=fine.time).values)
    confirm_mask=np.isfinite(confirmation.reindex(time=fine.time).values)
    assert not (search_mask&confirm_mask).any() and search_mask.any() and confirm_mask.any()
    assert np.isnan(parsed[0][1].ndvi.values[search_mask|confirm_mask]).all()
    assert np.isnan(parsed[2][1].ndvi.values[search_mask|confirm_mask]).all()
    np.testing.assert_array_equal(parsed[1][1].ndvi.values,coarse.ndvi.values)
    assert not search_mask[0,0,0] and not confirm_mask[0,0,0]
    assert protocol['masked_input_indices']==[0,2] and not protocol['confirmation_used_for_selection']
    assert protocol['input_time_count']==6 and protocol['external_test_used'] is False
    assert np.isfinite(fine.ndvi.values[1:]).all()


@pytest.mark.parametrize('t',[6,3,1])
def test_current_input_real_search_never_transplants_old_graph(monkeypatch,tmp_path,t):
    fine,coarse,inputs,request=setup(monkeypatch,tmp_path,t=t)
    if t==3:
        request['target_times']=[str(v) for v in sorted([*fine.time.values,fine.time.values[0]+np.timedelta64(4,'D')])]
    result=api.process_current_input(inputs,request,tmp_path/'run',budget=dict(objective_limit=160,probe_call_limit=0,proposal_limit=12000,wall_seconds=60),seed=11)
    assert result['status']=='PASS' and result['optimizer']['backend']=='random' and result['evolution_used']
    assert result['self_validation_protocol']['input_time_count']==t
    if t==3:
        assert result['confirmation'] is None
        assert not result['self_validation_protocol']['independent_confirmation_available']
    else:assert result['confirmation']['used_for_selection'] is False
    assert result['current_input_deployment']['status']=='PASS'
    with xr.open_dataset(result['product']) as out:
        assert out.sizes['time']==len(request['target_times'])
        np.testing.assert_array_equal(out.ndvi.sel(time=fine.time).values,fine.ndvi.values)
    records=json.loads((tmp_path/'run/graphs.json').read_text(encoding='utf-8'))
    for k in [r['key'] for r in result['current_input_deployment']['attempts'] if r['status']=='PASS']:
        pending=[k];seen=set()
        while pending:
            node=pending.pop()
            if node in seen:continue
            seen.add(node);row=records[node]
            if t==3:assert row['operation']!='reconstruction.dineof'
            pending+=row['parents']
    with xr.open_dataset(tmp_path/'run/self_validation_input.nc') as opened:assert not np.isfinite(opened.ndvi).all()


def test_strict_stale_dineof_is_rejected_but_current_mode_researches(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path,t=3)
    graph=dict(records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        d=dict(operation='reconstruction.dineof',parents=['s0'],parameters=dict(max_modes=5,internal_seed=0))),key='d')
    with pytest.raises(Exception,match='four dates'):
        api.execute_frozen_graph(inputs,q,graph,tmp_path/'strict')
    with pytest.raises(ValueError,match='prescribed graph'):
        api.process_current_input(inputs,dict(q,fixed_graph=graph),tmp_path/'not_current')


def test_current_deployment_automatically_rejects_stale_best_and_uses_next(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path,t=3);request,_=prepare_model_request(q);ctx,_,_=build_context(inputs,request)
    records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        stale=dict(operation='reconstruction.dineof',parents=['s0'],parameters=dict(max_modes=5,internal_seed=0)),
        legal=dict(operation='temporal.raster_linear',parents=['s0'],parameters={}))
    rows={k:dict(key=k,status='ELIGIBLE_OUTPUT',metrics=dict(objective=i)) for i,k in enumerate(['stale','legal'])}
    search=SimpleNamespace(records=records,evaluated=rows,rank=lambda r:r['metrics']['objective'],account=SearchAccount(10,30))
    selected,out,receipt=deploy_ranked(search,ctx,api.platform_registry(),request)
    assert selected['key']=='legal' and receipt['automatic_reselection']
    assert [r['status'] for r in receipt['attempts']]==['REJECTED','PASS']
    assert 'four dates' in receipt['attempts'][0]['error'] and not receipt['external_labels_used']
    assert np.isfinite(out.ndvi).all()


def test_no_visible_truth_stops_without_invented_score(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=3);fine.ndvi.values[:]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    with pytest.raises(ValueError,match='Insufficient visible|RawInputQualityGate BLOCK'):
        api.process_current_input(inputs,q,tmp_path/'run')
    assert not (tmp_path/'run/product.nc').exists()


def test_current_deployment_checks_quality_and_reports_no_valid_graph(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path)
    request,_=prepare_model_request(q);ctx,_,_=build_context(inputs,request)
    ctx.parameters['target_valid_range']=[float(fine.ndvi.min()),float(fine.ndvi.max())+.005]
    records={f's{i}':dict(operation='source',parents=[],parameters=dict(source=f'source{i}')) for i in range(2)}
    rows={k:dict(key=k,status='ELIGIBLE_OUTPUT',metrics=dict(objective=i)) for i,k in enumerate(['s1','s0'])}
    search=SimpleNamespace(records=records,evaluated=rows,rank=lambda r:r['metrics']['objective'],account=SearchAccount(10,30))
    selected,out,receipt=deploy_ranked(search,ctx,api.platform_registry(),request)
    assert selected['key']=='s0' and receipt['automatic_reselection']
    assert 'outside declared range' in receipt['attempts'][0]['error']
    search.evaluated={'s1':rows['s1']}
    selected,out,receipt=deploy_ranked(search,ctx,api.platform_registry(),request)
    assert selected is None and out is None and receipt['status']=='NO_EXECUTABLE_CURRENT_GRAPH'


def test_toolkit_routes_current_requests_without_external_labels(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path);items=[];calls=[]
    for item in inputs:
        path=tmp_path/item.filename;path.write_bytes(item.content)
        items.append(dict(path=str(path),role=item.role))
    def public(*args,**kwargs):calls.append((args,kwargs));return dict(status='PASS')
    monkeypatch.setattr(api,'process_current_input',public)
    config=dict(inputs=items,request=q,decision_policy='current_input',seed=29)
    assert api.submit_file_request(config,tmp_path/'toolkit')['status']=='PASS'
    assert calls[0][1]['seed']==29
    with pytest.raises(ValueError,match='external selection/test'):
        api.submit_file_request(dict(config,evaluation=dict(path='forbidden.nc')),tmp_path/'rejected')


def test_current_mode_rejects_external_selection_labels(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path)
    fine.ndvi.attrs['split']='search_validation'
    with pytest.raises(ValueError,match='visible-only'):
        api.submit_search(inputs,dict(q,decision_policy='current_input'),None,tmp_path/'run',mode='PortraitMutation',validation=api.SearchValidation(fine.ndvi,'ndvi'))


def test_search_rejects_low_error_graph_missing_requested_dates(monkeypatch,tmp_path):
    from ecodata.algorithms.field_mutation import GraphMutation
    fine,_,inputs,q=setup(monkeypatch,tmp_path)
    q['target_times']=[str(v) for v in sorted([*fine.time.values,fine.time.values[0]+np.timedelta64(4,'D')])]
    q['parameters']['raster_target_times']=q['target_times'];q['_deployment_reexecution']=True
    request,model=prepare_model_request(q);ctx,_,_=build_context(inputs,request)
    folder=tmp_path/'search';folder.mkdir()
    search=GraphMutation(ctx,api.platform_registry(),request,model,fine.ndvi,folder,dict(objective_limit=20,proposal_limit=500,wall_seconds=30),11)
    try:
        partial=search.intern('reconstruction.grid_gap_nearest',['s0'],{})
        row=search.execute(partial,'regression')
        assert row['status']=='INTERMEDIATE_ONLY' and 'target mismatch: time' in row['terminal_reason']
        full=search.intern('temporal.raster_linear',['s0'],{})
        assert search.execute(full,'regression')['status']=='ELIGIBLE_OUTPUT'
        assert search.best()['key']==full
    finally:search.close()


def test_search_keeps_reserved_deployment_time(monkeypatch):
    from ecodata.algorithms import search_accounting
    monkeypatch.setattr(search_accounting,'perf_counter',lambda:0.)
    account=SearchAccount(20,10);account.deployment_time_reserve=3.
    monkeypatch.setattr(search_accounting,'perf_counter',lambda:8.)
    assert account.exhausted('search') and not account.exhausted('deploy')


def test_page_and_toolkit_share_current_entry_and_changed_input_identity(monkeypatch,tmp_path):
    from ecodata.interface import search_page
    fine,_,inputs,q=setup(monkeypatch,tmp_path);calls=[]
    def public(*args,**kwargs):calls.append((args,kwargs));return dict(status='PASS')
    monkeypatch.setattr(search_page,'process_current_input',public)
    result=search_page.process_search_upload(inputs,dict(q,decision_policy='current_input'),None,tmp_path/'page',mode='PortraitMutation',validation=None,seed=29)
    assert result['status']=='PASS' and calls[0][1]['seed']==29
    first=search_page.request_fingerprint(inputs,q);fine.ndvi.values[0,0,0]+=.1
    changed=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),inputs[1]]
    assert first!=search_page.request_fingerprint(changed,q)
    assert first!=search_page.request_fingerprint(inputs,dict(q,target_times=['other']))
