from dataclasses import replace
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.current_input import raster_validation
from ecodata.algorithms.autonomous_field import build_context
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.contracts import AlgorithmSpec,AlgorithmResult,Compatibility
from ecodata.core.schemas import EcoDataBundle
from test_current_input import setup
from test_field_mutation import engine


def context(inputs,q):
    request,_=prepare_model_request(q)
    return request,build_context(inputs,request)[0]


def test_missing_dates_use_whole_blocks_disjoint_confirmation_and_aliases(monkeypatch,tmp_path):
    fine,coarse,inputs,q=setup(monkeypatch,tmp_path,t=8)
    fine=fine.isel(time=[0,1,2,5,6,7])
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    inputs.append(api.SearchInput('alias.nc','background',fine.to_netcdf()))
    request,ctx=context(inputs,q)
    changed,validation,protocol,confirmation=raster_validation(inputs,request,ctx)
    parsed,_=api.parse_inputs(changed)
    assert protocol['design']=='whole_date_temporal_block'
    assert protocol['requested_block_dates']==protocol['actual_search_block_dates']==2
    assert not protocol['support_limited'] and len(protocol['retained_anchor_dates'])==3
    search=np.isfinite(validation.reference.reindex(time=fine.time).values)
    confirm=np.isfinite(confirmation.reindex(time=fine.time).values)
    assert not (search&confirm).any()
    for mask in [search,confirm]:
        assert np.all(mask.any((1,2))==mask.all((1,2)))
        assert np.isnan(parsed[0][1].ndvi.values[mask]).all()
        assert np.isnan(parsed[2][1].ndvi.values[mask]).all()
    np.testing.assert_array_equal(parsed[1][1].ndvi.values,coarse.ndvi.values)
    again,ref2,policy2,conf2=raster_validation(inputs,request,ctx)
    xr.testing.assert_equal(validation.reference,ref2.reference)
    xr.testing.assert_equal(confirmation,conf2)
    assert policy2==protocol and not protocol['external_test_used']


def test_limited_temporal_support_is_declared_and_never_pixel_fallback(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=7)
    observed=fine.isel(time=[0,1,5,6])
    inputs[0]=api.SearchInput('fine.nc','primary',observed.to_netcdf())
    request,ctx=context(inputs,q)
    _,validation,protocol,confirmation=raster_validation(inputs,request,ctx)
    assert protocol['design']=='whole_date_temporal_block' and protocol['support_limited']
    assert protocol['requested_block_dates']==3 and protocol['actual_search_block_dates']==1
    assert len(validation.reference.time)==len(confirmation.time)==1
    observed=observed.isel(time=[0,1,3])
    inputs[0]=api.SearchInput('fine.nc','primary',observed.to_netcdf())
    request,ctx=context(inputs,q)
    with pytest.raises(ValueError,match='Insufficient visible dates for whole-date validation'):
        raster_validation(inputs,request,ctx)


class NumericStep:
    def __init__(self,identifier,fill=None,shift=0):
        self.fill=fill;self.shift=shift
        self.spec=AlgorithmSpec(identifier,identifier,'1','temporal_reconstruction','available',('series',),('continuous',),input_contract=dict(source_arity=1))
    def compatibility(self,context):return Compatibility(True,())
    def run(self,context):
        ds=context.sources[0].dataset.copy(deep=True);v=context.variable
        ds[v]=ds[v].fillna(self.fill) if self.fill is not None else ds[v]+self.shift
        ds[v].attrs=dict(context.sources[0].dataset[v].attrs)
        return AlgorithmResult(EcoDataBundle(ds),self.spec.id,{})


def test_executed_short_prefix_enters_ranking_without_new_pipeline_calls(tmp_path):
    search=engine(tmp_path)
    try:
        for plugin in [NumericStep('test.fill',fill=4),NumericStep('test.shift',shift=1)]:search.registry.register(plugin)
        short=search.intern('test.fill',['s0'],{})
        long=search.intern('test.shift',[short],{})
        row=search.execute(long,'mutation')
        assert row['metrics']['objective']==1
        assert search.evaluated[short]['origin']=='executed_prefix'
        assert search.evaluated[short]['metrics']['objective']==0
        assert search.best()['key']==short
        assert search.evaluated[short]['total_new_calls']==0
        assert search.executor.calls==2 and search.novel_executions==1
        search.execute(short,'mutation')
        assert search.executor.calls==2 and search.novel_executions==1
    finally:search.close()


def test_prefix_quality_gate_cannot_be_bypassed(tmp_path):
    search=engine(tmp_path)
    try:
        search.context=replace(search.context,model_constraints=dict(nonnegative=True))
        for plugin in [NumericStep('test.fill',fill=-4),NumericStep('test.shift',shift=10)]:search.registry.register(plugin)
        short=search.intern('test.fill',['s0'],{})
        long=search.intern('test.shift',[short],{})
        search.execute(long,'mutation')
        assert search.evaluated[short]['status']=='INTERMEDIATE_ONLY'
        assert 'Negative output' in search.evaluated[short]['terminal_reason']
        assert search.best()['key']!=short
    finally:search.close()


def test_whole_date_public_page_api_repeat_and_complete_product(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8)
    visible=fine.isel(time=[0,1,2,5,6,7])
    inputs[0]=api.SearchInput('fine.nc','primary',visible.to_netcdf())
    q['search_stopping']=dict(enabled=True,min_evaluations=2,patience_evaluations=3,
        no_novel_iterations=4,min_relative_gain=.01)
    budget=dict(objective_limit=160,probe_call_limit=0,proposal_limit=12000,wall_seconds=60)
    a=api.process_current_input(inputs,q,tmp_path/'a',budget=budget,seed=29)
    from ecodata.interface.search_page import process_search_upload
    b=process_search_upload(inputs,dict(q,decision_policy='current_input'),None,tmp_path/'b',budget=budget,seed=29)
    assert a['status']==b['status']=='PASS'
    assert a['self_validation_protocol']['design']=='whole_date_leave_one_out'
    assert a['best']['key']==b['best']['key'] and a['stop_reason']==b['stop_reason']
    with xr.open_dataset(a['product']) as x,xr.open_dataset(b['product']) as y:
        xr.testing.assert_equal(x.load(),y.load())
        np.testing.assert_array_equal(x.ndvi.sel(time=visible.time).values,visible.ndvi.values)


def test_page_identity_changes_when_public_mechanism_version_changes(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path)
    from ecodata.interface.search_page import request_fingerprint
    before=request_fingerprint(inputs,q)
    monkeypatch.setattr(api,'VERSION',api.VERSION+'_changed')
    assert request_fingerprint(inputs,q)!=before


def test_public_execution_scores_real_raster_prefix_without_injecting_it_as_candidate(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path,t=6)
    graph=dict(key='filtered',records=dict(
        s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        short=dict(operation='reconstruction.grid_gap_nearest',parents=['s0'],parameters={}),
        filtered=dict(operation='spatial.denoise_tv',parents=['short'],parameters=dict(noise_multiplier=1.))))
    result=api.submit_search(inputs,dict(q,mutation_control='fixed',fixed_graph=graph),None,tmp_path/'run',
        mode='PortraitMutation',budget=dict(objective_limit=20,probe_call_limit=0,proposal_limit=500,wall_seconds=60))
    assert result['status']=='PASS' and result['prefix_terminal_assessments']>=2
    import json
    rows=[json.loads(s) for s in (tmp_path/'run/attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    prefix=next(r for r in rows if r['key']=='short')
    assert prefix['origin']=='executed_prefix' and prefix['status']=='ELIGIBLE_OUTPUT'
    assert prefix['total_new_calls']==0
    assert any(n['node']=='short' and not n['cache_hit'] for r in rows for n in r.get('trace',[]))
    assert result['best']['metrics']['objective']<=prefix['metrics']['objective']
