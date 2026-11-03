"""Open-library discovery, unified objective, real edges, and honest stopping."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms import platform_search as api
from ecodata.algorithms.contracts import AlgorithmSpec,AlgorithmResult,Compatibility
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.core.schemas import EcoDataBundle


class Fill:
    def __init__(self,name,value):
        self.value=value
        self.spec=AlgorithmSpec(name,name,'1','test','available',('time',),('continuous',),
            input_contract=dict(source_arity=1,parameter_domains={},deterministic=True,state_complete=True))
    def compatibility(self,context):return Compatibility(len(context.sources)==1,())
    def run(self,context):
        assert context.validation_reference is None
        ds=context.sources[0].dataset.copy(deep=True)
        ds['p']=ds.p.fillna(self.value)
        return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(actual=True))


class Average:
    spec=AlgorithmSpec('average','average','1','test','available',('time',),('continuous',),
        input_contract=dict(source_arity=2,parameter_domains={},deterministic=True,state_complete=True))
    def compatibility(self,context):return Compatibility(len(context.sources)==2,())
    def run(self,context):
        ds=context.sources[0].dataset.copy(deep=True)
        ds.p.values[:]=(context.sources[0].dataset.p.values+context.sources[1].dataset.p.values)/2
        return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(two_parents=True))


def fixture():
    times=pd.date_range('2000-01-01',periods=75);values=np.full((75,1,1),5.)
    values[[8,9,23,41,42,62]]=np.nan
    ds=xr.Dataset(dict(p=(('time','lat','lon'),values)),coords=dict(time=times,lat=[1.],lon=[2.]),attrs=dict(crs='EPSG:4326'))
    ds.p.attrs.update(units='mm/day',physical_variable='precipitation_amount',temporal_semantics='accumulation',spatial_semantics='point')
    req=dict(task='temporal',variable='p',unit='mm/day',semantic_type='continuous',nonnegative=True,target_lat=[1.],target_lon=[2.],parameters={})
    return [api.SearchInput('station.nc','primary',ds.to_netcdf())],req


def run(tmp_path,monkeypatch,registry,**budget):
    monkeypatch.setattr(api,'platform_registry',lambda:registry)
    inputs,request=fixture()
    return api.submit_search(inputs,request,None,tmp_path,mode='Autonomous',
        budget=dict(dict(objective_limit=100,proposal_limit=1000,wall_seconds=15),**budget))


def test_registered_algorithm_enters_without_a_pool_and_all_paths_use_measured_error(tmp_path,monkeypatch):
    registry=AlgorithmRegistry();registry.register(Fill('zero',0));registry.register(Fill('ten',10));registry.register(Fill('newly_registered',5))
    result=run(tmp_path,monkeypatch,registry)
    assert result['status']=='PASS' and result['empirical_best']['metrics']['objective']==0
    assert result['empirical_best']['algorithm']=='newly_registered'
    assert result['handwritten_pool'] is False and result['max_depth'] is None
    assert result['soft_portrait_score_used'] is False
    assert result['certification']=='MEASURED_FINITE_STATE_OBJECTIVE_OPTIMUM'
    assert result['confirmation_used_for_selection'] is True
    assert set(result['protocol']['search_groups']).isdisjoint(result['protocol']['confirmation_groups'])
    assert len(json.loads((tmp_path/'catalog.json').read_text()))==3


def test_multi_parent_graph_is_discovered_and_no_infinite_depth_claim(tmp_path,monkeypatch):
    registry=AlgorithmRegistry();registry.register(Fill('fill0',0));registry.register(Fill('fill10',10));registry.register(Average())
    result=run(tmp_path,monkeypatch,registry)
    assert result['empirical_best']['metrics']['objective']==0
    assert result['empirical_best']['algorithm']=='average'
    assert len(result['empirical_best']['parents'])==2
    assert result['stop_reason']=='RESOURCE_BUDGET_EXHAUSTED'
    assert result['complete_space'] is False


def test_undocumented_domains_never_get_a_global_certificate(tmp_path,monkeypatch):
    registry=AlgorithmRegistry();plugin=Fill('default_only',5);plugin.spec=replace(plugin.spec,input_contract={});registry.register(plugin)
    result=run(tmp_path,monkeypatch,registry,proposal_limit=12)
    assert result['unbounded_or_undocumented_contracts']
    assert result['certification']=='NOT_CERTIFIED_CURRENT_BEST_ONLY'


def test_no_visible_reference_and_manual_pool_are_rejected(tmp_path,monkeypatch):
    inputs,req=fixture();ds=xr.open_dataset(__import__('io').BytesIO(inputs[0].content)).load();ds.p.values[:]=np.nan
    with pytest.raises(ValueError):api.submit_search([api.SearchInput('empty.nc','primary',ds.to_netcdf())],req,None,tmp_path/'no_truth',mode='Autonomous')
    with pytest.raises(ValueError,match='hand-authored'):
        api.submit_search(inputs,dict(req,composition={}),None,tmp_path/'pool',mode='Autonomous')


def test_page_handler_uses_same_public_decision_and_returns_real_edges(tmp_path,monkeypatch):
    from ecodata.interface.search_page import process_search_upload
    registry=AlgorithmRegistry();registry.register(Fill('measured_fill',5))
    monkeypatch.setattr(api,'platform_registry',lambda:registry)
    inputs,request=fixture()
    result=process_search_upload(inputs,request,None,tmp_path,mode='Autonomous',
        budget=dict(objective_limit=10,proposal_limit=20,wall_seconds=15))
    assert result['best']['graph'][-1]['parents']==['s0']
    assert result['best']['graph'][-1]['operation']=='measured_fill'
    assert result['best']['file_readback']=='PASS'


def test_unexecutable_declared_parameter_domain_cannot_certify_optimum(tmp_path,monkeypatch):
    registry=AlgorithmRegistry();plugin=Fill('unknown_domain',5)
    plugin.spec=replace(plugin.spec,input_contract=dict(plugin.spec.input_contract,
        parameter_domains=dict(setting=dict(source='undeclared_provider'))))
    registry.register(plugin)
    result=run(tmp_path,monkeypatch,registry)
    assert result['complete_space'] is False
    assert result['certification']=='NOT_CERTIFIED_CURRENT_BEST_ONLY'
    assert result['status']=='NO_ELIGIBLE_RESULT'
