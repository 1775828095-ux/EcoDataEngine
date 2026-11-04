"""Synthetic diagnostic contracts; real saved CMAQ execution is separate."""
from pathlib import Path
import io
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.stability import assess


def inputs():
 items=[]
 for role,v in [('primary',[1.,2.,3.]),('background',[0.,4.,5.])]:
  ds=xr.Dataset({'v':('x',v,dict(units='1'))},attrs=dict(source_id=role))
  items.append(api.SearchInput(role+'.nc',role,ds.to_netcdf()))
 return items


def baseline(tmp_path):
 p=tmp_path/'baseline.nc';xr.Dataset({'v':('x',[1.,2.,3.],dict(units='1'))}).to_netcdf(p)
 return dict(product=str(p),selected_graph=dict(key='a',records={'a':dict(operation='synthetic_test')}))


def execute(items,request,graph,folder):
 folder=Path(folder);folder.mkdir()
 with xr.open_dataset(io.BytesIO(items[0].content)) as d:ds=d.load()
 p=folder/'product.nc';ds.to_netcdf(p)
 return dict(product=str(p),cost=dict(actual_test_executions=1))


def test_real_salib_factors_are_scoped_and_no_reference_read(tmp_path):
 items=inputs();raw=[i.content for i in items]
 # Deliberately unreadable reference must neither be parsed nor perturbed.
 items.append(api.SearchInput('truth.nc','independent_validation',b'private hidden labels'))
 def guarded(items,request,graph,folder):
  assert items[-1].content==b'private hidden labels'
  with xr.open_dataset(io.BytesIO(items[1].content)) as d:assert float(d.v.min())>=0
  return execute(items,request,graph,folder)
 r=assess(items,dict(variable='v',semantic_type='continuous',nonnegative=True),baseline(tmp_path),tmp_path/'audit',execute=guarded)
 assert r['status']=='DIAGNOSTIC_COMPLETE' and r['new_executions']==6 and r['failures']==0
 assert not r['accuracy_assessed'] and not r['used_for_selection']
 by={s['index']:s for s in r['mean_output_sensitivity']}
 assert by[0]['mu_star']>0 and abs(by[1]['mu_star'])<1e-12
 assert [i.content for i in items[:2]]==raw
 assert (tmp_path/'audit/stability.json').is_file()


def test_failures_remain_incomplete_not_robust_pass(tmp_path):
 def fail(*a):raise ValueError('real execution failed')
 r=assess(inputs(),dict(variable='v',semantic_type='continuous'),baseline(tmp_path),tmp_path/'audit',execute=fail)
 assert r['status']=='INCOMPLETE' and r['new_executions']==6 and r['failures']==6
 assert 'mean_output_sensitivity' not in r


def test_no_graph_and_category_do_not_fake_scores(tmp_path):
 r=assess(inputs(),dict(variable='v'),{},tmp_path/'missing',execute=execute)
 assert r['status']=='NOT_EVALUATED' and not r['new_executions']
 r=assess(inputs(),dict(variable='v',semantic_type='categorical'),baseline(tmp_path),tmp_path/'category',execute=execute)
 assert r['status']=='NOT_EVALUATED' and not r['new_executions']


def test_constant_sources_do_not_invent_uncertainty(tmp_path):
 items=[api.SearchInput('x.nc','primary',xr.Dataset({'v':('x',[1.,1.],dict(units='1'))}).to_netcdf())]
 r=assess(items,dict(variable='v',semantic_type='continuous'),baseline(tmp_path),tmp_path/'audit',execute=execute)
 assert r['status']=='NOT_EVALUATED' and not r['new_executions']


def test_budget_stops_between_calls(tmp_path,monkeypatch):
 from ecodata.algorithms import stability
 clock=iter([0.,61.,62.]);monkeypatch.setattr(stability,'perf_counter',lambda:next(clock))
 r=assess(inputs(),dict(variable='v',semantic_type='continuous'),baseline(tmp_path),tmp_path/'audit',execute=execute)
 assert r['status']=='INCOMPLETE' and not r['new_executions']


def test_common_entry_adds_report_without_changing_selected_product(tmp_path,monkeypatch):
 fixed=baseline(tmp_path);before=Path(fixed['product']).read_bytes()
 monkeypatch.setattr(api,'_process_current_input',lambda *a,**k:fixed)
 monkeypatch.setattr(api,'execute_frozen_graph',execute)
 r=api.process_current_input(inputs(),dict(variable='v',semantic_type='continuous'),tmp_path/'service')
 assert r['product']==fixed['product'] and r['selected_graph']==fixed['selected_graph']
 assert Path(r['product']).read_bytes()==before and r['post_selection_stability']['status']=='DIAGNOSTIC_COMPLETE'

def test_download_assessment_keeps_diagnostic_separate_from_accuracy(tmp_path):
 from ecodata.models.fusion_assessment import assess_task
 b=baseline(tmp_path);r=assess(inputs(),dict(variable='v',semantic_type='continuous'),b,tmp_path/'audit',execute=execute)
 task=dict(field='v',result=dict(b,post_selection_stability=r))
 report=assess_task(tmp_path,task,b['product'])
 assert report['post_selection_stability']['status']=='DIAGNOSTIC_COMPLETE'
 assert report['accuracy_improvement']=='NOT_ESTABLISHED'
