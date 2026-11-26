"""Synthetic integration isolation, not environmental accuracy evidence."""
import json
import numpy as np
import pandas as pd
from ecodata.algorithms.platform_search import SearchInput,submit_search
from ecodata.algorithms.paired_station_search import prepare


def inputs():
    frame=pd.DataFrame(dict(station='S',grid_time=pd.date_range('2000-01-01',periods=24,freq='h'),
        background=280+np.arange(24.),observed=282+np.arange(24.),longitude=110.,latitude=30.,unit='K'))
    return frame


def request():
    return dict(task='temporal',adapter='paired_station_calibration',variable='temperature',unit='K',
        recommended_recipes=[dict(id='DS1',scope=dict(variable='temperature'),steps=['station.bias.holdout'])])


BUDGET=dict(objective_limit=80,total_call_limit=120,proposal_limit=256,wall_seconds=30)


def test_actual_common_mutation_and_recommendation_parent(tmp_path):
    data=inputs()
    result=submit_search([SearchInput('arbitrary.csv','paired_observations',data.to_csv(index=False).encode())],
        request(),None,tmp_path/'run',budget=BUDGET,seed=11)
    assert result['status']=='PASS',result
    run=result['stations'][0]
    assert len(run['best']['metrics']['block_metrics'])==3
    assert run['recommendation_review']['recommended']
    assert run['delivered_best']['key']==run['delivered_graph']['key']
    assert run['delivered_graph']['key'] in run['delivered_graph']['records']
    assert run['optimizer']['shared_driver']=='composition_optimizers.random_graph_search'
    assert run['recommendation_validation'][0]['status']=='PASS'
    assert run['evolution_used'] and run['mutation_actual_new_executions']>0
    traces=json.loads((tmp_path/'run/station-1/search/mutation_trace.json').read_text())
    parents=set(run['recommendation_validation'][0]['keys'])
    assert any(t.get('parent') in parents for t in traces)
    assert run['product'] and run['test_used_for_selection'] is False
    assert any(edge['from']=='__fitted_state.state' for node in run['best']['trace'] for edge in node['edges'])
    from ecodata.desktop.output_review import cards
    report=dict(mutation_search=run,methods=['station.bias.holdout'])
    product=dict(mode='selected_operations',original_report=[dict(task='temperature',backend='station_grid',report=[report])])
    card=cards(product,dict(source_formats=['CSV'],target_format='NetCDF',conversion_reason='Synthetic UI check'))[0]
    assert card['family']=='实际数值变异' and card['performed']
    assert any('实际新变异执行' in text for text in card['audit'])
    import os
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from ecodata.desktop.result_cards import ResultCards
    from ecodata.desktop.output_review import describe_output
    app=QApplication.instance() or QApplication([])
    widget=ResultCards();widget.render([card],describe_output(product,None));app.processEvents()
    widget.close()
    # Full search must stay identical when only the sealed confirmation labels
    # change. Only the post-selection confirmation is allowed to change.
    changed=data.copy();changed.loc[18:,'observed']+=10000
    other=submit_search([SearchInput('renamed.csv','paired_observations',changed.to_csv(index=False).encode())],
        request(),None,tmp_path/'other',budget=BUDGET,seed=11)['stations'][0]
    assert other['best']['key']==run['best']['key']
    assert other['best']['metrics']==run['best']['metrics']
    assert other['independent_evaluation']['RMSE']!=run['independent_evaluation']['RMSE']


def test_failed_review_keeps_background_without_reselecting_candidate(tmp_path,monkeypatch):
    from ecodata.algorithms.platform_search import BlockedReference
    actual=BlockedReference.__call__;calls=[]
    def confirmation(self,ds):
        freeze=tmp_path/'run/station-1/search/selection_freeze.json'
        assert freeze.exists()
        calls.append(json.loads(freeze.read_text())['key'])
        return actual(self,ds)
    monkeypatch.setattr(BlockedReference,'__call__',confirmation)
    data=inputs();data.loc[18:,'observed']=data.loc[18:,'background']
    result=submit_search([SearchInput('input.csv','paired_observations',data.to_csv(index=False).encode())],
        request(),None,tmp_path/'run',budget=BUDGET,seed=11)['stations'][0]
    assert result['recommendation_review']['status']=='CONFIRMATION_FAILED'
    assert result['delivery_status']=='BASELINE_RETAINED'
    assert result['delivered_best']['key']==result['delivered_graph']['key']=='s0'
    assert len(calls)==2 and set(calls)=={result['best']['key']} # one frozen candidate and its fixed baseline
    import xarray as xr
    with xr.open_dataset(result['product']) as delivered,xr.open_dataset(result['candidate_product']) as candidate:
        np.testing.assert_allclose(delivered.temperature.values[:,0,0],data.background.iloc[12:])
        assert not np.array_equal(delivered.temperature,candidate.temperature)
    from ecodata.desktop.output_review import cards,selection_summaries
    page=dict(mode='selected_operations',original_report=[dict(task='T',backend='station_grid',
        report=[dict(mutation_search=result,methods=['station.bias.holdout'])])])
    items=cards(page,dict(source_formats=['CSV'],target_format='NetCDF',conversion_reason='UI check'))
    chosen=next(r for r in items if r['selected'])
    assert chosen['selection_label']=='默认保留' and chosen['methods']==['source']
    assert selection_summaries(items,[],[])[-1]['steps']==['source']


def test_budget_stop_not_disguised_as_quality_failure_or_recommendation(tmp_path):
    data=inputs()
    result=submit_search([SearchInput('input.csv','paired_observations',data.to_csv(index=False).encode())],
        request(),None,tmp_path/'run',budget=dict(objective_limit=1,total_call_limit=2,proposal_limit=4,wall_seconds=10),seed=11)
    station=result['stations'][0]
    assert station['recommendation_review']['status']=='SEARCH_INCOMPLETE'
    assert not station['recommendation_review']['recommended']
    assert station['delivery_status']=='BASELINE_RETAINED'
    assert station['kernel_calls']<=2


def test_explicit_legacy_evaluation_keeps_archived_behavior(tmp_path):
    data=inputs();data.loc[18:,'observed']=data.loc[18:,'background']
    legacy=dict(request(),evaluation_protocol='legacy_single_v1')
    result=submit_search([SearchInput('input.csv','paired_observations',data.to_csv(index=False).encode())],
        legacy,None,tmp_path/'run',budget=BUDGET,seed=11)['stations'][0]
    assert result['recommendation_review']['status']=='LEGACY_NOT_REVIEWED'
    assert not result['recommendation_review']['recommended']
    assert 'block_metrics' not in result['best']['metrics']
    assert result['independent_evaluation']['RMSE']>result['baseline_final']['RMSE']


def test_fitting_state_and_predictor_do_not_contain_future_labels():
    a=inputs();x,state,val,test,protocol=prepare(a,'temperature','K')
    b=a.copy();b.loc[18:,'observed']+=10000
    y,other,v,t,_=prepare(b,'temperature','K')
    import xarray as xr
    xr.testing.assert_identical(x,y);xr.testing.assert_identical(state,other)
    xr.testing.assert_identical(val,v)
    assert not np.array_equal(test,t)
    assert pd.Timestamp(protocol['train_times'][-1])<pd.Timestamp(x.time.values[0])


def test_real_state_parent_changes_cache_and_result():
    from dataclasses import replace
    from ecodata.algorithms.autonomous_decision import Executor
    from ecodata.algorithms.contracts import AlgorithmContext
    from ecodata.algorithms.registry import AlgorithmRegistry
    from ecodata.algorithms.series_fusion import PriorSeriesCalibration
    from ecodata.core.schemas import EcoDataBundle
    data,state,*_=prepare(inputs(),'temperature','K')
    registry=AlgorithmRegistry();registry.register(PriorSeriesCalibration('affine'))
    context=AlgorithmContext((EcoDataBundle(data),),'temperature','continuous',parameters=dict(fitted_state=state))
    records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        correction=dict(operation='bias.prior_series_affine',parents=['s0'],parameters={}))
    executor=Executor(context,registry)
    first,trace=executor.execute(records,'correction')
    assert any(edge['from']=='__fitted_state.state' for node in trace for edge in node['edges'])
    changed=state.copy(deep=True);changed['affine_intercept']=changed.affine_intercept+1
    executor.context=replace(context,parameters=dict(fitted_state=changed))
    second,trace2=executor.execute(records,'correction')
    np.testing.assert_allclose(second.temperature-first.temperature,1)
    assert not trace2[-1]['cache_hit']


def test_missing_units_and_insufficient_observations_are_explicit():
    import pytest
    with pytest.raises(ValueError,match='nine'):prepare(inputs().iloc[:8],'temperature','K')
    with pytest.raises(ValueError,match='units'):prepare(inputs(),'temperature','UNKNOWN')


def test_desktop_worker_accepts_real_structured_search_progress():
    from ecodata.desktop.__main__ import Job
    events=[];finished=[]
    def operation(progress):
        progress(dict(event='mutation_execution',candidate='actual-node',status='ELIGIBLE_OUTPUT'))
        return 'done'
    job=Job(operation);job.progress.connect(events.append)
    job.done.connect(lambda result,error:finished.append((result,error)))
    job.run()
    assert finished==[('done',None)]
    assert json.loads(events[0])['event']=='mutation_execution'
