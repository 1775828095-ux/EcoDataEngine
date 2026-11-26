"""Integration boundaries, not environmental accuracy evidence."""
import json
import numpy as np
import pandas as pd
import pytest

from ecodata.evaluation.blocked_confirmation import mutation_acceptance
from ecodata.desktop.holdout_plan import reserve_mutation_confirmation


def test_recommendation_and_mutation_reserves_are_disjoint_before_values():
    task={'locked_holdout':{'days':['2020-01-01','2020-01-02','2020-01-03','2020-01-04']}}
    reserve_mutation_confirmation({},task)
    assert task['recipe_review_units']['days']==['2020-01-01','2020-01-02']
    assert task['mutation_confirmation_times']==['2020-01-03','2020-01-04']
    assert not set(task['mutation_reservation']['recommendation_review_times']) & set(task['mutation_confirmation_times'])
    assert task['locked_holdout']['days']==['2020-01-01','2020-01-02','2020-01-03','2020-01-04']


def test_one_reserved_unit_does_not_become_two_independent_evaluations():
    task={'locked_holdout':{'days':['2020-01-01']}}
    reserve_mutation_confirmation({},task)
    assert task['mutation_reservation']['status']=='INSUFFICIENT'
    assert 'mutation_confirmation_times' not in task


@pytest.mark.parametrize('frozen,reviewed,expected', [('child','child',True),('parent','child',False),('child','parent',False)])
def test_pre_mutation_acceptance_cannot_be_inherited(tmp_path,frozen,reviewed,expected):
    (tmp_path/'selection_freeze.json').write_text(json.dumps({'key':frozen}))
    result={'best':{'key':'child'},'recommendation_review':{'recommended':True,'status':'CONFIRMED','selection_key':reviewed}}
    receipt=mutation_acceptance(result,tmp_path)
    assert receipt['recommended']==expected
    assert receipt['probability_guarantee'] is None
    assert receipt['confirmation_retries']==0


def test_good_product_and_recommendation_only_are_not_numerical_acceptance(tmp_path):
    result={'status':'PASS','best':{'key':'child'},'product':'product.nc',
            'recommendation_review':{'recommended':True,'status':'PRE_MUTATION_CHECK'}}
    assert not mutation_acceptance(result,tmp_path)['recommended']


def test_insufficient_isolation_retains_unconfirmed_state(tmp_path):
    (tmp_path/'selection_freeze.json').write_text(json.dumps({'key':'child'}))
    result={'best':{'key':'child'},'recommendation_review':{'recommended':True,'status':'CONFIRMED'},
            'recommendation_isolation':{'status':'INSUFFICIENT'}}
    assert not mutation_acceptance(result,tmp_path)['recommended']


def test_desktop_displays_delivered_not_unconfirmed_winner():
    from ecodata.desktop.service import fusion_evidence
    data=fusion_evidence({'post_mutation_acceptance':{'status':'BASELINE_RETAINED','recommended':False,
                         'numerical_review':{'private_support':'not for DS'}}})
    assert data['post_mutation_acceptance']=={'status':'BASELINE_RETAINED','recommended':False}


def test_spatial_search_consumes_reservation_before_fitness(tmp_path):
    from ecodata.algorithms import platform_search as api
    from ecodata.algorithms.spatial_fusion_task import SpatialTask,SpatialSearch
    from test_spatial_relationship_fusion import fixture,BUDGET
    items,request,_=fixture()
    request.update(confirmation_times=['2000-01-10'],recommendation_isolation=dict(
        status='RESERVED',recommendation_review_times=['2000-01-09']))
    parsed,audit=api.parse_inputs(items)
    engine=SpatialSearch(SpatialTask.build(parsed,request,audit),tmp_path,BUDGET,11)
    assert list(engine.confirm_times)==[pd.Timestamp('2000-01-10')]
    assert pd.Timestamp('2000-01-09') in engine.search_times
    assert not set(engine.search_times)&set(engine.confirm_times)
    request['recommendation_isolation']['recommendation_review_times']=['2000-01-10']
    with pytest.raises(ValueError,match='overlap'):
        SpatialSearch(SpatialTask.build(parsed,request,audit),tmp_path/'conflict',BUDGET,11)


def test_reduced_scale_final_review_executes_reserved_dates_not_cached_selection(tmp_path):
    from test_required_target_grid import task_and_inputs,BUDGET
    from ecodata.algorithms.spatial_fusion_task import SpatialSearch
    task,_,_=task_and_inputs()
    task.request.update(confirmation_times=['2000-01-04'],recommendation_isolation=dict(
        status='RESERVED',recommendation_review_times=['2000-01-03']))
    search=SpatialSearch(task,tmp_path,BUDGET,11)
    assert search.reconstruction is not None
    key=search.leaves[1]
    search.execute(key)
    row=search.rows[-1]
    assert row['status']=='PASS'
    metrics=dict(row['metrics'])
    report=search.post_selection_assessment({'selected':key,'single_source_baseline':key})
    assert report['status']=='PASS'
    assert all(pd.Timestamp(t)==pd.Timestamp('2000-01-04') for t in report['review_support']['times'])
    assert report['schemes']['selected']['trace']
    assert search.rows[0]['metrics']==metrics
    assert report['used_for_selection'] is False


def test_component_format_summary_uses_validated_inventory_not_zip_name():
    from ecodata.desktop.output_review import format_conversion
    result={'source_formats':['NetCDF'],'target_format':'原版工程交付包','manifest':{
        'component_format_generated':True,'preparation':{'tasks':[{'source':{'upload':'input.nc'},'validation':{'records':3}}]},
        'output_inventory':[{'path':'components/SST.wps','sha256':'actual-file-hash'},
                            {'path':'components/conversion_report.json','sha256':'report-hash'}]}}
    report=format_conversion(result)
    assert report['target_formats']==['WPS intermediate v5（.wps）']
    assert report['files'][0]['status']=='CONVERTED'
    result['manifest']['component_format_generated']=False
    assert not format_conversion(result)['files']
