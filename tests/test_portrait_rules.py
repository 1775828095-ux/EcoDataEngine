"""Public-entry counterfactual acceptance; synthetic cases are not scientific gains."""
from copy import deepcopy
import json
import shutil

import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms import platform_search as platform
from ecodata.algorithms import profile_matching as pm
from ecodata.algorithms.portrait_rules import model_requirements
from ecodata.interface.search_page import process_search_upload
from test_profile_matching import fixture


def request_with(req, priorities=None, hard=None):
    result=deepcopy(req)
    result['profile_options']['candidate_limit']=64
    result['model_requirements']=dict(priorities=priorities or {}, hard=hard or [],
        evidence='Synthetic acceptance task contract; not inferred WRF sensitivity')
    return result


def run(ds, req, folder, *, reference=None, limit=1):
    inputs=[platform.SearchInput('visible.nc','primary',ds.to_netcdf())]
    evaluator=platform.DatasetReference(reference,req['variable']) if reference is not None else platform.EngineeringMetric(req['variable'])
    return platform.submit_search(inputs,req,evaluator,folder,mode='Profile',
                                  budget=dict(objective_limit=limit,wall_seconds=30))


def test_same_input_model_requirement_changes_algorithm_and_real_product(tmp_path):
    ds,_,req,reference=fixture()
    smooth=run(ds,request_with(req,{'continuity':1}),tmp_path/'continuous',reference=reference)
    samples=run(ds,request_with(req,{'no_mixing':1}),tmp_path/'samples',reference=reference)
    assert smooth['portraits']['data']==samples['portraits']['data']
    assert smooth['best']['steps'][0].endswith('bilinear')
    assert samples['best']['steps'][0].endswith('nearest')
    assert smooth['best']['graph']!=samples['best']['graph']
    with xr.open_dataset(smooth['best']['product']) as a, xr.open_dataset(samples['best']['product']) as b:
        np.testing.assert_allclose(a[req['variable']],reference)
        assert not np.allclose(a[req['variable']],b[req['variable']])
    for result in (smooth,samples):
        row=result['best'];assert row['numerical_execution_attempted'] and row['file_readback']=='PASS'
        assert row['matching']['terms'][0]['algorithm_evidence']
        assert row['matching']['terms'][0]['model_evidence']
        assert row['matching']['terms'][0]['activation']>0


def test_same_requirements_changed_data_adds_real_projection_edge(tmp_path):
    ds,_,req,_=fixture();req=request_with(req,{'nonnegative':1})
    positive=run(ds,req,tmp_path/'positive')
    signed=ds.copy(deep=True);signed[req['variable']].values-=12
    negative=run(signed,req,tmp_path/'signed')
    assert len(positive['best']['steps'])==1
    assert len(negative['best']['steps'])==2
    assert negative['best']['steps'][-1]=='raster_task.nonnegative'
    graph=negative['best']['graph'];tail=next(n for n in graph if n['operation']=='raster_task.nonnegative')
    assert len(tail['parents'])==1 and graph[tail['parents'][0]]['operation'] in negative['best']['steps']
    with xr.open_dataset(negative['best']['product']) as output:
        assert float(output[req['variable']].min())>=0
    assert positive['portraits']['model']['matching_requirements']==negative['portraits']['model']['matching_requirements']
    assert positive['best']['matching']['terms'][0]['activation']==0
    assert negative['best']['matching']['terms'][0]['activation']>0


def test_hard_filter_rejects_unknown_and_violated_guarantees_before_execution(tmp_path):
    ds,_,req,_=fixture()
    req=request_with(req,hard=['continuity','range_preservation'])
    result=run(ds,req,tmp_path/'hard',limit=30)
    ranking=json.loads((tmp_path/'hard/candidate_ranking.json').read_text())
    assert ranking['rejected']
    assert all(row['matching']['properties']['continuity'] is True for row in result['rows'])
    assert all(row['matching']['properties']['range_preservation'] is True for row in result['rows'])
    assert not any('cubic' in s or s.endswith('nearest') for row in result['rows'] for s in row['steps'])
    assert all(row['matching']['hard_failures'] for row in ranking['rejected'] if 'matching' in row)
    assert all(row['model_product_acceptance']['status']=='PASS' for row in result['rows'] if row['status']=='PASS')


def test_impossible_model_requirements_return_no_accepted_graph(tmp_path):
    ds,_,req,_=fixture();ds[req['variable']].values[:]=-5
    result=run(ds,request_with(req,hard=['range_preservation','nonnegative']),tmp_path/'impossible',limit=30)
    assert result['best'] is None and result['counts']['real_executions']==0
    assert result['counts']['rejected']>0


def test_unknown_is_interval_not_filled_and_no_reference_recommendation_only(tmp_path):
    ds,_,req,_=fixture();req=request_with(req,{'coarse_consistency':1})
    result=platform.submit_search([platform.SearchInput('input.nc','primary',ds.to_netcdf())],req,None,
                                 tmp_path/'unknown',mode='Profile')
    assert result['counts']['real_executions']==0
    cubic=next(c for c in result['recommendations'] if c['steps']==['spatial.xarray_cubic'])
    assert cubic['matching']['score'] is None
    term=cubic['matching']['terms'][0]
    assert term['contribution'] is None and term['interval'][0]<0<term['interval'][1]
    algo=next(a for a in result['portraits']['algorithms'] if a['id']=='spatial.xarray_cubic')
    assert algo['matching_properties']['noise_robustness'] is None
    assert result['portraits']['model']['sensitivity'] is None


def test_combination_scoring_propagates_not_adds_and_records_observed_gain(tmp_path):
    ds,_,req,reference=fixture();req=request_with(req,{'continuity':1})
    result=run(ds,req,tmp_path/'combination',reference=reference,limit=40)
    rows={tuple(r['steps']):r for r in result['rows']}
    base=rows[('spatial.bilinear',)]
    combined=rows[('spatial.bilinear','raster_task.nonnegative')]
    assert base['matching']['score']==combined['matching']['score']
    assert combined['matching']['properties']['continuity'] is True
    gain=next(g for g in result['combination_gains'] if g['key']==combined['key'])
    singles=[r['task_loss'] for r in result['rows'] if len(r['steps'])==1 and r['status']=='PASS']
    assert gain['gain']==pytest.approx(min(singles)-combined['task_loss'])
    assert gain['gain']<=0 and gain['accuracy_claim'] is True


def test_coarse_model_requirement_builds_and_executes_projection(tmp_path):
    ds,_,req,_=fixture();ds[req['variable']].values[:]=5
    req['target_lat']=(np.arange(8)*.5+29.75).tolist()
    req['target_lon']=(np.arange(8)*.5+109.75).tolist()
    result=run(ds,request_with(req,hard=['coarse_consistency','nonnegative']),tmp_path/'coarse',limit=30)
    assert result['best'] is not None
    assert result['best']['steps'][-1]=='raster_task.coarse_consistency'
    assert result['best']['matching']['properties']['coarse_consistency'] is True
    with xr.open_dataset(result['best']['product']) as output:
        np.testing.assert_allclose(output[req['variable']],5,atol=1e-5)
    assert result['best']['task_acceptance']


def test_outside_domain_cannot_borrow_interior_guarantee(tmp_path):
    ds,_,req,_=fixture();req['target_lat']=[29.,31.,34.]
    result=run(ds,request_with(req,hard=['range_preservation']),tmp_path/'outside')
    assert result['best'] is None and result['counts']['real_executions']==0


def test_changing_algorithm_portrait_is_consumed_by_public_filter(tmp_path,monkeypatch):
    import ecodata.algorithms.portrait_rules as rules
    original=rules.algorithm_properties
    def unavailable_evidence(plugin):
        result=original(plugin)
        if result['transfer'] in {'bilinear','pchip','cubic'}:
            result=dict(result,transfer='unknown',evidence=[])
        return result
    ds,_,req,_=fixture();req=request_with(req,hard=['continuity'])
    normal=run(ds,req,tmp_path/'normal')
    assert normal['best'] is not None
    monkeypatch.setattr(rules,'algorithm_properties',unavailable_evidence)
    unconfirmed=run(ds,req,tmp_path/'withdrawn_evidence')
    assert unconfirmed['best'] is None and unconfirmed['counts']['real_executions']==0


def test_page_and_file_clients_consume_identical_three_tables(tmp_path):
    ds,inputs,req,reference=fixture();req=request_with(req,{'no_mixing':1})
    first=process_search_upload(inputs,req,platform.DatasetReference(reference,req['variable']),tmp_path/'page',
                                mode='Profile',budget=dict(objective_limit=1,wall_seconds=30))
    ds.to_netcdf(tmp_path/'input.nc');reference.to_dataset(name=req['variable']).to_netcdf(tmp_path/'reference.nc')
    second=platform.submit_file_request(dict(inputs=[dict(path=str(tmp_path/'input.nc'),role='primary')],request=req,
        mode='Profile',evaluation=dict(kind='reference_error',path=str(tmp_path/'reference.nc')),
        budget=dict(objective_limit=1,wall_seconds=30)),tmp_path/'file')
    assert first['best']['graph']==second['best']['graph']
    assert first['best']['matching']==second['best']['matching']
    # Changing unrelated presentation metadata cannot alter the decision.
    ds.attrs['description']='different display title'
    third=run(ds,req,tmp_path/'unrelated',reference=reference)
    assert third['best']['graph']==first['best']['graph']


@pytest.mark.parametrize('value',[
    dict(priorities={'invented':1},evidence='x'),dict(priorities={'continuity':-1},evidence='x'),
    dict(priorities={'continuity':1}),dict(hard=['noise']),dict(priorities={'continuity':float('nan')},evidence='x')])
def test_unconsumed_or_unjustified_model_fields_block(value):
    with pytest.raises(ValueError):model_requirements(value)


def test_empirical_learning_populates_tables_and_reweights_metric_predictions(tmp_path,monkeypatch):
    monkeypatch.setattr(pm,'MODEL_ROOT',tmp_path/'models')
    folders=[]
    for i,offset in enumerate([-1.,1.,0.]):
        ds,_,req,reference=fixture();ds[req['variable']].values+=offset;reference.values+=offset
        req=request_with(req,{'continuity':1})
        req['profile_options'].update(evidence_role='development',group=f'g{i}')
        folder=tmp_path/f'development{i}'
        run(ds,req,folder,reference=reference,limit=40);folders.append(folder)
    report=platform.train_profile_matching(folders,'metrics',holdout_groups=['g2'])
    assert {'RMSE','MAE','abs_bias'}<=set(report['metric_models'])
    assert all(v['supported']>0 for v in report['metric_validation'].values())
    req['profile_options'].update(ranking_model_id='metrics',acknowledge_limited_development=True)
    req['preferences']=dict(weights={'RMSE':1},scales={'RMSE':2})
    a=run(ds,req,tmp_path/'learned_rmse',reference=reference,limit=40)
    req['preferences']=dict(weights={'MAE':1},scales={'MAE':3})
    b=run(ds,req,tmp_path/'learned_mae',reference=reference,limit=40)
    assert a['matching_status']==b['matching_status']=='THREE_PORTRAIT_JOINT_SCORE'
    assert a['learning_status']['affects_selection'] is False
    ar={r['key']:r for r in a['rows']};br={r['key']:r for r in b['rows']}
    assert set(ar)==set(br)
    for key in ar:
        assert ar[key]['predicted_metrics']==br[key]['predicted_metrics']
        assert ar[key]['predicted_task_loss']==pytest.approx(ar[key]['predicted_metrics']['RMSE']/2)
        assert br[key]['predicted_task_loss']==pytest.approx(br[key]['predicted_metrics']['MAE']/3)
    learned=next(x for x in a['portraits']['algorithms'] if x['id']=='spatial.bilinear')
    assert learned['empirical_cost'] and learned['empirical_side_effects']
    assert {c['source']['group'] for c in a['portraits']['historical_data_cases']}=={'g0','g1'}
    assert a['portraits']['model']['empirical_preparation_cases']
    assert a['portraits']['model']['sensitivity'] is None
    service=pm.PlanRankingService('metrics')
    vector=next(iter(ar.values()))['public_features']
    assert service.predict_metrics(dict(vector,model='unseen')) is None
    # Editable summaries cannot silently replace real numerical labels.
    corrupted=tmp_path/'corrupted_development'
    shutil.copytree(folders[0],corrupted)
    summary=corrupted/'summary.json';content=json.loads(summary.read_text(encoding='utf-8'));content['rows'][0]['RMSE']=123
    summary.write_text(json.dumps(content),encoding='utf-8')
    with pytest.raises(ValueError,match='differs from numerical execution'):
        pm.PlanRankingService.train([corrupted,*folders[1:]],'tampered',holdout_groups=['g2'])
