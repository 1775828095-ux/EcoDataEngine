"""Reference-free public recommendations; bounded synthetic acceptance only."""
from copy import deepcopy
import json
import shutil

import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms import platform_search as platform
from ecodata.algorithms import profile_matching as pm
from ecodata.interface.search_page import process_search_upload
from test_profile_matching import fixture


POOL=['raster_task.spatial.nearest','raster_task.spatial.bilinear','raster_task.pchip',
      'raster_task.spatial.xarray_cubic','raster_task.coarse_consistency']


def setup(offset=0.):
    ds,_,req,_=fixture()
    ds[req['variable']].values[:]=np.arange(16.).reshape(4,4)+5+offset
    req['target_lat']=(np.arange(8)*.5+29.75).tolist()
    req['target_lon']=(np.arange(8)*.5+109.75).tolist()
    req['requirements']=dict(try_coarse_consistency=True)
    req['profile_options']=dict(max_steps=2,candidate_limit=64,candidate_pool=dict(algorithms=POOL))
    reference=xr.DataArray(np.repeat(np.repeat(ds[req['variable']].values,2,axis=0),2,axis=1),dims=('lat','lon'),
        coords=dict(lat=req['target_lat'],lon=req['target_lon']),attrs=dict(ds[req['variable']].attrs))
    return ds,req,reference


def inputs(ds):return [platform.SearchInput('visible.nc','primary',ds.to_netcdf())]


def submit(ds,req,folder,evaluation=None,seed=11):
    return platform.submit_search(inputs(ds),req,evaluation,folder,mode='Profile',seed=seed,
                                  budget=dict(objective_limit=40,wall_seconds=30))


def test_pchip_and_cubic_properties_reach_public_scorer_and_execute(tmp_path):
    ds,_,req,reference=fixture()
    req['profile_options']=dict(max_steps=1,candidate_limit=64,candidate_pool=dict(algorithms=['raster_task.pchip','spatial.xarray_cubic']))
    req['model_requirements']=dict(priorities={'continuity':1},hard=[],evidence='functional task requirement')
    result=submit(ds,req,tmp_path/'reviewed',platform.DatasetReference(reference,req['variable']))
    assert len(result['rows'])==2 and all(r['status']=='PASS' for r in result['rows'])
    assert {r['matching']['properties']['continuity'] for r in result['rows']}=={True}
    for row in result['rows']:
        assert row['matching']['score']>0 and row['matching']['terms'][0]['algorithm_evidence']
        assert row['file_readback']=='PASS'
    req['model_requirements']['hard']=['range_preservation']
    filtered=submit(ds,req,tmp_path/'range')
    assert [r['steps'] for r in filtered['recommendations']]==[['raster_task.pchip']]
    p=next(a for a in filtered['portraits']['algorithms'] if a['id']=='raster_task.spatial.xarray_cubic')
    assert p['matching_properties']['noise_robustness'] is None
    assert p['matching_properties']['peak_preservation'] is None


def test_frozen_pool_is_eight_unique_graphs_with_alias_and_seed_invariance(tmp_path):
    ds,req,_=setup()
    a=submit(ds,req,tmp_path/'pool')
    assert a['counts']['candidate_count']==8
    plans={tuple(c['steps']) for c in a['recommendations']}
    bases=POOL[:4]
    assert plans=={(x,) for x in bases}|{(x,POOL[-1]) for x in bases}
    other=deepcopy(req)
    other['profile_options']['candidate_pool']['algorithms']=list(reversed(POOL))+['spatial.nearest','spatial.bilinear','spatial.xarray_cubic']
    b=submit(ds,other,tmp_path/'aliases',seed=99)
    assert b['candidate_pool']['fingerprint']==a['candidate_pool']['fingerprint']
    assert {c['key'] for c in a['recommendations']}=={c['key'] for c in b['recommendations']}
    assert len(b['recommendations'])==8
    pool=json.loads((tmp_path/'aliases/candidate_pool.json').read_text(encoding='utf8'))
    assert pool['aliases']['spatial.nearest']=='raster_task.spatial.nearest'
    assert 'boundary' in pool['boundary_policy'] or 'edge' in pool['boundary_policy']
    assert pool['generation_complete_within_cap']
    # No undeclared nonnegative tail or default registry method can leak in.
    assert all(set(c['steps'])<=set(POOL) for c in b['recommendations'])


def test_frozen_pool_and_hard_model_requirements_share_the_same_boundary(tmp_path):
    ds,req,_=setup()
    req['model_requirements']=dict(priorities={},hard=['coarse_consistency'],evidence='declared coarse operator required')
    result=submit(ds,req,tmp_path/'hard')
    assert result['counts']['candidate_count']==4
    assert all(c['steps'][-1]==POOL[-1] for c in result['recommendations'])
    assert result['candidate_pool']['grammar_candidate_count']==8
    assert result['candidate_pool']['legal_candidate_count']==4


@pytest.mark.parametrize('pool',[
    {'algorithms':['spatial.cubic']},{'algorithms':['invented']},{'algorithms':[]},
    {'algorithms':POOL,'preferred_winner':'x'}])
def test_invalid_frozen_pools_block(pool,tmp_path):
    ds,req,_=setup();req['profile_options']['candidate_pool']=pool
    with pytest.raises(ValueError):submit(ds,req,tmp_path/'bad')


def test_frozen_pool_cannot_silently_truncate(tmp_path):
    ds,req,_=setup();req['profile_options']['candidate_limit']=3
    with pytest.raises(ValueError,match='truncated'):submit(ds,req,tmp_path/'short')


@pytest.fixture
def learned(tmp_path,monkeypatch):
    monkeypatch.setattr(pm,'MODEL_ROOT',tmp_path/'models')
    folders=[]
    for i,offset in enumerate([-1.,1.,0.]):
        ds,req,reference=setup(offset)
        req['profile_options'].update(evidence_role='development',group=f'g{i}')
        folder=tmp_path/f'train{i}'
        result=submit(ds,req,folder,platform.DatasetReference(reference,req['variable']))
        assert result['stop_reason']=='CANDIDATE_POOL_COMPLETED'
        assert len(result['rows'])==8 and all(r['status']=='PASS' for r in result['rows'])
        folders.append(folder)
    platform.train_profile_matching(folders,'reference_free',holdout_groups=['g2'])
    ds,req,_=setup()
    req['profile_options'].update(ranking_model_id='reference_free',acknowledge_limited_development=True)
    req['preferences']=dict(weights={'RMSE':1},scales={'RMSE':1})
    return ds,req,folders


def test_no_reference_models_recommend_through_page_and_file_without_execution(learned,tmp_path,monkeypatch):
    ds,req,folders=learned
    def forbidden(*a,**k):pytest.fail('Recommendation must not execute a pipeline or evaluate reference labels')
    monkeypatch.setattr(platform.RegistryEvaluator,'execute',forbidden)
    monkeypatch.setattr(platform.DatasetReference,'__call__',forbidden)
    a=process_search_upload(inputs(ds),req,None,tmp_path/'page',mode='Profile')
    ds.to_netcdf(tmp_path/'visible.nc')
    b=platform.submit_file_request(dict(inputs=[dict(path=str(tmp_path/'visible.nc'),role='primary')],
        request=req,mode='Profile'),tmp_path/'toolkit')
    assert a['matching_status']==b['matching_status']=='RULE_COST_ORDER_NOT_ACCURACY'
    assert a['learning_status']['affects_selection'] is False
    assert a['counts']['real_executions']==b['counts']['real_executions']==0
    assert a['best'] is None and a['actual_accuracy']=='NOT_ASSESSED'
    assert a['accuracy_threshold_status']=='UNVERIFIED_NO_REFERENCE'
    assert [x['key'] for x in a['recommendations']]==[x['key'] for x in b['recommendations']]
    recorded=json.loads((folders[-1]/'candidate_ranking.json').read_text(encoding='utf8'))
    train_features={c['key']:c['public_features'] for c in recorded['candidates']}
    for c in a['recommendations']:
        assert c['public_features']==train_features[c['key']]
        assert not {'evaluation_kind','metric','reference_sha256'}&set(c['public_features'])
        assert c['predicted_task_loss']==pytest.approx(c['predicted_metrics']['RMSE'])
        assert 'objective' not in c['predicted_metrics']
    assert all(c['predicted_metrics'] is not None for c in a['recommendations'])
    assert len({round(c['predicted_metrics']['RMSE'],8) for c in a['recommendations']})>1


def test_reference_values_never_change_pre_execution_ranking(learned,tmp_path):
    ds,req,_=learned
    _,_,reference=setup();reference.values[:]=9999
    with_reference=submit(ds,req,tmp_path/'changed_truth',platform.DatasetReference(reference,req['variable']))
    without_reference=submit(ds,req,tmp_path/'no_truth')
    ranked=json.loads((tmp_path/'changed_truth/candidate_ranking.json').read_text(encoding='utf8'))['candidates']
    assert [c['key'] for c in ranked]==[c['key'] for c in without_reference['recommendations']]
    assert all(r['hidden_truth_passed_to_nodes'] is False for r in with_reference['rows'])
    # RMSE values are evaluated only afterwards; they are not predictions.
    assert min(r['RMSE'] for r in with_reference['rows'])>9000


def test_objective_semantics_not_crossed_and_changed_pool_abstains(learned,tmp_path):
    ds,req,folders=learned
    original=json.loads((folders[-1]/'summary.json').read_text(encoding='utf8'))
    row=original['rows'][0];service=pm.PlanRankingService('reference_free')
    accuracy=service.predict_metrics(row['public_features'],objective_head=pm.objective_semantics(original['evaluation_protocol']))
    coverage=service.predict_metrics(row['public_features'],objective_head=pm.objective_semantics(platform.EngineeringMetric(req['variable']).protocol))
    assert 'objective' in accuracy and 'objective' not in coverage
    assert accuracy['RMSE']==coverage['RMSE']
    assert service.predict_metrics(dict(row['public_features'],task_context='different target grid')) is None
    corrupted=tmp_path/'mixed_objective_record';shutil.copytree(folders[0],corrupted)
    report=json.loads((corrupted/'summary.json').read_text(encoding='utf8'))
    report['evaluation_protocol']=platform.EngineeringMetric(req['variable']).protocol
    (corrupted/'summary.json').write_text(json.dumps(report),encoding='utf8')
    with pytest.raises(ValueError,match='Different objective semantics'):
        platform.train_profile_matching([corrupted,*folders[1:]],'mixed_heads',holdout_groups=['g2'])
    req['profile_options']['candidate_pool']['algorithms']=POOL[:2]
    fallback=submit(ds,req,tmp_path/'changed_pool')
    assert fallback['matching_status']!='LEARNED_DEVELOPMENT_RANK'
    assert all(c['predicted_metrics'] is None for c in fallback['recommendations'])


def test_public_coarse_operator_matches_frozen_degradation_and_named_ports(tmp_path):
    from ecodata.algorithms.raster_reconstruction_task import degradation
    ds,req,reference=setup()
    coarse,expected=degradation(reference.to_dataset(name=req['variable']),4,req['variable'])
    np.testing.assert_allclose(coarse[req['variable']],ds[req['variable']])
    context,_,_=platform.make_context([('primary',ds)],req)
    np.testing.assert_allclose(context.parameters['degradation_operator'].toarray(),expected.toarray(),atol=1e-12)
    result=submit(ds,req,tmp_path/'ports')
    for candidate in result['recommendations']:
        for node in candidate['matching']['transfer_trace']:
            if node['operation']==POOL[-1]:
                assert node['input_bindings']['field'] in POOL[:4]
                assert node['properties']['coarse_consistency'] is True


def test_no_reference_unsupported_data_abstains_instead_of_claiming_accuracy(learned,tmp_path):
    ds,req,_=learned;ds[req['variable']].values[:]=10000
    result=submit(ds,req,tmp_path/'ood')
    assert result['matching_status']!='LEARNED_DEVELOPMENT_RANK'
    assert all(c['predicted_metrics'] is None for c in result['recommendations'])
    assert result['counts']['real_executions']==0 and result['actual_accuracy']=='NOT_ASSESSED'
