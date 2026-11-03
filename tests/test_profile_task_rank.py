from copy import deepcopy
import pytest
from ecodata.algorithms import platform_search as platform
from ecodata.algorithms import profile_matching as pm
from test_profile_reference_free import setup,submit,inputs


def test_rank_order_is_public_preference_bound_and_keeps_support_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(pm,'MODEL_ROOT',tmp_path/'models')
    folders=[]
    for i in range(3):
        ds,req,reference=setup(float(i));req['preferences']=dict(weights={'RMSE':1},scales={'RMSE':1})
        req['profile_options'].update(evidence_role='development',group=f'g{i}')
        folder=tmp_path/f'dev{i}';submit(ds,req,folder,platform.DatasetReference(reference,req['variable']));folders.append(folder)
    platform.train_profile_matching(folders,'rank',holdout_groups=['g2'])
    monkeypatch.setattr(platform.RegistryEvaluator,'execute',lambda *a,**k:pytest.fail('Recommendation must not execute a graph'))
    monkeypatch.setattr(platform.DatasetReference,'__call__',lambda *a,**k:pytest.fail('Recommendation must not read evaluation truth'))
    ds,req,_=setup();req['preferences']=dict(weights={'RMSE':1},scales={'RMSE':1})
    req['profile_options'].update(ranking_model_id='rank',acknowledge_limited_development=True,learning_order='task_rank')
    result=submit(ds,req,tmp_path/'rank')
    assert result['matching_status']=='RULE_COST_ORDER_NOT_ACCURACY'
    assert result['counts']['real_executions']==0 and result['best'] is None
    assert result['learning_status']['affects_selection'] is False
    assert all(c['support_diagnostics']['supported'] for c in result['recommendations'])
    platform.publish_profile_policy('rank','published',learning_order='task_rank',fallback_policy='development_fixed',
        validation=folders[2]/'summary.json')
    published=deepcopy(req);published['profile_options']['ranking_model_id']='published'
    del published['profile_options']['learning_order']
    from ecodata.interface.search_page import process_search_upload
    default=process_search_upload(inputs(ds),published,None,tmp_path/'page_defaults',mode='Profile')
    assert default['recommendations'][0]['key']==result['recommendations'][0]['key']
    assert default['learning_status']['learning_order']=='task_rank'
    assert default['learning_status']['fallback_policy']=='development_fixed'
    ds[req['variable']].values+=100
    rejected=submit(ds,req,tmp_path/'ood')
    assert rejected['matching_status']!='LEARNED_DEVELOPMENT_RANK'
    assert all(c['predicted_metrics'] is None for c in rejected['recommendations'])
    assert all(c['support_diagnostics']['reason']=='OUTSIDE_NUMERIC_SUPPORT' for c in rejected['recommendations'])
    fallback=deepcopy(req);fallback['profile_options']['fallback_policy']='development_fixed'
    fixed=submit(ds,fallback,tmp_path/'fixed_fallback')
    assert fixed['matching_status']=='RULE_COST_ORDER_NOT_ACCURACY'
    assert fixed['learning_status']['development_fixed_used'] is False
    assert fixed['learning_status']['supported_candidates']==0
    assert all(c['predicted_metrics'] is None for c in fixed['recommendations'])
    assert fixed['counts']['real_executions']==0
    changed=deepcopy(req);changed['preferences']=dict(weights={'MAE':1},scales={'MAE':1})
    different=submit(ds,changed,tmp_path/'different_preferences')
    assert different['learning_status']['affects_selection'] is False
    assert all(c['predicted_rank'] is None for c in different['recommendations'])
