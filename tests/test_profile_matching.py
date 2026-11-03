"""Synthetic functional fixtures, never evidence of environmental accuracy."""
import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms import profile_matching as pm
from ecodata.algorithms import platform_search as platform
from ecodata.interface.search_page import process_search_upload


def fixture(model="wrf"):
    table=pm.model_table(model)
    role,field=next((r,f) for r in table['roles'] if not r['record_only'] for f in r['fields']
                    if f.get('data_type',f.get('kind','continuous'))=='continuous'
                    and f.get('unit','').casefold() not in pm.GENERIC_UNITS)
    variable=field['name']; lat=np.arange(4.)+30;lon=np.arange(4.)+110
    ds=xr.Dataset({variable:(("lat","lon"),np.arange(16.).reshape(4,4)+5)},coords=dict(lat=lat,lon=lon),attrs=dict(crs="EPSG:4326"))
    ds[variable].attrs.update(units=field['unit'],data_kind='continuous',spatial_semantics='cell_mean',
                              physical_variable=field.get('standard_name',variable))
    target=[30.5,31.5,32.5];target_lon=[110.5,111.5,112.5]
    req=dict(task='spatial',variable=variable,unit=field['unit'],semantic_type='continuous',nonnegative=False,
             target_lat=target,target_lon=target_lon,spatial_semantics='cell_mean',parameters={},
             model_profile=dict(id=model,role=role['role'],field=variable),profile_options=dict(candidate_limit=16))
    reference=ds[variable].interp(lat=target,lon=target_lon)
    inputs=[platform.SearchInput('fixture.nc','primary',ds.to_netcdf())]
    return ds,inputs,req,reference


@pytest.mark.parametrize('model',pm.MODELS)
def test_seven_profiles_and_real_public_numerical_execution(tmp_path,monkeypatch,model):
    ds,inputs,req,reference=fixture(model)
    monkeypatch.setattr(platform,'run_numerical_search',lambda *a,**k:pytest.fail('Evolution must not be invoked'))
    result=platform.submit_search(inputs,req,platform.DatasetReference(reference,req['variable']),tmp_path/model,
                                  mode='Profile',budget=dict(objective_limit=12,wall_seconds=30))
    assert result['best'] and result['best']['file_readback']=='PASS'
    assert result['evolution_used'] is False and result['portraits']['model']['sensitivity'] is None
    assert result['portraits']['model']['model_id']==model
    assert result['native_model_acceptance']=='NOT_ASSESSED'
    assert all(len(r['steps'])<=3 and r['hidden_truth_passed_to_nodes'] is False for r in result['rows'])
    assert any('stage_portrait' in a['metadata'] for n in result['best']['node_trace'] for a in n.get('outputs',{}).values())
    with xr.open_dataset(result['best']['product']) as actual:
        assert actual[req['variable']].attrs['units']==req['unit']
    assert (tmp_path/model/'candidate_ranking.json').is_file()


def test_page_toolkit_and_preference_consumption(tmp_path):
    ds,inputs,req,reference=fixture()
    req['preferences']=dict(weights=dict(MAE=1),scales=dict(MAE=2))
    first=process_search_upload(inputs,req,platform.DatasetReference(reference,req['variable']),tmp_path/'page',mode='Profile',budget=dict(objective_limit=12,wall_seconds=30))
    input_path=tmp_path/'input.nc';ds.to_netcdf(input_path)
    reference_path=tmp_path/'reference.nc';reference.to_dataset(name=req['variable']).to_netcdf(reference_path)
    config=dict(inputs=[dict(path=str(input_path),role='primary')],request=req,mode='Profile',
                evaluation=dict(kind='reference_error',path=str(reference_path)),budget=dict(objective_limit=12,wall_seconds=30))
    second=platform.submit_file_request(config,tmp_path/'toolkit')
    assert first['best']['graph']==second['best']['graph']
    assert first['portraits']['model']==second['portraits']['model']
    assert first['best']['task_loss']==pytest.approx(first['best']['MAE']/2)
    changed=dict(req,preferences=dict(weights=dict(RMSE=1),scales=dict(RMSE=3)))
    _,a=pm.prepare_model_request(req);_,b=pm.prepare_model_request(changed)
    assert a['application_preferences']!=b['application_preferences']
    assert pm.task_loss(dict(objective=2,RMSE=2,MAE=1),a['application_preferences'])==.5
    assert pm.task_loss(dict(objective=2,RMSE=2,MAE=1),b['application_preferences'])==pytest.approx(2/3)


def test_profile_conflicts_and_unknown_are_not_scored():
    ds,inputs,req,_=fixture()
    with pytest.raises(ValueError,match='unit differs'):pm.prepare_model_request(dict(req,unit='wrong'))
    with pytest.raises(ValueError,match='field definition'):pm.prepare_model_request(dict(req,model_profile=dict(req['model_profile'],field='invented')))
    with pytest.raises(ValueError,match='reviewed profile'):pm.model_table('epicc')
    with pytest.raises(ValueError,match='no score consumer'):pm.validate_preferences(dict(unknown_dynamics=1))
    with pytest.raises(ValueError,match='unavailable'):pm.task_loss(dict(objective=1),pm.validate_preferences(dict(weights=dict(RMSE=1))))
    assert pm.task_loss(dict(objective=2),pm.validate_preferences(dict(thresholds=dict(objective=1)))) is None
    ctx,_,_=platform.make_context([('primary',ds)],req)
    _,table=pm.prepare_model_request(req)
    ds[req['variable']].attrs['physical_variable']='unrelated_concentration'
    with pytest.raises(ValueError,match='physical variable'):pm.validate_model_context(ctx,table)


def test_engineering_metric_cannot_satisfy_accuracy_preferences(tmp_path):
    _,inputs,req,_=fixture()
    req['preferences']=dict(weights=dict(RMSE=1))
    with pytest.raises(ValueError,match='Accuracy preference'):
        platform.submit_search(inputs,req,platform.EngineeringMetric(req['variable']),tmp_path/'bad',mode='Profile')


def test_without_reference_only_recommends_and_never_executes(tmp_path,monkeypatch):
    _,inputs,req,_=fixture()
    monkeypatch.setattr(platform.RegistryEvaluator,'execute',lambda *a,**k:pytest.fail('No numerical evaluation authorised'))
    result=platform.submit_search(inputs,req,None,tmp_path/'recommend',mode='Profile')
    assert result['stop_reason']=='RECOMMENDATION_ONLY'
    assert result['best'] is None and result['counts']['real_executions']==0
    assert result['recommendations']


@pytest.mark.parametrize('value',[dict(weights=dict(objective=-1)),dict(scales=dict(objective=0)),dict(tie_tolerance=float('nan')),dict(weights=dict(unknown=1))])
def test_invalid_scoring_protocol(value):
    with pytest.raises(ValueError):pm.validate_preferences(value)


def test_rank_training_requires_real_recorded_products_and_abstains(tmp_path,monkeypatch):
    monkeypatch.setattr(pm,'MODEL_ROOT',tmp_path/'models')
    folders=[]
    for index in range(3):
        _,inputs,req,reference=fixture()
        req['profile_options'].update(evidence_role='development',group='source'+str(index))
        folder=tmp_path/('run'+str(index))
        result=platform.submit_search(inputs,req,platform.DatasetReference(reference,req['variable']),folder,mode='Profile',budget=dict(objective_limit=40,wall_seconds=30))
        assert result['stop_reason']=='CANDIDATE_POOL_COMPLETED'
        # Independent fixtures must differ in content, not merely group labels.
        folders.append(folder)
    with pytest.raises(ValueError,match='Same source content'):
        pm.PlanRankingService.train(folders,'duplicate_fixture',holdout_groups=['source2'])


def test_rank_service_actual_fit_load_predict_and_abstention(tmp_path,monkeypatch):
    monkeypatch.setattr(pm,'MODEL_ROOT',tmp_path/'models')
    folders=[]
    for index,offset in enumerate([-1.,1.,0.]):
        ds,_,req,reference=fixture()
        ds[req['variable']].values+=offset
        reference.values+=offset
        req['profile_options'].update(evidence_role='development',group='region'+str(index))
        folder=tmp_path/('run'+str(index))
        result=platform.submit_search([platform.SearchInput('visible.nc','primary',ds.to_netcdf())],req,
            platform.DatasetReference(reference,req['variable']),folder,mode='Profile',budget=dict(objective_limit=40,wall_seconds=30))
        folders.append(folder)
    report=pm.PlanRankingService.train(folders,'functional_fixture',holdout_groups=['region2'])
    service=pm.PlanRankingService('functional_fixture')
    vector=next(r['public_features'] for r in result['rows'] if r['status']=='PASS')
    assert report['status']=='LIMITED_DEVELOPMENT'
    assert service.score(vector) is not None
    assert service.score(dict(vector,model='unseen')) is None
    numeric=next(k for k,v in vector.items() if not isinstance(v,str) and k.startswith('source'))
    assert service.score(dict(vector,**{numeric:1e20})) is None
    req['profile_options'].update(ranking_model_id='functional_fixture',acknowledge_limited_development=True)
    scored=platform.submit_search([platform.SearchInput('visible.nc','primary',ds.to_netcdf())],req,
        platform.DatasetReference(reference,req['variable']),tmp_path/'scored',mode='Profile',budget=dict(objective_limit=40,wall_seconds=30))
    ranking=json.loads((tmp_path/'scored/candidate_ranking.json').read_text())
    assert ranking['status']=='RULE_COST_ORDER_NOT_ACCURACY'
    assert scored['learning_status']['affects_selection'] is False
    assert all(r['predicted_rank'] is not None for r in scored['rows'])
    assert scored['hindsight_best']['task_loss']==min(r['task_loss'] for r in scored['rows'] if r['task_loss'] is not None)
    assert scored['best']['key']==scored['recommended_key']
    with pytest.raises(ValueError,match='Immutable'):
        pm.PlanRankingService.train(folders,'functional_fixture',holdout_groups=['region2'])
    with (tmp_path/'models/functional_fixture/model.joblib').open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='fingerprint'):pm.PlanRankingService('functional_fixture')


def test_streamlit_profile_mode_exposes_seven_models():
    from streamlit.testing.v1 import AppTest
    def app():
        from ecodata.interface.search_page import render_search_page
        render_search_page()
    app_test=AppTest.from_function(app,default_timeout=30).run()
    app_test.selectbox(key='unified_selection_mode').select('Profile').run()
    assert not app_test.exception
    assert app_test.selectbox(key='portrait_model').options==list(pm.MODELS)
    app_test.selectbox(key='portrait_model').select('wrf_chem').run()
    assert not app_test.exception
    assert app_test.selectbox(key='unified_selection_mode').value=='Profile'
