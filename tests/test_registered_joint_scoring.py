"""Synthetic end-to-end acceptance of registry-driven joint scoring/calibration."""
from copy import deepcopy
import json
import shutil

import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms import attribute_registry as registry
from ecodata.algorithms import platform_search as platform
from ecodata.algorithms import profile_matching as pm
from ecodata.algorithms import scoring_calibration as calibration
from ecodata.interface.search_page import process_search_upload
from test_profile_matching import fixture


@pytest.fixture(autouse=True)
def stores(tmp_path,monkeypatch):
    monkeypatch.setattr(registry,'ROOT',tmp_path/'registries')
    monkeypatch.setattr(calibration,'ROOT',tmp_path/'calibrations')
    monkeypatch.setattr(pm,'MODEL_ROOT',tmp_path/'legacy_models')


def request():
    ds,inputs,req,reference=fixture()
    req['profile_options']=dict(max_steps=2,candidate_limit=64,candidate_pool=dict(algorithms=['spatial.nearest','spatial.bilinear']))
    req['model_requirements']=dict(priorities={'continuity':.6,'no_mixing':.4},hard=[],evidence='synthetic declared preparation preference')
    req['preferences']=dict(weights={'RMSE':1},scales={'RMSE':1})
    return ds,req,reference


def submit(ds,req,folder,reference=None,limit=20):
    return platform.submit_search([platform.SearchInput('visible.nc','primary',ds.to_netcdf())],req,
        platform.DatasetReference(reference,req['variable']) if reference is not None else None,
        folder,mode='Profile',budget=dict(objective_limit=limit,wall_seconds=30))


def definitions():return deepcopy(registry.load()['attributes'])


def test_versioned_enable_flags_change_real_selection_and_old_version_replays(tmp_path):
    ds,req,ref=request()
    first=submit(ds,req,tmp_path/'first',ref)
    attrs=definitions();entry=next(a for a in attrs if a['id']=='continuity')
    entry['scoring_enabled']=False;entry['history'].append('Disable optional scoring term for counterfactual acceptance')
    published=platform.publish_attribute_registry('no-continuity',attrs,evidence='functional registry revision')
    req['scoring']=dict(registry_version='no-continuity')
    changed=submit(ds,req,tmp_path/'changed',ref)
    assert first['best']['steps'][0].endswith('bilinear')
    assert changed['best']['steps'][0].endswith('nearest')
    assert published['parent_sha256']==registry.load()['sha256']
    assert next(t for t in changed['best']['matching']['terms'] if t['dimension']=='continuity')['status']=='DISABLED'
    with xr.open_dataset(first['best']['product']) as a,xr.open_dataset(changed['best']['product']) as b:
        assert not np.allclose(a[req['variable']],b[req['variable']])
    # The numerical oracle is diagnostic; it must not silently replace the rule recommendation.
    assert changed['hindsight_best']['steps'][0].endswith('bilinear')
    del req['scoring']
    replay=submit(ds,req,tmp_path/'replayed',ref)
    assert replay['best']['graph']==first['best']['graph']
    with pytest.raises(FileExistsError):platform.publish_attribute_registry('no-continuity',attrs,evidence='cannot overwrite')


def test_new_registered_attribute_has_real_consumer_unknown_stays_unknown(tmp_path):
    ds,req,ref=request();attrs=definitions()
    copied=deepcopy(attrs[2]);copied.update(id='sample_fidelity',definition='Copy source sample values using reviewed no_mixing provider',hard_allowed=False,
        history=['Registered alias of the existing measured/proven property; no new scientific guarantee'])
    unknown=deepcopy(copied);unknown.update(id='noise_response',definition='Noise response not yet measured',provider='unavailable',property=None,
        activation='constant_one',evidence='Unknown; no reviewed noise measurement',history=['Registered unknown; no invented value'])
    attrs.extend([copied,unknown])
    platform.publish_attribute_registry('extended',attrs,evidence='synthetic additive registry acceptance')
    req['scoring']=dict(registry_version='extended')
    req['model_requirements']['priorities']={'continuity':1,'sample_fidelity':2,'noise_response':.1}
    result=submit(ds,req,tmp_path/'extended',ref)
    assert result['best']['steps'][0].endswith('nearest')
    term=next(t for t in result['best']['matching']['terms'] if t['dimension']=='noise_response')
    assert term['guarantee'] is None and term['contribution'] is None and term['status']=='UNKNOWN'
    assert result['best']['matching']['score'] is None
    assert term['interval'][0]<0<term['interval'][1]
    retired=deepcopy(registry.load('extended')['attributes'])
    entry=next(a for a in retired if a['id']=='sample_fidelity')
    entry['scoring_enabled']=False;entry['history'].append('Retire optional term from scoring; retain provenance')
    platform.publish_attribute_registry('retired',retired,parent_version='extended',evidence='functional retirement acceptance')
    req['scoring']['registry_version']='retired'
    after=submit(ds,req,tmp_path/'retired',ref)
    assert after['best']['steps'][0].endswith('bilinear')
    assert next(t for t in after['best']['matching']['terms'] if t['dimension']=='sample_fidelity')['status']=='DISABLED'
    req['model_requirements']['hard']=['noise_response']
    with pytest.raises(ValueError,match='hard requirement'):submit(ds,req,tmp_path/'unknown_hard')


def test_disabled_soft_attribute_cannot_disable_hard_filter_or_projection(tmp_path):
    ds,req,_=request();ds[req['variable']].values-=12
    attrs=definitions();nonnegative=next(a for a in attrs if a['id']=='nonnegative')
    nonnegative['scoring_enabled']=False;nonnegative['history'].append('Disable soft reward only; hard consumer remains')
    platform.publish_attribute_registry('soft-off',attrs,evidence='functional protected constraint test')
    req['scoring']=dict(registry_version='soft-off')
    req['model_requirements'].update(priorities={'nonnegative':1},hard=['nonnegative'])
    req['profile_options']['candidate_pool']['algorithms'].append('raster_task.nonnegative')
    reference=ds[req['variable']].interp(lat=req['target_lat'],lon=req['target_lon'])
    result=submit(ds,req,tmp_path/'hard',reference)
    assert result['best']['steps'][-1]=='raster_task.nonnegative'
    assert all(r['steps'][-1]=='raster_task.nonnegative' for r in result['rows'])
    assert result['best']['model_product_acceptance']['status']=='PASS'
    with xr.open_dataset(result['best']['product']) as output:assert float(output[req['variable']].min())>=0
    req['profile_options']['candidate_pool']['algorithms'].remove('raster_task.nonnegative')
    refused=submit(ds,req,tmp_path/'no_projection')
    assert not refused['recommendations'] and refused['counts']['real_executions']==0


def test_registry_rejects_hard_removal_unimplemented_provider_and_tampering(tmp_path):
    attrs=definitions()
    with pytest.raises(ValueError,match='Protected'):platform.publish_attribute_registry('removed',attrs[1:],evidence='must refuse')
    attrs[0]['provider']='eval:arbitrary_python'
    with pytest.raises(ValueError,match='provider'):platform.publish_attribute_registry('unsafe',attrs,evidence='must refuse')
    attrs=definitions();platform.publish_attribute_registry('retained',attrs,evidence='immutable snapshot')
    path=registry.ROOT/'retained.json';saved=json.loads(path.read_text(encoding='utf-8'));saved['snapshot']['evidence']='tamper'
    path.write_text(json.dumps(saved),encoding='utf-8')
    with pytest.raises(ValueError,match='fingerprint'):registry.load('retained')


def test_weights_on_same_data_change_real_combination_not_only_display(tmp_path):
    ds,req,_=request();ds[req['variable']].values-=12
    req['profile_options']['candidate_pool']['algorithms'].append('raster_task.nonnegative')
    reference=ds[req['variable']].interp(lat=req['target_lat'],lon=req['target_lon'])
    req['model_requirements']['priorities']={'continuity':1}
    smooth=submit(ds,req,tmp_path/'smooth',reference)
    req['model_requirements']['priorities']={'nonnegative':1}
    projected=submit(ds,req,tmp_path/'projected',reference)
    assert len(smooth['best']['steps'])==1
    assert len(projected['best']['steps'])==2 and projected['best']['steps'][-1]=='raster_task.nonnegative'
    assert smooth['portraits']['data']==projected['portraits']['data']
    with xr.open_dataset(smooth['best']['product']) as a,xr.open_dataset(projected['best']['product']) as b:
        assert float(a[req['variable']].min())<0 and float(b[req['variable']].min())>=0


def test_registered_declared_cost_uses_actual_graph_not_invented_accuracy(tmp_path):
    ds,req,_=request();attrs=definitions()
    cost=deepcopy(attrs[0]);cost.update(id='resource_tier',definition='Existing declared compute-tier cost of scientific graph nodes',
        provider='declared_cost',property=None,activation='constant_one',hard_allowed=False,
        evidence='AlgorithmSpec.compute tier mapping; not measured seconds',history=['Explicit optional cost consumer'])
    attrs.append(cost);platform.publish_attribute_registry('with-cost',attrs,evidence='functional declared-cost registration')
    req['scoring']=dict(registry_version='with-cost')
    req['model_requirements']['priorities']={'resource_tier':1}
    result=submit(ds,req,tmp_path/'cost')
    for row in result['recommendations']:
        term=row['matching']['terms'][0]
        assert term['status']=='DECLARED_COST' and term['guarantee'] is None
        assert term['utility']==pytest.approx(-row['declared_cost']/(1+row['declared_cost']))


@pytest.fixture
def development(tmp_path):
    folders=[]
    for i,offset in enumerate([-1.,1.,0.]):
        ds,req,reference=request();ds[req['variable']].values+=offset;reference.values+=offset
        req['model_requirements']['priorities']={'continuity':.2,'no_mixing':.8}
        req['profile_options'].update(evidence_role='development',group=f'g{i}')
        folder=tmp_path/f'dev{i}';result=submit(ds,req,folder,reference)
        assert result['stop_reason']=='CANDIDATE_POOL_COMPLETED' and all(r['file_readback']=='PASS' for r in result['rows'])
        folders.append(folder)
    return folders,ds,req,reference


def test_calibration_reuses_real_matrix_changes_public_recommendation_without_truth(development,tmp_path,monkeypatch):
    folders,ds,req,reference=development
    before=submit(ds,req,tmp_path/'initial')
    assert before['recommendations'][0]['steps'][0].endswith('nearest')
    def forbidden(*a,**k):pytest.fail('Calibration/recommendation must not execute a graph or read reference truth')
    monkeypatch.setattr(platform.RegistryEvaluator,'execute',forbidden)
    monkeypatch.setattr(platform.DatasetReference,'__call__',forbidden)
    report=platform.calibrate_profile_scoring(folders,'calibrated',holdout_groups=['g2'],
        parameter_grid={'no_mixing.multiplier':[0.,1.],'continuity.multiplier':[1.,2.]})
    assert report['real_executions_during_calibration']==0
    assert report['training']['groups']==['g0','g1'] and report['holdout']['groups']==['g2']
    assert all(t['training']['groups']==['g0','g1'] for t in report['trials'])
    assert report['parameters']['no_mixing']['multiplier']==0
    assert report['attribute_diagnostics']['no_mixing']['within_task_contribution_variance']['initial']>0
    assert report['attribute_diagnostics']['no_mixing']['within_task_contribution_variance']['calibrated']==0
    assert not report['hard_constraints_changed'] and not report['user_preferences_changed']
    req['scoring']=dict(calibration_id='calibrated')
    after=submit(ds,req,tmp_path/'calibrated')
    assert after['recommendations'][0]['steps'][0].endswith('bilinear')
    assert after['counts']['real_executions']==0 and after['actual_accuracy']=='NOT_ASSESSED'
    assert after['portraits']['model']['calibration']['status']=='LIMITED_DEVELOPMENT'
    assert after['scoring_policy']=='joint_portrait_v1'
    with pytest.raises(ValueError,match='immutable'):
        platform.calibrate_profile_scoring(folders,'calibrated',holdout_groups=['g2'],parameter_grid={'no_mixing.multiplier':[0]})
    changed=deepcopy(req);changed['model_requirements']['hard']=['continuity']
    with pytest.raises(ValueError,match='context differs'):submit(ds,changed,tmp_path/'wrong_scope')


def test_calibration_rejects_test_records_and_altered_labels(development,tmp_path):
    folders,ds,req,reference=development
    damaged=tmp_path/'damaged';shutil.copytree(folders[0],damaged)
    path=damaged/'summary.json';report=json.loads(path.read_text(encoding='utf-8'))
    report['evidence_role']='test';path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='development'):
        platform.calibrate_profile_scoring([damaged,*folders[1:]],'test_leak',holdout_groups=['g2'],parameter_grid={'no_mixing.multiplier':[0]})
    report['evidence_role']='development';report['rows'][0]['RMSE']=1234.;path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='labels differ'):
        platform.calibrate_profile_scoring([damaged,*folders[1:]],'bad_labels',holdout_groups=['g2'],parameter_grid={'no_mixing.multiplier':[0]})


def test_loaded_calibration_reaches_page_file_and_product_with_fingerprint_guard(development,tmp_path):
    folders,ds,req,reference=development
    platform.calibrate_profile_scoring(folders,'page-calibration',holdout_groups=['g2'],parameter_grid={'no_mixing.multiplier':[0,1]})
    req['scoring']=dict(calibration_id='page-calibration')
    ds=ds.copy(deep=True);reference=reference.copy(deep=True)
    ds[req['variable']].values+=.1;reference.values+=.1
    inputs=[platform.SearchInput('visible.nc','primary',ds.to_netcdf())]
    page=process_search_upload(inputs,req,platform.DatasetReference(reference,req['variable']),tmp_path/'page-calibrated',mode='Profile')
    ds.to_netcdf(tmp_path/'visible.nc');reference.to_dataset(name=req['variable']).to_netcdf(tmp_path/'ref.nc')
    file=platform.submit_file_request(dict(inputs=[dict(path=str(tmp_path/'visible.nc'),role='primary')],request=req,mode='Profile',
        evaluation=dict(kind='reference_error',path=str(tmp_path/'ref.nc'))),tmp_path/'file-calibrated')
    assert page['best']['steps'][0].endswith('bilinear')
    assert page['best']['key']==page['recommended_key'] and page['best']['file_readback']=='PASS'
    assert page['best']['prediction_fingerprint']==file['best']['prediction_fingerprint']
    with xr.open_dataset(page['best']['product']) as output:np.testing.assert_allclose(output[req['variable']],reference,atol=1e-10)
    path=calibration.ROOT/'page-calibration.json'
    saved=json.loads(path.read_text(encoding='utf-8'));saved['report']['parameters']['no_mixing']['multiplier']=1
    path.write_text(json.dumps(saved),encoding='utf-8')
    with pytest.raises(ValueError,match='fingerprint'):submit(ds,req,tmp_path/'tampered-calibration')


def test_page_and_file_use_same_registry_parameters_and_real_products(tmp_path):
    ds,req,reference=request();req['scoring']=dict(parameters={'continuity':{'multiplier':0}})
    inputs=[platform.SearchInput('visible.nc','primary',ds.to_netcdf())]
    page=process_search_upload(inputs,req,platform.DatasetReference(reference,req['variable']),tmp_path/'page',mode='Profile')
    ds.to_netcdf(tmp_path/'visible.nc');reference.to_dataset(name=req['variable']).to_netcdf(tmp_path/'ref.nc')
    file=platform.submit_file_request(dict(inputs=[dict(path=str(tmp_path/'visible.nc'),role='primary')],request=req,mode='Profile',
        evaluation=dict(kind='reference_error',path=str(tmp_path/'ref.nc'))),tmp_path/'file')
    assert page['best']['file_readback']==file['best']['file_readback']=='PASS'
    assert page['best']['graph']==file['best']['graph']
    assert page['best']['matching']==file['best']['matching']
    assert page['best']['prediction_fingerprint']==file['best']['prediction_fingerprint']


def test_legacy_learning_cannot_replace_joint_rule_order(development,tmp_path,monkeypatch):
    folders,ds,req,reference=development
    platform.train_profile_matching(folders,'diagnostic',holdout_groups=['g2'])
    req['model_requirements']['priorities']={'continuity':1}
    normal=submit(ds,req,tmp_path/'normal')
    monkeypatch.setattr(pm.PlanRankingService,'score',lambda self,x:1000. if 'nearest' in str(x) else -1000.)
    monkeypatch.setattr(pm.PlanRankingService,'predict_metrics',lambda *a,**k:dict(RMSE=0.,MAE=0.,abs_bias=0.))
    req['profile_options'].update(ranking_model_id='diagnostic',acknowledge_limited_development=True,learning_order='task_rank',fallback_policy='development_fixed')
    diagnostic=submit(ds,req,tmp_path/'diagnostic')
    assert diagnostic['recommendations'][0]['key']==normal['recommendations'][0]['key']
    assert diagnostic['recommendations'][0]['steps'][0].endswith('bilinear')
    assert diagnostic['learning_status']['affects_selection'] is False
    assert diagnostic['learning_status']['development_fixed_used'] is False
    assert diagnostic['matching_status']=='THREE_PORTRAIT_JOINT_SCORE'
