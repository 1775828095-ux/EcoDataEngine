"""Synthetic behavioral evidence; no environmental accuracy claims."""
import numpy as np
import pytest
from test_required_target_grid import task_and_inputs,BUDGET
from ecodata.algorithms.spatial_fusion_task import SpatialSearch
from ecodata.evaluation.fusion_protocol import prepare,degrade,remap


def test_all_sources_are_degraded_and_reference_is_geometry_selected():
    task,_,_=task_and_inputs();plan,r=prepare(task,task.times,11)
    assert plan['mode']=='REDUCED_SCALE_PRODUCT_RECONSTRUCTION'
    assert plan['reference_source']=='9' and plan['source_bonus']==0
    assert plan['independence']=='UNKNOWN' and not plan['accuracy_established']
    for source,shape in zip(task.native_fields,plan['degraded_shapes']):
        assert shape[1]<source.sizes['lat'] and shape[2]<source.sizes['lon']
    assert r['cohort']['y'].shape[1]==plan['conditioning_count']
    assert set(r['selected']).isdisjoint(r['test'])
    assert r['cohort']['query_count']==r['reference'].shape[1]


def test_existing_area_operator_preserves_constant_and_rejects_missing_support():
    task,_,_=task_and_inputs();ds=task.native_fields[0].copy(deep=True)
    ds.temperature.values[:]=7
    coarse=degrade(ds,'temperature')
    assert np.allclose(coarse.temperature,7)
    fine=ds.sel(lat=slice(float(coarse.lat[0]),float(coarse.lat[-1])),lon=slice(float(coarse.lon[0]),float(coarse.lon[-1])))
    assert np.allclose(remap(coarse,fine,'temperature'),7)


def test_outer_reference_never_changes_selection_fitness_or_candidate_inputs(tmp_path):
    task,_,_=task_and_inputs();a=SpatialSearch(task,tmp_path/'a',BUDGET,11)
    task,_,_=task_and_inputs();b=SpatialSearch(task,tmp_path/'b',BUDGET,11)
    r=b.reconstruction
    assert len(r['test'])
    original=[x.copy() for x in r['cohort']['sources']]
    r['reference'][:,r['test']]+=100000
    a.execute(a.leaves[1]);b.execute(b.leaves[1])
    assert a.rows[0]['metrics']==b.rows[0]['metrics']
    assert all(np.array_equal(x,y) for x,y in zip(original,r['cohort']['sources']))
    assert np.array_equal(a.reconstruction_predictions[a.leaves[1]],b.reconstruction_predictions[b.leaves[1]])


def test_no_source_count_bonus_and_real_ablations_share_one_queue(tmp_path):
    task,_,_=task_and_inputs();engine=SpatialSearch(task,tmp_path,BUDGET,11)
    keys=[*engine.leaves[:2],engine.intern('spatial.blend',engine.leaves[:2],dict(weight=.5))]
    for key in keys:engine.execute(key)
    assert all(r['status']=='PASS' for r in engine.rows)
    assert len(set(r['metrics']['count'] for r in engine.rows))==1
    for row in engine.rows:
        saved=np.load(tmp_path/(row['key']+'-evaluation.npz'))
        i=saved['selection_ids'];e=saved['prediction'][:,i]-saved['reference'][:,i]
        assert row['metrics']['RMSE']==pytest.approx(float(np.sqrt(np.mean(e**2))))
        assert row['metrics']['objective']==pytest.approx(row['blocked_validation']['objective'])
        assert row['metrics']['objective_name']=='equal_date_region_RMSE'
        assert row['coarse_reference_diagnostic']['used_for_selection'] is False
    report=engine.comparison_report(engine.best())
    assert len(report['source_ablations'])==2 and report['paired_selection']
    for item in report['source_ablations']:
        row=next(r for r in engine.rows if r['key']==item['best_key'])
        assert item['omitted_source'] not in row['source_identities']
    assessment=engine.post_selection_assessment(dict(selected=engine.best()['key']))
    assert assessment['used_for_selection'] is False


def test_missing_native_support_does_not_invent_fine_accuracy():
    task,_,_=task_and_inputs();task.native_fields=None
    plan,r=prepare(task,task.times,11)
    assert r is None and not plan['accuracy_established']
    assert plan['mode']=='COARSE_PRODUCT_RECONSTRUCTION_ONLY' and plan['missing_evidence']


def test_observation_mode_does_not_automatically_assert_independence():
    task,_,_=task_and_inputs();task.self_reference=False
    plan,r=prepare(task,task.times,11)
    assert r is None and plan['mode']=='OBSERVATION_HOLDOUT'
    assert plan['independence']=='UNKNOWN'


def test_unknown_policy_rejected_and_legacy_replay_is_explicit(tmp_path):
    task,_,_=task_and_inputs();task.request['evaluation_policy']='invented'
    with pytest.raises(ValueError,match='policy'):SpatialSearch(task,tmp_path/'bad',BUDGET,11)
    task.request['evaluation_policy']='legacy_primary_grid_v1'
    engine=SpatialSearch(task,tmp_path/'old',BUDGET,11)
    assert engine.reconstruction is None
    assert engine.evaluation_plan['mode']=='PRIMARY_GRID_RECONSTRUCTION_PROXY'


def test_desktop_evidence_exposes_new_score_and_separate_coarse_diagnostic(tmp_path):
    from ecodata.desktop.service import fusion_evidence
    task,_,_=task_and_inputs();engine=SpatialSearch(task,tmp_path,BUDGET,11)
    engine.execute(engine.leaves[0])
    comparison=engine.comparison_report(engine.best())
    info=fusion_evidence(dict(selection_comparison=comparison,evaluation_protocol=engine.evaluation_plan))
    assert info['reference_basis']=='REDUCED_SCALE_PRODUCT_RECONSTRUCTION'
    assert info['coarse_diagnostics']['SINGLE_SOURCE']['used_for_selection'] is False
    assert not info['independence_verified']


def test_evaluation_predictions_use_public_flat_download_contract():
    from ecodata.interface.downloads import package_results
    import io,zipfile
    values=io.BytesIO();np.savez_compressed(values,prediction=np.array([1.,2.]))
    name='pair-1-evaluation-a-evaluation.npz'
    raw=package_results([(name,values.getvalue()),('evaluation.json',b'{"scope":"synthetic download test"}')],[],[])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.testzip() is None
        assert np.array_equal(np.load(io.BytesIO(z.read(name)))['prediction'],[1.,2.])


def test_actual_result_cards_explain_proxy_and_ablations(tmp_path):
    from ecodata.desktop.output_review import cards,describe_output
    from ecodata.desktop.service import fusion_evidence
    task,_,_=task_and_inputs();engine=SpatialSearch(task,tmp_path,BUDGET,11)
    for key in engine.leaves[:2]:engine.execute(key)
    best=engine.best();comparison=engine.comparison_report(best)
    evidence=fusion_evidence(dict(selection_comparison=comparison,evaluation_protocol=engine.evaluation_plan))
    ranking=[dict(rank=i+1,key=r['key'],selected=r is best,algorithm=engine.records[r['key']]['operation'],family=r['source_family'],metrics=r['metrics'],parameters={}) for i,r in enumerate(engine.rows)]
    result=dict(mode='independent_grid_search',source_formats=['NetCDF'],target_format='NetCDF',files=[],local=dict(artifacts=[]),
        grid_searches=dict(test=dict(evidence=evidence,rankings=ranking)))
    entries=cards(result,describe_output(result,None))
    selected=next(c for c in entries if c['selected'])
    assert selected['proxy']
    text=' '.join(selected['audit'])
    assert '所有来源' in text and '不是独立实测' in text and '多源无额外加分' in text and '去掉' in text
