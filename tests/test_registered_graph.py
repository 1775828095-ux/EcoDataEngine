"""Functional graphs; synthetic values are not environmental accuracy evidence."""
from dataclasses import replace
import json
import numpy as np
import pytest

from ecodata.algorithms import platform_search as api
from ecodata.algorithms.registered_graph import discover, validate, execute_recommendation
from test_field_mutation import engine


def graph():
    return dict(nodes=[
        dict(id='clip',algorithm='constraint.series_nonnegative',parents=['source:0']),
        dict(id='fill',algorithm='fusion.series_missing_fallback',parents=['clip','source:1'])],
        output='fill')


def test_explicit_multibranch_graph_executes_real_parent_products(tmp_path):
    search=engine(tmp_path)
    try:
        search.states['s0'].precipitation.values[0]=-2
        search.context.sources[0].dataset.precipitation.values[0]=-2
        key=execute_recommendation(search,dict(graph=graph()))
        output=search.states[key].precipitation.values
        assert output[0]==0 and output[1]==2
        record=search.records[key]
        assert search.records[record['parents'][0]]['operation']=='constraint.series_nonnegative'
        assert record['parents'][1]=='s1'
        assert search.evaluated[key]['trace']
        assert search.registry.get('fusion.primary_weighted_mean').spec in search.available
    finally:search.close()


@pytest.mark.parametrize('change',[
    lambda g:g['nodes'][0].update(parents=['fill']),
    lambda g:g['nodes'][0].update(parameters={'fitted_state':'forged'}),
    lambda g:g.update(output='clip'),
    lambda g:g['nodes'][0].update(algorithm='invented'),
])
def test_invalid_graph_cannot_bypass_contracts(tmp_path,change):
    search=engine(tmp_path)
    try:
        proposed=graph();change(proposed)
        with pytest.raises((ValueError,KeyError)):validate(proposed,search.registry,2)
        assert search.executor.calls==0
    finally:search.close()


def test_discovery_inspects_every_algorithm_without_labels_or_execution(tmp_path):
    search=engine(tmp_path)
    try:
        original=discover(search.registry,search.context,max_bindings=1)
        altered=replace(search.context,validation_reference=search.reference*100)
        assert discover(search.registry,altered,max_bindings=1)==original
        assert {r['id'] for r in original['algorithms']}=={s.id for s in search.registry.specs()}
        assert search.executor.calls==0
        assert json.loads((tmp_path/'registry_discovery.json').read_text())['complete_registry_scan']
    finally:search.close()


def test_public_request_carries_explicit_graph_without_fixed_recipe():
    req=dict(variable='precipitation')
    row=dict(id='generated',task='task',graph=graph(),scope=dict(variable='precipitation'))
    result=api.with_recommendations(req,[row])
    assert result['recommended_graphs'][0]['graph']==graph()
    assert 'recommended_recipes' not in result


def test_no_processing_needed_is_a_valid_proposal(tmp_path):
    search=engine(tmp_path)
    try:
        search.execute('s1','source_baseline')
        assert execute_recommendation(search,dict(graph=dict(nodes=[],output='source:1')))=='s1'
    finally:search.close()


def test_spatial_registry_kernel_uses_only_actual_observation_parent(tmp_path):
    from test_recommendation_search import engine as spatial_engine
    search=spatial_engine(tmp_path,['spatial.blend'])
    operation='station.pykrige_ordinary'
    assert operation in search.registry_operations
    key=next(k for k in search.leaves if search.records[k]['operation']==operation)
    assert search.records[search.records[key]['parents'][0]]['operation']=='spatial.observations'
    executor=search.executors[0]
    baseline=executor.run(search.nodes(key),key)['outputs']['field'].value.copy()
    cohort=dict(search.folds[0]['cohort']);cohort['y']=cohort['y']+3
    changed=search.executor(cohort,'functional_test').run(search.nodes(key),key)['outputs']['field'].value
    np.testing.assert_allclose(changed-baseline,3,atol=1e-5)
    assert len(search.registry.specs())>len(search.registry_operations)


def test_spatial_explicit_graph_can_use_previously_unbound_registered_algorithm(tmp_path):
    from test_recommendation_search import engine as spatial_engine
    search=spatial_engine(tmp_path,['spatial.blend'])
    search.request['recommended_recipes']=[]
    search.request['recommended_graphs']=[dict(scope=dict(variable='temperature'),graph=dict(nodes=[
        dict(id='estimate',algorithm='station.pykrige_ordinary',parents=['observations']),
        dict(id='combine',algorithm='spatial.blend',parents=['estimate','source:0'],parameters=dict(weight=.5))],output='combine'))]
    search.validate_recommendations()
    receipt=search.recommendation_receipts[0]
    assert receipt['status']=='PASS',receipt
    sink=search.records[receipt['keys'][0]]
    assert search.records[sink['parents'][0]]['operation']=='station.pykrige_ordinary'
    assert search.rows[0]['traces']


def test_page_uses_public_recommendation_and_search_service(tmp_path,monkeypatch):
    from ecodata.interface.search_page import process_search_upload
    calls=[]
    provider=lambda payload:None
    monkeypatch.setattr(api,'recommend_and_submit',lambda *a,**kw:calls.append(kw) or dict(status='CHECK'))
    assert process_search_upload([],{},None,tmp_path,recommendation_provider=provider)['status']=='CHECK'
    assert calls[0]['provider'] is provider


def test_desktop_graph_is_search_admission_not_false_numerical_release():
    from ecodata.desktop.release_gate import selectable,allowed
    row=dict(graph=graph(),execution_release=dict(status='search_only'),
        search_admission=dict(protocol='registered-graph-search-v1',status='SEARCH_ONLY',route='station_grid'))
    assert selectable(row) and not allowed(row)


def test_last_legal_proposal_is_measured_without_exceeding_proposal_budget(tmp_path):
    search=engine(tmp_path)
    try:
        search.execute('s1','source_baseline')
        before=search.executor.calls
        def neighbourhood(sink,family,destruction):
            spec=search.registry.get('fusion.series_missing_fallback').spec
            row=search.candidate(sink,None,spec,['s0','s1'],dict(primary_index=0),'budget_boundary')
            search.attempts=search.budget['proposal_limit']
            yield row
        search.neighbourhood=neighbourhood
        child=search.propose('s1','structure','random',search.rng)
        assert child and search.executor.calls>before
        assert search.attempts==search.budget['proposal_limit']
        assert search.transitions[-1]['actual_execution']
    finally:search.close()


def test_every_spatial_registry_leaf_has_mutation_opportunity(tmp_path):
    from test_recommendation_search import engine as spatial_engine
    search=spatial_engine(tmp_path,['spatial.blend'])
    rng=np.random.default_rng(31)
    seen={search.records[search.candidate(search.leaves[0],'algorithm',rng)]['operation'] for _ in range(400)}
    assert set(search.registry_operations)<=seen


def test_target_geometry_and_missingness_reach_common_rule_consumer(tmp_path):
    from ecodata.algorithms.registered_graph import proposal_request
    search=engine(tmp_path)
    try:
        payload=proposal_request(search.registry,search.context,
            dict(task='temporal',target_times=[str(t) for t in search.states['s0'].time.values]))
        required=payload['requirements']['requirements']['required']
        assert 'temporal_missing_recovery' in required
        assert payload['target']['times']
        assert not payload['reference_used']
    finally:search.close()


def test_desktop_accepts_generated_graph_outside_legacy_recipes():
    from ecodata.desktop.recommendations import validate as admit,shared_algorithm_catalogue
    from ecodata.algorithms.spatial_fusion_task import graph_components
    registry=api.platform_registry()
    task=dict(id='C1:fusion',operation='fusion',allowed_recipes=[],context=dict(pair_details=[]),
        graph_protocol=dict(source_count=2,variable='temperature',components=graph_components(registry),resources=['observations']))
    proposed=dict(nodes=[dict(id='estimate',algorithm='station.pykrige_ordinary',parents=['observations']),
        dict(id='blend',algorithm='spatial.blend',parents=['estimate','source:0'])],output='blend')
    rows=admit(dict(recommendations=[dict(task=task['id'],recipe='generated_graph',graph=proposed,reason='contract proposal')]),
        dict(tasks=[task],shared_algorithm_catalogue=shared_algorithm_catalogue()))
    assert rows[0]['graph']['nodes'][0]['algorithm']=='station.pykrige_ordinary'
    assert rows[0]['scope']['variable']=='temperature' and not rows[0].get('advisory_only')


def test_resources_are_actual_edges_and_never_validation_labels(tmp_path):
    from ecodata.algorithms.autonomous_decision import Executor
    search=engine(tmp_path)
    try:
        aux=search.context.sources[1].dataset.copy(deep=True)
        context=replace(search.context,auxiliaries=(aux,),validation_reference=search.reference)
        executor=Executor(context,search.registry)
        key=search.intern('fusion.series_missing_fallback',['s0','s1'],dict(primary_index=0))
        _,trace=executor.execute(search.records,key)
        node=next(row for row in trace if row['operation']=='fusion.series_missing_fallback')
        assert any(edge['to'].endswith('.resource_auxiliaries') for edge in node['edges'])
        assert all('validation' not in row['operation'] for row in trace)
        assert 'validation_reference' not in executor.context_resources
    finally:search.close()
