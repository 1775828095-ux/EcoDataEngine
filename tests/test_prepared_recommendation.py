"""Functional transport/contract checks; synthetic fixtures are not accuracy evidence."""
import json
import pytest
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.registered_graph import recommendation_session,has_recommender


def test_old_public_spatial_entry_consumes_prepared_graph_and_hides_held_labels(tmp_path):
    from test_spatial_relationship_fusion import fixture,BUDGET
    items,request,_=fixture()
    payloads=[]
    def provider(payload):
        payloads.append(payload)
        assert payload['resources']['observations_summary']['positions']<5
        assert 'observations' in payload['resources']['named_ports']
        return dict(graph=dict(nodes=[dict(id='estimate',algorithm='station.pykrige_ordinary',parents=['observations']),
            dict(id='blend',algorithm='spatial.blend',parents=['estimate','source:0'])],output='blend'),reason='Functional edge test')
    with recommendation_session(provider):
        result=api.process_current_input(items,dict(request,initial_algorithms=['spatial.idw']),tmp_path/'run',budget=BUDGET)
    assert not has_recommender()
    assert len(payloads)==1
    receipt=json.loads((tmp_path/'run/prepared_recommendation.json').read_text())
    assert receipt['status']=='SEARCH_PROPOSAL'
    assert result['recommendation_validation'][0]['status']=='PASS'
    assert not json.loads((tmp_path/'run/request.json').read_text())['request']['initial_algorithms']


@pytest.mark.parametrize('route',['windowed_grid_recipe','pair_processing','observation_gap_search',
    'original_engineering','native_reference_search','station_grid','indexed_table_recipe','independent_grid_search','local'])
def test_legacy_plan_is_search_admission_not_fixed_recipe_release(route,monkeypatch):
    from ecodata.desktop import recommendations,release_gate,catalog_decision
    task=dict(id='C1:fusion',candidate='C1',operation='fusion',route=route,prepared_graph_service=True,
        allowed_recipes=[dict(id='R1',name='legacy preparation',steps=['units.harmonize','spatial.blend'])],
        context=dict(fields=['D1.F1'],pair_details=[dict(physical_family='temperature')]))
    plan=dict(tasks=[task],target_model=dict(id='data_only'))
    rows=recommendations.validate(dict(recommendations=[dict(task='C1:fusion',recipe='R1')]),plan)
    monkeypatch.setattr(recommendations,'assert_current',lambda scan:None)
    monkeypatch.setattr(catalog_decision,'local_fields',lambda scan,approved:({'D1.F1':(0,'D1.F1',dict(name='temperature'))},None))
    recommendations.review_holdouts({}, {}, plan,rows,{('C1','fusion'):dict(route=route)})
    assert release_gate.selectable(rows[0]) and not release_gate.allowed(rows[0])
    assert rows[0]['autonomous_recommendation']


def test_failed_online_proposal_continues_lawful_search_without_relaxing_quality(tmp_path):
    from test_spatial_relationship_fusion import fixture,BUDGET
    inputs,request,_=fixture()
    def failing(payload):raise TimeoutError('Functional transport failure')
    with recommendation_session(failing):
        result=api.process_current_input(inputs,request,tmp_path/'run',budget=BUDGET)
    receipt=json.loads((tmp_path/'run/prepared_recommendation.json').read_text())
    assert receipt['status']=='RECOMMENDATION_FAILED_OPEN_SEARCH_CONTINUES'
    assert result['product'] and not result['recommendation_validation']


def test_selected_workflow_owns_request_local_provider(monkeypatch,tmp_path):
    from ecodata.desktop import selected_processing,service
    provider=lambda payload:None
    monkeypatch.setattr(service,'prepared_graph_provider',lambda *args:provider)
    seen=[]
    monkeypatch.setattr(selected_processing,'_run',lambda *a,**kw:seen.append(has_recommender()) or {})
    selected_processing.run({},[('C1:fusion',dict(autonomous_recommendation=True))],tmp_path)
    assert seen==[True] and not has_recommender()


def test_original_workbench_owns_same_request_local_provider(monkeypatch,tmp_path):
    from ecodata.desktop import service
    monkeypatch.setattr(service,'prepared_graph_provider',lambda *args:lambda payload:None)
    seen=[]
    monkeypatch.setattr(service,'_run_original_workbench',lambda *a,**kw:seen.append(has_recommender()) or {})
    service.run_original_workbench({},'swat_plus',tmp_path,recommendations=[dict(autonomous_recommendation=True)])
    assert seen==[True] and not has_recommender()


def test_registry_session_does_not_escape_after_exception():
    with pytest.raises(ValueError):
        with recommendation_session(lambda payload:None):
            assert has_recommender();raise ValueError('stop')
    assert not has_recommender()

def test_static_entry_uses_same_graph_service_without_fabricated_time(tmp_path):
    import numpy as np
    import xarray as xr
    field=xr.Dataset({'temperature':(('lat','lon'),280+np.arange(64).reshape(8,8)/10,
        dict(units='K',physical_variable='temperature',physical_definition='synthetic boundary fixture',data_kind='continuous',spatial_semantics='point'))},
        coords={'lat':np.arange(8.)+30,'lon':np.arange(8.)+110},attrs={'crs':'EPSG:4326'})
    seen=[]
    def provider(payload):
        seen.append(payload)
        return dict(graph=dict(nodes=[],output='source:0'),reason='No extra processing assumed')
    request=dict(task='multisource',variable='temperature',physical_variable='temperature',unit='K',semantic_type='continuous',temporal_semantics='UNKNOWN')
    with recommendation_session(provider):
        result=api.submit_search([api.SearchInput('fixture.nc','primary',bytes(field.to_netcdf(engine='h5netcdf')))],request,None,tmp_path/'static',
            budget=dict(objective_limit=8,total_call_limit=12,proposal_limit=8,wall_seconds=8),seed=11)
    assert seen and result.get('product'),result
    assert result['static_protocol']['synthetic_timestamp'] is False
    assert result['delivery_status']=='BASELINE_RETAINED'
    with xr.open_dataset(result['product']) as product:
        assert 'time' not in product.dims
        xr.testing.assert_allclose(field,product.load())
    receipt=json.loads((tmp_path/'static/search/prepared_recommendation.json').read_text())
    assert receipt['status']=='SEARCH_PROPOSAL'

def test_old_static_windowed_entry_reaches_shared_numerical_search(tmp_path):
    import numpy as np
    import xarray as xr
    from ecodata.desktop import windowed_grid_recipe
    from test_desktop_joint_recipe_sampling import row,task_for
    paths=[];fields=[]
    for index in range(2):
        path=tmp_path/f'{index}.nc'
        attrs=dict(units='degC',physical_definition='synthetic comparable point temperature',spatial_semantics='point')
        ds=xr.Dataset({'temperature':(('lat','lon'),10+np.arange(64).reshape(8,8)/10+index,attrs)},
            coords={'lat':30+np.arange(8.)*.1,'lon':110+np.arange(8.)*.1},attrs={'crs':'EPSG:4326'})
        ds.to_netcdf(path,engine='h5netcdf');paths.append(path)
        fields.append(dict(name='temperature',unit='degC',**attrs))
    scan=dict(rows=[row(path,'NetCDF',field) for path,field in zip(paths,fields)])
    task,_=task_for('windowed_grid_recipe',fields,'raster_task.spatial.bilinear')
    with recommendation_session(lambda payload:dict(graph=dict(nodes=[],output='source:0'))):
        result=windowed_grid_recipe.run(scan,'C1:fusion',task,tmp_path/'run','data_only',None)
    assert result['files'] and result['grid_searches']['C1:fusion']['selected_graph']
    protocol=json.loads((tmp_path/'run/spatial-search/static_protocol.json').read_text())
    assert not protocol['synthetic_timestamp']
    receipt=json.loads((tmp_path/'run/spatial-search/search/prepared_recommendation.json').read_text())
    assert receipt['status']=='SEARCH_PROPOSAL'
