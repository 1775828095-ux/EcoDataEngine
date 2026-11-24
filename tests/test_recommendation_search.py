"""Advisory recipes affect real execution/parents, never replace measured fitness."""
import json
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.spatial_fusion_task import SpatialTask,SpatialSearch
from test_spatial_relationship_fusion import fixture,BUDGET


def recommendation(steps,variable='temperature'):
    return dict(task='pair:fusion',id='R1',steps=steps,scope=dict(variable=variable),reason='test proposal, not known winner')


def engine(tmp_path,steps,**updates):
    items,request,_=fixture()
    request=api.with_recommendations(dict(request,**updates),[recommendation(steps)])
    parsed,audit=api.parse_inputs(items)
    return SpatialSearch(SpatialTask.build(parsed,request,audit),tmp_path,BUDGET,11)


def test_real_recommendation_scored_before_mutation_and_consumed_as_parent(tmp_path):
    items,request,_=fixture()
    request=api.with_recommendations(request,[recommendation(['spatial.blend'])])
    result=api.process_current_input(items,request,tmp_path/'run',budget=BUDGET,seed=11)
    receipt=result['recommendation_validation'][0]
    assert receipt['status']=='PASS' and receipt['metrics']['count']>0
    rows=json.loads((tmp_path/'run/candidate_results.json').read_text())
    assert rows[0]['phase']=='recommended_initial' and rows[0]['traces']
    assert len({row['key'] for row in rows})==len(rows)
    transitions=json.loads((tmp_path/'run/transitions.json').read_text())
    first=transitions[0]
    assert first['parent']==receipt['keys'][0] and first['family']=='recommendation_parameters'
    assert first['actual_execution'] and first['child']!=first['parent']
    graphs=json.loads((tmp_path/'run/graphs.json').read_text())
    assert graphs[first['child']]['parents']==graphs[first['parent']]['parents']
    assert graphs[first['child']]['parameters']!=graphs[first['parent']]['parameters']
    assert result['best']['metrics']['objective']<=receipt['metrics']['objective']+1e-12


def test_recipe_order_is_real_parent_dependency(tmp_path):
    search=engine(tmp_path,['spatial.blend','spatial.residual_idw'])
    search.validate_recommendations()
    receipt=search.recommendation_receipts[0]
    assert receipt['status']=='PASS'
    output=receipt['keys'][0];row=search.records[output]
    assert row['operation']=='spatial.residual_idw'
    assert search.records[row['parents'][0]]['operation']=='spatial.blend'
    assert search.records[row['parents'][1]]['operation']=='spatial.observations'
    assert search.rows[0]['traces']


@pytest.mark.parametrize('steps,updates,reason',[
    (['spatial.blend','spatial.idw'],{},'does not consume'),
    (['spatial.affine'],dict(allow_training=False),'training permission'),
    (['spatial.regrid.conservative','spatial.blend'],{},'not actually been executed'),
    (['invented.algorithm'],{},'Unsupported recommended'),
])
def test_failed_contract_not_scored_or_promoted(tmp_path,steps,updates,reason):
    search=engine(tmp_path,steps,**updates);search.validate_recommendations()
    assert search.recommendation_receipts[0]['status']=='REJECTED'
    assert reason in search.recommendation_receipts[0]['error']
    assert not search.rows and search.recommendation_parent() is None


def test_scope_and_disabled_recommendations_do_not_change_default_request():
    request=dict(variable='temperature')
    assert api.with_recommendations(request,[])==request
    assert api.with_recommendations(request,[recommendation(['spatial.blend'],'precipitation')])==request
    with pytest.raises(ValueError,match='scope'):
        api.with_recommendations(request,[dict(steps=['spatial.blend'])])


def test_recommended_preparation_is_numerical_not_a_claim():
    from ecodata.models.engineering.preparation import prepare_recommended_sources
    sources=[]
    for i,axis in enumerate([np.array([.5,1.5]),np.array([.25,.75,1.25,1.75])]):
        ds=xr.Dataset(dict(value=(('lat','lon'),np.ones((len(axis),len(axis)))*(i+1),
            dict(units='mm',data_kind='continuous',spatial_semantics='cell_mean'))),
            coords=dict(lat=axis,lon=axis),attrs=dict(crs='EPSG:4326'))
        sources.append(dict(dataset=ds,field='value',source_id=str(i),sha256=str(i)))
    request=dict(recommended_recipes=[dict(steps=['spatial.regrid.conservative','spatial.blend'])])
    aligned=prepare_recommended_sources(sources,request)
    assert aligned[1]['dataset'].value.shape==(2,2)
    np.testing.assert_allclose(aligned[1]['dataset'].value,2.)
    assert aligned[1]['recommendation_preparation'][0]['status']=='PASS'
    assert sources[1]['dataset'].value.shape==(4,4)


def test_page_and_toolkit_forward_same_scoped_recipes(tmp_path,monkeypatch):
    import os,zipfile
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from ecodata.desktop.__main__ import Window
    from ecodata.desktop import service
    from ecodata.desktop.file_agent import scan_index
    from ecodata.algorithms import automatic_preparation
    source=tmp_path/'input.txt';source.write_text('metadata only integration fixture')
    scan,_=scan_index(source)
    rows=[recommendation(['spatial.blend'])];requests=[]
    def prepare(uploads,model,folder,**kwargs):
        request=kwargs['request_builder'](dict(variable='temperature',unit='K',semantic_type='continuous'))
        requests.append(request)
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/'result.zip'
        with zipfile.ZipFile(path,'w') as archive:archive.writestr('receipt.json','{}')
        return dict(archive=path,manifest=dict(status='integration_fixture'))
    monkeypatch.setattr(automatic_preparation,'prepare_uploads',prepare)
    service.run_original_workbench(scan,'swat_plus',tmp_path/'toolkit',recommendations=rows)
    app=QApplication.instance() or QApplication([])
    window=Window();window.scan=scan;window.decision_model='swat_plus';window.selected_recommendations=rows
    monkeypatch.setattr(window,'work',lambda fn,callback:fn(lambda _:None))
    window.execute_original()
    assert len(requests)==2 and requests[0]==requests[1]
    assert requests[0]['recommended_recipes'][0]['steps']==['spatial.blend']
    window.batch_timer.stop()
    if window.run_temp is not None:window.run_temp.cleanup()
    window.close();app.processEvents()
