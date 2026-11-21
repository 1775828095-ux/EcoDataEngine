"""Minimum component inputs and shared recommendation binding; no invented science."""
import io,json,zipfile
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.models.component_service import prepare_components
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.spatial_fusion_task import SpatialTask,SpatialSearch
from test_spatial_relationship_fusion import fixture,BUDGET
from test_desktop_simple_flow import window


def rain():
    ds=xr.Dataset(dict(precipitation=('time',[1.,2.,3.])),coords=dict(time=pd.date_range('2010-01-01',periods=3),lat=30.,lon=110.,elevation=40.),attrs=dict(crs='EPSG:4326',time_basis='UTC'))
    ds.precipitation.attrs.update(units='mm',temporal_semantics='accumulation',interval_seconds=86400)
    return ds


def upload(ds):
    b=io.BytesIO(ds.to_netcdf());b.name='rain.nc';return b


def test_only_precipitation_without_engineering_or_temperature_exports_native_component(tmp_path):
    result=prepare_components([upload(rain())],'swat_plus',tmp_path)
    assert result['manifest']['status']=='COMPONENT_EXPORTED'
    assert not result['manifest']['complete_model_input']
    with zipfile.ZipFile(result['archive']) as z:
        assert 'components/station.pcp' in z.namelist()
        assert not any(n.endswith('file.cio') for n in z.namelist())
        assert '2010 1 1.0' in z.read('components/station.pcp').decode()
    task=result['manifest']['preparation']['tasks'][0]
    assert task['validation']['auto_completion']
    assert task['configuration']['station']['elevation']==40.


def test_missing_labels_complete_but_scientific_metadata_never_guessed(tmp_path):
    ds=rain();ds.attrs.pop('time_basis')
    with pytest.raises(ValueError,match='时间基准'):
        prepare_components([upload(ds)],'swat_plus',tmp_path,task=dict(type='model_component',operation='precipitation',input='rain.nc',configuration={}))


def test_declared_coordinate_conflict_is_not_silently_overwritten(tmp_path):
    with pytest.raises(ValueError,match='conflicts'):
        prepare_components([upload(rain())],'swat_plus',tmp_path,task=dict(type='model_component',operation='precipitation',input='rain.nc',configuration=dict(station=dict(latitude=10.))))


def test_bound_workflow_and_concrete_recommendations_are_scored_then_mutable(tmp_path):
    items,q,_=fixture();q=dict(q,recommended_recipes=[dict(id='r',steps=['engineering.port_binding','spatial.residual_idw'])],completed_preparation=[dict(operation='engineering.port_binding',status='PASS',resource='actual native port')])
    parsed,audit=api.parse_inputs(items);search=SpatialSearch(SpatialTask.build(parsed,q,audit),tmp_path,BUDGET,11)
    search.validate_recommendations();receipt=search.recommendation_receipts[0]
    assert receipt['status']=='PASS' and receipt['keys']
    key=receipt['keys'][0]
    assert search.recommendation_parent()==key
    assert any(n.operation=='spatial.observations' for n in search.nodes(key))
    assert search.rows[0]['phase']=='recommended_initial' and search.rows[0]['metrics']['count']>0
    q.pop('completed_preparation');bad=SpatialSearch(SpatialTask.build(parsed,q,audit),tmp_path,BUDGET,11)
    bad.validate_recommendations();assert bad.recommendation_receipts[0]['status']=='REJECTED' and not bad.rows


def test_data_only_request_does_not_resolve_a_model_profile():
    from ecodata.desktop.local_search import model_request
    q=dict(variable='precipitation',unit='mm')
    assert model_request(q,'data_only')==q


def test_data_only_desktop_metadata_entry_needs_no_engineering(tmp_path):
    from ecodata.desktop.scan import Scanner
    from ecodata.desktop import discovery
    (tmp_path/'a.nc').write_bytes(rain().to_netcdf())
    ds=rain();ds.precipitation.values[:]+=1
    (tmp_path/'b.nc').write_bytes(ds.to_netcdf())
    scan,_=Scanner().scan(tmp_path)
    summary,_=discovery.discover(scan,'data_only',task_goal='evaluate_gain')
    assert summary['target_model']['id']=='data_only'
    assert not scan.get('native_model_probes',{}).get('data_only')


def test_minimum_header_completion_is_bounded_and_does_not_infer_science(tmp_path):
    from ecodata.desktop.file_agent import FileTools,complete_pair_headers
    for n in ('a.nc','b.nc','c.nc'):(tmp_path/n).write_bytes(rain().to_netcdf())
    tools=FileTools(tmp_path);tools.call('list_directory',dict(path=''))
    records=[];calls=complete_pair_headers(tools,0,records,lambda _:None)
    assert calls==2 and len(records)==2
    assert {e['tool'] for e in tools.events[1:]}=={'read_header'}
    assert 'c.nc' not in tools.rows
    assert complete_pair_headers(tools,calls,records,lambda _:None)==calls


def test_shared_online_recommendation_uses_checked_catalog_and_marks_fallback(monkeypatch):
    from ecodata.desktop import recommendations,service
    plan=dict(stage='recommendation',target_model={'id':'swat_plus','name':'SWAT+'},tasks=[dict(
        id='C1:fusion',candidate='C1',operation='fusion',context=dict(pair_details=[dict(physical_family='precipitation')]),
        allowed_recipes=[dict(id='R1',steps=['spatial.blend'],name='blend',description='actual catalog')])])
    calls=[]
    def send(approved,digest,mock,**kwargs):
        calls.append(json.loads(approved.summary))
        return dict(recommendations=recommendations.validate(dict(recommendations=[dict(task='C1:fusion',recipe='R1',reason='valid proposal')]),plan))
    monkeypatch.setattr(service,'send',send)
    previews=[]
    rows,receipt=recommendations.recommend_plan(plan,[],on_request=previews.append)
    assert calls[0]['stage']=='recommendation'
    assert previews[0].digest==receipt['request_sha256']
    assert rows[0]['recommendation_origin']=='deepseek' and receipt['status']=='COMPLETED'
    def failed(*args,**kwargs):raise OSError('bounded network failure')
    monkeypatch.setattr(service,'send',failed)
    rows,receipt=recommendations.recommend_plan(plan,[])
    assert receipt['status']=='PLATFORM_FALLBACK'
    assert rows[0]['recommendation_origin']=='platform_fallback'


def test_single_day_and_missing_optional_labels_still_convert(tmp_path):
    ds=rain().isel(time=slice(0,1))
    result=prepare_components([upload(ds)],'swat_plus',tmp_path)
    assert result['manifest']['preparation']['tasks'][0]['validation']['records']==1


def test_component_and_data_only_review_do_not_require_whole_model():
    from ecodata.desktop import output_review
    component=output_review.describe_output(dict(manifest=dict(status='COMPONENT_EXPORTED')),dict(id='swat_plus',inputs=[dict(name='soil',required=True)]))
    assert component['requirements']==[]
    assert output_review.review_request(component,[])['task_scope']=='component_conversion'
    plain=output_review.describe_output({},None)
    assert output_review.review_request(plain,[])['task_scope']=='data_only'
    assert '未请求模型工程' in output_review.availability_text(plain)


def test_desktop_component_confirmation_reaches_real_public_conversion(tmp_path,window,monkeypatch):
    from ecodata.desktop.scan import Scanner
    from ecodata.desktop import discovery
    folder=tmp_path/'input';folder.mkdir();source=folder/'rain.nc';source.write_bytes(rain().to_netcdf())
    before=source.read_bytes();scan,_=Scanner().scan(folder)
    discovery.discover(scan,'swat_plus',task_goal='evaluate_gain')
    assert scan['native_model_probes']['swat_plus']['component_candidate']
    window.set_decision_model('swat_plus');window.scan=scan;window.candidates={}
    window.mock.setChecked(True)  # Only final DS review is simulated, no API in this test.
    monkeypatch.setattr(window,'error',lambda e:pytest.fail(str(e)))
    monkeypatch.setattr(window,'work',lambda fn,then,**kw:then(fn(window.show_progress)))
    window.continue_automatically(dict(failures=[]),[])
    assert window.tabs.currentIndex()==1 and window.result is None
    window.search_button.click()
    assert window.result['manifest']['status']=='COMPONENT_EXPORTED'
    assert window.result['manifest']['final_check']['format']['status']=='PASS'
    assert window.result['manifest']['final_check']['project']['status']=='NOT_REQUESTED'
    assert source.read_bytes()==before
