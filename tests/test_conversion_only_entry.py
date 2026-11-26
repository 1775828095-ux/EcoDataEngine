"""Only-convert UI/API behavior; synthetic weather is not environmental validation."""
import zipfile
import pytest
from test_desktop_simple_flow import window
from test_minimum_task_inputs import rain,upload
from ecodata.desktop import service
from ecodata.desktop.scan import Scanner


def forbid_fusion(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Only conversion must not recommend, fuse or search')
    monkeypatch.setattr('ecodata.algorithms.platform_search.submit_search',forbidden)
    monkeypatch.setattr('ecodata.algorithms.automatic_preparation.prepare_uploads',forbidden)
    monkeypatch.setattr(service,'send',forbidden)


def test_only_conversion_public_service_writes_real_native_component_without_fusion(tmp_path,monkeypatch):
    forbid_fusion(monkeypatch)
    source=tmp_path/'input';source.mkdir();p=source/'rain.nc';p.write_bytes(rain().to_netcdf());before=p.read_bytes()
    scan,_=Scanner().scan(source,shallow=True)
    result=service.run_original_workbench(scan,'swat_plus',tmp_path/'out',declared_only=True)
    assert result['manifest']['status']=='COMPONENT_EXPORTED'
    gate=result['manifest']['final_check']
    assert gate['format']['status']=='PASS' and gate['project']['status']=='NOT_REQUESTED'
    assert result['grid_searches']=={} and not result['manifest']['native_model_executed']
    with zipfile.ZipFile(result['original_archive']) as z:
        assert z.testzip() is None
        assert '2010 1 1.0' in z.read('components/station.pcp').decode()
    assert p.read_bytes()==before


def test_page_mode_needs_no_api_and_keeps_explicit_confirmation(tmp_path,window,monkeypatch):
    forbid_fusion(monkeypatch)
    monkeypatch.setenv('DEEPSEEK_API_KEY','')
    folder=tmp_path/'input';folder.mkdir();(folder/'rain.nc').write_bytes(rain().to_netcdf())
    monkeypatch.setattr(window,'error',lambda e:pytest.fail(str(e)))
    monkeypatch.setattr(window,'work',lambda fn,then,**kw:then(fn(window.show_progress)))
    window.conversion_only.setChecked(True);window.set_decision_model('swat_plus');window.path.setText(str(folder))
    window.start_button.click()
    assert window.tabs.currentIndex()==1 and window.result is None
    assert window.search_button.isEnabled() and not window.recommendation_rows
    window.search_button.click()
    assert window.result['manifest']['status']=='COMPONENT_EXPORTED'
    assert window.result['manifest']['final_check']['format']['status']=='PASS'
    assert window.export_button.isEnabled()


def test_fusion_and_conversion_modes_cannot_be_silently_combined(window):
    window.set_decision_model('data_only');window.conversion_only.setChecked(True)
    assert window.decision_model is None
    window.set_decision_model('swat_plus');window.set_decision_model('data_only')
    assert not window.conversion_only.isChecked()
    assert not window.target_grid_row.isHidden()


def test_missing_critical_metadata_still_blocks_conversion(tmp_path):
    source=tmp_path/'input';source.mkdir();ds=rain();ds.attrs.pop('time_basis');(source/'rain.nc').write_bytes(ds.to_netcdf())
    scan,_=Scanner().scan(source,shallow=True)
    with pytest.raises(ValueError,match='时间基准'):
        service.run_original_workbench(scan,'swat_plus',tmp_path/'out',declared_only=True)


def test_only_conversion_cannot_use_data_only_as_target(tmp_path):
    source=tmp_path/'input';source.mkdir();(source/'rain.nc').write_bytes(rain().to_netcdf())
    scan,_=Scanner().scan(source,shallow=True)
    with pytest.raises(ValueError,match='明确目标模型'):
        service.run_original_workbench(scan,'data_only',tmp_path/'out',declared_only=True)
