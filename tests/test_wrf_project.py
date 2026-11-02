import json
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.interface import wrf_project as p,independent_cases as cases

def settings():
    return dict(start='2016-10-06_18:00:00',end='2016-10-07_00:00:00',interval_seconds=21600,duration_seconds=1800,latitude=27.,longitude=-78.,dx=12000,nx=49,ny=49,levels=48,top_pressure=5000,time_step=60,physics_suite='CONUS')

@pytest.mark.parametrize('changes',[{'latitude':0.},{'longitude':120.},{'physics_suite':'unreviewed'},{'duration_seconds':86400},{'interval_seconds':3700},{'time_step':180},{'nx':True},{'end':'2016-10-06_18:00:00'},{'extra':'script'}])
def test_out_of_scope_settings_fail_without_native(changes):
    with pytest.raises((ValueError,TypeError)):
        p.start([],settings()|changes)

def test_regions_and_grid_sizes_not_bound_to_registered_values():
    assert p.validate_settings(settings()|{'latitude':40.,'longitude':-100.,'nx':37,'ny':41,'dx':18000})['nx']==37

@pytest.mark.parametrize('raw',[b'#!/bin/sh'+b'x'*1200,b'GRIB'+bytes([0,0,0,3])+b'x'*1200,b'CDF'+b'x'*1200])
def test_non_grib_payload_rejected(raw):
    with patch('threading.Thread') as native:
        with pytest.raises(ValueError,match='GRIB'):p.start([('one',raw),('two',raw)],settings())
        native.assert_not_called()

def test_parallel_registered_job_prevents_new_native(monkeypatch):
    monkeypatch.setattr(cases,'ACTIVE','running-native')
    raw=b'GRIB'+bytes([0,0,0,2])+b'x'*1200
    with pytest.raises(ValueError,match='串行'):p.start([('one',raw),('two',raw)],settings())

def test_custom_ui_requires_actual_uploads_never_uses_demo():
    page=AppTest.from_string('from ecodata.interface.wrf_run_page import render_wrf_run_page; render_wrf_run_page()').run()
    page.selectbox[0].set_value('自定义原始GFS/FNL新建工程（试验）').run()
    with patch.object(p,'start') as launch:
        page.button(key='wrf_project_start').click().run()
        launch.assert_not_called()
    assert not page.exception and '不会回退' in page.error[0].value

def test_incompatible_conversion_mode_rejected_on_page():
    page=AppTest.from_string("from ecodata.interface.wrf_project import render; render(mode='sst')").run()
    assert not page.exception and not page.button
    assert '不能作为完整三维' in page.error[0].value

def test_stream_preserves_original_values_and_rejects_truncation(tmp_path):
    from test_wrf_assembly import field
    from ecodata.interface.wrf_source_crop import crop,stream_fields
    raw=field('TT',85000,[1,2,3,4])+field('RH',85000,[10,20,30,40])
    a,b=tmp_path/'source',tmp_path/'subset';a.write_bytes(raw)
    assert crop(a,b,0.,-130.)==2 and b.read_bytes()==raw
    a.write_bytes(raw[:-2])
    with pytest.raises(ValueError,match='Truncated'):list(stream_fields(a))
