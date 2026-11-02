"""Render real engineering pages and switch both routes without starting models."""
import pytest
from streamlit.testing.v1 import AppTest

@pytest.mark.parametrize('module,kwargs',[
    ('swat',"converted_inputs=b'current-zip',input_mode='soil'"),
    ('wrf',"converted_zip=b'current-zip',input_mode='sst'"),
    ('calmet',"converted_zip=b'current-zip',mode='receptors'"),
    ('camx',"converted_zip=b'current-zip',mode='initial_concentrations'"),
    ('cmaq',"converted_zip=b'current-zip',mode='boundary_concentrations'")])
def test_both_real_routes_render(module,kwargs):
    app=AppTest.from_string(f'from ecodata.interface.{module}_run_page import render_{module}_run_page\nrender_{module}_run_page({kwargs})').run(timeout=30)
    assert not app.exception
    assert len(app.radio[0].options)==2
    assert app.get('graphviz_chart')
    app.radio[0].set_value(app.radio[0].options[1]).run(timeout=30)
    assert not app.exception
    assert app.get('graphviz_chart')
    if module=='cmaq':
        app.checkbox[0].check().run(timeout=30)
        assert not app.exception
        assert len(app.get('graphviz_chart'))==2
