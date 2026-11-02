from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.interface import independent_cases as cases
from ecodata.interface import cmaq_listos_meteorology as met


@pytest.mark.parametrize('inputs',[[],[b'bad'],[b'x'*1024]*4])
def test_invalid_inputs_stop_before_native_call(tmp_path,inputs):
    with patch.object(met.subprocess,'run') as run:
        with pytest.raises(ValueError):
            met.build(tmp_path/'new',inputs,tmp_path)
    run.assert_not_called()
    assert not (tmp_path/'new').exists()


def test_selected_wrf_route_requires_upload_without_fallback():
    page=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page; render_cmaq_run_page()').run()
    page.selectbox[0].set_value(cases.PROFILES['cmaq']['label']).run()
    page.radio[0].set_value(page.radio[0].options[1]).run()
    page.checkbox(key='cmaq_uploaded_wrf_enabled').check().run()
    with patch.object(cases,'start') as start:
        page.button(key='cmaq_independent_start').click().run()
    start.assert_not_called()
    assert not page.exception and '不会回退配套气象' in page.error[0].value


@pytest.mark.parametrize('model,route',[('camx','rebuild'),('cmaq','existing')])
def test_wrf_cannot_change_another_route(model,route):
    with pytest.raises(ValueError,match='仅用于CMAQ'):
        cases.start(model,route,meteorology_files=[b'bad'])
