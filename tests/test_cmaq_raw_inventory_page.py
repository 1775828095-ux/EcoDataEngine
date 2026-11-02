from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.interface import cmaq_smoke_inventory as smoke
from ecodata.interface import independent_cases as cases


def test_unreviewed_nei_stops_before_execution(tmp_path):
    with patch.object(smoke.subprocess,'run') as run:
        with pytest.raises(ValueError,match='不将其他清单替换'):
            smoke.build(tmp_path/'new',b'different inventory',tmp_path)
    run.assert_not_called()
    assert not (tmp_path/'new').exists()


def test_nei_selection_requires_upload():
    page=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page; render_cmaq_run_page()').run()
    page.selectbox[0].set_value(cases.PROFILES['cmaq']['label']).run()
    page.radio[0].set_value(page.radio[0].options[1]).run()
    page.checkbox(key='cmaq_raw_oilgas_enabled').check().run()
    with patch.object(cases,'start') as start:
        page.button(key='cmaq_independent_start').click().run()
    start.assert_not_called()
    assert not page.exception and '不会回退原点源部门' in page.error[0].value


def test_non_egu_wrong_inventory_stops_before_native_execution(tmp_path):
    with patch.object(smoke.subprocess,'run') as run:
        with pytest.raises(ValueError,match='不将其他清单替换'):
            smoke.build(tmp_path/'new',b'wrong sector',tmp_path,sector='non_egu')
    run.assert_not_called()
    assert not (tmp_path/'new').exists()


def test_non_egu_selection_requires_matching_upload():
    page=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page; render_cmaq_run_page()').run()
    page.selectbox[0].set_value(cases.PROFILES['cmaq']['label']).run()
    page.radio[0].set_value(page.radio[0].options[1]).run()
    page.checkbox(key='cmaq_raw_non_egu_enabled').check().run()
    with patch.object(cases,'start') as start:
        page.button(key='cmaq_independent_start').click().run()
    start.assert_not_called()
    assert not page.exception and '不会回退原部门' in page.error[0].value
    assert any('33条' in warning.value for warning in page.warning)


def test_non_egu_cannot_be_applied_to_other_routes():
    with pytest.raises(ValueError,match='仅用于CMAQ新建'):
        cases.start('cmaq','existing',raw_non_egu_inventory=b'any')
    with pytest.raises(ValueError,match='仅用于CMAQ新建'):
        cases.start('wrf','rebuild',raw_non_egu_inventory=b'any')


def test_ship_hourly_mismatch_is_rejected_before_native(tmp_path):
    with patch.object(smoke.subprocess,'run') as run:
        with pytest.raises(ValueError,match='不补缺小时'):
            smoke.validate_hourly(b'unmatched hours')
    run.assert_not_called()


def test_ship_selection_requires_both_files():
    page=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page; render_cmaq_run_page()').run()
    page.selectbox[0].set_value(cases.PROFILES['cmaq']['label']).run()
    page.radio[0].set_value(page.radio[0].options[1]).run()
    page.checkbox(key='cmaq_raw_cmv_enabled').check().run()
    with patch.object(cases,'start') as start:
        page.button(key='cmaq_independent_start').click().run()
    start.assert_not_called()
    assert not page.exception and '同时上传' in page.error[0].value
    assert any('225个源' in w.value and '不是实测' in w.value for w in page.warning)


def test_ship_inputs_cannot_be_sent_to_existing_route():
    with pytest.raises(ValueError,match='仅用于CMAQ新建'):
        cases.start('cmaq','existing',raw_cmv_hourly=b'unmatched')
