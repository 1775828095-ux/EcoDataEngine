from unittest.mock import patch
import json
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.interface import wrf_project as project


def test_existing_custom_requires_new_conversion():
    with patch.object(project.threading,'Thread') as launch:
        with pytest.raises(ValueError,match='本次平台转换'):project.start_existing('a'*32,None)
    launch.assert_not_called()


def test_failed_project_not_silently_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(project,'RUNS',tmp_path)
    p=tmp_path/('a'*32);p.mkdir();(p/'status.json').write_text(json.dumps({'state':'failed'}))
    with patch.object(project.threading,'Thread') as launch:
        with pytest.raises(ValueError,match='不回退'):project.start_existing('a'*32,b'not-used')
    launch.assert_not_called()


def test_page_existing_route_requires_both_inputs():
    page=AppTest.from_string('from ecodata.interface.wrf_project import render; render()').run()
    page.radio(key='wrf_custom_route').set_value('接入已成功的本机自定义工程').run()
    with patch.object(project,'start_existing') as launch:
        page.button(key='wrf_custom_existing_start').click().run()
    launch.assert_not_called()
    assert not page.exception and '不会回退' in page.error[0].value
