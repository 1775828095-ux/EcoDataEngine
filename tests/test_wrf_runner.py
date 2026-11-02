import io
import zipfile
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.core.exceptions import SchemaError
from ecodata.interface import wrf_runner


def package(names):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as archive:
        for name in names:archive.writestr(name,b'data')
    return out.getvalue()


@pytest.mark.parametrize('name',['../field.wps','folder/field.wps','C:field.wps','folder\\field.wps'])
def test_zip_rejects_paths(name):
    with pytest.raises(SchemaError,match='平铺'):
        wrf_runner.read_update_zip(package([name]))


def test_zip_rejects_case_duplicate_and_no_fields():
    with pytest.raises(SchemaError,match='重复'):
        wrf_runner.read_update_zip(package(['A.wps','a.wps']))
    with pytest.raises(SchemaError,match='没有WPS'):
        wrf_runner.read_update_zip(package(['README.txt']))


def test_page_routes_and_upload_requirement():
    page=AppTest.from_string('from ecodata.interface.wrf_run_page import render_wrf_run_page; render_wrf_run_page()').run()
    assert not page.exception
    assert not page.button[0].disabled
    page.radio[0].set_value('从原始资料重新搭建').run()
    assert any('geogrid和ungrib' in element.value for element in page.markdown)
    page.radio[1].set_value('上传平台转换结果ZIP').run()
    assert page.button[0].disabled


def test_invalid_route_and_job_id():
    with pytest.raises(SchemaError):wrf_runner.start('bad')
    with pytest.raises(SchemaError):wrf_runner.status('../bad')
