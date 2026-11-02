import io
import json
import zipfile
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.interface import independent_cases as cases


def netcdf_payload(value=1., units='mol/s', origin=0., model_time=10000, clock=120000):
    from netCDF4 import Dataset
    d=Dataset('memory', 'w', memory=8192, format='NETCDF3_CLASSIC')
    d.createDimension('x',1)
    d.XORIG=origin
    d.CTIME=clock; d.WTIME=clock; d.SDATE=2018217
    v=d.createVariable('NO','f4',('x',)); v.units=units; v[:]=value
    d.createVariable('TFLAG','i4',('x',))[:]=model_time
    return bytes(d.close())


def test_netcdf_clock_only_change_preserves_scientific_equivalence():
    assert cases.scientific_equal('emissions.nc',netcdf_payload(clock=120000),netcdf_payload(clock=130000))


@pytest.mark.parametrize('change',[{'value':2.},{'units':'kg/s'},{'origin':1000.},{'model_time':20000}])
def test_netcdf_scientific_changes_are_not_treated_as_clock_metadata(change):
    assert not cases.scientific_equal('emissions.nc',netcdf_payload(),netcdf_payload(**change))


@pytest.mark.parametrize('model,module,function', [
    ('wrf','wrf_run_page','render_wrf_run_page'),
    ('calpuff','calmet_run_page','render_calmet_run_page'),
    ('camx','camx_run_page','render_camx_run_page'),
    ('cmaq','cmaq_run_page','render_cmaq_run_page'),
])
def test_case_and_route_reach_backend_without_old_case_fallback(model,module,function):
    page=AppTest.from_string(f'from ecodata.interface.{module} import {function}; {function}()').run()
    page.selectbox[0].set_value(cases.PROFILES[model]['label']).run()
    assert not page.exception
    page.radio[0].set_value(page.radio[0].options[1]).run()
    with patch.object(cases,'start',side_effect=RuntimeError('explicit new-case launch failure')) as start:
        page.button(key=model+'_independent_start').click().run()
    start.assert_called_once_with(model,'rebuild',None,None)
    assert not page.exception and not page.success
    assert 'explicit new-case launch failure' in page.error[0].value


@pytest.mark.parametrize('model',list(cases.PROFILES))
def test_unverified_category_cannot_launch_or_fallback(model):
    with patch('ecodata.round1.convert') as convert:
        with pytest.raises(ValueError,match='分类'):
            cases.start(model,'rebuild',mode='unverified-category')
        convert.assert_not_called()


@pytest.mark.parametrize('name',['../a.nc','nested/a.nc','C:a.nc','a\\b.nc'])
def test_uploaded_output_paths_rejected(name):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr(name,b'not executed')
    with pytest.raises(ValueError,match='平铺'):
        cases.members(out.getvalue())


def test_invalid_job_id_cannot_read_outside_runs():
    with pytest.raises(ValueError):cases.status('../status')


def test_conversion_mismatch_rejected_before_native_thread(tmp_path,monkeypatch):
    monkeypatch.setattr(cases,'RUNS',tmp_path/'runs')
    source=tmp_path/'source';inputs=source/'platform_inputs/WRF_Colorado';inputs.mkdir(parents=True)
    (inputs/'config.json').write_text('{}');(inputs/'original_pressure_fields.nc').write_bytes(b'raw')
    monkeypatch.setattr(cases,'SOURCE',source)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:z.writestr('field.wps',b'different scientific input')
    with patch('ecodata.round1.convert',return_value=({'field.wps':b'expected scientific input'},{})),patch.object(cases.threading,'Thread') as thread:
        with pytest.raises(ValueError,match='不匹配'):
            cases.start('wrf','existing',buffer.getvalue())
        thread.assert_not_called()
    assert cases.ACTIVE is None


def test_transient_windows_status_sharing_violation_is_retried(tmp_path,monkeypatch):
    from pathlib import Path
    original=Path.replace
    calls=[]
    def replace(path,target):
        calls.append(path)
        if len(calls)==1:
            raise PermissionError('Windows sharing violation')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',replace)
    cases.save(tmp_path,{'state':'failed','error':'preserved native failure'})
    assert json.loads((tmp_path/'status.json').read_text())=={'state':'failed','error':'preserved native failure'}
    assert len(calls)==2


def test_launch_failure_does_not_display_previous_success():
    page=AppTest.from_string('from ecodata.interface.camx_run_page import render_camx_run_page; render_camx_run_page()').run()
    page.selectbox[0].set_value(cases.PROFILES['camx']['label']).run()
    page.session_state['camx_independent_job']='a'*32
    with patch.object(cases,'start',side_effect=ValueError('current payload rejected')),patch.object(cases,'status') as read_previous:
        page.button(key='camx_independent_start').click().run()
    read_previous.assert_not_called()
    assert not page.exception and not page.success
    assert 'current payload rejected' in page.error[0].value


