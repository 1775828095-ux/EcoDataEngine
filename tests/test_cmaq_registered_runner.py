from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.interface import cmaq_runner as runner

@pytest.mark.parametrize('ident',['../foo','','x'*32,'a'*33])
def test_invalid_job_rejected(ident):
    with pytest.raises(ValueError):runner.directory(ident)

def test_engine_integrity_failure_does_not_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'RUNS',tmp_path);monkeypatch.setattr(runner,'sha',lambda p:'invalid')
    ident='b'*32;folder=runner.directory(ident);folder.mkdir();runner.save(folder,dict(status='running',route='existing'))
    assert runner.LOCK.acquire(blocking=False)
    with patch.object(runner.subprocess,'run') as native:
        runner.execute(ident,'existing');native.assert_not_called()
    assert runner.status(ident)['status']=='failed' and not runner.LOCK.locked()

def test_native_lease_rejects_overlapping_processes(tmp_path,monkeypatch):
    import subprocess,sys
    monkeypatch.setattr(runner,'RUNS',tmp_path)
    held=runner.acquire_lease()
    try:
        with pytest.raises(ValueError,match='另一进程'):runner.acquire_lease()
        code=f"import msvcrt; f=open({str(tmp_path/'native.lock')!r},'r+b'); msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)"
        assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode!=0
    finally:held.close()
    assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode==0
    available=runner.acquire_lease();available.close()

@pytest.mark.parametrize('state',['running','failed'])
def test_page_non_success_has_no_success(state):
    app=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page\nrender_cmaq_run_page()');app.query_params['cmaq_job']='a'*32
    with patch('ecodata.interface.cmaq_run_page.status',return_value=dict(status=state,route='rebuild',error='injected failure')):app.run()
    assert not app.exception and not app.success and not app.metric
    assert app.radio[0].value=='用配套资料重建登记工程'
    if state=='failed':assert 'injected failure' in app.error[0].value

def test_start_failure_does_not_continue_to_old_result():
    app=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page\nrender_cmaq_run_page()');app.run()
    with patch('ecodata.interface.cmaq_run_page.start',side_effect=RuntimeError('injected start failure')):app.button[0].click().run()
    assert not app.exception and not app.success and 'injected start failure' in app.error[0].value


def test_raw_meteorology_rejected_on_existing_route():
    with pytest.raises(ValueError,match='新建路线'):runner.start('existing',new_meteorology=True)


def test_raw_meteorology_page_flag_reaches_backend_without_fallback():
    app=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page\nrender_cmaq_run_page()').run()
    app.radio[0].set_value('用配套资料重建登记工程').run()
    app.checkbox[0].check().run()
    with patch('ecodata.interface.cmaq_run_page.start',side_effect=RuntimeError('stop before native')) as launch:
        app.button[0].click().run()
    assert not app.exception
    launch.assert_called_once_with('rebuild',None,None,True)
    assert not app.success
