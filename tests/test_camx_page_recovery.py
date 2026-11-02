from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

SCRIPT='from ecodata.interface.camx_run_page import render_camx_run_page\nrender_camx_run_page()'

@pytest.mark.parametrize('ident,route',[('936642b3726349039780a40ece4d7fae','更新已有工程'),('524de2a9c3c646d3b2214d5a8886cf1d','用配套资料重新组装登记工程')])
def test_restore_saved_job(ident,route):
    app=AppTest.from_string(SCRIPT);app.query_params['camx_job']=ident;app.run()
    assert not app.exception and app.radio[0].value==route
    assert app.success
    assert any('83.244190' in item.value for item in app.markdown)

def test_new_start_failure_hides_old_success():
    app=AppTest.from_string(SCRIPT);app.query_params['camx_job']='524de2a9c3c646d3b2214d5a8886cf1d';app.run()
    with patch('ecodata.interface.camx_run_page.start',side_effect=RuntimeError('injected start failure')):
        app.button[0].click().run()
    assert not app.exception and app.error and 'injected start failure' in app.error[0].value
    assert not app.success

@pytest.mark.parametrize('state',['running','failed'])
def test_non_success_state_has_no_success_output(state):
    record=dict(route='existing',status=state,error='injected native failure')
    app=AppTest.from_string(SCRIPT);app.query_params['camx_job']='a'*32
    with patch('ecodata.interface.camx_run_page.status',return_value=record):app.run()
    assert not app.exception and not app.success
    if state=='failed':assert app.error and 'injected native failure' in app.error[0].value

def test_untrusted_engine_rejected_before_launch(tmp_path,monkeypatch):
    from ecodata.interface import camx_runner as runner
    monkeypatch.setattr(runner,'RUNS',tmp_path)
    monkeypatch.setattr(runner,'sha',lambda path:'invalid hash')
    ident='b'*32;folder=runner.directory(ident);folder.mkdir()
    runner.save(folder,dict(id=ident,route='existing',status='running'))
    assert runner.LOCK.acquire(blocking=False)
    with patch.object(runner.subprocess,'run') as native:
        runner.execute(ident,'existing')
        native.assert_not_called()
    assert runner.status(ident)['status']=='failed'
    assert '哈希' in runner.status(ident)['error']
    assert not runner.LOCK.locked()
