"""Quality-screen suggestions are not fusion goals or waived quality gates."""
import pytest
from PySide6.QtWidgets import QCheckBox
from test_desktop_simple_flow import window
from ecodata.desktop import discovery,issues
import ecodata.desktop.__main__ as desktop


def rows(candidate='C1'):
    return [dict(candidate=candidate,action='fusion',status='待工程核验',next_step='original_workbench'),
        dict(candidate='C1',action='quality_screen',status='待执行质量核验',next_step='runtime_quality_gate',reason='public gates still required')]


def test_pending_quality_has_same_candidate_bound_runtime_or_blocks():
    assert not discovery.blocking_checks(rows())
    assert discovery.blocking_checks(rows('C2'))
    assert discovery.blocking_checks(rows()[1:])
    assert not issues.checks(rows())


def test_real_missing_data_not_hidden_by_quality_plan():
    checks=rows()+[dict(candidate='C1',action='align_time',status='条件不足',reason='missing time basis')]
    assert len(discovery.blocking_checks(checks))==1
    assert discovery.blocking_checks(checks)[0]['reason']=='missing time basis'


def test_auto_and_manual_buttons_keep_pending_quality_and_enter_same_route(window,monkeypatch):
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None)
    check=QCheckBox();check.setChecked(True)
    window.scan={};window.recommendation_rows=[dict(task='C1:fusion',steps=['engineering.port_binding'])]
    window.recommendation_checks=[check];calls=[]
    monkeypatch.setattr(window,'execute_now',lambda:calls.append(list(window.checked_decisions)))
    window.continue_automatically(dict(failures=[]),rows())
    assert calls==[] and window.tabs.currentIndex()==1
    window.search_button.setEnabled(True);window.search_button.click()
    assert len(calls)==1
    assert all(any(r['action']=='quality_screen' and r['status']!='通过' for r in run) for run in calls)


def test_manual_no_runtime_for_quality_still_blocks(window):
    window.update_task_checks(rows()[1:])
    with pytest.raises(ValueError,match='quality_screen'):window.require_engineering_checks()


def test_exact_saved_model_context_does_not_depend_on_agent_reading_control(tmp_path,monkeypatch):
    import json,hashlib
    from ecodata.models import input_context
    from ecodata.desktop import native_dispatch
    from ecodata.desktop.file_agent import scan_index
    store=tmp_path/'contexts';store.mkdir();monkeypatch.setattr(input_context,'STORE',store)
    source=tmp_path/'input.bin';source.write_bytes(b'functional source, not an actual model project')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    ctx=dict(schema='ecodata.saved-input-context.v1',source_sha256=sha,model_id='wrf',status='WORKING_ASSUMPTION_ONLY',
        documents={'metadata.json':dict(schema='ecodata.engineering-semantics.v1',model_id='wrf',resources=[dict(role='model_input',confirmed=True,resource='not-yet-bound')])})
    (store/(sha+'.json')).write_text(json.dumps(ctx))
    scan,_=scan_index(source);result=native_dispatch.inspect(scan,'wrf')
    assert result['status']=='project' and not result['controls']
    assert result['declarations'][0]['origin']=='hash_bound_saved_input_context'
    assert result['declarations'][0]['evidence_status']=='WORKING_ASSUMPTION_ONLY'
    source.write_bytes(b'different source');scan,_=scan_index(source)
    scan['saved_input_context']=ctx  # A forged stale client field is not trusted.
    assert native_dispatch.inspect(scan,'wrf')['status']=='unavailable'
