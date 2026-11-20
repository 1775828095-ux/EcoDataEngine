"""UI behavior fixtures; mocked scientific work is not model-run evidence."""
import json
import os
import pytest
from PySide6.QtWidgets import QApplication,QCheckBox
from PySide6.QtGui import QFont,QFontDatabase
from ecodata.desktop.__main__ import Window
from ecodata.desktop import output_review


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-not-a-real-key')
    app=QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    app.setFont(QFont('Microsoft YaHei',10))
    w=Window();yield w
    w.batch_timer.stop();w.thread=None;w.close();app.processEvents()


def test_initial_screen_hides_legacy_steps(window):
    assert '模拟技术引擎' in window.windowTitle()
    assert window.start_button.text()=='开始自动处理'
    assert window.scan_button.isHidden() and window.decision_send.isHidden()
    assert window.alg.isHidden() and window.execute_button.isHidden()
    assert window.audit_panel.isHidden()


def test_start_checks_resources_then_indexes_once(window,monkeypatch,tmp_path):
    path=tmp_path/'input.csv';path.write_text('value\n1\n')
    window.path.setText(str(path));window.set_decision_model('swat_plus')
    calls=[];monkeypatch.setattr(window,'scan_now',lambda:calls.append(window.auto_phase))
    window.start_workflow();assert calls==['index'] and window.auto_run
    window.thread=object();window.start_workflow();assert calls==['index']
    window.thread=None


def test_missing_key_does_not_scan(window,monkeypatch,tmp_path):
    monkeypatch.delenv('DEEPSEEK_API_KEY');window.path.setText(str(tmp_path/'input.csv'))
    window.set_decision_model('swat_plus')
    monkeypatch.setattr(window,'scan_now',lambda:pytest.fail('must not scan'))
    with pytest.raises(ValueError,match='密钥'):window.start_workflow()


def test_prepared_request_automatically_enters_shared_decision(window,monkeypatch):
    window.scan=dict(rows=[],signature=[]);window.decision_model='swat_plus'
    window.decision_json.blockSignals(True)
    window.decision_json.setPlainText(json.dumps(dict(stage='decision',catalog_first=True,sources=[],target_model={'id':'swat_plus','name':'SWAT+'})))
    window.decision_json.blockSignals(False)
    window.auto_run=True;window.auto_phase='prepare';calls=[]
    monkeypatch.setattr(window,'send_decision',lambda:calls.append('shared_decision'))
    monkeypatch.setattr(window,'work',lambda fn,then,**kw:then(fn(lambda _:None)))
    window.prepare_edited_batches()
    assert calls==['shared_decision'] and window.auto_phase=='analyze'


def test_checked_recipes_pause_for_user_choice_before_execution(window,monkeypatch):
    row=dict(task='C1:fusion',steps=['spatial.blend'])
    check=QCheckBox();window.recommendation_rows=[row];window.recommendation_checks=[check]
    calls=[];monkeypatch.setattr(window,'continue_recommendations',lambda:calls.append(check.isChecked()))
    window.auto_run=True;window.continue_automatically(dict(failures=[]),[dict(status='通过')])
    assert calls==[] and not window.auto_run
    assert window.tabs.currentIndex()==1 and check.isChecked()
    assert '勾选方案' in window.workflow_status.text()


def test_blocked_engineering_never_gets_blanket_permission(window,monkeypatch):
    row=dict(task='C1:fusion',steps=['engineering.port_binding'])
    window.recommendation_rows=[row];window.recommendation_checks=[QCheckBox()]
    monkeypatch.setattr(window,'continue_recommendations',lambda:pytest.fail('blocked'))
    window.continue_automatically(dict(failures=[]),[dict(status='未通过')])
    assert not window.original_recommendation_allowed
    assert '未放行' in window.workflow_status.text()


def test_advisory_or_failed_plan_does_not_fake_processing(window,monkeypatch):
    monkeypatch.setattr(window,'continue_recommendations',lambda:pytest.fail('must pause'))
    window.recommendation_rows=[];window.recommendation_checks=[]
    window.continue_automatically(dict(failures=[]),[])
    assert '尚未运行融合' in window.workflow_status.text()
    window.continue_automatically(dict(failures=['offline']),[])
    assert '未完成' in window.workflow_status.text()


def test_format_transition_uses_written_record_not_planned_port():
    result=dict(source_formats=['ZIP','NetCDF'],manifest=dict(native_writeback_verified=True,
        changes=[dict(path='case/rain.pcp')],preparation=dict(tasks=[dict(port=dict(resource='case/rain.pcp'),
        sources=[dict(resource='source/rain.nc')]),dict(port=dict(resource='never_written.tmp'),sources=[dict(resource='x.csv')])])))
    report=output_review.format_conversion(result)
    assert len(report['files'])==1
    assert report['files'][0]['source_format']=='NetCDF' and report['files'][0]['target_format']=='SWAT+ 降水文本（.pcp）'
    text=output_review.conversion_text(report)
    assert 'NetCDF → SWAT+ 降水文本（.pcp）' in text and 'ZIP是下载容器' in text


def test_same_format_is_explicit_and_unknown_not_invented():
    report=output_review.format_conversion(dict(source_formats=['NetCDF'],local=dict(artifacts=[dict(name='updated.nc')])))
    assert '格式相同' in output_review.conversion_text(report)
    unknown=output_review.format_conversion(dict(original_archive='unknown.zip'))
    assert not unknown['source_formats'] and not unknown['target_formats']


def test_download_has_format_record_and_preserves_original(tmp_path):
    import zipfile
    from ecodata.desktop.service import export_original_zip
    original=tmp_path/'original.zip'
    with zipfile.ZipFile(original,'w') as z:z.writestr('input.txt','functional fixture')
    before=original.read_bytes()
    result=dict(original_archive=original,manifest={},source_formats=['CSV'],target_format='NetCDF')
    destination=tmp_path/'download.zip';export_original_zip(result,destination,None)
    with zipfile.ZipFile(destination) as z:
        record=json.loads(z.read('desktop_format_conversion.json'))
        assert record['source_formats']==['CSV'] and record['target_formats']==['NetCDF']
        assert z.read('input.txt')==b'functional fixture'
    assert original.read_bytes()==before
