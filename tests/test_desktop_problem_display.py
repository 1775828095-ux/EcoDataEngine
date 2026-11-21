"""Persistent failure presentation; no numerical or model experiments."""
import json,zipfile
from PySide6.QtWidgets import QMessageBox,QFileDialog
from test_desktop_simple_flow import window
from ecodata.desktop import issues
from ecodata.interface.engineering_completeness import IncompleteProject,report,finding


def test_original_validator_names_and_actions_survive(window,monkeypatch):
    window.show()
    exc=IncompleteProject(report('WRF',[finding('wrfbdy_d01','missing','控制文件引用了此边界文件，但包内未找到','提供同工程、同时间窗口的wrfbdy_d01')]))
    monkeypatch.setattr(QMessageBox,'warning',lambda *a:None)
    window.error(exc)
    assert not window.problem_panel.isVisible()
    assert window.problem_table.rowCount()==1
    assert 'wrfbdy_d01' in window.problem_table.item(0,0).text()
    assert '未找到' in window.problem_table.item(0,2).text()
    assert '同时间窗口' in window.problem_table.item(0,3).text()
    window.tabs.setCurrentIndex(1)
    assert window.problem_panel.isVisible()
    window.issue_button.click()
    assert window.drawer.pages.currentWidget() is window.issue_details
    window.drawer.hide()
    window.tabs.setCurrentIndex(2)
    assert not window.problem_panel.isVisible() and window.problem_table.rowCount()==1


def test_no_executable_plan_lists_the_actual_reason(window):
    window.continue_automatically(dict(failures=[]),[dict(candidate='C1',action='fusion',status='条件不足',
        reason='POWER来源未声明UTC或LST统计日界',block_code='time_support_unknown')])
    assert window.problem_table.rowCount()==1
    assert 'UTC或LST' in window.problem_table.item(0,2).text()
    assert '原始请求回执' in window.problem_table.item(0,3).text()


def test_problem_identifies_exact_files_from_existing_candidate_binding(window):
    window.scan=dict(rows=[dict(name='folder/sourceA.nc'),dict(name='other/sourceB.nc')])
    window.candidates={'C1':((0,'D1.F1',dict(name='rain')),(1,'D2.F1',dict(name='precip')),['fusion'])}
    row=issues.item('输入核验','C1','时间口径未确认')
    window.present_issues([row]);text=window.problem_table.item(0,0).text()
    assert 'sourceA.nc' in text and 'sourceB.nc' in text
    assert window.current_issues[0]['source_paths']==['folder/sourceA.nc','other/sourceB.nc']
    window.present_issues([*window.current_issues,row]);assert window.problem_table.rowCount()==1
    window.present_issues([issues.item('处理','C1:fusion','缺少统计区间')])
    assert 'sourceA.nc' in window.problem_table.item(0,0).text()


def test_missing_reason_is_unknown_not_an_invented_file(window):
    window.continue_automatically(dict(failures=[]),[])
    assert '后台未提供具体缺项' in window.problem_table.item(0,2).text()


def test_export_issue_list_without_product_and_reset(window,monkeypatch,tmp_path):
    destination=tmp_path/'issues.json'
    window.present_issues([issues.item('时间核验','source.nc','缺少时间统计区间','提供原始时间定义')])
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(destination),'JSON'))
    window.export_issues()
    assert json.loads(destination.read_text(encoding='utf8'))['issues'][0]['target']=='source.nc'
    window.reset_target();assert window.problem_panel.isHidden() and window.problem_table.rowCount()==0


def test_model_not_run_does_not_change_passed_format():
    result=dict(manifest=dict(final_check=dict(format=dict(status='PASS'),project=dict(status='PASS'),
        native=dict(status='NOT_RUN',reason='本机缺少匹配版本模型程序'))))
    rows=issues.output(result)
    assert len(rows)==1 and rows[0]['stage']=='模型试读'
    assert result['manifest']['final_check']['format']['status']=='PASS'


def test_partial_output_preserves_all_failures():
    result=dict(failed_tasks=[dict(task='A',reason='缺少站点坐标'),dict(task='B',reason='单位冲突')],manifest={})
    rows=issues.output(result);assert len(rows)==2
    assert {r['target'] for r in rows}=={'A','B'}


def test_existing_admission_and_nested_failure_details_are_not_lost():
    detail=issues.item('工程依赖','met.nc','缺少工程引用文件','补充工程原文件')
    result=dict(failed_tasks=[dict(task='A',reason='summary',issues=[detail])],
        manifest=dict(preparation=dict(source_admission=dict(required_information=[dict(code='TIME_UNKNOWN',message='未声明统计日界',resources=['rain.nc'])]))))
    rows=issues.output(result)
    assert any(r['target']=='met.nc' and r['action']=='补充工程原文件' for r in rows)
    assert any('统计日界' in r['problem'] and 'rain.nc' in r['target'] for r in rows)


def test_downloaded_project_contains_problem_report_without_touching_source(tmp_path):
    from ecodata.desktop.service import export_original_zip
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as z:z.writestr('original.txt','original bytes')
    before=source.read_bytes();target=tmp_path/'output.zip'
    export_original_zip(dict(original_archive=source,manifest={},failed_tasks=[dict(task='C1',reason='缺少明确日界')]),target,None)
    with zipfile.ZipFile(target) as z:
        rows=json.loads(z.read('desktop_issues.json'))
        assert rows[0]['target']=='C1' and '日界' in rows[0]['problem']
        assert z.read('original.txt')==b'original bytes'
    assert source.read_bytes()==before
