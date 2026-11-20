import json,zipfile
from pathlib import Path
import pytest
import xarray as xr
from ecodata.desktop import recommendations
from ecodata.desktop.result_summary import ranked_candidates,candidate_options,export_evaluated_candidate
from ecodata.algorithms import platform_search as api
from test_spatial_relationship_fusion import fixture,BUDGET


def advised_row():
    task=dict(id='C1:fusion',context=dict(pair_details=[dict(id='D1.F1',variable_name='rain',physical_family='precipitation',unit='mm',grid_interval=dict(x=dict(step=.5,unit='degrees_east')))]),
        engineering_note='同物理量；执行前检查统计日界',allowed_recipes=[dict(id='R1',steps=['spatial.blend'],name='加权融合',description='两个来源')])
    return recommendations.validate(dict(recommendations=[dict(task='C1:fusion',recipe='R1',reason='两来源需要比较；尚不能保证更好')]),dict(tasks=[task]))[0]


@pytest.fixture
def completed(tmp_path):
    inputs,request,_=fixture()
    folder=tmp_path/'search'
    result=api.process_current_input(inputs,request,folder,budget=BUDGET,seed=11)
    rows=ranked_candidates(folder,result)
    display=dict(grid_searches=dict(task=dict(search_folder=str(folder),rankings=rows,evidence=result['selection_comparison'])))
    return folder,result,display


def test_explanation_separates_facts_unknown_and_advice():
    row=advised_row();text=recommendations.explanation_text(row)
    assert 'D1.F1' in text and 'mm' in text and '0.5' in text
    assert 'UNKNOWN' in text and '尚不能保证更好' in text
    assert row['explanation']['unknowns'] and '适用性建议' in text


def test_select_other_passed_candidate_generates_real_product_without_reselection(tmp_path,completed):
    folder,result,display=completed
    options=candidate_options(display)
    chosen=next(row for row in options if not row['selected'] and row['phase']=='mutation')
    original=(folder/'product.nc').read_bytes()
    destination=tmp_path/'choice.zip'
    export_evaluated_candidate(folder,chosen['key'],destination)
    with zipfile.ZipFile(destination) as archive:
        receipt=json.loads(archive.read('selection.json'))
        assert receipt['chosen_key']==chosen['key'] and receipt['selection']=='USER_SELECTED'
        assert receipt['deployment']['selection_performed'] is False
        assert receipt['native_writeback'] is False
        path=tmp_path/'product.nc';path.write_bytes(archive.read('selected-product.nc'))
        with xr.open_dataset(path) as ds:assert ds.temperature.notnull().all()
    assert (folder/'product.nc').read_bytes()==original
    assert json.loads((folder/'summary.json').read_text())['best']['key']==result['best']['key']


def test_unmeasured_and_changed_input_cannot_be_exported(tmp_path,completed):
    folder,result,_=completed
    with pytest.raises(ValueError,match='评价记录'):export_evaluated_candidate(folder,'not-executed',tmp_path/'invalid.zip')
    index=json.loads((folder/'replay_inputs/index.json').read_text())
    path=folder/'replay_inputs'/index[0]['path'];path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='指纹'):export_evaluated_candidate(folder,result['best']['key'],tmp_path/'changed.zip')
    assert not (tmp_path/'changed.zip').exists()
    inputs=[api.SearchInput(r['filename'],r['role'],(folder/'replay_inputs'/r['path']).read_bytes(),r.get('metadata')) for r in index]
    with pytest.raises(ValueError,match='changed inputs'):
        api.execute_evaluated_candidate(inputs,folder,result['best']['key'],tmp_path/'deployment')


def test_explanation_toggle_and_candidate_choice_are_real_widgets(tmp_path,completed,monkeypatch):
    import os
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication,QPlainTextEdit,QPushButton
    from ecodata.desktop.__main__ import Window
    app=QApplication.instance() or QApplication([])
    window=Window();window.render_recommendations(dict(mode='界面功能测试',recommendations=[advised_row()]))
    evidence=next(w for w in window.findChildren(QPlainTextEdit) if w.accessibleName()=='数据依据与推荐边界')
    toggle=next(w for w in window.findChildren(QPushButton) if w.text()=='展开数据依据与检查前提')
    assert evidence.isHidden();toggle.click();assert not evidence.isHidden()
    _,_,display=completed;window.set_candidate_options(display)
    assert window.candidate_export.isEnabled() and window.candidate_select.count()>1
    window.candidate_select.setCurrentIndex(1)
    assert 'MAE' in window.candidate_note.text() and '不替换默认工程包' in window.candidate_note.text()
    window.set_candidate_options(None);assert not window.candidate_export.isEnabled()
    window.batch_timer.stop();window.close();app.processEvents()


def test_page_export_handler_uses_actual_user_choice(tmp_path,completed,monkeypatch):
    import os
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication,QFileDialog
    from ecodata.desktop.__main__ import Window
    from ecodata.desktop.file_agent import scan_index
    app=QApplication.instance() or QApplication([])
    source=tmp_path/'user-source.txt';source.write_text('read-only input fixture')
    window=Window();window.scan,_=scan_index(source)
    _,_,display=completed;window.result=display;window.set_candidate_options(display)
    window.candidate_select.setCurrentIndex(1)
    chosen=window.candidate_select.currentData()['key'];destination=tmp_path/'user-choice.zip'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(destination),'ZIP'))
    monkeypatch.setattr(window,'work',lambda fn,callback:callback(fn(lambda _:None)))
    window.export_candidate_now()
    with zipfile.ZipFile(destination) as archive:assert json.loads(archive.read('selection.json'))['chosen_key']==chosen
    assert source.read_text()=='read-only input fixture'
    window.batch_timer.stop();window.close();app.processEvents()
