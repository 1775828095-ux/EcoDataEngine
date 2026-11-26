from pathlib import Path

import pytest
from test_desktop_simple_flow import window
from ecodata.core.temporary import TemporaryDirectory


def test_window_startup_invokes_shared_recovery(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from ecodata.core import temporary
    from ecodata.desktop.__main__ import Window
    app=QApplication.instance() or QApplication([])
    calls=[]
    report=dict(removed=[],retained=[],failures=[],released_bytes=0)
    monkeypatch.setattr(temporary,'sweep_abandoned',lambda:calls.append('startup') or report)
    w=Window()
    assert calls==['startup'] and w.startup_cleanup is report
    w.close();app.processEvents()


def test_successful_download_cleans_work_and_keeps_saved_result(window,tmp_path):
    output=tmp_path/'saved.zip';output.write_bytes(b'saved-result')
    original=tmp_path/'input.nc';original.write_bytes(b'input')
    old=Path(window.task.name)
    window.run_temp=TemporaryDirectory(dir=old)
    (Path(window.run_temp.name)/'large-work').write_bytes(b'x'*1024)
    window.result={'large':'state'};window.prepared={'large':'data'}
    window.finish_download(output)
    assert not old.exists() and output.read_bytes()==b'saved-result'
    assert original.read_bytes()==b'input' and Path(window.task.name).exists()
    assert window.result is None and window.prepared is None and window.run_temp is None
    assert not window.export_button.isEnabled() and not window.search_button.isEnabled()


def test_cancel_or_failed_save_does_not_delete_undownloaded_product(window,tmp_path):
    old=Path(window.task.name);product=old/'product.nc';product.write_bytes(b'keep')
    with pytest.raises(FileNotFoundError):window.finish_download(tmp_path/'not-saved.zip')
    assert product.read_bytes()==b'keep'


def test_save_inside_work_tree_cannot_delete_its_own_product(window):
    product=Path(window.task.name)/'saved.zip';product.write_bytes(b'keep')
    window.finish_download(product)
    assert product.read_bytes()==b'keep' and '已保留' in window.statusBar().currentMessage()
