from threading import Event

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton

from test_desktop_simple_flow import window


def test_unified_log_keeps_receiving_with_drawer_closed(window):
    window.show()
    assert window.log_button.property('logState') == 'idle'
    assert window.findChild(QPushButton, 'assistantOrb') is None
    started, release = Event(), Event()

    def work(progress):
        progress('DeepSeek 正在核验候选')
        started.set()
        release.wait(5)
        progress('已完成候选评价 · RMSE 1.25')

    window.work(work, lambda _: None)
    try:
        assert started.wait(2)
        QTest.qWait(40)
        assert window.log_button.property('logState') == 'busy'
        assert window.drawer.isHidden()
        assert '[AI 辅助]' in window.workflow_log.toPlainText()
        QTest.mouseClick(window.log_button, Qt.MouseButton.LeftButton)
        QTest.qWait(240)
        assert window.drawer.width() == 500 and window.drawer.scrim.isVisible()
        assert window.drawer.title.text() == '实时推演日志'
        QTest.mouseClick(window.drawer.scrim, Qt.MouseButton.LeftButton)
        assert window.drawer.isHidden() and not window.drawer.scrim.isVisible()
    finally:
        release.set()
        for _ in range(100):
            if window.thread is None:
                break
            QTest.qWait(20)
    assert window.thread is None and window.log_button.property('logState') == 'ready'
    text = window.workflow_log.toPlainText()
    assert text.index('DeepSeek') < text.index('RMSE 1.25')
    assert '[算法执行]' in text
    html = window.workflow_log.document().toHtml()
    assert 'color:#000000' in html
    assert '#2563eb' not in html and '#065f46' not in html


def test_open_log_follows_new_entries_and_preserves_literal_text(window):
    window.show()
    for index in range(45):
        window.log_event(f'读取成员 {index}')
    window.log_button.click()
    QTest.qWait(260)
    scrollbar = window.workflow_log.verticalScrollBar()
    assert scrollbar.maximum() > 0 and scrollbar.value() == scrollbar.maximum()
    scrollbar.setValue(0)
    window.log_event('本地依据：<字段> & 新条目')
    QTest.qWait(260)
    assert scrollbar.value() == scrollbar.maximum()
    assert '本地依据：<字段> & 新条目' in window.workflow_log.toPlainText()
    assert window.drawer.title.text() == '实时推演日志'
    window.drawer.close_button.click()
    assert not window.drawer.scrim.isVisible()


def test_technical_drawer_uses_same_dismissible_overlay(window):
    window.show()
    window.technical_toggle.click()
    assert window.drawer.scrim.isVisible() and window.drawer.close_button.text() == '×'
    QTest.mouseClick(window.drawer.scrim, Qt.MouseButton.LeftButton)
    assert window.drawer.isHidden() and not window.drawer.scrim.isVisible()
