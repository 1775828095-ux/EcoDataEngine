from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from ecodata.desktop.principles import STEPS
from test_desktop_simple_flow import window


def test_principles_entry_is_icon_only_and_opens_a_modal_timeline(window):
    window.show()
    button = window.principles_button
    assert not button.text() and not button.icon().isNull()
    assert button.toolTip() == '平台运行原理'

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    dialog = button.dialog
    assert dialog.isVisible() and dialog.isModal()
    assert dialog.windowTitle() == '环境数据工作台运行逻辑'
    assert dialog.panel.geometry().center() == dialog.rect().center()
    assert [row.text() for row in dialog.findChildren(QLabel, 'principleText')] == [body for _, body in STEPS]
    assert [row.text() for row in dialog.findChildren(QLabel, 'principleAiBadge')] == ['DeepSeek'] * 2
    assert dialog.findChildren(QLabel, 'principleStepTitle')[4].text() == '核心算法编译与寻优'
    numbers = dialog.findChildren(QLabel, 'principleNumber')
    assert [row.text() for row in numbers] == list('123456')
    assert all(row.property('core') is None for row in numbers)

    QTest.mouseClick(dialog.scrim, Qt.MouseButton.LeftButton, pos=dialog.scrim.rect().topLeft())
    assert not dialog.isVisible()
    button.click()
    QTest.mouseClick(dialog.close_button, Qt.MouseButton.LeftButton)
    assert not dialog.isVisible()
    button.click()
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert not dialog.isVisible()
