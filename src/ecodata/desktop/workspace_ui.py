"""Native presentation widgets for the desktop workspace."""

from PySide6.QtCore import QEasingCurve, QEvent, QPoint, QPropertyAnimation, Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QVBoxLayout,
)


def card_shadow(widget, *, primary=False):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(19 if primary else 14)
    shadow.setOffset(0, 4 if primary else 2)
    shadow.setColor(QColor('#08734c' if primary else '#183a2c'))
    color = shadow.color()
    color.setAlpha(42 if primary else 15)
    shadow.setColor(color)
    widget.setGraphicsEffect(shadow)


class SideDrawer(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('sideDrawer')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(23, 20, 23, 20)
        layout.setSpacing(14)
        header = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName('drawerTitle')
        header.addWidget(self.title, 1)
        self.close_button = QPushButton('×')
        self.close_button.setObjectName('drawerClose')
        self.close_button.setAccessibleName('关闭抽屉')
        self.close_button.clicked.connect(self.hide)
        header.addWidget(self.close_button)
        layout.addLayout(header)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.motion = QPropertyAnimation(self, b'pos', self)
        self.motion.setDuration(210)
        self.motion.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.escape = QShortcut(QKeySequence('Esc'), self)
        self.escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.escape.activated.connect(self.hide)
        self.trigger = None
        self.drawer_width = 830
        self.scrim = QPushButton(parent)
        self.scrim.setObjectName('drawerScrim')
        self.scrim.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scrim.setAccessibleName('点击遮罩关闭抽屉')
        self.scrim.clicked.connect(self.hide)
        self.scrim.hide()
        parent.installEventFilter(self)
        self.hide()

    def add_page(self, panel):
        self.pages.addWidget(panel)

    def place(self):
        host = self.parentWidget()
        width = min(self.drawer_width, host.width())
        self.setGeometry(host.width() - width, 0, width, host.height())
        self.scrim.setGeometry(host.rect())

    def open_page(self, title, panel, trigger=None, *, width=830):
        opening = self.isHidden()
        self.trigger = trigger
        self.drawer_width = width
        self.title.setText(title)
        self.pages.setCurrentWidget(panel)
        self.motion.stop()
        self.place()
        self.scrim.show()
        self.scrim.raise_()
        if opening:
            destination = self.pos()
            self.move(self.parentWidget().width(), 0)
            self.show()
            self.motion.setStartValue(QPoint(self.pos()))
            self.motion.setEndValue(destination)
            self.motion.start()
        self.raise_()
        self.close_button.setFocus()

    def hideEvent(self, event):
        self.motion.stop()
        self.scrim.hide()
        if self.trigger is not None and self.trigger.isVisible() and self.trigger.isEnabled():
            self.trigger.setFocus()
        self.trigger = None
        super().hideEvent(event)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize and not self.isHidden():
            self.motion.stop()
            self.place()
        return super().eventFilter(watched, event)
