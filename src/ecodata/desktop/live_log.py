"""Desktop trigger and live timeline for the existing workflow log events."""

from html import escape

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import QPushButton, QTextBrowser


class LogTrigger(QPushButton):
    LABELS = {
        'idle': '实时推演日志 (待命)',
        'busy': '实时推演日志 (推演中...)',
        'ready': '查看实时推演日志',
    }

    def __init__(self):
        super().__init__()
        self.setObjectName('liveLogTrigger')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QSize(16, 16))
        self.pulse = QVariantAnimation(self)
        self.pulse.setDuration(1250)
        self.pulse.setLoopCount(-1)
        self.pulse.setKeyValueAt(0.0, 0.4)
        self.pulse.setKeyValueAt(0.5, 1.0)
        self.pulse.setKeyValueAt(1.0, 0.4)
        self.pulse.valueChanged.connect(self.draw_icon)
        self.set_state('idle')

    def set_state(self, state):
        if self.property('logState') == state:
            return
        self.pulse.stop()
        self.setProperty('logState', state)
        self.setText(self.LABELS[state])
        self.style().unpolish(self)
        self.style().polish(self)
        self.draw_icon(1.0)
        if state == 'busy':
            self.pulse.start()

    def draw_icon(self, opacity):
        state = self.property('logState')
        pixmap = QPixmap(32, 32)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor({'idle': '#475569', 'busy': '#16a34a', 'ready': '#ffffff'}[state])
        if state == 'busy':
            painter.setOpacity(opacity)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(3, 3, 10, 10))
        else:
            pen = QPen(color, 1.5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            if state == 'ready':
                painter.drawLine(QPointF(2.5, 8), QPointF(6.3, 12))
                painter.drawLine(QPointF(6.3, 12), QPointF(13.5, 4))
            else:
                painter.drawRoundedRect(QRectF(1, 2.5, 14, 11), 2.5, 2.5)
                painter.drawLine(QPointF(4, 6), QPointF(6, 8))
                painter.drawLine(QPointF(6, 8), QPointF(4, 10))
                painter.drawLine(QPointF(8, 10), QPointF(11.5, 10))
        painter.end()
        self.setIcon(QIcon(pixmap))


class LiveTimeline(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setObjectName('liveTimeline')
        self.setAccessibleName('实时推演日志时间线')
        self.setPlaceholderText('任务开始后，处理步骤和日志会按时间显示在这里。')
        self.document().setDocumentMargin(12)
        self.setOpenLinks(False)
        bar = self.verticalScrollBar()
        self.scroll = QPropertyAnimation(bar, b'value', self)
        self.scroll.setDuration(190)
        self.scroll.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.follow = QTimer(self)
        self.follow.setSingleShot(True)
        self.follow.timeout.connect(self.scroll_to_latest)
        bar.rangeChanged.connect(lambda *_: self.queue_follow())

    def append_entry(self, timestamp, message, category):
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.document().isEmpty():
            cursor.insertBlock()
        name = {
            'ai': 'AI 辅助',
            'algorithm': '算法执行',
            'system': '',
        }[category]
        tag = f'&nbsp; <span style="color:#000000;font-weight:600">[{name}]</span>' if name else ''
        body = escape(message).replace('\n', '<br>')
        cursor.insertHtml(
            f'<div style="margin-bottom:11px">'
            f'<span style="color:#000000;font-size:9pt">{escape(timestamp)}</span>{tag}<br>'
            f'<span style="color:#000000;font-size:{10 if name else 9}pt">{body}</span></div>'
        )
        self.queue_follow()

    def queue_follow(self):
        if self.isVisible() and not self.follow.isActive():
            self.follow.start(0)

    def scroll_to_latest(self):
        bar = self.verticalScrollBar()
        if bar.value() == bar.maximum():
            return
        self.scroll.stop()
        bar.setValue(bar.maximum())

    def showEvent(self, event):
        super().showEvent(event)
        self.queue_follow()

    def hideEvent(self, event):
        self.follow.stop()
        self.scroll.stop()
        super().hideEvent(event)
