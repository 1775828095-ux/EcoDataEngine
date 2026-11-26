"""Read-only explanation of the workspace workflow."""

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)


STEPS = (
    ('数据接入与解析', '上传多源数据，平台自动解析并扫描文件基础结构。'),
    ('安全抽样与脱敏', '平台从上传的数据中抽取极少量样本，并隐去敏感信息，确保隐私安全。'),
    ('数据类型识别', '将脱敏样本发给 DeepSeek 进行读取，判断数据的具体类型（如工程说明或纯数值数据）。随后，AI 会向平台索要对应的关键表头与具体数据列。'),
    ('算法方案推荐', '平台按要求提供脱敏后的数据，DeepSeek 根据数据特征，推荐基础的数据处理算法组合方案。'),
    ('核心算法编译与寻优', '平台以 AI 推荐的方案作为起点，启动自研的【变异算法引擎】进行深度编译与组合，对方案进行打分评测，最终筛选出评分最高的最优算法。'),
    ('生成依据与导出', '平台给出选择该最优算法的科学依据，用户确认后即可导出处理结果。'),
)


def bulb_icon(color):
    image = QPixmap(44, 44)
    image.setDevicePixelRatio(2)
    image.fill(Qt.GlobalColor.transparent)
    with QPainter(image) as painter:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color), 1.65)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        bulb = QPainterPath(QPointF(8.2, 15.6))
        bulb.cubicTo(8.2, 13.7, 5.6, 12.7, 5.6, 9.2)
        bulb.cubicTo(5.6, 6.2, 8, 4, 11, 4)
        bulb.cubicTo(14, 4, 16.4, 6.2, 16.4, 9.2)
        bulb.cubicTo(16.4, 12.7, 13.8, 13.7, 13.8, 15.6)
        painter.drawPath(bulb)
        painter.drawLine(QPointF(8.5, 16.6), QPointF(13.5, 16.6))
        painter.drawLine(QPointF(9.7, 19), QPointF(12.3, 19))
    return QIcon(image)


class PrinciplesButton(QPushButton):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('principlesTrigger')
        self.setToolTip('平台运行原理')
        self.setAccessibleName('平台运行原理')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(38, 38)
        self.setIconSize(QSize(22, 22))
        self.default_icon = bulb_icon('#52645c')
        self.hover_icon = bulb_icon('#047857')
        self.setIcon(self.default_icon)
        self.dialog = PrinciplesDialog(parent)
        self.clicked.connect(self.dialog.open_centered)

    def enterEvent(self, event):
        self.setIcon(self.hover_icon)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setIcon(self.default_icon)
        super().leaveEvent(event)


class PrinciplesDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName('principlesOverlay')
        self.setWindowTitle('环境数据工作台运行逻辑')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layers = QGridLayout(self)
        layers.setContentsMargins(0, 0, 0, 0)
        self.scrim = QPushButton(self)
        self.scrim.setObjectName('drawerScrim')
        self.scrim.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scrim.setAccessibleName('点击遮罩关闭说明')
        self.scrim.clicked.connect(self.reject)
        layers.addWidget(self.scrim, 0, 0)

        self.panel = QFrame(self)
        self.panel.setObjectName('principlesPanel')
        layers.addWidget(self.panel, 0, 0, Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(25, 17, 22, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel(self.windowTitle(), objectName='drawerTitle')
        header.addWidget(title, 1)
        self.close_button = QPushButton('×', objectName='drawerClose')
        self.close_button.setAccessibleName('关闭运行原理说明')
        self.close_button.clicked.connect(self.reject)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        scroller = QScrollArea(objectName='principlesScroll')
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        timeline = QWidget(objectName='principlesTimeline')
        rows = QVBoxLayout(timeline)
        rows.setContentsMargins(0, 3, 5, 0)
        rows.setSpacing(0)
        for index, (title, text) in enumerate(STEPS, start=1):
            rows.addWidget(self.stage(index, title, text))
        rows.addStretch()
        scroller.setWidget(timeline)
        layout.addWidget(scroller, 1)

    def open_centered(self):
        host = self.parentWidget()
        self.panel.setFixedSize(min(688, host.width() - 40), min(608, host.height() - 40))
        self.setGeometry(QRect(host.mapToGlobal(QPoint()), host.size()))
        self.open()
        self.close_button.setFocus()

    def stage(self, index, title, text):
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(12)
        rail = QVBoxLayout()
        rail.setSpacing(0)
        number = QLabel(str(index), objectName='principleNumber')
        number.setFixedSize(28, 28)
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rail.addWidget(number, 0, Qt.AlignmentFlag.AlignHCenter)
        if index < len(STEPS):
            connector = QFrame(objectName='principleTrack')
            connector.setFixedWidth(2)
            rail.addWidget(connector, 1, Qt.AlignmentFlag.AlignHCenter)
        else:
            rail.addStretch(1)
        line.addLayout(rail)

        stack = QVBoxLayout()
        panel = QFrame(objectName='principleStage')
        stack.addWidget(panel)
        stack.addSpacing(9)
        line.addLayout(stack, 1)
        content = QVBoxLayout(panel)
        content.setContentsMargins(13, 8, 13, 9)
        content.setSpacing(3)
        heading = QHBoxLayout()
        name = QLabel(title, objectName='principleStepTitle')
        name.setAccessibleName(f'步骤 {index}：{title}')
        if index in (3, 4):
            heading.addWidget(name)
            heading.addWidget(QLabel('DeepSeek', objectName='principleAiBadge'))
            heading.addStretch()
        else:
            name.setWordWrap(True)
            heading.addWidget(name, 1)
        content.addLayout(heading)
        description = QLabel(text, objectName='principleText')
        description.setTextFormat(Qt.TextFormat.PlainText)
        description.setWordWrap(True)
        content.addWidget(description)
        return row
