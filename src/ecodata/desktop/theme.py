"""Light sage palette and native Qt control styling."""
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap

def mark(size=40):
    pix=QPixmap(QSize(size,size))
    pix.fill(Qt.GlobalColor.transparent)
    p=QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size/40,size/40)
    pen=QPen(QColor("#3a765d"),1.65)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    top=QPainterPath(QPointF(5,13))
    for x,y in [(20,5.5),(35,13),(20,20.5),(5,13)]:
        top.lineTo(x,y)
    p.fillPath(top,QColor("#dceee2"))
    p.drawPath(top)
    for y in [20,27]:
        layer=QPainterPath(QPointF(5,y))
        layer.lineTo(20,y+7.5)
        layer.lineTo(35,y)
        p.drawPath(layer)
    p.end()
    return pix


def warning_mark():
    pixmap = QPixmap(44, 44)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    with QPainter(pixmap) as painter:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor('#f59e0b'), 1.7)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        triangle = QPainterPath(QPointF(11, 3.5))
        triangle.lineTo(19.5, 18.5)
        triangle.lineTo(2.5, 18.5)
        triangle.closeSubpath()
        painter.drawPath(triangle)
        painter.drawLine(QPointF(11, 8.7), QPointF(11, 12.4))
        painter.drawPoint(QPointF(11, 15.3))
    return pixmap


WORKSPACE_BACKGROUND = '#f0f7f4'

STYLE = """
QMainWindow, QWidget#workspaceSurface { background:__WORKSPACE_BACKGROUND__; }
QDialog, QMessageBox { background:#f0f7f4; color:#172620; }
QMessageBox QLabel { background:transparent; color:#20382e; }
QWidget { color:#34453d; font-size:13px; font-family:'Microsoft YaHei UI','Segoe UI'; }
QWidget:disabled { color:#819087; }
QLabel#title { font-size:26px; font-weight:700; color:#131e18; }
QLabel#subtitle, QLabel[quiet="true"] { color:#536d5d; }
QLabel#workspace {
    background:#e0efe4; color:#35664e; padding:6px 12px;
    border:1px solid #cce1d2; border-radius:12px; font-weight:600;
}
QLabel#pageCount { color:#486557; padding:0 5px; }
QTabWidget::pane { border:1px solid #cdded2; background:#fbfdfa; border-radius:10px; }
QTabWidget#workspaceTabs::pane { background:transparent; border:0; }
QTabBar::tab {
    background:#e6efe8; color:#4b6557; border:1px solid #d6e3d9;
    border-radius:7px; padding:9px 17px; margin-right:6px; margin-bottom:5px;
    font-weight:600;
}
QTabBar::tab:selected { background:#dcefe5; color:#065f46; border:1px solid #add6c0; }
QTabBar::tab:!selected:hover { background:#e0eee3; color:#264f3a; }
QPushButton {
    background:#fbfdfa; color:#315744; border:1px solid #c6dacd;
    border-radius:18px; padding:0 16px; min-height:34px; font-weight:600;
}
QPushButton:hover { background:#e8f3ea; border-color:#9bc1a8; }
QPushButton:pressed { background:#d8eadc; }
QPushButton:focus { border-color:#418264; }
QPushButton[primary="true"] { background:#047857; color:#ffffff; border-color:#047857; }
QPushButton[primary="true"]:hover { background:#245a40; border-color:#245a40; }
QPushButton[primary="true"]:pressed { background:#1a4832; }
QPushButton[primary="true"]:focus { border:1px solid #92c9a7; }
QPushButton:disabled, QPushButton[primary="true"]:disabled {
    background:#e8eee9; color:#819087; border-color:#d7e2d9;
}
QLineEdit, QComboBox {
    background:#f1f6f2; border:1px solid #ceded2; border-radius:6px;
    padding:4px 8px; min-height:20px; selection-background-color:#cce9d5;
    selection-color:#18372a; placeholder-text-color:#62776b;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border-color:#669a7d; }
QComboBox::drop-down { width:24px; border:0; }
QComboBox QAbstractItemView {
    background:#fbfdfa; border:1px solid #aec8b8; padding:3px;
    selection-background-color:#d6eadb; selection-color:#214e37;
}
QPlainTextEdit, QTableWidget {
    background:#ffffff; alternate-background-color:#f5f9f5;
    border:1px solid #d4e3d8; border-radius:8px;
    selection-background-color:#d8ebde; selection-color:#18372a;
}
QPlainTextEdit { padding:8px; }
QPlainTextEdit#outputCheckCard {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(246, 249, 228, 245), stop:0.40 rgba(234, 239, 208, 241),
        stop:1 rgba(227, 234, 197, 242));
    border:1px solid #cdd8ad;
    border-top-color:#f8fbe9;
    border-radius:12px;
    padding:10px 12px;
}
QTableWidget::item:selected { background:#d2ead9; color:#18372a; }
QTableWidget::item:hover:!selected { background:#eef6ef; }
QTableWidget QComboBox { background:transparent; border-color:transparent; padding:0 7px; }
QTableWidget QComboBox:hover { background:#e6f2e8; }
QHeaderView { background:#e9f2eb; }
QHeaderView::section {
    background:#e9f2eb; color:#426352; padding:8px 7px; font-weight:600;
    border:0; border-bottom:1px solid #d4e3d8;
}
QTableCornerButton::section { background:#e9f2eb; border:none; }
QCheckBox { spacing:7px; }
QSplitter::handle { background:#edf4ef; }
QSplitter::handle:hover { background:#c6dfce; }
QStatusBar { color:#596b62; background:#f0f7f4; border-top:1px solid #dce9e0; }
QStatusBar::item { border:0; }
QFrame#recommendationCard { background:#ffffff; border:1px solid #cdded2; border-radius:10px; }
QFrame#recommendationCard[winner="true"] { background:#f2faf3; border:1px solid #a9cfb5; }
QFrame#recommendationCard QLabel { background:transparent; }
QLabel#recommendationTitle { font-size:15px; font-weight:700; color:#214e37; }
QToolTip { background:#fbfdfa; color:#20382e; border:1px solid #b6cebf; padding:5px; }
QFrame#inputCard, QFrame#sideDrawer { background:#fff; border:1px solid #dfeae3; border-radius:12px; }
QFrame#sideDrawer { border-radius:0; border-left:2px solid #bcd8c9; }
QLabel#sectionTitle { color:#17241d; font-size:17px; font-weight:700; }
QLabel#selectionScope { color:#536d5d; font-size:12px; }
QLabel#selectionMetrics { color:#065f46; font-size:16px; font-weight:700; }
QLabel#selectionExplanation { color:#244b3c; background:#edf8f1; border-radius:8px; padding:8px 10px; }
QLabel#drawerTitle { color:#17241d; font-size:21px; font-weight:700; }
QLabel#inputHint, QLabel#emptyNote { color:#596d62; }
QLabel#inputHint { padding:1px 5px; }
QLabel#emptyTitle { font-size:24px; font-weight:700; color:#16261d; }
QLabel#emptyNote { font-size:14px; }
QLabel#workflowConclusion { color:#165940; font-size:17px; font-weight:600; padding:4px 0; }
QPushButton#liveLogTrigger { min-width:182px; }
QPushButton#liveLogTrigger[logState="idle"] { background:#f1f5f9; color:#334155; border-color:#cbd5e1; }
QPushButton#liveLogTrigger[logState="idle"]:hover { background:#e2e8f0; }
QPushButton#liveLogTrigger[logState="busy"] { background:#ecfdf5; color:#065f46; border-color:#059669; }
QPushButton#liveLogTrigger[logState="busy"]:hover { background:#d1fae5; }
QPushButton#liveLogTrigger[logState="ready"] { background:#047857; color:#fff; border-color:#047857; }
QPushButton#liveLogTrigger[logState="ready"]:hover { background:#065f46; }
QPushButton#liveLogTrigger:focus { border:2px solid #059669; }
QPushButton#drawerScrim, QPushButton#drawerScrim:hover, QPushButton#drawerScrim:pressed {
    background:rgba(90,90,90,22); border:0; border-radius:0; padding:0;
}
QPushButton#drawerClose { min-width:34px; padding:0; font-size:22px; color:#000000; }
QPushButton#principlesTrigger {
    background:#f5f8f6; border:1px solid #dce7df; border-radius:14px; min-height:0; padding:0;
}
QPushButton#principlesTrigger:hover, QPushButton#principlesTrigger:focus {
    background:#e6f4eb; border-color:#9dcbb2;
}
QDialog#principlesOverlay { background:transparent; }
QFrame#principlesPanel { background:#ffffff; border:1px solid #d4e4da; border-radius:16px; }
QScrollArea#principlesScroll, QWidget#principlesTimeline { background:#ffffff; }
QFrame#principleTrack { background:#dbe9e0; border:0; }
QLabel#principleNumber { background:#eef5f0; border:1px solid #dce9e0; border-radius:14px; color:#315b49; font-weight:700; }
QFrame#principleStage { background:#f6f5f2; border:1px solid #e8e5df; border-radius:9px; }
QLabel#principleStepTitle { font-size:15px; font-weight:700; color:#17241d; }
QLabel#principleText { color:#394c43; }
QLabel#principleAiBadge { background:#dbeafe; color:#1d4ed8; font-size:11px; font-weight:700; padding:2px 7px; border-radius:8px; }
QTextBrowser#liveTimeline { background:#ffffff; color:#000000; placeholder-text-color:#000000; border:1px solid #e2e2e2; border-radius:10px; }
QPushButton#targetModelButton { background:#edf7f2; color:#075a42; }
QPushButton#targetModelButton:hover { background:#dcefe5; }
QLabel#apiIcon { color:#1685ed; font-size:20px; }
QLabel#apiStatus { color:#596d62; }
QPushButton#apiConnect { min-height:29px; border-radius:15px; }
QPushButton#apiBadge { background:#ecfdf5; color:#047857; border:1px solid #bfe8d4; min-height:28px; border-radius:14px; padding:0 12px; }
QPushButton#apiBadge:hover, QPushButton#apiBadge:focus { background:#dcf7e9; border-color:#82bd9d; }
QPushButton#startPill { min-height:47px; min-width:166px; border-radius:24px; font-size:16px; padding:0 22px; }
QFrame#issueBanner { background:#fffbeb; border:0; border-radius:12px; }
QLabel#issueBannerText { color:#78350f; font-weight:500; }
QFrame#issueBanner QPushButton { background:#fffdf6; color:#70420a; border-color:#e7c679; }
QFrame#issueBanner QPushButton:hover { background:#ffefc5; border-color:#bc8630; }
QLabel#issueBadge { color:#80510e; background:#fff0ca; border-radius:10px; padding:2px 4px; font-weight:600; }
QLabel#issueBadge[severe="true"] { color:#ab2929; background:#fce7e7; }
""".replace('__WORKSPACE_BACKGROUND__', WORKSPACE_BACKGROUND)


