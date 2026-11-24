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

STYLE = """
QMainWindow { background: #edf4ef; }
QDialog, QMessageBox { background: #edf4ef; color:#20382e; }
QMessageBox QLabel { background:transparent; color:#20382e; }
QWidget { color:#20382e; font-size:13px; }
QWidget:disabled { color:#819087; }
QLabel#title { font-size:24px; font-weight:700; color:#18372a; }
QLabel#subtitle, QLabel[quiet="true"] { color:#536d5d; }
QLabel#workspace {
    background:#e0efe4; color:#35664e; padding:6px 12px;
    border:1px solid #cce1d2; border-radius:12px; font-weight:600;
}
QLabel#pageCount { color:#486557; padding:0 5px; }
QTabWidget::pane { border:1px solid #cdded2; background:#fbfdfa; border-radius:10px; }
QTabBar::tab {
    background:#e6efe8; color:#4b6557; border:1px solid #d6e3d9;
    border-radius:7px; padding:9px 17px; margin-right:6px; margin-bottom:5px;
    font-weight:600;
}
QTabBar::tab:selected { background:#d6eadb; color:#1e553b; border:1px solid #a9cfb5; }
QTabBar::tab:!selected:hover { background:#e0eee3; color:#264f3a; }
QPushButton {
    background:#fbfdfa; color:#315744; border:1px solid #c6dacd;
    border-radius:18px; padding:0 16px; min-height:34px; font-weight:600;
}
QPushButton:hover { background:#e8f3ea; border-color:#9bc1a8; }
QPushButton:pressed { background:#d8eadc; }
QPushButton:focus { border-color:#418264; }
QPushButton[primary="true"] { background:#2e6a4d; color:#ffffff; border-color:#2e6a4d; }
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
QStatusBar { color:#4c6959; background:#e5f0e7; border-top:1px solid #d3e1d6; }
QStatusBar::item { border:0; }
QFrame#recommendationCard { background:#ffffff; border:1px solid #cdded2; border-radius:10px; }
QFrame#recommendationCard[winner="true"] { background:#f2faf3; border:1px solid #a9cfb5; }
QFrame#recommendationCard QLabel { background:transparent; }
QLabel#recommendationTitle { font-size:15px; font-weight:700; color:#214e37; }
QToolTip { background:#fbfdfa; color:#20382e; border:1px solid #b6cebf; padding:5px; }
"""


