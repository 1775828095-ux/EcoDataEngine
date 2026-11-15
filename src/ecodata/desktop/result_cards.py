"""Native desktop presentation for ranked results and output checks."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QFrame, QGraphicsDropShadowEffect, QLabel, QPlainTextEdit,
                               QScrollArea, QVBoxLayout, QWidget)

from .output_review import availability_text


def label(text, name=None):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


class ResultCards(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status = label("")
        layout.addWidget(self.status)
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        self.cards = QVBoxLayout(panel)
        self.cards.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cards.setContentsMargins(0, 0, 6, 0)
        scroller.setWidget(panel)
        layout.addWidget(scroller, 1)
        self.output_check = QPlainTextEdit()
        self.output_check.setReadOnly(True)
        self.output_check.setMaximumHeight(122)
        self.output_check.setAccessibleName("输出包检查")
        self.output_check.setObjectName("outputCheckCard")
        shadow = QGraphicsDropShadowEffect(self.output_check)
        shadow.setBlurRadius(13)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(77, 89, 54, 38))
        self.output_check.setGraphicsEffect(shadow)
        self.reset()

    def reset(self, status="处理完成后在这里查看方案排名和输出包检查。"):
        while self.cards.count():
            widget = self.cards.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        self.status.setText(status)
        self.notes = []
        self.output_check.clear()
        self.output_check.hide()

    def render(self, items, check):
        self.reset("方案排名 · 各任务按本地评分排列" if any(row["metrics"] for row in items) else "输出结果")
        task = None
        for row in items:
            if row["task"] and row["task"] != task:
                task = row["task"]
                name = task.replace(":fusion", " · 融合评估")
                self.cards.addWidget(label(name, "recommendationTitle"))
            card = QFrame()
            card.setObjectName("recommendationCard")
            if row["selected"]:
                card.setProperty("winner", True)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(14, 10, 14, 11)
            layout.setSpacing(5)
            heading = row["title"]
            if row["metrics"]:
                heading = f"第 {row['rank']} 名 · {heading} · {row['family']}"
            if row["selected"]:
                heading += " · 本次最优"
            layout.addWidget(label(heading, "recommendationTitle"))
            metric = row["metrics"]
            if metric:
                score = f"RMSE {metric['RMSE']:.4g}   ·   MAE {metric['MAE']:.4g}   ·   样本 {metric['count']}"
                if row["proxy"]:
                    score += "   ·   格点代理评分"
                layout.addWidget(label(score))
            if row.get("audit"):
                layout.addWidget(label(" · ".join(row["audit"])))
            details = row["format_line"]
            if row["parameters"]:
                details += f"   ·   {row['parameters']}"
            layout.addWidget(label(details))
            note = label(row["reason"])
            layout.addWidget(note)
            self.notes.append((note, row["reason"]))
            self.cards.addWidget(card)
        self.update_check(check)

    def update_check(self, check, review=None, notice=None):
        addition = (review or {}).get("conversion_benefit", "")
        for widget, reason in self.notes:
            widget.setText(f"{reason} {addition}".strip())
        text = availability_text(check, review)
        if notice:
            text += "\n" + notice
        self.output_check.setPlainText(text)
        self.output_check.show()
