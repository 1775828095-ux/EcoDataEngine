"""Native desktop presentation for ranked results and output checks."""
from math import isfinite

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QFrame, QGraphicsDropShadowEffect, QLabel, QPlainTextEdit,
                               QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from .empty_state import EmptyDataIllustration
from .output_review import availability_text,conversion_text


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
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.empty_page = QWidget()
        empty = QVBoxLayout(self.empty_page)
        empty.addStretch(1)
        self.illustration = EmptyDataIllustration()
        empty.addWidget(self.illustration, 0, Qt.AlignmentFlag.AlignHCenter)
        self.empty_title = label('', 'emptyTitle')
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.addWidget(self.empty_title)
        self.status = label('', 'emptyNote')
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.addWidget(self.status)
        empty.addStretch(1)
        self.pages.addWidget(self.empty_page)
        self.formats=label('')
        self.formats.setAccessibleName('原始与输出格式说明')
        self.progress_log=QPlainTextEdit(self)
        self.progress_log.hide()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumBlockCount(300)
        self.progress_log.setAccessibleName('真实处理与搜索进度')
        self.progress_log.setPlaceholderText('开始本地执行后，检查、候选评价与写回记录会显示在这里。')
        self.results_page = scroller = QScrollArea()
        scroller.setStyleSheet('QScrollArea, QScrollArea > QWidget > QWidget { background:#fbfdfa; }')
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        panel.setStyleSheet('background:#fbfdfa;')
        self.cards = QVBoxLayout(panel)
        self.cards.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cards.setContentsMargins(0, 0, 6, 0)
        scroller.setWidget(panel)
        self.pages.addWidget(scroller)
        layout.addWidget(self.formats)
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

    def reset(self, status="处理完成后在这里查看方案排名和输出包检查。", *, keep_progress=False, processing=False):
        while self.cards.count():
            widget = self.cards.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        self.empty_title.setText('正在处理' if processing else '等待处理结果')
        self.status.setText(status)
        self.pages.setCurrentWidget(self.empty_page)
        self.illustration.set_running(processing)
        self.formats.clear()
        self.plan_notes = {}
        self.output_check.clear()
        self.output_check.hide()
        if not keep_progress:
            self.progress_log.clear()

    def append_progress(self,message):
        from datetime import datetime
        self.progress_log.appendPlainText(datetime.now().strftime('%H:%M:%S')+'  '+message)

    def render_selection(self, summaries):
        self.illustration.set_running(False)
        self.pages.setCurrentWidget(self.results_page)
        self.cards.addWidget(label('最终方案与评分', 'sectionTitle'))
        for row in summaries:
            card = QFrame()
            card.setObjectName('recommendationCard')
            metrics = row.get('metrics')
            if metrics:
                card.setProperty('winner', True)
            body = QVBoxLayout(card)
            body.setContentsMargins(15, 12, 15, 12)
            body.setSpacing(6)
            if row.get('task'):
                body.addWidget(label(row['task'], 'selectionScope'))
            title=(row.get('selection_label', '本次选中')+' · ' if metrics else
                   '已执行 · ' if row.get('performed') else '')+row['title']
            body.addWidget(label(title, 'recommendationTitle'))
            if metrics:
                fields=[('RMSE','RMSE'),('MAE','MAE'),('count','样本')]
                if metrics.get('objective')!=metrics.get('RMSE'):
                    fields.append(('objective','目标值'))
                scores=[f'{name} {metrics[key]:.5g}' for key,name in fields
                        if isinstance(metrics.get(key),(int,float)) and isfinite(metrics[key])]
                if scores:
                    body.addWidget(label('   ·   '.join(scores),'selectionMetrics'))
                structure=metrics.get('spatial_structure') or {}
                if structure:
                    detail='空间结构：'+structure.get('status','未评价')
                    if structure.get('gradient_RMSE_per_cell') is not None:
                        detail+=f"；梯度误差 {structure['gradient_RMSE_per_cell']:.4g} / 网格间隔"
                    if structure.get('normalized_spectrum_L1') is not None:
                        detail+=f"；频谱形状差 {structure['normalized_spectrum_L1']:.4g}"
                    if structure.get('FSS_status')=='THRESHOLD_NOT_DECLARED':detail+='；未声明事件阈值，不计算FSS'
                    body.addWidget(label(detail+'。结构诊断不等于独立精度认证。'))
                if row.get('parameters'):
                    body.addWidget(label(f"参数：{row['parameters']}"))
            if row.get('proposed'):
                body.addWidget(label(f"尝试的推荐方案：{row['proposed']}"))
            lead='选择依据' if metrics else '失败原因' if row.get('failed') else '处理说明' if row.get('performed') else '评分状态'
            body.addWidget(label(f"{lead}：{row['reason']}"))
            if row.get('reference'):
                body.addWidget(label(f"评价样本：{row['reference']}"))
            if row.get('seed_usage'):
                evidence=row['seed_usage']
                message=f"推荐候选实际评分通过 {evidence['scored']} / {evidence['recommended']}"
                if evidence['mutations'] is not None:
                    message+=f"；以推荐方案为父代实际执行变异 {evidence['mutations']} 次"
                body.addWidget(label(message))
            if row.get('search_note'):
                body.addWidget(label(row['search_note']))
            if row.get('review_id'):
                note=label('', 'selectionExplanation')
                note.hide()
                self.plan_notes[row['review_id']]=note
                body.addWidget(note)
            self.cards.addWidget(card)

    def render(self, items, check, *, summaries=()):
        self.reset(keep_progress=True)
        if items:
            self.pages.setCurrentWidget(self.results_page)
        if summaries:
            self.render_selection(summaries)
            has_rankings = any(row['metrics'] for row in items)
            self.cards.addWidget(label('候选排名与处理记录' if has_rankings else '处理记录', 'sectionTitle'))
        if check.get('format_conversion'):self.formats.setText(conversion_text(check['format_conversion']))
        task = None
        task_number=0
        for row in items:
            if row["task"] and row["task"] != task:
                task = row["task"]
                task_number+=1
                name = f'工程输入处理 · 任务 {task_number}' if '/' in task or '\\' in task else task.replace(":fusion", " · 融合评估")
                self.cards.addWidget(label(name, "recommendationTitle"))
            card = QFrame()
            card.setObjectName("recommendationCard")
            if row["selected"] and row["metrics"]:
                card.setProperty("winner", True)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(14, 10, 14, 11)
            layout.setSpacing(5)
            heading = row["title"]
            if row["metrics"] and row.get('rank') is not None:
                heading = f"第 {row['rank']} 名 · {heading} · {row['family']}"
            if row["selected"] and row["metrics"]:
                heading += ' · '+row.get('selection_label','本次最优')
            elif row.get("performed"):
                heading += " · 已执行"
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
            layout.addWidget(label(row["reason"]))
            self.cards.addWidget(card)
        self.update_check(check)

    def update_check(self, check, review=None, notice=None, *, pending=(), simulated=False):
        explanations=(review or {}).get('plan_explanations',{})
        for plan_id,note in self.plan_notes.items():
            message=''
            if explanation:=explanations.get(plan_id):
                source='模拟处理收益' if simulated else 'DeepSeek 处理收益'
                message=f"{source}：{explanation['benefit']}\n适用边界：{explanation['limits']}"
            elif plan_id in pending:
                message='正在生成这项方案的处理收益说明…'
            note.setText(message)
            note.setVisible(bool(message))
        text = availability_text(check, review)
        if review and review.get('conversion_benefit'):
            text += '\n输出用途 · '+review['conversion_benefit']
        if notice:
            text += "\n" + notice
        self.output_check.setPlainText(text)
        self.output_check.show()
