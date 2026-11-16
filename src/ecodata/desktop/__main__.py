import argparse
import json
import os
import re
from pathlib import PurePosixPath
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
    QPushButton,QMenu,QLineEdit,QComboBox,QLabel,QPlainTextEdit,QFileDialog,QTabWidget,
    QTableWidget,QTableWidgetItem,QAbstractItemView,QSplitter,QCheckBox,QMessageBox,QScrollArea,QFrame)
from ecodata.models.requirements import RequirementStore
from .scan import Scanner, assert_current
from . import service as s
from . import discovery, output_review, recommendations, request_batches
from .result_cards import ResultCards
from .theme import STYLE, mark

class Job(QObject):
    done=Signal(object,object)
    progress=Signal(str)
    batch_feedback=Signal(object)
    def __init__(self,fn,*,with_feedback=False):
        super().__init__()
        self.fn=fn
        self.with_feedback=with_feedback

    @Slot()
    def run(self):
        try:
            args=(self.progress.emit,self.batch_feedback.emit) if self.with_feedback else (self.progress.emit,)
            self.done.emit(self.fn(*args),None)
        except Exception as exc: self.done.emit(None,exc)

def page():
    e=QPlainTextEdit()
    e.setReadOnly(True)
    return e

def render_local_check(record, labels):
    title=labels.get(record["check"],"超出授权的核验")
    terms={"equivalent":"单位等价","convertible":"单位可换算",
           "physical_identity_unconfirmed":"物理量未确认","unit_unknown":"单位未知",
           "unit_not_convertible":"单位不可换算",
           "ranges_overlap":"时间范围重叠","ranges_disjoint":"时间范围不重叠",
           "sampled_ranges_overlap":"抽样时间范围重叠",
           "overlap":"抽样坐标范围重叠","disjoint":"抽样坐标范围未重叠","unknown":"关系待核验"}
    if record["status"]=="outside_scope":
        text="超出授权范围"
    elif record["status"]!="checked":
        reasons={"sample_format_unavailable":"此文件格式尚未识别到可读取的数值字段",
                 "no_numeric_samples":"抽样未取得有限的数值",
                 "position_verification_needs_supported_coordinates":"此组合缺少当前同位置核验支持的坐标结构",
                 "sampled_time_records_not_verified_equal":"未证实抽样记录的时间与口径一致",
                 "too_few_matching_sample_locations":"本次抽样可对应位置少于 10 个，无法直接核对同位置的值",
                 "verified_geographic_coordinates_missing":"缺少可核验的地理坐标"}
        text=reasons.get(record.get("basis"),"有效证据不足")
    elif "sample_fields" in record:
        text="；".join(f"{item['field']} 返回 {len(item.get('perturbed_samples',[]))} 个扰动数值" for item in record["sample_fields"])
    elif "perturbed_samples" in record:
        text=f"返回 {len(record['perturbed_samples'])} 个扰动数值"
    elif "sampled_pairs" in record:
        text=f"{record['sampled_pairs']} 对抽样位置"
        if "agreement_percent" in record:
            text+=f"，一致率约 {record['agreement_percent']}%"
        else:
            text+=f"，MAE 约 {record['mae']}，RMSE 约 {record['rmse']}（统一单位）"
    elif "sources" in record:
        text="；".join(f"来源 {name} 抽样 {row['sampled']} 个，"
                       + (f"有限值约 {row['finite_percent']}%" if "finite_percent" in row else "样本不足")
                       + ("，有质量标记信息" if row["quality_flag_metadata_present"] else "")
                       for name,row in zip(("A","B"),record["sources"]))
    else:
        status=record.get("unit_status",record.get("relationship","unknown"))
        text=terms.get(status,"关系待核验")
    return f"{record['candidate']} · 自动核验 · {title} · {text}"


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EcoDataEngine | 环境数据工作台")
        self.resize(1200,790)
        self.scanner=Scanner()
        self.decision_model=None
        self.scan=None
        self.task=TemporaryDirectory(prefix="ecodata-desktop-")
        self.profiles={p["id"]:p for p in RequirementStore().all()}
        self.thread=None
        self.attempt=0
        self.run_temp=None
        self.config_fingerprint=None
        self.batch_replies={}
        self.decision_batches=[]
        self.batch_error=""
        self.batch_state={}
        self.batch_response={}
        self.batch_timer=QTimer(self)
        self.batch_timer.setSingleShot(True)
        self.batch_timer.timeout.connect(self.prepare_edited_batches)
        self.root=QWidget()
        self.setCentralWidget(self.root)
        root=QVBoxLayout(self.root)
        root.setContentsMargins(20,15,20,12)
        root.setSpacing(12)
        logo=mark()
        self.setWindowIcon(QIcon(logo))
        header=QHBoxLayout()
        header.setSpacing(11)
        emblem=QLabel()
        emblem.setPixmap(logo)
        header.addWidget(emblem)
        brand=QVBoxLayout()
        brand.setSpacing(0)
        title=QLabel("EcoDataEngine")
        title.setObjectName("title")
        subtitle=QLabel("环境数据准备工作台")
        subtitle.setObjectName("subtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()
        workspace=QLabel("本地工作空间")
        workspace.setObjectName("workspace")
        header.addWidget(workspace)
        root.addLayout(header)
        line=QHBoxLayout()
        self.path=QLineEdit()
        self.path.setPlaceholderText("输入本地文件、目录或压缩包路径")
        line.addWidget(QLabel("本地数据路径"))
        line.addWidget(self.path,1)
        self.path.setToolTip("输入或拖入文件、压缩包或文件夹；留空点击扫描可选择")
        self.setAcceptDrops(True)
        root.addLayout(line)
        self.tabs=QTabWidget()
        root.addWidget(self.tabs,1)

        scan=QWidget()
        scanbox=QVBoxLayout(scan)
        hint=QLabel("先扫描表头、时空分辨率和工程资料，再审核发送。DeepSeek 判断是否需要处理，平台核验能否实施。")
        hint.setWordWrap(True)
        scanbox.addWidget(hint)
        split=QSplitter()
        self.detail=page()
        self.detail.setPlaceholderText("本地扫描结果与双重核验结果")
        self.decision_json=QPlainTextEdit()
        self.decision_json.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.decision_json.setPlaceholderText("这里会显示可编辑的脱敏 JSON。请勿填写真实数值、路径、敏感字段名或工程原文。")
        self.local_explanation=page()
        self.local_explanation.setPlainText("选择数据路径并点击「扫描数据」，这里会说明发现了哪些候选以及判断依据。")
        self.send_explanation=page()
        for title,explanation,raw in [
            ("本地证据与决策结果",self.local_explanation,self.detail),
            ("待发送给 DeepSeek",self.send_explanation,self.decision_json),
        ]:
            panel=QWidget()
            box=QVBoxLayout(panel)
            box.setContentsMargins(0,0,0,0)
            box.addWidget(QLabel(title))
            views=QTabWidget()
            views.addTab(explanation,"文字说明")
            if raw is self.detail:
                views.addTab(raw,"原始 JSON")
            else:
                requests=QTabWidget()
                requests.addTab(raw,"融合判断 · 可编辑")
                self.decision_batch_json=page()
                self.decision_batch_index=self.add_batch_view(requests,self.decision_batch_json,"融合判断 · 实际分批")
                self.recommend_json=page()
                self.recommend_json.setPlaceholderText("确认发送并完成核验后，这里会显示算法推荐的完整请求。")
                self.recommend_batch_index=self.add_batch_view(requests,self.recommend_json,"算法推荐 · 完整请求")
                views.addTab(requests,"查看待发 JSON")
            box.addWidget(views,1)
            split.addWidget(panel)
        split.setSizes([550,550])
        scanbox.addWidget(split,1)
        connection=QHBoxLayout()
        connection.addWidget(QLabel("DeepSeek 线上 API"))
        self.api_key=QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("在此粘贴 DeepSeek API 密钥（仅本次运行）")
        connection.addWidget(self.api_key,1)
        self.api_status=QLabel()
        connection.addWidget(self.api_status)
        scanbox.addLayout(connection)
        actions=QHBoxLayout()
        self.scan_button=self.button(actions,"扫描数据",self.scan_now,primary=True)
        self.model_button=self.button(actions,"选择模型",self.choose_decision_model)
        actions.addStretch(1)
        self.decision_send=self.button(actions,"确认发送",self.send_decision,primary=True)
        self.ai_advice_button=self.button(actions,"获取 AI 建议",self.get_recommendations)
        self.ai_advice_button.setToolTip("先确认发送；点击后发送算法推荐请求，收到回复后查看推荐卡片")
        scanbox.addLayout(actions)
        self.tabs.addTab(scan,"1  扫描与决策")

        plan=QWidget(self)
        plan.hide()
        planbox=QVBoxLayout(plan)
        self.table=QTableWidget(0,4)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["使用","文件 / 成员","格式","来源角色"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0,50)
        self.table.setColumnWidth(1,390)
        self.table.setColumnWidth(2,82)
        self.table.setMaximumHeight(140)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        planbox.addWidget(self.table)
        nav=QHBoxLayout()
        self.button(nav,"上一页",lambda:self.paginate(-1))
        self.page_label=QLabel()
        nav.addWidget(self.page_label)
        self.button(nav,"下一页",lambda:self.paginate(1))
        self.show_all=QCheckBox("显示全部成员")
        self.show_all.toggled.connect(self.filter_members)
        nav.addWidget(self.show_all)
        nav.addStretch()
        planbox.addLayout(nav)
        form=QFormLayout()
        form.setVerticalSpacing(5)
        form.setHorizontalSpacing(14)
        self.mode=QComboBox()
        for label,key in [("直接使用 / 格式转换","convert"),("空间融合与变异寻优","fusion"),("CMAQ / CAMx 工程原生转换","native")]:
            self.mode.addItem(label,key)
        self.model=QComboBox()
        self.model.addItem("通用交换数据",None)
        for k,p in self.profiles.items(): self.model.addItem(p["name"],k)
        self.role=QComboBox()
        self.fmt=QComboBox()
        self.fmt.addItems(s.EXPORTS)
        self.variable=QLineEdit()
        self.unit=QLineEdit()
        self.temporal=QComboBox()
        self.temporal.addItems(["","instantaneous","mean","accumulation"])
        self.purpose=QLineEdit()
        self.purpose.setPlaceholderText("融合时填写用途，仅保存在本机")
        self.config=QLineEdit()
        self.config.setPlaceholderText("原生转换配置：本地 JSON 文件路径")
        for label,w in [("处理方式",self.mode),("目标模型",self.model),("输入类别",self.role),("输出格式",self.fmt),
                        ("字段（模型映射后可自动选择）",self.variable),("融合物理单位",self.unit),
                        ("融合时间语义",self.temporal),("融合用途",self.purpose),("工程配置",self.config)]:
            form.addRow(label,w)
        planbox.addLayout(form)
        actions=QHBoxLayout()
        self.button(actions,"查看轻扫建议",self.show_suggestions)
        self.button(actions,"深读所选并核验",self.check_now, primary=True)
        self.button(actions,"生成 DeepSeek 规划预览",lambda:self.make_preview("plan"))
        planbox.addLayout(actions)
        self.local=page()
        self.local.setPlaceholderText("本地建议和核验结果会显示在这里。")
        planbox.addWidget(self.local,1)
        recommend=QWidget()
        recbox=QVBoxLayout(recommend)
        self.recommendation_checks=[]
        cards=QScrollArea()
        cards.setWidgetResizable(True)
        cards.setFrameShape(QFrame.Shape.NoFrame)
        card_panel=QWidget()
        self.recommend_cards=QVBoxLayout(card_panel)
        self.recommend_cards.setContentsMargins(0,0,6,0)
        self.recommend_cards.setAlignment(Qt.AlignmentFlag.AlignTop)
        cards.setWidget(card_panel)
        self.recommendation_status=QLabel("请在首页点击「获取 AI 建议」。")
        recbox.addWidget(self.recommendation_status)
        recbox.addWidget(cards,1)
        recactions=QHBoxLayout()
        recactions.addStretch()
        self.search_button=self.button(recactions,"本地寻优",self.continue_recommendations,primary=True)
        recbox.addLayout(recactions)
        self.tabs.addTab(recommend,"2  AI 算法推荐")

        ai=QWidget(self)
        ai.hide()
        aibox=QVBoxLayout(ai)
        aibox.addWidget(QLabel("上方是将发送的实际 JSON 请求体。只有点击审核发送后才调用线上服务。"))
        self.body=page()
        self.advice=page()
        self.body.setPlaceholderText("生成预览后，在此检查实际待发送的 JSON。")
        self.advice.setPlaceholderText("审核发送后，AI 建议或复核理由会显示在这里。")
        ai_split=QSplitter(Qt.Orientation.Vertical)
        ai_split.addWidget(self.body)
        ai_split.addWidget(self.advice)
        aibox.addWidget(ai_split,1)
        buttons=QHBoxLayout()
        self.mock=QCheckBox("使用模拟回复（不联网）")
        buttons.addWidget(self.mock)
        self.button(buttons,"已审核，发送当前预览",self.send_now, primary=True)
        aibox.addLayout(buttons)
        self.review_page=ai

        result=QWidget()
        rbox=QVBoxLayout(result)
        self.alg=QLineEdit()
        self.alg.setPlaceholderText("逗号分隔，填写要确认的 AI 推荐算法编号；留空用原版初始化")
        rbox.addWidget(self.alg)
        row=QHBoxLayout()
        self.execute_button=self.button(row,"确认并执行本地处理",self.execute_now,primary=True)
        rbox.addLayout(row)
        self.result_cards=ResultCards()
        rbox.addWidget(self.result_cards,1)
        rbox.addSpacing(7)
        footer=QHBoxLayout()
        footer.setContentsMargins(2, 0, 2, 7)
        footer.setSpacing(13)
        footer.addWidget(self.result_cards.output_check,1)
        self.export_button=self.button(footer,"导出 ZIP",self.export_now,primary=True)
        rbox.addLayout(footer)
        self.tabs.addTab(result,"3  本地寻优与导出")
        self.model.currentIndexChanged.connect(self.model_changed)
        for widget in (self.mode,self.role,self.fmt,self.temporal):
            widget.currentIndexChanged.connect(self.reset_target)
        for widget in (self.variable,self.unit,self.config,self.purpose):
            widget.textChanged.connect(self.reset_target)
        self.mock.toggled.connect(self.reset_target)
        self.mock.toggled.connect(self.decision_edited)
        self.path.textChanged.connect(self.reset_path)
        self.table.itemChanged.connect(self.selection_changed)
        self.decision_json.textChanged.connect(self.decision_edited)
        self.api_key.textChanged.connect(self.clear_batch_replies)
        for panel in (scanbox,planbox,aibox,rbox,recbox):
            panel.setContentsMargins(13,12,13,12)
            panel.setSpacing(9)
        self.setStyleSheet(STYLE)
        self.statusBar().showMessage("就绪 · 请选择本地数据")
        self.reset_path()

    def button(self,layout,label,fn,*,primary=False):
        b=QPushButton(label)
        b.setProperty("primary",primary)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda:self.safe(fn))
        layout.addWidget(b)
        return b

    def safe(self,fn):
        try: fn()
        except Exception as exc: self.error(exc)

    def error(self,exc):
        self.statusBar().showMessage("操作未完成")
        QMessageBox.warning(self,"操作未完成",str(exc))

    def work(self,fn,then,*,feedback=None):
        if self.thread is not None: return
        self.root.setEnabled(False)
        self.statusBar().showMessage("正在处理…")
        self.thread=QThread()
        self.job=Job(fn,with_feedback=feedback is not None)
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.run)
        self.job.done.connect(self.receive,Qt.ConnectionType.QueuedConnection)
        self.callback=then
        self.job.progress.connect(self.statusBar().showMessage)
        self.feedback_callback=feedback
        if feedback is not None:
            self.job.batch_feedback.connect(self.receive_batch_feedback,Qt.ConnectionType.QueuedConnection)
        self.thread.finished.connect(self.job.deleteLater)
        self.thread.start()

    @Slot(object)
    def receive_batch_feedback(self,event):
        if self.feedback_callback is not None:
            self.feedback_callback(event)

    @Slot(object,object)
    def receive(self,value,error):
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.thread=None
        self.job=None
        self.root.setEnabled(True)
        if error is not None:
            if self.tabs.currentIndex()==2 and self.result is None:
                self.result_cards.status.setText("本地处理未完成。请查看错误提示，修正后返回推荐页重试。")
            self.error(error)
        else:
            self.statusBar().showMessage("操作完成")
            self.safe(lambda:self.callback(value))

    def reset_path(self,*_):
        self.scan=None
        self.selected={}
        self.page_number=0
        self.reset_target()
        self.table.setRowCount(0)
        self.detail.clear()
        self.local_explanation.setPlainText("选择数据路径并点击「扫描数据」，这里会说明发现了哪些候选以及判断依据。")
        self.decision_json.clear()
        self.candidates={}
        self.gated={}
        self.refresh_decision_send()
        self.page_label.clear()

    def reset_target(self,*_):
        self.original_recommendation_allowed=False
        self.selected_recommendations=[]
        self.recommendation_request=None
        self.recommendation_rows=[]
        self.set_batch_view(self.recommend_batch_index,[])
        self.clear_recommendation_cards()
        self.recommendation_status.setText("请在首页点击「获取 AI 建议」。")
        self.ai_advice_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.engineering_continuation=False
        self.execute_button.setText("确认并执行本地处理")
        if self.run_temp is not None:
            self.run_temp.cleanup()
            self.run_temp=None
        self.prepared=self.approved=self.plan_reply=self.review_reply=self.result=None
        self.config_fingerprint=None
        self.body.clear()
        self.advice.clear()
        self.local.clear()
        self.result_cards.reset()
        self.export_button.setEnabled(False)
        self.alg.show()
        self.execute_button.show()
        self.alg.clear()

    def model_changed(self,*_):
        self.role.clear()
        p=self.profile()
        if p: self.role.addItems([r["name"] for r in p["inputs"]])
        self.reset_target()

    def profile(self):
        return self.profiles.get(self.model.currentData())

    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self,event):
        self.path.setText(event.mimeData().urls()[0].toLocalFile())
        event.acceptProposedAction()

    def clear_batch_replies(self):
        self.batch_replies.clear()
        self.refresh_decision_send()

    def add_batch_view(self, tabs, editor, title):
        panel=QWidget()
        layout=QVBoxLayout(panel)
        selector=QComboBox()
        layout.addWidget(selector)
        feedback=page()
        feedback.setMaximumHeight(145)
        feedback.setPlaceholderText("本批回复会在这里自动显示。")
        self.batch_response[selector]=(feedback,{})
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(editor,1)
        layout.addWidget(feedback)
        self.batch_state[selector]=(editor, [])
        selector.currentIndexChanged.connect(lambda index:self.render_batch(selector,index))
        tabs.addTab(panel,title)
        return selector

    def set_batch_view(self, selector, batches):
        editor,_=self.batch_state[selector]
        entries=[]
        for index,batch in enumerate(batches,1):
            title=f"第 {index}/{len(batches)} 批 · {', '.join(batch.ids) or '工程上下文'}"
            if batch.blocked:
                title+=" · 超限"
            entries.append((title,batch))
        feedback,_=self.batch_response[selector]
        self.batch_response[selector]=(feedback,{})
        self.batch_state[selector]=(editor,entries)
        selector.blockSignals(True)
        selector.clear()
        selector.addItems([title for title,_ in entries])
        selector.setEnabled(bool(entries))
        selector.blockSignals(False)
        self.render_batch(selector,selector.currentIndex())

    def render_batch(self, selector, index):
        editor,entries=self.batch_state[selector]
        feedback,replies=self.batch_response[selector]
        if index < 0:
            editor.clear()
            feedback.clear()
            return
        batch=entries[index][1]
        editor.setPlainText(batch.request.body.decode("utf8"))
        feedback.setPlainText(replies.get(index,batch.blocked))

    def show_batch_reply(self, selector, event):
        index=event["batch"]-1
        _,entries=self.batch_state[selector]
        batch=event.get("request_batch")
        if batch is not None and index==len(entries):
            title=f"第 {index+1} 批 · {', '.join(batch.ids) or '工程上下文'}"
            entries.append((title,batch))
            selector.addItem(title)
        feedback,replies=self.batch_response[selector]
        failure=event["failure"]
        text=s.dump(dict(提示=event.get("notice"),失败原因=failure["reason"] if failure else None,线上响应=event.get("api_receipts",[]),
                         复用本地结果=event.get("local_cache",False),
                         自动回传核验=event.get("local_checks",[]),最终结果=event["rows"]))
        replies[index]=text
        if selector.currentIndex()==index:
            feedback.setPlainText(text)
        else:
            selector.setCurrentIndex(index)

    def decision_edited(self):
        self.batch_replies.clear()
        self.decision_batches=[]
        self.batch_error=""
        self.set_batch_view(self.decision_batch_index,[])
        self.gated={}
        self.reset_target()
        self.decision_send.setEnabled(False)
        if self.scan is not None:
            self.show_local_explanation()
            self.update_original_background()
        if self.decision_json.document().isEmpty():
            self.batch_timer.stop()
            self.refresh_decision_send()
        else:
            self.batch_timer.start(280)

    def prepare_edited_batches(self):
        if self.thread is not None:
            self.batch_timer.start(200)
            return
        text=self.decision_json.toPlainText()

        def finish(result):
            self.decision_batches,self.batch_error=result
            self.set_batch_view(self.decision_batch_index,self.decision_batches)
            self.refresh_decision_send()

        def prepare(_):
            try:
                draft=json.loads(text)
                if isinstance(draft,dict) and draft.get("stage")=="decision" and draft.get("catalog_first") is True and isinstance(draft.get("sources"),list):
                    return request_batches.prepare(draft),""
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                return [],str(error)
            return [],""

        self.work(prepare,finish)

    def refresh_decision_send(self):
        has_key=bool(self.api_key.text().strip() or os.environ.get("DEEPSEEK_API_KEY","").strip())
        mock=self.mock.isChecked()
        self.api_status.setText("模拟 · 不联网" if mock else ("密钥已填写 · 点击确认发送联网" if has_key else "请填写密钥"))
        try:
            draft=json.loads(self.decision_json.toPlainText())
            valid=isinstance(draft,dict) and draft.get("stage")=="decision" and draft.get("catalog_first") is True and isinstance(draft.get("sources"),list)
        except ValueError:
            draft={}
            valid=False
        self.decision_send.setEnabled(self.scan is not None and valid and any(not batch.blocked for batch in self.decision_batches)
                                      and self.decision_model is not None and (has_key or mock))
        lines=["发送你在「查看 / 编辑 JSON」中确认的内容，供 DeepSeek 判断是否需要融合或统一分辨率。",
               "将发送扫描到的全部原表头、变量元数据、时空分辨率和工程说明。DeepSeek 先从这些数据中提出组合。",
               "平台只改待发送副本。正文保留段落和表格。文件引用用编号代替，匹配到的敏感信息用占位符替换，普通数值加随机小数。精确时间戳、坐标和完整观测序列留在本机。请在 JSON 中复核工程说明里的项目名、地名等敏感文字。"]
        lines.append("确认发送也授权本轮自动核验。DeepSeek 可追问其提出组合的单位、时间覆盖、空间重叠、"
                     "数据质量和对应位置一致性，也可以按字段请求数值样本。平台给样本加随机小数后回传。"
                     "本地拦截融合时，同一次确认也授权 DeepSeek 复核预览范围内的候选及脱敏拦截原因。")
        if self.decision_model is None:
            lines.append("请先选择目标模型，再确认发送。")
        else:
            lines.append(f"目标模型：{discovery.MODELS[self.decision_model]}。更换模型会重新生成待审核内容。")
        if valid:
            if self.decision_batches:
                lines.append(f"已整理为 {len(self.decision_batches)} 批融合判断。在「融合判断 · 实际分批」查看请求。"
                             "平台优先把完整字段目录和已读取的脱敏工程全文一次发送。"
                             "先一次发送完整 JSON。若 DeepSeek 明确拒绝请求大小或上下文长度，平台自动分批重试。"
                             "只需确认一次，收到回复时自动切换批次。")
                for batch in self.decision_batches:
                    if batch.blocked:
                        lines.append(f"待处理 · {'、'.join(batch.ids)} · {batch.blocked}")
            elif self.batch_error:
                lines.append(f"生成待发批次失败：{self.batch_error}")
            documents=draft.get("engineering_documents",[])
            modes={"full":"脱敏全文","mixed":"保留说明，替换原始数据",
                   "data":"仅发数据结构","unknown":"用途待确认，仅发结构","unread":"未读取正文"}
            if isinstance(documents,list):
                readable=[doc for doc in documents if isinstance(doc,dict)]
                complete=sum(doc.get("sharing")=="full" for doc in readable)
                sources=draft.get("sources",[])
                count=sum(len(source.get("fields",[])) for source in sources)
                lines.append(f"当前草稿包含 {len(sources)} 个数据来源、{count} 个字段和 {complete} 份工程说明全文。")
                for doc in readable:
                    portions=[row.get("path","") for row in doc.get("replaced_sections",[]) if isinstance(row,dict)]
                    hint=(" · 替换位置 " + "、".join(portions[:12])) if portions else ""
                    if len(portions)>12:
                        hint+=f" 等 {len(portions)} 处，完整位置见 JSON"
                    lines.append(f"{doc.get('id','工程成员')} · {modes.get(doc.get('sharing'),'见 JSON')} · "
                                 f"{doc.get('classification_reason','')}{hint}")
            comparisons={"same":"相同","different":"不同","unknown":"待核验","not_comparable":"基准或单位不可直接比较",
                         "ranges_overlap":"范围重叠","ranges_disjoint":"范围不重叠","sampled_ranges_overlap":"抽样范围重叠"}
            for pair in draft.get("candidates",[])[:40]:
                if not isinstance(pair,dict) or not isinstance(pair.get("resolution_comparison"),dict):
                    continue
                facts=pair["resolution_comparison"]
                def word(key):
                    value=facts.get(key)
                    return comparisons.get(value,"待核验") if isinstance(value,str) else "待核验"
                lines.append(f"{pair.get('id','候选')} · 网格间距{word('grid_spacing')}，"
                             f"时间间隔{word('time_spacing')}，时间覆盖：{word('time_coverage')}。")
        elif self.decision_json.toPlainText().strip():
            lines.append("当前 JSON 格式有误，请修正后发送。")
        self.send_explanation.setPlainText("\n\n".join(lines))

    def show_local_explanation(self, reply=None, checked=()):
        labels={"fusion":"融合","align_time":"统一时间分辨率","align_space":"统一空间分辨率",
                "compare_station_grid":"站点与网格配对","convert_units":"单位换算","convert_format":"格式转换",
                "direct":"分别使用","insufficient":"依据不足","reject":"明确反对处理","pending_goal":"待确认用途"}
        count=sum(bool(row["fields"]) for row in self.scan["rows"])
        model_name=discovery.MODELS.get(self.decision_model,"尚未选择")
        lines=[f"目标模型：{model_name}。",f"已登记 {len(self.scan['rows'])} 个成员，读取到 {count} 份数据的字段信息。",
               (f"DeepSeek 提出 {len(self.candidates)} 组候选。" if reply is not None
                else "确认发送后，DeepSeek 将从完整字段目录中选择候选。")]
        deferred=[row for row in self.scan["rows"] if not row["fields"] and
                  any("预算" in reason or "定向" in reason for reason in row["pending"])]
        if deferred:
            lines.append(f"{len(deferred)} 个成员的字段尚未核验。可在「原始 JSON」查看具体文件和原因。")
        if self.scan.get("candidate_scan_limit"):
            lines.append("本次已达到候选配对预算。已读字段仍列在本地字段清单。")
        research = self.scan.get("research_tasks",{}).get(self.decision_model,[])
        if research:
            lines.append("已识别科研任务："+"、".join(task["label"] for task in research)+"。")
        from .native_dispatch import explain as explain_native
        lines.extend(explain_native(self.scan.get("native_model_probes",{}).get(self.decision_model)))
        if self.original_background_ready():
            lines.append("已识别原版处理任务。点击「本地寻优」时，平台会在后台一并执行。")
        if self.decision_model=="cmaq":
            from .cmaq_grid_audit import explain
            lines.extend(explain(self.scan.get("cmaq_grid_checks",[])))
        for cid,(left,right,actions) in self.candidates.items():
            sources=[]
            for index,_,field in (left,right):
                row=self.scan["rows"][index]
                roles="、".join(discovery.ROLE_LABELS[r] for r in row["roles"] if r in discovery.ROLE_LABELS) or "用途待核验"
                sources.append(f"{PurePosixPath(row['name']).name} · {field['name']}（{roles}）")
            lines.append(f"{cid}  {' 与 '.join(sources)}\n  可核验操作：{'、'.join(labels[a] for a in actions)}")
        if reply is None:
            lines.append("确认发送后，DeepSeek 给出处理建议，平台再检查单位、语义、时空覆盖和算法输入条件。工程资料只提供来源背景。")
        else:
            decisions=reply.get("decisions",[])
            passed=sum(row["status"]=="通过" for row in checked)
            planned=sum(row["status"]!="通过" and row.get("next_step") in {"recommendation_only","alignment_then_fusion"} for row in checked)
            result=[f"{reply['mode']}：{len(decisions)} 条判断，{passed+planned} 项进入算法推荐。"]
            usage=request_batches.describe_usage(reply)
            if usage:
                result.append(usage)
            if reply.get("failures"):
                for failure in reply["failures"]:
                    state="全文超限，未发送" if failure.get("blocked") else "待重试"
                    result.append(f"第 {failure['batch']} 批{state} · {'、'.join(failure['ids'])} · {failure['reason']}")
                if any(not failure.get("blocked") for failure in reply["failures"]):
                    result.append("再次点击「确认发送」仅补发未完成批次。")
            if not passed and any(row["status"]=="待确认用途" for row in checked):
                result.append("用途仍待确认。平台会另行核验可比较的分辨率差异。")
            elif not passed and not planned:
                result.append("当前没有可进入算法推荐的任务。")
            if passed or planned:
                result.append("核对以下判断后，可在右侧查看待发请求，再点击「获取 AI 建议」。")
            from .decision_checks import CHECKS
            for record in reply.get("local_checks",[]):
                result.append(render_local_check(record,CHECKS))
            for decision in decisions:
                action=labels.get(decision["action"],decision["action"])
                goal=decision.get("goal")
                purpose=discovery.TASK_GOALS.get(goal,"待确认用途")
                author="平台（对话状态）" if decision.get("origin")=="platform" else "DeepSeek"
                result.append(f"{decision['candidate']} · {author}：{action}。任务目标：{purpose}。{decision['reason'] or '未给出理由'}")
            for local in checked:
                action=labels.get(local["action"],local["action"])
                result.append(f"{local['candidate']} · 平台核验{action}：{local['status']}。{local['reason']}")
                if review:=local.get("fusion_review"):
                    verdict={"uphold":"认可拦截","retain":"保留融合建议","uncertain":"证据仍不足",
                             "unavailable":"未取得有效复核"}[review["verdict"]]
                    result.append(f"{local['candidate']} · DeepSeek 复核：{verdict}。{review['reason']}")
            lines=result+lines
        self.local_explanation.setPlainText("\n\n".join(lines))

    def choose_decision_model(self):
        menu=QMenu(self)
        for model_id,name in discovery.MODELS.items():
            if model_id not in self.profiles:
                continue
            action=menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(model_id==self.decision_model)
            action.triggered.connect(lambda checked=False,mid=model_id:self.safe(lambda:self.set_decision_model(mid)))
        menu.exec(self.model_button.mapToGlobal(self.model_button.rect().bottomLeft()))

    def set_decision_model(self, model_id):
        if model_id==self.decision_model:
            return
        self.decision_model=model_id
        self.model_button.setText(f"目标模型 · {discovery.MODELS[model_id]}")
        self.model.setCurrentIndex(self.model.findData(model_id))
        if self.scan is not None:
            scan=self.scan
            self.decision_json.clear()
            def finish(value):
                summary,self.candidates=value
                self.detail.setPlainText(s.dump(discovery.local_overview(scan,self.candidates)))
                self.decision_json.setPlainText(s.dump(summary))
                self.show_local_explanation()
                self.update_original_background()
            self.work(lambda log:discovery.discover(scan,model_id),finish)

    def scan_now(self):
        path=self.path.text().strip().strip('"')
        if not path:
            path=QFileDialog.getOpenFileName(self,"选择数据文件或压缩包（文件夹可直接输入或拖入）")[0]
            if not path:
                return
            self.path.setText(path)
        candidate=path.strip("'").strip()
        if re.fullmatch(r"(?:Bearer\s+)?sk-[A-Za-z0-9_-]{8,}",candidate,re.IGNORECASE):
            self.path.clear()
            self.api_key.setFocus()
            raise ValueError("这里需要本地数据路径。请把密钥粘贴到首页下方的「DeepSeek 线上 API」输入框。")
        def scan_job(log):
            log("正在读取包索引、表头和工程资料…")
            scanned,metrics=self.scanner.scan(path)
            log("正在识别候选并整理脱敏工程资料…")
            summary,candidates=discovery.discover(scanned,self.decision_model)
            return scanned,metrics,summary,candidates
        self.work(scan_job,self.show_scan)

    def show_scan(self,value):
        scan,self.scan_metrics,summary,candidates=value
        self.reset_path()
        self.scan=scan
        self.candidates=candidates
        self.fill_page()
        self.detail.setPlainText(s.dump(discovery.local_overview(scan,candidates)))
        self.decision_json.setPlainText(s.dump(summary))
        self.show_local_explanation()
        self.update_original_background()
        self.tabs.setCurrentIndex(0)

    def original_background_ready(self):
        if self.scan is None:
            return False
        report = self.scan.get("native_model_probes",{}).get(self.decision_model,{})
        return (bool(self.scan.get("research_tasks",{}).get(self.decision_model))
                or report.get("status")=="declared"
                or (report.get("status")=="project" and not self.candidates))

    def update_original_background(self):
        if self.original_background_ready():
            self.recommendation_status.setText("已识别本地工程任务。点击「本地寻优」运行适配与处理。")
            self.search_button.setEnabled(True)

    def send_decision(self):
        assert_current(self.scan)
        if self.decision_model is None:
            raise ValueError("请先选择目标模型")
        edited=self.decision_json.toPlainText()
        summary=json.loads(edited)
        if not isinstance(summary,dict) or summary.get("stage")!="decision" or summary.get("catalog_first") is not True or not isinstance(summary.get("sources"),list):
            raise ValueError("请保留 stage=decision、catalog_first 和 sources 字段目录")
        if summary.get("target_model")!=dict(id=self.decision_model,name=discovery.MODELS[self.decision_model]):
            raise ValueError("JSON 中的目标模型需与首页选择一致")
        batches=self.decision_batches
        if not any(not batch.blocked for batch in batches):
            raise ValueError(self.batch_error or "未生成待发批次")
        scan,mock=self.scan,self.mock.isChecked()
        self.gated={}
        key=self.api_key.text().strip()
        self.reset_target()
        def decide(log,feedback):
            assert_current(scan)
            from .catalog_decision import decide as decide_catalog
            reply,checked,gated,candidates,approved=decide_catalog(
                scan,summary,batches,self.batch_replies,mock,key,log,feedback)
            eligible=any(row["status"]=="通过" or row.get("next_step") in {"original_workbench","recommendation_only","alignment_then_fusion"}
                         for row in checked)
            if eligible:
                log("正在准备算法推荐请求…")
            request=(request_batches.prepare(recommendations.make_request(
                scan,approved,checked,gated,local_checks=reply.get("local_checks",[]))) if eligible else None)
            return reply,checked,gated,request,candidates
        def finish(value):
            reply,checked,gated,request,candidates=value
            self.candidates=candidates
            self.gated=gated
            self.show_local_explanation(reply,checked)
            self.detail.setPlainText(s.dump(dict(
                DeepSeek=reply, 本地核验=checked,
                待准备算法=[row for row in checked if row["status"]=="通过"],
                扫描依据=discovery.local_overview(scan,candidates))))
            if gated:
                passed=[(key,task) for key,task in gated.items() if task["action"]=="fusion" and "state" in task]
                if not passed:
                    self.local.setPlainText(s.dump(dict(已通过本地核验的候选=[
                        row for row in checked if row["status"]=="通过"],
                        下一步="生成 DeepSeek 规划预览并审核发送以获取重采样方法建议")))
                    self.show_recommendation_request(request)
                    self.update_original_background()
                    self.tabs.setCurrentIndex(0)
                    return
                (cid,_),task=passed[0]
                state=task["state"]
                self.mode.setCurrentIndex(self.mode.findData("fusion"))
                self.model.setCurrentIndex(0)
                self.variable.setText(state["variable"])
                self.unit.setText(state["request"]["unit"])
                self.temporal.setCurrentText(state["request"]["temporal_semantics"])
                self.purpose.setText("用观测校正背景场")
                self.selected=dict(task["selection"])
                self.fill_page()
                state["decision_candidate"]=cid
                self.prepared=state
                self.local.setPlainText(s.dump(dict(已通过本地核验的候选=cid,
                      本地可用算法=state["eligible"], 下一步="生成 DeepSeek 规划预览并审核发送以获取算法推荐")))
            engineering=[row for row in checked if row.get("next_step")=="original_workbench"]
            # The original dispatcher consumes the entire project. Never use this
            # continuation when an audited operation was explicitly blocked.
            if engineering and not gated and all(row.get("next_step")=="original_workbench" for row in checked):
                self.engineering_continuation=True
                self.execute_button.setText("继续原版工程核验与处理")
                self.local.setPlainText(s.dump(dict(放行至原版工程核验=engineering)))
            self.original_recommendation_allowed=bool(engineering) and all(
                row.get("next_step")=="original_workbench" for row in checked) and not reply["failures"]
            if request:
                self.show_recommendation_request(request)
            self.update_original_background()
            self.tabs.setCurrentIndex(0)
        self.work(decide,finish,feedback=lambda event:self.show_batch_reply(self.decision_batch_index,event))

    def show_recommendation_request(self,batches):
        self.recommendation_request=batches
        self.set_batch_view(self.recommend_batch_index,batches)
        self.ai_advice_button.setEnabled(any(not batch.blocked for batch in batches))
        for batch in batches:
            if batch.blocked:
                self.send_explanation.appendPlainText(f"\n算法推荐未发送 · {'、'.join(batch.ids)} · {batch.blocked}")

    def clear_recommendation_cards(self):
        self.recommendation_checks.clear()
        while self.recommend_cards.count():
            item=self.recommend_cards.takeAt(0)
            widget=item.widget()
            widget.hide()
            widget.deleteLater()

    def add_recommendation_card(self,rows):
        card=QFrame()
        card.setObjectName("recommendationCard")
        layout=QVBoxLayout(card)
        layout.setContentsMargins(14,10,14,11)
        layout.setSpacing(7)
        recipe=rows[0]
        steps=recipe["steps"]
        kind="算法组合" if len(steps)>1 else "算法"
        title=QLabel(recipe["name"] if recipe.get("advisory_only") else f"{kind} · {recommendations.recipe_name(steps)}")
        title.setObjectName("recommendationTitle")
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        layout.addWidget(title)
        details=QLabel(recipe["description"])
        details.setTextFormat(Qt.TextFormat.PlainText)
        details.setWordWrap(True)
        details.setToolTip("算法编号："+ " → ".join(steps))
        layout.addWidget(details)
        layout.addWidget(QLabel(f"适用任务 · {len(rows)} 项"))
        for row in rows:
            advisory=row.get("advisory_only")
            if row.get("alignment_plan"):
                layout.addWidget(QLabel(f"{row['task']} · 对齐与融合评估方案"))
                check=None
            else:
                check=QCheckBox(f"{row['task']} · 执行待核验" if advisory else f"{row['task']} · 加入本地寻优")
                check.setEnabled(not advisory)
                layout.addWidget(check)
            reason=QLabel("推荐理由：" + (row["reason"].strip() or "DeepSeek 未提供理由。"))
            reason.setTextFormat(Qt.TextFormat.PlainText)
            reason.setWordWrap(True)
            layout.addWidget(reason)
            self.recommendation_checks.append(check)
        self.recommend_cards.addWidget(card)

    def get_recommendations(self):
        if not self.recommendation_request:
            raise ValueError("请先完成首页核验")
        assert_current(self.scan)
        draft=self.recommendation_request
        scan=self.scan
        self.recommendation_status.setText("正在获取 AI 推荐…")
        self.recommendation_rows=[]
        self.clear_recommendation_cards()
        self.search_button.setEnabled(False)
        key=self.api_key.text().strip()
        mock=self.mock.isChecked()
        def finish(reply):
            rows=reply["recommendations"]
            groups={}
            for row in rows:
                group=(row["task"],row["id"]) if row.get("advisory_only") else tuple(row["steps"])
                groups.setdefault(group,[]).append(row)
            self.recommendation_rows=[row for group in groups.values() for row in group]
            for group in groups.values():
                self.add_recommendation_card(group)
            self.search_button.setEnabled(self.original_background_ready() or
                                          any(not row.get("advisory_only") for row in rows))
            executable=sum(any(not row.get("advisory_only") for row in group) for group in groups.values())
            self.recommendation_status.setText(f"{reply['mode']} · {executable} 种方案可勾选，"
                                                f"{len(groups)-executable} 种待核验"
                                                + ("\n"+usage if (usage:=request_batches.describe_usage(reply)) else ""))
            if reply["failures"]:
                self.tabs.setCurrentIndex(0)
                failed="；".join(f"第 {row['batch']} 批 {'、'.join(row['ids'])} · {row['reason']}"
                                for row in reply["failures"])
                self.send_explanation.appendPlainText("\n算法推荐待重试 · "+failed
                    +("\n再次点击「获取 AI 建议」仅补发未完成批次。"
                       if any(not row.get("blocked") for row in reply["failures"]) else ""))
            else:
                self.tabs.setCurrentIndex(1)
        def request(log,feedback):
            return request_batches.send(draft,self.batch_replies,mock,key,log,lambda:assert_current(scan),
                                        feedback=feedback)
        self.work(request,finish,feedback=lambda event:self.show_batch_reply(self.recommend_batch_index,event))

    def continue_recommendations(self):
        selected=[row for row,check in zip(self.recommendation_rows,self.recommendation_checks)
                  if check is not None and check.isChecked()]
        if not selected:
            if self.original_background_ready():
                if self.scan.get("research_tasks",{}).get(self.decision_model):
                    self.selected_recommendations=[]
                    self.tabs.setCurrentIndex(2)
                    self.execute_selected()
                else:
                    self.execute_original()
                return
            raise ValueError("请至少勾选一个推荐方案")
        assert_current(self.scan)
        engineering=[row for row in selected if row["steps"][0].startswith("engineering.")]
        if engineering and len(engineering)!=len(selected):
            raise ValueError("原版整包工程和逐项寻优请分开选择")
        if engineering and not self.original_recommendation_allowed:
            raise ValueError("整包工程入口无法隔离未放行任务，请先缩小输入范围")
        self.selected_recommendations=selected
        self.engineering_continuation=bool(engineering)
        self.execute_button.setText("继续原版工程核验与处理" if self.engineering_continuation else "确认并执行本地处理")
        task_ids={row["task"] for row in selected}
        local_task=(self.prepared or {}).get("decision_candidate")
        if not engineering and len(task_ids)==1:
            cid,operation=next(iter(task_ids)).split(":")
            task=self.gated.get((cid,operation))
            if operation=="fusion" and task and "state" in task:
                self.prepared=task["state"]
                self.prepared["decision_candidate"]=cid
                local_task=cid
        seeds=list(dict.fromkeys(step for row in selected if row["task"]==f"{local_task}:fusion"
                                 for step in row["steps"] if step in s.ALGORITHMS))
        self.alg.setText(",".join(seeds))
        self.plan_reply=dict(algorithms=seeds,recommendations=selected)
        self.alg.hide()
        self.execute_button.hide()
        self.tabs.setCurrentIndex(2)
        self.execute_now()

    def filter_members(self):
        if self.scan:
            self.page_number=0
            self.fill_page()

    def fill_page(self):
        self.table.blockSignals(True)
        start=self.page_number*120
        self.display=[i for i,r in enumerate(self.scan["rows"]) if self.show_all.isChecked() or r["format"]!="其他"]
        visible=self.display[start:start+120]
        self.visible=visible
        self.table.setRowCount(len(visible))
        for j,i in enumerate(visible):
            row=self.scan["rows"][i]
            check=QTableWidgetItem("")
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(Qt.CheckState.Checked if i in self.selected else Qt.CheckState.Unchecked)
            self.table.setItem(j,0,check)
            self.table.setItem(j,1,QTableWidgetItem(row["name"]))
            self.table.setItem(j,2,QTableWidgetItem(row["format"]))
            role=QComboBox()
            role.addItems(s.ROLES)
            role.setCurrentText(self.selected.get(i,row["roles"][0] if len(row["roles"])==1 else "primary"))
            role.currentTextChanged.connect(lambda r,j=j:self.selection_changed(self.table.item(j,0)))
            self.table.setCellWidget(j,3,role)
        self.table.blockSignals(False)
        self.page_label.setText(f"{self.page_number+1} / {max(1,(len(self.display)+119)//120)} 页 · {len(self.display)} 个数据成员")

    def paginate(self,delta):
        if self.scan and 0<=self.page_number+delta<max(1,(len(self.display)+119)//120):
            self.page_number+=delta
            self.fill_page()

    def selection_changed(self,item):
        if not item or item.column()!=0: return
        j=item.row()
        i=self.visible[j]
        if item.checkState()==Qt.CheckState.Checked: self.selected[i]=self.table.cellWidget(j,3).currentText()
        else: self.selected.pop(i,None)
        self.reset_target()

    def inputs(self):
        if not self.scan: raise ValueError("请先扫描")
        if not self.selected: raise ValueError("请勾选要处理的数据")
        return self.scan,sorted(self.selected.items()),self.mode.currentData(),self.profile(),self.role.currentText(),self.fmt.currentText()

    def show_suggestions(self):
        if not self.scan: raise ValueError("请先扫描")
        self.local.setPlainText(s.dump(s.suggest(self.scan,self.profile(),self.role.currentText(),self.fmt.currentText())))

    def check_now(self):
        args=self.inputs()
        variable,unit,temporal=self.variable.text().strip(),self.unit.text().strip(),self.temporal.currentText()
        config=self.config.text().strip()
        if args[2]=="fusion" and not self.purpose.text().strip(): raise ValueError("请填写明确的融合用途")
        purpose=self.purpose.text().strip()
        def run(_):
            import hashlib
            raw=Path(config).read_bytes() if args[2]=="native" and config else b"{}"
            cfg=raw.decode("utf-8-sig")
            state, report=s.prepare(*args,variable,unit,temporal,cfg)
            state["purpose"]=purpose
            return state,report,hashlib.sha256(raw).digest() if args[2]=="native" else None
        def finish(value):
            self.prepared,report,self.config_fingerprint=value
            self.local.setPlainText(s.dump(report))
        self.work(run,finish)

    def check_config(self):
        if self.mode.currentData()=="native" and self.config_fingerprint is not None:
            import hashlib
            path=Path(self.config.text().strip())
            if hashlib.sha256(path.read_bytes()).digest()!=self.config_fingerprint:
                self.reset_target()
                raise ValueError("本地工程配置已变化，请重新核验和审核")

    def make_preview(self,stage):
        self.check_config()
        alignment=[(key,task) for key,task in self.gated.items() if task["action"]!="fusion"]
        if stage=="plan" and alignment and not (self.prepared and self.prepared.get("decision_candidate")):
            assert_current(self.scan)
            self.approved=s.preview(dict(stage="algorithm",tasks=[
                dict(candidate=key[0],operation=key[1],methods=task.get("methods",["spatial.regrid.conservative"]))
                for key,task in alignment]),Path(self.task.name)/"algorithm.json")
            self.body.setPlainText(self.approved.body.decode("utf8"))
            self.advice.clear()
            self.tabs.setCurrentWidget(self.review_page)
            return
        if stage=="plan":
            scan,selected,mode,profile,role,fmt=self.inputs()
            assert_current(scan)
            if mode=="fusion" and not (self.prepared and (self.prepared.get("decision_candidate"),"fusion") in self.gated):
                raise ValueError("先在首页确认发送，待融合建议通过本地核验后再请求算法")
            req=next((r for r in (profile or {}).get("inputs",[]) if r["name"]==role),None)
            brief=s.planning_summary(scan,selected,mode,"NetCDF" if mode=="fusion" else fmt,req)
            if self.prepared and "eligible" in self.prepared:
                brief["allowed_algorithms"]=[v for v in brief["allowed_algorithms"] if v["id"] in self.prepared["eligible"]]
            self.plan_reply=None
            self.alg.clear()
        else:
            if not self.result: raise ValueError("请先执行并通过本地输出检查")
            if "original_archive" in self.result:
                raise ValueError("原版工程流程的核验与缺项见导出 ZIP 内的工程报告")
            assert_current(self.scan)
            brief=s.review_summary(self.result)
            self.review_reply=None
        self.approved=s.preview(brief,Path(self.task.name)/f"{stage}.json")
        self.body.setPlainText(self.approved.body.decode("utf8"))
        self.advice.clear()
        self.tabs.setCurrentWidget(self.review_page)

    def send_now(self):
        self.check_config()
        if not self.approved: raise ValueError("请先生成预览")
        assert_current(self.scan)
        approved=self.approved
        self.approved=None
        mock=self.mock.isChecked()
        def finish(reply):
            if approved.stage=="plan":
                self.plan_reply=reply
                self.alg.setPlaceholderText("确认后填写（可留空）：" + "，".join(reply["algorithms"]))
            elif approved.stage=="review": self.review_reply=reply
            self.advice.setPlainText(s.dump(reply))
        key=self.api_key.text().strip()
        self.work(lambda _:s.send(approved,approved.digest,mock,api_key=key),finish)

    def execute_now(self):
        if not self.selected_recommendations and self.scan and self.scan.get("research_tasks",{}).get(self.decision_model):
            self.tabs.setCurrentIndex(2)
            self.execute_selected()
            return
        if self.engineering_continuation or (not self.selected_recommendations and self.original_background_ready()):
            self.execute_original()
            return
        if self.selected_recommendations:
            self.execute_selected()
            return
        self.check_config()
        if not self.prepared: raise ValueError("请先按需深读并完成本地核验")
        assert_current(self.scan)
        confirmed=[x.strip() for x in self.alg.text().replace("，",",").split(",") if x.strip()]
        if not set(confirmed)<=set((self.plan_reply or {}).get("algorithms",[])):
            raise ValueError("只能确认当前已审核请求返回的推荐编号")
        if self.run_temp is not None: self.run_temp.cleanup()
        self.run_temp=TemporaryDirectory(dir=self.task.name,prefix="run-")
        folder=Path(self.run_temp.name)
        self.result=self.review_reply=self.approved=None
        self.export_button.setEnabled(False)
        self.result_cards.reset("正在本地处理，请稍候…")
        state=self.prepared
        model=(state.get("profile") or {}).get("id")
        self.work(lambda log:s.run(state,folder,confirmed,log),lambda value:self.show_result(value,model))

    def execute_selected(self):
        from .selected_processing import run
        assert_current(self.scan)
        self.alg.hide()
        self.execute_button.hide()
        recipes={}
        for row in self.selected_recommendations:
            recipes.setdefault(row["task"],[]).append(row["steps"])
        tasks=[(task_id,dict(self.gated[tuple(task_id.split(":",1))],recipes=steps))
               for task_id,steps in recipes.items()]
        if self.run_temp is not None:
            self.run_temp.cleanup()
        self.run_temp=TemporaryDirectory(dir=self.task.name,prefix="selected-")
        folder=Path(self.run_temp.name)
        scan,model_id=self.scan,self.decision_model
        self.result=self.review_reply=self.approved=None
        self.export_button.setEnabled(False)
        self.result_cards.reset("正在按已勾选方案调用原版处理，请稍候…")
        self.work(lambda log:run(scan,tasks,folder,progress=log,model_id=model_id),
                  lambda value:self.show_result(value,model_id))

    def execute_original(self):
        if not self.scan or not self.decision_model:
            raise ValueError("请先在首页选择目标模型并扫描工程")
        assert_current(self.scan)
        self.tabs.setCurrentIndex(2)
        self.alg.hide()
        self.execute_button.hide()
        if self.run_temp is not None:
            self.run_temp.cleanup()
        self.run_temp=TemporaryDirectory(dir=self.task.name,prefix="original-")
        folder=Path(self.run_temp.name)
        scan,model=self.scan,self.decision_model
        self.result=self.review_reply=self.approved=None
        self.export_button.setEnabled(False)
        self.result_cards.reset("正在生成原版工程输出包…")
        self.work(lambda log:s.run_original_workbench(scan,model,folder,
                  lambda msg:log(msg if isinstance(msg,str) else "正在核验原版候选与质量…")),
                  lambda value:self.show_result(value,model))

    def show_result(self, result, model_id):
        self.result=result
        check=output_review.describe_output(result,self.profiles.get(model_id))
        items=output_review.cards(result,check)
        self.result_cards.render(items,check)
        self.export_button.setEnabled(True)
        key=self.api_key.text().strip() or os.environ.get("DEEPSEEK_API_KEY","").strip()
        mock=self.mock.isChecked()
        if not key and not mock:
            self.result_cards.update_check(check,notice="填写首页 API 密钥后可自动补充线上复核。")
            return

        self.result_cards.update_check(check,notice="正在复核脱敏的输出包清单…")
        request=output_review.review_request(check,items)
        summary_file=Path(self.task.name)/"output-review-request.json"

        def review(_):
            try:
                approved=s.preview(request,summary_file)
                return s.send(approved,approved.digest,mock,api_key=key),None
            except (ValueError, OSError, TypeError, KeyError, IndexError):
                return None,"线上复核未完成。当前显示本地检查结果。"

        def finish(value):
            reply,notice=value
            self.review_reply=reply
            if reply and mock:
                notice="模拟复核 · 未调用线上 API"
            self.result_cards.update_check(check,reply,notice)

        self.work(review,finish)

    def export_now(self):
        self.check_config()
        if not self.result: raise ValueError("请先完成本地检查")
        assert_current(self.scan)
        path=QFileDialog.getSaveFileName(self,"导出 ZIP","EcoDataEngine-result.zip","ZIP (*.zip)")[0]
        if not path: return
        if not path.lower().endswith(".zip"): path+=".zip"
        dst=Path(path).resolve()
        if any(p==dst for p,*_ in self.scan["signature"]): raise ValueError("不能覆盖原始输入")
        if "original_archive" in self.result:
            result,reply=self.result,self.review_reply
            self.work(lambda _:s.export_original_zip(result,dst,reply),
                      lambda path:self.statusBar().showMessage(f"已导出 {path.name}"))
            return
        planning,review,result=self.plan_reply,self.review_reply,self.result
        self.work(lambda _:s.export_bundle(result,dst,planning,review),lambda n:self.statusBar().showMessage(f"已导出 {n:,} 字节"))

    def closeEvent(self,event):
        if self.thread is not None:
            event.ignore()
            self.statusBar().showMessage("当前操作完成后可关闭窗口")
        else:
            if self.run_temp is not None: self.run_temp.cleanup()
            self.task.cleanup()
            event.accept()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("path",nargs="?")
    args=parser.parse_args()
    app=QApplication(sys.argv[:1])
    window=Window()
    window.show()
    if args.path:
        window.path.setText(args.path)

    return app.exec()

if __name__=="__main__":
    raise SystemExit(main())




