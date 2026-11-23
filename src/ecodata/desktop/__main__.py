import argparse
import json
import os
import re
from pathlib import PurePosixPath
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Metadata inspection does not need a machine-wide BLAS thread pool. Respect
# explicitly configured limits and let numerical backends manage their own work.
for thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(thread_variable, '1')

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
    QPushButton,QMenu,QLineEdit,QComboBox,QLabel,QPlainTextEdit,QFileDialog,QTabWidget,
    QTableWidget,QTableWidgetItem,QAbstractItemView,QSplitter,QCheckBox,QMessageBox,QScrollArea,QFrame,QInputDialog)
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
                 "preprocessing_plan_missing":"尚未提供完整的统计窗口和换算方案",
                 "preprocessing_evidence_missing":"前处理依据尚未核实",
                 "position_verification_needs_supported_coordinates":"此组合缺少当前同位置核验支持的坐标结构",
                 "sampled_time_records_not_verified_equal":"未证实抽样记录的时间与口径一致",
                 "too_few_matching_sample_locations":"本次抽样可对应位置少于 10 个，无法直接核对同位置的值",
                 "verified_geographic_coordinates_missing":"缺少可核验的地理坐标",
                 "annual_and_yearly_pair_required":"需同时提供平均年度表和逐年输出表",
                 "same_scenario_and_output_variable_unconfirmed":"未证实两表属于同一工程和同一输出变量",
                 "annual_unit_not_confirmed":"两表单位缺失或未统一",
                 "average_annual_window_unconfirmed":"未证实平均年度输出覆盖哪段年份",
                 "native_yearly_output_unreadable":"原生输出读取未完成"}
        text=reasons.get(record.get("basis"),"有效证据不足")
        if record.get("basis")=="preprocessing_evidence_missing":
            text=record["reason"]
        if record.get("basis")=="native_yearly_output_unreadable":
            source={"average_annual":"平均年度表","yearly":"逐年表"}.get(record.get("failed_output"),"输出表")
            errors={"record_column_count_mismatch":"记录列数与表头不符",
                    "unrecognized_trailing_columns":"尾部额外列缺少可识别的文本标记",
                    "invalid_numeric_value":"声明的数值列有缺测或无效值",
                    "invalid_identity_or_numeric_prefix":"对象编号、年份或声明的数值列无法解析",
                    "duplicate_year_or_invalid_value":"同一对象年份重复或数值无效",
                    "no_output_records":"没有数据记录"}
            location=f"第 {record['line']} 行" if record.get("line") else ""
            text=f"{source}{location} · {errors.get(record.get('read_error'),'本地读取器未完成解析')}"
    elif record["check"]=="preprocessing_plan":
        plan=record["plan"]
        text=f"{plan['start_year']}–{plan['end_year']} 年，目标单位 {plan['target_unit']}，已试算两来源各 {record['output_records']} 条记录"
    elif record["check"]=="annual_mean_comparison":
        period=record["year_window"]
        text=(f"{period['start_year']}–{period['end_year']} 年，共 {record['matched_objects']} 个对应对象，"
              f"{record['compared_objects']} 个覆盖完整年份，其中 {record['matching_means']} 个均值相符")
        if record.get("trailing_annotations"):
            text+=f"；已识别 {len(record['trailing_annotations'])} 行尾部附注"
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
    elif record.get("basis")=="local_field_context":
        context=record.get("sources",[record.get("metadata",{})])
        items=[]
        for field in context:
            crs=field.get("coordinate_reference") or {}
            facts=[]
            if crs.get("status")=="declared":
                facts.append(f"坐标参考 {crs.get('authority') or '已声明'}")
            if field.get("engineering_mentions"):
                facts.append("查到工程文字引用")
            if field.get("declarations"):
                facts.append("查到字段属性")
            items.append(f"{field['field']} · {'、'.join(facts) or '未查到新增语义证据'}")
        text="；".join(items)
    elif "sources" in record:
        text="；".join(f"来源 {name} 抽样 {row['sampled']} 个，"
                       + (f"有限值约 {row['finite_percent']}%" if "finite_percent" in row else "样本不足")
                       + ("，有质量标记信息" if row["quality_flag_metadata_present"] else "")
                       for name,row in zip(("A","B"),record["sources"]))
    else:
        status=record.get("unit_status",record.get("relationship","unknown"))
        text=terms.get(status,"关系待核验")
    question=f"{record['question']} · " if record.get("question") else ""
    return f"{record['candidate']} · 自动核验 · {title} · {question}{text}"


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模拟技术引擎 | 环境数据工作台")
        self.resize(1200,790)
        self.scanner=Scanner()
        self.decision_model=None
        self.auto_run=False
        self.auto_phase=None
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
        title=QLabel("模拟技术引擎")
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
        self.path.setToolTip("选择、拖入或输入文件、压缩包、文件夹路径；点击开始后由DS按需检查")
        self.button(line,"选择文件",self.browse_file)
        self.button(line,"选择文件夹",self.browse_folder)
        self.setAcceptDrops(True)
        root.addLayout(line)
        self.current_issues=[]
        self.problem_panel=QFrame()
        self.problem_panel.setObjectName('recommendationCard')
        problems=QVBoxLayout(self.problem_panel)
        heading=QHBoxLayout()
        self.problem_title=QLabel('问题与待核实事项')
        self.problem_title.setWordWrap(True)
        heading.addWidget(self.problem_title,1)
        self.button(heading,'下载问题清单',self.export_issues)
        problems.addLayout(heading)
        self.problem_table=QTableWidget(0,4)
        self.problem_table.setHorizontalHeaderLabels(['检查阶段 / 对象','状态','具体问题或缺项','需要补充或核对'])
        self.problem_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.problem_table.setWordWrap(True)
        self.problem_table.setColumnWidth(0,180)
        self.problem_table.setColumnWidth(1,90)
        self.problem_table.setColumnWidth(2,450)
        self.problem_table.horizontalHeader().setStretchLastSection(True)
        self.problem_table.setMinimumHeight(100)
        self.problem_table.setMaximumHeight(190)
        problems.addWidget(self.problem_table)
        root.addWidget(self.problem_panel)
        self.problem_panel.hide()
        self.tabs=QTabWidget()
        root.addWidget(self.tabs,1)

        scan=QWidget()
        scanbox=QVBoxLayout(scan)
        hint=QLabel("选好数据和目标模型，点击「开始自动处理」：评估多源数据的处理价值，准备模型输入。DS按需检查，平台验证后执行；不预设融合必然更好。所需摘要和少量样例发送至DeepSeek，原始文件只读。")
        hint.setWordWrap(True)
        scanbox.addWidget(hint)
        self.target_grid_row=QWidget()
        target_layout=QHBoxLayout(self.target_grid_row)
        target_layout.setContentsMargins(0,0,0,0)
        target_layout.addWidget(QLabel('目标网格间距（纬度,经度；度）'))
        self.target_grid_input=QLineEdit()
        self.target_grid_input.setPlaceholderText('可选，例如 0.1,0.1；填写后作为硬性输出要求')
        self.target_grid_input.textChanged.connect(self.reset_target)
        target_layout.addWidget(self.target_grid_input,1)
        scanbox.addWidget(self.target_grid_row);self.target_grid_row.hide()
        self.workflow_status=QLabel("待开始 · 选择数据与目标模型即可")
        self.workflow_status.setWordWrap(True)
        scanbox.addWidget(self.workflow_status)
        self.basis_summary=page()
        self.basis_summary.setPlainText('决策依据将在检查后显示：数据事实 / 采用的约定 / 处理操作 / 验证结论。')
        self.basis_summary.setMinimumHeight(120)
        self.basis_summary.setMaximumHeight(200)
        evidence_frame=QFrame()
        evidence_frame.setObjectName('recommendationCard')
        evidence_box=QVBoxLayout(evidence_frame)
        evidence_box.setContentsMargins(14,10,14,10)
        evidence_title=QLabel('决策依据与执行链')
        evidence_title.setObjectName('recommendationTitle')
        evidence_box.addWidget(evidence_title)
        evidence_box.addWidget(self.basis_summary)
        scanbox.addWidget(evidence_frame)
        self.workflow_log=page()
        self.workflow_log.setPlaceholderText("运行后显示当前步骤、处理依据和需要补充的条件。无需手动扫描或编辑JSON。")
        scanbox.addWidget(self.workflow_log,1)
        self.audit_panel=QWidget()
        auditbox=QVBoxLayout(self.audit_panel)
        split=QSplitter()
        self.detail=page()
        self.detail.setPlaceholderText("本地扫描结果与双重核验结果")
        self.decision_json=QPlainTextEdit()
        self.decision_json.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.decision_json.setPlaceholderText("这里会显示可编辑的脱敏 JSON。请勿填写真实数值、路径、敏感字段名或工程原文。")
        self.local_explanation=page()
        self.local_explanation.setPlainText("选好数据和目标模型后点击「开始自动处理」，这里展示检查结果与依据。")
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
                self.recommend_json.setPlaceholderText("自动检查完成后，这里保留算法推荐请求供核查。")
                self.recommend_batch_index=self.add_batch_view(requests,self.recommend_json,"算法推荐 · 完整请求")
                views.addTab(requests,"查看待发 JSON")
            box.addWidget(views,1)
            split.addWidget(panel)
        split.setSizes([550,550])
        auditbox.addWidget(split,1)
        connection=QHBoxLayout()
        connection.addWidget(QLabel("DeepSeek 线上 API"))
        self.api_key=QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("已配置环境变量时无需填写；也可在此填写本次密钥")
        connection.addWidget(self.api_key,1)
        self.api_status=QLabel()
        connection.addWidget(self.api_status)
        auditbox.addLayout(connection)
        scanbox.addWidget(self.audit_panel,1)
        self.audit_panel.hide()
        actions=QHBoxLayout()
        self.scan_button=self.button(actions,"扫描数据",self.scan_now,primary=True)
        self.model_button=self.button(actions,"选择模型",self.choose_decision_model)
        actions.removeWidget(self.model_button)
        line.addWidget(self.model_button)
        actions.addStretch(1)
        self.start_button=self.button(actions,"开始自动处理",self.start_workflow,primary=True)
        self.decision_send=self.button(actions,"确认发送",self.send_decision,primary=True)
        self.ai_advice_button=self.button(actions,"查看算法方案",lambda:self.tabs.setCurrentIndex(1))
        self.ai_advice_button.setToolTip("查看推荐理由、勾选方案，再确认继续处理")
        self.scan_button.hide()
        self.decision_send.hide()
        self.ai_advice_button.hide()
        details=QCheckBox("查看技术记录与API设置")
        self.technical_toggle=details
        details.toggled.connect(self.audit_panel.setVisible)
        actions.addWidget(details)
        scanbox.addLayout(actions)
        self.tabs.addTab(scan,"1  数据与任务")

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
        self.recommendation_status=QLabel("点击首页「开始自动处理」后，此处自动显示处理方案及理由。")
        recbox.addWidget(self.recommendation_status)
        recbox.addWidget(cards,1)
        recactions=QHBoxLayout()
        recactions.addStretch()
        self.search_button=self.button(recactions,"按当前方案继续处理",self.continue_recommendations,primary=True)
        recbox.addLayout(recactions)
        self.tabs.addTab(recommend,"2  处理依据与方案")

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
        choice=QHBoxLayout()
        self.candidate_label=QLabel('其他已验证方案（可选）')
        choice.addWidget(self.candidate_label)
        self.candidate_select=QComboBox()
        self.candidate_select.setMinimumWidth(280)
        choice.addWidget(self.candidate_select,1)
        self.candidate_export=self.button(choice,'导出所选方案数据产品',self.export_candidate_now)
        rbox.addLayout(choice)
        self.candidate_note=QLabel()
        self.candidate_note.setWordWrap(True)
        self.candidate_note.setTextFormat(Qt.TextFormat.PlainText)
        rbox.addWidget(self.candidate_note)
        self.candidate_select.currentIndexChanged.connect(self.candidate_changed)
        self.set_candidate_options(None)
        rbox.addSpacing(7)
        footer=QHBoxLayout()
        footer.setContentsMargins(2, 0, 2, 7)
        footer.setSpacing(13)
        footer.addWidget(self.result_cards.output_check,1)
        self.export_button=self.button(footer,"下载结果与检查记录",self.export_now,primary=True)
        rbox.addLayout(footer)
        self.tabs.addTab(result,"3  结果与下载")
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
        from . import issues
        self.present_issues([*self.current_issues,*issues.exception(exc)])
        if isinstance(exc,json.JSONDecodeError):
            self.detail.appendPlainText('结构化资料解析失败：'+str(exc))
            exc=ValueError('结构化资料格式不完整，已暂停；详细原因可在技术记录中查看，未冒充处理成功。')
        self.auto_run=False
        self.auto_phase=None
        self.workflow_status.setText("已暂停 · "+str(exc))
        self.workflow_log.appendPlainText("需要处理："+str(exc))
        self.statusBar().showMessage("操作未完成")
        QMessageBox.warning(self,"操作未完成",str(exc))

    def present_issues(self,rows):
        from .issues import unique,bind_sources
        self.current_issues=bind_sources(unique(rows),getattr(self,'candidates',{}),getattr(self,'scan',None))
        self.problem_table.setRowCount(len(self.current_issues))
        for number,row in enumerate(self.current_issues):
            for col,value in enumerate((row['stage']+' / '+row['target'],row['status'],row['problem'],row['action'])):
                cell=QTableWidgetItem(value);cell.setToolTip(value+('\n'+'\n'.join(row.get('source_paths',[])) if col==0 else ''));self.problem_table.setItem(number,col,cell)
        self.problem_table.resizeRowsToContents()
        self.problem_title.setText(f'问题与待核实事项 · {len(self.current_issues)}项（已完成步骤保留）')
        self.problem_panel.setVisible(bool(self.current_issues))
        if self.current_issues:
            (Path(self.task.name)/'current_issues.json').write_text(s.dump(self.current_issues),encoding='utf8')

    def export_issues(self):
        if not self.current_issues:return
        path=QFileDialog.getSaveFileName(self,'保存具体问题与缺项清单','模拟技术引擎-问题清单.json','JSON (*.json)')[0]
        if path:
            target=Path(path).resolve()
            source=self.path.text().strip().strip('"')
            if source:
                original=Path(source).resolve()
                if target==original or original.is_dir() and target.is_relative_to(original):
                    raise ValueError('问题清单不能覆盖原始输入或写入本次原始数据目录，请另选保存位置。')
            target.write_text(s.dump(dict(issues=self.current_issues)),encoding='utf8')
            self.statusBar().showMessage('已保存问题清单')

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
        self.job.progress.connect(self.show_progress)
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
        self.auto_run=False
        self.auto_phase=None
        self.scan=None
        self.selected={}
        self.page_number=0
        self.reset_target()
        self.table.setRowCount(0)
        self.detail.clear()
        self.local_explanation.setPlainText("点击「开始自动处理」后，DS按需检查数据，平台展示判断依据。")
        self.decision_json.clear()
        self.candidates={}
        self.gated={}
        self.refresh_decision_send()
        self.page_label.clear()

    def reset_target(self,*_):
        self.present_issues([])
        self.set_candidate_options(None)
        self.original_recommendation_allowed=False
        self.checked_decisions=[]
        self.selected_recommendations=[]
        self.recommendation_rows=[]
        self.set_batch_view(self.recommend_batch_index,[])
        self.clear_recommendation_cards()
        self.recommendation_status.setText("点击首页「开始自动处理」后，此处自动显示处理方案及理由。")
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
        self.alg.hide()
        self.execute_button.hide()
        self.alg.clear()

    def model_changed(self,*_):
        self.role.clear()
        p=self.profile()
        if p: self.role.addItems([r["name"] for r in p["inputs"]])
        self.reset_target()

    def profile(self):
        return self.profiles.get(self.model.currentData())

    def browse_file(self):
        path=QFileDialog.getOpenFileName(self,"选择数据或工程压缩包")[0]
        if path:self.path.setText(path)

    def browse_folder(self):
        path=QFileDialog.getExistingDirectory(self,"选择数据与工程所在文件夹")
        if path:self.path.setText(path)

    def show_progress(self,message):
        self.statusBar().showMessage(message)
        if self.tabs.currentIndex()==2:
            self.result_cards.append_progress(message)
        if '次尝试' in message:
            self.detail.appendPlainText(message)
            self.workflow_status.setText('DS正在分析数据与工程依据，请稍候…')
            return
        self.workflow_status.setText(message)
        self.workflow_log.appendPlainText(message)

    def start_workflow(self):
        if self.thread is not None:return
        if not self.path.text().strip():
            self.browse_file()
            if not self.path.text().strip():return
        if self.decision_model is None:raise ValueError("请选择需要接收处理结果的目标模型")
        if not (self.mock.isChecked() or self.api_key.text().strip() or os.environ.get('DEEPSEEK_API_KEY','').strip()):
            self.technical_toggle.setChecked(True)
            raise ValueError("尚未配置DeepSeek密钥，请在连接设置中填写；已配置环境变量时无需重复填写")
        self.auto_run=True
        self.auto_phase='index'
        self.workflow_log.clear()
        self.show_progress("1/4 · DS检查准备：仅列目录，不预先深读全部数据")
        self.tabs.setCurrentIndex(0)
        self.scan_now()

    def continue_automatically(self,reply,checked):
        """UI chaining only: every recipe still comes from the shared checked plan."""
        from .issues import checks
        self.present_issues(checks(checked,reply.get('failures',[])))
        self.auto_run=False
        self.auto_phase=None
        if reply.get('failures'):
            self.show_progress("已暂停 · 部分检查未完成，请查看处理依据中的失败原因后重试")
            return
        executable=[(row,check) for row,check in zip(self.recommendation_rows,self.recommendation_checks)
                    if check is not None and check.isEnabled() and not row.get('advisory_only')]
        if not executable:
            if self.component_confirmation_ready():
                self.search_button.setEnabled(True)
                self.tabs.setCurrentIndex(1)
                self.recommendation_status.setText('已识别独立模型组件转换候选；无需完整工程或无关变量。继续后由公共写出器检查最低必要条件，尚未标记转换通过。')
                self.show_progress('2/4 · 无需融合选型；请确认继续组件转换，缺少必要科学信息时将明确说明')
                return
            if not self.current_issues:
                from .issues import item
                self.present_issues([item('算法匹配',None,'当前没有可执行方案，后台未提供具体缺项；不能判断是缺文件、元数据还是暂不支持。','查看处理依据与检查记录；此状态不证明数据损坏。')])
            self.show_progress("需要补充条件 · 暂无通过核验的可执行方案，详见处理依据；尚未运行融合")
            self.tabs.setCurrentIndex(1)
            return
        engineering=[row for row,_ in executable if row['steps'][0].startswith('engineering.')]
        if engineering and len(engineering)!=len(executable):
            self.show_progress("需要选择处理范围 · 整包工程与独立数据任务不能混用，请在处理依据页选择一类")
            self.tabs.setCurrentIndex(1)
            return
        # Never authorize a whole-package continuation if another checked task
        # remains blocked. The existing execution service rechecks contracts.
        self.update_task_checks(checked)
        if engineering and not self.original_recommendation_allowed:
            self.show_progress("需要补充条件 · 整包工程仍包含未放行任务，不能自动执行；请查看处理依据")
            self.tabs.setCurrentIndex(1)
            return
        for _,check in executable:check.setChecked(True)
        self.tabs.setCurrentIndex(1)
        self.show_progress("2/4 · 推荐方案已准备，请查看理由并勾选方案，点击「按当前方案继续处理」后再执行")

    def update_task_checks(self,checked):
        self.checked_decisions=[dict(row) for row in checked]
        self.original_recommendation_allowed=bool(self.checked_decisions) and not discovery.blocking_checks(self.checked_decisions)

    def require_engineering_checks(self):
        self.update_task_checks(self.checked_decisions)
        if self.original_recommendation_allowed:return
        blockers=discovery.blocking_checks(self.checked_decisions)
        details='；'.join(f"{row.get('candidate','任务')} / {row.get('action','检查')}：{row.get('reason') or row.get('status','未确认')}" for row in blockers[:5])
        error=ValueError('本次工程暂不能继续：'+(details or '没有当前数据的有效核验记录，请从首页重新开始检查。'))
        error.checked_decisions=self.checked_decisions
        raise error

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
        if batch is not None and index<=len(entries):
            phase={1:"文件分类",2:"完整资料判断",3:"集中追问回传"}.get(event.get("phase"),"请求")
            title=f"{phase} · 第 {index+1} 批"
            if index==len(entries):
                entries.append((title,batch))
                selector.addItem(title)
            else:
                entries[index]=(title,batch)
                selector.setItemText(index,title)
                if selector.currentIndex()==index:
                    self.render_batch(selector,index)
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
            if self.auto_run and self.auto_phase=='prepare':
                if self.decision_send.isEnabled():
                    self.auto_phase='analyze'
                    self.show_progress("2/4 · DS按需查表头、工程依赖并提出方案；平台检查真实契约")
                    self.send_decision()
                else:
                    self.auto_run=False
                    self.auto_phase=None
                    self.show_progress("需要补充条件 · "+(self.batch_error or '数据请求、目标模型或API连接未就绪'))

        def prepare(_):
            try:
                draft=json.loads(text)
                if isinstance(draft,dict) and draft.get("stage")=="decision" and draft.get("catalog_first") is True and isinstance(draft.get("sources"),list):
                    from .three_phase import classification_request
                    classified=classification_request(self.scan,draft)
                    first=request_batches.Batch(s.preview(classified),tuple(row["id"] for row in classified["files"]))
                    return [first],""
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                return [],str(error)
            return [],""

        self.work(prepare,finish)

    def refresh_decision_send(self):
        has_key=bool(self.api_key.text().strip() or os.environ.get("DEEPSEEK_API_KEY","").strip())
        mock=self.mock.isChecked()
        self.api_status.setText("模拟模式 · 不联网" if mock else ("已连接密钥 · 开始后按需调用" if has_key else "未配置密钥"))
        try:
            draft=json.loads(self.decision_json.toPlainText())
            valid=isinstance(draft,dict) and draft.get("stage")=="decision" and draft.get("catalog_first") is True and isinstance(draft.get("sources"),list)
        except ValueError:
            draft={}
            valid=False
        self.decision_send.setEnabled(self.scan is not None and valid and any(not batch.blocked for batch in self.decision_batches)
                                      and self.decision_model is not None and (has_key or mock))
        lines=["开始后自动完成目录检查、按需读取与方案核验；在第二步确认方案后执行处理。",
               "先发每个文件的结构与脱敏片段。DeepSeek 给出分类后，平台发送完整字段目录、变量引用和对应的脱敏工程说明。",
               "证据足够时直接生成算法组合。证据不足时，DeepSeek 一次列出全部问题，平台补充指定字段的完整元数据和随机小数扰动样本。",
               "原始观测序列、精确时刻与位置留在本机。各阶段优先整包发送，官网明确拒绝容量时自动分批。"]
        if self.scan is not None and self.scan.get('on_demand'):
            lines=["开始后，DeepSeek在本次输入范围内按需列目录、读表头、查说明和工程依赖。",
                   "所请求的元数据、文本片段和少量表格样例将发送至DeepSeek；未请求文件不深读。",
                   "每次只读工具在独立子进程执行，临时文件随后清理；工具与依据保留在决策记录中。"]
        if self.decision_model is None:
            lines.append("请在首页选择目标模型，然后开始自动处理。")
        else:
            lines.append(f"目标模型：{discovery.MODELS[self.decision_model]}。更换模型会重新生成待审核内容。")
        if valid:
            if self.decision_batches:
                lines.append("可在「融合判断 · 实际分批」查看各阶段的请求和反馈。")
                for batch in self.decision_batches:
                    if batch.blocked:
                        lines.append(f"待处理 · {'、'.join(batch.ids)} · {batch.blocked}")
            elif self.batch_error:
                lines.append(f"生成待发批次失败：{self.batch_error}")
            if self.scan is not None and self.scan.get("shallow"):
                lines.append("当前仅列出输入目录，待DeepSeek按需检查。" if self.scan.get('on_demand') else
                             f"已浅读 {len(self.scan['rows'])} 个文件，完整字段和工程说明在分类后读取。")
            documents=draft.get("engineering_documents",[])
            modes={"full":"脱敏全文","mixed":"保留说明，替换原始数据",
                   "data":"仅发数据结构","unknown":"用途待确认，仅发结构","unread":"未读取正文"}
            if isinstance(documents,list):
                readable=[doc for doc in documents if isinstance(doc,dict)]
                complete=sum(doc.get("sharing")=="full" for doc in readable)
                sources=draft.get("sources",[])
                count=sum(len(source.get("fields",[])) for source in sources)
                if sources or readable:
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
                "preprocess":"统一统计窗口及换算","direct":"分别使用","insufficient":"依据不足","reject":"明确反对处理","pending_goal":"处理目标待确认"}
        self.operation_labels=labels
        count=sum(bool(row["fields"]) for row in self.scan["rows"])
        model_name=discovery.MODELS.get(self.decision_model,"尚未选择")
        lines=[f"目标模型：{model_name}。",(f"已浅读 {len(self.scan['rows'])} 个成员，由 DeepSeek 在自动流程中按需检查。"
                if self.scan.get("shallow") else f"已登记 {len(self.scan['rows'])} 个成员，读取到 {count} 份数据的字段信息。"),
               (f"DeepSeek 提出 {len(self.candidates)} 组候选。" if reply is not None
                else "DeepSeek 在自动流程中根据实际字段匹配处理候选。")]
        if self.scan.get('on_demand'):
            lines[1] = ("当前仅列目录；自动流程将由DS按需读取表头、配置及依赖。" if self.scan.get('shallow') else
                        f"已按需检查 {len(self.scan['rows'])} 份文件，其余文件尚未读取。")
            if reply is None:
                lines[2] = "DeepSeek 将根据工具实际读取的证据提出候选，未知信息保留待核验。"
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
            lines.append("已识别工程任务；通过契约检查后由自动流程继续处理。")
        if self.decision_model=="cmaq":
            from .cmaq_grid_audit import explain
            lines.extend(explain(self.scan.get("cmaq_grid_checks",[])))
        field_uses=reply.get("field_uses",{}) if reply is not None else {}
        for cid,(left,right,actions) in self.candidates.items():
            sources=[]
            for index,fid,field in (left,right):
                row=self.scan["rows"][index]
                roles="、".join(discovery.ROLE_LABELS[r] for r in row["roles"] if r in discovery.ROLE_LABELS)
                if not roles and fid in field_uses:
                    use=field_uses[fid]
                    roles=f"{use['use']}；依据：{use['evidence']}"
                if not roles:
                    if reply is None:
                        roles="待 DeepSeek 判断用途"
                    elif reply.get("failures"):
                        roles="用途判断未完成"
                    else:
                        roles="用途待核验"
                sources.append(f"{PurePosixPath(row['name']).name} · {field['name']}（{roles}）")
            lines.append(f"{cid}  {' 与 '.join(sources)}\n  可核验操作：{'、'.join(labels[a] for a in actions)}")
        if reply is None:
            lines.append("DeepSeek给出建议后，平台核对单位、语义、时空覆盖和算法输入条件；缺少依据会明确暂停。")
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
                    result.append("本次有未完成检查；保留失败原因，不自动无限重试。")
            if not passed and any(row["status"]=="待确认用途" for row in checked):
                result.append("本次处理目标仍待确认。平台会另行核验可比较的分辨率差异。")
            elif not passed and not planned:
                result.append("当前没有可进入算法推荐的任务。")
            if passed or planned:
                result.append("判断与算法建议已生成，详见处理依据页。")
            from .decision_checks import CHECKS
            for record in reply.get("local_checks",[]):
                result.append(render_local_check(record,CHECKS))
            for decision in decisions:
                action=labels.get(decision["action"],decision["action"])
                goal=decision.get("goal")
                purpose=discovery.TASK_GOALS.get(goal,"待定")
                author="平台（对话状态）" if decision.get("origin")=="platform" else "DeepSeek"
                result.append(f"{decision['candidate']} · {author}：{action}。任务目标：{purpose}。{decision['reason'] or '未给出理由'}")
            for local in checked:
                action=labels.get(local["action"],local["action"])
                status="处理目标待确认" if local["status"]=="待确认用途" else local["status"]
                result.append(f"{local['candidate']} · 平台核验{action}：{status}。{local['reason']}")
                if review:=local.get("fusion_review"):
                    verdict={"uphold":"认可拦截","retain":"保留融合建议","uncertain":"证据仍不足",
                             "unavailable":"未取得有效复核"}[review["verdict"]]
                    result.append(f"{local['candidate']} · DeepSeek 复核：{verdict}。{review['reason']}")
            lines=result+lines
        self.local_explanation.setPlainText("\n\n".join(lines))

    def choose_decision_model(self):
        menu=QMenu(self)
        for model_id,name in discovery.MODELS.items():
            if model_id not in self.profiles and model_id!='data_only':
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
        self.target_grid_row.setVisible(model_id=='data_only')
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
        requirements=self.task_requirements()
        def scan_job(log):
            log("正在列出输入目录，等待 DeepSeek 按需检查…")
            from .file_agent import scan_index
            scanned,metrics=scan_index(path,progress=log)
            log("正在识别候选并整理脱敏工程资料…")
            summary,candidates=discovery.discover(scanned,self.decision_model,task_goal='evaluate_gain')
            summary['task_requirements']=requirements
            return scanned,metrics,summary,candidates
        self.work(scan_job,self.show_scan)

    def task_requirements(self):
        if self.decision_model!='data_only' or not self.target_grid_input.text().strip():return {}
        from ecodata.spatial.regridding import target_grid_spec
        parts=re.split(r'[,，×x\s]+',self.target_grid_input.text().strip())
        if len(parts)!=2:raise ValueError('目标网格请填写纬度和经度两个间距，例如0.1,0.1；单位为度')
        return dict(target_grid=target_grid_spec(dict(crs='EPSG:4326',latitude_step=float(parts[0]),longitude_step=float(parts[1]),extent='intersection')))

    def show_scan(self,value):
        scan,self.scan_metrics,summary,candidates=value
        automatic=self.auto_run
        self.reset_path()
        self.auto_run=automatic
        self.auto_phase='prepare' if automatic else None
        self.scan=scan
        context=scan.get('saved_input_context')
        if context:
            self.basis_summary.setPlainText('历史上下文 · 原包SHA256匹配\n证据状态 · '+context['status']+'\n'+context.get('explanation',''))
            self.workflow_log.appendPlainText(self.basis_summary.toPlainText())
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

    def component_confirmation_ready(self):
        report=(self.scan or {}).get('native_model_probes',{}).get(self.decision_model,{})
        return bool(report.get('component_candidate') and not self.candidates and not discovery.blocking_checks(self.checked_decisions))

    def update_original_background(self):
        if self.original_background_ready():
            self.recommendation_status.setText('已识别组件转换候选；确认继续后核验最低必要数据。' if self.component_confirmation_ready()
                                               else '已识别工程任务；查看方案与检查依据后确认继续。')
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
            from .three_phase import run
            reply,checked,gated,candidates,approved,full_scan=run(scan,summary,self.batch_replies,mock,key,log,feedback)
            if full_scan.get('file_agent_audit'):
                (Path(self.task.name)/'file-inspection.json').write_text(s.dump(dict(
                    inspection=full_scan['file_agent_audit'], rows=full_scan['rows'])), encoding='utf8')
            eligible=any(row["status"]=="通过" or row.get("next_step") in {"original_workbench","recommendation_only","alignment_then_fusion"}
                         for row in checked)
            rows=[]
            if eligible:
                plan=recommendations.make_request(full_scan,approved,checked,gated,
                                                  local_checks=reply.get("local_checks",[]),include_samples=False)
                rows,receipt=recommendations.recommend_plan(plan,reply['decisions'],mock=mock,api_key=key,progress=log)
                log("正在本地复查留出的数据窗口…")
                recommendations.review_holdouts(full_scan,approved,plan,rows,gated)
                reply['algorithm_recommendation']=receipt
                reply['api_receipts']=[*reply.get('api_receipts',[]),*receipt['api_receipts']]
                reply['batch_count']=len(reply['api_receipts'])
                (Path(self.task.name)/'algorithm-recommendation.json').write_text(s.dump(dict(plan=plan,receipt=receipt,rows=rows)),encoding='utf8')
            return reply,checked,gated,candidates,rows,full_scan

        def finish(value):
            reply,checked,gated,candidates,rows,full_scan=value
            self.scan=full_scan
            self.fill_page()
            self.candidates=candidates
            self.gated=gated
            self.update_task_checks(checked)
            self.show_local_explanation(reply,checked)
            self.detail.setPlainText(s.dump(dict(
                DeepSeek=reply, 本地核验=checked, 文档核对=full_scan.get('document_evidence_checks',[]), 待选算法=rows,
                扫描依据=discovery.local_overview(full_scan,candidates))))
            self.render_recommendations(dict(reply,recommendations=rows))
            facts=[f"已按需读取 {len(full_scan['rows'])} 份资料；形成 {len(candidates)} 组候选，{len(rows)} 项建议。"]
            for source in [v for v in full_scan['rows'] if any(not f.get('axis') for f in v['fields'])][:3]:
                fields=[f"{f['name']}（{f.get('unit') or '单位未确认'}）" for f in source['fields'] if not f.get('axis')][:2]
                facts.append('数据事实 · '+PurePosixPath(source['name']).name+' · '+'、'.join(fields))
            for item in checked[:3]:facts.append(f"{item['candidate']} · {self.operation_labels.get(item['action'],item['action'])} · {item['status']}：{item['reason'][:160]}")
            heldout=[result for row in rows for result in row.get('local_holdout_review',[])]
            if heldout:
                counts={status:sum(item['status']==status for item in heldout) for status in ('checked','pending','failed')}
                facts.append(f"本地留出回检 · 已核验 {counts['checked']} · 待核实 {counts['pending']} · 失败 {counts['failed']}")
            if full_scan.get('saved_input_context'):facts.append('历史约定 · '+full_scan['saved_input_context'].get('explanation',''))
            self.basis_summary.setPlainText('\n'.join(facts))
            self.ai_advice_button.setEnabled(bool(rows))
            self.update_original_background()
            self.tabs.setCurrentIndex(1 if rows and not reply["failures"] else 0)
            self.workflow_log.appendPlainText(self.local_explanation.toPlainText())
            if self.auto_run:self.continue_automatically(reply,checked)
        self.work(decide,finish,feedback=lambda event:self.show_batch_reply(self.decision_batch_index,event))

    def clear_recommendation_cards(self):
        self.recommendation_checks.clear()
        while self.recommend_cards.count():
            item=self.recommend_cards.takeAt(0)
            widget=item.widget()
            widget.hide()
            widget.deleteLater()

    def add_recommendation_card(self,rows):
        from .release_gate import allowed
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
                check.setEnabled(allowed(row))
                layout.addWidget(check)
            if row.get("execution_release") and not allowed(row):
                from .release_gate import explanation
                from .fallback_evidence import approved_fields, eligible
                gate = QLabel(explanation(row["execution_release"]))
                gate.setWordWrap(True)
                gate.setTextFormat(Qt.TextFormat.PlainText)
                layout.addWidget(gate)
                if eligible(row) and self.scan and approved_fields(self.scan, row):
                    button = QPushButton("补充样本或审核全量副本")
                    button.clicked.connect(lambda checked=False, item=row: self.safe(lambda: self.review_full_fallback(item)))
                    layout.addWidget(button)
                if row.get("independent_required"):
                    validation_button = QPushButton("选择新的独立数据复验")
                    validation_button.clicked.connect(lambda checked=False, item=row: self.safe(lambda: self.review_new_independent_data(item)))
                    layout.addWidget(validation_button)
                if row["execution_release"]["status"] == "failed" and row.get("local_recipe_options"):
                    adjustment = QPushButton("调整候选并重新复验")
                    adjustment.clicked.connect(lambda checked=False, item=row: self.safe(lambda: self.adjust_review_candidate(item)))
                    layout.addWidget(adjustment)
                if self.scan and self.scan.get("engineering_docs"):
                    document_button = QPushButton("复查说明章节或观测表")
                    document_button.clicked.connect(lambda: self.safe(self.review_source_document))
                    layout.addWidget(document_button)
            reason=QLabel("推荐理由：" + (row["reason"].strip() or "DeepSeek 未提供理由。"))
            reason.setTextFormat(Qt.TextFormat.PlainText)
            reason.setWordWrap(True)
            layout.addWidget(reason)
            explanation=QPlainTextEdit()
            explanation.setReadOnly(True)
            members = row.get("execution_release", {}).get("members", ())
            explanation.setPlainText(recommendations.explanation_text(row)
                + ("\n工程成员放行与外发状态\n" + s.dump(members) if members else ""))
            explanation.setMaximumHeight(230)
            explanation.setAccessibleName('数据依据与推荐边界')
            explanation.hide()
            toggle=QPushButton('展开数据依据与检查前提')
            toggle.setCheckable(True)
            toggle.toggled.connect(explanation.setVisible)
            layout.addWidget(toggle)
            layout.addWidget(explanation)
            self.recommendation_checks.append(check)
        self.recommend_cards.addWidget(card)

    def render_recommendations(self, reply):
        from .release_gate import allowed
        self.clear_recommendation_cards()
        rows=reply["recommendations"]
        groups={}
        for row in rows:
            group=(row["task"],row["id"]) if row.get("advisory_only") else tuple(row["steps"])
            groups.setdefault(group,[]).append(row)
        self.recommendation_rows=[row for group in groups.values() for row in group]
        for group in groups.values():
            self.add_recommendation_card(group)
        self.search_button.setEnabled(self.original_background_ready() or
                                      any(allowed(row) for row in rows))
        executable=sum(any(allowed(row) for row in group) for group in groups.values())
        self.recommendation_status.setText(f"{reply['mode']} · {executable} 种方案可勾选，"
                                            f"{len(groups)-executable} 种待核验"
                                            + ("\n"+usage if (usage:=request_batches.describe_usage(reply)) else ""))

    def review_full_fallback(self, recommendation):
        from uuid import uuid4
        from . import fallback_evidence as fallback

        scan = self.scan
        fields = fallback.approved_fields(scan, recommendation)
        if not fallback.eligible(recommendation) or not fields:
            raise ValueError("目前缺少科学依据或独立参考，请先补齐原始资料")
        key = self.api_key.text().strip()
        if not key or self.mock.isChecked():
            raise ValueError("发送补充证据需要已连接的 DeepSeek API")
        members = {scan["rows"][index]["name"]: index for index, _ in fields}
        if len(members) > 1 and any("!/" in name for name in members):
            name, ok = QInputDialog.getItem(self, "选择需要补样的 ZIP 成员", "授权范围",
                                          ["全部获准成员", *members], 0, False)
            if not ok:
                return
            if name in members:
                fields = [item for item in fields if item[0] == members[name]]
        targets = {}
        tables = {scan["rows"][index]["name"]: index for index, _ in fields
                  if scan["rows"][index]["format"] in {"CSV", "TSV", "Excel"}}
        if tables:
            name, ok = QInputDialog.getItem(self, "定向补样", "选择站点或工作表所在文件",
                                          ["按当前覆盖范围补样", *tables], 0, False)
            if not ok:
                return
            if name in tables:
                index = tables[name]
                choices = ["站点", "工作表"] if scan["rows"][index]["format"] == "Excel" else ["站点"]
                mode, ok = QInputDialog.getItem(self, "定向补样", "分组方式", choices, 0, False)
                if not ok:
                    return
                group, ok = QInputDialog.getText(self, "定向补样", f"输入{mode}名称，只用来在本地定位")
                if not ok or not group.strip():
                    return
                targets[index] = ("station" if mode == "站点" else "sheet", group.strip())
        archive = Path(self.task.name) / f"outbound-evidence-{uuid4().hex}.zip"
        selected_classification = []

        def build_copy():
            classifications = fallback.classification_fields(scan, fields)
            if classifications:
                options = {f"{scan['rows'][index]['name']} · {field['name']}": (index, field)
                           for index, field in classifications}
                selected, ok = QInputDialog.getItem(self, "分类波段授权", "原样保留的分类波段",
                    ["仅连续数值", *options], 0, False)
                if not ok:
                    return
                if selected in options:
                    selected_classification.append(options[selected])
            self.work(prepare, confirm)

        def prepare(log):
            if self.scan is not scan:
                raise ValueError("当前任务已变化")
            entries = fallback.build(scan, [*fields, *selected_classification], archive, log)
            return entries, fallback.digest_file(archive)

        def confirm(value):
            entries, digest = value
            count = sum(item["records"] for item in entries)
            size = sum(item["bytes"] for item in entries)
            batches = sum(len(item["parts"]) for item in entries)
            scope = "\n".join(f"{scan['rows'][index]['name']} · {field['name']}（"
                              + ("分类编码原样" if field.get("approved_classification") else str(field.get("unit"))) + "）"
                              for index, field in [*fields, *selected_classification])
            answer = QMessageBox.question(self, "审核全量扰动证据",
                f"准备向 api.deepseek.com 分批发送 {count:,} 条获准数值记录，约 {size:,} 字节，{batches:,} 次请求。\n\n"
                f"{scope}\n\n副本保留必要的网格或时间区间信息，省略站点编号和非选定字段。"
                "随机扰动不提供正式的隐私保证。全量范围需重新取得独立回检证据。\n\n是否批准发送这个副本？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                archive.unlink(missing_ok=True)
                self.show_progress("全量扰动证据未发送")
                return

            from .independent_review import fields as bound_fields, invalidate

            def identity(index, field):
                return index, field["name"], field.get("sheet"), field.get("band")

            approved_sources = {identity(index, field) for index, field in [*fields, *selected_classification]}
            outbound = scan.setdefault("member_outbound", {})
            for item in entries:
                outbound[item["source_member"]] = "approved_pending_transmission"
            def overlaps(item):
                return any(identity(index, field) in approved_sources for index, _, field in bound_fields(scan, item))

            def existing_task(item):
                return self.gated.get(tuple(item["task"].split(":", 1)))

            for item in self.recommendation_rows:
                if overlaps(item):
                    invalidate(scan, item, existing_task(item))
            self.render_recommendations(dict(recommendations=list(self.recommendation_rows), mode="已批准外发范围"))
            archive.with_suffix(".approval.json").write_text(s.dump(dict(
                digest=digest, fields=entries, trigger="补样覆盖不足", records=count,
                bytes=size, requests=batches, approved=True)), encoding="utf8")

            def transmit(log):
                assert_current(scan)
                return fallback.transmit(archive, entries, digest, api_key=key, progress=log,
                                         recipes=recommendation.get("local_recipe_options", ()))

            def sent(result):
                from .independent_review import start
                for item in entries:
                    outbound[item["source_member"]] = "transmitted"
                candidate = start(recommendation, result)
                rows = []
                for item in self.recommendation_rows:
                    if candidate is not None and item["task"] == recommendation["task"]:
                        continue
                    if overlaps(item):
                        invalidate(scan, item, existing_task(item))
                    rows.append(item)
                if candidate is not None:
                    invalidate(scan, candidate, existing_task(candidate))
                    rows.append(candidate)
                self.render_recommendations(dict(recommendations=rows, mode="全量分片复核"))
                self.show_progress("已形成新候选，等待独立数据复验" if candidate else
                                   "已收齐分片回复，各分片依据尚不足以形成一致候选")
                self.detail.appendPlainText(s.dump(dict(分片回执=str(result["receipts"]), 候选支持=result["votes"],
                                                        形成候选=result.get("candidate"))))

            self.work(transmit, sent)

        def consider(samples):
            supplemental = [item for item in samples if item["perturbed_samples"]]
            if not supplemental:
                build_copy()
                return
            answer = QMessageBox.question(self, "先补充小样本",
                "已找到仍在推荐范围内、未发送的少量记录。先向 DeepSeek 发送这些扰动样本再判断？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return

            def send_supplement(log):
                messages = [
                    dict(role="system", content='仅判断额外小样本是否解决覆盖问题，返回 JSON 键 needs_more（布尔值）和 reason。'),
                    dict(role="user", content=s.dump(dict(samples=supplemental,
                        blockers=[dict(reason=item["reason"], format=item.get("format"))
                                  for item in recommendation["execution_release"]["blockers"]]))),
                ]
                message, receipt = s.tool_turn(messages, [], key, log)
                answer = json.loads(message.get("content") or "{}")
                if type(answer.get("needs_more")) is not bool:
                    raise ValueError("DeepSeek 未明确说明补样是否充足")
                return answer, receipt

            def reviewed(value):
                answer, receipt = value
                for index, _ in fields:
                    scan.setdefault("member_outbound", {})[scan["rows"][index]["name"]] = "supplement_reviewed"
                self.detail.appendPlainText(s.dump(dict(补样结论=answer, 回执=receipt)))
                if answer["needs_more"]:
                    build_copy()
                else:
                    self.show_progress("补样已完成 · 请核对回复并用独立数据复验")

            self.work(send_supplement, reviewed)

        self.work(lambda _: fallback.supplement(scan, fields, targets), consider)

    def review_new_independent_data(self, candidate):
        from .independent_review import apply, fields, review

        scan = self.scan
        entries = {f"{fid} · {scan['rows'][index]['name']} · {field['name']}": fid
                   for index, fid, field in fields(scan, candidate)}
        if not entries:
            raise ValueError("该候选没有可复验的字段")
        entry, accepted = QInputDialog.getItem(self, "独立复验", "选择要复验的字段", list(entries), 0, False)
        if not accepted:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择未用于推荐或外发的独立数据")
        if not path:
            return
        answer = QMessageBox.question(self, "确认独立数据来源",
            "确认这份资料的观测记录没有用于之前的推荐、调参或全量外发。该文件只用于本地复验。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        task = self.gated.get(tuple(candidate["task"].split(":", 1)))

        def reviewed(report):
            if self.scan is not scan:
                return
            apply(scan, candidate, task, report)
            self.render_recommendations(dict(recommendations=list(self.recommendation_rows), mode="独立复验"))
            self.detail.appendPlainText(s.dump(dict(独立复验=report, 放行=candidate["execution_release"])))

        self.work(lambda _: review(scan, candidate, task, entries[entry], path), reviewed)

    def adjust_review_candidate(self, previous):
        from .independent_review import fields
        from .release_gate import assess

        options = {f"{option['id']} · {option['name']}": option for option in previous["local_recipe_options"]
                   if option["id"] != previous["id"]}
        if not options:
            raise ValueError("当前任务没有其他已登记的候选方案")
        name, accepted = QInputDialog.getItem(self, "调整候选", "从已登记方案选择", list(options), 0, False)
        if not accepted:
            return
        choice = options[name]
        candidate = dict(previous, **{key: choice[key] for key in ("id", "name", "steps", "description")})
        candidate.update(independent_required=True, local_holdout_review=[],
                         previous_independent_hashes=list(dict.fromkeys([
                             *previous.get("previous_independent_hashes", ()),
                             *(item["independent_sha256"] for item in previous.get("local_holdout_review", ())
                               if item.get("independent_sha256"))])))
        task = self.gated.get(tuple(candidate["task"].split(":", 1)))
        candidate["execution_release"] = assess(self.scan, candidate, task, fields(self.scan, candidate))
        rows = [candidate if row is previous else row for row in self.recommendation_rows]
        self.render_recommendations(dict(recommendations=rows, mode="候选已调整"))
        self.detail.appendPlainText(s.dump(dict(上一轮失败=previous["execution_release"], 新候选=choice["id"])))

    def review_source_document(self):
        scan = self.scan
        documents = list(dict.fromkeys(name for name, _ in scan.get("engineering_docs", ())))
        if not documents:
            raise ValueError("目前没有可复查的工程说明")
        name, ok = QInputDialog.getItem(self, "复查工程说明", "选择原始文件", documents, 0, False)
        if not ok:
            return
        action, ok = QInputDialog.getItem(self, "读取范围", "操作",
            ["复查相关章节或页面", "提取观测表到本地 CSV"], 0, False)
        if not ok:
            return
        section, ok = QInputDialog.getInt(self, "读取范围",
            "PDF 页码、Word 表格号或段落批次、文本分页，从 1 开始", 1, 1, 10000)
        if not ok:
            return
        root = Path(scan["path"])
        locator = name.removeprefix(root.name + "!/") if root.is_file() else name
        if root.is_file() and name == root.name:
            locator = ""
        extension = PurePosixPath(name).suffix.lower()

        def run(_):
            from . import document_evidence, file_tools
            assert_current(scan)
            if action == "提取观测表到本地 CSV":
                with file_tools.selected(root, locator) as (path, _, __):
                    table = document_evidence.observation_table(path, section)
                assert_current(scan)
                return {"table": table}
            offset = section - 1
            if extension == ".docx":
                offset *= 20
            elif extension != ".pdf":
                offset *= 16384
            if extension in {".pdf", ".docx"}:
                with file_tools.selected(root, locator) as (path, _, __):
                    evidence = document_evidence.extract(path, extension, offset, prefer_outline=False)
            else:
                evidence = file_tools.inspect(root, "read_text", dict(path=locator, offset=offset))
            assert_current(scan)
            checks = document_evidence.cross_check(dict(scan, engineering_docs=[(name, evidence["text"])]))
            return {"evidence": evidence.get("evidence", []), "checks": checks}

        def reviewed(result):
            if "table" in result:
                self.save_document_observations(scan, result["table"])
                return
            self.apply_document_review(scan, result["checks"])
            self.detail.appendPlainText(s.dump(dict(复核原文=name, **result)))
            self.show_progress("本地说明复查完成 · 查看检查记录及原文位置")

        self.work(run, reviewed)

    def apply_document_review(self, scan, updates):
        from .release_gate import assess, resolve_field, task_fields

        if self.scan is not scan or not updates:
            return
        def identity(record):
            return tuple(record.get(key) for key in ("document", "check", "reference", "data", "variable"))
        merged = {identity(record): record for record in scan.get("document_evidence_checks", ())}
        merged.update({identity(record): record for record in updates})
        scan["document_evidence_checks"] = list(merged.values())
        for recommendation in self.recommendation_rows:
            if "execution_release" not in recommendation:
                continue
            fields = [(index, fid, field) for index, name, fid in
                      task_fields({"fields": recommendation.get("execution_fields", ())})
                      if (field := resolve_field(scan["rows"][index], index, name, fid))]
            task = self.gated.get(tuple(recommendation["task"].split(":", 1)))
            recommendation["execution_release"] = assess(scan, recommendation, task, fields)
        self.render_recommendations(dict(recommendations=list(self.recommendation_rows), mode="本地复核"))

    def save_document_observations(self, scan, table):
        import csv
        location, _ = QFileDialog.getSaveFileName(self, "保存复查用观测表", "观测表.csv", "CSV (*.csv)")
        if not location:
            return
        destination, source = Path(location).resolve(), Path(scan["path"]).resolve()
        if destination == source or source.is_dir() and destination.is_relative_to(source):
            raise ValueError("请将提取表保存到本次输入目录之外")
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(table)
        self.show_progress("已保存观测表 · 可转入 CSV 核验")
        proceed = QMessageBox.question(self, "转入表格核验",
            "已经保存原表日期、表头、单位和数值。确认表格结构后，是否用 CSV 流程开始核验这份观测表？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if proceed == QMessageBox.StandardButton.Yes:
            self.path.setText(str(destination))
            self.scan_now()

    def continue_recommendations(self):
        from .release_gate import require
        selected=[row for row,check in zip(self.recommendation_rows,self.recommendation_checks)
                  if check is not None and check.isChecked()]
        if not selected:
            if self.original_background_ready():
                if not self.component_confirmation_ready():self.require_engineering_checks()
                if self.scan.get("research_tasks",{}).get(self.decision_model):
                    self.selected_recommendations=[]
                    self.tabs.setCurrentIndex(2)
                    self.execute_selected()
                else:
                    self.execute_original()
                return
            raise ValueError("请至少勾选一个推荐方案")
        assert_current(self.scan)
        require(selected)
        engineering=[row for row in selected if row["steps"][0].startswith("engineering.")]
        if engineering and len(engineering)!=len(selected):
            raise ValueError("原版整包工程和逐项寻优请分开选择")
        if engineering:self.require_engineering_checks()
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
        if self.scan.get('on_demand') and self.scan.get('shallow'):
            count=self.scan.get('initial_directory',{}).get('total',0)
            self.page_label.setText(f"目录包含 {count} 个条目；自动流程中由DS按需读取")

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
        self.work(lambda log:s.send(approved,approved.digest,mock,api_key=key,on_progress=log),finish)

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
        self.set_candidate_options(None)
        self.result_cards.reset("正在本地处理，请稍候…")
        state=self.prepared
        model=(state.get("profile") or {}).get("id")
        self.work(lambda log:s.run(state,folder,confirmed,log),lambda value:self.show_result(value,model))

    def execute_selected(self):
        from .selected_processing import run
        from .release_gate import require, require_whole_scope
        assert_current(self.scan)
        require(self.selected_recommendations)
        if not self.selected_recommendations:
            require_whole_scope(self.recommendation_rows)
        self.alg.hide()
        self.execute_button.hide()
        recipes={}
        for row in self.selected_recommendations:
            recipes.setdefault(row["task"],[]).append(row["steps"])
        tasks=[(task_id,dict(self.gated[tuple(task_id.split(":",1))],recipes=steps,
                             runtime_fields=list(dict.fromkeys(
                                 field for row in self.selected_recommendations if row["task"] == task_id
                                 for field in row.get("execution_fields", ())))))
               for task_id,steps in recipes.items()]
        if self.run_temp is not None:
            self.run_temp.cleanup()
        self.run_temp=TemporaryDirectory(dir=self.task.name,prefix="selected-")
        folder=Path(self.run_temp.name)
        scan,model_id=self.scan,self.decision_model
        self.result=self.review_reply=self.approved=None
        self.export_button.setEnabled(False)
        self.set_candidate_options(None)
        self.result_cards.reset("正在执行已核验方案并生成结果，请稍候…")
        self.work(lambda log:run(scan,tasks,folder,progress=log,model_id=model_id),
                  lambda value:self.show_result(value,model_id))

    def execute_original(self):
        from .release_gate import require, require_whole_scope, runtime
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
        selected=tuple(self.selected_recommendations)
        require(selected)
        if not selected:
            require_whole_scope(self.recommendation_rows)
        self.result=self.review_reply=self.approved=None
        self.export_button.setEnabled(False)
        self.set_candidate_options(None)
        self.result_cards.reset("正在生成原版工程输出包…")
        self.result_cards.append_progress('已接收所选方案；正在准备工程数据。推荐能否作为搜索初始图，仍须真实契约检查；结果仅为预算内最佳。')
        def perform(log):
            fields = list(dict.fromkeys(field for row in selected for field in row.get("execution_fields", ())))
            review = runtime(scan, {"fields": fields, "steps": [step for row in selected for step in row.get("steps", ())]}, folder)
            result = s.run_original_workbench(scan,model,folder,
                lambda msg:log(msg if isinstance(msg,str) else "正在核验原版候选与质量…"), recommendations=selected)
            result["input_runtime_checks"] = review
            return result
        self.work(perform, lambda value:self.show_result(value,model))

    def show_result(self, result, model_id):
        self.show_progress("4/4 · 产品已生成，正在核对输入/输出格式、原生要求和实际试读状态")
        if self.scan and self.scan.get('file_agent_audit'):
            result['file_inspection'] = self.scan['file_agent_audit']
        self.result=result
        from .issues import output
        self.present_issues(output(result))
        check=output_review.describe_output(result,self.profiles.get(model_id))
        gate=result.get('manifest',{}).get('final_check')
        if gate:
            self.basis_summary.setPlainText(self.basis_summary.toPlainText()+'\n执行反馈 · 格式 '+gate['format']['status']+' / 工程 '+gate['project']['status']+' / 模型试读 '+gate['native']['status']+'；具体依据和范围见结果页。')
        items=output_review.cards(result,check)
        self.result_cards.render(items,check)
        self.set_candidate_options(result)
        self.export_button.setEnabled(True)
        key=self.api_key.text().strip() or os.environ.get("DEEPSEEK_API_KEY","").strip()
        mock=self.mock.isChecked()
        if not key and not mock:
            self.result_cards.update_check(check,notice="填写首页 API 密钥后可自动补充线上复核。")
            return

        self.result_cards.update_check(check,notice="正在复核脱敏的输出包清单…")
        request=output_review.review_request(check,items)
        summary_file=Path(self.task.name)/"output-review-request.json"

        def review(log):
            summary_file.write_text(s.dump(request),encoding='utf8')
            return s.review_model_output(result,self.profiles.get(model_id),api_key=key,mock=mock,progress=log)

        def finish(value):
            reply,notice=value
            self.review_reply=reply
            from .issues import output,item
            rows=output(result,reply)
            if notice:rows.append(item('最终线上核对',None,notice))
            self.present_issues(rows)
            if reply and mock:
                notice="模拟复核 · 未调用线上 API"
            self.result_cards.update_check(check,reply,notice)
            self.show_progress("处理已结束 · 请在结果页查看格式转换、检查结论与未完成项")

        self.work(review,finish)

    def set_candidate_options(self,result):
        from .result_summary import candidate_options
        self.candidate_select.clear()
        for row in candidate_options(result) if result else []:
            score=row['metrics']
            title=recommendations.recipe_name(row['steps'])
            origin={'mutation':'变异方案','recommended_initial':'DS推荐验证'}.get(row['phase'],'已评价方案')
            default=' · 默认' if row['selected'] else ''
            self.candidate_select.addItem(f"{origin} · {title} · RMSE {score['RMSE']:.4g}{default}",row)
        available=self.candidate_select.count()>0
        self.candidate_select.setEnabled(available)
        self.candidate_export.setEnabled(available)
        for widget in (self.candidate_label,self.candidate_select,self.candidate_export,self.candidate_note):widget.setVisible(available)
        if not available:
            self.candidate_note.setText('完成支持候选回放的搜索后，可选择已通过评价的方案。默认结果仍按原流程导出。')
        else:self.candidate_changed()

    def candidate_changed(self,*_):
        row=self.candidate_select.currentData()
        if not row:return
        reference=('格点代理，不能代表独立实测精度' if row['reference_basis'] in {'PRIMARY_GRID_RECONSTRUCTION_PROXY','REDUCED_SCALE_PRODUCT_RECONSTRUCTION','COARSE_PRODUCT_RECONSTRUCTION_ONLY'}
                   else '按已声明的验证参考；独立性以验收记录为准')
        parameters=json.dumps(row['parameters'],ensure_ascii=False)
        self.candidate_note.setText(f"同队列数值评价通过 · MAE {row['metrics']['MAE']:.4g} · 样本 {row['metrics']['count']} · {reference}\n"
            f"参数 {parameters}。每任务显示前10项，同验证预测的重复图合并。导出前重新执行并核验产品；不替换默认工程包。")

    def export_candidate_now(self):
        from .result_summary import export_evaluated_candidate
        row=self.candidate_select.currentData()
        if not row or not self.result:raise ValueError('请先完成搜索并选择已评价的候选')
        assert_current(self.scan)
        path=QFileDialog.getSaveFileName(self,'导出所选方案的数据产品','EcoData-selected-product.zip','ZIP (*.zip)')[0]
        if not path:return
        if not path.lower().endswith('.zip'):path+='.zip'
        destination=Path(path).resolve()
        protected={p.resolve() for p,*_ in self.scan['signature']}
        if self.result.get('original_archive'):protected.add(Path(self.result['original_archive']).resolve())
        if destination in protected:raise ValueError('不能覆盖原始输入或默认工程包')
        def finished(path):
            import zipfile
            with zipfile.ZipFile(path) as archive:receipt=json.loads(archive.read('selection.json'))
            same=receipt['default_product_comparison']['same_values']
            notice='本输入上的产品数值与默认方案相同。' if same else '所选产品与默认方案存在数值差异，不代表精度更高。'
            self.candidate_note.setText(self.candidate_note.text()+'\n'+notice)
            self.statusBar().showMessage('所选方案数据已导出；默认工程包未改变：'+path.name)
        self.work(lambda _:export_evaluated_candidate(row['search_folder'],row['key'],destination),
            finished)

    def export_now(self):
        self.check_config()
        if not self.result: raise ValueError("请先完成本地检查")
        assert_current(self.scan)
        path=QFileDialog.getSaveFileName(self,"下载处理结果","模拟技术引擎-结果.zip","ZIP (*.zip)")[0]
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




