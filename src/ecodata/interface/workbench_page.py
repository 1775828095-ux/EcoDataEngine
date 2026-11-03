"""The two-input user-facing workbench; scientific controls remain internal."""
from pathlib import Path
import hashlib
import html
import uuid

import streamlit as st


MODELS = {"SWAT+": "swat_plus", "WRF": "wrf", "CMAQ": "cmaq",
          "CAMx": "camx", "CALPUFF": "calpuff"}


def input_identity(uploads, model):
    digest = hashlib.sha256(model.encode())
    for upload in uploads or []:
        digest.update(upload.name.encode())
        digest.update(hashlib.sha256(upload.getvalue()).digest())
    return digest.hexdigest()


def stage_strip(state="empty", fusion_status=None):
    states = {"empty": ("等待数据", "等待识别", "等待处理"),
              "running": ("处理中", "自动执行", "等待处理"),
              "AWAITING_CONVERTER": ("已完成", "已完成", "待接入"),
              "CONVERTED": ("已完成", "已完成", "已完成"),
              "CONVERSION_FAILED": ("已完成", "已完成", "转化失败")}
    labels = states.get(state, states["empty"])
    if fusion_status in {'NOT_APPLICABLE_SAME_PRODUCT','PRESERVE_SINGLE_SOURCE'}:
        labels = (labels[0], '已检查，无需重复融合', labels[2])
    outcomes={'SINGLE_SOURCE_SELECTED':'已处理，最终采用单源',
              'SINGLE_SOURCE_PRESERVED':'保留单源，未融合',
              'MULTISOURCE_GRAPH_EFFECT_UNVERIFIED':'多源方案已执行，贡献未验证',
              'NOT_DETERMINABLE':'处理已完成，融合证据不足',
              'MIXED_OR_NOT_DETERMINABLE':'部分处理的融合证据不足'}
    if fusion_status in outcomes:
        labels=(labels[0],outcomes[fusion_status],labels[2])
    return '<div class="stages">' + ''.join(
        f'<div class="stage"><span class="stage-number">0{i + 1}</span><div><strong>{title}</strong><span>{label}</span></div></div>'
        for i, (title, label) in enumerate(zip(("自动识别", "自动融合", "自动转化"), labels))) + '</div>'


def render_workbench(root, version):
    style = Path(__file__).with_name("workbench.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
    st.markdown('''<div class="brand"><svg width="35" height="35" viewBox="0 0 35 35" aria-hidden="true"><path d="M3 10L17 3l15 7-15 7zM3 18l14 7 15-7M3 26l14 7 15-7" fill="none" stroke="currentColor" stroke-width="2"/></svg><div><strong>EcoData<span> Engine</span></strong><small>环境数据准备工作台</small></div><span class="local-label">本地工作空间</span></div>''', unsafe_allow_html=True)
    st.title("让数据，进入模型。")
    st.markdown('<p class="intro">提交数据，选择目标模型。其余交给平台自动识别、融合与转化。</p>', unsafe_allow_html=True)
    left, right = st.columns([0.95, 1.35], gap="large")
    with left:
        with st.container(border=True, key="inputs"):
            st.subheader("输入数据")
            st.caption("可同时上传多个来源；平台根据物理变量与数据关系判断如何处理。")
            uploads = st.file_uploader("上传压缩包或数据文件", accept_multiple_files=True,
                type=["nc", "nc4", "tif", "tiff", "csv", "tsv", "xlsx", "parquet", "h5", "zip"],
                help="可直接拖入 ZIP 压缩包或多个数据文件。文件应保留变量、单位、时间及坐标信息。")
            st.markdown('<div class="file-help">NetCDF · GeoTIFF · CSV / Excel · 数据 ZIP</div>', unsafe_allow_html=True)
            folder_path = st.text_input("或输入本地文件夹路径", placeholder=r"例如 D:\数据\流域气象",
                help="读取运行平台这台电脑上的文件夹，自动包含子文件夹。可与上方上传文件一起提交。")
            input_error = None
            if folder_path.strip():
                from ecodata.interface.engineering_upload import local_folder_uploads
                try:
                    local_uploads = local_folder_uploads(folder_path, include_all=True)
                    uploads = [*(uploads or []), *local_uploads]
                    st.caption(f"已找到 {len(local_uploads)} 个数据文件或压缩包，将自动读取。")
                except (ValueError, OSError) as exc:
                    input_error = str(exc)
                    st.error(input_error)
            model = st.selectbox("目标模型", list(MODELS))
            st.caption("按照模型已有输入契约准备数据。")
            try:
                identity = input_identity(uploads, model)
            except (OSError, ValueError):
                identity = "unreadable"
                input_error = "文件已移动或无法读取，请检查文件夹路径。"
                st.error(input_error)
            if input_error:
                identity += folder_path
            previous = st.session_state.get("workbench_result")
            if previous and previous["identity"] != identity:
                del st.session_state["workbench_result"]
            start = st.button("开始自动处理", type="primary", use_container_width=True, disabled=not uploads or bool(input_error))
            st.markdown('<p class="privacy">数据在本地处理，原始文件保留。</p>', unsafe_allow_html=True)
    with right:
        with st.container(border=True, key="output"):
            st.subheader("处理与输出")
            result = st.session_state.get("workbench_result")
            stages = st.empty()
            stages.markdown(stage_strip(result["data"]["manifest"]["status"] if result else "empty",
                result['data']['manifest'].get('preparation',{}).get('fusion_decision',{}).get('status') if result else None), unsafe_allow_html=True)
            body = st.empty()
            if start:
                stages.markdown(stage_strip("running"), unsafe_allow_html=True)
                try:
                    from ecodata.algorithms.automatic_preparation import prepare_uploads
                    folder = Path(root) / "output" / "workbench" / uuid.uuid4().hex
                    with st.status("正在识别数据并自动处理…", expanded=True) as status:
                        data = prepare_uploads(uploads, MODELS[model], folder,
                            progress=lambda message: status.update(label=message if isinstance(message, str) else "正在真实执行候选组合并检验质量…"))
                        status.update(label="数据处理完成", state="complete", expanded=False)
                    result = dict(identity=identity, data=data, model=model)
                    st.session_state["workbench_result"] = result
                    stages.markdown(stage_strip(data["manifest"]["status"], data['manifest'].get('preparation',{}).get('fusion_decision',{}).get('status')), unsafe_allow_html=True)
                except Exception as exc:
                    stages.markdown(stage_strip(), unsafe_allow_html=True)
                    body.error(f"暂未完成处理：{exc}")
                    st.caption("请检查提示中的数据或元信息后重新提交。原始数据未被改写。")
                    return
            if result:
                manifest = result["data"]["manifest"]
                if manifest["status"] == "CONVERTED":
                    if manifest.get('preparation',{}).get('route')=='wrf_daily_sst_bundle_v1':
                        st.success("格式转化完成，可下载本次自动处理后的大气驱动与海温补充文件；尚未合并为可直接运行的完整 WRF 工程。")
                    else:
                        st.success("格式转化完成，可以下载。")
                    label = "下载模型输入包"
                else:
                    st.markdown('<div class="result-heading">数据已准备，等待格式转化</div>', unsafe_allow_html=True)
                    st.write("队友的格式转化模块尚未接入。已处理数据和交接清单已打包，接入后将自动完成最后一步。")
                    if manifest["status"] == "CONVERSION_FAILED":
                        st.error("转化模块未通过校验：" + manifest.get("conversion_error", "未知原因"))
                    st.caption("当前下载为数据准备包，尚不是模型原生输入文件。")
                    label = "下载数据准备包"
                path = Path(result["data"]["archive"])
                st.download_button(label, data=path.read_bytes(), file_name=f"EcoData-{result['model']}.zip",
                                   mime="application/zip", use_container_width=True)
            elif not start:
                body.markdown('''<div class="empty-output"><svg width="112" height="88" viewBox="0 0 112 88" aria-hidden="true"><path d="M10 42L56 18l46 24-46 24zM10 54l46 24 46-24M10 30L56 6l46 24" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M56 18v48" stroke="currentColor" stroke-dasharray="3 4"/></svg><strong>从一份数据开始</strong><p>处理状态与输出结果将在这里呈现。</p></div>''', unsafe_allow_html=True)
            st.markdown('<div class="handoff-note"><strong>格式转化接入口已预留</strong><span>融合后的数据将直接交给模型转化模块。</span></div>', unsafe_allow_html=True)
    st.markdown(f'<footer>EcoData Engine <span>自动识别 → 自动融合 → 自动转化</span><small>{html.escape(str(version))}</small></footer>', unsafe_allow_html=True)
