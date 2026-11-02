"""EcoData Engine's single local entry point."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ecodata.interface.workflow import run_p3_sample
from ecodata.versioning import version


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "examples" / "datasets"


def _disable_browser_translation() -> None:
    """Avoid third-party translation extensions mutating Streamlit's React DOM."""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.documentElement.setAttribute("translate", "no");
        doc.documentElement.classList.add("notranslate");
        doc.body.setAttribute("translate", "no");
        doc.body.classList.add("notranslate");
        if (!doc.head.querySelector('meta[name="google"]')) {
          const meta = doc.createElement("meta");
          meta.name = "google";
          meta.content = "notranslate";
          doc.head.appendChild(meta);
        }
        </script>
        """,
        height=0,
    )


def run_selected(method: str, resolution: float, hour: int):
    """Backward-compatible P3 API; no longer exposed as a separate page."""
    result, truth = run_p3_sample(method, resolution, hour, DATA)
    metrics = dict(result.report.get("精度指标", {}))
    metrics["runtime_seconds"] = result.runtime_seconds
    return result.original, truth, result.output, metrics


def render_app() -> None:
    current_version = version()
    st.set_page_config(page_title=f"EcoData Engine {current_version}", layout="wide")
    _disable_browser_translation()
    st.title("EcoData Engine 1.0")
    st.caption(f"环境数据转换与模型数据准备 · 本地运行 · {current_version}")
    entry = st.sidebar.radio("功能", ["数据转换", "模型数据准备", "融合重构实验", "首轮专项转换", "SWAT+ 工程运行", "WRF 工程运行", "CALMET/CALPUFF 工程运行", "CAMx 工程运行", "CMAQ 工程运行"], index={"swat-run":4,"wrf-run":5,"calpuff-run":6,"camx-run":7,"cmaq-run":8}.get(st.query_params.get("page"),0), key="main_entry")

    if entry == "CMAQ 工程运行":
        from ecodata.interface.cmaq_run_page import render_cmaq_run_page
        render_cmaq_run_page()
        return

    if entry == "CAMx 工程运行":
        from ecodata.interface.camx_run_page import render_camx_run_page
        render_camx_run_page()
        return

    if entry == "CALMET/CALPUFF 工程运行":
        from ecodata.interface.calmet_run_page import render_calmet_run_page
        render_calmet_run_page()
        return
    st.sidebar.caption("已有时间转换、空间转换、融合和约束算法保留在后台，由任务流程统一调用。")
    if entry == "WRF 工程运行":
        from ecodata.interface.wrf_run_page import render_wrf_run_page
        render_wrf_run_page()
    elif entry == "SWAT+ 工程运行":
        from ecodata.interface.swat_run_page import render_swat_run_page
        render_swat_run_page()
    elif entry == "首轮专项转换":
        from ecodata.interface.round1_page import render_round1_page
        render_round1_page()
    elif entry == "模型数据准备":
        from ecodata.interface.model_page import render_model_page
        render_model_page()
    elif entry == "融合重构实验":
        from ecodata.interface.fusion_experiment_page import render_fusion_experiment_page
        render_fusion_experiment_page()
    else:
        from ecodata.interface.task_page import render_task_page
        render_task_page()


if __name__ == "__main__":
    render_app()
