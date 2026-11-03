"""EcoData Engine's single local entry point."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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
    from ecodata.interface.workflow import run_p3_sample
    result, truth = run_p3_sample(method, resolution, hour, DATA)
    metrics = dict(result.report.get("精度指标", {}))
    metrics["runtime_seconds"] = result.runtime_seconds
    return result.original, truth, result.output, metrics


def render_app() -> None:
    current_version = version()
    st.set_page_config(page_title=f"EcoData Engine {current_version}", layout="wide")
    from ecodata.interface.workbench_page import render_workbench
    render_workbench(ROOT, current_version)


if __name__ == "__main__":
    render_app()
