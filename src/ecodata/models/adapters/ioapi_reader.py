"""Read I/O API with PseudoNetCDF in its compatible, isolated environment."""
import json
import os
from pathlib import Path
import subprocess
import sys

from ecodata.core.exceptions import SchemaError


class CmaqIOAPIReader:
    def __init__(self, python=None):
        root = Path(__file__).resolve().parents[4]
        configured = python or os.environ.get("ECODATA_PNC_PYTHON")
        bundled = root/"_runtime"/"python.exe"
        isolated = root/"_pnc_runtime"/"Scripts"/"python.exe"
        sibling_isolated = Path(sys.executable).resolve().parent.parent/"_pnc_runtime"/"Scripts"/"python.exe"
        legacy_isolated = root/"_pnc_runtime"/"python.exe"
        self.python = Path(configured) if configured else (
            isolated if isolated.is_file() else sibling_isolated if sibling_isolated.is_file()
            else legacy_isolated if legacy_isolated.is_file()
            else bundled if (root/"_runtime"/"Lib"/"site-packages"/"PseudoNetCDF").is_dir()
            else root/".venv-pnc"/"Scripts"/"python.exe")

    def inspect(self, path, *, format="ioapi"):
        if format not in {"ioapi", "icon_profile", "bcon_profile", "griddesc"}:
            raise SchemaError("只接受明确的 PseudoNetCDF CMAQ/I/O API 格式")
        if not self.python.is_file():
            raise SchemaError("未配置 PseudoNetCDF 独立环境；不能假装完成原生读取")
        worker = Path(__file__).with_name("ioapi_worker.py")
        result = subprocess.run([str(self.python), "-X", "utf8", str(worker), str(Path(path).resolve()), format],
                                capture_output=True, timeout=60)
        # Native libraries may emit locale-encoded diagnostics. Keep stderr
        # readable even on failure without letting a decoder thread mask it.
        diagnostics = result.stderr.decode("utf-8", errors="replace")
        if result.returncode:
            raise SchemaError("PseudoNetCDF 读取失败：" + diagnostics[-1500:])
        try:
            report = json.loads(result.stdout.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise SchemaError("PseudoNetCDF 未返回有效UTF-8 JSON报告：" + diagnostics[-1500:]) from exc
        report["reader_diagnostics"] = diagnostics[-1500:]
        report["native_model_input_verified"] = False
        report["scope"] = "真实库读取与 I/O API 关键属性检查，不等于对应版本 CMAQ 运行验证"
        return report
