"""Existing integrity checks plus independent standards and skill backends."""
from importlib.metadata import version
from pathlib import Path
import json
import tempfile

import numpy as np
import xarray as xr

from ecodata.core.exceptions import SchemaError


class ValidationEngine:
    def validate(self, *args, **kwargs):
        from ecodata.evaluation.self_check import PostConversionValidator
        return PostConversionValidator().validate(*args, **kwargs)

    @staticmethod
    def skill(prediction, reference, *, independent=False):
        if not independent or reference is None:
            raise SchemaError("xskillscore 精度评估需要明确的独立参考，训练拟合误差不是独立精度")
        import xskillscore as xs
        actual, expected = xr.align(prediction, reference, join="exact")
        valid = np.isfinite(actual.values) & np.isfinite(expected.values)
        if not valid.any():
            return {"available": False, "reason": "没有共同有效独立验证样本"}
        a = xr.DataArray(actual.values[valid], dims="sample")
        b = xr.DataArray(expected.values[valid], dims="sample")
        correlation = float(xs.pearson_r(a, b, dim="sample")) if a.size > 1 and a.std() > 0 and b.std() > 0 else None
        return {"available": True, "backend": "xskillscore", "sample_count": int(a.size),
                "RMSE": float(xs.rmse(a, b, dim="sample")), "MAE": float(xs.mae(a, b, dim="sample")),
                "bias": float((a-b).mean()), "correlation": correlation}

    @staticmethod
    def compliance(content, *, acdd=False):
        from compliance_checker.runner import ComplianceChecker
        from compliance_checker.suite import CheckSuite
        suite = CheckSuite()
        suite.load_all_available_checkers()
        checkers = ["cf:1.8"] + (["acdd:1.3"] if acdd else [])
        try:
            with tempfile.TemporaryDirectory(prefix="ecodata_cf_") as folder:
                path, report = Path(folder)/"output.nc", Path(folder)/"standards.json"
                path.write_bytes(content)
                passed, errors = ComplianceChecker.run_checker(str(path), checkers, 0, "normal",
                    output_filename=str(report), output_format="json")
                details = json.loads(report.read_text(encoding="utf-8"))
            return {"backend": "compliance-checker", "version": version("compliance-checker"),
                    "status": "ERROR" if errors else "PASS" if passed else "FAIL", "criteria": "normal",
                    "checkers": checkers, "report": details, "acdd_applicable": acdd,
                    "scope": "CF/ACDD 文件规范，不是模型原生输入验证"}
        except Exception as exc:
            return {"backend": "compliance-checker", "status": "ERROR", "reason": str(exc),
                    "checkers": checkers, "scope": "检查器未完成，不能声称 CF/ACDD 通过"}
