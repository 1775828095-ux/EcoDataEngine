"""Two-stage quality gates used before algorithm selection and model delivery."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


LEVELS = {"PASS": 0, "WARN": 1, "BLOCK": 2}


def _item(check_id: str, category: str, status: str, message: str, *, value: Any = None,
          threshold: Any = None, locations: list[str] | None = None, decision: str = "none") -> dict[str, Any]:
    return {"id": check_id, "category": category, "status": status, "message": message,
            "value": value, "threshold": threshold, "locations": locations or [], "decision": decision}


def _report(stage: str, checks: list[dict[str, Any]], *, accuracy_assessed: bool = False) -> dict[str, Any]:
    status = max((item["status"] for item in checks), key=lambda value: LEVELS[value], default="PASS")
    counts = {level: sum(item["status"] == level for item in checks) for level in LEVELS}
    return {"stage": stage, "status": status, "passed": status != "BLOCK", "checks": checks,
            "summary": counts, "decision": {"PASS": "continue", "WARN": "confirm_required", "BLOCK": "blocked"}[status],
            "accuracy_assessed": accuracy_assessed,
            "accuracy_note": "已使用独立参考数据评价" if accuracy_assessed else "没有独立真值；仅检查结构、元数据和物理合理性，不代表预测精度"}


def _coordinate(dataset: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray | None:
    for name in names:
        if name in dataset.coords:
            return dataset.coords[name]
        if name in dataset:
            return dataset[name]
    return None


class RawInputQualityGate:
    """Read-only checks. This gate never edits, fills or drops source values."""

    def validate(self, dataset: xr.Dataset, *, variable: str | None = None,
                 intended_operation: str = "conversion", expected_unit: str | None = None,
                 requires_spatial: bool | None = None) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        spatial = requires_spatial if requires_spatial is not None else intended_operation in {
            "spatial", "reconstruction", "fusion", "model_adaptation"
        }
        if not dataset.data_vars or any(size == 0 for size in dataset.sizes.values()):
            checks.append(_item("readable_structure", "完整性", "BLOCK", "数据没有可处理变量或维度为空", decision="禁止继续"))
            return _report("raw_input", checks)
        checks.append(_item("readable_structure", "完整性", "PASS", "数据结构可解析"))

        names = [variable] if variable else list(dataset.data_vars)
        for name in names:
            if name not in dataset:
                checks.append(_item(f"{name}.present", "完整性", "BLOCK", f"关键变量 {name} 不存在", decision="补充或重新映射字段"))
                continue
            field = dataset[name]
            missing = pd.isna(field.values)
            ratio = float(missing.mean()) if field.size else 1.0
            level = "BLOCK" if ratio >= 1 else ("WARN" if ratio > 0 else "PASS")
            checks.append(_item(f"{name}.missing", "完整性", level, f"{name} 缺失率 {ratio:.2%}",
                                value=ratio, threshold={"warn": ">0", "block": "100%"},
                                decision="保留原始缺失并由用户确认" if level == "WARN" else ("补充有效数据" if level == "BLOCK" else "none")))
            if field.dtype.kind in "biuf":
                values = np.asarray(field.values, dtype=float)
                inf = np.argwhere(np.isinf(values))
                checks.append(_item(f"{name}.finite", "有效性", "BLOCK" if len(inf) else "PASS",
                                    f"{name} {'包含 Inf' if len(inf) else '未发现 Inf'}", value=int(len(inf)),
                                    locations=[str(tuple(index)) for index in inf[:20]], decision="修正非法值" if len(inf) else "none"))
                finite = values[np.isfinite(values)]
                if finite.size >= 8:
                    q1, q3 = np.quantile(finite, [0.25, 0.75]); spread = q3 - q1
                    outliers = int(((finite < q1 - 3 * spread) | (finite > q3 + 3 * spread)).sum()) if spread else 0
                    checks.append(_item(f"{name}.outliers", "有效性", "WARN" if outliers else "PASS",
                                        f"{name} {'发现疑似异常值' if outliers else '未发现明显极端离群'}", value=outliers,
                                        threshold="3×IQR", decision="人工核对，不自动删除" if outliers else "none"))
            elif field.dtype.kind in "OUS":
                text = pd.Series(np.asarray(field.values).ravel()).dropna().astype(str)
                broken = int(text.str.contains("\ufffd|\x00", regex=True).sum())
                checks.append(_item(f"{name}.encoding", "有效性", "WARN" if broken else "PASS",
                                    f"{name} {'含疑似乱码' if broken else '文本编码未见明显异常'}", value=broken,
                                    decision="核对源编码" if broken else "none"))

        time = _coordinate(dataset, ("time", "datetime", "date", "DATATIME"))
        if time is not None:
            parsed = pd.to_datetime(np.asarray(time.values).ravel(), errors="coerce")
            invalid = int(pd.isna(parsed).sum()); duplicate = int(pd.Series(parsed).duplicated().sum())
            checks.append(_item("time.parse", "时间", "BLOCK" if invalid else "PASS", "时间字段可解析" if not invalid else "存在无法解析的时间", value=invalid))
            checks.append(_item("time.duplicates", "时间", "WARN" if duplicate else "PASS", "存在重复时间" if duplicate else "未发现重复时间", value=duplicate,
                                decision="确认站点/层次维度后再处理" if duplicate else "none"))
            valid = pd.Series(parsed).dropna().sort_values().drop_duplicates()
            if len(valid) > 2:
                gaps = valid.diff().dropna(); irregular = int((gaps != gaps.mode().iloc[0]).sum())
                checks.append(_item("time.interval", "时间", "WARN" if irregular else "PASS", "时间间隔不规则" if irregular else "时间间隔规则", value=irregular,
                                    decision="保留缺测并确认时间语义" if irregular else "none"))

        lon = _coordinate(dataset, ("lon", "longitude", "x")); lat = _coordinate(dataset, ("lat", "latitude", "y"))
        if spatial and (lon is None or lat is None):
            checks.append(_item("spatial.coordinates", "空间", "BLOCK", "当前任务需要空间坐标，但未识别到完整 x/y 或经纬度", decision="补充坐标或改为非空间任务"))
        elif lon is not None and lat is not None:
            checks.append(_item("spatial.coordinates", "空间", "PASS", "已识别空间坐标"))
            if (lon.name or "").lower() in {"lon", "longitude"} and np.isfinite(lon).any():
                bad = bool(((lon < -180) | (lon > 360)).any())
                checks.append(_item("spatial.longitude_range", "空间", "BLOCK" if bad else "PASS", "经度超出合理范围" if bad else "经度范围合理"))
            if (lat.name or "").lower() in {"lat", "latitude"} and np.isfinite(lat).any():
                bad = bool(((lat < -90) | (lat > 90)).any())
                checks.append(_item("spatial.latitude_range", "空间", "BLOCK" if bad else "PASS", "纬度超出合理范围" if bad else "纬度范围合理"))
            if spatial and not dataset.attrs.get("crs"):
                checks.append(_item("spatial.crs", "空间", "BLOCK", "空间任务缺少 CRS", decision="确认坐标基准"))
            else:
                checks.append(_item("spatial.crs", "空间", "PASS", "CRS 已记录" if dataset.attrs.get("crs") else "当前阶段不要求 CRS；如选择空间处理将在后续确认"))

        if variable and variable in dataset:
            unit = str(dataset[variable].attrs.get("units", "")).strip()
            if not unit:
                checks.append(_item("physical.unit", "物理", "BLOCK" if intended_operation in {"fusion", "model_adaptation"} else "WARN",
                                    f"{variable} 缺少单位", decision="用户确认单位"))
            elif expected_unit and unit != expected_unit:
                checks.append(_item("physical.unit", "物理", "BLOCK", f"单位 {unit} 与要求 {expected_unit} 不一致", value=unit, threshold=expected_unit,
                                    decision="执行可追溯单位换算或更正映射"))
            else:
                checks.append(_item("physical.unit", "物理", "PASS", f"单位：{unit}"))
            lower, upper = dataset[variable].attrs.get("valid_min"), dataset[variable].attrs.get("valid_max")
            values = np.asarray(dataset[variable].values)
            finite = values[np.isfinite(values)] if values.dtype.kind in "biuf" else np.array([])
            violations = (int((finite < lower).sum()) if lower is not None else 0) + (int((finite > upper).sum()) if upper is not None else 0)
            if lower is not None or upper is not None:
                checks.append(_item("physical.range", "物理", "BLOCK" if violations else "PASS", "存在超出声明物理范围的值" if violations else "数值满足声明物理范围", value=violations,
                                    threshold={"min": lower, "max": upper}))
        checks.append(_item("source.confidence", "可信度", "PASS", "未提供独立参考观测：不评价数据源准确性，保留不确定性"))
        return _report("raw_input", checks)


class ResultQualityGate:
    def validate(self, source: xr.Dataset, output: xr.Dataset, *, variable: str,
                 semantic_type: str = "continuous", nonnegative: bool = False,
                 model_check: dict[str, Any] | None = None, reference: xr.DataArray | None = None,
                 validation_role: str | None = None) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        if variable not in output:
            checks.append(_item("output.variable", "结构", "BLOCK", f"输出缺少变量 {variable}", decision="禁止模型适配与交付"))
            return _report("result", checks)
        values = np.asarray(output[variable].values)
        numeric = values.dtype.kind in "biuf"
        invalid = int((~np.isfinite(values)).sum()) if numeric else int(pd.isna(values).sum())
        all_invalid = invalid == values.size
        checks.append(_item("output.valid_values", "数值", "BLOCK" if all_invalid else ("WARN" if invalid else "PASS"),
                            "输出全部无效" if all_invalid else ("输出存在缺失值" if invalid else "输出值有效"), value=invalid,
                            decision="禁止交付" if all_invalid else ("保留 mask 并说明" if invalid else "none")))
        unit = output[variable].attrs.get("units")
        source_unit = source[variable].attrs.get("units") if variable in source else None
        checks.append(_item("output.unit", "单位", "BLOCK" if not unit or (source_unit and unit != source_unit) else "PASS",
                            "单位缺失或意外改变" if not unit or (source_unit and unit != source_unit) else f"单位保持为 {unit}",
                            value=unit, threshold=source_unit))
        source_crs = source.attrs.get("crs"); output_crs = output.attrs.get("crs")
        checks.append(_item("output.crs", "空间", "BLOCK" if source_crs and not output_crs else "PASS",
                            "输出 CRS 丢失" if source_crs and not output_crs else "CRS 检查通过"))
        if numeric and nonnegative:
            negative = int((values[np.isfinite(values)] < 0).sum())
            checks.append(_item("output.nonnegative", "物理", "BLOCK" if negative else "PASS", "出现物理上不允许的负值" if negative else "满足非负约束", value=negative))
        if semantic_type == "categorical" and variable in source:
            allowed = set(np.asarray(source[variable].values).ravel().tolist())
            introduced = sorted(set(values.ravel().tolist()) - allowed, key=str)
            checks.append(_item("output.categories", "物理", "BLOCK" if introduced else "PASS", "生成了源数据不存在的类别" if introduced else "分类值集合有效", value=[str(v) for v in introduced]))
        if model_check is not None:
            complete = bool(model_check.get("complete", model_check.get("passed", False)))
            checks.append(_item("model.compatibility", "模型需求", "PASS" if complete else "BLOCK", "模型需求复检通过" if complete else "模型需求复检未通过",
                                value=model_check, decision="禁止生成完整模型输入" if not complete else "none"))
        metrics: dict[str, Any] = {}
        assessed = reference is not None
        if reference is not None:
            if validation_role not in {"validation", "independent_validation"}:
                checks.append(_item("accuracy.reference_role", "精度", "BLOCK", "精度评价只能使用独立 validation 数据"))
            else:
                predicted, truth = xr.align(output[variable], reference, join="inner")
                delta = np.asarray(predicted - truth, dtype=float); finite = np.isfinite(delta)
                metrics = {"RMSE": float(np.sqrt(np.mean(delta[finite] ** 2))), "MAE": float(np.mean(np.abs(delta[finite]))),
                           "bias": float(np.mean(delta[finite]))} if finite.any() else {}
                checks.append(_item("accuracy.independent", "精度", "PASS" if metrics else "BLOCK", "已计算独立验证指标" if metrics else "独立验证无有效重叠", value=metrics))
        report = _report("result", checks, accuracy_assessed=assessed)
        report["metrics"] = metrics
        return report
