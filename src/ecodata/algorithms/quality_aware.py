from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext, AlgorithmResult, AlgorithmSpec, Compatibility
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord


def fit_affine_source(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Shared existing source-calibration component; callers enforce fit isolation."""
    valid = np.isfinite(prediction) & np.isfinite(reference)
    if valid.sum() < 3:
        raise SchemaError("至少需要3条有效训练配对拟合偏差校正")
    design = np.column_stack([prediction[valid], np.ones(valid.sum())])
    if np.linalg.matrix_rank(design) != 2:
        raise SchemaError("偏差校正训练背景无变化，无法识别斜率")
    slope, intercept = np.linalg.lstsq(design, reference[valid], rcond=None)[0]
    return float(slope), float(intercept)


def fit_affine_frames(prediction, reference):
    """Vectorized OLS over time x retained station pairs; unidentified dates use identity.

    Returns the number of such dates so the caller must expose that fallback.
    This is the same affine model as fit_affine_source, not a learned preference.
    """
    valid=np.isfinite(prediction)&np.isfinite(reference);count=valid.sum(axis=1,keepdims=True)
    xm=np.divide(np.where(valid,prediction,0).sum(axis=1,keepdims=True),count,out=np.zeros_like(count,dtype=float),where=count>0)
    ym=np.divide(np.where(valid,reference,0).sum(axis=1,keepdims=True),count,out=np.zeros_like(count,dtype=float),where=count>0)
    xx=np.where(valid,prediction-xm,0);yy=np.where(valid,reference-ym,0);den=(xx*xx).sum(axis=1,keepdims=True)
    okay=(count>=3)&(den>np.finfo(float).eps)
    slope=np.divide((xx*yy).sum(axis=1,keepdims=True),den,out=np.ones_like(den),where=okay)
    intercept=np.where(okay,ym-slope*xm,0.)
    return slope,intercept,int((~okay).sum())


def fit_ratio_frames(prediction, reference):
    """Existing spatial ratio of sums, with explicit identifiable-date mask."""
    valid=np.isfinite(prediction)&np.isfinite(reference)
    denominator=np.where(valid,prediction,0).sum(axis=1,keepdims=True)
    numerator=np.where(valid,reference,0).sum(axis=1,keepdims=True)
    okay=(valid.sum(axis=1,keepdims=True)>=2)&(denominator>0)
    return np.divide(numerator,denominator,out=np.ones_like(numerator),where=okay),okay


class QualityAwareFusionPlugin:
    """Bias-correct gridded sources and weight them using held-out validation observations."""

    spec = AlgorithmSpec(
        id="fusion.quality_aware",
        name="Observation-validated quality-aware fusion",
        version="1",
        category="multi_source_fusion",
        status="available",
        input_kinds=("aligned_grids", "station_table"),
        semantic_types=("continuous",),
        requires_observations=True,
        physical_constraints=("same_variable", "same_unit", "same_crs", "train_validation_separation"),
        compute="low",
        license="project",
        backend="NumPy least-squares + inverse validation-error weighting",
        notes="Training observations fit source-specific linear bias correction; validation observations select weights. Test observations are never used to fit or weight.",
    )

    @staticmethod
    def _grid(dataset: xr.Dataset, variable: str) -> xr.DataArray:
        if variable not in dataset:
            raise SchemaError(f"输入缺少目标变量 {variable}")
        field = dataset[variable]
        if set(field.dims) != {"lat", "lon"}:
            raise SchemaError("当前质量感知融合要求每个来源为二维 lat/lon 网格")
        return field.transpose("lat", "lon")

    def compatibility(self, context: AlgorithmContext) -> Compatibility:
        reasons = []
        if len(context.sources) < 2:
            reasons.append("质量感知融合至少需要两个目标变量来源")
        units = set()
        crs_values = set()
        for source in context.sources:
            try:
                field = self._grid(source.dataset, context.variable)
            except SchemaError as exc:
                reasons.append(str(exc))
                continue
            unit = field.attrs.get("units")
            if not unit:
                reasons.append("每个来源必须明确目标变量单位")
            else:
                units.add(str(unit))
            crs = source.dataset.attrs.get("crs")
            if not crs:
                reasons.append("每个来源必须明确 CRS")
            else:
                crs_values.add(str(crs))
        if len(units) > 1:
            reasons.append("目标变量单位不一致，必须先换算")
        if len(crs_values) > 1:
            reasons.append("来源 CRS 不一致，必须先转换")
        observations = context.observations
        required = {"longitude", "latitude", "value", "split"}
        if observations is None or not required.issubset(observations.columns):
            reasons.append("需要含 longitude/latitude/value/split 的可靠观测")
        else:
            if len(observations[observations.split.eq("train")]) < 3:
                reasons.append("至少需要3条训练观测拟合偏差校正")
            if len(observations[observations.split.eq("validation")]) < 2:
                reasons.append("至少需要2条独立验证观测确定数据源权重")
        return Compatibility(not reasons, tuple(dict.fromkeys(reasons)))

    @staticmethod
    def _sample(field: xr.DataArray, observations: pd.DataFrame) -> np.ndarray:
        points = "point"
        sampled = field.interp(
            lat=xr.DataArray(observations.latitude.to_numpy(float), dims=points),
            lon=xr.DataArray(observations.longitude.to_numpy(float), dims=points),
            method="linear",
        )
        return np.asarray(sampled.values, dtype=float)

    @staticmethod
    def _metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        valid = np.isfinite(prediction) & np.isfinite(reference)
        if not valid.any():
            return {"count": 0, "RMSE": float("nan"), "MAE": float("nan"), "bias": float("nan")}
        error = prediction[valid] - reference[valid]
        return {
            "count": int(valid.sum()),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAE": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }

    def run(self, context: AlgorithmContext) -> AlgorithmResult:
        compatibility = self.compatibility(context)
        if not compatibility.eligible:
            raise SchemaError("质量感知融合不可执行：" + "；".join(compatibility.reasons))
        started = perf_counter()
        observations = context.observations
        assert observations is not None
        train = observations[observations.split.eq("train")].dropna(subset=["longitude", "latitude", "value"])
        validation = observations[observations.split.eq("validation")].dropna(subset=["longitude", "latitude", "value"])
        test = observations[observations.split.eq("test")].dropna(subset=["longitude", "latitude", "value"])

        reference = self._grid(context.sources[0].dataset, context.variable)
        corrected_fields = []
        reports = []
        for index, source in enumerate(context.sources):
            dataset = source.dataset
            field = self._grid(dataset, context.variable)
            same_grid = np.array_equal(field.lat.values, reference.lat.values) and np.array_equal(field.lon.values, reference.lon.values)
            if not same_grid:
                field = field.interp(lat=reference.lat, lon=reference.lon, method="linear")
            if not np.isfinite(field.values).all():
                raise SchemaError("来源尺度匹配后含 NaN；空间范围必须完整覆盖参考网格")
            train_prediction = self._sample(field, train)
            valid = np.isfinite(train_prediction) & np.isfinite(train.value.to_numpy(float))
            if valid.sum() < 3:
                raise SchemaError("某来源与训练观测有效重叠不足3条")
            slope, intercept = fit_affine_source(train_prediction[valid], train.value.to_numpy(float)[valid])
            corrected = slope * field + intercept
            validation_raw = self._metrics(self._sample(field, validation), validation.value.to_numpy(float))
            validation_corrected = self._metrics(self._sample(corrected, validation), validation.value.to_numpy(float))
            source_id = str(dataset.attrs.get("source_id", f"source_{index + 1}"))
            reports.append({
                "source_id": source_id,
                "scale_matching": "none" if same_grid else "bilinear_to_reference_grid",
                "bias_correction": {"slope": float(slope), "intercept": float(intercept), "fit_split": "train"},
                "validation_raw": validation_raw,
                "validation_corrected": validation_corrected,
            })
            corrected_fields.append(corrected)

        rmse = np.asarray([item["validation_corrected"]["RMSE"] for item in reports], dtype=float)
        if not np.isfinite(rmse).all():
            raise SchemaError("验证观测与某来源无有效重叠，不能自动判断数据源质量")
        inverse_error = 1.0 / np.maximum(rmse, float(context.parameters.get("rmse_floor", 1e-6))) ** 2
        weights = inverse_error / inverse_error.sum()
        stack = xr.concat(corrected_fields, dim="source")
        weight_array = xr.DataArray(weights, dims="source")
        fused = (stack * weight_array).sum("source")
        uncertainty = np.sqrt((((stack - fused) ** 2) * weight_array).sum("source"))

        output = xr.Dataset(
            {
                context.variable: fused.astype(np.float32),
                "fusion_uncertainty": uncertainty.astype(np.float32),
            },
            attrs={
                **context.sources[0].dataset.attrs,
                "processing_stage": "quality_aware_fused",
                "fusion_method": self.spec.id,
                "weight_basis": "inverse squared validation RMSE after train-only linear bias correction",
            },
        )
        output[context.variable].attrs.update(reference.attrs)
        output["fusion_uncertainty"].attrs.update(units=reference.attrs.get("units", ""), long_name="weighted source spread")
        for item, weight in zip(reports, weights):
            item["weight"] = float(weight)

        validation_fused = self._metrics(self._sample(fused, validation), validation.value.to_numpy(float))
        test_fused = self._metrics(self._sample(fused, test), test.value.to_numpy(float)) if len(test) else None
        elapsed = perf_counter() - started
        report = {
            "status": "PASS",
            "sources": reports,
            "validation_fused": validation_fused,
            "test_fused_independent": test_fused,
            "test_used_for_fitting_or_weighting": False,
            "uncertainty_mean": float(uncertainty.mean()),
            "runtime_seconds": elapsed,
            "limitations": ["验证集用于确定权重，因此 validation_fused 不是最终独立精度", "只有 test 指标可作为本次独立评价"],
        }
        record = TransformRecord(
            self.spec.id,
            {"weights": {item["source_id"]: item["weight"] for item in reports}, "fit_split": "train", "weight_split": "validation"},
            sum(int(self._grid(source.dataset, context.variable).size) for source in context.sources),
            int(fused.size),
            elapsed,
        )
        history = tuple(record for source in context.sources for record in source.history) + (record,)
        return AlgorithmResult(EcoDataBundle(output, history), self.spec.id, report, test_fused or {})
