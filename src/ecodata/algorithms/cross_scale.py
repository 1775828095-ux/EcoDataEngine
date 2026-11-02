from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext, AlgorithmResult, AlgorithmSpec, Compatibility
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord
from ecodata.spatial.regridding import GridRegridder, RegridConfig


class SemanticCrossScaleStackPlugin:
    """Build a dimensionless feature stack after semantics-aware spatial alignment."""

    spec = AlgorithmSpec(
        id="scale.semantic_stack",
        name="Semantic cross-scale feature stack",
        version="1",
        category="multi_source_feature_preparation",
        status="available",
        input_kinds=("rectilinear_grids",),
        semantic_types=("multidimensional_array",),
        requires_auxiliary=True,
        physical_constraints=("variable_semantics", "unit_preservation", "categorical_fraction", "conservation_when_declared"),
        compute="medium",
        license="project / xarray-regrid backend",
        backend="EcoData semantic routing + xarray-regrid",
        notes="Creates aligned raw variables and normalized model features. A finer output grid does not imply finer physical information.",
    )

    @staticmethod
    def _datasets(context: AlgorithmContext) -> tuple[xr.Dataset, ...]:
        return tuple(source.dataset for source in context.sources) + tuple(context.auxiliaries)

    def compatibility(self, context: AlgorithmContext) -> Compatibility:
        reasons = []
        if context.variable != "feature_stack":
            reasons.append("该插件生成 feature_stack，不生成任意目标物理变量")
        if context.target_lat is None or context.target_lon is None:
            reasons.append("缺少目标 lat/lon 网格")
        if len(self._datasets(context)) < 2:
            reasons.append("跨尺度特征栈至少需要两个来源")
        crs_values = set()
        usable = 0
        for dataset in self._datasets(context):
            crs = dataset.attrs.get("crs")
            if crs:
                crs_values.add(str(crs))
            for name, field in dataset.data_vars.items():
                if {"lat", "lon"}.issubset(field.dims):
                    usable += 1
                    kind = field.attrs.get("data_kind")
                    if kind not in {"continuous", "categorical"}:
                        reasons.append(f"{name} 缺少受支持的 data_kind（continuous/categorical）")
                    if not field.attrs.get("units"):
                        reasons.append(f"{name} 缺少单位；分类变量应使用 units=1")
        if usable < 2:
            reasons.append("至少需要两个带 lat/lon 的可用变量")
        if len(crs_values) != 1 or any(not dataset.attrs.get("crs") for dataset in self._datasets(context)):
            reasons.append("所有来源必须明确且使用同一 CRS")
        return Compatibility(not reasons, tuple(dict.fromkeys(reasons)), ("网格加密只改变表达网格，不代表真实信息尺度提高",))

    @staticmethod
    def _is_finer(source: xr.DataArray, target_lat, target_lon) -> bool:
        source_step = abs(float(np.median(np.diff(source.lat.values)))) * abs(float(np.median(np.diff(source.lon.values))))
        target_step = abs(float(np.median(np.diff(target_lat)))) * abs(float(np.median(np.diff(target_lon))))
        return target_step < source_step

    @staticmethod
    def _regrid(field: xr.DataArray, method: str, target_lat, target_lon) -> xr.DataArray:
        dataset = field.to_dataset(name=field.name)
        dataset.attrs.update(crs=field.attrs.get("crs", ""), data_kind=field.attrs.get("data_kind"), spatial_semantics=field.attrs.get("spatial_semantics"))
        bundle = EcoDataBundle(dataset)
        output = GridRegridder(RegridConfig(method, tuple(target_lat), tuple(target_lon), field.name)).apply(bundle).data.dataset[field.name]
        if not np.isfinite(output.values).all():
            raise SchemaError(f"{field.name} 尺度匹配后出现 NaN；源数据必须覆盖完整目标范围")
        return output

    def run(self, context: AlgorithmContext) -> AlgorithmResult:
        compatibility = self.compatibility(context)
        if not compatibility.eligible:
            raise SchemaError("跨尺度特征准备不可执行：" + "；".join(compatibility.reasons))
        started = perf_counter()
        target_lat = np.asarray(context.target_lat, dtype=float)
        target_lon = np.asarray(context.target_lon, dtype=float)
        raw_variables = {}
        feature_arrays = []
        feature_names = []
        transformations = []
        seen_names = set()

        for dataset in self._datasets(context):
            for name, original in dataset.data_vars.items():
                if not {"lat", "lon"}.issubset(original.dims):
                    continue
                if name in seen_names:
                    raise SchemaError(f"跨来源变量名重复：{name}；请在输入阶段明确重命名")
                seen_names.add(name)
                field = original.transpose("lat", "lon").copy()
                field.attrs["crs"] = dataset.attrs["crs"]
                kind = field.attrs["data_kind"]
                finer_target = self._is_finer(field, target_lat, target_lon)

                if kind == "categorical":
                    categories = [value.item() if hasattr(value, "item") else value for value in np.unique(field.values[np.isfinite(field.values)])]
                    categories = [int(value) if isinstance(value, float) and value.is_integer() else value for value in categories]
                    for category in categories:
                        fraction_name = f"{name}__fraction_{category}"
                        indicator = xr.where(field == category, 1.0, 0.0).rename(fraction_name)
                        indicator.attrs.update(units="1", data_kind="continuous", spatial_semantics="cell_mean", source_variable=name, category_value=category)
                        method = "nearest" if finer_target else "conservative"
                        aligned = self._regrid(indicator, method, target_lat, target_lon)
                        raw_variables[fraction_name] = aligned
                        feature_arrays.append(aligned)
                        feature_names.append(fraction_name)
                    transformations.append({"variable": name, "semantic": "categorical", "method": "nearest one-hot" if finer_target else "conservative category fractions", "information_scale_increased": False})
                    continue

                semantics = field.attrs.get("spatial_semantics")
                if finer_target:
                    method = "bilinear"
                elif semantics in {"cell_mean", "flux"}:
                    method = "conservative"
                else:
                    raise SchemaError(f"{name} 向更粗网格聚合前必须声明 spatial_semantics=cell_mean/flux")
                aligned = self._regrid(field, method, target_lat, target_lon)
                raw_variables[name] = aligned
                feature_arrays.append(aligned)
                feature_names.append(name)
                transformations.append({"variable": name, "semantic": "continuous", "method": method, "information_scale_increased": False})

        normalized = []
        normalization = {}
        for name, field in zip(feature_names, feature_arrays):
            values = np.asarray(field.values, dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            scale = std if std > 0 else 1.0
            normalized.append(((field - mean) / scale).expand_dims(feature=[name]))
            normalization[name] = {"mean": mean, "std": scale, "source_units": field.attrs.get("units", "")}
        feature_stack = xr.concat(normalized, dim="feature").astype(np.float32)
        output = xr.Dataset(raw_variables, coords={"lat": target_lat, "lon": target_lon})
        output["feature_stack"] = feature_stack
        output["feature_stack"].attrs.update(units="1", long_name="normalized semantics-aware cross-scale features")
        output.attrs.update(
            crs=self._datasets(context)[0].attrs["crs"],
            processing_stage="semantic_cross_scale_feature_preparation",
            normalization=json.dumps(normalization, ensure_ascii=False),
            resolution_claim="Aligned grid resolution is not a claim of recovered physical detail.",
        )
        elapsed = perf_counter() - started
        record = TransformRecord(self.spec.id, {"transformations": transformations}, sum(field.size for field in feature_arrays), feature_stack.size, elapsed, compatibility.warnings)
        report = {"status": "PASS", "features": feature_names, "transformations": transformations, "runtime_seconds": elapsed, "physical_detail_claim": False}
        return AlgorithmResult(EcoDataBundle(output, (record,)), self.spec.id, report)
