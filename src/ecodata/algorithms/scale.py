from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext


@dataclass(frozen=True)
class ScaleDecision:
    target_lat: tuple[float, ...] | None
    target_lon: tuple[float, ...] | None
    scale_factor: float
    recommended_method: str
    reason: str
    warnings: tuple[str, ...]


def spatial_scale_factor(context: AlgorithmContext) -> float:
    target_lat, target_lon = context.target_lat, context.target_lon
    grid = context.parameters.get("target_grid")
    if target_lat is None and isinstance(grid, xr.Dataset) and "lat" in grid.coords:
        target_lat = grid.lat
    if target_lon is None and isinstance(grid, xr.Dataset) and "lon" in grid.coords:
        target_lon = grid.lon
    if (not context.sources or target_lat is None or target_lon is None
            or context.variable not in context.sources[0].dataset):
        return 1.0
    field = context.sources[0].dataset[context.variable]
    if not {"lat", "lon"}.issubset(field.dims):
        return 1.0
    return max(len(target_lat) / field.sizes["lat"], len(target_lon) / field.sizes["lon"])


class TargetScaleAdvisor:
    def decide(self, context: AlgorithmContext) -> ScaleDecision:
        semantic = context.semantic_type
        constraints = context.model_constraints
        warnings = []
        target_lat, target_lon = context.target_lat, context.target_lon
        reason = "使用用户指定目标网格"
        grid = context.parameters.get("target_grid")
        if (target_lat is None or target_lon is None) and isinstance(grid, xr.Dataset) and {"lat", "lon"}.issubset(grid.coords):
            target_lat, target_lon = tuple(grid.lat.values), tuple(grid.lon.values)
        if target_lat is None or target_lon is None:
            requested = constraints.get("target_grid")
            if requested:
                target_lat = tuple(requested["lat"])
                target_lon = tuple(requested["lon"])
                reason = "使用目标模型 Model Schema 声明的网格"
            elif context.sources and {"lat", "lon"}.issubset(context.sources[0].dataset.coords):
                target_lat = tuple(np.asarray(context.sources[0].dataset.lat, float))
                target_lon = tuple(np.asarray(context.sources[0].dataset.lon, float))
                reason = "未提供可靠目标精度依据，保持源网格"
        factor = spatial_scale_factor(AlgorithmContext(**{**context.__dict__, "target_lat": target_lat, "target_lon": target_lon}))
        if semantic == "categorical":
            method = "spatial.nearest"
        elif semantic in {"integer_count", "cell_total", "flux", "accumulation"}:
            method = "spatial.conservative"
        elif semantic == "continuous":
            method = "spatial.bilinear"
        else:
            method = "none"
        if factor > 1:
            warnings.append("输出网格加密不等于真实空间精度提高；没有独立高分辨率真值时只能验证一致性和物理约束")
        return ScaleDecision(target_lat, target_lon, factor, method, reason, tuple(warnings))

