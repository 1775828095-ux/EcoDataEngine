from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult


def normalize_grid_axes(dataset: xr.Dataset, axes=("lat", "lon")):
    """Reverse descending axes together with all dependent arrays."""
    reversed_axes = []
    for axis in axes:
        if axis not in dataset.dims:
            continue
        values = np.asarray(dataset[axis].values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise SchemaError(f"空间坐标 {axis} 必须是一维有限数值")
        differences = np.diff(values)
        if np.all(differences > 0):
            continue
        if np.all(differences < 0):
            dataset = dataset.isel({axis: slice(None, None, -1)})
            reversed_axes.append(axis)
        else:
            raise SchemaError(f"空间坐标 {axis} 存在重复或非单调排列，无法自动翻转")
    return dataset, reversed_axes


@dataclass(frozen=True)
class RegridConfig:
    method: str
    target_lat: tuple[float, ...]
    target_lon: tuple[float, ...]
    variable: str = "value"


class GridRegridder:
    name = "spatial.regrid"

    def __init__(self, config: RegridConfig):
        self.config = config

    def validate_input(self, data: EcoDataBundle) -> None:
        dataset = data.dataset
        if self.config.method not in {"nearest", "bilinear", "conservative"}:
            raise SchemaError(f"Unsupported regridding method: {self.config.method}")
        if self.config.variable not in dataset:
            raise SchemaError(f"Variable '{self.config.variable}' is missing")
        if not {"lat", "lon"}.issubset(dataset[self.config.variable].dims):
            raise SchemaError("Spatial regridding requires rectilinear lat and lon dimensions")
        if len(self.config.target_lat) < 2 or len(self.config.target_lon) < 2:
            raise SchemaError("Target grid requires at least two coordinates per axis")
        if not (_strictly_increasing(dataset["lat"].values) and _strictly_increasing(dataset["lon"].values)):
            raise SchemaError("Source lat/lon coordinates must be strictly increasing")
        if not (_strictly_increasing(self.config.target_lat) and _strictly_increasing(self.config.target_lon)):
            raise SchemaError("Target lat/lon coordinates must be strictly increasing")
        kind = dataset[self.config.variable].attrs.get("data_kind", dataset.attrs.get("data_kind"))
        if self.config.method != "nearest" and kind != "continuous":
            raise SemanticError("Bilinear and conservative regridding require continuous data")
        spatial = dataset[self.config.variable].attrs.get(
            "spatial_semantics", dataset.attrs.get("spatial_semantics")
        )
        if self.config.method == "conservative" and spatial not in {"cell_mean", "flux"}:
            raise SemanticError("Conservative regridding requires cell_mean or flux spatial semantics")

    def apply(self, data: EcoDataBundle) -> TransformResult:
        normalized, reversed_axes = normalize_grid_axes(data.dataset)
        data = EcoDataBundle(normalized, data.history)
        self.validate_input(data)
        started = perf_counter()
        variable = self.config.variable
        selected = data.dataset[[variable]]
        target = xr.Dataset(coords={"lat": np.asarray(self.config.target_lat), "lon": np.asarray(self.config.target_lon)})
        from ecodata.engines.grid import GridEngine
        output, backend = GridEngine().regrid(selected, target, self.config.method)
        output.attrs.update(data.dataset.attrs)
        output.attrs.update(
            {"processing_stage": "spatial_regridded", "spatial_regridding_method": self.config.method}
        )
        output[variable].attrs.update(data.dataset[variable].attrs)
        for axis in ("lat", "lon"):
            output[axis].attrs.update(data.dataset[axis].attrs)
        record = TransformRecord(
            method=f"{self.name}.{self.config.method}",
            parameters={"method": self.config.method, "variable": variable, "reversed_axes": reversed_axes, **backend},
            input_rows=int(data.dataset[variable].size),
            output_rows=int(output[variable].size),
            elapsed_seconds=perf_counter() - started,
        )
        bundle = EcoDataBundle(output, data.history + (record,))
        return TransformResult(bundle, record)


def _strictly_increasing(values) -> bool:
    values = np.asarray(values, dtype=float)
    return values.ndim == 1 and np.isfinite(values).all() and np.all(np.diff(values) > 0)
