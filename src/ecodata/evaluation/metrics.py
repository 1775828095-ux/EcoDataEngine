from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class FidelityMetrics:
    mae: float
    rmse: float
    max_abs_error: float


def fidelity_metrics(actual: xr.DataArray, expected: xr.DataArray) -> FidelityMetrics:
    actual, expected = xr.align(actual, expected, join="exact")
    error = np.asarray(actual.values - expected.values, dtype=float)
    return FidelityMetrics(
        mae=float(np.nanmean(np.abs(error))),
        rmse=float(np.sqrt(np.nanmean(error**2))),
        max_abs_error=float(np.nanmax(np.abs(error))),
    )


def spherical_cell_areas(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    lat_edges = _coordinate_edges(lat.values, clip=(-90.0, 90.0))
    lon_edges = _coordinate_edges(lon.values)
    lat_factor = np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
    lon_factor = np.deg2rad(lon_edges[1:] - lon_edges[:-1])
    area = (6_371_000.0**2) * lat_factor[:, None] * lon_factor[None, :]
    return xr.DataArray(area, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def area_integral(field: xr.DataArray) -> float:
    return float((field * spherical_cell_areas(field["lat"], field["lon"])).sum(skipna=True))


def relative_conservation_error(before: xr.DataArray, after: xr.DataArray) -> float:
    reference = area_integral(before)
    if reference == 0:
        return abs(area_integral(after))
    return abs(area_integral(after) - reference) / abs(reference)


def metadata_completeness(dataset: xr.Dataset, variable: str = "value") -> tuple[bool, tuple[str, ...]]:
    required_dataset = ("dataset_id", "crs", "temporal_semantics", "spatial_semantics")
    required_variable = ("units", "data_kind", "temporal_semantics", "spatial_semantics")
    missing = [f"dataset:{key}" for key in required_dataset if key not in dataset.attrs]
    if variable not in dataset:
        missing.append(f"variable:{variable}")
    else:
        missing.extend(f"{variable}:{key}" for key in required_variable if key not in dataset[variable].attrs)
    return not missing, tuple(missing)


def _coordinate_edges(values, clip=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("Coordinates must be one-dimensional and strictly increasing")
    mid = (values[:-1] + values[1:]) / 2
    edges = np.concatenate([[values[0] - (mid[0] - values[0])], mid, [values[-1] + (values[-1] - mid[-1])]])
    if clip is not None:
        edges = np.clip(edges, clip[0], clip[1])
    return edges

