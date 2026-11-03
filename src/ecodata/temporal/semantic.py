from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult


SUPPORTED_SEMANTICS = {"instantaneous", "mean", "accumulation"}


@dataclass(frozen=True)
class SemanticTimeConfig:
    frequency: str
    variable: str = "value"
    interpolation: str = "linear"


class SemanticTimeResampler:
    name = "temporal.semantic_resample"

    def __init__(self, config: SemanticTimeConfig):
        self.config = config

    def validate_input(self, data: EcoDataBundle) -> None:
        if self.config.interpolation not in {"linear","nearest"}:
            raise SemanticError("Instantaneous interpolation must be linear or nearest")
        dataset = data.dataset
        if self.config.variable not in dataset:
            raise SchemaError(f"Variable '{self.config.variable}' is missing")
        if "time" not in dataset[self.config.variable].dims:
            raise SchemaError("Temporal resampling requires a time dimension")
        time_index = pd.DatetimeIndex(dataset["time"].values)
        if not time_index.is_monotonic_increasing or time_index.has_duplicates:
            raise SchemaError("Time must be unique and strictly increasing; out-of-order input is rejected")
        semantics = _semantics(dataset, self.config.variable)
        if semantics not in SUPPORTED_SEMANTICS:
            raise SemanticError(f"Unsupported temporal semantics: {semantics!r}")
        if dataset[self.config.variable].attrs.get("data_kind", dataset.attrs.get("data_kind")) != "continuous":
            raise SemanticError("Temporal resampling requires continuous data")
        try:
            pd.tseries.frequencies.to_offset(self.config.frequency)
        except ValueError as exc:
            raise SchemaError(f"Invalid target frequency: {self.config.frequency}") from exc
        if semantics != "instantaneous":
            _validated_bounds(dataset)

    def apply(self, data: EcoDataBundle) -> TransformResult:
        self.validate_input(data)
        started = perf_counter()
        semantics = _semantics(data.dataset, self.config.variable)
        if semantics == "instantaneous":
            output = self._instantaneous(data.dataset)
            action = "interpolated"
        else:
            output = self._interval_resample(data.dataset, semantics)
            action = "overlap_weighted"

        output.attrs.update(data.dataset.attrs)
        output.attrs.update(
            {
                "processing_stage": "time_resampled",
                "time_frequency": self.config.frequency,
                "time_resampling_action": action,
            }
        )
        output[self.config.variable].attrs.update(data.dataset[self.config.variable].attrs)
        record = TransformRecord(
            method=self.name,
            parameters={
                "frequency": self.config.frequency,
                "variable": self.config.variable,
                "temporal_semantics": semantics,
                "extrapolation": False,
            },
            input_rows=int(data.dataset[self.config.variable].size),
            output_rows=int(output[self.config.variable].size),
            elapsed_seconds=perf_counter() - started,
        )
        bundle = EcoDataBundle(output, data.history + (record,))
        return TransformResult(bundle, record)

    def _instantaneous(self, dataset: xr.Dataset) -> xr.Dataset:
        source_time = pd.DatetimeIndex(dataset["time"].values)
        target = pd.date_range(
            source_time.min().ceil(self.config.frequency),
            source_time.max().floor(self.config.frequency),
            freq=self.config.frequency,
        )
        if target.empty:
            raise SchemaError("Source range contains no target timestamps")
        output = dataset[[self.config.variable]].interp(time=target,method=self.config.interpolation)
        output["time_qc"] = ("time", np.full(target.size, "INTERPOLATED", dtype=str))
        output["time_qc"].attrs["long_name"] = "temporal processing quality flag"
        return output

    def _interval_resample(self, dataset: xr.Dataset, semantics: str) -> xr.Dataset:
        source_bounds = _validated_bounds(dataset)
        start = pd.Timestamp(source_bounds[0, 0])
        end = pd.Timestamp(source_bounds[-1, 1])
        edges = pd.date_range(start, end, freq=self.config.frequency)
        if len(edges) < 2 or edges[-1] != end:
            raise SchemaError("Target frequency must exactly cover the source interval")
        target_bounds = np.column_stack(
            [edges[:-1].to_numpy(dtype="datetime64[ns]"), edges[1:].to_numpy(dtype="datetime64[ns]")]
        )
        if np.array_equal(source_bounds,target_bounds):
            # Identity intervals preserve missing values locally and need no
            # quadratic overlap matrix. Native labels remain interval-end.
            output=dataset[[self.config.variable,'time_bounds']].copy(deep=True)
            output=output.assign_coords(time=target_bounds[:,1])
            output['time_qc']=('time',np.full(len(target_bounds),'AGGREGATED'))
            output['time_qc'].attrs['long_name']='temporal processing quality flag'
            return output
        overlap = _overlap_seconds(source_bounds, target_bounds)
        source_seconds = _duration_seconds(source_bounds)
        target_seconds = _duration_seconds(target_bounds)
        coverage = overlap.sum(axis=0)
        if not np.allclose(coverage, target_seconds):
            raise SchemaError("Source intervals do not completely cover every target interval")

        if semantics == "mean":
            weights = overlap / target_seconds[None, :]
            time_qc = "AGGREGATED" if len(source_bounds) >= len(target_bounds) else "DISAGGREGATED"
        else:
            weights = overlap / source_seconds[:, None]
            time_qc = "AGGREGATED" if len(source_bounds) >= len(target_bounds) else "REDISTRIBUTED"

        source = dataset[self.config.variable].transpose("time", ...)
        finite=np.isfinite(source.values)
        values = np.tensordot(weights.T, np.where(finite,source.values,0), axes=(1, 0))
        unknown=np.tensordot((weights.T>0).astype(np.int64),(~finite).astype(np.int64),axes=(1,0))>0
        values=np.where(unknown,np.nan,values)
        dims = ("time",) + source.dims[1:]
        coords = {name: source.coords[name] for name in source.dims[1:]}
        coords["time"] = target_bounds[:, 1]
        output = xr.Dataset(
            {
                self.config.variable: (dims, values),
                "time_bounds": (("time", "bounds"), target_bounds),
                "time_qc": ("time", np.full(len(target_bounds), time_qc)),
            },
            coords={**coords, "bounds": [0, 1]},
        )
        output["time_bounds"].attrs["long_name"] = "time interval bounds"
        output["time_qc"].attrs["long_name"] = "temporal processing quality flag"
        return output


def _semantics(dataset: xr.Dataset, variable: str) -> str | None:
    return dataset[variable].attrs.get("temporal_semantics", dataset.attrs.get("temporal_semantics"))


def _validated_bounds(dataset: xr.Dataset) -> np.ndarray:
    if "time_bounds" not in dataset:
        raise SchemaError("Mean and accumulation data require time_bounds")
    bounds = dataset["time_bounds"].values.astype("datetime64[ns]")
    if bounds.ndim != 2 or bounds.shape != (dataset.sizes["time"], 2):
        raise SchemaError("time_bounds must have shape (time, 2)")
    start = bounds[:, 0].astype("int64")
    end = bounds[:, 1].astype("int64")
    if np.any(end <= start) or np.any(start[1:] != end[:-1]):
        raise SchemaError("time_bounds must be positive, ordered and contiguous")
    return bounds


def _duration_seconds(bounds: np.ndarray) -> np.ndarray:
    return (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")


def _overlap_seconds(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_start = source[:, 0].astype("datetime64[ns]").astype("int64")[:, None]
    source_end = source[:, 1].astype("datetime64[ns]").astype("int64")[:, None]
    target_start = target[:, 0].astype("datetime64[ns]").astype("int64")[None, :]
    target_end = target[:, 1].astype("datetime64[ns]").astype("int64")[None, :]
    overlap_ns = np.maximum(0, np.minimum(source_end, target_end) - np.maximum(source_start, target_start))
    return overlap_ns / 1_000_000_000.0

