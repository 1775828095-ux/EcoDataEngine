from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle
from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler


def interval_bundle(values, semantics: str) -> EcoDataBundle:
    edges = pd.date_range("2026-01-01", periods=5, freq="1h").to_numpy(dtype="datetime64[ns]")
    bounds = np.column_stack([edges[:-1], edges[1:]])
    ds = xr.Dataset(
        {
            "value": ("time", np.asarray(values, dtype=float)),
            "time_bounds": (("time", "bounds"), bounds),
        },
        coords={"time": bounds[:, 1], "bounds": [0, 1]},
        attrs={
            "dataset_id": "temporal-synthetic",
            "crs": "EPSG:4326",
            "temporal_semantics": semantics,
            "spatial_semantics": "point",
            "data_kind": "continuous",
        },
    )
    ds["value"].attrs.update(
        {
            "units": "ug m-3",
            "data_kind": "continuous",
            "temporal_semantics": semantics,
            "spatial_semantics": "point",
        }
    )
    return EcoDataBundle(ds)


def test_interval_mean_uses_overlap_weighted_average() -> None:
    output = SemanticTimeResampler(SemanticTimeConfig("2h")).apply(
        interval_bundle([2, 4, 6, 8], "mean")
    ).data.dataset
    np.testing.assert_allclose(output["value"], [3, 7])
    assert output["value"].attrs["temporal_semantics"] == "mean"
    assert output["time_qc"].values.tolist() == ["AGGREGATED", "AGGREGATED"]


def test_accumulation_is_conserved_when_disaggregated() -> None:
    source = interval_bundle([1, 2, 3, 4], "accumulation")
    output = SemanticTimeResampler(SemanticTimeConfig("30min")).apply(source).data.dataset
    np.testing.assert_allclose(output["value"], [0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2])
    assert float(output["value"].sum()) == pytest.approx(float(source.dataset["value"].sum()))


def test_interval_semantics_require_valid_bounds() -> None:
    source = interval_bundle([1, 2, 3, 4], "mean")
    source.dataset["time_bounds"].values[1, 0] += np.timedelta64(5, "m")
    with pytest.raises(SchemaError, match="contiguous"):
        SemanticTimeResampler(SemanticTimeConfig("2h")).apply(source)


def test_instantaneous_grid_interpolation_has_no_extrapolation() -> None:
    time = pd.to_datetime(["2026-01-01T00:10", "2026-01-01T02:50"])
    ds = xr.Dataset(
        {"value": ("time", [10.0, 30.0])},
        coords={"time": time},
        attrs={"temporal_semantics": "instantaneous", "data_kind": "continuous"},
    )
    ds["value"].attrs.update({"temporal_semantics": "instantaneous", "data_kind": "continuous"})
    output = SemanticTimeResampler(SemanticTimeConfig("1h")).apply(EcoDataBundle(ds)).data.dataset
    expected = pd.to_datetime(["2026-01-01T01:00", "2026-01-01T02:00"]).to_numpy()
    np.testing.assert_array_equal(output.time.values, expected)
    assert not np.isnan(output.value.values).any()


def test_out_of_order_interval_time_is_rejected() -> None:
    source = interval_bundle([2, 4, 6, 8], "mean")
    disordered = EcoDataBundle(source.dataset.isel(time=[0, 2, 1, 3]), source.history)
    with pytest.raises(SchemaError, match="strictly increasing"):
        SemanticTimeResampler(SemanticTimeConfig("2h")).apply(disordered)


def _hourly_interval_bundle(values, semantics):
    edges = pd.date_range("2026-01-01", periods=len(values) + 1, freq="1h")
    bounds = np.column_stack([edges[:-1], edges[1:]]).astype("datetime64[ns]")
    ds = xr.Dataset(
        {
            "value": ("time", np.asarray(values, dtype=float)),
            "time_bounds": (("time", "bounds"), bounds),
        },
        coords={"time": bounds[:, 1], "bounds": [0, 1]},
        attrs={
            "dataset_id": "temporal-synthetic",
            "crs": "EPSG:4326",
            "temporal_semantics": semantics,
            "spatial_semantics": "point",
            "data_kind": "continuous",
        },
    )
    ds["value"].attrs.update(
        {
            "units": "ug m-3",
            "data_kind": "continuous",
            "temporal_semantics": semantics,
            "spatial_semantics": "point",
        }
    )
    return EcoDataBundle(ds)


def test_hourly_mean_to_daily_mean_is_weighted_average() -> None:
    values = np.linspace(0.0, 23.0, 24)
    output = SemanticTimeResampler(SemanticTimeConfig("1D")).apply(
        _hourly_interval_bundle(values, "mean")
    ).data.dataset
    np.testing.assert_allclose(output["value"], [float(values.mean())])
    assert output["time_qc"].values.tolist() == ["AGGREGATED"]
    assert output["time"].values[0] == pd.Timestamp("2026-01-02T00:00")


def test_hourly_accumulation_to_daily_preserves_total_and_labels_aggregation() -> None:
    values = np.arange(1.0, 25.0)
    output = SemanticTimeResampler(SemanticTimeConfig("1D")).apply(
        _hourly_interval_bundle(values, "accumulation")
    ).data.dataset
    np.testing.assert_allclose(output["value"], [float(values.sum())])
    assert output["time_qc"].values.tolist() == ["AGGREGATED"]

