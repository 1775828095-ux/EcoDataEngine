from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SemanticError
from ecodata.core.pipeline import SequentialPipeline
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.metrics import fidelity_metrics, metadata_completeness
from ecodata.evaluation.validation import P2Validator
from ecodata.fusion.residual import ResidualFieldFusion
from ecodata.fusion.weighted import UncertaintyWeightedFusion
from ecodata.io.readers import _frame_to_dataset
from ecodata.spatial.regridding import GridRegridder, RegridConfig
from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler


def source(value: float, uncertainty: float, unit="ug m-3") -> EcoDataBundle:
    ds = xr.Dataset(
        {
            "value": (("lat", "lon"), np.full((2, 2), value)),
            "uncertainty": (("lat", "lon"), np.full((2, 2), uncertainty)),
        },
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        attrs={"dataset_id": "source", "crs": "EPSG:4326"},
    )
    ds["value"].attrs.update(
        {"units": unit, "physical_variable": "PM2.5", "data_kind": "continuous"}
    )
    ds["uncertainty"].attrs["units"] = unit
    return EcoDataBundle(ds)


def test_inverse_variance_fusion() -> None:
    result = UncertaintyWeightedFusion().apply((source(10, 1), source(14, 2)))
    np.testing.assert_allclose(result.data.dataset["value"], 10.8)
    np.testing.assert_allclose(result.data.dataset["uncertainty"], np.sqrt(0.8))
    assert result.data.dataset.attrs["source_count"] == 2
    assert result.record.elapsed_seconds is not None


def test_weighted_fusion_rejects_unit_mismatch() -> None:
    with pytest.raises(SemanticError, match="units"):
        UncertaintyWeightedFusion().apply((source(10, 1), source(14, 2, "mg m-3")))


def test_weighted_fusion_rejects_different_physical_variables() -> None:
    other = source(14, 2)
    other.dataset["value"].attrs["physical_variable"] = "O3"
    with pytest.raises(SemanticError, match="physical variables"):
        UncertaintyWeightedFusion().apply((source(10, 1), other))


def background_and_observations():
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 1.0, 2.0])
    truth = lat[:, None] + lon[None, :]
    background = xr.Dataset(
        {"value": (("lat", "lon"), truth - 2.0)},
        coords={"lat": lat, "lon": lon},
        attrs={
            "dataset_id": "biased-background",
            "crs": "EPSG:4326",
            "temporal_semantics": "instantaneous",
            "spatial_semantics": "cell_mean",
            "data_kind": "continuous",
        },
    )
    background["value"].attrs.update(
        {
            "units": "ug m-3",
            "physical_variable": "PM2.5",
            "data_kind": "continuous",
            "temporal_semantics": "instantaneous",
            "spatial_semantics": "cell_mean",
        }
    )
    points = [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]
    frame = pd.DataFrame(
        [
            [f"S{i}", pd.Timestamp("2026-01-01"), la, lo, "PM2.5", la + lo, "ug m-3", "VALID"]
            for i, (la, lo) in enumerate(points)
        ],
        columns=["station_id", "time", "latitude", "longitude", "variable", "value", "unit", "qc_flag"],
    )
    observations = _frame_to_dataset(frame)
    observations.attrs.update({"dataset_id": "stations", "crs": "EPSG:4326"})
    return EcoDataBundle(background), EcoDataBundle(observations), truth


def test_residual_fusion_removes_known_constant_bias() -> None:
    background, observations, truth = background_and_observations()
    result = ResidualFieldFusion().apply(background, observations)
    metrics = fidelity_metrics(result.data.dataset["value"], xr.DataArray(truth, dims=("lat", "lon"), coords={"lat": [0, 1, 2], "lon": [0, 1, 2]}))
    assert metrics.rmse < 1e-12
    np.testing.assert_allclose(result.data.dataset["background_value"], background.dataset["value"])
    assert result.data.dataset.attrs["observation_count"] == 5
    report = P2Validator().validate(
        result.data,
        xr.DataArray(truth, dims=("lat", "lon"), coords={"lat": [0, 1, 2], "lon": [0, 1, 2]}),
    )
    assert report.passed


def test_sequential_pipeline_connects_temporal_and_spatial_transforms() -> None:
    time = pd.to_datetime(["2026-01-01T00:00", "2026-01-01T02:00"])
    lat = np.array([0.0, 2.0])
    lon = np.array([0.0, 2.0])
    values = np.stack([lat[:, None] + lon[None, :], 2 + lat[:, None] + lon[None, :]])
    ds = xr.Dataset(
        {"value": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "dataset_id": "pipeline-synthetic",
            "crs": "EPSG:4326",
            "temporal_semantics": "instantaneous",
            "spatial_semantics": "cell_mean",
            "data_kind": "continuous",
        },
    )
    ds["value"].attrs.update(
        {
            "units": "ug m-3",
            "data_kind": "continuous",
            "temporal_semantics": "instantaneous",
            "spatial_semantics": "cell_mean",
        }
    )
    result = SequentialPipeline().run(
        EcoDataBundle(ds),
        [
            SemanticTimeResampler(SemanticTimeConfig("1h")),
            GridRegridder(RegridConfig("bilinear", (0, 1, 2), (0, 1, 2))),
        ],
    )
    assert result.methods == ("temporal.semantic_resample", "spatial.regrid")
    assert result.data.dataset["value"].shape == (3, 3, 3)
    assert len(result.data.history) == 2
    assert result.original.dataset["value"].shape == (2, 2, 2)
    complete, missing = metadata_completeness(result.data.dataset)
    assert complete, missing
