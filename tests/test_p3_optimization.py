from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.core.pipeline import P3Pipeline
from ecodata.core.schemas import EcoDataBundle
from ecodata.optimization.operators import aggregation_operator, difference_operator
from ecodata.optimization.downscaling import adaptive_edge_weights
from ecodata.optimization.multiscale import extract_reconstruction_hints


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "datasets"


@pytest.fixture(scope="module")
def p3_data():
    coarse = xr.load_dataset(DATA / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf")
    truth = xr.load_dataset(DATA / "p3_pm25_truth_2p5km_48h.nc", engine="h5netcdf")
    train = pd.read_csv(DATA / "p3_pm25_train_stations.csv", parse_dates=["time"])
    validation = pd.read_csv(DATA / "p3_pm25_validation_stations.csv", parse_dates=["time"])
    return coarse, truth, train, validation


def test_sparse_operators_have_expected_structure() -> None:
    aggregation = aggregation_operator((2, 3), 4)
    differences, edges = difference_operator((8, 12))
    assert aggregation.shape == (6, 96)
    np.testing.assert_allclose(np.asarray(aggregation.sum(axis=1)).ravel(), 1.0)
    assert differences.shape == (8 * 11 + 7 * 12, 96)
    assert edges.shape[0] == differences.shape[0]


def test_dataset_split_has_no_station_or_cell_leakage(p3_data) -> None:
    _, _, train, validation = p3_data
    assert set(train.station_id).isdisjoint(validation.station_id)
    train_cells = set(zip(train.grid_y_index, train.grid_x_index))
    validation_cells = set(zip(validation.grid_y_index, validation.grid_x_index))
    assert train_cells.isdisjoint(validation_cells)
    assert train.time.nunique() == validation.time.nunique() == 48


def test_optimizer_rejects_validation_split(p3_data) -> None:
    coarse, truth, _, validation = p3_data
    hour = coarse.isel(time=slice(0, 1))
    validation_hour = validation.loc[validation.time.eq(validation.time.min())]
    with pytest.raises(SchemaError, match="training observations only"):
        P3Pipeline().run(
            EcoDataBundle(hour),
            "peak_aware",
            tuple(truth.lat.values),
            tuple(truth.lon.values),
            validation_hour,
        )


def test_constraints_and_peak_protection_on_known_hour(p3_data) -> None:
    coarse, truth, train, _ = p3_data
    hour_index = 8
    coarse_hour = coarse.isel(time=slice(hour_index, hour_index + 1))
    truth_hour = truth.isel(time=hour_index)
    train_hour = train.loc[train.time.eq(pd.Timestamp(coarse_hour.time.values[0]))]
    pipeline = P3Pipeline()
    constrained = pipeline.run(
        EcoDataBundle(coarse_hour), "constrained", tuple(truth.lat.values), tuple(truth.lon.values), train_hour
    )
    peak_aware = pipeline.run(
        EcoDataBundle(coarse_hour), "peak_aware", tuple(truth.lat.values), tuple(truth.lon.values), train_hour
    )
    aggregation = aggregation_operator((6, 6), 4)
    for result in (constrained, peak_aware):
        values = result.dataset["value"].values.squeeze()
        assert values.min() >= -1e-8
        np.testing.assert_allclose(
            aggregation @ values.ravel(),
            coarse_hour["value"].values.squeeze().ravel(),
            atol=1e-5,
        )
        assert result.dataset.attrs["validation_data_used"] == "false"
        assert result.dataset.attrs["solver"] == "OSQP"
    truth_peak = float(truth_hour["value"].max())
    constrained_error = abs(float(constrained.dataset["value"].max()) - truth_peak)
    peak_aware_error = abs(float(peak_aware.dataset["value"].max()) - truth_peak)
    assert peak_aware_error < constrained_error


def test_adaptive_weights_are_fixed_bounded_and_peak_sensitive() -> None:
    initial = np.array([[0.0, 1.0], [1.0, 5.0]])
    _, edges = difference_operator((2, 2))
    weights = adaptive_edge_weights(
        initial, edges, np.array([3]), np.array([8.0]), np.array([1.0]), strength=0.75, minimum=0.08
    )
    assert np.all((weights >= 0.08) & (weights <= 1.0))
    incident = (edges[:, 0] == 3) | (edges[:, 1] == 3)
    assert weights[incident].mean() < weights[~incident].mean()


@pytest.mark.parametrize("method", ["adaptive_smoothing", "adaptive_peak"])
def test_p32_methods_preserve_hard_constraints_without_validation_data(p3_data, method) -> None:
    coarse, truth, train, _ = p3_data
    hour = coarse.isel(time=slice(8, 9))
    train_hour = train.loc[train.time.eq(pd.Timestamp(hour.time.values[0]))]
    result = P3Pipeline().run(
        EcoDataBundle(hour), method, tuple(truth.lat.values), tuple(truth.lon.values), train_hour
    )
    values = result.dataset["value"].values.squeeze()
    aggregation = aggregation_operator((6, 6), 4)
    np.testing.assert_allclose(aggregation @ values.ravel(), hour["value"].values.squeeze().ravel(), atol=1e-5)
    assert values.min() >= -1e-8
    assert result.dataset.attrs["validation_data_used"] == "false"
    assert result.dataset.attrs["weights_determined_before_optimization"] == "true"
    if method == "adaptive_peak":
        assert result.dataset.attrs["peak_constraint_type"] == "explicit_soft"


def test_reconstruction_hint_interface_records_peaks_and_statistics(p3_data) -> None:
    _, truth, _, _ = p3_data
    hints = extract_reconstruction_hints(truth["value"].isel(time=0), top_k=2)
    assert len(hints.peak_values) == len(hints.peak_indices) == 2
    assert hints.peak_values[0] >= hints.peak_values[1]
    assert hints.maximum == pytest.approx(float(truth["value"].isel(time=0).max()))
