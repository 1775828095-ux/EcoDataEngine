from __future__ import annotations

import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.interface.workflow import TrialRequest, run_grid_trial, run_multi_grid_trial
from ecodata.optimization.downscaling import QPOptions


DATA = Path(__file__).resolve().parents[1] / "examples" / "datasets"


@pytest.fixture
def uploaded_inputs():
    source = xr.load_dataset(DATA / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf").isel(time=slice(8, 9))
    train = pd.read_csv(DATA / "p3_pm25_train_stations.csv", parse_dates=["time"])
    train = train.loc[train.time.eq(pd.Timestamp(source.time.values[0]))].reset_index(drop=True)
    return source, train


def qp_request(**overrides) -> TrialRequest:
    return replace(
        TrialRequest(
            target_spatial_resolution_km=2.5,
            spatial_method="constrained",
            variable="value",
            temporal_semantics="instantaneous",
            unit="ug m-3",
            target_model="custom",
        ),
        **overrides,
    )


def assert_consistent(fine: xr.DataArray, coarse: xr.DataArray) -> None:
    restored = fine.coarsen(lat=4, lon=4, boundary="exact").mean()
    np.testing.assert_allclose(restored.values, coarse.values, atol=1e-5, rtol=0)
    assert float(fine.min()) >= -1e-5


@pytest.mark.parametrize("method", ["constrained", "peak_aware", "adaptive_smoothing", "adaptive_peak"])
def test_uploaded_qp_methods_roundtrip_without_reference_leakage(uploaded_inputs, method) -> None:
    source, train = uploaded_inputs
    original = source.copy(deep=True)
    original_train = train.copy(deep=True)
    options = QPOptions(initial_method="nearest", alpha=1.2, beta=18.0, gamma=0.6, peak_threshold=40.0,
                        observation_tolerance=0.4, soft_constraint_penalty=70.0, max_iter=25_000,
                        eps_abs=5e-6, eps_rel=5e-6)
    result = run_grid_trial(source, qp_request(spatial_method=method, optimization_options=options),
                            training_observations=train)
    assert result.output["value"].shape == (1, 24, 24)
    assert_consistent(result.output["value"], source["value"])
    np.testing.assert_array_equal(result.output.time.values, source.time.values)
    for attr in ("units", "physical_variable", "temporal_semantics"):
        assert result.output["value"].attrs[attr] == source["value"].attrs[attr]
    assert result.output.attrs["validation_data_used"] == "false"
    assert result.output.attrs["training_station_count"] == train.station_id.nunique()
    assert result.report["精度指标可用"] is False
    assert "精度指标" not in result.report
    assert result.report["优化诊断"]["粗网格算术平均一致性最大误差"] < 1e-5
    parameters = result.history[-1]["parameters"]
    for name, value in asdict(options).items():
        assert parameters[name] == value
    assert len(parameters["solver_iterations"]) == 1
    assert 0 < parameters["solver_iterations"][0] <= options.max_iter
    assert parameters["solver_statuses"] == ["optimal"]
    with xr.open_dataset(io.BytesIO(result.netcdf_bytes())) as reopened:
        np.testing.assert_allclose(reopened["value"], result.output["value"])
        assert json.loads(reopened.attrs["transform_history"])[-1]["parameters"]["max_iter"] == 25_000
        assert reopened["value"].attrs["units"] == "ug m-3"
    xr.testing.assert_identical(source, original)
    pd.testing.assert_frame_equal(train, original_train)


def test_uploaded_qp_accepts_selected_variable_and_dimension_order(uploaded_inputs) -> None:
    source, train = uploaded_inputs
    source = source.rename({"value": "pm25"}).transpose("lon", "time", "lat")
    result = run_grid_trial(source, qp_request(variable="pm25"), training_observations=train)
    assert "pm25" in result.output and "value" not in result.output
    assert_consistent(result.output.pm25, source.pm25.transpose("time", "lat", "lon"))
    assert result.output.pm25.attrs["physical_variable"] == "PM2.5"


def test_multi_source_fusion_then_uploaded_qp_preserves_history(uploaded_inputs) -> None:
    first, train = uploaded_inputs
    first["uncertainty"] = xr.full_like(first["value"], 1.0)
    first.uncertainty.attrs["units"] = "ug m-3"
    second = first.copy(deep=True)
    second["value"].values += 4.0
    second["uncertainty"] = xr.full_like(first["value"], 2.0)
    second.uncertainty.attrs["units"] = "ug m-3"
    result = run_multi_grid_trial([first, second], qp_request(), training_observations=train)
    assert result.history[0]["method"] == "fusion.uncertainty_weighted"
    assert result.history[-1]["method"] == "downscale.constrained_baseline"
    assert result.report["算法决策"][0]["步骤"] == "多源融合"
    assert_consistent(result.output["value"], first["value"] + 0.8)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("validation", "split=train"), ("test", "split=train"),
        ("missing_split", "缺少字段"), ("invalid_qc", "非 VALID"),
        ("duplicate", "重复"), ("outside", "超出"),
        ("time_mismatch", "全部时间点"), ("wrong_unit", "单位不一致"),
        ("wrong_variable", "physical_variable 不一致"),
        ("missing_stations", "训练监测站"),
    ],
)
def test_invalid_training_uploads_are_rejected(uploaded_inputs, case, message) -> None:
    source, train = uploaded_inputs
    if case in {"validation", "test"}:
        train.loc[0, "split"] = case
    elif case == "missing_split":
        train = train.drop(columns="split")
    elif case == "invalid_qc":
        train.loc[0, "qc_flag"] = "INVALID"
    elif case == "duplicate":
        train = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    elif case == "outside":
        train.loc[0, "longitude"] = 0.0
    elif case == "time_mismatch":
        train["time"] += pd.Timedelta(days=1)
    elif case == "wrong_unit":
        train["unit"] = "mg m-3"
    elif case == "wrong_variable":
        train["variable"] = "NO2"
    elif case == "missing_stations":
        train = None
    with pytest.raises(SchemaError, match=message):
        run_grid_trial(source, qp_request(), training_observations=train)


@pytest.mark.parametrize(
    ("case", "message"),
    [("negative", "有限且非负"), ("categorical", "连续物理变量"),
     ("coarser", "目标 km 必须小于源 km"), ("same_resolution", "目标 km 必须小于源 km"),
     ("mean", "瞬时值"), ("unit_relabel", "明确的单位转换")],
)
def test_incompatible_grid_or_request_is_rejected(uploaded_inputs, case, message) -> None:
    source, train = uploaded_inputs
    request = qp_request()
    if case == "negative":
        source["value"].values[0, 0, 0] = -1.0
    elif case == "categorical":
        request = replace(request, semantic_type="categorical")
    elif case == "coarser":
        request = replace(request, target_spatial_resolution_km=20.0)
    elif case == "same_resolution":
        request = replace(request, target_spatial_resolution_km=10.0)
    elif case == "mean":
        request = replace(request, temporal_semantics="mean")
    elif case == "unit_relabel":
        request = replace(request, unit="mg m-3")
    with pytest.raises(SchemaError, match=message):
        run_grid_trial(source, request, training_observations=train)


def test_training_must_cover_new_timestamps_after_temporal_interpolation(uploaded_inputs) -> None:
    source, train = uploaded_inputs
    next_time = pd.Timestamp(source.time.values[0]) + pd.Timedelta(hours=1)
    source = xr.concat([source, source.assign_coords(time=[next_time])], dim="time")
    train = pd.concat([train, train.assign(time=next_time)], ignore_index=True)
    with pytest.raises(SchemaError, match="全部时间点"):
        run_grid_trial(source, qp_request(target_time_frequency="30min"), training_observations=train)


@pytest.mark.parametrize(("case", "message"), [("missing_station", "全部训练站"), ("moving_station", "坐标随时间变化")])
def test_station_coverage_and_position_are_consistent_across_time(uploaded_inputs, case, message) -> None:
    source, train = uploaded_inputs
    next_time = pd.Timestamp(source.time.values[0]) + pd.Timedelta(hours=1)
    source = xr.concat([source, source.assign_coords(time=[next_time])], dim="time")
    second = train.assign(time=next_time)
    if case == "missing_station":
        second = second.iloc[1:]
    else:
        second.loc[0, "latitude"] += 0.001
    train = pd.concat([train, second], ignore_index=True)
    with pytest.raises(SchemaError, match=message):
        run_grid_trial(source, qp_request(), training_observations=train)


def test_tiny_iteration_limit_reports_failure_instead_of_success(uploaded_inputs) -> None:
    source, train = uploaded_inputs
    with pytest.raises(SchemaError, match="OSQP"):
        run_grid_trial(source, qp_request(optimization_options=QPOptions(max_iter=1)),
                       training_observations=train)
