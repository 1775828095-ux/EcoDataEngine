from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.fords import ForDsPressurePlugin, source_verified
from ecodata.algorithms.open_source_bias import XsdbaPlugin
from ecodata.algorithms.open_source_time import XclimStatisticsPlugin
from ecodata.core.schemas import EcoDataBundle
from ecodata.core.exceptions import SchemaError


def daily_context(operation="mean", meaning="state"):
    a = xr.DataArray(np.arange(1., 11.), dims="time", coords={"time": pd.date_range("2000-01-01", periods=10)},
                     attrs={"units": "mm" if meaning == "interval_total" else "K", "temporal_semantics": meaning}, name="value")
    return AlgorithmContext((EcoDataBundle(a.to_dataset()),), "value", "continuous",
                            model_constraints={"output_unit": {"duration": "d", "frequency": "1"}[operation]} if operation in {"duration", "frequency"} else {},
                            parameters={"statistic": operation, "target_frequency": "5D", "threshold": "5 K"})


@pytest.mark.parametrize("operation,meaning,expected", [
    ("mean", "state", [3, 8]), ("min", "state", [1, 6]), ("max", "state", [5, 10]),
    ("sum", "interval_total", [15, 40]), ("accumulation", "interval_total", [15, 40]),
    ("frequency", "state", [1, 5]), ("duration", "state", [1, 5])])
def test_xclim_real_api_statistics(operation, meaning, expected):
    pytest.importorskip("yamale")
    plugin = XclimStatisticsPlugin()
    if plugin.spec.status != "available":
        pytest.skip("optional xclim not installed")
    result = plugin.run(daily_context(operation, meaning)).data.dataset
    np.testing.assert_allclose(result.value, expected)
    assert result.time.size == 2
    assert result.value.attrs["units"]
    np.testing.assert_array_equal(result.contributing_sample_count, [5, 5])


@pytest.mark.parametrize("problem", ["nan", "irregular", "sum_state", "no_semantics", "refinement", "no_units"])
def test_xclim_rejects_invalid_semantics_and_missing_information(problem):
    context = daily_context()
    if problem == "nan":
        context.sources[0].dataset.value.values[2] = np.nan
    elif problem == "irregular":
        context = replace(context, sources=(EcoDataBundle(context.sources[0].dataset.isel(time=[0, 1, 3, 4])),))
    elif problem == "sum_state":
        context.parameters["statistic"] = "sum"
    elif problem == "no_semantics":
        context.sources[0].dataset.value.attrs.pop("temporal_semantics")
    elif problem == "no_units":
        context.sources[0].dataset.value.attrs.pop("units")
    else:
        context.parameters["target_frequency"] = "12h"
    assert not XclimStatisticsPlugin().compatibility(context).eligible


def pressure_context():
    coords = {"lat": [35., 36.], "lon": [110., 111., 112.]}
    ds = xr.Dataset({"pressure": (("lat", "lon"), np.full((2, 3), 100000.))}, coords=coords, attrs={"crs": "EPSG:4326"})
    ds.pressure.attrs.update(units="Pa", standard_name="surface_air_pressure")
    aux = xr.Dataset({k: (("lat", "lon"), np.full((2, 3), v)) for k, v in
                      {"temperature": 280., "temperature_reference": 280., "elevation_reference": 0., "elevation": 1000.}.items()},
                     coords=coords, attrs={"crs": "EPSG:4326"})
    for name in aux:
        aux[name].attrs["units"] = "m" if name.startswith("elevation") else "K"
    return AlgorithmContext((EcoDataBundle(ds),), "pressure", "continuous", auxiliaries=(aux,))


def test_fords_unmodified_pressure_function_and_identity():
    if not source_verified():
        pytest.skip("optional pinned ForDs source unavailable")
    context = pressure_context()
    result = ForDsPressurePlugin().run(context)
    expected = 100000 * np.exp(-9.81 * 1000 / (287 * 280))
    np.testing.assert_allclose(result.data.dataset.pressure, expected)
    assert result.data.dataset.pressure.attrs["units"] == "Pa"
    xr.testing.assert_equal(result.data.dataset.lat, context.sources[0].dataset.lat)
    context.auxiliaries[0]["elevation"][:] = 0
    np.testing.assert_allclose(ForDsPressurePlugin().run(context).data.dataset.pressure, 100000)


@pytest.mark.parametrize("problem", ["units", "crs", "coordinates", "missing", "nonpositive"])
def test_fords_requires_physical_inputs(problem):
    context = pressure_context()
    aux = context.auxiliaries[0]
    if problem == "units":
        aux.temperature.attrs["units"] = "degC"
    elif problem == "crs":
        aux.attrs["crs"] = "EPSG:3857"
    elif problem == "coordinates":
        context = replace(context, auxiliaries=(aux.assign_coords(lat=[35., 37.]),))
    elif problem == "missing":
        context = replace(context, auxiliaries=(aux.drop_vars("elevation_reference"),))
    else:
        aux.temperature[:] = -1
    assert not ForDsPressurePlugin().compatibility(context).eligible


def extreme_context():
    # Synthetic unpaired daily rainfall periods, not evidence of environmental skill.
    rng = np.random.default_rng(42)
    arrays = []
    for year, scale in ((1980, 1.), (1980, 1.4), (2000, 1.6)):
        values = rng.gamma(1.5, 4 * scale, 365 * 5)
        values[::3] = 0
        arrays.append(xr.DataArray(values, dims="time", coords={"time": pd.date_range(str(year), periods=len(values))},
                                  attrs={"units": "mm/day", "split": "train"}))
    return AlgorithmContext((EcoDataBundle(arrays[2].to_dataset(name="rain")),), "rain", "continuous",
        parameters={"training_reference": arrays[0], "historical_simulation": arrays[1], "cluster_thresh": "1 mm/day",
                    "q_thresh": 0.8, "nquantiles": 20, "adjustment_kind": "*"}, model_constraints={"nonnegative": True})


def test_extreme_values_actual_official_fit_adjust_and_readback(tmp_path):
    plugin = XsdbaPlugin("extreme")
    if plugin.spec.status != "available":
        pytest.skip("optional xsdba unavailable")
    context = extreme_context()
    out = plugin.run(context).data.dataset
    assert out.rain.shape == (1825,)
    assert np.isfinite(out.rain).all() and (out.rain >= 0).all()
    assert out.rain.attrs["units"] == "mm/day"
    xr.testing.assert_equal(out.time, context.sources[0].dataset.time)
    path = tmp_path / "extreme.nc"
    out.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as reread:
        xr.testing.assert_allclose(out.rain, reread.rain)
    assert "extreme_value_correction" in plugin.spec.capabilities


@pytest.mark.parametrize("problem", ["short", "no_threshold", "nan", "overlap", "irregular", "quantile"])
def test_extremes_refuse_invalid_training(problem):
    context = extreme_context()
    if problem == "short":
        context = replace(context, sources=(EcoDataBundle(context.sources[0].dataset.isel(time=slice(0, 100))),))
    elif problem == "no_threshold":
        context.parameters.pop("cluster_thresh")
    elif problem == "nan":
        context.parameters["training_reference"].values[5] = np.nan
    elif problem == "overlap":
        context = replace(context, sources=(EcoDataBundle(context.parameters["training_reference"].to_dataset(name="rain")),))
    elif problem == "irregular":
        context = replace(context, sources=(EcoDataBundle(context.sources[0].dataset.isel(time=np.r_[0:50, 51:1825])),))
    else:
        context.parameters["q_thresh"] = 1
    assert not XsdbaPlugin("extreme").compatibility(context).eligible


def test_derived_statistics_need_explicit_output_unit_contract():
    from ecodata.evaluation.quality_gates import ResultQualityGate
    source = daily_context().sources[0].dataset
    output = source.copy(deep=True)
    output.value.attrs["units"] = "d"
    gate = ResultQualityGate()
    assert gate.validate(source, output, variable="value")["status"] == "BLOCK"
    assert gate.validate(source, output, variable="value", expected_output_unit="d")["status"] != "BLOCK"
    output.value.attrs["units"] = "mm"
    assert gate.validate(source, output, variable="value", expected_output_unit="d")["status"] == "BLOCK"


def test_research_candidates_cannot_execute_or_cover_capabilities():
    from ecodata.algorithms.registry import create_default_registry
    from ecodata.algorithms.capabilities import describe_algorithm
    registry = create_default_registry()
    for spec in registry.specs():
        if spec.id.startswith("research."):
            assert spec.verification_status in {"BLOCKED", "RESEARCH_CANDIDATE"}
            assert not describe_algorithm(spec).executable
            assert not registry.get(spec.id).compatibility(daily_context()).eligible


def test_xclim_rate_integrates_time_with_declared_unit():
    plugin = XclimStatisticsPlugin()
    if plugin.spec.status != "available":
        pytest.skip("optional xclim unavailable")
    context = daily_context("accumulation", "rate")
    context.sources[0].dataset.value.attrs["units"] = "mm/day"
    context = replace(context, model_constraints={"output_unit": "mm"})
    out = plugin.run(context).data.dataset.value
    assert out.attrs["units"] == "mm"
    np.testing.assert_allclose(out, [15, 40])


def test_extremes_reject_missing_time_without_crashing():
    context = extreme_context()
    context.parameters["training_reference"] = context.parameters["training_reference"].drop_vars("time")
    assert not XsdbaPlugin("extreme").compatibility(context).eligible
