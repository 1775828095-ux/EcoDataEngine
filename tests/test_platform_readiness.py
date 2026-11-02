import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.output_quality import validate_export
from ecodata.fusion.alignment import align_and_fuse
from ecodata.fusion.weighted import UncertaintyWeightedFusion
from ecodata.io.structures import export_structure
from ecodata.interface.workflow import TrialRequest
from ecodata.fusion.standardized_residual import run_residual_task
from ecodata.io.structures import table_dataset


def grid(value=1., unit="mg/L", frequency="2h"):
    times = pd.date_range("2021-01-01", periods=1 + 4 // int(frequency[0]), freq=frequency)
    shape = (len(times), 3, 3)
    ds = xr.Dataset({"value": (("time", "lat", "lon"), np.full(shape, value)),
                     "uncertainty": (("time", "lat", "lon"), np.full(shape, 1. if unit == "mg/L" else .001))},
                    coords={"time": times, "lat": [2700500., 2701500., 2702500.], "lon": [500500., 501500., 502500.]},
                    attrs={"crs": "EPSG:32618", "grid_resolution_km": 1., "structure": "grid"})
    ds.value.attrs.update(units=unit, physical_variable="COD", temporal_semantics="instantaneous")
    ds.uncertainty.attrs.update(units=unit)
    return ds


def test_align_units_time_and_resolution_then_fuse():
    result, report = align_and_fuse([grid(), grid(.003, "g/L", "4h")], "value", frequency="1h", resolution=.5)
    assert result.dataset.value.shape == (5, 6, 6)
    np.testing.assert_allclose(result.dataset.value, 2.)
    assert len(report["steps"]) >= 3
    content, name, _ = export_structure(result.dataset, "NetCDF")
    assert validate_export(result.dataset, content, name, "NetCDF")["passed"]


def test_missing_measurement_does_not_bias_weighted_fusion_to_zero():
    first, second = grid(), grid(3.)
    second.value.values[0, 0, 0] = np.nan
    result = UncertaintyWeightedFusion().apply([EcoDataBundle(first), EcoDataBundle(second)]).data.dataset
    assert result.value.values[0, 0, 0] == 1.
    assert result.value.values[0, 0, 1] == 2.


def test_alignment_excludes_invalid_quality_without_losing_provenance():
    first, second = grid(), grid(100.)
    second["qc_flag"] = xr.full_like(second.value, "INVALID", dtype="U7")
    result, report = align_and_fuse([first, second], "value")
    np.testing.assert_allclose(result.dataset.value, 1.)
    assert (result.dataset.qc_flag == "FUSED_VALID").all()
    assert any(step.get("original_flags") == {"INVALID": 27} for step in report["steps"])


@pytest.mark.parametrize("problem", ["crs", "unit", "variable", "uncertainty", "time"])
def test_alignment_rejects_unreliable_sources(problem):
    first, second = grid(), grid(3.)
    if problem == "crs":
        second.attrs.pop("crs")
    elif problem == "unit":
        second.value.attrs["units"] = "unknown"
    elif problem == "variable":
        second.value.attrs["physical_variable"] = "TN"
    elif problem == "uncertainty":
        second.uncertainty.values[:] = 0
    else:
        second = second.assign_coords(time=second.time + np.timedelta64(10, "D"))
    with pytest.raises(SchemaError):
        align_and_fuse([first, second], "value")


def test_quality_rejects_all_nan_and_empty_files():
    ds = grid()
    with pytest.raises(SchemaError, match="为空"):
        validate_export(ds, b"", "empty.nc", "NetCDF")
    ds = ds.drop_vars("uncertainty")
    ds.value.values[:] = np.nan
    content, name, _ = export_structure(ds, "NetCDF")
    with pytest.raises(SchemaError, match="全部"):
        validate_export(ds, content, name, "NetCDF")


@pytest.mark.parametrize("problem", ["crs", "variable", "unit", "coordinate"])
def test_quality_detects_actual_export_loss(problem):
    ds = grid()
    corrupted = ds.copy(deep=True)
    if problem == "crs":
        corrupted.attrs.pop("crs")
    elif problem == "variable":
        corrupted = corrupted.drop_vars("value")
    elif problem == "unit":
        corrupted.value.attrs.pop("units")
    else:
        corrupted = corrupted.assign_coords(lon=corrupted.lon+1)
    content, name, _ = export_structure(corrupted, "NetCDF")
    with pytest.raises(SchemaError):
        validate_export(ds, content, name, "NetCDF")


def test_existing_residual_fusion_keeps_background_and_quality_flags():
    bg = grid().isel(time=slice(0, 1)).assign_coords(lat=[30., 30.01, 30.02], lon=[110., 110.01, 110.02])
    bg.attrs["crs"] = "EPSG:4326"
    stations = table_dataset(pd.DataFrame({"station_id": ["a", "b", "c", "bad"],
        "time": [pd.Timestamp(bg.time.values[0])]*4, "latitude": [30., 30., 30.02, 30.01],
        "longitude": [110., 110.02, 110., 110.01], "variable": ["COD"]*4,
        "value": [.003, .003, .003, .1], "unit": ["g/L"]*4, "qc_flag": ["VALID"]*3+["INVALID"]}), crs="EPSG:4326")
    result = run_residual_task([bg, stations], TrialRequest(variable="value", fusion_method="residual"))
    np.testing.assert_allclose(result.output.value, 3.)
    np.testing.assert_allclose(result.output.background_value, 1.)
    result.export("NetCDF")
    assert result.report["自动质量检查"]["passed"]


def test_export_accepts_equivalent_crs84_and_wgs84_without_changing_pixels():
    ds = grid().isel(time=0, drop=True).assign_coords(lat=[30., 30.01, 30.02], lon=[110., 110.01, 110.02])
    ds.attrs.update(crs="OGC:CRS84", export_variable="value")
    content, filename, _ = export_structure(ds, "GeoTIFF")
    assert validate_export(ds, content, filename, "GeoTIFF")["passed"]
