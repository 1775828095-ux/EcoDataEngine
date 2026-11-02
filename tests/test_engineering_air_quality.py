"""Offline real-source case checks; no network downloads during regression."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.warp import transform_bounds
from ecodata.io.structures import raster_plane, read_structure
from ecodata.models.adaptation import prepare_input, suggest_mapping
from ecodata.recognition import inspect_dataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engineering_case", ROOT / "examples/run_engineering_case_air_quality.py")
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


def test_ascii_parser_rejects_shape_and_missing_axis():
    valid = b"field, [1][1][2][2]\n[0][0][0], 1, 2\n[0][0][1], 3, 4\ntime, [1]\n738887\nlat, [2]\n34,34.25\nlon, [2]\n-118,-117.75\n"
    values, axes = case.parse_ascii(valid, "field")
    np.testing.assert_array_equal(values, [[[1, 2], [3, 4]]])
    assert len(axes["time"]) == 1
    with pytest.raises(ValueError):
        case.parse_ascii(valid.replace(b"3, 4", b"3"), "field")
    with pytest.raises(ValueError):
        case.parse_ascii(valid.replace(b"lon,", b"other,"), "field")


def test_real_engineering_output_roundtrip_and_alignment():
    path = case.ROOT / "output/model_input.nc"
    if not path.exists():
        pytest.skip("Run the public-data engineering case first")
    output = read_structure(path.name, path.read_bytes())
    assert dict(output.sizes) == {"time": 8, "lat": 3, "lon": 3, "bnds": 2}
    for field, unit in case.FIELDS.items():
        assert output[field].dtype.kind == "f"
        assert output[field].attrs["units"] == unit
        assert np.isfinite(output[field]).all()
    np.testing.assert_array_equal(output.time_bnds[:, 1] - output.time_bnds[:, 0], np.repeat(np.timedelta64(1, "h"), 8))
    np.testing.assert_array_equal(output.time.values, output.time_bnds[:, 0].values + np.timedelta64(30, "m"))
    assert output.attrs["crs"] == "EPSG:4326"
    frame = pd.read_csv(case.ROOT / "observation/pm25_station.csv", dtype={"station_id": str})
    assert frame.station_id.str.len().eq(9).all()
    assert set(pd.to_datetime(frame.time)) == set(pd.to_datetime(output.time.values))
    assert frame.latitude.between(34, 34.5).all()
    assert frame.longitude.between(-118.25, -117.75).all()
    bg = read_structure("bg.nc", (case.ROOT / "background/pm25_background.nc").read_bytes())
    met = read_structure("met.nc", (case.ROOT / "meteorology/era5_sample.nc").read_bytes())
    expected_t = met.temperature.sel(time=bg.time).interp(lat=bg.lat, lon=bg.lon) - 273.15
    np.testing.assert_allclose(output.temperature, expected_t, atol=1e-10)
    fused = read_structure("fused.nc", (case.ROOT / "output/fused_pm25.nc").read_bytes())
    np.testing.assert_allclose(fused.PM25, fused.background_value + fused.residual_correction)
    np.testing.assert_array_equal(fused.background_value, bg.PM25)
    np.testing.assert_allclose(output.PM25, np.maximum(fused.PM25, 0))
    assert float(output.PM25.min()) >= 0
    report = json.loads((case.ROOT / "output/engineering_report.json").read_text(encoding="utf-8"))
    assert report["success"] and report["quality"]["passed"]
    assert report["model_matching"]["ready"]
    assert not report["model_matching"]["can_generate_native_model_input"]


def test_real_dem_is_projected_elevation_not_hillshade():
    path = case.ROOT / "terrain/dem.tif"
    if not path.exists():
        pytest.skip("Public DEM not downloaded")
    dem = read_structure(path.name, path.read_bytes())
    values, affine, crs = raster_plane(dem, "elevation")
    assert crs.to_epsg() == 3857
    assert values.shape == (128, 128)
    assert dem.elevation.attrs["units"] == "m"
    assert np.isfinite(values).all() and values.max() > 255
    from rasterio.transform import array_bounds
    bounds = transform_bounds(crs, "EPSG:4326", *array_bounds(*values.shape, affine))
    # USGS adjusts longitude extent to keep square projected pixels. Use actual
    # transform, never assume the requested geographic bbox was retained exactly.
    assert bounds[0] <= -118.375 and bounds[2] >= -117.625
    np.testing.assert_allclose([bounds[1], bounds[3]], [33.875, 34.625], atol=1e-6)


def test_netcdf_quality_accepts_dimension_insertion_order_change():
    from ecodata.evaluation.output_quality import validate_export
    from ecodata.io.structures import export_structure
    ds = xr.Dataset(coords={"lat": [34., 34.25], "lon": [-118., -117.75], "time": pd.date_range("2024-01-01", periods=2)})
    ds["temperature"] = (("time", "lat", "lon"), np.arange(8.).reshape(2, 2, 2))
    content, name, _ = export_structure(ds, "NetCDF")
    assert validate_export(ds, content, name, "NetCDF")["passed"]


def test_missing_physical_variable_does_not_pass_model_matching():
    path = case.ROOT / "output/model_input.nc"
    if not path.exists():
        pytest.skip("Run engineering case first")
    output = read_structure(path.name, path.read_bytes()).drop_vars("humidity")
    profile = json.loads((case.ROOT / "output/model_requirement.json").read_text(encoding="utf-8"))
    descriptor = inspect_dataset("missing.nc", output)
    mapping = suggest_mapping(profile["inputs"][0]["fields"], descriptor)
    _, report = prepare_input(output, descriptor, profile, mapping)
    assert not report["ready"] and "humidity" in report["missing"]
