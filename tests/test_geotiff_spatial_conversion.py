from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle
from ecodata.interface.workflow import TrialRequest, recommended_spatial_resolutions, target_grid
from ecodata.io.structures import StructureRequest, read_structure
from ecodata.recognition import recognise_file
from ecodata.spatial.regridding import GridRegridder, RegridConfig
from ecodata.task import ConversionTaskEngine


ROOT = Path(__file__).resolve().parents[1]


def sample_tiff(*, reverse_x=False, metadata_unit=None, band_unit=None):
    values = np.arange(1, 17, dtype="float64").reshape(4, 4)
    transform = rasterio.Affine(-10000 if reverse_x else 10000, 0, 500000, 0, -10000, 2700000)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", height=4, width=4, count=1, dtype="float64", crs="EPSG:32618", transform=transform, nodata=0) as output:
            output.write(values, 1)
            if metadata_unit:
                output.update_tags(1, units=metadata_unit)
            if band_unit:
                output.set_band_unit(1, band_unit)
        return memory.read()


def run(content, method, target_km, unit="unknown"):
    dataset, _ = recognise_file("source.tif", content)
    request = TrialRequest(variable="value", unit=unit, target_spatial_resolution_km=target_km,
                           spatial_method=method, spatial_semantics="cell_mean", output_format="NetCDF")
    return ConversionTaskEngine().execute(
        [dataset], StructureRequest(variable="value", x="x", y="y", unit=unit or ""), request,
    )


@pytest.mark.parametrize("reverse_x", [False, True])
def test_coordinate_flip_preserves_pixel_location_values_crs_and_bounds(reverse_x):
    content = sample_tiff(reverse_x=reverse_x)
    original = read_structure("source.tif", content)
    result = run(content, "nearest", 10)
    restored = read_structure("result.nc", result.export("NetCDF")[0])
    assert np.all(np.diff(restored.y) > 0) and np.all(np.diff(restored.x) > 0)
    for y in original.y.values:
        for x in original.x.values:
            assert restored.value.sel(y=y, x=x).item() == original.value.sel(y=y, x=x).item()
    assert rasterio.crs.CRS.from_wkt(restored.attrs["crs"]).to_epsg() == 32618
    exported = result.export("GeoTIFF")[0]
    with MemoryFile(content) as original_memory, original_memory.open() as before, MemoryFile(exported) as output_memory, output_memory.open() as after:
        np.testing.assert_allclose(tuple(before.bounds), tuple(after.bounds) if not reverse_x else
                                   (after.bounds.right, after.bounds.bottom, after.bounds.left, after.bounds.top))
    assert any(item["method"] == "spatial.coordinate_orientation" for item in result.history)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "conservative"])
@pytest.mark.parametrize("target_km", [5, 2.5, 1, 3])
def test_geotiff_spatial_resolution_supports_regular_and_noninteger_refinement(method, target_km):
    result = run(sample_tiff(), method, target_km)
    output = read_structure("result.nc", result.export("NetCDF")[0])
    count = round(4 * 10 / target_km)
    assert output.value.shape == (count, count)
    assert output.value.attrs["units"] == "unknown"
    assert np.isfinite(output.value).all()
    assert np.all(np.diff(output.y) > 0) and np.all(np.diff(output.x) > 0)
    dx, dy = float(output.x[1] - output.x[0]), float(output.y[1] - output.y[0])
    assert float(output.x.min()) - dx / 2 == pytest.approx(500000)
    assert float(output.x.max()) + dx / 2 == pytest.approx(540000)
    assert float(output.y.min()) - dy / 2 == pytest.approx(2660000)
    assert float(output.y.max()) + dy / 2 == pytest.approx(2700000)
    assert result.report["空间分辨率"]["请求_km"] == target_km
    assert result.report["空间分辨率"]["实际x_km"] == pytest.approx(40 / count)
    if method == "conservative":
        # Independent planar-area integral, not spherical latitude weights.
        expected_integral = np.arange(1, 17).sum() * 10000 ** 2
        assert float(output.value.sum()) * dx * dy == pytest.approx(expected_integral, rel=1e-10)


@pytest.mark.parametrize("metadata_unit,band_unit,expected", [("mg/L", None, "mg/L"), (None, "K", "K")])
def test_geotiff_unit_metadata_used_without_manual_input(metadata_unit, band_unit, expected):
    content = sample_tiff(metadata_unit=metadata_unit, band_unit=band_unit)
    dataset, info = recognise_file("source.tif", content)
    assert info.units["value"] == expected
    result = run(content, "nearest", 5, unit=None)
    assert read_structure("result.nc", result.export("NetCDF")[0]).value.attrs["units"] == expected


@pytest.mark.parametrize("unit", ["unknown", "1", "无量纲"])
def test_user_confirmed_unknown_or_dimensionless_unit_saved(unit):
    result = run(sample_tiff(), "bilinear", 5, unit=unit)
    assert read_structure("result.nc", result.export("NetCDF")[0]).value.attrs["units"] == unit


def test_unit_prompt_and_recommended_resolution_through_page():
    content = sample_tiff()
    upload = SimpleNamespace(name="source.tif", getvalue=lambda: content)
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        assert not app.exception and not app.error
        recommendation = next(item for item in app.selectbox if item.label == "推荐目标空间分辨率")
        assert {"5 km（细化）", "2.5 km（细化）", "1 km（细化）"} <= set(recommendation.options)
        recommendation.set_value(5.0).run()
        assert next(item for item in app.number_input if item.label.startswith("目标空间分辨率")).value == 5
        button = next(item for item in app.button if item.label == "执行数据转换")
        assert button.disabled
        next(item for item in app.text_input if item.label == "value").set_value("unknown").run()
        next(item for item in app.button if item.label == "执行数据转换").click().run()
    assert not app.exception and not app.error
    result = app.session_state["task_result"]
    assert result.output.value.shape == (8, 8)
    assert result.output.value.attrs["units"] == "unknown"


def test_recommendations_obey_memory_limit():
    dataset, _ = recognise_file("source.tif", sample_tiff())
    assert 5 in recommended_spatial_resolutions(dataset, 10)
    large = dataset.isel(y=0, x=0, drop=True).expand_dims(y=np.arange(1000), x=np.arange(1000))
    choices = recommended_spatial_resolutions(large, 10)
    assert 5 not in choices and 2.5 not in choices and 1 not in choices


def test_direct_regrid_handles_both_descending_axes_without_mutating_source():
    import xarray as xr
    dataset = xr.Dataset({"value": (("lat", "lon"), [[1., 2.], [3., 4.]])}, coords={"lat": [2., 1.], "lon": [2., 1.]})
    dataset.value.attrs["data_kind"] = "continuous"
    before = dataset.copy(deep=True)
    output = GridRegridder(RegridConfig("bilinear", (1., 2.), (1., 2.))).apply(EcoDataBundle(dataset)).data.dataset
    np.testing.assert_array_equal(output.value, [[4., 3.], [2., 1.]])
    xr.testing.assert_identical(dataset, before)


def test_conservative_preserves_nodata_instead_of_turning_it_into_zero():
    import xarray as xr
    dataset = xr.Dataset({"value": (("lat", "lon"), [[np.nan, 2.], [3., 4.]])}, coords={"lat": [1., 2.], "lon": [1., 2.]})
    dataset.value.attrs.update(data_kind="continuous", spatial_semantics="cell_mean")
    output = GridRegridder(RegridConfig("conservative", (1., 2.), (1., 2.))).apply(EcoDataBundle(dataset)).data.dataset
    np.testing.assert_allclose(output.value, dataset.value, equal_nan=True)


@pytest.mark.parametrize("bad_target", [0, -1, np.nan, np.inf, 1e-100, 1000])
def test_invalid_or_impractical_resolutions_report_reason(bad_target):
    dataset, _ = recognise_file("source.tif", sample_tiff())
    grid = dataset.rename({"x": "lon", "y": "lat"})
    with pytest.raises(SchemaError):
        target_grid(grid, 10, bad_target)
