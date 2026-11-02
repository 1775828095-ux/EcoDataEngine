from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from streamlit.testing.v1 import AppTest

from ecodata.interface.workflow import TrialRequest
from ecodata.io.structures import StructureRequest, read_structure
from ecodata.recognition import recognise_file
from ecodata.task import ConversionTaskEngine


ROOT = Path(__file__).resolve().parents[1]


def geotiff(crs, width, height):
    left, top = (110, 30) if crs == "EPSG:4326" else (500000, 2700000)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=3, height=2, count=1, dtype="uint8", crs=crs,
                         transform=rasterio.Affine(width, 0, left, 0, -height, top)) as target:
            target.write(np.arange(1, 7, dtype="uint8").reshape(2, 3), 1)
        return memory.read()


@pytest.mark.parametrize("crs,pixel_size,expected", [
    ("EPSG:32618", 300, 0.3),
    ("EPSG:2277", 1000, 0.3048006096012192),
])
def test_projected_resolution_converts_crs_linear_units(crs, pixel_size, expected):
    dataset, info = recognise_file("sample.tif", geotiff(crs, pixel_size, pixel_size))
    assert info.spatial_resolution_km == pytest.approx(expected)
    assert info.pixel_width_km == info.pixel_height_km == pytest.approx(expected)
    assert dataset.attrs["pixel_width"] == dataset.attrs["pixel_height"] == pixel_size
    assert "raster_transform" in dataset.attrs
    assert info.spatial_resolution_reason is None


@pytest.mark.parametrize("crs,width,height,reason", [
    ("EPSG:4326", 0.01, 0.01, "非投影坐标"),
    ("EPSG:32618", 300, 600, "像元宽高不同"),
])
def test_resolution_cannot_be_assumed_for_angular_or_rectangular_pixels(crs, width, height, reason):
    dataset, info = recognise_file("sample.tif", geotiff(crs, width, height))
    assert info.spatial_resolution_km is None
    assert "grid_resolution_km" not in dataset.attrs
    assert reason in info.spatial_resolution_reason
    if crs == "EPSG:32618":
        assert info.pixel_width_km == pytest.approx(0.3)
        assert info.pixel_height_km == pytest.approx(0.6)


@pytest.mark.parametrize("crs,width,height,expected", [
    ("EPSG:32618", 300, 300, 0.3),
    ("EPSG:4326", 0.01, 0.01, None),
])
def test_page_displays_detected_resolution_or_blank_with_reason(crs, width, height, expected):
    content = geotiff(crs, width, height)
    upload = SimpleNamespace(name="sample.tif", getvalue=lambda: content)
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
    assert not app.exception
    control = next(item for item in app.number_input if item.label.startswith("源空间分辨率"))
    assert control.value == expected
    if expected is None:
        assert any("源空间分辨率未识别" in item.value for item in app.warning)


def test_geotiff_netcdf_roundtrip_preserves_values_transform_and_resolution():
    content = geotiff("EPSG:32618", 300, 300)
    dataset, _ = recognise_file("sample.tif", content)
    engine = ConversionTaskEngine()
    result = engine.execute([dataset], StructureRequest(variable="value"), TrialRequest(variable="value", output_format="NetCDF"))
    nc, _, _ = result.export("NetCDF")
    restored = read_structure("sample.nc", nc)
    assert restored.attrs["grid_resolution_km"] == pytest.approx(0.3)
    result = engine.execute([restored], StructureRequest(variable="value"), TrialRequest(variable="value", output_format="GeoTIFF"))
    tif, _, _ = result.export("GeoTIFF")
    with MemoryFile(content) as original_memory, original_memory.open() as original, MemoryFile(tif) as output_memory, output_memory.open() as output:
        np.testing.assert_array_equal(original.read(1), output.read(1))
        assert original.crs == output.crs
        assert original.transform.almost_equals(output.transform)
