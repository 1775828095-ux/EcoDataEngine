import numpy as np
import pytest
import xarray as xr
from rasterio.io import MemoryFile

from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import export_structure, validate_geotiff
from ecodata.recognition import recognise_file


def projected(values):
    return xr.Dataset({"value": (("lat", "lon"), values)},
                      coords={"lat": [2700000, 2700300], "lon": [500000, 500300]},
                      attrs={"crs": "EPSG:32618", "spatial_x": "lon", "spatial_y": "lat"})


def test_projected_labels_and_nodata_roundtrip():
    dataset = projected([[1., np.nan], [3., 4.]])
    nc, _, _ = export_structure(dataset, "NetCDF")
    restored, info = recognise_file("test.nc", nc)
    assert set(restored.dims) == {"x", "y"}
    np.testing.assert_array_equal(restored.x, dataset.lon)
    np.testing.assert_array_equal(restored.y, dataset.lat)
    content, _, _ = export_structure(restored, "GeoTIFF")
    check = validate_geotiff(content)
    assert (check["width"], check["height"], check["bands"], check["valid_pixels"]) == (2, 2, 1, 3)
    assert check["CRS"] == "EPSG:32618"
    assert check["nodata"] == "NaN"
    with MemoryFile(content) as memory, memory.open() as source:
        np.testing.assert_allclose(source.read(1), [[3, 4], [1, np.nan]], equal_nan=True)


def test_all_missing_is_rejected():
    with pytest.raises(SchemaError, match="全部为 NaN"):
        export_structure(projected(np.full((2, 2), np.nan)), "GeoTIFF")


def test_corrupt_file_is_rejected():
    with pytest.raises(SchemaError, match="回读检查失败"):
        validate_geotiff(b"not a TIFF")


def test_geographic_names_are_retained():
    dataset = projected([[1., 2.], [3., 4.]])
    dataset.attrs["crs"] = "EPSG:4326"
    nc, _, _ = export_structure(dataset, "NetCDF")
    restored, _ = recognise_file("test.nc", nc)
    assert set(restored.dims) == {"lat", "lon"}
