import io
import json
import zipfile
from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapefile
import xarray as xr
from rasterio.io import MemoryFile

from ecodata.core.exceptions import SchemaError
from ecodata.interface.workflow import run_structure_trial
from ecodata.io.structures import (
    GEOMETRY, StructureRequest, describe_hdf, export_structure, read_structure, table_dataset, table_frame,
)


@pytest.fixture
def grid():
    ds = xr.Dataset({"value": (("lat", "lon"), np.array([[1., np.nan, 3.], [4., 5., 6.]]))}, coords={"lat": [30., 31.], "lon": [110., 111., 112.]}, attrs={"crs": "EPSG:4326", "source": "synthetic"})
    ds.value.attrs.update(units="ug m-3", physical_variable="PM2.5")
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    return ds


@pytest.fixture
def mixed_table():
    return table_dataset(pd.DataFrame({"station_id": ["001", "002", "003", "004"], "longitude": [110., 111., 110., 111.], "latitude": [30., 30., 31., 31.], "value": [1., 2., 3., 4.], "qc_flag": ["VALID"]*4, "label": ["甲", "乙", "丙", "丁"]}))


@pytest.mark.parametrize("format", ["CSV", "TSV", "JSON", "Excel", "Parquet", "NetCDF", "HDF5-NetCDF"])
def test_mixed_table_format_roundtrip(mixed_table, format):
    original = mixed_table.copy(deep=True)
    result = run_structure_trial(mixed_table, StructureRequest(unit="ug m-3"), format)
    payload, name, _ = result.export(format)
    reopened = read_structure(name, payload)
    frame = table_frame(reopened)
    assert list(frame.station_id.astype(str)) == ["001", "002", "003", "004"]
    assert list(frame.label) == ["甲", "乙", "丙", "丁"]
    np.testing.assert_allclose(pd.to_numeric(frame.value), [1., 2., 3., 4.])
    assert list(frame.qc_flag) == ["VALID"]*4
    assert result.report["精度指标可用"] is False
    assert result.history[0]["method"] == "structure.convert"
    if format not in {"CSV", "TSV", "JSON"}:
        assert reopened.value.attrs["units"] == "ug m-3"
    xr.testing.assert_identical(mixed_table, original)


def test_table_to_grid_keeps_text_and_quality(mixed_table):
    result = run_structure_trial(mixed_table, StructureRequest(operation="table_to_grid", crs="EPSG:4326", unit="ug m-3"), "NetCDF")
    assert result.output.value.shape == (2, 2)
    assert result.output.station_id.sel(latitude=30, longitude=110).item() == "001"
    assert result.output.label.sel(latitude=31, longitude=111).item() == "丁"
    assert result.output.value.attrs["units"] == "ug m-3"
    payload, name, _ = result.export("NetCDF")
    reopened = read_structure(name, payload)
    np.testing.assert_allclose(reopened.value, [[1, 2], [3, 4]])
    assert set(reopened.qc_flag.values.ravel()) == {"VALID"}


def test_table_duplicate_coordinates_rejected(mixed_table):
    frame = table_frame(mixed_table)
    source = table_dataset(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(SchemaError, match="重复"):
        run_structure_trial(source, StructureRequest(operation="table_to_grid"), "NetCDF")


def test_incomplete_numeric_table_leaves_missing_grid_cells():
    source = table_dataset(pd.DataFrame({"longitude": [0., 1., 0.], "latitude": [0., 0., 1.], "value": [1., 2., 3.]}))
    result = run_structure_trial(source, StructureRequest(operation="table_to_grid"), "NetCDF")
    assert np.isnan(result.output.value.sel(latitude=1, longitude=1))
    assert any("缺少 1" in message for message in result.report["注意事项"])


def test_grid_geotiff_roundtrip_preserves_values_crs_nodata_and_unit(grid):
    result = run_structure_trial(grid, StructureRequest(), "GeoTIFF")
    payload, name, _ = result.export("GeoTIFF")
    with MemoryFile(payload) as memory, memory.open() as raster:
        assert raster.crs.to_epsg() == 4326
        assert raster.bounds.left == 109.5
        assert raster.bounds.top == 31.5
        assert np.isnan(raster.nodata)
        np.testing.assert_allclose(raster.read(1), grid.value.values[::-1], equal_nan=True)
    reopened = read_structure(name, payload)
    assert reopened.value.attrs["units"] == "ug m-3"
    np.testing.assert_allclose(reopened.value.values, grid.value.values[::-1], equal_nan=True)
    np.testing.assert_allclose(reopened.x, grid.lon)
    np.testing.assert_allclose(reopened.y, grid.lat[::-1])
    nc = run_structure_trial(reopened, StructureRequest(), "NetCDF")
    restored = read_structure("restored.nc", nc.export("NetCDF")[0])
    np.testing.assert_allclose(restored.value, reopened.value, equal_nan=True)


def test_multidimensional_raster_requires_slice(grid):
    source = grid.expand_dims(time=pd.date_range("2026-01-01", periods=2))
    with pytest.raises(SchemaError, match="二维切片"):
        run_structure_trial(source, StructureRequest(), "GeoTIFF")
    result = run_structure_trial(source, StructureRequest(slices=(("time", 1),)), "GeoTIFF")
    assert str(result.output.time.values).startswith("2026-01-02")
    assert result.history[0]["parameters"]["slices"] == (("time", 1),)
    restored = read_structure("slice.tif", result.export("GeoTIFF")[0])
    assert restored.time.values == source.time.values[1]


def test_raw_hdf_explicit_mapping_and_fill_value():
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        value = handle.create_dataset("science/data", data=np.array([[1., -9999.], [3., 4.]]))
        value.attrs["_FillValue"] = -9999.
        value.attrs["units"] = "K"
        handle.create_dataset("x", data=[100., 101.])
        handle.create_dataset("y", data=[30., 31.])
    content = buffer.getvalue()
    assert {item["path"] for item in describe_hdf(content)} == {"science/data", "x", "y"}
    with pytest.raises(SchemaError, match="明确指定"):
        read_structure("raw.h5", content, hdf_value="science/data")
    ds = read_structure("raw.h5", content, hdf_value="science/data", hdf_x="x", hdf_y="y", crs="EPSG:4326")
    assert np.isnan(ds.value.values[0, 1])
    result = run_structure_trial(ds, StructureRequest(), "GeoTIFF")
    restored = read_structure("result.tif", result.export("GeoTIFF")[0])
    assert restored.value.attrs["units"] == "K"
    np.testing.assert_allclose(restored.value, ds.value[::-1], equal_nan=True)


def polygon_source():
    payload = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"class_id": 1}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 2], [0, 2], [0, 0]]]}},
        {"type": "Feature", "properties": {"class_id": 2}, "geometry": {"type": "Polygon", "coordinates": [[[1, 0], [2, 0], [2, 2], [1, 2], [1, 0]]]}},
    ]}
    return read_structure("regions.geojson", json.dumps(payload).encode())


def test_polygon_rasterize_polygonize_roundtrip():
    vector = polygon_source()
    request = StructureRequest(operation="rasterize", variable="class_id", bounds=(0., 0., 2., 2.), width=2, height=2)
    raster = run_structure_trial(vector, request, "GeoTIFF")
    np.testing.assert_array_equal(raster.output.class_id, [[1, 2], [1, 2]])
    polygon = run_structure_trial(raster.output, StructureRequest(operation="polygonize", variable="class_id"), "GeoJSON")
    payload = json.loads(polygon.export("GeoJSON")[0])
    assert len(payload["features"]) == 2
    assert {item["geometry"]["type"] for item in payload["features"]} == {"Polygon"}
    again = run_structure_trial(polygon.output, request, "GeoTIFF")
    np.testing.assert_array_equal(raster.output.class_id, again.output.class_id)


def test_rasterization_overlap_rule_is_explicit():
    source = polygon_source()
    frame = table_frame(source)
    duplicate = frame.iloc[[0]].copy()
    duplicate["class_id"] = 9
    source = table_dataset(pd.concat([frame, duplicate], ignore_index=True), crs="EPSG:4326")
    request = StructureRequest(operation="rasterize", variable="class_id", bounds=(0., 0., 2., 2.), width=2, height=2)
    with pytest.raises(SchemaError, match="覆盖同一像元"):
        run_structure_trial(source, request, "GeoTIFF")
    first = run_structure_trial(source, replace(request, overlap="first"), "GeoTIFF")
    last = run_structure_trial(source, replace(request, overlap="last"), "GeoTIFF")
    assert first.output.class_id.values[0, 0] == 1
    assert last.output.class_id.values[0, 0] == 9


def test_table_points_shapefile_zip_roundtrip(mixed_table):
    result = run_structure_trial(mixed_table, StructureRequest(operation="table_points", crs="EPSG:4326"), "Shapefile ZIP")
    content = result.export("Shapefile ZIP")[0]
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert {"result.shp", "result.shx", "result.dbf", "result.prj", "result.cpg", "ecodata_metadata.json"}.issubset(archive.namelist())
        with shapefile.Reader(shp=io.BytesIO(archive.read("result.shp")), dbf=io.BytesIO(archive.read("result.dbf")), shx=io.BytesIO(archive.read("result.shx")), encoding="utf-8") as reader:
            assert reader.records()[0].as_dict()["station_id"] == "001"
            assert reader.records()[0].as_dict()["label"] == "甲"
            np.testing.assert_array_equal(reader.shapes()[0].points[0], [110., 30.])
    restored = read_structure("points.zip", content)
    assert table_frame(restored).label.tolist() == ["甲", "乙", "丙", "丁"]
    geo = run_structure_trial(restored, StructureRequest(), "GeoJSON")
    assert json.loads(geo.export("GeoJSON")[0])["features"][0]["properties"]["station_id"] == "001"


def test_projected_points_reproject_for_geojson():
    ds = table_dataset(pd.DataFrame({"x": [111319.49079327357], "y": [0.], "value": [2.]}))
    result = run_structure_trial(ds, StructureRequest(operation="table_points", x="x", y="y", crs="EPSG:3857"), "GeoJSON")
    coordinates = json.loads(result.export("GeoJSON")[0])["features"][0]["geometry"]["coordinates"]
    np.testing.assert_allclose(coordinates, [1., 0.], atol=1e-8)


def test_grid_points_keep_missing_and_attributes(grid):
    result = run_structure_trial(grid, StructureRequest(operation="grid_points"), "GeoJSON")
    payload = json.loads(result.export("GeoJSON")[0])
    assert len(payload["features"]) == 6
    assert payload["features"][1]["properties"]["value"] is None
    assert result.output.value.attrs["units"] == "ug m-3"


def test_missing_crs_irregular_grid_and_continuous_polygonize_rejected(grid):
    no_crs = grid.copy()
    no_crs.attrs = {}
    with pytest.raises(SchemaError, match="CRS"):
        run_structure_trial(no_crs, StructureRequest(), "GeoTIFF")
    with pytest.raises(SchemaError, match="等间隔"):
        run_structure_trial(grid.assign_coords(lon=[110., 111., 114.]), StructureRequest(), "GeoTIFF")
    continuous = grid.copy(deep=True)
    continuous.value.values[0, 0] = 1.2
    with pytest.raises(SchemaError, match="整数分类"):
        run_structure_trial(continuous, StructureRequest(operation="polygonize"), "GeoJSON")
    with pytest.raises(SchemaError, match="不能通过更改标签"):
        run_structure_trial(grid, StructureRequest(crs="EPSG:3857"), "NetCDF")


def test_shapefile_long_fields_not_silently_truncated(mixed_table):
    source = mixed_table.rename({"label": "a_very_long_field"})
    with pytest.raises(SchemaError, match="10 字符"):
        run_structure_trial(source, StructureRequest(operation="table_points", crs="EPSG:4326"), "Shapefile ZIP")
