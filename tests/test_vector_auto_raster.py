import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import shapefile
from rasterio.crs import CRS
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.interface.workflow import TrialRequest
from ecodata.io.structures import StructureRequest, vector_raster_grid, validate_geotiff
from ecodata.recognition import recognise_file
from ecodata.task import ConversionTaskEngine

ROOT = Path(__file__).resolve().parents[1]


def shp(numeric=False):
    parts = [io.BytesIO() for _ in range(3)]
    with shapefile.Writer(shp=parts[0], shx=parts[1], dbf=parts[2], shapeType=shapefile.POLYGON) as writer:
        writer.field("name", "C")
        if numeric:
            writer.field("level", "N", decimal=2)
        writer.poly([[[0, 0], [0, 1000], [1000, 1000], [1000, 0], [0, 0]]])
        writer.record("lake", *([2.5] if numeric else []))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as target:
        for ext, part in zip(("shp", "shx", "dbf"), parts):
            target.writestr("lake."+ext, part.getvalue())
        target.writestr("lake.prj", CRS.from_epsg(32618).to_wkt())
    return archive.getvalue()


@pytest.mark.parametrize("numeric", [True, False])
def test_shp_metadata_and_page_conversion(numeric):
    content = shp(numeric)
    source, info = recognise_file("lake.zip", content)
    assert info.extent == (0, 0, 1000, 1000)
    assert info.geometry_types == ("Polygon",)
    assert CRS.from_user_input(info.crs).to_epsg() == 32618
    assert info.rasterizable_fields == (("level",) if numeric else ())
    assert info.field_types["name"] in ("object", "<U4")
    upload = SimpleNamespace(name="lake.zip", getvalue=lambda: content)
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        assert not app.exception
        assert not [item for item in app.text_input if item.label in {"CRS", "输出范围 xmin,ymin,xmax,ymax"}]
        assert not [item for item in app.number_input if item.label.startswith("源空间分辨率") or item.label in {"输出列数", "输出行数"}]
        next(item for item in app.number_input if item.label.startswith("目标栅格分辨率")).set_value(100).run()
        next(item for item in app.selectbox if item.label == "输出格式").set_value("GeoTIFF").run()
        next(item for item in app.button if item.label == "执行数据转换").click().run()
        assert not app.exception
        assert not app.error
        result = app.session_state["task_result"]
        variable = "level" if numeric else "value"
        np.testing.assert_array_equal(result.output[variable], np.full((10, 10), 2.5 if numeric else 1))
        assert validate_geotiff(result.export("GeoTIFF")[0])["valid_pixels"] == 100
        assert result.report["精度指标可用"] is False
        if not numeric:
            assert result.output.value.attrs["units"] == "1"
    assert "value" not in source


@pytest.mark.parametrize("geometry,extent", [
    ({"type": "Point", "coordinates": [110, 30]}, (110, 30, 110, 30)),
    ({"type": "LineString", "coordinates": [[110, 30], [111, 30]]}, (110, 30, 111, 30)),
])
def test_geographic_point_and_line_mask(geometry, extent):
    content = json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": geometry, "properties": {"name": "site"}}]}).encode()
    source, info = recognise_file("test.geojson", content)
    assert info.extent == extent
    bounds, width, height = vector_raster_grid(info.extent, .1)
    result = ConversionTaskEngine().execute([source], StructureRequest(operation="rasterize", bounds=bounds, width=width, height=height,
        raster_mask=True, overlap="last", all_touched=True), TrialRequest(variable="value", output_format="GeoTIFF"))
    assert validate_geotiff(result.export("GeoTIFF")[0])["valid_pixels"] > 0


def test_resolution_extent_alignment_and_memory_limit():
    bounds, width, height = vector_raster_grid((0, 0, 1000, 500), 300)
    assert (bounds, width, height) == ((0, 0, 1200, 600), 4, 2)
    with pytest.raises(SchemaError, match="超过"):
        vector_raster_grid((0, 0, 1000, 1000), .01)
    with pytest.raises(SchemaError, match="正数"):
        vector_raster_grid((0, 0, 1, 1), 0)
