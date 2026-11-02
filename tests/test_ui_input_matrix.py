import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

from ecodata.interface.workflow import run_structure_trial
from ecodata.io.structures import StructureRequest, read_structure


ROOT = Path(__file__).resolve().parents[1]


def _upload(name, content):
    return SimpleNamespace(name=name, getvalue=lambda: content)


def _cases():
    frame = pd.DataFrame({"断面名称": ["A"], "监测时间": ["2021-02-01 00:00:00"], "站点编码": ["001"], "总磷": [0.1], "总氮": [2.0]})
    excel = io.BytesIO(); frame.to_excel(excel, index=False)
    parquet = io.BytesIO(); frame.to_parquet(parquet, index=False)
    grid = xr.Dataset({"value": (("lat", "lon"), np.arange(4, dtype=float).reshape(2, 2))}, coords={"lat": [30., 31.], "lon": [110., 111.]}, attrs={"crs": "EPSG:4326", "grid_resolution_km": 10.0})
    grid.value.attrs["units"] = "mg/L"
    grid.attrs.update(source_file="grid.nc", source_sha256="test")
    tif = run_structure_trial(grid, StructureRequest(variable="value"), "GeoTIFF").export("GeoTIFF")[0]
    geojson = b'{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"value":1},"geometry":{"type":"Point","coordinates":[110,30]}}]}'
    vector = read_structure("points.geojson", geojson)
    shp = run_structure_trial(vector, StructureRequest(variable="value"), "Shapefile ZIP").export("Shapefile ZIP")[0]
    return [
        _upload("water.csv", frame.to_csv(index=False).encode("utf-8-sig")),
        _upload("water.xlsx", excel.getvalue()),
        _upload("water.parquet", parquet.getvalue()),
        _upload("grid.nc", bytes(grid.to_netcdf())),
        _upload("grid.tif", tif),
        _upload("points.geojson", geojson),
        _upload("points.zip", shp),
    ]


@pytest.mark.parametrize("upload", _cases(), ids=lambda item: item.name)
def test_supported_input_types_render_without_uncaught_error(upload):
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert any(item.label == "输出格式" for item in app.selectbox)


def test_malformed_input_is_explained_in_page_not_raised():
    with patch("streamlit.file_uploader", return_value=[_upload("broken.xlsx", b"not an excel workbook")]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.error


def test_unit_headers_are_normalized_in_task_page():
    content = (
        "断面名称,化学需氧量(mg/L),氨氮(mg/L),总磷(mg/L),总氮(mg/L)\n"
        "站点A,5.1,0.2,0.1,2.0\n"
    ).encode("utf-8-sig")
    with patch("streamlit.file_uploader", return_value=[_upload("water.csv", content)]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()

    assert not app.exception
    selector = next(item for item in app.multiselect if item.label == "需要处理的变量（可多选）")
    assert {"化学需氧量", "氨氮", "总磷", "总氮"} <= set(selector.options)
    assert not any("(mg/L)" in name for name in selector.options)
    units = {item.label: item.value for item in app.text_input}
    assert all(units[name] == "mg/L" for name in ("化学需氧量", "氨氮", "总磷", "总氮"))
    recognition = next(item.value for item in app.caption if item.value.startswith("识别类型："))
    assert all(f"{name}＝连续数值" in recognition for name in ("化学需氧量", "氨氮", "总磷", "总氮"))
    with patch("streamlit.file_uploader", return_value=[_upload("water.csv", content)]):
        next(item for item in app.button if item.label == "执行数据转换").click().run()
    assert not app.exception
    assert not app.error
    result = app.session_state["task_result"]
    payload, _, _ = result.export("NetCDF")
    reopened = read_structure("water.nc", payload)
    for name in ("化学需氧量", "氨氮", "总磷", "总氮"):
        assert reopened[name].dtype.kind == "f"
        assert reopened[name].attrs["units"] == "mg/L"
