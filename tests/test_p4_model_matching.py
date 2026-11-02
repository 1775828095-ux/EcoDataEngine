from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from ecodata.interface.workflow import run_structure_trial
from ecodata.io.structures import StructureRequest, read_structure
from ecodata.models import RequirementStore
from ecodata.models.schema import normalize_profile, validate_profile
from ecodata.recognition import recognise_file


ROOT = Path(__file__).resolve().parents[1]


def _custom_profile(*, formats=("NetCDF",), data_type="continuous", temporal=None):
    return {
        "schema_version": "1.0", "id": "demo", "name": "Demo", "category": "测试", "version": "1",
        "reference_sources": ["https://example.test/model"], "status": "requirements_index",
        "output_format": "NetCDF", "temporal_resolution": temporal, "spatial_resolution_km": None,
        "inputs": [{"name": "temperature", "standard_name": "air_temperature", "unit": "K", "data_type": data_type,
                    "temporal_resolution": temporal, "spatial_resolution_km": None, "formats": list(formats),
                    "required": True, "depends_on": [], "aliases": ["T"]}],
    }


def test_one_store_exposes_all_24_models_with_one_schema():
    profiles = RequirementStore().all()
    assert len(profiles) == 24
    assert {item["id"] for item in profiles} >= {"calpuff", "cmaq", "wrf", "swat", "swat_plus", "wasp", "modflow6"}
    assert all(item["schema_version"] == "1.0" and item["inputs"] for item in profiles)


def test_verified_native_status_requires_real_sample_evidence():
    profile = _custom_profile()
    profile["status"] = "verified_native_adapter"
    with pytest.raises(ValueError, match="真实样例"):
        validate_profile(profile)


def test_match_report_separates_satisfied_convertible_and_missing():
    store = RequirementStore()
    profile = _custom_profile(formats=("NetCDF",))
    _, descriptor = recognise_file("temperature.csv", b"temperature\n273.15\n")
    descriptor = descriptor.__class__(**{**descriptor.as_dict(), "units": {"temperature": "K"}})
    report = store.match(profile, [descriptor], {})
    assert report["convertible"][0]["conversion"] == "CSV → NetCDF"
    assert not report["satisfied"] and not report["missing"] and not report["unsupported"]
    missing = store.match(profile, [], {})
    assert missing["missing"] == ["temperature"] and not missing["ready"]


def test_match_report_rejects_geometry_without_coordinates_or_crs():
    profile = _custom_profile(formats=("CSV",), data_type="geometry")
    _, descriptor = recognise_file("segments.csv", b"temperature\n1\n")
    report = RequirementStore.match(profile, [descriptor], {"segments.csv": "temperature"})
    assert report["unsupported"]
    assert "空间" in report["unsupported"][0]["reason"]


def test_wasp_closure_is_preprocessing_only_and_lists_missing_inputs():
    store = RequirementStore()
    profile = store.get("wasp")
    _, descriptor = recognise_file("state.csv", b"time,water_quality_state\n2026-01-01,1.2\n")
    report = store.match(profile, [descriptor], {})
    assert profile["status"] == "standardized_preprocessing"
    assert report["satisfied"][0]["requirement"] == "water_quality_state"
    assert {"segment_geometry", "flow", "boundary_concentrations", "loads", "kinetic_parameters"} <= set(report["missing"])
    assert not report["can_generate_native_model_input"]


def test_spatial_csv_geojson_netcdf_geotiff_chain_preserves_crs_and_values():
    content = (
        "station_id,longitude,latitude,crs,value,unit,qc_flag\n"
        "A,110.25,30.25,EPSG:4326,1.0,mg/L,VALID\n"
        "B,110.75,30.25,EPSG:4326,2.0,mg/L,VALID\n"
        "C,110.25,30.75,EPSG:4326,3.0,mg/L,VALID\n"
        "D,110.75,30.75,EPSG:4326,4.0,mg/L,VALID\n"
    ).encode("utf-8")
    table, descriptor = recognise_file("spatial.csv", content)
    assert descriptor.crs and descriptor.x_coordinate == "longitude" and descriptor.y_coordinate == "latitude"

    points = run_structure_trial(
        table, StructureRequest(operation="table_points", variable="value", x="longitude", y="latitude", crs="EPSG:4326", unit="mg/L"), "GeoJSON"
    )
    geojson, _, _ = points.export("GeoJSON")
    assert len(json.loads(geojson)["features"]) == 4
    vector = read_structure("points.geojson", geojson)

    grid = run_structure_trial(
        vector,
        StructureRequest(operation="rasterize", variable="value", bounds=(110.0, 30.0, 111.0, 31.0), width=2, height=2),
        "NetCDF",
    )
    netcdf, _, _ = grid.export("NetCDF")
    reopened = read_structure("grid.nc", netcdf)
    geotiff = run_structure_trial(reopened, StructureRequest(variable="value"), "GeoTIFF")
    raster = read_structure("grid.tif", geotiff.export("GeoTIFF")[0])
    assert raster.attrs["crs"] and sorted(raster.value.values.ravel()) == [1.0, 2.0, 3.0, 4.0]


def test_home_uses_simplified_wasp_data_package_flow():
    upload = SimpleNamespace(
        name="state.csv",
        getvalue=lambda: b"time,water_quality_state\n2026-01-01,1.2\n",
    )
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(item for item in app.radio if item.label == "输出目标").set_value("环境模型").run()
        next(item for item in app.selectbox if item.label == "目标模型").set_value("WASP").run()
        next(item for item in app.button if item.label == "自动分析并生成方案").click().run()
    assert not app.exception
    assert not app.error
    assert not any(item.label == "我要把当前数据转换成哪一种模型输入" for item in app.selectbox)
    role_table = next(item.value for item in app.dataframe if "自动识别角色" in item.value.columns)
    assert role_table.iloc[0]["自动识别角色"] == "unclassified"
    assert not any(item.label == "执行模型数据准备" for item in app.button)
    assert not any("WIF" in item.value or "INP" in item.value for item in app.caption)
