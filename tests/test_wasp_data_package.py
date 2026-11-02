import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

from ecodata.models.adapters.wasp_package import WASPDataPackageExporter
from ecodata.models.contracts import ScenarioInput
from ecodata.models.scenario import recognise_inputs


RAW = Path(__file__).resolve().parents[1] / "examples/real_wasp_narragansett/raw_source_data"


def _real(*paths):
    return recognise_inputs([(path.name, path.read_bytes()) for path in paths])


def test_real_usgs_nbc_noaa_and_ri_dem_generate_supported_files_only():
    ri_dem = next((RAW / "ri_dem").rglob("F4.nbc.PDB040709.RPT.txt"))
    inputs = _real(RAW / "usgs/01113895_daily_2009.rdb",
                   RAW / "nbc/2009-NBC-River-and-Bay-Nutrients-Data.xlsx",
                   RAW / "noaa/725070-14765-2009.gz", ri_dem)
    result = WASPDataPackageExporter().build(inputs)
    assert {"tributary_flow.csv", "tributary_concentration.csv", "environmental_forcing.csv",
            "observations.csv", "mapping.json", "manifest.json", "validation_report.json"} <= set(result.files)
    assert "point_source_load.csv" not in result.files
    assert result.status == {"raw_data_recognized": True, "standardized_preprocessing": True,
                             "wasp_role_matched": True, "wasp_data_package_generated": True,
                             "segment_mapping_status": "NEEDS_PROJECT_TEMPLATE"}
    mapping = json.loads(result.files["mapping.json"])
    assert mapping["segment_mapping_status"] == "NEEDS_PROJECT_TEMPLATE"
    assert all(item["segment_id"] is None for item in mapping["mappings"])
    validation = json.loads(result.files["validation_report.json"])
    assert validation["status"] == "WARN" and not validation["failures"]
    assert validation["load_derivation"]["generated_records"] == 0
    with zipfile.ZipFile(io.BytesIO(result.zip_bytes())) as archive:
        assert set(archive.namelist()) == set(result.files)


def test_point_source_load_requires_same_source_time_units_and_qa():
    time = np.array(["2009-01-01", "2009-01-02"], dtype="datetime64[ns]")
    common = {"time": ("record", time), "source_id": ("record", ["wwtp-1", "wwtp-1"]),
              "station_id": ("record", ["outfall-1", "outfall-1"]),
              "quality_code": ("record", ["A", "A"])}
    flow = xr.Dataset({**common, "flow": ("record", [1.0, 2.0])}, attrs={"data_role": "point_source_flow"})
    flow.flow.attrs["units"] = "m^3/s"
    concentration = xr.Dataset({**common, "concentration": ("record", [10.0, 20.0]),
                                "variable": ("record", ["NH3", "NH3"])},
                               attrs={"data_role": "point_source_concentration"})
    concentration.concentration.attrs["units"] = "mg/L"
    inputs = [ScenarioInput("flow.csv", flow, "a" * 64), ScenarioInput("concentration.csv", concentration, "b" * 64)]
    result = WASPDataPackageExporter().build(inputs)
    assert "point_source_load.csv" in result.files
    text = result.files["point_source_load.csv"].decode("utf-8-sig")
    assert "864.0" in text and "3456.0" in text
    assert "flow_m3_s * concentration_mg_L * 86.4" in text


def test_missing_unit_fails_and_does_not_emit_data_file():
    ds = xr.Dataset({"time": ("record", np.array(["2009-01-01"], dtype="datetime64[ns]")),
                     "station_id": ("record", ["1"]), "value": ("record", [1.0]),
                     "quality_code": ("record", ["A"]), "parameter_code": ("record", ["00060"])},
                    attrs={"data_role": "tributary_inflow"})
    result = WASPDataPackageExporter().build([ScenarioInput("flow.csv", ds, "a" * 64)])
    assert "tributary_flow.csv" not in result.files
    validation = json.loads(result.files["validation_report.json"])
    assert validation["status"] == "FAIL" and "缺少单位" in validation["failures"][0]["reason"]


def test_no_load_when_flow_is_missing():
    path = RAW / "nbc/2009-NBC-River-and-Bay-Nutrients-Data.xlsx"
    result = WASPDataPackageExporter().build(_real(path))
    assert "point_source_load.csv" not in result.files
    validation = json.loads(result.files["validation_report.json"])
    assert validation["load_derivation"]["generated_records"] == 0


def test_streamlit_wasp_flow_builds_download_from_real_usgs():
    path = RAW / "usgs/01113895_daily_2009.rdb"
    upload = SimpleNamespace(name=path.name, getvalue=path.read_bytes)
    root = Path(__file__).resolve().parents[1]
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(root / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(item for item in app.radio if item.label == "输出目标").set_value("环境模型").run()
        next(item for item in app.selectbox if item.label == "目标模型").set_value("WASP").run()
        next(item for item in app.button if item.label == "自动分析并生成方案").click().run()
    assert not app.exception and not app.error
    assert any("tributary_flow.csv" in str(item.value) for item in app.markdown)
    assert any(item.label == "下载 wasp_package.zip" for item in app.get("download_button"))
