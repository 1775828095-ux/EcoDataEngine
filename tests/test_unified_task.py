from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.interface.workflow import TrialRequest, run_structure_trial
from ecodata.io.structures import StructureRequest, read_structure
from ecodata.models.requirements import RequirementStore
from ecodata.optimization.downscaling import QPOptions
from ecodata.recognition import recognise_file
from ecodata.task import ConversionTaskEngine


ROOT = Path(__file__).resolve().parents[1]


def test_home_keeps_conversion_model_preparation_and_fusion_experiment_navigation():
    app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.sidebar.radio) == 1
    assert app.sidebar.radio[0].options == ["数据转换", "模型数据准备", "融合重构实验", "首轮专项转换"]
    assert app.sidebar.radio[0].value == "数据转换"
    assert any("数据转换任务" in item.value for item in app.subheader)


def test_recognition_reports_structure_semantics_and_confirmation():
    content = b"station_id,longitude,latitude,value\n001,110,30,4.5\n002,111,31,5.5\n"
    _, info = recognise_file("sample.csv", content)
    assert info.structure == "table"
    assert info.x_coordinate == "longitude" and info.y_coordinate == "latitude"
    assert info.semantic_types["station_id"] == "string_identifier"
    assert info.semantic_types["value"] == "continuous"
    assert "变量单位" in info.confirmations_required


def test_chinese_water_table_recognition_hides_missing_spatial_semantics():
    content = "断面名称,监测时间,自动监测站编码,总磷,总氮,高锰酸盐指数,氨氮\n站点A,2021-02-01 00:00:00,A001,0.1,2.0,3.0,0.2\n".encode("utf-8-sig")
    _, info = recognise_file("water.csv", content)
    assert info.time_coordinate == "监测时间"
    assert info.x_coordinate is None and info.y_coordinate is None and info.crs is None
    assert {name for name, kind in info.semantic_types.items() if kind == "continuous"} == {"总磷", "总氮", "高锰酸盐指数", "氨氮"}


def test_units_and_year_month_are_recognised_from_realistic_headers():
    content = "断面名称,采样时间,监测年份,监测月份,水质类别,氨氮(mg/L),化学需氧量(mg/L)\nA,2021-01-01,2021,1,III,0.2,5.2\nB,2021-02-01,2021,2,II,,6.1\n".encode("utf-8-sig")
    dataset, info = recognise_file("water.csv", content)
    assert {"氨氮", "化学需氧量"} <= set(info.variables)
    assert info.units["氨氮"] == info.units["化学需氧量"] == "mg/L"
    assert info.semantic_types["氨氮"] == info.semantic_types["化学需氧量"] == "continuous"
    assert dataset["氨氮"].attrs["source_field_name"] == "氨氮(mg/L)"
    assert dataset["化学需氧量"].attrs["source_field_name"] == "化学需氧量(mg/L)"
    assert info.semantic_types["采样时间"] == "datetime"
    assert info.semantic_types["监测年份"] == "datetime"
    assert info.semantic_types["监测月份"] == "datetime"
    assert info.semantic_types["水质类别"] == "categorical"
    _, english = recognise_file("panel.csv", b"year,month,month_start,TP_mg_L,TP_flag\n2021,1,2021-01-01,0.1,VALID\n")
    assert english.time_coordinate == "month_start"
    assert english.units["TP_mg_L"] == "mg/L"
    assert english.semantic_types["TP_flag"] == "quality_metadata"
    _, missing_numeric = recognise_file("missing.csv", "站点,总磷(mg/L)\nA,0.1\nB,\n".encode("utf-8-sig"))
    assert missing_numeric.semantic_types["总磷"] == "continuous"


def test_unit_header_split_rejects_duplicate_clean_variable_names():
    content = "氨氮,氨氮(mg/L)\n0.1,0.2\n".encode("utf-8-sig")
    with pytest.raises(SchemaError, match="重名"):
        recognise_file("duplicate.csv", content)


@pytest.mark.parametrize("source_format", ["csv", "xlsx"])
def test_water_quality_table_netcdf_roundtrip_preserves_semantic_dtypes_and_units(source_format):
    frame = pd.DataFrame({
        "断面名称": ["站点A", "站点B"],
        "监测年份": [2021, 2021],
        "监测月份": [1, 2],
        "水质类别": ["III", "II"],
        "自动监测站编码": ["001", "002"],
        "氨氮(mg/L)": ["0.11", "0.20"],
        "化学需氧量(mg/L)": ["5.2", "6.1"],
        "总磷(mg/L)": ["0.01", ""],
    })
    if source_format == "csv":
        filename = "water.csv"
        content = frame.to_csv(index=False).encode("utf-8-sig")
    else:
        from io import BytesIO
        buffer = BytesIO()
        frame.to_excel(buffer, index=False)
        filename = "water.xlsx"
        content = buffer.getvalue()

    dataset, info = recognise_file(filename, content)
    assert dataset["氨氮"].dtype.kind == dataset["化学需氧量"].dtype.kind == "f"
    assert dataset["监测年份"].dtype.kind == dataset["监测月份"].dtype.kind == "i"
    assert dataset["水质类别"].dtype.kind in "OUS"
    result = ConversionTaskEngine().execute(
        [dataset], StructureRequest(operation="preserve", variable="氨氮"),
        TrialRequest(variable="氨氮", output_format="NetCDF"),
    )
    payload, _, _ = result.export("NetCDF")
    reopened = read_structure("water.nc", payload)
    assert reopened["氨氮"].dtype.kind == reopened["化学需氧量"].dtype.kind == "f"
    assert reopened["监测年份"].dtype.kind == reopened["监测月份"].dtype.kind == "i"
    assert reopened["水质类别"].dtype.kind in "OUS"
    assert reopened["氨氮"].attrs["units"] == reopened["化学需氧量"].attrs["units"] == "mg/L"
    np.testing.assert_allclose(reopened["氨氮"], [0.11, 0.20])
    np.testing.assert_allclose(reopened["总磷"], [0.01, np.nan], equal_nan=True)
    assert reopened["水质类别"].values.tolist() == ["III", "II"]
    assert reopened["自动监测站编码"].values.tolist() == ["001", "002"]
    assert reopened["氨氮"].attrs["source_field_name"] == "氨氮(mg/L)"


def test_missing_year_month_store_as_integer_with_fill_value():
    from io import BytesIO
    dataset, _ = recognise_file("missing.csv", b"year,month,value\n2021,1,0.11\n,,\n")
    result = run_structure_trial(dataset, StructureRequest(variable="value"), "NetCDF")
    payload, _, _ = result.export("NetCDF")
    with xr.open_dataset(BytesIO(payload), engine="h5netcdf", mask_and_scale=False) as raw:
        assert raw.year.dtype == raw.month.dtype == np.dtype("int32")
        assert raw.year.values.tolist() == [2021, np.iinfo(np.int32).min]
    reopened = read_structure("missing.nc", payload)
    np.testing.assert_allclose(reopened.year, [2021, np.nan], equal_nan=True)
    np.testing.assert_allclose(reopened.month, [1, np.nan], equal_nan=True)


def test_netcdf_recognition_preserves_existing_numeric_storage_types():
    dataset = xr.Dataset({"value": ("row", np.array([1, 2], dtype=np.int64))})
    from ecodata.io.structures import export_structure
    payload, _, _ = export_structure(dataset, "NetCDF")
    reopened, _ = recognise_file("integers.nc", payload)
    assert reopened.value.dtype == np.dtype("int64")


def test_editable_requirement_store_roundtrip(tmp_path):
    store = RequirementStore(tmp_path / "requirements.json")
    profile = {"schema_version": "1.0", "id": "demo", "name": "Demo", "category": "测试", "version": "1",
               "reference_sources": ["https://example.test/model"], "status": "requirements_index", "output_format": "NetCDF",
               "temporal_resolution": "1h", "spatial_resolution_km": 5.0,
               "inputs": [{"name": "temperature", "standard_name": "air_temperature", "unit": "K", "data_type": "continuous",
                           "temporal_resolution": "1h", "spatial_resolution_km": 5.0, "formats": ["CSV", "NetCDF"],
                           "required": True, "depends_on": [], "aliases": []}]}
    store.save(profile)
    saved = store.get("demo")
    assert saved["id"] == "demo" and saved["inputs"] == profile["inputs"]
    assert json.loads((tmp_path / "requirements.json").read_text(encoding="utf-8"))[0]["id"] == "demo"
    _, descriptor = recognise_file("temperature.csv", b"time,temperature,unit\n2026-01-01T00:00:00,273.15,K\n")
    descriptor = descriptor.__class__(**{**descriptor.as_dict(), "units": {"temperature": "K", "unit": ""}})
    match = store.match(profile, [descriptor], {})
    assert match["ready"] and match["matched"] == ["temperature"]


def test_unified_engine_structure_then_spatial_conversion():
    content = b"longitude,latitude,value\n110,30,1\n111,30,2\n110,31,3\n111,31,4\n"
    source = read_structure("grid.csv", content)
    source.attrs["grid_resolution_km"] = 10.0
    result = ConversionTaskEngine().execute(
        [source],
        StructureRequest(operation="table_to_grid", variable="value", x="longitude", y="latitude", crs="EPSG:4326", unit="ug m-3"),
        TrialRequest(target_spatial_resolution_km=5.0, variable="value", unit="ug m-3", temporal_semantics="instantaneous", output_format="NetCDF"),
    )
    assert result.output.value.shape == (4, 4)
    stages = [item["阶段"] for item in result.report["自动处理流程"]]
    assert stages == ["数据识别", "结构转换", "空间处理", "输出"]


def test_unified_engine_fuses_after_matching_scale():
    coords = {"lat": [0.0, 1.0], "lon": [0.0, 1.0]}
    attrs = {"grid_resolution_km": 10.0, "crs": "EPSG:4326"}
    first = xr.Dataset({"value": (("lat", "lon"), np.ones((2, 2))), "uncertainty": (("lat", "lon"), np.ones((2, 2)))}, coords=coords, attrs=attrs)
    second = xr.Dataset({"value": (("lat", "lon"), np.full((2, 2), 3.0)), "uncertainty": (("lat", "lon"), np.ones((2, 2)))}, coords=coords, attrs=attrs)
    for dataset in (first, second):
        dataset.value.attrs["units"] = "ug m-3"
        dataset.value.attrs["physical_variable"] = "PM2.5"
        dataset.uncertainty.attrs["units"] = "ug m-3"
        dataset.attrs["source_file"] = "source.nc"
        dataset.attrs["source_sha256"] = "abc"
    result = ConversionTaskEngine().execute(
        [first, second], StructureRequest(variable="value", unit="ug m-3"),
        TrialRequest(variable="value", unit="ug m-3", output_format="NetCDF"), fuse=True,
    )
    np.testing.assert_allclose(result.output.value, 2.0)
    assert result.report["算法决策"][0]["步骤"] == "多源融合"


@pytest.mark.parametrize("solver", ["OSQP", "CLARABEL", "SCS"])
def test_unified_engine_passes_user_qp_options_and_training_stations(solver):
    time = pd.to_datetime(["2026-01-01 00:00:00"])
    source = xr.Dataset(
        {"value": (("time", "lat", "lon"), [[[10.0, 12.0], [14.0, 16.0]]])},
        coords={"time": time, "lat": [30.0, 31.0], "lon": [110.0, 111.0]},
        attrs={"grid_resolution_km": 10.0, "temporal_semantics": "instantaneous"},
    )
    source.value.attrs.update(units="ug m-3", physical_variable="PM2.5", temporal_semantics="instantaneous")
    observations = pd.DataFrame({
        "station_id": ["S1", "S2"], "time": [time[0], time[0]],
        "latitude": [30.2, 30.8], "longitude": [110.2, 110.8],
        "variable": ["PM2.5", "PM2.5"], "value": [10.5, 15.5],
        "unit": ["ug m-3", "ug m-3"], "qc_flag": ["VALID", "VALID"],
        "split": ["train", "train"],
    })
    options = QPOptions(solver=solver, max_iter=12_000, eps_abs=1e-4, eps_rel=1e-4)
    result = ConversionTaskEngine().execute(
        [source], StructureRequest(variable="value", unit="ug m-3"),
        TrialRequest(
            target_spatial_resolution_km=5.0, spatial_method="constrained",
            variable="value", unit="ug m-3", temporal_semantics="instantaneous",
            output_format="NetCDF", optimization_options=options,
        ),
        training_observations=observations,
    )
    spatial_step = next(item for item in result.report["自动处理流程"] if item["阶段"] == "空间处理")
    assert solver in spatial_step["处理"] and "12000" in spatial_step["处理"]
    assert result.history[-1]["parameters"]["max_iter"] == 12_000
    assert result.history[-1]["parameters"]["solver"] == solver
    assert result.report["优化诊断"]["求解状态"] == ["optimal"]


def test_wide_water_table_resamples_multiple_variables_without_coordinates():
    content = "断面名称,监测时间,自动监测站编码,总磷,总氮\n站点A,2021-02-01 00:00:00,A001,0.1,2.0\n站点A,2021-02-01 02:00:00,A001,0.3,4.0\n".encode("utf-8-sig")
    source = read_structure("water.csv", content)
    result = ConversionTaskEngine().execute(
        [source], StructureRequest(operation="preserve", variable="总磷", x="自动监测站编码", time="监测时间"),
        TrialRequest(target_time_frequency="1h", variable="总磷", variables=("总磷", "总氮"), output_format="Parquet"),
    )
    frame = result.output.to_dataframe().reset_index(drop=True)
    assert list(frame["总磷"].astype(float)) == [0.1, 0.2, 0.3]
    assert list(frame["总氮"].astype(float)) == [2.0, 3.0, 4.0]
    assert list(frame["总磷_qc_flag"]) == ["ORIGINAL", "INTERPOLATED", "ORIGINAL"]
