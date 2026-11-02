from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.interface.workflow import (
    TrialRequest,
    recognise_csv,
    recognise_netcdf,
    run_grid_trial,
    run_multi_grid_trial,
    run_station_csv_trial,
)
from ecodata.recognition import recognise_file


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "datasets"


def test_complete_uploaded_netcdf_to_downloads() -> None:
    source = xr.load_dataset(DATA / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf")
    recognition = recognise_netcdf("uploaded.nc", source)
    assert recognition.selected_variable == "value"
    assert recognition.unit == "ug m-3"
    assert recognition.spatial_resolution_km == 10.0

    result = run_grid_trial(
        source,
        TrialRequest(
            target_spatial_resolution_km=2.5,
            variable="value",
            temporal_semantics="instantaneous",
            unit="ug m-3",
        ),
    )
    assert result.output["value"].shape == (48, 24, 24)
    assert result.report["精度指标可用"] is False
    assert "不计算" in result.report["精度说明"]
    assert result.history[-1]["method"] == "spatial.regrid.bilinear"
    assert len(result.netcdf_bytes()) > 100
    assert b"value" in result.csv_bytes()
    assert json.loads(result.report_bytes())["状态"] == "成功"


def test_missing_unit_requires_confirmation() -> None:
    source = xr.load_dataset(DATA / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf")
    del source["value"].attrs["units"]
    recognition = recognise_netcdf("unknown.nc", source)
    assert "数据单位" in recognition.confirmations_required
    with pytest.raises(SchemaError, match="单位"):
        run_grid_trial(source, TrialRequest(target_spatial_resolution_km=2.5, variable="value"))


def test_standard_csv_recognition_and_conversion() -> None:
    content = (ROOT / "examples" / "p1_irregular_observations.csv").read_bytes()
    frame = pd.read_csv(ROOT / "examples" / "p1_irregular_observations.csv")
    recognition = recognise_csv("stations.csv", frame)
    assert recognition.selected_variable == "value"
    assert recognition.unit == "ug m-3"
    result = run_station_csv_trial(
        content,
        TrialRequest(target_time_frequency="1h", temporal_semantics="instantaneous"),
    )
    assert result.output.sizes["observation"] == 4
    assert result.report["精度指标可用"] is False
    assert result.output["station_id"].dtype.kind in {"U", "O"}
    assert set(result.output["qc_flag"].values) == {"VALID", "INTERPOLATED"}


def test_csv_target_fields_keep_text_and_numeric_columns() -> None:
    content = (ROOT / "examples" / "p1_irregular_observations.csv").read_bytes()
    result = run_station_csv_trial(
        content,
        TrialRequest(
            target_time_frequency="1h",
            temporal_semantics="instantaneous",
            target_fields=("station_id", "time", "value", "unit", "qc_flag"),
        ),
    )
    assert set(result.output.data_vars) == {"station_id", "time", "value", "unit", "qc_flag"}
    assert result.output["station_id"].dtype.kind in {"U", "O"}


def test_noninteger_spatial_ratio_is_supported() -> None:
    source = xr.load_dataset(DATA / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf")
    result = run_grid_trial(
        source,
        TrialRequest(target_spatial_resolution_km=3.0, variable="value", unit="ug m-3", temporal_semantics="instantaneous"),
    )
    assert result.output.sizes["lat"] == result.output.sizes["lon"] == 20


def test_confirmed_multiple_grids_are_fused_before_transform() -> None:
    first = xr.load_dataset(DATA / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf").isel(time=slice(0, 2))
    first["uncertainty"] = xr.full_like(first["value"], 1.0)
    first["uncertainty"].attrs["units"] = first["value"].attrs["units"]
    second = first.copy(deep=True)
    second["value"] = first["value"] + 4.0
    second["value"].attrs.update(first["value"].attrs)
    second["uncertainty"] = xr.full_like(first["value"], 2.0)
    second["uncertainty"].attrs["units"] = first["value"].attrs["units"]
    result = run_multi_grid_trial(
        [first, second],
        TrialRequest(target_spatial_resolution_km=5.0, variable="value", unit="ug m-3", temporal_semantics="instantaneous"),
    )
    assert result.history[0]["method"] == "fusion.uncertainty_weighted"
    assert result.report["算法决策"][0]["步骤"] == "多源融合"


@pytest.mark.parametrize(
    ("output_format", "expected_filename"),
    [
        ("CSV", "ecodata_result.csv"),
        ("TSV", "ecodata_result.tsv"),
        ("JSON", "ecodata_result.json"),
        ("Excel", "ecodata_result.xlsx"),
        ("NetCDF", "ecodata_result.nc"),
        ("HDF5-NetCDF", "ecodata_result.h5"),
        ("Parquet", "ecodata_result.parquet"),
        ("GeoJSON", "ecodata_result.geojson"),
    ],
)
def test_verified_exports_can_be_read_back(output_format: str, expected_filename: str) -> None:
    content = (ROOT / "examples" / "p1_irregular_observations.csv").read_bytes()
    result = run_station_csv_trial(content, TrialRequest(target_time_frequency="1h", temporal_semantics="instantaneous"))
    exported, filename, _ = result.export(output_format)
    assert filename == expected_filename
    assert len(exported) > 20

    if output_format == "CSV":
        assert len(pd.read_csv(io.BytesIO(exported))) == 4
    elif output_format == "TSV":
        assert len(pd.read_csv(io.BytesIO(exported), sep="\t")) == 4
    elif output_format == "JSON":
        assert len(json.loads(exported)) == 4
    elif output_format == "Excel":
        assert len(pd.read_excel(io.BytesIO(exported), engine="openpyxl")) == 4
    elif output_format == "NetCDF":
        with xr.open_dataset(io.BytesIO(exported)) as reopened:
            assert reopened.sizes["observation"] == 4
    elif output_format == "HDF5-NetCDF":
        with xr.open_dataset(io.BytesIO(exported), engine="h5netcdf") as reopened:
            assert reopened.sizes["observation"] == 4
    elif output_format == "Parquet":
        assert len(pd.read_parquet(io.BytesIO(exported), engine="pyarrow")) == 4
    else:
        payload = json.loads(exported)
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) == 4


def test_excel_and_parquet_uploads_are_actually_read() -> None:
    frame = pd.read_csv(ROOT / "examples" / "p1_irregular_observations.csv")
    excel = io.BytesIO()
    frame.to_excel(excel, index=False, engine="openpyxl")
    parsed_excel, excel_info = recognise_file("stations.xlsx", excel.getvalue())
    parquet = io.BytesIO()
    frame.to_parquet(parquet, index=False, engine="pyarrow")
    parsed_parquet, parquet_info = recognise_file("stations.parquet", parquet.getvalue())
    assert parsed_excel.sizes[parsed_excel.attrs["row_dimension"]] == len(frame) and excel_info.format == "Excel"
    assert parsed_parquet.sizes[parsed_parquet.attrs["row_dimension"]] == len(frame) and parquet_info.format == "Parquet"


def test_recognition_treats_uncertainty_as_auxiliary_variable() -> None:
    source = xr.load_dataset(DATA / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf").isel(time=slice(0, 1))
    source["uncertainty"] = xr.full_like(source["value"], 2.0)
    source["uncertainty"].attrs["units"] = source["value"].attrs["units"]
    recognition = recognise_netcdf("fusion.nc", source)
    assert recognition.selected_variable == "value"
    assert recognition.unit == "ug m-3"
