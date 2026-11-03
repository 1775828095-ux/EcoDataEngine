"""Independent self-checks and gates; reuse real conversion/fusion code."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

from examples.run_information_loss_checks import water_fixture, resampling_fixture, fusion_fixture
from ecodata.core.exceptions import SchemaError, ValidationError
from ecodata.core.pipeline import SequentialPipeline
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult
from ecodata.evaluation.self_check import PostConversionValidator, profile_dataset, require_usable, validation_files
from ecodata.interface.workflow import TrialRequest
from ecodata.io.structures import GEOMETRY, StructureRequest, table_dataset
from ecodata.recognition import recognise_file
from ecodata.task import ConversionTaskEngine
from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_contains_statistics_semantics_and_metadata():
    ds = water_fixture()
    profile = profile_dataset(ds)
    assert profile["record_count"] == 3 and profile["variable_count"] == 8
    assert profile["variables"]["氨氮"]["unit"] == "mg/L"
    assert profile["variables"]["氨氮"]["missing"] == 1
    assert profile["variables"]["氨氮"]["mean"] == pytest.approx(.155)
    assert profile["variables"]["水质类别"]["categorical_values"] == ["II", "III"]
    assert "qc_flag" in profile["quality_flags"]
    assert profile["source"]["source_file"] == "synthetic_water_qa.csv"
    assert profile["time_range"] and profile["time_resolution"]
    assert profile["CRS"] is None and profile["resolution_km"] is None
    json.dumps(profile, allow_nan=False)


@pytest.mark.parametrize("output_format", ["NetCDF", "HDF5-NetCDF"])
def test_lossless_water_conversion_reopens_file_and_emits_three_reports(output_format):
    ds = water_fixture()
    result = ConversionTaskEngine().execute([ds], StructureRequest(variable="氨氮"), TrialRequest(variable="氨氮"))
    content, filename, _ = result.export(output_format)
    reopened, _ = recognise_file(filename, content, record_read_source=False)
    assert reopened["氨氮"].dtype.kind == "f"
    assert reopened["年份"].dtype.kind == reopened["月份"].dtype.kind == "i"
    assert reopened["水质类别"].dtype.kind in "OUS"
    report = result.report["validation_report"]
    assert report["status"] == report["file_validation"]["status"] == "PASS"
    assert report["numerical_comparison"]["氨氮"]["changed_value_count"] == 0
    assert report["numerical_comparison"]["氨氮"]["missing_positions_equal"]
    assert set(result.validation_files()) == {"before_profile.json", "after_profile.json", "validation_report.json"}
    for payload in result.validation_files().values():
        json.loads(payload)


@pytest.mark.parametrize("fault", ["value", "unit", "metadata", "metadata_value", "time", "missing_position", "category", "quality_flag", "numeric_to_string", "variable", "new_nan"])
def test_lossless_corruption_fails(fault):
    source = water_fixture()
    after = source.copy(deep=True)
    if fault == "value":
        after["氨氮"].values[0] += .1
    elif fault == "unit":
        after["氨氮"].attrs.pop("units")
    elif fault == "metadata":
        after.attrs.pop("source_file")
    elif fault == "metadata_value":
        after["氨氮"].attrs["source_field_name"] = "fake"
    elif fault == "time":
        after["采样时间"].values[1] = "2021-02-04"
    elif fault == "missing_position":
        after["氨氮"].values[:] = [np.nan, .20, .11]
    elif fault == "category":
        after["水质类别"].values[1] = "IV"
    elif fault == "quality_flag":
        after.qc_flag.values[0] = "INVALID"
    elif fault == "numeric_to_string":
        after["氨氮"] = after["氨氮"].astype(str)
    elif fault == "variable":
        after = after.drop_vars("氨氮")
    else:
        after["broken"] = (after["氨氮"].dims, [np.nan]*3)
    report = PostConversionValidator().validate(source, after)
    assert report.status == "FAIL" and report.details["errors"]
    with pytest.raises(ValidationError) as exc:
        require_usable(report)
    assert exc.value.report is report


def test_float32_roundoff_is_allowed():
    source = water_fixture()
    after = source.copy(deep=True)
    after["氨氮"] = after["氨氮"].astype("float32")
    report = PostConversionValidator().validate(source, after)
    assert report.status == "PASS"
    assert report.details["numerical_comparison"]["氨氮"]["changed_value_count"] == 2
    assert report.details["numerical_comparison"]["氨氮"]["unacceptable_value_count"] == 0


def test_integer_counts_are_exact_not_subject_to_relative_float_tolerance():
    source = xr.Dataset({"sample_count": ("row", np.array([10_000_000], dtype="int64"))})
    after = source.copy(deep=True)
    after.sample_count.values[0] += 1
    report = PostConversionValidator().validate(source, after)
    assert report.status == "FAIL"
    assert report.details["numerical_comparison"]["sample_count"]["unacceptable_value_count"] == 1


def test_preexisting_optional_empty_column_is_not_mistaken_for_conversion_loss():
    source = water_fixture()
    source["optional_measurement"] = (source["氨氮"].dims, [np.nan]*3)
    report = PostConversionValidator().validate(source, source.copy(deep=True), critical_fields=["氨氮"])
    assert report.status == "WARN" and report.passed
    report = PostConversionValidator().validate(source, source, critical_fields=["optional_measurement"])
    assert report.status == "FAIL"


@pytest.mark.parametrize("fault", ["crs_missing", "crs_changed", "large_nan", "infinity", "negative"])
def test_processed_field_corruption_fails(fault):
    source = resampling_fixture()
    after = source.copy(deep=True)
    if fault == "crs_missing":
        after.attrs.pop("crs")
    elif fault == "crs_changed":
        after.attrs["crs"] = "EPSG:4326"
    elif fault == "large_nan":
        after.value.values[:15] = np.nan
    elif fault == "infinity":
        after.value.values[0, 0] = np.inf
    else:
        after.value.values[0, 0] = -1.
    assert PostConversionValidator().validate(source, after, kind="resampling").status == "FAIL"


def test_1km_to_5km_peak_loss_warns_and_integral_passes():
    source = resampling_fixture()
    result = ConversionTaskEngine().execute([source], StructureRequest(variable="value", x="x", y="y"),
        TrialRequest(variable="value", target_spatial_resolution_km=5, spatial_method="conservative", spatial_semantics="cell_mean"))
    result.export("NetCDF")
    report = result.report["validation_report"]
    assert report["status"] == "WARN"
    assert report["before_profile"]["resolution_km"] == pytest.approx(1.)
    assert report["after_profile"]["resolution_km"] == pytest.approx(5.)
    assert report["information_loss"]["peak_loss"]["value"]["amplitude_percent"] == pytest.approx(84.83720930232558)
    assert report["information_loss"]["conservation_error"]["value"]["relative_error"] == pytest.approx(0., abs=1e-12)
    assert report["checks_by_category"]["conservation"] == "PASS"
    assert report["information_loss"]["dropped_records"] == 0
    assert result.report["validation_report"]["after_profile"]["pixel_extent"] == [0., 0., 20000., 20000.]
    assert set(report["information_loss"]["peak_loss"]["value"]["after_position"]) == {"x", "y"}


def test_broken_conservation_fails_but_cropped_support_is_not_claimed_conserved():
    source = resampling_fixture()
    source.value.attrs["spatial_semantics"] = "flux"
    after = source.copy(deep=True)
    after.value.values[:] *= 2.
    report = PostConversionValidator().validate(source, after, kind="resampling")
    assert report.status == "FAIL" and report.details["checks_by_category"]["conservation"] == "FAIL"
    cropped = after.isel(x=slice(0, 10))
    report = PostConversionValidator().validate(source, cropped, kind="resampling")
    assert report.status == "WARN"
    assert report.details["information_loss"]["conservation_error"]["value"]["relative_error"] is None


def test_declared_peak_sensitive_limit_can_block_otherwise_reasonable_resampling():
    source = resampling_fixture()
    source.value.attrs["max_peak_loss_percent"] = 20.
    after = source.copy(deep=True)
    after.value.values[:] = 10.
    report = PostConversionValidator().validate(source, after, kind="resampling")
    assert report.status == "FAIL"
    assert any("峰值损失超出" in message for message in report.details["errors"])


@pytest.mark.parametrize("semantics", ["mean", "accumulation"])
def test_temporal_interval_conservation_and_semantics(semantics):
    edges = pd.date_range("2026-01-01", periods=5, freq="1h").to_numpy()
    bounds = np.column_stack([edges[:-1], edges[1:]])
    source = xr.Dataset({"value": ("time", [2., 4., 6., 8.]), "time_bounds": (("time", "bounds"), bounds)},
        coords={"time": bounds[:, 1], "bounds": [0, 1]}, attrs={"temporal_semantics": semantics, "data_kind": "continuous"})
    source.value.attrs.update(units="mm" if semantics == "accumulation" else "mg/L", temporal_semantics=semantics, spatial_semantics="point")
    transformed = SemanticTimeResampler(SemanticTimeConfig("2h")).apply(EcoDataBundle(source))
    report = PostConversionValidator().validate(source, transformed.data.dataset, history=[transformed.record])
    assert report.status == "WARN"
    assert report.details["information_loss"]["conservation_error"]["value"]["relative_error"] == pytest.approx(0.)
    broken = transformed.data.dataset.copy(deep=True)
    broken.value.attrs["temporal_semantics"] = "instantaneous"
    assert PostConversionValidator().validate(source, broken, kind="resampling").status == "FAIL"


def test_fusion_changes_are_not_fake_accuracy_and_stations_are_geometrically_checked():
    background, stations = fusion_fixture()
    result = ConversionTaskEngine().execute([background, stations], StructureRequest(variable="value"),
        TrialRequest(variable="value", fusion_method="residual", spatial_method="nearest"), fuse=True)
    result.export("NetCDF")
    report = result.report["validation_report"]
    assert report["status"] == "WARN" and not report["accuracy"]["available"]
    assert report["numerical_comparison"]["value"]["correction_distribution"]["mean"] == 2.
    assert report["observation_placement"]["matched_count"] == 3
    assert report["observation_placement"]["max_nearest_cell_distance_m"] == 0.
    assert report["extrapolated_spatial_cell_count"] == 3
    assert report["information_loss"]["extrapolation_applied"]
    assert len(report["source_profiles"]) == 1
    assert "background_value" in result.output and "residual_correction" in result.output


def test_independent_validation_metrics_and_training_leakage_rejection():
    background, stations = fusion_fixture()
    after = background.copy(deep=True)
    after.value.values[:] = 12.
    frame = pd.DataFrame({"latitude": [30.], "longitude": [110.], "time": ["2026-01-01"], "value": [13.],
                          "variable": ["PM2.5"], "unit": ["ug/m3"], "qc_flag": ["VALID"], "split": ["validation"]})
    report = PostConversionValidator().validate(background, after, kind="fusion", validation_stations=frame)
    assert report.details["accuracy"]["RMSE"] == report.details["accuracy"]["MAE"] == 1.
    assert report.details["accuracy"]["bias"] == -1.
    frame["split"] = "train"
    with pytest.raises(SchemaError, match="训练站"):
        PostConversionValidator().validate(background, after, kind="fusion", validation_stations=frame)


def test_structure_loss_and_geometry_corruption_are_reported():
    geometry = json.dumps({"type": "Point", "coordinates": [110., 30.]})
    source = table_dataset(pd.DataFrame({GEOMETRY: [geometry], "station_note": ["keep me"], "value": [1.]}))
    source.attrs["crs"] = "EPSG:4326"
    output = source.drop_vars([GEOMETRY, "station_note"])
    report = PostConversionValidator().validate(source, output, kind="structure")
    assert report.status == "WARN"
    assert set(report.details["information_loss"]["lost_fields"]) == {GEOMETRY, "station_note"}
    changed = source.copy(deep=True)
    changed[GEOMETRY].values[0] = json.dumps({"type": "Point", "coordinates": [111., 30.]})
    assert PostConversionValidator().validate(source, changed, kind="structure").status == "FAIL"


def test_model_incomplete_fails_and_native_status_is_not_invented():
    ds = water_fixture()
    report = PostConversionValidator().validate(ds, ds, kind="model_adaptation", model_check={"complete": False, "missing": ["wind"], "native_verified": False})
    assert report.status == "FAIL"
    assert report.details["model_compatibility_check"]["missing"] == ["wind"]
    assert report.details["model_compatibility_check"]["native_verified"] is False


def test_unit_conversion_does_not_create_fake_peak_loss_or_correction_error():
    source = xr.Dataset({"temperature": ("row", [273.15, 283.15])}, attrs={"structure": "table", "row_dimension": "row"})
    source.temperature.attrs["units"] = "K"
    after = source.copy(deep=True)
    after.temperature.values[:] -= 273.15
    after.temperature.attrs.update(units="degC", source_units="K", unit_conversion='{"scale": 1.0, "offset": -273.15}')
    report = PostConversionValidator().validate(source, after, kind="model_adaptation")
    assert report.status == "WARN"
    assert report.details["information_loss"]["peak_loss"]["temperature"]["amplitude_percent"] == 0.
    assert report.details["numerical_comparison"]["temperature"]["max_abs_error"] == 0.
    assert report.details["before_profile"]["variables"]["temperature"]["max"] == 283.15
    assert report.details["statistical_changes"]["temperature"]["comparison_unit"] == "degC"


@pytest.mark.parametrize("export_method", ["export", "netcdf_bytes", "csv_bytes"])
def test_fail_gate_blocks_modified_result_exports(export_method):
    ds = water_fixture()
    result = ConversionTaskEngine().execute([ds], StructureRequest(variable="氨氮"), TrialRequest(variable="氨氮"))
    result.output["氨氮"].values[0] = 100.
    with pytest.raises(ValidationError):
        result.export("NetCDF") if export_method == "export" else getattr(result, export_method)()


def test_pipeline_blocks_bad_transform_and_prior_failure():
    class BadTransform:
        name = "format.corrupt"
        def validate_input(self, data):
            pass
        def apply(self, data):
            ds = data.dataset.copy(deep=True)
            ds["氨氮"].values[0] += 10.
            record = TransformRecord(self.name, {}, 3, 3)
            return TransformResult(EcoDataBundle(ds, data.history + (record,)), record)
    source = water_fixture()
    with pytest.raises(ValidationError):
        SequentialPipeline().run(EcoDataBundle(source), [BadTransform()])
    failed = PostConversionValidator().validate(source, source.drop_vars("氨氮"))
    with pytest.raises(ValidationError):
        SequentialPipeline().run(EcoDataBundle(source, validation=failed), [])


@pytest.mark.parametrize("fault", ["empty_file", "unreadable_file", "changed_value", "lost_unit"])
def test_serialized_output_failure_still_produces_downloadable_fail_report(fault):
    from ecodata.interface.workflow import TrialResult
    from ecodata.io.structures import export_structure
    source = water_fixture()
    result = ConversionTaskEngine().execute([source], StructureRequest(variable="氨氮"), TrialRequest(variable="氨氮"))
    if fault in {"empty_file", "unreadable_file"}:
        payload = b"" if fault == "empty_file" else b"not a netcdf file"
    else:
        broken = result.output.copy(deep=True)
        if fault == "changed_value":
            broken["氨氮"].values[0] += 1.
        else:
            broken["氨氮"].attrs.pop("units")
        payload = export_structure(broken, "NetCDF")[0]
    with patch.object(TrialResult, "_export_unchecked", return_value=(payload, "broken.nc", "application/x-netcdf")):
        with pytest.raises(ValidationError) as failure:
            result.export("NetCDF")
    report = failure.value.report
    assert report.status == "FAIL" and not report.passed
    assert report.details["before_profile"]["record_count"] == 3
    assert set(validation_files(report)) == {"before_profile.json", "after_profile.json", "validation_report.json"}
    assert report.details["after_profile_available"] == (fault in {"changed_value", "lost_unit"})


def test_streamlit_summary_and_automatic_downloads():
    upload = SimpleNamespace(name="water.csv", getvalue=lambda: "断面名称,氨氮(mg/L)\nA,0.11\nB,0.20\n".encode("utf-8"))
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value('数据转换').run()
        next(button for button in app.button if button.label == "执行数据转换").click().run()
    assert not app.exception and not app.error
    assert any("PASS" in message.value for message in app.success)
    labels = [button.label for button in app.get("download_button")]
    assert all("下载自查 " + name in labels for name in ("before_profile.json", "after_profile.json", "validation_report.json"))


def test_streamlit_resampling_warn_summary():
    from ecodata.io.structures import export_structure
    payload = export_structure(resampling_fixture(), "NetCDF")[0]
    upload = SimpleNamespace(name="grid.nc", getvalue=lambda: payload)
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value('数据转换').run()
        next(widget for widget in app.number_input if widget.label.startswith("目标空间分辨率")).set_value(5.).run()
        next(widget for widget in app.selectbox if widget.label == "空间转换算法").set_value("conservative").run()
        next(button for button in app.button if button.label == "执行数据转换").click().run()
    assert not app.exception and not app.error
    assert any("转换结果：WARN" in message.value for message in app.warning)
    assert app.session_state["task_result"].report["validation_report"]["checks_by_category"]["conservation"] == "PASS"


def test_streamlit_failed_file_has_diagnostics_but_no_result_download():
    from ecodata.interface.workflow import TrialResult
    upload = SimpleNamespace(name="water.csv", getvalue=lambda: b"station_id,value\nA,0.11\nB,0.20\n")
    with patch("streamlit.file_uploader", return_value=[upload]), patch.object(TrialResult, "_export_unchecked", return_value=(b"broken", "bad.nc", "application/x-netcdf")):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value('数据转换').run()
        next(button for button in app.button if button.label == "执行数据转换").click().run()
    assert not app.exception and app.error
    labels = [button.label for button in app.get("download_button")]
    assert "下载转换结果" not in labels
    assert len(labels) == 3 and all("下载失败诊断" in label for label in labels)
