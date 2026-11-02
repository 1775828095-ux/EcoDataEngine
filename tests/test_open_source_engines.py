from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.engines.fusion import SpatialFusionEngine
from ecodata.engines.grid import GridEngine
from ecodata.engines.units import UnitEngine
from ecodata.engines.validation import ValidationEngine
from ecodata.fusion.planning import FusionPlanner
from ecodata.interface.workflow import TrialRequest
from ecodata.io.structures import StructureRequest, table_dataset
from ecodata.models.adapters.ioapi_reader import CmaqIOAPIReader
from ecodata.recognition import inspect_dataset, recognise_file
from ecodata.task import ConversionTaskEngine


from examples.open_source_integration.datasets import kriging_fixture


def test_cf_evidence_recognizes_nonstandard_coordinate_names():
    ds = xr.Dataset({"t": ("date_index", [290.], {"units": "K", "standard_name": "air_temperature"})},
        coords={"eastings": ("eastings", [1., 2.], {"axis": "X", "standard_name": "projection_x_coordinate", "units": "m"}),
                "northings": ("northings", [3., 4.], {"axis": "Y", "standard_name": "projection_y_coordinate", "units": "m"}),
                "date_index": ("date_index", pd.date_range("2024-01-01", periods=1), {"axis": "T"})})
    info = inspect_dataset("cf.nc", ds)
    assert info.x_coordinate == "eastings" and info.y_coordinate == "northings"
    assert info.time_coordinate == "date_index"
    assert info.cf_metadata["standard_names"]["air_temperature"] == ["t"]
    assert info.crs is None  # no guess from coordinate values


def test_csv_unit_conversion_and_export_automatic_standards():
    ds, info = recognise_file("water.csv", "station_id,氨氮(mg/L),年份,地区\nA,0.11,2021,甲\nB,0.23,2021,乙\n".encode())
    assert info.semantic_types["氨氮"] == "continuous"
    ds["氨氮"] = UnitEngine().convert(ds["氨氮"], "ug/L")
    assert ds["氨氮"].values.tolist() == [110., 230.]
    result = ConversionTaskEngine().execute([ds], StructureRequest(variable="氨氮"), TrialRequest(variable="氨氮"))
    content, name, _ = result.export("NetCDF")
    restored, _ = recognise_file(name, content)
    assert restored["氨氮"].dtype.kind == "f" and restored["年份"].dtype.kind == "i"
    assert restored["氨氮"].attrs["units"] == "ug/L"
    assert "pint-xarray" in restored["氨氮"].attrs["unit_conversion"]
    assert "standards_compliance" in result.report["validation_report"]
    assert result.report["validation_report"]["standards_compliance"]["status"] == "FAIL"  # CF ASCII naming, not data loss


@pytest.mark.parametrize("source,target,expected", [("degC", "K", (1.,273.15)), ("mg/L", "ug/L", (1000.,0.)),
    ("ug m-3", "mg/m3", (.001,0.)), ("m/s", "km/h", (3.6,0.)), ("1", "%", (100.,0.))])
def test_pint_physical_units(source, target, expected):
    assert UnitEngine().affine(source, target) == pytest.approx(expected)


@pytest.mark.parametrize("source,target", [("K", "mg/L"), ("unknown", "m"), ("ug/m3", "ug/kg")])
def test_pint_rejects_unknown_and_incompatible(source, target):
    with pytest.raises(SchemaError):
        UnitEngine().affine(source, target)


def test_grid_fallback_records_real_backend():
    source = xr.Dataset({"value": (("lat", "lon"), [[1.,2.],[3.,4.]])},
        coords={"lat": [1.,2.], "lon": [1.,2.]}, attrs={"crs": "EPSG:4326"})
    target = xr.Dataset(coords={"lat": [1.,1.5,2.], "lon": [1.,1.5,2.]})
    output, record = GridEngine().regrid(source, target, "bilinear", backend="fallback")
    assert output.value.sel(lat=1.5, lon=1.5) == 2.5
    assert record["backend"] == "xarray-regrid"


def test_xesmf_real_two_grid_integration():
    status = GridEngine.availability()
    if not status["available"]:
        pytest.skip("真实 xESMF 未运行：" + status["reason"])
    source = xr.Dataset({"value": (("lat", "lon"), [[1.,2.],[3.,4.]])},
        coords={"lat": [1.,2.], "lon": [1.,2.]}, attrs={"crs": "EPSG:4326"})
    target = xr.Dataset(coords={"lat": [1.25,1.75], "lon": [1.25,1.75]})
    output, record = GridEngine().regrid(source, target, "bilinear", backend="xesmf")
    assert record["backend"] == "xESMF" and np.isfinite(output.value).all()


def test_ordinary_external_drift_pipeline_and_independent_validation():
    target, stations, auxiliary, truth = kriging_fixture()
    results = {}
    for method in ("ordinary_kriging", "external_drift_kriging"):
        result = ConversionTaskEngine().execute([target, stations, auxiliary], StructureRequest(variable="PM25"),
            TrialRequest(variable="PM25", fusion_method=method, covariates=("elevation",) if method.startswith("external") else ()), fuse=True)
        held = table_frame_for_test(stations).query("split == 'validation'")
        prediction = result.output.PM25.interp(x=xr.DataArray(held.x.values, dims="station"), y=xr.DataArray(held.y.values, dims="station"))
        score = ValidationEngine.skill(prediction, xr.DataArray(held.PM25.values, dims="station"), independent=True)
        results[method] = score
        assert result.report["GSTools"][0]["training_count"] == 60
        assert "original_background_PM25" in result.output and "kriging_variance" in result.output
        assert np.isfinite(result.output.PM25).all()
    assert results["external_drift_kriging"]["RMSE"] < results["ordinary_kriging"]["RMSE"]


def table_frame_for_test(stations):
    from ecodata.io.structures import table_frame
    return table_frame(stations)


@pytest.mark.parametrize("method", ["ordinary_kriging", "universal_kriging"])
def test_gstools_native_estimators(method):
    rng = np.random.default_rng(51)
    positions = rng.uniform(0, 1000, (2, 25))
    values = 10 + .01*positions[0] + np.sin(positions[1]/100)
    field, variance, record, krige = SpatialFusionEngine().estimate(positions, values, positions, method=method, split=["train"]*25)
    assert np.isfinite(field).all() and (variance >= -1e-8).all()
    assert record["backend"] == "GSTools" and callable(krige)
    conditioned = SpatialFusionEngine.conditioned_field(krige, positions, seed=10)
    assert np.isfinite(conditioned).all()


def test_kriging_rejects_validation_leakage():
    with pytest.raises(SchemaError, match="验证站"):
        SpatialFusionEngine().estimate(np.ones((2,10)), np.arange(10), np.ones((2,10)), split=["validation"]*10)


@pytest.mark.parametrize("problem", ["coverage", "rank", "relationship"])
def test_external_drift_refuses_unreliable_auxiliary(problem):
    rng = np.random.default_rng(17)
    positions = rng.uniform(0, 1000, (2, 60))
    values = rng.normal(size=60)
    drift = rng.normal(size=(1,60))
    target_drift = drift.copy()
    if problem == "coverage":
        target_drift[0,0] = np.nan
    elif problem == "rank":
        drift = np.vstack([drift,drift]); target_drift = drift.copy()
    with pytest.raises(SchemaError):
        SpatialFusionEngine().estimate(positions, values, positions, method="external_drift_kriging", drift=drift,
            target_drift=target_drift, split=["train"]*60)


def test_planner_does_not_add_dem_to_pm25():
    target, stations, auxiliary, _ = kriging_fixture()
    plan = FusionPlanner().plan([target,stations,auxiliary], "PM25", "auto", ("elevation",))
    assert plan["method"] == "external_drift_kriging"
    assert plan["roles"] == ["target_state","observation","static_constraint"]


def test_observations_dem_without_fabricated_background_and_auto_plan():
    _, stations, dem, _ = kriging_fixture()
    result = ConversionTaskEngine().execute([dem, stations], StructureRequest(variable="PM25"),
        TrialRequest(variable="PM25", fusion_method="auto"), fuse=True)
    assert result.report["fusion_plan"]["method"] == "external_drift_kriging"
    assert "original_background_PM25" not in result.output
    assert np.isfinite(result.output.PM25).all()


def test_validation_values_cannot_change_fitted_field():
    target, stations, dem, _ = kriging_fixture()
    request = TrialRequest(variable="PM25",fusion_method="external_drift_kriging",covariates=("elevation",))
    first = ConversionTaskEngine().execute([target,stations,dem],StructureRequest(variable="PM25"),request,fuse=True)
    changed = stations.copy(deep=True)
    changed.PM25.values[changed.split.values == "validation"] += 1000
    second = ConversionTaskEngine().execute([target,changed,dem],StructureRequest(variable="PM25"),request,fuse=True)
    xr.testing.assert_identical(first.output.PM25,second.output.PM25)
    assert first.report["独立验证"]["RMSE"] != second.report["独立验证"]["RMSE"]


@pytest.mark.parametrize("kind", ["categorical", "quality_auxiliary"])
def test_nonphysical_auxiliary_is_not_forced_into_drift(kind):
    target, stations, auxiliary, _ = kriging_fixture()
    if kind == "categorical":
        auxiliary.elevation.attrs.update(data_kind="categorical", flag_values=[1,2])
    else:
        auxiliary.attrs["data_role"] = kind
    with pytest.raises(SchemaError):
        ConversionTaskEngine().execute([target,stations,auxiliary],StructureRequest(variable="PM25"),
            TrialRequest(variable="PM25",fusion_method="external_drift_kriging",covariates=("elevation",)),fuse=True)


def test_skill_requires_independent_reference():
    with pytest.raises(SchemaError):
        ValidationEngine.skill(xr.DataArray([1.]), xr.DataArray([1.]))


def test_nonlinear_units_are_not_silently_treated_as_affine():
    with pytest.raises(SchemaError, match="非线性"):
        UnitEngine().affine("dB", "1")


def test_cf_checker_real_pass_and_fail():
    good = xr.Dataset({"temperature": ("row", [280.,281.], {"units":"K","standard_name":"air_temperature"})},
        attrs={"Conventions":"CF-1.8","title":"Explicit test metadata","history":"Synthetic integration test"})
    report = ValidationEngine.compliance(bytes(good.to_netcdf()))
    assert report["status"] == "PASS", report
    bad = good.copy(deep=True); bad.temperature.attrs["units"] = "mg/L"
    assert ValidationEngine.compliance(bytes(bad.to_netcdf()))["status"] == "FAIL"


def test_cmaq_official_profile_real_reader():
    path = Path(__file__).resolve().parents[1]/"examples/open_source_integration/cmaq_official/ic_profile_CB05.dat"
    if not path.exists() or not CmaqIOAPIReader().python.exists():
        pytest.skip("EPA 官方 profile 或 PseudoNetCDF 独立环境未配置")
    report = CmaqIOAPIReader().inspect(path, format="icon_profile")
    assert report["attributes"]["NVARS"] > 0 and "VAR-LIST" in report["attributes"]
    assert "VGLVLS" in report["attributes"]
    assert not report["native_model_input_verified"]


def test_streamlit_external_drift_upload_selection_and_export():
    from types import SimpleNamespace
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    from ecodata.io.structures import export_structure
    target, stations, dem, _ = kriging_fixture()
    uploads = [SimpleNamespace(name=name, getvalue=lambda data=export_structure(ds,"NetCDF")[0]: data)
               for name, ds in [("background.nc",target),("stations.nc",stations),("dem.nc",dem)]]
    root = Path(__file__).resolve().parents[1]
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(str(root/"src/ecodata/interface/app.py"),default_timeout=30).run()
        next(w for w in app.checkbox if w.label == "对多个兼容网格进行数据融合").check().run()
        next(w for w in app.selectbox if w.label == "融合方式").set_value("external_drift_kriging").run()
        next(w for w in app.multiselect if w.label.startswith("辅助变量")).set_value(["elevation"]).run()
        next(b for b in app.button if b.label == "执行数据转换").click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert app.session_state["task_result"].report["GSTools"][0]["method"] == "external_drift_kriging"
    assert "下载转换结果" in [b.label for b in app.get("download_button")]
