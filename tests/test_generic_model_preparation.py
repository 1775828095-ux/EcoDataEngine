"""Framework tests, not native-model validation claims."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
import pytest
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import read_structure
from ecodata.models import RequirementStore
from ecodata.models.contracts import ModelSchema
from ecodata.models.schema import editable_template, validate_profile
from ecodata.models.scenario import recognise_inputs, suggest_roles, plan_scenario, execute_scenario


ROOT = Path(__file__).resolve().parents[1]


def water_profile():
    # This is an explicit TEST preparation contract, not a fabricated WASP
    # native specification. It uses the same framework as the real CMAQ case.
    profile = editable_template()
    profile.update(id="water_contract_test", name="水质准备配置测试", temporal_resolution=None)
    profile["inputs"] = [{"name": "water_quality", "standard_name": "water_quality", "unit": "variable-specific",
        "data_type": "continuous", "temporal_resolution": None, "spatial_resolution_km": None,
        "formats": ["CSV", "NetCDF"], "required": True, "depends_on": [], "aliases": [],
        "dimensions": ["record"], "fields": [
            {"name": "NH3", "standard_name": "ammonia", "aliases": ["氨氮"], "unit": "mg/L", "data_type": "continuous"},
            {"name": "COD", "aliases": ["化学需氧量"], "unit": "mg/L", "data_type": "continuous"}]}]
    profile["reference_sources"] = ["用户测试契约；不是WASP原生输入规范"]
    return profile


def water_inputs():
    return recognise_inputs([("water.csv", "站点,采样时间,氨氮(g/L),化学需氧量(mg/L)\nA,2024-01-01,0.00011,8\nA,2024-01-02,0.00012,9\n".encode("utf-8"))])


def test_all_existing_models_use_one_schema_and_config_selected_adapter():
    profiles = RequirementStore().all()
    assert len(profiles) == 24
    for identifier in ("cmaq", "wrf", "swat_plus", "wasp", "modflow6", "hec_ras"):
        schema = ModelSchema.from_profile(next(p for p in profiles if p["id"] == identifier))
        assert schema.inputs and schema.profile["reference_sources"]
        assert schema.preparation["adapter"] in {"standardized", "cmaq_auxiliary", "wasp_package", "wasp_text", "modflow6_flopy"}
        assert all("depends_on" in r and "allow_derived" in r and "dimensions" in r for r in schema.inputs)


def test_csv_is_convertible_not_missing_and_units_are_really_converted():
    profile, inputs = water_profile(), water_inputs()
    roles = suggest_roles(profile, inputs)
    analysis, plan = plan_scenario(profile, inputs, roles)
    match = plan["matches"][0]
    assert plan["can_process"] and match["classification"] == "convertible"
    assert not match["missing"] and len(analysis) == 1
    assert any(o["operation"] == "unit_conversion" for o in match["operations"])
    assert any(o["operation"] == "format_conversion" for o in match["operations"])
    result = execute_scenario(profile, inputs, roles)
    restored = read_structure("prepared_water_quality.nc", result.files["prepared_water_quality.nc"])
    np.testing.assert_allclose(restored.NH3, [.11, .12])
    assert restored.NH3.dtype.kind == "f" and restored.NH3.attrs["units"] == "mg/L"
    assert restored["站点"].values.astype(str).tolist() == ["A", "A"]
    assert result.quality["completion"]["standardized_preprocessing_completed"]
    assert result.quality["completion"]["model_format_matched"]
    assert not result.quality["completion"]["native_model_input_verified"]
    assert "model_input.nc" not in result.files
    assert inputs[0].dataset["氨氮"].attrs["units"] == "g/L"
    assert result.plan["executed_steps"][0]["sha256"] == inputs[0].sha256


def test_same_file_can_serve_two_model_roles_without_overwriting():
    profile, inputs = water_profile(), water_inputs()
    second = deepcopy(profile["inputs"][0])
    second["name"] = "observation_constraint"
    second["depends_on"] = ["water_quality"]
    profile["inputs"].append(second)
    roles = suggest_roles(profile, inputs)
    assert set(roles.values()) == {"water.csv"}
    result = execute_scenario(profile, inputs, roles)
    assert len([f for f in result.files if f.endswith(".nc")]) == 2
    assert result.plan["execution_order"] == ["water_quality", "observation_constraint"]


def test_partial_execution_never_generates_complete_model_input():
    profile, inputs = water_profile(), water_inputs()
    other = deepcopy(profile["inputs"][0])
    other.update(name="required_missing_nitrogen", fields=[{"name": "TN", "unit": "mg/L", "data_type": "continuous"}])
    profile["inputs"].append(other)
    roles = suggest_roles(profile, inputs)
    _, plan = plan_scenario(profile, inputs, roles)
    assert not plan["can_process"] and plan["can_process_partial"]
    assert plan["matches"][1]["classification"] == "missing"
    result = execute_scenario(profile, inputs, roles)
    assert "prepared_water_quality.nc" in result.files
    assert result.quality["postcheck"]["missing_roles"] == ["required_missing_nitrogen"]
    assert not any(result.quality["completion"][key] for key in ("standardized_preprocessing_completed", "model_format_matched", "complete_model_input"))


def test_unknown_units_are_confirmation_problem_not_invented_values():
    profile, inputs = water_profile(), water_inputs()
    inputs[0].dataset["氨氮"].attrs["units"] = "unknown"
    _, plan = plan_scenario(profile, inputs, {"water_quality": "water.csv"})
    assert not plan["can_process"]
    assert not plan["matches"][0]["missing"]
    assert any("单位未知" in reason for reason in plan["matches"][0]["reasons"])


def test_ambiguous_fields_are_not_guessed():
    profile, inputs = water_profile(), water_inputs()
    inputs[0].dataset["NH3"] = inputs[0].dataset["氨氮"]
    _, plan = plan_scenario(profile, inputs, {"water_quality": "water.csv"})
    assert not plan["can_process"]
    assert any("多个字段" in r for r in plan["matches"][0]["reasons"])
    assert "NH3" not in plan["matches"][0]["missing"]


def test_hec_ras_field_contract_rejects_unrelated_water_table_and_never_claims_native_support():
    profile = RequirementStore().get("hec_ras")
    role = ModelSchema.from_profile(profile).inputs[0]["role"]
    _, plan = plan_scenario(profile, water_inputs(), {role: "water.csv"})
    match = next(m for m in plan["matches"] if m["role"] == role)
    assert not match["can_process"] and match["classification"] == "missing"
    assert match["missing"]
    assert not plan["native_model_input"]


def test_dependency_cycle_and_transitive_missing_are_blocked():
    profile = water_profile()
    first = profile["inputs"][0]
    second = deepcopy(first)
    second.update(name="second", depends_on=["water_quality"])
    first["depends_on"] = ["second"]
    profile["inputs"].append(second)
    with pytest.raises(SchemaError, match="循环"):
        plan_scenario(profile, water_inputs(), {"water_quality": "water.csv", "second": "water.csv"})
    first["depends_on"] = []
    third = deepcopy(first)
    third.update(name="third", depends_on=["second"])
    profile["inputs"] = [third, second, first]
    _, plan = plan_scenario(profile, water_inputs(), {"second": "water.csv", "third": "water.csv"})
    assert not plan["can_process_partial"]


def test_adding_new_model_only_requires_store_configuration(tmp_path):
    store = RequirementStore(path=tmp_path / "models.json")
    profile = water_profile()
    store.save(profile)
    profile = store.get(profile["id"])
    validate_profile(profile)
    result = execute_scenario(profile, water_inputs(), {"water_quality": "water.csv"})
    assert result.quality["postcheck"]["complete"]


def test_schema_triggers_existing_time_resampling():
    profile = water_profile()
    profile["inputs"][0]["temporal_resolution"] = "1h"
    text = "station_id,time,氨氮(mg/L),化学需氧量(mg/L)\nA,2024-01-01 00:00,1,2\nA,2024-01-01 02:00,3,6\n"
    inputs = recognise_inputs([("water.csv", text.encode())], {"water.csv": {"temporal_semantics": "instantaneous"}})
    from ecodata.recognition import inspect_dataset
    import pandas as pd
    assert pd.Timedelta(inspect_dataset("water.csv", inputs[0].dataset).temporal_resolution) == pd.Timedelta(hours=2)
    result = execute_scenario(profile, inputs, {"water_quality": "water.csv"})
    np.testing.assert_allclose(result.output.NH3, [1, 2, 3])
    assert result.quality["postcheck"]["complete"]
    assert result.plan["executed_steps"][0]["source_dimensions"] != result.plan["executed_steps"][0]["output_dimensions"]


def test_table_to_grid_is_inferred_from_required_dimensions():
    profile = water_profile()
    profile["inputs"][0]["dimensions"] = ["y", "x"]
    text = "latitude,longitude,氨氮(mg/L),化学需氧量(mg/L)\n34,-118,1,2\n34,-117,3,4\n35,-118,5,6\n35,-117,7,8\n"
    inputs = recognise_inputs([("water.csv", text.encode())], {"water.csv": {"crs": "EPSG:4326"}})
    assignments = suggest_roles(profile, inputs)
    _, plan = plan_scenario(profile, inputs, assignments)
    assert plan["can_process"]
    assert any(op.get("method") == "table_to_grid" for op in plan["matches"][0]["operations"])
    result = execute_scenario(profile, inputs, assignments)
    assert result.output.NH3.shape == (2, 2) and result.quality["postcheck"]["complete"]


def test_postcheck_does_not_trust_plan_or_native_verified_flag():
    from ecodata.models.adapters.standardized import ModelAdapter
    from ecodata.models.matching import ModelRequirementMatcher
    profile = water_profile()
    profile.update(status="verified_native_adapter", verified_sample="claimed evidence is not a runtime validator")
    result = execute_scenario(profile, water_inputs(), {"water_quality": "water.csv"})
    assert not result.quality["completion"]["native_model_input_verified"]
    schema = ModelSchema.from_profile(profile)
    output = result.output.copy(deep=True)
    output.NH3.attrs["units"] = "g/L"
    check = ModelRequirementMatcher(schema, ModelAdapter()).check_outputs({"water_quality": output})
    assert not check["complete"]
    output = result.output.drop_vars("NH3")
    assert not ModelRequirementMatcher(schema, ModelAdapter()).check_outputs({"water_quality": output})["complete"]


def test_schema_composes_existing_fusion_without_model_name_rules():
    from ecodata.models.contracts import ScenarioInput
    profile = water_profile()
    field = {"name": "COD", "unit": "mg/L", "data_type": "continuous"}
    role = profile["inputs"][0]
    role.update(name="source_a", fields=[field], dimensions=["y", "x"])
    role_b = deepcopy(role)
    role_b["name"] = "source_b"
    fused = deepcopy(role)
    fused.update(name="fused", allow_derived=True, processing={"fusion": {"method": "weighted", "sources": ["source_a", "source_b"]}})
    profile["inputs"] = [fused, role, role_b]
    inputs = []
    for name, value in (("a.nc", 2.), ("b.nc", 4.)):
        ds = xr.Dataset({"COD": (("lat", "lon"), np.full((2, 2), value)), "uncertainty": (("lat", "lon"), np.ones((2, 2)))},
                        coords={"lat": [34., 35.], "lon": [-118., -117.]}, attrs={"structure": "grid", "crs": "EPSG:4326"})
        ds.COD.attrs.update(units="mg/L", physical_variable="COD", data_kind="continuous")
        ds.uncertainty.attrs.update(units="mg/L")
        inputs.append(ScenarioInput(name, ds, "test-source-" + name))
    assignments = {"source_a": "a.nc", "source_b": "b.nc"}
    _, plan = plan_scenario(profile, inputs, assignments)
    assert plan["can_process"] and plan["matches"][0]["classification"] == "convertible"
    result = execute_scenario(profile, inputs, assignments)
    output = read_structure("prepared_fused.nc", result.files["prepared_fused.nc"])
    np.testing.assert_allclose(output.COD, 3.)
    assert result.quality["postcheck"]["complete"]
    assert len([name for name in result.files if name.endswith(".nc")]) == 3


def test_wind_derivation_requires_schema_permission():
    from tests.test_model_scenario import case_inputs
    profile, inputs = RequirementStore().get("cmaq"), case_inputs()
    for item in inputs:
        if item.filename == "geos_cf_meteorology.nc":
            del item.dataset["wind"]
    _, plan = plan_scenario(profile, inputs, suggest_roles(profile, inputs))
    assert plan["can_process"]
    profile["preparation_scenario"]["derivable_fields"] = []
    _, plan = plan_scenario(profile, inputs, {"background": "pm25_background.nc", "terrain": "dem.tif", "observation": "pm25_station.csv", "meteorology": "geos_cf_meteorology.nc"})
    assert not plan["can_process"]
    assert "wind" in next(m for m in plan["matches"] if m["role"] == "meteorology")["missing"]


def test_streamlit_non_cmaq_uses_same_flow_and_partial_downloads():
    profile = water_profile()
    files = [("water.csv", "站点,氨氮(g/L),化学需氧量(mg/L)\nA,0.00011,8\nA,0.00012,9\n".encode())]
    uploads = [SimpleNamespace(name=name, getvalue=lambda content=content: content) for name, content in files]
    with patch("streamlit.file_uploader", return_value=uploads), patch("ecodata.models.RequirementStore.all", return_value=[RequirementStore().get("cmaq"), profile]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(x for x in app.radio if x.label == "功能").set_value("模型数据准备").run()
        next(x for x in app.selectbox if x.label == "目标模型").set_value(profile["name"]).run()
        next(x for x in app.button if x.label == "自动分析并生成方案").click().run()
        next(x for x in app.button if x.label == "执行模型数据准备").click().run()
        assert not app.exception and not app.error
        assert app.session_state["scenario_result"].quality["postcheck"]["complete"]
        assert len(app.get("download_button")) == len(app.session_state["scenario_result"].files) + 1
