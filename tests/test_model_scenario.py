import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.core.exceptions import SchemaError
from ecodata.models import RequirementStore
from ecodata.models.schema import validate_profile
from ecodata.models.scenario import recognise_inputs, suggest_roles, plan_scenario, execute_scenario
from ecodata.io.structures import read_structure

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cmaq_case"


def case_inputs():
    files = [(path.name, path.read_bytes()) for path in sorted((CASE / "input").iterdir())]
    return recognise_inputs(files, {"pm25_station.csv": {"temporal_semantics": "mean"}})


def test_model_scenario_schema_and_role_matching():
    profile = RequirementStore().get("cmaq")
    validate_profile(profile)
    inputs = case_inputs()
    roles = suggest_roles(profile, inputs)
    analysis, plan = plan_scenario(profile, inputs, roles)
    assert plan["can_process"] and not plan["native_model_input"]
    assert roles["observation"] == "pm25_station.csv"
    assert len(analysis) == 4
    assert any(item["role"] == "boundary" and not item["can_process"] for item in plan["matches"])
    broken = deepcopy(profile)
    broken["preparation_scenario"]["inputs"][0].pop("dimensions")
    with pytest.raises(ValueError):
        validate_profile(broken)


def test_missing_input_and_ambiguous_role_do_not_pass():
    profile, inputs = RequirementStore().get("cmaq"), case_inputs()
    roles = suggest_roles(profile, inputs)
    roles["terrain"] = ""
    _, plan = plan_scenario(profile, inputs, roles)
    assert not plan["can_process"] and any("地形" in item for item in plan["errors"])
    roles["terrain"] = roles["background"]
    _, plan = plan_scenario(profile, inputs, roles)
    assert not plan["can_process"]


def test_nonoverlapping_times_block_plan_before_execution():
    from ecodata.models.scenario import ScenarioInput
    profile, inputs = RequirementStore().get("cmaq"), case_inputs()
    inputs = [ScenarioInput(item.filename, item.dataset.assign_coords(time=item.dataset.time + np.timedelta64(30, "D")), item.sha256)
              if item.filename == "geos_cf_meteorology.nc" else item for item in inputs]
    _, plan = plan_scenario(profile, inputs, suggest_roles(profile, inputs))
    assert not plan["can_process"]
    assert any("共同真实时间" in error for error in plan["errors"])


def test_real_cmaq_preparation_execution_and_roundtrip():
    profile, inputs = RequirementStore().get("cmaq"), case_inputs()
    result = execute_scenario(profile, inputs, suggest_roles(profile, inputs))
    restored = read_structure("model_input.nc", result.files["model_input.nc"])
    assert result.quality["passed"] and not result.quality["native_model_input"]
    assert restored.PM25.shape == (8, 3, 3)
    assert restored.elevation.shape == (3, 3)
    assert int(restored.PM25_bound_flag.sum()) == result.quality["negative_fused_cells"] == 30
    np.testing.assert_array_equal(restored.PM25_background, result.fused.background_value)
    np.testing.assert_array_equal(restored.PM25_raw_fused, result.fused.PM25)
    np.testing.assert_array_equal(restored.PM25, np.maximum(restored.PM25_raw_fused, 0))
    assert result.quality["model_input.nc"]["passed"]
    assert json.loads(result.files["processing_plan.json"])["scope"] == "auxiliary_preparation"


@pytest.mark.parametrize("problem", ["unit", "crs", "time_bounds", "negative", "time_semantics", "physical_variable"])
def test_scenario_refuses_incompatible_physical_data(problem):
    profile, inputs = RequirementStore().get("cmaq"), case_inputs()
    roles = suggest_roles(profile, inputs)
    by_name = {item.filename: item.dataset for item in inputs}
    if problem == "unit":
        by_name[roles["background"]].PM25.attrs["units"] = "unknown"
    elif problem == "crs":
        by_name[roles["terrain"]].attrs.pop("crs")
    elif problem == "time_bounds":
        by_name[roles["meteorology"]].time_bnds.values[:] += np.timedelta64(1, "h")
    elif problem == "time_semantics":
        by_name[roles["observation"]].attrs["temporal_semantics"] = "instantaneous"
    elif problem == "physical_variable":
        by_name[roles["observation"]]["variable"] = ("__row__", np.repeat("NO2", by_name[roles["observation"]].sizes["__row__"]))
    with pytest.raises(SchemaError):
        execute_scenario(profile, inputs, roles, negative_policy="reject" if problem == "negative" else "clip")


def test_streamlit_real_upload_model_execution_download_and_stale_clear():
    uploads = [SimpleNamespace(name=p.name, getvalue=p.read_bytes) for p in sorted((CASE / "input").iterdir())]
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(x for x in app.radio if x.label == "功能").set_value("模型数据准备").run()
        next(x for x in app.selectbox if "pm25_station.csv 时间语义" in x.label).set_value("mean").run()
        next(x for x in app.button if x.label == "自动分析并生成方案").click().run()
        next(x for x in app.button if x.label == "执行模型数据准备").click().run()
        assert not app.exception and not app.error
        result = app.session_state["scenario_result"]
        assert result.quality["passed"] and len(app.get("download_button")) == len(result.files) + 1
        assert {"before_profile.json", "after_profile.json", "validation_report.json"} <= set(result.files)
        next(x for x in app.selectbox if x.label == "负浓度处理").set_value("阻止输出").run()
        assert not app.get("download_button")
        next(x for x in app.button if x.label == "自动分析并生成方案").click().run()
        next(x for x in app.button if x.label == "执行模型数据准备").click().run()
        assert app.error and not app.exception
        assert not app.get("download_button")
