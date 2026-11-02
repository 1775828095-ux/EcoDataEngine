import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

from ecodata.models import RequirementStore
from ecodata.models.adaptation import input_profile, prepare_input, suggest_mapping
from ecodata.recognition import recognise_file


def test_selected_type_does_not_require_other_model_inputs():
    full = RequirementStore().get("wasp")
    selected = input_profile(full, "water_quality_state")
    data, info = recognise_file("water.csv", "采样时间,氨氮(mg/L),断面名称\n2021-01-01,0.11,A\n".encode())
    result, report = prepare_input(data, info, selected, {}, ("氨氮",))
    assert report["ready"] and report["missing"] == []
    assert report["scope"] == "selected_input" and not report["field_schema_verified"]
    assert not report["can_generate_native_model_input"]
    assert "flow" in RequirementStore.match(full, [info], {})["missing"]
    assert result["氨氮"].item() == .11


def test_explicit_mapping_and_unit_conversion_keeps_original():
    selected = input_profile(RequirementStore().get("wasp"), "water_quality_state")
    selected["inputs"][0]["fields"] = [{"name": "NH3", "aliases": ["氨氮"], "unit": "ug/L", "data_type": "continuous", "required": True}]
    data, info = recognise_file("water.csv", "采样时间,氨氮(mg/L),断面名称\n2021-01-01,0.11,A\n".encode())
    mapping = suggest_mapping(selected["inputs"][0]["fields"], info)
    assert mapping == {"NH3": "氨氮"}
    output, report = prepare_input(data, info, selected, mapping)
    assert report["ready"]
    np.testing.assert_allclose(output.NH3, [110])
    assert output.NH3.attrs["units"] == "ug/L"
    assert data["氨氮"].item() == .11 and data["氨氮"].attrs["units"] == "mg/L"


def test_missing_physical_field_not_generated():
    selected = input_profile(RequirementStore().get("aermod"), "source_parameters")
    selected["inputs"][0]["fields"] = [{"name": "emission_rate", "unit": "g/s", "data_type": "continuous"}]
    data, info = recognise_file("water.csv", b"COD\n1.2\n")
    output, report = prepare_input(data, info, selected, {})
    assert not report["ready"] and report["missing"] == ["emission_rate"]
    assert "emission_rate" not in output


def test_incompatible_unit_is_not_relabelled():
    selected = input_profile(RequirementStore().get("wasp"), "water_quality_state")
    selected["inputs"][0]["fields"] = [{"name": "flow", "unit": "m3/s", "data_type": "continuous"}]
    data, info = recognise_file("water.csv", b"COD(mg/L)\n1.2\n")
    _, report = prepare_input(data, info, selected, {"flow": "COD"})
    assert not report["ready"] and report["unit_issues"]


def test_input_specific_frequency_not_global_default():
    full = RequirementStore().get("cmaq")
    assert full["temporal_resolution"] == "1h"
    assert input_profile(full, "grid_definition")["temporal_resolution"] is None


def test_page_mapping_unit_conversion_and_unrelated_missing_input():
    profile = RequirementStore.template()
    profile["name"] = "Mapping demo"
    profile["temporal_resolution"] = None
    profile["inputs"][0]["temporal_resolution"] = None
    other = dict(profile["inputs"][0], name="humidity", standard_name="humidity", aliases=[], unit="1")
    profile["inputs"].append(other)
    upload = SimpleNamespace(name="temperatures.csv", getvalue=lambda: b"T(degC),station_id\n0,A\n20,B\n")
    root = Path(__file__).resolve().parents[1]
    with patch("streamlit.file_uploader", return_value=[upload]), patch.object(RequirementStore, "all", return_value=[profile]), patch.object(RequirementStore, "get", return_value=profile):
        app = AppTest.from_file(str(root / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(x for x in app.radio if x.label == "输出目标").set_value("环境模型").run()
        assert not app.error
        next(x for x in app.button if x.label == "自动分析并生成方案").click().run()
        next(x for x in app.button if x.label == "执行模型数据准备").click().run()
        assert not app.exception and not app.error, [item.value for item in app.error]
        output = app.session_state["scenario_result"].output
        np.testing.assert_allclose(output.temperature, [273.15, 293.15])
        assert output.temperature.attrs["units"] == "K"
