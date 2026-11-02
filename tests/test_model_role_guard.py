"""Physical-role gates cover automatic, manual and legacy export paths."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.fusion.planning import DataRoleClassifier
from ecodata.models import RequirementStore
from ecodata.models.adaptation import input_profile, prepare_input
from ecodata.models.scenario import (recognise_inputs, suggest_roles, plan_scenario,
                                     execute_scenario, apply_field_mappings)
from ecodata.recognition import inspect_dataset
from ecodata.task import ConversionTaskEngine
from ecodata.interface.workflow import TrialRequest
from ecodata.io.structures import StructureRequest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cmaq_case/input"


def inputs():
    return recognise_inputs([(p.name, p.read_bytes()) for p in sorted(CASE.iterdir())],
                            {"pm25_station.csv": {"temporal_semantics": "mean"}})


@pytest.mark.parametrize("role,filename", [("meteorology", "dem.tif"), ("terrain", "geos_cf_meteorology.nc")])
def test_wrong_roles_fail_before_adapter_execution(role, filename):
    profile, data = RequirementStore().get("cmaq"), inputs()
    assignments = suggest_roles(profile, data)
    assignments[role] = filename
    _, plan = plan_scenario(profile, data, assignments)
    assert plan["role_conflicts"] and not plan["can_process"] and not plan["can_process_partial"]
    with patch("ecodata.models.adapters.cmaq.CMAQAdapter.execute", side_effect=AssertionError("must not execute")):
        with pytest.raises(SchemaError):
            execute_scenario(profile, data, assignments)


def test_observation_dem_and_multiple_files_automatically_assigned():
    profile, data = RequirementStore().get("cmaq"), inputs()
    assignments = suggest_roles(profile, data)
    assert assignments["terrain"] == "dem.tif"
    assert assignments["observation"] == "pm25_station.csv"
    assert assignments["background"] == "pm25_background.nc"
    assert assignments["meteorology"] == "geos_cf_meteorology.nc"
    _, plan = plan_scenario(profile, data, assignments)
    assert plan["can_process"] and not plan["role_conflicts"]
    classifier = DataRoleClassifier()
    datasets = {i.filename: i.dataset for i in data}
    assert classifier.classify(datasets["dem.tif"], "elevation") == "static_constraint"
    assert classifier.classify(datasets["pm25_station.csv"], "PM25") == "observation"


def test_forced_field_mapping_cannot_disguise_elevation_as_temperature():
    profile, data = RequirementStore().get("cmaq"), inputs()
    assignments = suggest_roles(profile, data)
    assignments["meteorology"] = "dem.tif"
    with pytest.raises(SchemaError, match="不兼容"):
        apply_field_mappings(profile, data, assignments, {"meteorology": {"temperature": "elevation"}})


def test_user_declared_role_cannot_override_terrain_evidence():
    dem = next(i.dataset.copy(deep=True) for i in inputs() if i.filename == "dem.tif")
    dem.attrs["data_role"] = "forcing"
    classifier = DataRoleClassifier()
    assert classifier.classify(dem, "elevation") == "static_constraint"
    assert not classifier.compatibility(dem, {"role": "meteorology", "fields": []}, variables=("elevation",))["passed"]


def test_legacy_selected_meteorology_cannot_export_terrain():
    dem = next(i.dataset for i in inputs() if i.filename == "dem.tif")
    profile = input_profile(RequirementStore().get("cmaq"), "meteorology")
    _, report = prepare_input(dem, inspect_dataset("dem.tif", dem), profile, {}, ("elevation",))
    assert not report["ready"] and not report["role_compatibility"]["passed"]
    with patch.object(ConversionTaskEngine, "_execute", side_effect=AssertionError("must not convert")):
        with pytest.raises(SchemaError, match="不兼容"):
            ConversionTaskEngine().execute([dem], StructureRequest(variable="elevation"),
                TrialRequest(variable="elevation"), model=profile)


def test_generic_partial_processing_also_blocks_any_role_conflict():
    profile = deepcopy(RequirementStore().get("cmaq"))
    profile["preparation_scenario"]["adapter"] = "standardized"
    data = inputs()
    assignments = suggest_roles(profile, data)
    assignments["meteorology"] = "dem.tif"
    _, plan = plan_scenario(profile, data, assignments)
    assert not plan["can_process_partial"]
    with pytest.raises(SchemaError):
        execute_scenario(profile, data, assignments)


def test_single_dem_prepares_only_verified_terrain_role():
    profile = RequirementStore().get("cmaq")
    data = [item for item in inputs() if item.filename == "dem.tif"]
    assignments = suggest_roles(profile, data)
    _, plan = plan_scenario(profile, data, assignments)
    assert not plan["can_process"] and plan["can_process_partial"]
    result = execute_scenario(profile, data, assignments)
    assert "prepared_terrain.nc" in result.files
    assert not any("meteorology" in name or name == "model_input.nc" for name in result.files)
    assert not result.quality["completion"]["complete_model_input"]
    assert not result.quality["completion"]["standardized_preprocessing_completed"]


@pytest.mark.parametrize("role", ["meteorology", "initial_concentrations", "boundary_concentrations"])
def test_legacy_dem_cannot_become_any_concentration_or_meteorology(role):
    dem = next(i.dataset for i in inputs() if i.filename == "dem.tif")
    profile = input_profile(RequirementStore().get("cmaq"), role)
    _, report = prepare_input(dem, inspect_dataset("dem.tif", dem), profile, {}, ("elevation",))
    assert not report["ready"]


def test_page_analysis_then_execute_and_forced_role_has_no_download():
    uploads = [SimpleNamespace(name=p.name, getvalue=p.read_bytes) for p in sorted(CASE.iterdir())]
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(x for x in app.radio if x.label == "功能").set_value("模型数据准备").run()
        assert not app.exception and not app.error
        assert not any(x.label in {"我要把当前数据转换成哪一种模型输入", "该输入类型使用的当前变量"} for x in app.selectbox)
        assert not any(x.label == "执行模型数据准备" for x in app.button)
        next(x for x in app.selectbox if "pm25_station.csv 时间语义" in x.label).set_value("mean").run()
        next(x for x in app.button if x.label == "自动分析并生成方案").click().run()
        assert not app.error
        assert not next(x for x in app.button if x.label == "执行模型数据准备").disabled
        next(x for x in app.button if x.label == "执行模型数据准备").click().run()
        assert not app.error and app.get("download_button")
        next(x for x in app.selectbox if x.label == "修改 气象场 的来源").set_value("dem.tif").run()
        assert not app.get("download_button")
        next(x for x in app.button if x.label == "自动分析并生成方案").click().run()
        assert app.error and not app.exception
        assert not app.get("download_button")
        assert not any(x.label == "执行模型数据准备" and not x.disabled for x in app.button)
