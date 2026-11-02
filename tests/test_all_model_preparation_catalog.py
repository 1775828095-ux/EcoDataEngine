"""Every built-in model has an executable minimum data-preparation contract."""
import hashlib

import numpy as np
import pytest
import xarray as xr

from ecodata.models import RequirementStore
from ecodata.models.contracts import ModelSchema, ScenarioInput
from ecodata.models.scenario import execute_scenario, plan_scenario


CATALOG_MODELS = (
    "aermod", "aerscreen", "ce_qual_w2", "clm_ctsm", "efdc", "epanet",
    "hec_hms", "hec_ras", "hysplit", "mt3d_usgs", "prms", "seawat",
    "smoke", "swat", "swmm", "vic", "wrf_chem",
)


def _dataset(role):
    variables = {}
    for field in role["fields"]:
        kind = field["data_type"]
        if kind == "datetime":
            values = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        elif kind in {"string_identifier", "categorical"}:
            values = np.array([field["name"] + "_1", field["name"] + "_2"], dtype=str)
        else:
            values = np.array([1.0, 2.0])
        variables[field["name"]] = (("record",), values, {
            "units": field.get("unit", ""), "physical_variable": field["name"],
        })
    return xr.Dataset(variables, attrs={
        "structure": "table", "row_dimension": "record", "crs": "EPSG:4326",
        "preparation_test_fixture": "known-value schema contract; not native model evidence",
    })


def test_all_24_models_have_field_level_preparation_contracts():
    profiles = RequirementStore().all()
    assert len(profiles) == 24
    for profile in profiles:
        schema = ModelSchema.from_profile(profile)
        assert profile["status"] == "standardized_preprocessing"
        assert schema.inputs and all(role["fields"] for role in schema.inputs)
        assert not schema.preparation["native_model_input"]


@pytest.mark.parametrize("model_id", CATALOG_MODELS)
def test_each_new_catalog_model_executes_one_real_standardization_path(model_id):
    profile = RequirementStore().get(model_id)
    role = ModelSchema.from_profile(profile).inputs[0]
    dataset = _dataset(role)
    item = ScenarioInput(f"{model_id}.nc", dataset, hashlib.sha256(model_id.encode()).hexdigest())
    assignments = {role["role"]: item.filename}
    _, plan = plan_scenario(profile, [item], assignments)
    assert plan["can_process_partial"] or plan["can_process"]
    result = execute_scenario(profile, [item], assignments)
    assert result.files
    assert f"prepared_{role['role']}.nc" in result.files
    assert result.quality["output_checks"][role["role"]]["passed"]
    assert result.quality["completion"]["standardized_preprocessing_completed"] is False
    assert not result.quality["completion"]["native_model_input_verified"]


@pytest.mark.parametrize("model_id", CATALOG_MODELS)
def test_each_new_catalog_model_accepts_a_complete_explicit_role_set(model_id):
    profile = RequirementStore().get(model_id)
    schema = ModelSchema.from_profile(profile)
    inputs, assignments = [], {}
    for index, role in enumerate(schema.inputs):
        filename = f"{model_id}_{role['role']}.nc"
        inputs.append(ScenarioInput(filename, _dataset(role), hashlib.sha256(filename.encode()).hexdigest()))
        assignments[role["role"]] = filename
    _, plan = plan_scenario(profile, inputs, assignments)
    assert plan["can_process"], plan["errors"]
    assert not plan["native_model_input"]
