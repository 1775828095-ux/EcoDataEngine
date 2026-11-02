from ecodata.models import RequirementStore
from ecodata.models.contracts import ModelSchema
from ecodata.models.schema import validate_profile


def test_five_target_models_have_standardized_scenarios_and_no_native_claim():
    for model_id in ("calpuff", "swat_plus", "wrf", "camx", "cmaq"):
        profile = RequirementStore().get(model_id)
        validate_profile(profile)
        schema = ModelSchema.from_profile(profile)
        assert schema.preparation["native_model_input"] is False
        assert schema.preparation["status"] == "standardized_preprocessing"
        assert schema.inputs
        assert all(role["fields"] for role in schema.inputs if not role.get("record_only"))


def test_calpuff_keeps_six_core_inputs_with_standardized_preparation_status():
    profile = RequirementStore().get("calpuff")
    assert profile["status"] == "standardized_preprocessing"
    assert {item["name"] for item in profile["required_variables"]} >= {
        "control_parameters",
        "meteorology",
        "terrain",
        "land_use",
        "emissions",
        "receptors",
    }
    assert "CALMET.DAT" in profile["requirements"]["files"]


def test_wrf_camx_and_swat_scenarios_do_not_claim_native_verification():
    for model_id in ("swat_plus", "wrf", "camx"):
        schema = ModelSchema.from_profile(RequirementStore().get(model_id))
        assert schema.preparation["adapter"] == "standardized"
        assert schema.preparation["format_scope"] == "standardized_only"
        assert schema.preparation["native_model_input"] is False
        assert schema.preparation["native_gaps"]
