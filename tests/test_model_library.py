from __future__ import annotations

import numpy as np
import xarray as xr

from ecodata.core.data_types import FORMAT_CAPABILITIES, SEMANTIC_TYPES, recommended_spatial_method
from ecodata.interface.workflow import TrialRequest, run_grid_trial
from ecodata.models import ModelLibrary
from ecodata.models.schema import MODEL_STATUSES, SCHEMA_VERSION


def test_library_contains_24_independent_versioned_profiles() -> None:
    library = ModelLibrary()
    profiles = library.list()
    assert len(profiles) == 24
    assert len(list(library.directory.glob("*.json"))) == 24
    assert {'calpuff', 'swat', 'swat_plus'} <= {p['id'] for p in profiles}
    assert all(profile["official_sources"] for profile in profiles)
    assert all(profile["reference_version"] for profile in profiles)
    assert all(profile["schema_version"] == SCHEMA_VERSION for profile in profiles)
    assert all(profile["status"] in MODEL_STATUSES for profile in profiles)
    assert sum(profile["status"] == "standardized_preprocessing" for profile in profiles) == 24
    assert ModelLibrary().get("wasp")["status"] == "standardized_preprocessing"
    required_input_fields = {"name", "standard_name", "unit", "data_type", "temporal_resolution", "spatial_resolution_km", "formats", "required", "depends_on", "aliases"}
    assert all(required_input_fields <= set(item) for profile in profiles for item in profile["inputs"])


def test_priority_profiles_have_detailed_requirements() -> None:
    library = ModelLibrary()
    for model_id in ("cmaq", "wrf", "swat_plus", "wasp", "modflow6"):
        profile = library.get(model_id)
        assert len(profile["required_variables"]) >= 5
        assert profile["requirements"].get("files")
        assert profile["requirements"].get("dimensions")
        assert profile["outputs"]
        assert profile["exchange_notes"]


def test_calpuff_profile_has_standardized_preparation_and_lists_core_inputs() -> None:
    profile = ModelLibrary().get("calpuff")
    assert profile["name"] == "CALPUFF"
    assert profile["status"] == "standardized_preprocessing"
    assert profile["input_formats"]
    assert {item["name"] for item in profile["required_variables"]} >= {
        "control_parameters",
        "meteorology",
        "terrain",
        "land_use",
        "emissions",
        "receptors",
    }
    assert profile["requirements"]["files"]
    assert "CALMET.DAT" in profile["requirements"]["files"]


def test_model_match_reports_missing_variables_and_never_claims_unverified_adapter() -> None:
    report = ModelLibrary().match("cmaq", {"PM2.5"}, {"NetCDF"})
    assert "meteorology" in report.missing_required_variables
    assert not report.can_generate_model_input
    assert report.adapter_status == "standardized_preprocessing"
    assert any("不会凭空生成" in message for message in report.messages)


def test_user_confirmed_file_roles_satisfy_matching_without_claiming_adapter() -> None:
    assignments = {
        "source_parameters": ["sources.csv"],
        "receptors": ["receptors.csv"],
        "meteorology": ["surface.sfc", "profile.pfl"],
        "terrain": ["terrain.dem"],
    }
    report = ModelLibrary().match("aermod", set(), {"keyword text"}, assignments)
    assert not report.missing_required_variables
    assert report.assigned_inputs["meteorology"] == ("surface.sfc", "profile.pfl")
    assert not report.can_generate_model_input


def test_role_suggestion_does_not_guess_ambiguous_files() -> None:
    library = ModelLibrary()
    assert library.suggest_role("aermod", "uploaded.csv", {"value"}) is None
    assert library.suggest_role("aermod", "terrain.csv", set()) == "terrain"


def test_semantic_and_format_capability_registry_is_explicit() -> None:
    assert set(SEMANTIC_TYPES) == {
        "continuous", "integer_count", "categorical", "string_identifier",
        "datetime", "geometry", "multidimensional_array", "quality_metadata",
    }
    assert FORMAT_CAPABILITIES["CSV"] == "read_write_verified"
    assert FORMAT_CAPABILITIES["NetCDF"] == "read_write_verified"
    assert FORMAT_CAPABILITIES["GeoTIFF"] == "structure_page_verified"
    assert FORMAT_CAPABILITIES["GRIB"] == "registered_not_implemented"
    assert recommended_spatial_method("categorical", downscaling=True) == "nearest"


def test_categorical_grid_uses_nearest_not_numeric_interpolation() -> None:
    dataset = xr.Dataset(
        {"land_class": (("lat", "lon"), np.array([[1, 2], [3, 4]], dtype=int))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        attrs={"grid_resolution_km": 10.0, "spatial_semantics": "cell_mean"},
    )
    result = run_grid_trial(
        dataset,
        TrialRequest(
            target_spatial_resolution_km=5.0,
            variable="land_class",
            unit="class_id",
            semantic_type="categorical",
        ),
    )
    assert result.history[-1]["parameters"]["method"] == "nearest"
    assert set(np.unique(result.output["land_class"])) <= {1, 2, 3, 4}
