from datetime import datetime, timezone

import geopandas as gpd
import pytest
import xarray as xr
from shapely.geometry import Point, box, LineString

from ecodata.models.contracts import ScenarioInput
from ecodata.models.engineering import (EngineeringProject, SpatialEngineeringMapper,
    engineering_mapping_document, get_engineering_profile, prepare_for_model)


MODELS = ["WASP", "CALPUFF", "SWAT+", "WRF", "CAMx", "CMAQ"]


def _project(object_type="GRID"):
    objects = gpd.GeoDataFrame({"object_id": ["A", "B"], "object_type": [object_type] * 2,
        "geometry": [box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)]}, crs="EPSG:3857")
    return EngineeringProject("test", datetime(2009, 1, 1, tzinfo=timezone.utc),
                              crs=objects.crs, spatial_objects=objects)


def _input(role="observation"):
    ds = xr.Dataset({"longitude": ("record", [100.0]), "latitude": ("record", [100.0]),
                     "station_id": ("record", ["S1"]), "value": ("record", [1.0])},
                    attrs={"crs": "EPSG:3857", "engineering_role": role})
    return ScenarioInput("station.csv", ds, "a" * 64)


@pytest.mark.parametrize("model", MODELS)
def test_six_models_share_one_prepare_interface_and_profile(model):
    profile = get_engineering_profile(model)
    assert profile.model_name == model
    project = _project("POLYGON" if model in {"WASP", "SWAT+"} else "GRID")
    result = prepare_for_model([_input()], model, project)
    assert result.standardized_data
    assert result.engineering_mapping[0]["target_id"] == "A"
    assert set(result.__dict__) == {"standardized_data", "engineering_mapping", "model_package",
                                    "missing_requirements", "confirmation_requests", "validation_report"}


def test_missing_project_context_does_not_block_standardization():
    result = prepare_for_model([_input()], "CMAQ")
    assert result.standardized_data
    assert result.missing_requirements == [{"requirement": "project_context", "status": "NEEDS_PROJECT_CONTEXT"}]
    assert result.validation_report["engineering_context"] == "NEEDS_PROJECT_CONTEXT"


def test_existing_wasp_roles_are_normalized_by_shared_role_resolver():
    item = _input()
    item.dataset.attrs["engineering_role"] = "water_quality_observation"
    result = prepare_for_model([item], "WASP")
    assert result.validation_report["role_checks"][0]["role"] == "observation"
    assert result.validation_report["role_checks"][0]["status"] == "MATCHED"


def test_expert_mapping_overrides_automatic_geometry():
    project = _project()
    project.existing_mappings = [{"source_id": "S1", "target_id": "B"}]
    result = prepare_for_model([_input()], "CMAQ", project)
    assert result.engineering_mapping[0]["target_id"] == "B"
    assert result.engineering_mapping[0]["mapping_method"] == "project_template"


def test_nearest_line_and_network_reach_use_shared_mapper():
    objects = gpd.GeoDataFrame({"object_id": ["R1"], "object_type": ["NETWORK_EDGE"],
                                "geometry": [LineString([(0, 0), (1000, 0)])]}, crs="EPSG:3857")
    project = EngineeringProject("SWAT+", crs=objects.crs, spatial_objects=objects)
    sources = gpd.GeoDataFrame({"source_id": ["S"], "geometry": [Point(10, 25)]}, crs="EPSG:3857")
    result = SpatialEngineeringMapper(100).map(sources, project, "NETWORK_EDGE", "nearest_line")
    assert result[0]["target_id"] == "R1" and result[0]["distance"] == pytest.approx(25)


def test_polygon_to_grid_uses_maximum_overlap():
    project = _project("GRID")
    sources = gpd.GeoDataFrame({"source_id": ["P"],
                                "geometry": [box(100, 100, 800, 800)]}, crs="EPSG:3857")
    result = SpatialEngineeringMapper().map(sources, project, "GRID")
    assert result[0]["target_id"] == "A"
    assert result[0]["mapping_method"] == "maximum_overlap"


def test_confirmed_mapping_document_is_reusable_and_auditable():
    document = engineering_mapping_document(
        [{"source_id": "S", "target_id": "A", "target_type": "GRID",
          "confidence": "HIGH", "confirmed": True, "mapping_method": "within"}],
        "CMAQ", "project-1", confirmed_by_user=True, provenance={"file": "grid.nc"})
    assert document["mappings"][0]["confirmed_by_user"] is True
    assert document["mappings"][0]["provenance"] == {"file": "grid.nc"}
    assert document["generated_at"]


def test_wasp_template_has_lossless_bridge_to_generic_project():
    from ecodata.models.engineering import ModelProjectTemplate
    segments = gpd.GeoDataFrame({"segment_id": ["S1"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    old = ModelProjectTemplate("WASP", segments, source_mappings=[{"station_id": "A", "segment_id": "S1"}])
    new = old.to_engineering_project()
    assert new.spatial_objects.iloc[0].object_id == "S1"
    assert new.existing_mappings[0]["target_id"] == "S1"
