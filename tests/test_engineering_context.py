from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from ecodata.core.exceptions import SchemaError
from ecodata.models.engineering import (BoundaryMapper, EngineeringContextResolver,
                                        InitialConditionMapper, ModelProjectTemplate,
                                        SegmentMapper)


def _segments(crs="EPSG:3857"):
    return gpd.GeoDataFrame({"segment_id": ["S1", "S2"],
                             "geometry": [box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)]}, crs=crs)


def _template(**kwargs):
    values = {"model_name": "WASP", "segments": _segments(),
              "connections": [{"from_segment": "S1", "to_segment": "S2"}],
              "simulation_start": datetime(2009, 1, 1, tzinfo=timezone.utc),
              "state_variables": [{"name": "DO", "unit": "mg/L"}]}
    values.update(kwargs)
    return ModelProjectTemplate(**values)


def _station(x, y, station_id="A", crs="EPSG:3857"):
    return gpd.GeoDataFrame({"station_id": [station_id], "geometry": [Point(x, y)]}, crs=crs)


def test_within_polygon_is_high_confidence():
    row = SegmentMapper().map(_station(100, 100), _template())[0]
    assert (row["segment_id"], row["confidence"], row["confirmed"]) == ("S1", "HIGH", True)


def test_nearest_within_limit_is_medium_and_unconfirmed():
    row = SegmentMapper(max_distance_m=300).map(_station(2100, 500), _template())[0]
    assert row["segment_id"] == "S2"
    assert row["confidence"] == "MEDIUM" and not row["confirmed"]
    assert row["distance_m"] == pytest.approx(100)


def test_different_crs_is_reprojected_before_mapping():
    row = SegmentMapper().map(_station(0.005, 0.005, crs="EPSG:4326"), _template())[0]
    assert row["segment_id"] == "S1" and row["confidence"] == "HIGH"


def test_nearest_beyond_limit_refuses_mapping():
    row = SegmentMapper(max_distance_m=50).map(_station(2100, 500), _template())[0]
    assert row["segment_id"] is None and row["confidence"] == "NEEDS_CONFIRMATION"


def test_missing_station_crs_fails():
    with pytest.raises(SchemaError, match="CRS"):
        SegmentMapper().map(_station(100, 100, crs=None), _template())


def test_project_template_mapping_overrides_spatial_candidate_and_reports_conflict():
    template = _template(source_mappings=[{"station_id": "A", "segment_id": "S2"}])
    row = SegmentMapper().map(_station(100, 100), template)[0]
    assert row["segment_id"] == "S2" and row["mapping_conflict"] is True
    assert row["mapping_method"] == "project_template"


def test_explicit_boundary_mapping_is_final():
    template = _template(source_mappings=[{"station_id": "A", "segment_id": "S1", "boundary_id": "B1"}])
    rows = [{"station_id": "A", "source_role": "tributary_flow"}]
    result = BoundaryMapper().map(rows, [{"station_id": "A", "segment_id": "S1"}], template)[0]
    assert result["engineering_use"] == "BOUNDARY" and result["confirmed"]


def test_topology_only_produces_boundary_candidate():
    rows = [{"station_id": "A", "source_role": "tributary_flow"}]
    result = BoundaryMapper().map(rows, [{"station_id": "A", "segment_id": "S1"}], _template())[0]
    assert result["engineering_use"] == "BOUNDARY_CANDIDATE" and not result["confirmed"]


def test_initial_candidate_uses_closest_qa_passed_observation():
    observations = pd.DataFrame([
        {"station_id": "A", "time": "2008-12-31T00:00:00Z", "variable": "DO", "value": 5., "unit": "mg/L", "quality_code": "VALID"},
        {"station_id": "A", "time": "2008-12-20T00:00:00Z", "variable": "DO", "value": 9., "unit": "mg/L", "quality_code": "VALID"},
    ])
    mappings = [{"station_id": "A", "segment_id": "S1", "distance_m": 0}]
    result = InitialConditionMapper("7D").map(observations, mappings, _template())
    s1 = next(item for item in result if item["segment_id"] == "S1")
    assert s1["value"] == 5 and s1["confidence"] == "HIGH"


def test_initial_observation_outside_time_gap_is_missing():
    observations = pd.DataFrame([{"station_id": "A", "time": "2008-01-01T00:00:00Z",
                                  "variable": "DO", "value": 5., "unit": "mg/L", "quality_code": "VALID"}])
    result = InitialConditionMapper("7D").map(observations,
        [{"station_id": "A", "segment_id": "S1", "distance_m": 0}], _template())
    assert next(item for item in result if item["segment_id"] == "S1")["confidence"] == "MISSING"


def test_initial_qa_failure_is_missing():
    observations = pd.DataFrame([{"station_id": "A", "time": "2009-01-01T00:00:00Z",
                                  "variable": "DO", "value": 5., "unit": "mg/L", "quality_code": "REJECTED"}])
    result = InitialConditionMapper().map(observations,
        [{"station_id": "A", "segment_id": "S1", "distance_m": 0}], _template())
    assert next(item for item in result if item["segment_id"] == "S1")["confidence"] == "MISSING"


def test_no_project_template_keeps_package_mapping_possible():
    context = EngineeringContextResolver().resolve(
        [{"source_id": "A", "station_id": "A", "source_role": "tributary_flow"}])
    assert context.status == "NEEDS_PROJECT_TEMPLATE"
    assert context.segment_mappings[0]["segment_id"] is None


def test_template_without_station_coordinates_requires_confirmation():
    context = EngineeringContextResolver().resolve(
        [{"source_id": "A", "station_id": "A", "source_role": "tributary_flow"}], template=_template())
    assert context.status == "NEEDS_CONFIRMATION"
    assert context.segment_mappings[0]["mapping_method"] == "station_coordinates_missing"
