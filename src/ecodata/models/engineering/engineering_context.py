"""Orchestrate project-template, spatial, boundary and initial mappings."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

from .boundary_mapper import BoundaryMapper
from .initial_condition_mapper import InitialConditionMapper
from .project_template import ModelProjectTemplate
from .spatial_mapper import SegmentMapper
from .general import SpatialEngineeringMapper


@dataclass(frozen=True)
class EngineeringContext:
    status: str
    segment_mappings: list[dict]
    boundary_mappings: list[dict]
    initial_condition_candidates: list[dict]


class EngineeringContextResolver:
    def __init__(self, max_distance_m=5000.0, max_initial_time_gap="7D",
                 auto_confirm_threshold="HIGH"):
        self.segment_mapper = SegmentMapper(max_distance_m, auto_confirm_threshold)
        self.boundary_mapper = BoundaryMapper()
        self.initial_mapper = InitialConditionMapper(max_initial_time_gap)

    def resolve(self, source_rows: list[dict], stations: gpd.GeoDataFrame | None = None,
                observations: pd.DataFrame | None = None,
                template: ModelProjectTemplate | None = None) -> EngineeringContext:
        unique = {}
        for row in source_rows:
            station_id = str(row.get("station_id", row.get("source_id", "")))
            unique.setdefault(station_id, row)
        if template is None:
            mappings = [self._without_template(station_id, row) for station_id, row in unique.items()]
            status = "CONFIRMED" if mappings and all(item.get("confirmed") for item in mappings) else "NEEDS_PROJECT_TEMPLATE"
            return EngineeringContext(status, mappings, [], [])
        if stations is None or stations.empty:
            mappings = [self._without_coordinates(station_id, row, template) for station_id, row in unique.items()]
        else:
            generic_sources = stations.rename(columns={"station_id": "source_id"})
            generic = SpatialEngineeringMapper(self.segment_mapper.max_distance_m,
                                               self.segment_mapper.auto_confirm_threshold).map(
                                                   generic_sources, template.to_engineering_project(), "POLYGON")
            spatial = {item["source_id"]: {"station_id": item["source_id"],
                       "segment_id": item["target_id"], "mapping_method": item["mapping_method"],
                       "distance_m": item["distance"], "confidence": item["confidence"],
                       "confirmed": item["confirmed"], "reason": item["evidence"],
                       "evidence": {"generic_engineering_mapper": True}}
                       for item in generic}
            mappings = []
            for station_id, row in unique.items():
                item = spatial.get(station_id, self._without_coordinates(station_id, row, template))
                mappings.append({**self._base(row), **item})
        boundaries = self.boundary_mapper.map(source_rows, mappings, template)
        initials = self.initial_mapper.map(observations if observations is not None else pd.DataFrame(),
                                           mappings, template)
        status = "CONFIRMED" if mappings and all(item.get("confirmed") for item in mappings) else "NEEDS_CONFIRMATION"
        return EngineeringContext(status, mappings, boundaries, initials)

    @staticmethod
    def _base(row):
        role = row.get("source_role", row.get("target_role", ""))
        return {"source_id": row.get("source_id", ""), "station_id": row.get("station_id", ""),
                "source_role": role, "target_role": row.get("target_role", role),
                "boundary_id": None, "engineering_use": None}

    def _without_template(self, station_id, row):
        if row.get("segment_id") not in (None, ""):
            return {**self._base(row), "station_id": station_id, "segment_id": str(row["segment_id"]),
                    "mapping_method": "explicit_source_metadata", "distance_m": None,
                    "confidence": "HIGH", "confirmed": True,
                    "reason": "输入数据明确提供 segment_id", "evidence": {"project_template": False}}
        return {**self._base(row), "station_id": station_id, "segment_id": None,
                "mapping_method": "project_template_missing", "distance_m": None,
                "confidence": "NEEDS_CONFIRMATION", "confirmed": False,
                "reason": "缺少模型工程模板/segment 几何，不能自动建立工程用途映射",
                "evidence": {"project_template": False}}

    def _without_coordinates(self, station_id, row, template):
        explicit = template.explicit_mapping(station_id)
        if explicit and explicit.get("segment_id") not in (None, ""):
            return {**self._base(row), "station_id": station_id, "segment_id": str(explicit["segment_id"]),
                    "mapping_method": "project_template", "distance_m": None,
                    "confidence": "HIGH", "confirmed": True,
                    "reason": "使用工程模板中的专家映射", "evidence": {"project_template": True}}
        return {**self._base(row), "station_id": station_id, "segment_id": None,
                "mapping_method": "station_coordinates_missing", "distance_m": None,
                "confidence": "NEEDS_CONFIRMATION", "confirmed": False,
                "reason": "站点缺少可核验坐标，不能执行空间映射",
                "evidence": {"project_template": True, "station_coordinates": False}}
