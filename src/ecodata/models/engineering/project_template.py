"""Configuration-backed project geometry and engineering metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import geopandas as gpd

from ecodata.core.exceptions import SchemaError


@dataclass
class ModelProjectTemplate:
    model_name: str
    segments: gpd.GeoDataFrame
    simulation_start: datetime | None = None
    simulation_end: datetime | None = None
    connections: list[dict[str, Any]] = field(default_factory=list)
    boundaries: list[dict[str, Any]] = field(default_factory=list)
    state_variables: list[dict[str, Any]] = field(default_factory=list)
    source_mappings: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.segments, gpd.GeoDataFrame):
            raise SchemaError("工程模板 segments 必须是 GeoDataFrame")
        if "segment_id" not in self.segments:
            raise SchemaError("工程模板缺少 segment_id")
        if self.segments.segment_id.astype(str).duplicated().any():
            raise SchemaError("工程模板 segment_id 必须唯一")
        if not self.segments.empty and self.segments.crs is None:
            raise SchemaError("工程模板 segment 几何缺少 CRS")
        self.segments = self.segments.copy()
        self.segments["segment_id"] = self.segments.segment_id.astype(str)

    @property
    def crs(self):
        return self.segments.crs

    @classmethod
    def from_segments(cls, model_name: str, segments: gpd.GeoDataFrame, **kwargs):
        """Build a template from an existing GIS layer without inventing metadata."""
        return cls(model_name=model_name, segments=segments, **kwargs)

    def explicit_mapping(self, station_id: str) -> dict[str, Any] | None:
        station_id = str(station_id)
        return next((item for item in self.source_mappings
                     if str(item.get("station_id", item.get("source_id", ""))) == station_id), None)

    def to_engineering_project(self):
        """Compatibility bridge from the WASP template to the shared project model."""
        from .general import EngineeringProject
        objects = self.segments.rename(columns={"segment_id": "object_id"}).copy()
        objects["object_type"] = "POLYGON"
        mappings = [{**item,
                     "source_id": item.get("source_id", item.get("station_id")),
                     "target_id": item.get("target_id", item.get("segment_id"))}
                    for item in self.source_mappings]
        return EngineeringProject(self.model_name, self.simulation_start, self.simulation_end,
                                  self.crs, objects, network=self.connections,
                                  boundaries=self.boundaries, state_variables=self.state_variables,
                                  existing_mappings=mappings, provenance=self.provenance)
