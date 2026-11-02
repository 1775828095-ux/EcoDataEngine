"""Model-independent engineering context, mapping and preparation contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ecodata.core.exceptions import SchemaError


OBJECT_TYPES = {"POINT", "LINE", "POLYGON", "GRID", "NETWORK_NODE",
                "NETWORK_EDGE", "VOLUME", "LAYER"}
ROLES = {"observation", "forcing", "initial_condition", "boundary_condition",
         "source", "load", "emission", "static_field", "terrain", "landuse",
         "hydrology", "meteorology", "calibration", "validation", "quality_auxiliary"}


@dataclass
class EngineeringProject:
    model_name: str
    simulation_start: datetime | None = None
    simulation_end: datetime | None = None
    crs: Any | None = None
    spatial_objects: gpd.GeoDataFrame | None = None
    network: list[dict] = field(default_factory=list)
    domains: list[dict] = field(default_factory=list)
    boundaries: list[dict] = field(default_factory=list)
    vertical_structure: list[dict] = field(default_factory=list)
    source_locations: list[dict] = field(default_factory=list)
    state_variables: list[dict] = field(default_factory=list)
    existing_mappings: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.spatial_objects is None:
            self.spatial_objects = gpd.GeoDataFrame(
                {"object_id": [], "object_type": []}, geometry=[], crs=self.crs)
            return
        if not isinstance(self.spatial_objects, gpd.GeoDataFrame):
            raise SchemaError("spatial_objects 必须是 GeoDataFrame")
        required = {"object_id", "object_type"}
        if not required <= set(self.spatial_objects):
            raise SchemaError("spatial_objects 缺少 object_id/object_type")
        invalid = set(self.spatial_objects.object_type.astype(str).str.upper()) - OBJECT_TYPES
        if invalid:
            raise SchemaError(f"不支持的工程空间对象：{sorted(invalid)}")
        if not self.spatial_objects.empty and self.spatial_objects.crs is None:
            raise SchemaError("工程空间对象缺少 CRS")
        self.spatial_objects = self.spatial_objects.copy()
        self.spatial_objects["object_id"] = self.spatial_objects.object_id.astype(str)
        self.spatial_objects["object_type"] = self.spatial_objects.object_type.astype(str).str.upper()
        self.crs = self.spatial_objects.crs or self.crs

    def explicit_mapping(self, source_id):
        source_id = str(source_id)
        return next((item for item in self.existing_mappings
                     if str(item.get("source_id", item.get("station_id", ""))) == source_id), None)


@dataclass(frozen=True)
class ModelEngineeringProfile:
    model_name: str
    spatial_structure_type: tuple[str, ...]
    required_engineering_objects: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    initial_condition_rules: dict = field(default_factory=dict)
    boundary_rules: dict = field(default_factory=dict)
    source_rules: dict = field(default_factory=dict)
    mapping_rules: tuple[dict, ...] = ()
    required_fields: dict = field(default_factory=dict)
    validation_status: str = "profile_only"


class SpatialEngineeringMapper:
    """GIS mappings backed by GeoPandas/Shapely after explicit CRS alignment."""
    def __init__(self, max_distance_m=5000.0, auto_confirm_threshold="HIGH"):
        self.max_distance_m = float(max_distance_m)
        self.auto_confirm_threshold = auto_confirm_threshold

    @staticmethod
    def _metric(source, target):
        if source.crs is None or target.crs is None:
            raise SchemaError("源数据或工程对象缺少 CRS，禁止空间映射")
        source = source.to_crs(target.crs)
        if target.crs.is_projected:
            return source, target
        metric = target.estimate_utm_crs() or source.estimate_utm_crs()
        if metric is None:
            raise SchemaError("无法确定用于距离计算的投影 CRS")
        return source.to_crs(metric), target.to_crs(metric)

    def map(self, sources: gpd.GeoDataFrame, project: EngineeringProject,
            target_type: str, method: str | None = None) -> list[dict]:
        if "source_id" not in sources:
            raise SchemaError("空间源数据缺少 source_id")
        target_type = target_type.upper()
        targets = project.spatial_objects[
            project.spatial_objects.object_type == target_type].copy()
        if targets.empty:
            return [self._record(str(row.source_id), None, target_type, "target_missing",
                                 None, "MISSING", False, "工程缺少目标空间对象")
                    for _, row in sources.drop_duplicates("source_id").iterrows()]
        source, target = self._metric(sources.drop_duplicates("source_id").copy(), targets)
        result = []
        for _, row in source.iterrows():
            source_id = str(row.source_id)
            explicit = project.explicit_mapping(source_id)
            if explicit and explicit.get("target_id"):
                result.append(self._record(source_id, str(explicit["target_id"]), target_type,
                                           "project_template", 0.0, "HIGH", True,
                                           "使用已有专家工程映射"))
                continue
            record = self._map_one(row, target, target_type, method)
            result.append(record)
        return result

    def _map_one(self, row, targets, target_type, method):
        source_id = str(row.source_id)
        polygons = target_type in {"POLYGON", "GRID", "VOLUME", "LAYER"}
        if polygons and row.geometry.geom_type == "Point" and method in {None, "within", "point_to_polygon", "point_to_grid"}:
            hits = targets[targets.geometry.covers(row.geometry)]
            if len(hits) == 1:
                return self._record(source_id, hits.iloc[0].object_id, target_type, "within",
                                    0.0, "HIGH", True, "点位于唯一工程对象内部")
            if len(hits) > 1:
                return self._record(source_id, None, target_type, "within_ambiguous", 0.0,
                                    "NEEDS_CONFIRMATION", False, "点同时位于多个工程对象")
        if polygons and row.geometry.geom_type != "Point":
            overlap = targets.geometry.intersection(row.geometry).area
            if overlap.max() > 0:
                best = targets[overlap == overlap.max()]
                if len(best) == 1:
                    ratio = float(overlap.max() / row.geometry.area) if row.geometry.area else None
                    return self._record(source_id, best.iloc[0].object_id, target_type,
                                        "maximum_overlap", 0.0, "HIGH", True,
                                        f"最大空间重叠；source_overlap_ratio={ratio}")
                return self._record(source_id, None, target_type, "overlap_ambiguous", 0.0,
                                    "NEEDS_CONFIRMATION", False, "多个工程对象具有相同最大重叠面积")
        distances = targets.geometry.distance(row.geometry)
        minimum = float(distances.min())
        best = targets[distances == minimum]
        if len(best) != 1 or minimum > self.max_distance_m:
            return self._record(source_id, None, target_type, "nearest_refused", minimum,
                                "NEEDS_CONFIRMATION", False, "最近对象不唯一或超过允许距离")
        return self._record(source_id, best.iloc[0].object_id, target_type,
                            "nearest", minimum, "MEDIUM",
                            self.auto_confirm_threshold == "MEDIUM", "最近工程对象候选")

    @staticmethod
    def _record(source_id, target_id, target_type, method, distance, confidence, confirmed, evidence):
        return {"source_id": source_id, "target_id": None if target_id is None else str(target_id),
                "target_type": target_type, "mapping_method": method,
                "distance": distance, "confidence": confidence, "evidence": evidence,
                "confirmed": confirmed}


class EngineeringRoleResolver:
    ALIASES = {"water_quality_observation": "observation", "pollutant_observation": "observation",
               "tributary_inflow": "hydrology", "tributary_flow": "hydrology",
               "tributary_concentration": "boundary_condition",
               "boundary_concentration": "boundary_condition",
               "point_source_flow": "source", "point_source_concentration": "source",
               "point_source_load": "load", "static_constraint": "static_field",
               "target_state": "initial_condition"}

    def resolve(self, role: str, profile: ModelEngineeringProfile, variable=""):
        role = str(role or "").casefold()
        role = self.ALIASES.get(role, role)
        if role not in ROLES:
            return {"role": role or "unclassified", "status": "NEEDS_CONFIRMATION",
                    "reason": "数据角色无法可靠识别"}
        if role not in profile.allowed_roles:
            return {"role": role, "status": "INCOMPATIBLE",
                    "reason": f"{profile.model_name} Profile 不允许该角色"}
        permitted = profile.required_fields.get(role, ())
        if variable and permitted and variable.casefold() not in {str(v).casefold() for v in permitted}:
            return {"role": role, "status": "NEEDS_CONFIRMATION",
                    "reason": "变量不在该角色的已知字段中"}
        return {"role": role, "status": "MATCHED", "reason": "符合模型工程 Profile"}


class InitialConditionResolver:
    GOOD_QA = {"VALID", "GOOD", "APPROVED", "EXACT", "A", "PASS"}

    def resolve(self, observations: pd.DataFrame, mappings: list[dict],
                project: EngineeringProject, profile: ModelEngineeringProfile):
        if project.simulation_start is None:
            return [{"status": "NEEDS_PROJECT_CONTEXT", "reason": "缺少 simulation_start"}]
        if observations.empty:
            return [{"status": "MISSING", "reason": "没有初始状态候选观测"}]
        required = {"source_id", "time", "variable", "value", "unit"}
        if not required <= set(observations):
            return [{"status": "MISSING", "reason": f"缺少字段 {sorted(required-set(observations))}"}]
        frame = observations.copy()
        frame["time"] = pd.to_datetime(frame.time, errors="coerce", utc=True)
        start = pd.Timestamp(project.simulation_start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        frame["gap"] = (frame.time - start).abs()
        max_gap = pd.Timedelta(profile.initial_condition_rules.get("max_time_gap", "7D"))
        if "quality_code" in frame:
            frame = frame[frame.quality_code.astype(str).str.upper().isin(self.GOOD_QA)]
        frame = frame[frame.gap <= max_gap]
        target_by_source = {str(item["source_id"]): item.get("target_id") for item in mappings}
        frame["target_id"] = frame.source_id.astype(str).map(target_by_source)
        result = []
        for (target, variable), group in frame.dropna(subset=["target_id"]).groupby(["target_id", "variable"]):
            best = group.sort_values(["gap", "source_id"]).iloc[0]
            result.append({"target_id": target, "variable": variable, "value": float(best.value),
                           "unit": str(best.unit), "source_id": str(best.source_id),
                           "status": "CANDIDATE", "confidence": "HIGH", "confirmed": False,
                           "time_offset_seconds": float(best.gap.total_seconds())})
        return result or [{"status": "MISSING", "reason": "没有满足时间、QA和空间要求的候选"}]


class BoundaryConditionResolver:
    def resolve(self, mappings: list[dict], project: EngineeringProject,
                profile: ModelEngineeringProfile):
        explicit = {str(item.get("source_id")): item for item in project.existing_mappings
                    if item.get("boundary_id")}
        results = []
        for mapping in mappings:
            source_id = str(mapping["source_id"])
            if source_id in explicit:
                results.append({**mapping, "boundary_id": explicit[source_id]["boundary_id"],
                                "status": "CONFIRMED", "confidence": "HIGH",
                                "evidence": "已有专家工程边界配置"})
            elif not project.boundaries and not project.network:
                results.append({**mapping, "boundary_id": None,
                                "status": "NEEDS_PROJECT_CONTEXT", "confidence": "NEEDS_CONFIRMATION",
                                "evidence": "缺少边界或拓扑结构，不能自动判定"})
            else:
                results.append({**mapping, "boundary_id": None, "status": "CANDIDATE",
                                "confidence": "NEEDS_CONFIRMATION",
                                "evidence": profile.boundary_rules.get("evidence", "需按模型边界规则确认")})
        return results


@dataclass
class EngineeringPreparationResult:
    standardized_data: list
    engineering_mapping: list[dict]
    model_package: Any
    missing_requirements: list[dict]
    confirmation_requests: list[dict]
    validation_report: dict


class EngineeringContextEngine:
    def __init__(self, profile: ModelEngineeringProfile, max_distance_m=5000.0):
        self.profile = profile
        self.mapper = SpatialEngineeringMapper(max_distance_m)
        self.roles = EngineeringRoleResolver()

    def prepare(self, data, project_context: EngineeringProject | None = None):
        data = list(data or [])
        role_results = []
        for item in data:
            dataset = getattr(item, "dataset", item)
            role_results.append(self.roles.resolve(dataset.attrs.get("engineering_role", dataset.attrs.get("data_role", "")), self.profile))
        missing = []
        mappings = []
        confirmations = [item for item in role_results if item["status"] == "NEEDS_CONFIRMATION"]
        if project_context is None:
            missing.append({"requirement": "project_context", "status": "NEEDS_PROJECT_CONTEXT"})
        else:
            present = set(project_context.spatial_objects.object_type)
            for required in self.profile.required_engineering_objects:
                if required.upper() not in present:
                    missing.append({"requirement": required, "status": "MISSING"})
            sources = self._source_points(data)
            for rule in self.profile.mapping_rules:
                if not sources.empty:
                    mapped = self.mapper.map(sources, project_context, rule["target_type"], rule.get("method"))
                    mappings.extend(mapped)
                    confirmations.extend(item for item in mapped if not item["confirmed"])
            if mappings:
                boundary = BoundaryConditionResolver().resolve(mappings, project_context, self.profile)
                mappings.extend({**item, "mapping_kind": "boundary"} for item in boundary)
                observations = self._observations(data)
                initial = InitialConditionResolver().resolve(observations, mappings,
                                                              project_context, self.profile)
                mappings.extend({**item, "mapping_kind": "initial_condition"} for item in initial)
                confirmations.extend(item for item in boundary + initial
                                     if item.get("confirmed") is False or item.get("status") in {"CANDIDATE", "NEEDS_PROJECT_CONTEXT"})
        status = "PASS" if not missing and not confirmations else "WARN"
        return EngineeringPreparationResult(data, mappings, None, missing, confirmations,
            {"status": status, "model": self.profile.model_name,
             "profile_validation_status": self.profile.validation_status,
             "standardized_preprocessing": True,
             "engineering_context": "AVAILABLE" if project_context else "NEEDS_PROJECT_CONTEXT",
             "role_checks": role_results})

    @staticmethod
    def _source_points(data):
        rows = []
        crs = None
        for item in data:
            ds = getattr(item, "dataset", item)
            names = set(ds.variables)
            lon = next((v for v in ("longitude", "lon", "x") if v in names), None)
            lat = next((v for v in ("latitude", "lat", "y") if v in names), None)
            sid = next((v for v in ("station_id", "source_id", "id") if v in names), None)
            if not lon or not lat:
                continue
            crs = ds.attrs.get("crs", "EPSG:4326")
            values = zip(ds[lon].values.ravel(), ds[lat].values.ravel(),
                         ds[sid].values.ravel() if sid else range(ds[lon].size))
            rows.extend({"source_id": str(i), "geometry": Point(float(x), float(y))}
                        for x, y, i in values if pd.notna(x) and pd.notna(y))
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs or "EPSG:4326")

    @staticmethod
    def _observations(data):
        frames = []
        required = {"time", "variable", "value", "unit"}
        for item in data:
            ds = getattr(item, "dataset", item)
            if not required <= set(ds.variables):
                continue
            frame = ds[list(required | ({"source_id"} if "source_id" in ds else {"station_id"}))].to_dataframe().reset_index()
            if "source_id" not in frame and "station_id" in frame:
                frame = frame.rename(columns={"station_id": "source_id"})
            if "quality_code" in ds:
                frame["quality_code"] = ds.quality_code.values.ravel()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def engineering_mapping_document(mappings, model, project="", confirmed_by_user=False, provenance=None):
    """Serializable confirmation record; expert mappings remain highest priority."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return {"model": model, "project": project, "generated_at": timestamp,
            "mappings": [{"source": item.get("source_id"), "target": item.get("target_id"),
                          "target_type": item.get("target_type"),
                          "confidence": item.get("confidence"),
                          "confirmed_by_user": bool(confirmed_by_user or item.get("confirmed_by_user")),
                          "timestamp": item.get("timestamp", timestamp),
                          "provenance": item.get("provenance", provenance or {}),
                          "mapping_method": item.get("mapping_method")}
                         for item in mappings if item.get("source_id") is not None]}
