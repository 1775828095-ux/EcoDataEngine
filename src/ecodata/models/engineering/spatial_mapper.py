"""Evidence-bearing station-to-segment mapping using GeoPandas joins."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from ecodata.core.exceptions import SchemaError
from .project_template import ModelProjectTemplate


class SegmentMapper:
    def __init__(self, max_distance_m: float = 5000.0, auto_confirm_threshold: str = "HIGH"):
        if max_distance_m <= 0:
            raise ValueError("max_distance_m 必须大于 0")
        self.max_distance_m = float(max_distance_m)
        if auto_confirm_threshold not in {"HIGH", "MEDIUM"}:
            raise ValueError("auto_confirm_threshold 仅支持 HIGH 或 MEDIUM")
        self.auto_confirm_threshold = auto_confirm_threshold

    @staticmethod
    def _projected(stations: gpd.GeoDataFrame, segments: gpd.GeoDataFrame):
        if stations.crs is None:
            raise SchemaError("站点缺少 CRS，禁止计算空间距离")
        if segments.crs is None:
            raise SchemaError("segment 几何缺少 CRS，禁止空间映射")
        stations = stations.to_crs(segments.crs)
        if segments.crs.is_projected:
            return stations, segments
        target = segments.estimate_utm_crs() or stations.estimate_utm_crs()
        if target is None:
            raise SchemaError("无法为工程范围确定可靠的投影 CRS")
        return stations.to_crs(target), segments.to_crs(target)

    def map(self, stations: gpd.GeoDataFrame, template: ModelProjectTemplate) -> list[dict]:
        if "station_id" not in stations:
            raise SchemaError("站点数据缺少 station_id")
        if stations.station_id.astype(str).duplicated().any():
            stations = stations.drop_duplicates("station_id", keep="first").copy()
        points, polygons = self._projected(stations.copy(), template.segments.copy())
        points["station_id"] = points.station_id.astype(str)
        polygon_view = polygons[["segment_id", "geometry"]]
        within = gpd.sjoin(points, polygon_view, how="left", predicate="within")
        inside = within.dropna(subset=["segment_id"]).groupby("station_id")["segment_id"].agg(list).to_dict()

        result = []
        for _, point in points.iterrows():
            station_id = str(point.station_id)
            candidates = list(dict.fromkeys(str(value) for value in inside.get(station_id, [])))
            automatic = None
            if len(candidates) == 1:
                automatic = self._record(station_id, candidates[0], "within", 0.0, "HIGH", True,
                                         "站点位于唯一 segment 多边形内部")
            elif len(candidates) > 1:
                automatic = self._record(station_id, None, "within_ambiguous", 0.0,
                                         "NEEDS_CONFIRMATION", False,
                                         "站点同时落入多个 segment，不能自动选择")
            else:
                one = gpd.GeoDataFrame([point], geometry="geometry", crs=points.crs)
                nearest = gpd.sjoin_nearest(one, polygon_view, how="left",
                                            max_distance=self.max_distance_m,
                                            distance_col="distance_m")
                nearest = nearest.dropna(subset=["segment_id"])
                if nearest.empty:
                    automatic = self._record(station_id, None, "nearest_refused", None,
                                             "NEEDS_CONFIRMATION", False,
                                             f"最近 segment 超过最大距离 {self.max_distance_m:g} m")
                else:
                    best_distance = float(nearest.distance_m.min())
                    best = nearest[nearest.distance_m == best_distance]
                    ids = list(dict.fromkeys(best.segment_id.astype(str)))
                    if len(ids) != 1:
                        automatic = self._record(station_id, None, "nearest_ambiguous", best_distance,
                                                 "NEEDS_CONFIRMATION", False,
                                                 "存在等距 segment，不能自动选择")
                    else:
                        automatic = self._record(station_id, ids[0], "nearest", best_distance,
                                                 "MEDIUM", self.auto_confirm_threshold == "MEDIUM",
                                                 "最近 segment 在允许距离内，需用户确认")

            explicit = template.explicit_mapping(station_id)
            if explicit and explicit.get("segment_id") not in (None, ""):
                segment_id = str(explicit["segment_id"])
                if segment_id not in set(template.segments.segment_id):
                    raise SchemaError(f"显式映射引用不存在的 segment_id：{segment_id}")
                conflict = automatic.get("segment_id") not in (None, segment_id)
                reason = "使用工程模板中的专家映射"
                if conflict:
                    reason += f"；与空间自动候选 {automatic['segment_id']} 冲突，模板优先"
                automatic = self._record(station_id, segment_id, "project_template", 0.0,
                                         "HIGH", True, reason)
                automatic["mapping_conflict"] = conflict
            result.append(automatic)
        return result

    @staticmethod
    def _record(station_id, segment_id, method, distance_m, confidence, confirmed, reason):
        return {"station_id": station_id, "segment_id": segment_id,
                "mapping_method": method, "distance_m": distance_m,
                "confidence": confidence, "confirmed": confirmed,
                "reason": reason,
                "evidence": {"station_id": station_id, "spatial_method": method}}
