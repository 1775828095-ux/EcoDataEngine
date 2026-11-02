"""Conservative boundary-candidate inference from explicit topology evidence."""
from __future__ import annotations

from .project_template import ModelProjectTemplate


class BoundaryMapper:
    BOUNDARY_ROLES = {"tributary_flow", "tributary_concentration", "boundary_concentration"}

    def map(self, source_rows: list[dict], segment_mappings: list[dict],
            template: ModelProjectTemplate) -> list[dict]:
        mapped = {item["station_id"]: item for item in segment_mappings}
        incoming = {str(item.get("to_segment")) for item in template.connections if item.get("to_segment") is not None}
        upstream = {str(item.get("from_segment")) for item in template.connections if item.get("from_segment") is not None}
        entry_segments = upstream - incoming
        results = []
        seen = set()
        for row in source_rows:
            role = row.get("source_role", row.get("target_role", ""))
            station_id = str(row.get("station_id", row.get("source_id", "")))
            if role not in self.BOUNDARY_ROLES or (station_id, role) in seen:
                continue
            seen.add((station_id, role))
            explicit = template.explicit_mapping(station_id) or {}
            if explicit.get("boundary_id"):
                results.append(self._record(station_id, explicit.get("segment_id"), explicit["boundary_id"],
                                            "project_template", "HIGH", True,
                                            "工程模板明确指定 boundary", explicit.get("boundary_type")))
                continue
            spatial = mapped.get(station_id, {})
            segment_id = spatial.get("segment_id")
            if not template.connections:
                results.append(self._record(station_id, segment_id, None, "topology_unavailable",
                                            "NEEDS_CONFIRMATION", False,
                                            "工程模板没有连接/流向拓扑，不能判定边界"))
            elif segment_id in entry_segments:
                results.append(self._record(station_id, segment_id, None, "network_entry_candidate",
                                            "MEDIUM", False,
                                            "站点映射到无上游入边的网络入口；仅作为边界候选"))
            else:
                results.append(self._record(station_id, segment_id, None, "not_network_entry",
                                            "NEEDS_CONFIRMATION", False,
                                            "当前证据不足以证明该位置是模型边界"))
        return results

    @staticmethod
    def _record(station_id, segment_id, boundary_id, method, confidence, confirmed, reason,
                boundary_type=None):
        return {"station_id": station_id, "segment_id": segment_id, "boundary_id": boundary_id,
                "boundary_type": boundary_type,
                "engineering_use": "BOUNDARY" if confirmed else "BOUNDARY_CANDIDATE",
                "mapping_method": method, "confidence": confidence, "confirmed": confirmed,
                "reason": reason, "evidence": {"topology_method": method}}
