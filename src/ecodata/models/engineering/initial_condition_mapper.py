"""Select auditable initial-condition candidates without interpolation."""
from __future__ import annotations

import pandas as pd

from .project_template import ModelProjectTemplate


class InitialConditionMapper:
    GOOD_QA = {"VALID", "GOOD", "APPROVED", "EXACT", "A", "PASS"}

    def __init__(self, max_time_gap="7D"):
        self.max_time_gap = pd.Timedelta(max_time_gap)

    def map(self, observations: pd.DataFrame, segment_mappings: list[dict],
            template: ModelProjectTemplate) -> list[dict]:
        if template.simulation_start is None or not template.state_variables:
            return []
        mapping = {str(item["station_id"]): item for item in segment_mappings if item.get("segment_id")}
        frame = observations.copy()
        if frame.empty:
            return [self._missing(segment, variable, "没有观测记录")
                    for segment in template.segments.segment_id for variable in self._variables(template)]
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
        start = pd.Timestamp(template.simulation_start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        frame["station_id"] = frame.station_id.astype(str)
        frame["segment_id"] = frame.station_id.map(lambda value: mapping.get(value, {}).get("segment_id"))
        frame["distance_m"] = frame.station_id.map(lambda value: mapping.get(value, {}).get("distance_m"))
        quality = frame["quality_code"] if "quality_code" in frame else pd.Series("", index=frame.index)
        frame["qa_ok"] = quality.astype(str).str.upper().isin(self.GOOD_QA)
        frame["time_gap"] = (frame.time - start).abs()
        results = []
        for segment in template.segments.segment_id.astype(str):
            for variable in self._variables(template):
                candidates = frame[(frame.segment_id == segment) & (frame.variable.astype(str) == variable["name"])
                                   & frame.qa_ok & frame.value.notna()].copy()
                candidates = candidates[candidates.time_gap <= self.max_time_gap]
                if candidates.empty:
                    results.append(self._missing(segment, variable, "无通过 QA 且在最大时间差内的观测"))
                    continue
                expected_unit = variable.get("unit")
                if expected_unit:
                    candidates = candidates[candidates.unit.astype(str) == str(expected_unit)]
                if candidates.empty:
                    results.append(self._missing(segment, variable, "观测单位与状态变量要求不一致"))
                    continue
                candidates["distance_sort"] = pd.to_numeric(candidates.distance_m, errors="coerce").fillna(float("inf"))
                candidates = candidates.sort_values(["time_gap", "distance_sort", "station_id"])
                best = candidates.iloc[0]
                tied = candidates[(candidates.time_gap == best.time_gap)
                                  & (candidates.distance_sort == best.distance_sort)]
                ambiguous = len(tied) > 1
                results.append({"segment_id": segment, "variable": variable["name"],
                                "station_id": str(best.station_id), "value": float(best.value),
                                "unit": str(best.unit), "observation_time": best.time.isoformat(),
                                "time_offset_seconds": float(best.time_gap.total_seconds()),
                                "distance_m": None if pd.isna(best.distance_m) else float(best.distance_m),
                                "engineering_use": "INITIAL_CONDITION_CANDIDATE",
                                "mapping_method": "qa_time_space_ranking",
                                "confidence": "NEEDS_CONFIRMATION" if ambiguous else "HIGH",
                                "confirmed": False,
                                "reason": "多个同等候选需确认" if ambiguous else "QA通过，且按时间差、空间距离排序最优",
                                "source": str(best.get("source_file", "")),
                                "evidence": {"quality_code": str(best.get("quality_code", "")),
                                             "max_time_gap": str(self.max_time_gap)}})
        return results

    @staticmethod
    def _variables(template):
        return [item if isinstance(item, dict) else {"name": str(item)} for item in template.state_variables]

    @staticmethod
    def _missing(segment, variable, reason):
        return {"segment_id": str(segment), "variable": variable["name"], "station_id": None,
                "engineering_use": "INITIAL_CONDITION_CANDIDATE", "mapping_method": "no_candidate",
                "confidence": "MISSING", "confirmed": False, "reason": reason, "evidence": {}}
