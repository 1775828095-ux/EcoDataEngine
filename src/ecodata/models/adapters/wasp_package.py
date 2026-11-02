"""WASP upstream data package exporter.

This module standardises evidence-bearing environmental data.  It does not
write WIF/INP files and never invents segment mappings or model parameters.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

from ecodata.core.exceptions import SchemaError
from ecodata.engines.units import UnitEngine
from ecodata.evaluation.self_check import PostConversionValidator
from ecodata.models.engineering import EngineeringContextResolver, ModelProjectTemplate


OUTPUTS = {
    "tributary_flow": "tributary_flow.csv",
    "tributary_concentration": "tributary_concentration.csv",
    "boundary_concentration": "boundary_concentration.csv",
    "initial_condition": "initial_condition.csv",
    "point_source_flow": "point_source_flow.csv",
    "point_source_concentration": "point_source_concentration.csv",
    "point_source_load": "point_source_load.csv",
    "environmental_forcing": "environmental_forcing.csv",
    "observations": "observations.csv",
}


@dataclass(frozen=True)
class WASPPackageResult:
    files: dict[str, bytes]
    status: dict

    def zip_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in self.files.items():
                if not content:
                    raise SchemaError(f"WASP 数据包文件为空：{name}")
                archive.writestr(name, content)
        return buffer.getvalue()


class WASPDataPackageExporter:
    """Convert recognised datasets to an auditable WASP upstream package."""

    def build(self, inputs, project_template: ModelProjectTemplate | None = None,
              max_distance_m: float = 5000.0, max_initial_time_gap: str = "7D",
              auto_confirm_threshold: str = "HIGH") -> WASPPackageResult:
        rows = {role: [] for role in OUTPUTS}
        mapping, sources, failures, warnings, audits = [], [], [], [], []
        for item in inputs:
            ds = item.dataset
            role = str(ds.attrs.get("data_role", "unclassified"))
            sources.append({"file": item.filename, "sha256": item.sha256,
                            "recognised_role": role, "variables": list(ds.data_vars)})
            try:
                if role == "tributary_inflow":
                    rows["tributary_flow"].extend(self._flow(item, audits))
                elif role == "forcing":
                    rows["environmental_forcing"].extend(self._forcing(item, audits))
                elif role in {"point_source_flow", "point_source_concentration"}:
                    rows[role].extend(self._point_source(item, role, audits))
                elif role == "initial_condition":
                    rows[role].extend(self._initial(item, audits))
                elif role == "water_quality_observation":
                    concentration, observation = self._water_quality(item, audits)
                    rows["tributary_concentration"].extend(concentration)
                    rows["observations"].extend(observation)
                else:
                    warnings.append(f"{item.filename}：已识别但没有足够证据映射为 WASP 数据角色")
            except (SchemaError, ValueError, TypeError) as exc:
                failures.append({"file": item.filename, "reason": str(exc)})

        mapping_seed = []
        for role in ("tributary_flow", "tributary_concentration", "observations",
                     "environmental_forcing", "point_source_flow",
                     "point_source_concentration", "initial_condition"):
            mapping_seed.extend(self._mapping_rows(role, rows[role]))

        load_rows, load_notes = self._derive_load(rows["point_source_flow"], rows["point_source_concentration"])
        rows["point_source_load"].extend(load_rows)
        warnings.extend(load_notes)
        mapping_seed.extend(self._mapping_rows("point_source_load", load_rows))

        context = EngineeringContextResolver(max_distance_m, max_initial_time_gap,
                                             auto_confirm_threshold).resolve(
            mapping_seed, stations=self._station_frame(rows),
            observations=pd.DataFrame(rows["observations"]), template=project_template)
        mapping = context.segment_mappings
        boundary_by_station = {item["station_id"]: item for item in context.boundary_mappings}
        for item in mapping:
            boundary = boundary_by_station.get(item["station_id"])
            if boundary:
                item.update({key: boundary.get(key) for key in ("boundary_id", "engineering_use")})

        data_files = {}
        file_quality = {}
        for role, filename in OUTPUTS.items():
            if not rows[role]:
                continue
            frame = pd.DataFrame(rows[role])
            payload = frame.to_csv(index=False).encode("utf-8-sig")
            data_files[filename] = payload
            file_quality[filename] = self._quality(frame, role)

        segment_status = context.status
        status = {
            "raw_data_recognized": bool(inputs),
            "standardized_preprocessing": bool(data_files) and not failures,
            "wasp_role_matched": bool(data_files),
            "wasp_data_package_generated": bool(data_files) and not failures,
            "segment_mapping_status": segment_status,
        }
        validation = {
            "status": "FAIL" if failures else "WARN" if warnings or segment_status != "CONFIRMED" else "PASS",
            "checks": file_quality,
            "failures": failures,
            "warnings": warnings,
            "conversion_audits": audits,
            "dropped_records": int(sum(item.get("dropped_records", 0) for item in audits)),
            "segment_mapping_status": segment_status,
            "load_derivation": {
                "formula": "load_kg_day = flow_m3_s * concentration_mg_L * 86.4",
                "generated_records": len(load_rows),
                "requirements": ["same source/location", "matching time", "explicit compatible units", "QA pass"],
            },
            "scope": "WASP upstream standard data package; not WIF/INP/native model input",
        }
        manifest = {
            "model": "WASP",
            "package_scope": "standardized_upstream_data",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "sources": sources,
            "files": {name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                      for name, content in data_files.items()},
        }
        files = dict(data_files)
        files["mapping.json"] = self._json({"segment_mapping_status": segment_status,
                                              "mappings": mapping,
                                              "boundary_candidates": context.boundary_mappings,
                                              "initial_condition_candidates": context.initial_condition_candidates})
        files["manifest.json"] = self._json(manifest)
        files["validation_report.json"] = self._json(validation)
        return WASPPackageResult(files, status)

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False).encode("utf-8")

    @staticmethod
    def _times(ds):
        if "time" not in ds:
            raise SchemaError("缺少可靠时间字段")
        return pd.to_datetime(ds.time.values.ravel(), errors="coerce")

    def _convert(self, values, source_unit, target_unit, variable, audits):
        if not source_unit:
            raise SchemaError(f"{variable} 缺少单位，禁止生成 WASP 标准数据")
        source = xr.Dataset({variable: xr.DataArray(np.asarray(values, dtype=float), dims="record",
                                                    attrs={"units": source_unit})})
        converted = source[variable].copy() if source_unit == target_unit else UnitEngine().convert(source[variable], target_unit)
        converted.attrs["units"] = target_unit
        after = xr.Dataset({variable: converted})
        check = PostConversionValidator().validate(after, after, kind="model_adaptation",
            history=[{"method": "unit_conversion", "variable": variable,
                      "source_unit": source_unit, "target_unit": target_unit}])
        audits.append({"variable": variable, "source_unit": source_unit, "target_unit": target_unit,
                       "input_records": int(source[variable].size), "output_records": int(converted.size),
                       "dropped_records": 0, "validator_status": check.details["status"]})
        return converted.values

    def _flow(self, item, audits):
        ds = item.dataset
        if "value" not in ds or "station_id" not in ds:
            raise SchemaError("USGS 流量缺少 value/station_id")
        values = self._convert(ds.value.values, ds.value.attrs.get("units", ""), "m^3/s", "flow", audits)
        times = self._times(ds)
        result = []
        for index, value in enumerate(values):
            if pd.isna(times[index]) or not np.isfinite(value):
                continue
            result.append({"time": times[index].isoformat(), "source_id": str(ds.station_id.values[index]),
                           "station_id": str(ds.station_id.values[index]), "flow": float(value), "unit": "m^3/s",
                           "quality_code": str(ds.quality_code.values[index]) if "quality_code" in ds else "",
                           "source_file": item.filename, "source_parameter": str(ds.parameter_code.values[index]) if "parameter_code" in ds else "",
                           "source_value": str(ds.raw_value.values[index]) if "raw_value" in ds else str(ds.value.values[index]),
                           "source_unit": ds.value.attrs.get("units", ""), "segment_id": ""})
        audits[-1]["dropped_records"] = len(values) - len(result)
        return result

    def _point_source(self, item, role, audits):
        ds, times = item.dataset, self._times(item.dataset)
        value_name = "flow" if role == "point_source_flow" else "concentration"
        if value_name not in ds or "source_id" not in ds:
            raise SchemaError(f"{role} 缺少 {value_name}/source_id")
        target_unit = "m^3/s" if role == "point_source_flow" else "mg/L"
        values = self._convert(ds[value_name].values, ds[value_name].attrs.get("units", ""),
                               target_unit, value_name, audits)
        result = []
        for index, value in enumerate(values):
            if pd.isna(times[index]) or not np.isfinite(value):
                continue
            row = {"time": times[index].isoformat(), "source_id": str(ds.source_id.values[index]),
                   "station_id": str(ds.station_id.values[index]) if "station_id" in ds else str(ds.source_id.values[index]),
                   value_name: float(value), "unit": target_unit,
                   "quality_code": str(ds.quality_code.values[index]) if "quality_code" in ds else "",
                   "source_file": item.filename, "segment_id": str(ds.segment_id.values[index]) if "segment_id" in ds else ""}
            if role == "point_source_concentration":
                if "variable" not in ds:
                    raise SchemaError("point_source_concentration 缺少 variable")
                row["variable"] = str(ds.variable.values[index])
            result.append(row)
        audits[-1]["dropped_records"] = len(values) - len(result)
        return result

    def _initial(self, item, audits):
        ds = item.dataset
        if not {"segment_id", "variable", "value"} <= set(ds):
            raise SchemaError("initial_condition 缺少 segment_id/variable/value")
        unit = ds.value.attrs.get("units", "")
        values = self._convert(ds.value.values, unit, "mg/L", "initial_condition", audits)
        return [{"source_id": item.filename, "station_id": "", "segment_id": str(ds.segment_id.values[index]),
                 "variable": str(ds.variable.values[index]), "value": float(value), "unit": "mg/L",
                 "quality_code": str(ds.quality_code.values[index]) if "quality_code" in ds else "",
                 "source_file": item.filename}
                for index, value in enumerate(values) if np.isfinite(value)]

    def _water_quality(self, item, audits):
        ds, times = item.dataset, self._times(item.dataset)
        station_name = next((name for name in ("Station", "station_id", "station") if name in ds), None)
        if not station_name:
            station_values = np.repeat(item.filename, len(times))
        else:
            station_values = ds[station_name].values
        waterbody = ds["Waterbody"].values if "Waterbody" in ds else np.repeat("", len(times))
        concentration, observation = [], []
        variables = [name for name in ds.data_vars if name.endswith("__numeric") or
                     (ds[name].dtype.kind in "fi" and ds[name].attrs.get("physical_variable"))]
        if not variables:
            raise SchemaError("水质观测没有可用的连续物理变量")
        for name in variables:
            physical = ds[name].attrs.get("physical_variable", name)
            unit = ds[name].attrs.get("units", "")
            target = "mg/L" if unit in {"ug/L", "mg/L"} else unit
            values = self._convert(ds[name].values, unit, target, physical, audits)
            raw_name = str(ds[name].attrs.get("ancillary_variables", ""))
            raw = ds[raw_name].values if raw_name in ds else ds[name].values
            for index, value in enumerate(values):
                token = str(raw[index]).strip()
                quality = "EXACT" if np.isfinite(value) else "CENSORED_OR_MISSING" if token else "MISSING"
                if pd.isna(times[index]):
                    continue
                record = {"time": times[index].isoformat(), "source_id": str(station_values[index]),
                          "station_id": str(station_values[index]), "variable": physical,
                          "value": None if not np.isfinite(value) else float(value), "unit": target,
                          "quality_code": quality, "raw_value": token, "source_unit": unit,
                          "source_file": item.filename, "segment_id": ""}
                record.update(self._coordinates(ds, index))
                if str(waterbody[index]).strip().upper() == "RIVER":
                    concentration.append({**{key: value for key, value in record.items() if key != "value"},
                                          "concentration": record["value"]})
                else:
                    observation.append(record)
            audits[-1]["dropped_records"] = int(pd.isna(times).sum())
        return concentration, observation

    def _forcing(self, item, audits):
        ds, times = item.dataset, self._times(item.dataset)
        result = []
        ignored = {"latitude", "longitude", "time"}
        for name in ds.data_vars:
            if name in ignored or name.endswith("_quality_code") or name.endswith("_hours") or ds[name].dtype.kind not in "fi":
                continue
            unit = ds[name].attrs.get("units", "")
            if not unit:
                raise SchemaError(f"{name} 缺少单位，禁止生成 forcing")
            values = self._convert(ds[name].values, unit, unit, name, audits)
            quality_name = name + "_quality_code"
            for index, value in enumerate(values):
                if pd.isna(times[index]) or not np.isfinite(value):
                    continue
                result.append({"time": times[index].isoformat(), "source_id": str(ds.station.values[index]) if "station" in ds else item.filename,
                               "station_id": str(ds.station.values[index]) if "station" in ds else "",
                               "variable": name, "value": float(value), "unit": unit,
                               "quality_code": str(ds[quality_name].values[index]) if quality_name in ds else "",
                               "source_file": item.filename, "segment_id": "",
                               **self._coordinates(ds, index)})
            audits[-1]["dropped_records"] = len(values) - sum(np.isfinite(values) & ~pd.isna(times))
        return result

    @staticmethod
    def _mapping_rows(role, rows):
        seen, result = set(), []
        for row in rows:
            key = (row.get("source_file", ""), row.get("source_id", ""), role)
            if key in seen:
                continue
            seen.add(key)
            segment = row.get("segment_id", "")
            result.append({"source_id": row.get("source_id", ""), "station_id": row.get("station_id", ""),
                           "source_role": role, "target_role": role, "segment_id": segment or None})
        return result

    @staticmethod
    def _station_frame(rows):
        records = []
        for values in rows.values():
            for row in values:
                latitude, longitude = row.get("latitude"), row.get("longitude")
                if latitude in (None, "") or longitude in (None, ""):
                    continue
                records.append({"station_id": str(row.get("station_id", row.get("source_id", ""))),
                                "geometry": Point(float(longitude), float(latitude))})
        if not records:
            return None
        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326").drop_duplicates("station_id")

    @staticmethod
    def _coordinates(ds, index):
        result = {}
        for source, target in (("latitude", "latitude"), ("lat", "latitude"),
                               ("longitude", "longitude"), ("lon", "longitude")):
            if source not in ds or target in result:
                continue
            values = np.asarray(ds[source].values).ravel()
            value = values[0] if values.size == 1 else values[index]
            if np.isfinite(value):
                result[target] = float(value)
        return result

    @staticmethod
    def _derive_load(flow_rows, concentration_rows):
        if not flow_rows or not concentration_rows:
            return [], ["point_source load 未生成：缺少 point_source_flow 或 point_source_concentration"]
        flow = pd.DataFrame(flow_rows)
        concentration = pd.DataFrame(concentration_rows)
        keys = [key for key in ("source_id", "station_id", "time") if key in flow and key in concentration]
        if set(keys) != {"source_id", "station_id", "time"}:
            return [], ["point_source load 未生成：来源/位置/时间键不完整"]
        merged = flow.merge(concentration, on=keys, suffixes=("_flow", "_concentration"))
        merged = merged[(merged.unit_flow == "m^3/s") & (merged.unit_concentration == "mg/L")]
        merged = merged[merged.quality_code_flow.ne("") & merged.quality_code_concentration.ne("")]
        result = []
        for row in merged.itertuples():
            result.append({"time": row.time, "source_id": row.source_id, "station_id": row.station_id,
                           "variable": row.variable, "load": float(row.flow * row.concentration * 86.4),
                           "unit": "kg/day", "quality_code": f"{row.quality_code_flow}|{row.quality_code_concentration}",
                           "source_file": f"{row.source_file_flow};{row.source_file_concentration}",
                           "formula": "flow_m3_s * concentration_mg_L * 86.4", "segment_id": ""})
        return result, [] if result else ["point_source load 未生成：没有同来源、同位置、同时间且 QA/单位合格的记录"]

    @staticmethod
    def _quality(frame, role):
        numeric = [name for name in frame if pd.api.types.is_numeric_dtype(frame[name])]
        return {"status": "PASS", "role": role, "records": len(frame), "columns": list(frame),
                "missing_values": {name: int(frame[name].isna().sum()) for name in frame},
                "numeric_min_max": {name: [float(frame[name].min()), float(frame[name].max())] for name in numeric if frame[name].notna().any()},
                "source_trace_present": "source_file" in frame and frame.source_file.astype(bool).all(),
                "quality_preserved": "quality_code" in frame,
                "segment_mapping_status": "CONFIRMED" if "segment_id" in frame and frame.segment_id.astype(bool).all() else "NEEDS_CONFIRMATION"}
