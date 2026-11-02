"""Editable unified model requirements and conservative data matching."""
from __future__ import annotations

import json
from pathlib import Path

from ecodata.models.library import ModelLibrary
from ecodata.models.schema import editable_template, normalize_profile, validate_profile


CONVERTIBLE_FORMATS = {
    "CSV", "TSV", "Excel", "JSON", "Parquet", "NetCDF", "HDF5", "HDF5-NetCDF",
    "GeoTIFF", "GeoJSON", "Shapefile ZIP",
}
GENERIC_UNITS = {
    "", "model-specific", "model/project-specific", "state-variable-specific",
    "state/parameter-specific", "parameter-specific", "variable-specific",
    "file/variable-specific", "consistent model units",
}


class RequirementStore:
    """One store backed by all 24 built-in profiles plus editable overrides."""

    def __init__(self, path: Path | None = None, library: ModelLibrary | None = None):
        self.path = path or Path(__file__).resolve().parents[3] / "data" / "model_requirements.json"
        self.library = library or ModelLibrary()

    def all(self) -> list[dict]:
        profiles = {item["id"]: item for item in self.library.list()}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            profiles.update({item["id"]: normalize_profile(item) for item in saved})
        return sorted(profiles.values(), key=lambda item: item["name"])

    def get(self, model_id: str) -> dict:
        for item in self.all():
            if item["id"] == model_id:
                return item
        raise KeyError(model_id)

    def save(self, profile: dict) -> None:
        normalized = normalize_profile(profile)
        validate_profile(normalized)
        custom = []
        if self.path.exists():
            custom = json.loads(self.path.read_text(encoding="utf-8"))
        custom = [item for item in custom if item["id"] != normalized["id"]]
        saved = {key: value for key, value in normalized.items() if key not in {"required_variables", "adapter_status"}}
        custom.append(saved)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(custom, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def template() -> dict:
        return editable_template()

    @staticmethod
    def match(profile: dict, descriptors: list, assignments: dict[str, str]) -> dict:
        """Classify every required input as satisfied, convertible, missing or unsupported."""
        profile = normalize_profile(profile)
        by_filename = {item.filename: item for item in descriptors}
        variables: dict[str, list[tuple[object, str]]] = {}
        for descriptor in descriptors:
            for name in descriptor.variables:
                variables.setdefault(name.casefold(), []).append((descriptor, name))

        satisfied, convertible, missing, unsupported, unit_issues = [], [], [], [], []
        conversion_plan = []
        for requirement in profile["inputs"]:
            role = requirement["name"]
            keys = {role.casefold(), requirement["standard_name"].casefold(), *(item.casefold() for item in requirement["aliases"])}
            candidates = [item for key in keys for item in variables.get(key, [])]
            assigned_file = next((filename for filename, assigned_role in assignments.items() if assigned_role == role), None)
            if assigned_file and assigned_file in by_filename:
                descriptor = by_filename[assigned_file]
                matched_name = role
                source = "用户确认文件角色"
            elif candidates:
                descriptor, matched_name = candidates[0]
                source = "变量名/标准名/别名精确匹配"
            else:
                if requirement["required"]:
                    missing.append(role)
                continue

            reason = _unsupported_reason(requirement, descriptor)
            if reason:
                unsupported.append({"requirement": role, "file": descriptor.filename, "reason": reason})
                continue

            actual_unit = descriptor.units.get(matched_name, "")
            expected_unit = requirement["unit"]
            if expected_unit.casefold() not in GENERIC_UNITS and actual_unit and actual_unit != expected_unit:
                unit_issues.append({"variable": role, "file": descriptor.filename, "expected": expected_unit, "actual": actual_unit})
                continue

            accepted = set(requirement["formats"])
            if descriptor.format in accepted:
                satisfied.append({"requirement": role, "file": descriptor.filename, "reason": source})
            elif descriptor.format in CONVERTIBLE_FORMATS and profile["output_format"] in CONVERTIBLE_FORMATS:
                step = f"{descriptor.format} → {profile['output_format']}"
                convertible.append({"requirement": role, "file": descriptor.filename, "conversion": step, "reason": source})
                conversion_plan.append({"input": role, "file": descriptor.filename, "operation": step})
            else:
                unsupported.append({"requirement": role, "file": descriptor.filename, "reason": f"当前不能把 {descriptor.format} 转换为该需求接受的格式"})

        ready = not missing and not unsupported and not unit_issues
        native = ready and profile["status"] == "verified_native_adapter"
        return {
            "model_id": profile["id"], "model_name": profile["name"], "model_version": profile["version"],
            "model_status": profile["status"], "satisfied": satisfied, "convertible": convertible,
            "missing": missing, "unsupported": unsupported, "unit_issues": unit_issues,
            "conversion_plan": conversion_plan, "target_format": profile["output_format"],
            "target_time_resolution": profile.get("temporal_resolution"),
            "target_spatial_resolution_km": profile.get("spatial_resolution_km"),
            "ready_for_standardized_preprocessing": ready,
            "can_generate_native_model_input": native, "ready": ready,
            "matched": [item["requirement"] for item in satisfied + convertible],
        }


def _unsupported_reason(requirement: dict, descriptor) -> str | None:
    if requirement["data_type"] == "geometry":
        spatial = descriptor.structure in {"grid", "vector"} or bool(descriptor.x_coordinate and descriptor.y_coordinate)
        if not spatial:
            return "该输入需要空间几何或longitude/latitude，上传文件没有可靠空间坐标"
        if not descriptor.crs:
            return "该输入需要CRS，上传文件没有可靠CRS"
    if requirement["temporal_resolution"] and not descriptor.time_coordinate:
        return "该输入需要时间字段，上传文件未识别到可靠时间坐标"
    return None
