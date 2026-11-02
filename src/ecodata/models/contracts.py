"""Shared model preparation contracts; no model-specific processing rules."""
from dataclasses import dataclass
from copy import deepcopy

import xarray as xr
import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.models.schema import normalize_profile
from ecodata.models.requirements import GENERIC_UNITS


@dataclass(frozen=True)
class ScenarioInput:
    filename: str
    dataset: xr.Dataset
    sha256: str


@dataclass(frozen=True)
class ScenarioResult:
    output: xr.Dataset
    fused: xr.Dataset
    files: dict[str, bytes]
    analysis: list[dict]
    plan: dict
    quality: dict


@dataclass(frozen=True)
class ModelSchema:
    """Executable view of the same RequirementStore, not a second model library.

    Human-readable requirements remain documentation. Only explicit processing
    settings authorize transformations; a requirement index is not an adapter.
    """
    profile: dict
    preparation: dict

    @classmethod
    def from_profile(cls, profile):
        profile = normalize_profile(profile)
        preparation = deepcopy(profile.get("preparation_scenario"))
        if preparation is None:
            roles = []
            for item in profile["inputs"]:
                fields = deepcopy(item.get("fields", []))
                if not fields and item["data_type"] not in {"multidimensional_array", "geometry"} and item["unit"].casefold() not in GENERIC_UNITS:
                    fields = [{key: deepcopy(item[key]) for key in ("name", "standard_name", "unit", "data_type", "aliases")}]
                roles.append({**deepcopy(item), "role": item["name"], "fields": fields,
                              "dimensions": item.get("dimensions", []), "crs": item.get("crs", "未明确"),
                              "time": item.get("temporal_resolution"), "space": item.get("spatial_resolution_km")})
            preparation = {"id": profile["id"] + "_preparation", "name": profile["name"] + " 标准化准备",
                           "adapter": "standardized", "inputs": roles, "target_crs": None,
                           "format_scope": "declared_requirements",
                           "output_format": profile["output_format"], "native_model_input": False,
                           "native_gaps": ["尚未通过对应版本原生文件样例和模型读取验收"]}
        preparation.setdefault("adapter", "standardized")
        preparation.setdefault("output_format", profile["output_format"])
        names = [item["role"] for item in preparation["inputs"]]
        if len(names) != len(set(names)):
            raise SchemaError("模型输入角色重复")
        for role in preparation["inputs"]:
            role.setdefault("record_only", role["role"] in preparation.get("record_only_roles", []))
            role.setdefault("depends_on", [])
            role.setdefault("allow_derived", False)
            role.setdefault("processing", {})
            role.setdefault("output_format", preparation["output_format"])
            role.setdefault("standard_name", role["role"])
            role.setdefault("aliases", [])
            for field in role["fields"]:
                field.setdefault("standard_name", field["name"])
                field.setdefault("aliases", [])
                field.setdefault("required", True)
                field.setdefault("allow_derived", role["allow_derived"] or field["name"] in preparation.get("derivable_fields", []))
            settings = role["processing"]
            if isinstance(settings, dict):
                fusion = settings.get("fusion")
                if fusion:
                    if fusion.get("method") not in {"weighted", "residual", "auto", "ordinary_kriging", "universal_kriging", "external_drift_kriging"} or len(fusion.get("sources", [])) < 2:
                        raise SchemaError("融合配置需引用至少两个输入角色，并使用已接入的 weighted/residual/auto/GSTools Kriging 方法")
                    if not role["allow_derived"]:
                        raise SchemaError("融合派生输入需显式声明allow_derived=true")
                    role["depends_on"] = list(dict.fromkeys([*role["depends_on"], *fusion["sources"]]))
                frequency = role.get("temporal_resolution")
                if frequency and "target_time_frequency" not in settings:
                    try:
                        pd.tseries.frequencies.to_offset(frequency)
                        settings["target_time_frequency"] = frequency
                    except (ValueError, TypeError):
                        # A phrase such as 'simulation-specific' is NOT a frequency.
                        pass
                if role.get("spatial_resolution_km") is not None:
                    settings.setdefault("target_spatial_resolution_km", role["spatial_resolution_km"])
            if set(role["depends_on"]) - set(names):
                raise SchemaError(f"{role['role']} 引用了未定义输入依赖")
        return cls(profile, preparation)

    @property
    def inputs(self):
        return self.preparation["inputs"]
