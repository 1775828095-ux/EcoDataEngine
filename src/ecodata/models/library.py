from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecodata.models.schema import normalize_profile


@dataclass(frozen=True)
class ModelMatchReport:
    model_id: str
    model_name: str
    adapter_status: str
    matched_variables: tuple[str, ...]
    missing_required_variables: tuple[str, ...]
    accepted_input_formats: tuple[str, ...]
    uploaded_formats: tuple[str, ...]
    format_compatible: bool
    can_generate_model_input: bool
    messages: tuple[str, ...]
    assigned_inputs: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "adapter_status": self.adapter_status,
            "matched_variables": list(self.matched_variables),
            "missing_required_variables": list(self.missing_required_variables),
            "accepted_input_formats": list(self.accepted_input_formats),
            "uploaded_formats": list(self.uploaded_formats),
            "format_compatible": self.format_compatible,
            "can_generate_model_input": self.can_generate_model_input,
            "messages": list(self.messages),
            "assigned_inputs": {key: list(value) for key, value in self.assigned_inputs.items()},
        }


class ModelLibrary:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or Path(__file__).with_name("profiles")

    def list(self) -> list[dict[str, Any]]:
        return [self._load(path) for path in sorted(self.directory.glob("*.json"))]

    def get(self, model_id: str) -> dict[str, Any]:
        path = self.directory / f"{model_id}.json"
        if not path.exists():
            raise KeyError(f"Unknown model profile: {model_id}")
        return self._load(path)

    def match(
        self,
        model_id: str,
        variables: set[str],
        formats: set[str],
        assignments: dict[str, list[str]] | None = None,
    ) -> ModelMatchReport:
        profile = self.get(model_id)
        normalized = {item.strip().lower() for item in variables}
        required = profile["inputs"]
        assigned = {key: tuple(value) for key, value in (assignments or {}).items() if value}
        matched, missing = [], []
        for requirement in required:
            name = requirement["name"]
            candidates = {name.lower(), requirement["standard_name"].lower(), *(alias.lower() for alias in requirement["aliases"])}
            (matched if name in assigned or candidates & normalized else missing).append(name)
        accepted = tuple(sorted({item for requirement in required for item in requirement["formats"]}))
        uploaded = tuple(sorted(formats))
        compatible = bool({item.lower() for item in accepted} & {item.lower() for item in formats})
        status = profile["status"]
        messages = []
        if missing:
            messages.append("缺少 required inputs，软件不会凭空生成。")
        if not compatible:
            messages.append("上传格式与模型原生输入格式不匹配；是否可转换需由已验证适配器决定。")
        if status != "verified_native_adapter":
            messages.append("当前仅提供需求检查，尚无对应版本真实样例验证的输出适配器。")
        can_generate = not missing and compatible and status == "verified_native_adapter"
        return ModelMatchReport(
            profile["id"], profile["name"], status, tuple(matched), tuple(missing),
            accepted, uploaded, compatible, can_generate, tuple(messages), assigned,
        )

    def suggest_role(self, model_id: str, filename: str, variables: set[str]) -> str | None:
        """Return only an exact token/alias suggestion; ambiguous files remain unassigned."""
        profile = self.get(model_id)
        searchable = " ".join({filename, *variables}).lower().replace("-", "_")
        matches = []
        for requirement in profile["inputs"]:
            role = requirement["name"]
            tokens = [role, requirement["standard_name"], *requirement["aliases"]]
            if any(token.lower().replace("-", "_") in searchable for token in tokens):
                matches.append(role)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
        required = {"id", "name", "category", "reference_version", "official_sources", "input_formats", "required_variables", "adapter_status"}
        missing = required - set(profile)
        if missing:
            raise ValueError(f"Profile {path.name} missing keys: {sorted(missing)}")
        normalized = normalize_profile(profile)
        from ecodata.models.preparation_catalog import apply_preparation_catalog
        return apply_preparation_catalog(normalized)
