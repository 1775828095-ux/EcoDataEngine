from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import xarray as xr


@dataclass(frozen=True)
class TransformRecord:
    method: str
    parameters: dict[str, Any]
    input_rows: int
    output_rows: int
    elapsed_seconds: float | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "parameters": self.parameters,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "elapsed_seconds": self.elapsed_seconds,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EcoDataBundle:
    dataset: xr.Dataset
    history: tuple[TransformRecord, ...] = ()
    validation: ValidationReport | None = None


@dataclass(frozen=True)
class TransformResult:
    data: EcoDataBundle
    record: TransformRecord
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.details.get("status", "PASS" if self.passed else "FAIL")

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks, **self.details, "status": self.status}


@dataclass
class InformationLossRecord:
    lost_fields: list[str] = field(default_factory=list)
    lost_metadata: list[str] = field(default_factory=list)
    changed_units: dict[str, Any] = field(default_factory=dict)
    changed_resolution: dict[str, Any] = field(default_factory=dict)
    changed_time_semantics: dict[str, Any] = field(default_factory=dict)
    changed_crs: dict[str, Any] = field(default_factory=dict)
    interpolation_applied: bool = False
    extrapolation_applied: bool = False
    clipped_values: int = 0
    filled_values: int = 0
    dropped_records: int = 0
    peak_loss: dict[str, Any] = field(default_factory=dict)
    conservation_error: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Transform(Protocol):
    """Stable extension point for temporal, spatial, fusion and optimization work."""

    name: str

    def validate_input(self, data: EcoDataBundle) -> None: ...

    def apply(self, data: EcoDataBundle) -> TransformResult: ...


class SpatialTransform(Transform, Protocol):
    """P2 extension contract; no spatial algorithm is implemented in P1."""


class FusionTransform(Transform, Protocol):
    """P2 extension contract; no fusion algorithm is implemented in P1."""


class DownscalingTransform(SpatialTransform, Protocol):
    """P3 extension contract for constrained and peak-aware downscaling."""
