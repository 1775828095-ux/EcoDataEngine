from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import pandas as pd
import xarray as xr

from ecodata.core.schemas import EcoDataBundle


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    version: str
    category: str
    status: str
    input_kinds: tuple[str, ...]
    semantic_types: tuple[str, ...]
    scale_range: tuple[float, float] = (1.0, float("inf"))
    requires_training: bool = False
    requires_weights: bool = False
    requires_observations: bool = False
    requires_auxiliary: bool = False
    physical_constraints: tuple[str, ...] = ()
    compute: str = "low"
    license: str = "project"
    backend: str = "EcoData Engine"
    source_url: str | None = None
    notes: str = ""
    capabilities: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    input_contract: dict[str, Any] = field(default_factory=dict)
    output_contract: dict[str, Any] = field(default_factory=dict)
    verification_evidence: tuple[str, ...] = ()
    auxiliary_inputs: tuple[str, ...] = ()
    temporal_requirements: tuple[str, ...] = ()
    spatial_requirements: tuple[str, ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    verification_status: str = "UNVERIFIED"
    official_entry_point: str = ""

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["scale_range"] = list(self.scale_range)
        return value


@dataclass(frozen=True)
class AlgorithmContext:
    sources: tuple[EcoDataBundle, ...]
    variable: str
    semantic_type: str
    target_lat: tuple[float, ...] | None = None
    target_lon: tuple[float, ...] | None = None
    observations: pd.DataFrame | None = None
    auxiliaries: tuple[xr.Dataset, ...] = ()
    validation_reference: xr.DataArray | None = None
    validation_role: str | None = None
    model_constraints: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    input_quality: dict[str, Any] | None = None


@dataclass(frozen=True)
class Compatibility:
    eligible: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlgorithmResult:
    data: EcoDataBundle
    algorithm_id: str
    report: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


class AlgorithmPlugin(Protocol):
    spec: AlgorithmSpec

    def compatibility(self, context: AlgorithmContext) -> Compatibility: ...

    def run(self, context: AlgorithmContext) -> AlgorithmResult: ...

