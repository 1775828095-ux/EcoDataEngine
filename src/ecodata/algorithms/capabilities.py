"""Capability contracts projected from the executable Algorithm Registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ecodata.algorithms.contracts import AlgorithmSpec
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.semantic_requirements import CapabilityRequirements


DEFAULT_CAPABILITIES = {
    "spatial.nearest": ("continuous_spatial_resampling", "categorical_scale_processing"),
    "spatial.bilinear": ("continuous_spatial_resampling",),
    "spatial.conservative": ("continuous_spatial_resampling", "spatial_conservation"),
    "temporal.semantic_resample": ("temporal_accumulation", "temporal_mean_resampling", "temporal_interpolation"),
    "fusion.residual": ("multi_source_fusion", "observation_constraint"),
    "fusion.uncertainty_weighted": ("multi_source_fusion", "uncertainty_weighting"),
    "fusion.quality_aware": ("multi_source_fusion", "quality_aware_fusion", "bias_correction"),
    "fusion.idw": ("observation_interpolation",),
    "downscale.qp": ("continuous_spatial_resampling", "nonnegative_validation", "coarse_scale_consistency"),
    "neural.fno": ("continuous_spatial_resampling", "observation_constraint"),
    "neural.multiscale_uno": ("continuous_spatial_resampling", "observation_constraint"),
    "neural.observation_guided_uno": ("continuous_spatial_resampling", "observation_constraint"),
    "scale.semantic_stack": ("multiscale_feature_preparation", "fractional_categorical_aggregation"),
    "features.galileo_nano": ("pretrained_feature_encoding",),
}


@dataclass(frozen=True)
class CapabilityRecord:
    algorithm_id: str
    capabilities: tuple[str, ...]
    preconditions: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    compute_cost: float
    executable: bool
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    evidence: tuple[str, ...]
    semantic_types: tuple[str, ...] = ()
    input_kinds: tuple[str, ...] = ()

    def as_dict(self):
        return asdict(self)


def describe_algorithm(spec: AlgorithmSpec) -> CapabilityRecord:
    cost = {"low": 1.0, "medium": 2.0, "high": 5.0, "gpu": 8.0}.get(spec.compute, 3.0)
    capabilities = spec.capabilities or DEFAULT_CAPABILITIES.get(spec.id, ())
    if spec.category in {"temporal_scale", "temporal_statistics", "irregular_time", "temporal_reconstruction"}:
        capabilities=tuple(dict.fromkeys((*capabilities,"temporal_processing")))
    return CapabilityRecord(spec.id, tuple(capabilities), spec.preconditions, spec.physical_constraints,
                            cost, spec.status == "available" and spec.category != "model_preprocessing", spec.input_contract, spec.output_contract,
                            spec.verification_evidence, spec.semantic_types, spec.input_kinds)


def capability_catalog(registry: AlgorithmRegistry) -> tuple[CapabilityRecord, ...]:
    return tuple(describe_algorithm(spec) for spec in registry.specs())


def coverage(requirements: CapabilityRequirements, selected: list[CapabilityRecord]) -> dict[str, Any]:
    provided = {capability for item in selected if item.executable for capability in item.capabilities}
    missing = sorted(set(requirements.required) - provided)
    return {"valid": not missing and all(item.executable for item in selected), "provided": sorted(provided),
            "missing": missing, "non_executable": [item.algorithm_id for item in selected if not item.executable]}
