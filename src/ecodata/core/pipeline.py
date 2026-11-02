from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ecodata.core.exceptions import ValidationError
from ecodata.core.schemas import EcoDataBundle, Transform, ValidationReport
from ecodata.evaluation.validation import P1Validator
from ecodata.evaluation.self_check import PostConversionValidator, profile_dataset, require_usable, validation_files
from ecodata.io.readers import CsvReadConfig, NetCdfReader, StationCsvReader
from ecodata.io.writers import NetCdfWriter
from ecodata.temporal.resampling import LinearTimeConfig, LinearTimeInterpolator
from ecodata.optimization.downscaling import ConstrainedDownscaler, DownscalingConfig, QPOptions, supported_regrid
from ecodata.spatial.regridding import GridRegridder, RegridConfig
import pandas as pd


@dataclass(frozen=True)
class PipelineResult:
    data: EcoDataBundle
    report: ValidationReport
    output_path: Path


class P1Pipeline:
    def run(
        self,
        source: Path,
        target: Path,
        read_config: CsvReadConfig,
        time_config: LinearTimeConfig,
    ) -> PipelineResult:
        raw = StationCsvReader().read(source, read_config)
        baseline = profile_dataset(raw.dataset)
        transformed = LinearTimeInterpolator(time_config).apply(raw).data
        audit = PostConversionValidator().validate(raw.dataset, transformed.dataset, history=transformed.history, before_profile=baseline, critical_fields=("value",))
        require_usable(audit)
        NetCdfWriter().write(transformed, target)
        restored = NetCdfReader().read(target)
        report = P1Validator().validate(transformed, restored, target)
        if not report.passed:
            failed = [name for name, passed in report.checks.items() if not passed]
            raise ValidationError(f"P1 output validation failed: {', '.join(failed)}")
        report.details["post_conversion"] = audit.as_dict()
        from ecodata.engines.validation import ValidationEngine
        audit.details["standards_compliance"] = ValidationEngine.compliance(target.read_bytes(),
            acdd="ACDD" in str(restored.dataset.attrs.get("Conventions", "")))
        for name, content in validation_files(audit).items():
            (target.parent / name).write_bytes(content)
        return PipelineResult(data=restored, report=report, output_path=target)


@dataclass(frozen=True)
class SequentialPipelineResult:
    original: EcoDataBundle
    data: EcoDataBundle
    methods: tuple[str, ...]
    validation_report: ValidationReport | None = None


class SequentialPipeline:
    """Run compatible transforms in order without mutating the original bundle."""

    def run(
        self,
        initial: EcoDataBundle,
        transforms: tuple[Transform, ...] | list[Transform],
    ) -> SequentialPipelineResult:
        original = EcoDataBundle(dataset=initial.dataset.copy(deep=True), history=initial.history)
        baseline = profile_dataset(original.dataset)
        if initial.validation:
            require_usable(initial.validation)
        current = initial
        methods: list[str] = []
        for transform in transforms:
            transform.validate_input(current)
            current = transform.apply(current).data
            methods.append(transform.name)
        critical = tuple(getattr(getattr(t, "config", None), "variable", "") for t in transforms)
        report = PostConversionValidator().validate(original.dataset, current.dataset, history=current.history[len(initial.history):], before_profile=baseline, critical_fields=critical)
        require_usable(report)
        return SequentialPipelineResult(original=original, data=replace(current, validation=report), methods=tuple(methods), validation_report=report)


class P3Pipeline:
    """Single entry point used by experiments and the Streamlit test interface."""

    def run(
        self,
        coarse: EcoDataBundle,
        method: str,
        target_lat: tuple[float, ...],
        target_lon: tuple[float, ...],
        training_observations: pd.DataFrame | None = None,
        *,
        variable: str = "value",
        optimization_options: QPOptions | None = None,
    ) -> EcoDataBundle:
        if method in {"nearest", "bilinear", "conservative"}:
            result = supported_regrid(coarse, method, target_lat, target_lon, variable)
            return self._validated(coarse, result, variable)
        if method not in {"constrained", "peak_aware", "adaptive_smoothing", "adaptive_peak"}:
            raise ValueError(f"Unknown P3 method: {method}")
        if training_observations is None:
            raise ValueError(f"Method '{method}' requires training observations")
        config = DownscalingConfig(
            target_lat=target_lat,
            target_lon=target_lon,
            adaptive=method == "peak_aware",
            adaptive_smoothing=method in {"adaptive_smoothing", "adaptive_peak"},
            trusted_peak_constraints=method == "adaptive_peak",
            variable=variable,
            **asdict(optimization_options or QPOptions()),
        )
        return self._validated(coarse, ConstrainedDownscaler(config).apply(coarse, training_observations).data, variable)

    @staticmethod
    def _validated(source, output, variable):
        report = PostConversionValidator().validate(source.dataset, output.dataset, kind="resampling", history=output.history[len(source.history):], critical_fields=(variable,))
        require_usable(report)
        return replace(output, validation=report)
