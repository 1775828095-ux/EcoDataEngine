from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from ecodata.core.schemas import EcoDataBundle, ValidationReport
from ecodata.evaluation.metrics import (
    fidelity_metrics,
    metadata_completeness,
    relative_conservation_error,
)
from ecodata.io.readers import REQUIRED_COLUMNS
from ecodata.evaluation.self_check import PostConversionValidator, ValidationThresholds


CRITICAL_ATTRS = (
    "dataset_id",
    "source_name",
    "source_format",
    "crs",
    "time_zone",
    "source_time_zone",
    "temporal_semantics",
    "data_kind",
    "Conventions",
    "processing_stage",
    "time_interpolation_method",
    "time_frequency",
)


class P1Validator:
    """Checks persisted output independently from the transform implementation."""

    def validate(
        self,
        before: EcoDataBundle,
        after: EcoDataBundle,
        output_path: Path,
    ) -> ValidationReport:
        source = before.dataset
        restored = after.dataset
        checks = {
            "output_exists": output_path.is_file() and output_path.stat().st_size > 0,
            "required_variables": set(REQUIRED_COLUMNS).issubset(restored.data_vars),
            "row_count": source.sizes.get("observation") == restored.sizes.get("observation"),
            "values_equal": np.allclose(source["value"].values, restored["value"].values, equal_nan=True),
            "times_equal": np.array_equal(source["time"].values, restored["time"].values),
            "units_equal": np.array_equal(source["unit"].values, restored["unit"].values),
            "qc_flags_equal": np.array_equal(source["qc_flag"].values, restored["qc_flag"].values),
            "metadata_equal": all(source.attrs.get(key) == restored.attrs.get(key) for key in CRITICAL_ATTRS),
            "variable_metadata_equal": source["value"].attrs == restored["value"].attrs,
            "history_present": self._valid_history(restored.attrs.get("transform_history")),
            "time_monotonic_by_group": self._time_monotonic(restored),
        }
        return ValidationReport(
            passed=all(checks.values()),
            checks=checks,
            details={"output": str(output_path), "rows": restored.sizes.get("observation", 0)},
        )

    @staticmethod
    def _valid_history(raw: object) -> bool:
        try:
            history = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            return False
        return bool(history) and history[-1].get("method") == "temporal.linear"

    @staticmethod
    def _time_monotonic(dataset) -> bool:
        station = dataset["station_id"].values.astype(str)
        variable = dataset["variable"].values.astype(str)
        time = dataset["time"].values
        for key in sorted(set(zip(station, variable))):
            mask = (station == key[0]) & (variable == key[1])
            values = time[mask].astype("datetime64[ns]").astype("int64")
            if len(values) > 1 and not np.all(np.diff(values) > 0):
                return False
        return True


class P2Validator:
    """Independent numerical, conservation, metadata and timing checks."""

    def validate(
        self,
        data: EcoDataBundle,
        expected: xr.DataArray,
        *,
        variable: str = "value",
        conservation_source: xr.DataArray | None = None,
        max_rmse: float = 1e-9,
        max_conservation_error: float = 1e-9,
    ) -> ValidationReport:
        metrics = fidelity_metrics(data.dataset[variable], expected)
        metadata_ok, missing = metadata_completeness(data.dataset, variable)
        conservation_error = None
        if conservation_source is not None:
            conservation_error = relative_conservation_error(conservation_source, data.dataset[variable])
        elapsed = [record.elapsed_seconds for record in data.history]
        timing_ok = bool(elapsed) and all(value is not None and value >= 0 for value in elapsed)
        checks = {
            "accuracy": metrics.rmse <= max_rmse,
            "metadata_complete": metadata_ok,
            "timing_recorded": timing_ok,
            "conservation": conservation_error is None or conservation_error <= max_conservation_error,
        }
        return ValidationReport(
            passed=all(checks.values()),
            checks=checks,
            details={
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "max_abs_error": metrics.max_abs_error,
                "relative_conservation_error": conservation_error,
                "missing_metadata": list(missing),
                "elapsed_seconds": elapsed,
            },
        )
