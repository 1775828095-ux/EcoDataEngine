"""Optional xclim daily statistics; no changes to the existing interval resampler."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.algorithms.open_source_stations import installed
from ecodata.core.exceptions import SchemaError


class XclimStatisticsPlugin:
    def __init__(self):
        self.spec = AlgorithmSpec(
            "temporal.xclim_statistics", "xclim daily semantic statistics", installed("xclim") or "missing",
            "temporal_statistics", "available" if installed("xclim") else "dependency_missing",
            ("time_series", "time_grid"), ("continuous", "accumulation", "flux"),
            license="Apache-2.0", backend="xclim", source_url="https://github.com/Ouranosinc/xclim",
            capabilities=("temporal_statistical_aggregation", "temporal_event_statistics"),
            preconditions=("explicit_units", "regular_daily", "declared_statistic", "declared_temporal_semantics"),
            temporal_requirements=("daily contiguous; NaNs require explicit handling before this adapter",),
            runtime_requirements=("xclim==0.59.1; isolated subprocess",),
            input_contract={"operations": ["mean", "sum", "min", "max", "duration", "frequency", "accumulation"]},
            output_contract={"kind": "time_statistics", "units": "upstream-derived"},
            verification_evidence=("tests/test_algorithm_library_v2.py",),
            official_entry_point="xclim.indices.generic.select_resample_op/threshold_count/spell_length_statistics",
            verification_status="VERIFIED" if installed("xclim") == "0.59.1" else "UNVERIFIED")

    def compatibility(self, context):
        reasons = []
        if self.spec.status != "available" or context.semantic_type not in self.spec.semantic_types:
            reasons.append("Unavailable backend or incompatible semantics")
        if not context.sources or context.variable not in context.sources[0].dataset:
            return Compatibility(False, ("INPUT_REQUIREMENT_MISSING: variable",))
        a = context.sources[0].dataset[context.variable]
        p = context.parameters
        if not a.attrs.get("units") or "time" not in a.coords:
            return Compatibility(False, ("INPUT_REQUIREMENT_MISSING: units and time",))
        try:
            times = pd.DatetimeIndex(a.time.values)
            if len(times) < 2 or not np.all(np.diff(times.values.astype("datetime64[ns]")) == np.timedelta64(1, "D")):
                reasons.append("Only contiguous daily input verified; use existing interval resampler first")
            if not p.get("target_frequency"):
                reasons.append("INPUT_REQUIREMENT_MISSING: target_frequency")
            else:
                offset = pd.tseries.frequencies.to_offset(p["target_frequency"])
                if (times[0] + offset) - times[0] < pd.Timedelta(days=1):
                    reasons.append("Statistics cannot refine daily inputs")
        except (ValueError, TypeError):
            reasons.append("Invalid time or target_frequency")
        if not np.isfinite(a.values).all():
            reasons.append("Missing/invalid samples: explicit missing-data decision required; no silent skip")
        operation = p.get("statistic")
        if operation not in self.spec.input_contract["operations"]:
            reasons.append("INPUT_REQUIREMENT_MISSING: supported statistic")
        meaning = p.get("temporal_semantics", a.attrs.get("temporal_semantics"))
        if meaning not in {"state", "interval_mean", "interval_total", "rate"}:
            reasons.append("INPUT_REQUIREMENT_MISSING: confirmed temporal_semantics")
        if operation == "sum" and meaning != "interval_total":
            reasons.append("sum requires interval totals; do not sum temperature or rates")
        if meaning in {"interval_total", "rate"} and a.attrs.get("units") in {"K", "degC", "degree_Celsius"}:
            reasons.append("Temperature is a state, not an accumulated total or rate")
        if operation == "accumulation" and meaning not in {"rate", "interval_total"}:
            reasons.append("accumulation requires a rate or interval totals")
        if operation == "mean" and meaning == "interval_total":
            reasons.append("Totals need explicit interval-total aggregation, not mean")
        if operation in {"duration", "frequency"} and not p.get("threshold"):
            reasons.append("INPUT_REQUIREMENT_MISSING: threshold with units")
        derived_unit = {"duration": "d", "frequency": "1"}.get(operation)
        if derived_unit and context.model_constraints.get("output_unit") != derived_unit:
            reasons.append("INPUT_REQUIREMENT_MISSING: explicit derived output_unit=" + derived_unit)
        if operation == "accumulation" and meaning == "rate" and not context.model_constraints.get("output_unit"):
            reasons.append("Rate integration requires an explicit output_unit contract")
        if p.get("comparison", ">=") not in {">", ">=", "<", "<="}:
            reasons.append("Unsupported threshold comparison")
        return Compatibility(not reasons, tuple(reasons), ("Incomplete edge periods are reported, not assumed complete",))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        a = context.sources[0].dataset[context.variable]
        p = context.parameters
        options = {k: p[k] for k in ("statistic", "target_frequency", "threshold", "comparison") if k in p}
        options["temporal_semantics"] = p.get("temporal_semantics", a.attrs.get("temporal_semantics"))
        with tempfile.TemporaryDirectory(prefix="ecodata_xclim_") as directory:
            folder = Path(directory)
            a.to_netcdf(folder / "input.nc", engine="h5netcdf")
            (folder / "options.json").write_text(json.dumps(options), encoding="utf-8")
            runner = Path(__file__).resolve().parents[3] / "tools/run_xclim_statistics.py"
            proc = subprocess.run([sys.executable, str(runner), str(folder)], capture_output=True, text=True, timeout=120)
            if proc.returncode:
                raise SchemaError("Official xclim statistics failed: " + proc.stderr[-2000:])
            with xr.open_dataset(folder / "output.nc", engine="h5netcdf") as ds:
                output = ds.load()
        output = output.rename({"result": context.variable})
        if not np.isfinite(output[context.variable].values).all():
            raise SchemaError("xclim produced nonfinite statistics")
        return finish(context, self.spec, output, {"backend": "xclim", **options,
            "input_samples": a.time.size, "accuracy_verified": False,
            "missing_policy": "reject", "output_units": output[context.variable].attrs["units"]}, started)
