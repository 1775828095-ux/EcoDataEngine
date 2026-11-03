"""Unmodified official DualTimesField interpolation behind the existing contracts."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.core.exceptions import SchemaError

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / ".venv-capability-v3/Scripts/python.exe"
SOURCE = ROOT / "external/dualtimesfield"
COMMIT = "6a2263fd6b7cee4f0a662406df0eb850543fe086"


def source_ready():
    manifest = SOURCE / "source_evidence.json"
    if not manifest.exists() or not RUNTIME.exists():
        return False
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entries = {r["path"]: r for r in data["files"]}
    return data["commit"] == COMMIT and all((SOURCE / p).exists() and hashlib.sha256((SOURCE / p).read_bytes()).hexdigest() == entries[p]["sha256"]
        for p in ("README.md", "interpolation/models.py"))


def execution_verified():
    folder = ROOT / "output/algorithm_library_expansion/v3"
    path = folder / "dualtimesfield_execution.json"
    if not path.exists():
        return False
    r = json.loads(path.read_text(encoding="utf-8"))
    output = folder / r["output"]
    return r.get("status") == "VERIFIED" and r.get("commit") == COMMIT and output.is_file() and r.get("sha256") == hashlib.sha256(output.read_bytes()).hexdigest()


class DualTimesFieldPlugin:
    def __init__(self):
        self.spec = AlgorithmSpec("temporal.dualtimesfield", "DualTimesField irregular interpolation", COMMIT,
            "irregular_time", "available" if source_ready() else "dependency_missing", ("time_series",), ("continuous",),
            requires_training=True, compute="medium", license="MIT declared in official README", backend="official DualTimesFieldInterpolator",
            source_url="https://github.com/WisdomTogether/DualTimesField",
            capabilities=("irregular_time_interpolation", "continuous_time_representation", "irregular_sampling"),
            preconditions=("explicit_units", "unique_increasing_times", "at_least_four_finite_observations", "bounded_query_times"),
            input_contract={"shape": "time only", "mask": "NaN denotes unobserved; no held-out values used for fitting", "max_samples": 512},
            output_contract={"kind": "interpolated_time_series", "extrapolation": False, "observed_values": "retained"},
            temporal_requirements=("interpolation within input time support; not forecasting",),
            runtime_requirements=(".venv-capability-v3; shared read-only Torch binaries, isolated extra packages",),
            official_entry_point="interpolation/models.py:DualTimesFieldInterpolator.fit/predict",
            verification_status="VERIFIED" if source_ready() and execution_verified() else "PARTIAL", verification_evidence=("tests/test_algorithm_library_v3.py",),
            notes="Per-series fitting; official README declares MIT, no standalone LICENSE at pinned commit. Smoke execution is not paper-level accuracy validation.")

    def compatibility(self, context):
        reasons = []
        if not source_ready():
            reasons.append("Pinned official code or isolated runtime unavailable")
        if context.semantic_type != "continuous" or not context.sources or context.variable not in context.sources[0].dataset:
            return Compatibility(False, tuple(reasons + ["Continuous time-series variable required"]))
        a = context.sources[0].dataset[context.variable]
        if a.dims != ("time",) or "time" not in a.coords or not a.attrs.get("units"):
            return Compatibility(False, tuple(reasons + ["One-dimensional time and explicit units required"]))
        t = a.time.values
        if t.dtype.kind not in "Mifu" or (np.isnat(t).any() if t.dtype.kind == "M" else not np.isfinite(t).all()):
            return Compatibility(False, ("Finite numeric or datetime timestamps required",))
        if len(t) < 4 or len(t) > 512 or np.any(np.diff(t) <= (np.timedelta64(0, "ns") if t.dtype.kind == "M" else 0)):
            reasons.append("Require 4..512 unique, strictly increasing timestamps")
        if np.isinf(a.values).any() or np.isfinite(a.values).sum() < 4:
            reasons.append("At least four finite observations; Inf forbidden, NaN is an explicit missing mask")
        support = t[np.isfinite(a.values)]
        queries = np.asarray(context.parameters.get("target_times", t))
        if queries.ndim != 1 or not queries.size or queries.size > 1024:
            reasons.append("Require 1..1024 query times")
        else:
            try:
                if not support.size or queries.min() < support.min() or queries.max() > support.max() or np.any(queries[1:] <= queries[:-1]):
                    reasons.append("Queries must be increasing and inside input time support; no forecasting")
            except (TypeError, ValueError):
                reasons.append("Query/source timestamp types differ")
        epochs = context.parameters.get("epochs", 10)
        if not isinstance(epochs, int) or not 1 <= epochs <= 100:
            reasons.append("Smoke adapter permits 1..100 epochs; not unbounded training")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        a = context.sources[0].dataset[context.variable]
        times = a.time.values
        queries = np.asarray(context.parameters.get("target_times", times), dtype=times.dtype)
        numeric = times.astype("datetime64[ns]").astype(np.int64).astype(float) if times.dtype.kind == "M" else times.astype(float)
        qnumeric = queries.astype("datetime64[ns]").astype(np.int64).astype(float) if times.dtype.kind == "M" else queries.astype(float)
        origin = numeric.min()
        scale = numeric.max() - origin
        with tempfile.TemporaryDirectory(prefix="ecodata_dualtime_") as directory:
            folder = Path(directory)
            np.savez(folder / "input.npz", times=(numeric-origin)/scale, queries=(qnumeric-origin)/scale, values=a.values)
            options = {"epochs": context.parameters.get("epochs", 10), "seed": context.parameters.get("seed", 42)}
            (folder / "options.json").write_text(json.dumps(options), encoding="utf-8")
            proc = subprocess.run([str(RUNTIME), str(ROOT / "tools/run_dualtimesfield.py"), str(folder)],
                capture_output=True, text=True, timeout=180)
            if proc.returncode:
                raise SchemaError("Official DualTimesField failed: " + proc.stderr[-2500:])
            values = np.load(folder / "prediction.npy", allow_pickle=False)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if values.shape != queries.shape or not np.isfinite(values).all():
            raise SchemaError("DualTimesField output shape/finite check failed")
        # Filling is explicit: retain actual observations, only estimate absent/query values.
        observed = np.zeros(queries.size, dtype=bool)
        for i, time in enumerate(queries):
            match = np.flatnonzero(times == time)
            if match.size and np.isfinite(a.values[match[0]]):
                values[i] = a.values[match[0]]
                observed[i] = True
        output = xr.Dataset({context.variable: ("time", values), "observed_mask": ("time", observed.astype(np.int8))},
                            coords={"time": queries}, attrs=dict(context.sources[0].dataset.attrs))
        output[context.variable].attrs = dict(a.attrs)
        output.time.attrs = dict(a.time.attrs)
        output.observed_mask.attrs.update(units="1", long_name="retained actual observations; zero means model estimate")
        report.update(commit=COMMIT, filled_or_queried_count=int((~observed).sum()), original_observations_retained=True,
                      accuracy_proven=False, extrapolation=False)
        return finish(context, self.spec, output, report, started)
