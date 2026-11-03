"""Official mpspline2 via independent R; never manufactures missing horizons."""
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.algorithms.expansion_v5 import ROOT, verified
from ecodata.core.exceptions import SchemaError

RSCRIPT = ROOT / "external/R-4.5.2/bin/Rscript.exe"


class SoilSplinePlugin:
    def __init__(self):
        self.spec = AlgorithmSpec("soil.mpspline2", "mpspline2 soil depth spline", "0.1.9", "soil_profile",
            "available" if RSCRIPT.exists() and (ROOT/"external/R-library/mpspline2/DESCRIPTION").exists() else "dependency_missing",
            ("soil_profile",), ("continuous",), license="GPL (upstream DESCRIPTION)", backend="mpspline2::mpspline",
            source_url="https://github.com/obrl-soil/mpspline2", capabilities=("soil_profile_processing",),
            input_contract={"depth_units": "cm", "single_profile": True, "missing_horizons_require_confirmation": True},
            output_contract={"kind": "soil_depth_intervals"},
            verification_status="VERIFIED" if verified("soil.mpspline2") else "PARTIAL",
            official_entry_point="mpspline2::mpspline", verification_evidence=("output/algorithm_library_expansion/v5/execution_manifest.json",))

    def compatibility(self, context):
        if self.spec.status != "available" or not context.sources or context.variable not in context.sources[0].dataset:
            return Compatibility(False, ("R/mpspline2 and soil variable required",))
        ds, p = context.sources[0].dataset, context.parameters
        a = ds[context.variable]
        if context.semantic_type != "continuous" or a.dims != ("horizon",) or not all(k in ds for k in ("upper", "lower")):
            return Compatibility(False, ("Single soil profile with horizon upper/lower required",))
        reasons = []
        if a.dtype.kind not in "fiu" or not a.attrs.get("units") or ds.upper.attrs.get("units") != "cm" or ds.lower.attrs.get("units") != "cm":
            return Compatibility(False, ("Numeric values, explicit property units and cm depths required",))
        u, l = ds.upper.values, ds.lower.values
        if u.shape != a.shape or l.shape != a.shape or not np.isfinite(u).all() or not np.isfinite(l).all() or (u < 0).any() or (l <= u).any() or np.any(u[1:] < l[:-1]):
            reasons.append("Depths must be ordered, positive thickness and non-overlapping; no automatic repair")
        valid = np.isfinite(a.values)
        if np.isinf(a.values).any() or not valid.any():
            reasons.append("No valid soil observations or infinite values")
        if not valid.all() and not p.get("allow_missing_horizons"):
            reasons.append("Missing horizons need explicit user confirmation; original records retained")
        d = np.asarray(p.get("depth_boundaries_cm", []))
        if d.ndim != 1 or len(d) < 2 or d.dtype.kind not in "fiu" or not np.isfinite(d).all() or not np.equal(d, np.floor(d)).all() or np.any(np.diff(d) <= 0):
            reasons.append("Increasing integer target boundaries in cm required")
        elif valid.any() and (d[0] < u[valid].min() or d[-1] > l[valid].max()):
            reasons.append("Target depths outside observed support; extrapolation blocked")
        bounds = p.get("physical_range")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            reasons.append("Explicit physical_range required")
        elif valid.any() and ((a.values[valid] < bounds[0]).any() or (a.values[valid] > bounds[1]).any()):
            reasons.append("Source values outside confirmed range")
        if p.get("conserve_depth_integral") and a.attrs.get("units") == "pH":
            reasons.append("pH is logarithmic; do not impose a mass integral on pH")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        c = self.compatibility(context)
        if not c.eligible:
            raise SchemaError("; ".join(c.reasons))
        start = perf_counter()
        ds, p = context.sources[0].dataset, context.parameters
        a = ds[context.variable]
        with tempfile.TemporaryDirectory(prefix="soil_v5_") as temporary:
            path = Path(temporary)
            pd.DataFrame({"id": "profile", "upper": ds.upper.values, "lower": ds.lower.values, "value": a.values}).to_csv(path/"input.csv", index=False, na_rep="NA")
            process = subprocess.run([str(RSCRIPT), str(ROOT/"tools/run_mpspline_v5.R"), str(ROOT/"external/R-library"), str(path/"input.csv"), str(path/"out.csv"),
                ",".join(str(int(v)) for v in p["depth_boundaries_cm"]), str(p.get("lambda", .1)), *map(str, p["physical_range"])],
                capture_output=True, check=True, timeout=60)
            frame = pd.read_csv(path/"out.csv")
            one_cm = pd.read_csv(path/"out.csv.one_cm.csv").value.values
            report = {"backend": "mpspline2::mpspline", "dropped_missing_horizons": int(a.isnull().sum()),
                "original_data_modified": False, "missing_policy_confirmed": bool(p.get("allow_missing_horizons")),
                "depth_integral_applicable": bool(p.get("conserve_depth_integral")),
                "r_messages": process.stderr.decode(errors="replace")}
            if p.get("conserve_depth_integral"):
                if not np.isfinite(a).all() or p["depth_boundaries_cm"][0] != 0 or p["depth_boundaries_cm"][-1] != float(ds.lower[-1]) or np.any(ds.upper.values[1:] != ds.lower.values[:-1]):
                    raise SchemaError("Integral check requires complete contiguous full-depth support")
                original = float(np.sum(a.values*(ds.lower-ds.upper).values))
                error = float(abs(one_cm.sum()-original)/max(abs(original), 1e-12))
                report.update(depth_integral_input=original, depth_integral_output=float(one_cm.sum()), conservation_relative_error=error)
                if error > p.get("conservation_tolerance", .02):
                    raise SchemaError("Official spline exceeds specified depth-integral tolerance; result blocked")
        output = xr.Dataset({context.variable: ("depth", frame.value), "upper": ("depth", frame.upper_cm), "lower": ("depth", frame.lower_cm)}, coords={"depth": (frame.upper_cm+frame.lower_cm)/2})
        for key in ("depth", "upper", "lower"):
            output[key].attrs["units"] = "cm"
        output[context.variable].attrs.update(a.attrs)
        if not np.isfinite(frame.value).all() or (frame.value < p["physical_range"][0]).any() or (frame.value > p["physical_range"][1]).any():
            raise SchemaError("Invalid soil spline output")
        return finish(context, self.spec, output, report, start)
