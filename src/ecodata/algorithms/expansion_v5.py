"""Contract-limited official V5 backends; model preprocessing is explicit opt-in."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.core.exceptions import SchemaError

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/algorithm_library_expansion/v5"
RUNTIME = ROOT / ".venv-capability-v5/Scripts/python.exe"


def target_grid(context):
    target = context.parameters.get("target_grid")
    if isinstance(target, xr.Dataset):
        return target
    if context.target_lat is not None and context.target_lon is not None and context.sources:
        return xr.Dataset(coords={"lat": list(context.target_lat), "lon": list(context.target_lon)},
            attrs={"crs": context.sources[0].dataset.attrs.get("crs")})
    return None


def verified(algorithm):
    manifest = OUT / "execution_manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    return any(r["algorithm"] == algorithm and r["status"] in {"PASS", "WARN"} and (OUT/r["file"]).is_file()
        and hashlib.sha256((OUT/r["file"]).read_bytes()).hexdigest() == r["sha256"] for r in rows)


class V5Plugin:
    def __init__(self, kind):
        self.kind = kind
        learn = kind == "liif"
        emission = kind.startswith("emiproc")
        repository = "yinboc/liif" if learn else ("C2SM-RCM/emiproc" if emission else "pytroll/pyresample")
        package = "emiproc" if emission else "pyresample"
        available = RUNTIME.exists() and ((OUT/"liif_training.json").exists() and (OUT/"liif_era5_local.pt").exists()
            if learn else (RUNTIME.parent.parent/"Lib/site-packages"/package/"__init__.py").exists())
        self.spec = AlgorithmSpec("v5."+kind, kind, "official pinned source" if learn else ("2.10.0" if emission else "1.35.0"),
            "model_preprocessing" if emission else "single_source_scale", "available" if available else "dependency_missing",
            ("time_grid",) if learn else (("emission_grid",) if emission else ("curvilinear_grid",)),
            ("cell_total",) if emission else ("continuous",), scale_range=(1.5, 4.) if learn else (1., float("inf")),
            requires_weights=learn, license="BSD-3-Clause" if learn or emission else "LGPL-3.0-or-later",
            backend=repository, source_url="https://github.com/"+repository,
            capabilities=("learned_arbitrary_environmental_reconstruction", "continuous_spatial_resampling") if learn else (("emission_spatial_allocation",) if kind == "emiproc_spatial" else
                (("emission_temporal_allocation",) if emission else ("multi_grid_regridding", "curvilinear_regridding"))),
            input_contract={"model_preprocessing_opt_in": emission, "trained_variable": "air_temperature K" if learn else None},
            output_contract={"kind": "grid", "arbitrary_variables": False},
            official_entry_point=kind, verification_status="VERIFIED" if verified("v5."+kind) else "PARTIAL",
            verification_evidence=("output/algorithm_library_expansion/v5/execution_manifest.json",))

    def compatibility(self, context):
        if not RUNTIME.exists() or not context.sources or context.variable not in context.sources[0].dataset:
            return Compatibility(False, ("V5 environment and target field required",))
        ds = context.sources[0].dataset
        a = ds[context.variable]
        target = target_grid(context)
        if not isinstance(target, xr.Dataset):
            return Compatibility(False, ("Explicit target_grid required",))
        reasons = []
        for axis, explicit in (("lat",context.target_lat),("lon",context.target_lon)):
            if explicit is not None and (axis not in target.coords or len(explicit)!=target[axis].size or not np.allclose(explicit,target[axis])):
                reasons.append("Explicit target coordinates conflict with target_grid")
        if context.semantic_type not in self.spec.semantic_types or a.dtype.kind not in "fiu" or not np.isfinite(a.values).all():
            return Compatibility(False, ("Finite numeric field with compatible physical semantics required",))
        if not ds.attrs.get("crs") or target.attrs.get("crs") != ds.attrs.get("crs") or not a.attrs.get("units"):
            reasons.append("Explicit matching CRS and units required")
        if self.kind == "liif":
            contract_path = OUT / "liif_training.json"
            if not contract_path.exists() or not (OUT/"liif_era5_local.pt").exists():
                return Compatibility(False, ("Locally trained LIIF checkpoint missing",))
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if not contract["optimizer_steps"] or hashlib.sha256((OUT/"liif_era5_local.pt").read_bytes()).hexdigest() != contract["sha256"]:
                reasons.append("Checkpoint integrity/training evidence failed")
            evidence = ROOT/"external/liif/source_evidence.json"
            entries = json.loads(evidence.read_text(encoding="utf-8")) if evidence.exists() else {}
            if entries.get("commit") != contract["commit"] or not all((ROOT/"external/liif"/e["path"]).is_file() and
                hashlib.sha256((ROOT/"external/liif"/e["path"]).read_bytes()).hexdigest() == e["sha256"] for e in entries.get("files", [])):
                reasons.append("Official LIIF source integrity failed")
            if a.dims != ("time", "lat", "lon") or a.shape[-2:] != (6, 6):
                return Compatibility(False, ("LIIF expects time x 6 x 6",))
            if a.attrs.get("standard_name") != "air_temperature" or a.attrs["units"] != "K" or ds.attrs.get("crs") != "EPSG:4326":
                reasons.append("Only trained regional ERA5 air_temperature K supported, not LAI/NPP")
            for axis, bounds in (("lat", contract["bounds"][:2]), ("lon", contract["bounds"][2:])):
                if axis not in target.coords or target[axis].ndim != 1 or not 9 <= target[axis].size <= 24:
                    reasons.append("Continuous-query sizes limited to verified 1.5x..4x range")
                    continue
                n = target[axis].size
                expected = np.linspace(bounds[0]+(bounds[1]-bounds[0])/n/2, bounds[1]-(bounds[1]-bounds[0])/n/2, n)
                if not np.allclose(target[axis], expected):
                    reasons.append("Query coordinates must be cell centers of trained footprint")
                if not np.allclose(ds[axis], np.linspace(bounds[0]+.5, bounds[1]-.5, 6)):
                    reasons.append("Input differs from trained regional grid")
        elif self.kind.startswith("pyresample"):
            if a.ndim not in (2, 3) or "lat" not in ds.coords or "lon" not in ds.coords or ds.lat.ndim != 2 or ds.lon.shape != ds.lat.shape or a.shape[-2:] != ds.lat.shape:
                reasons.append("Two-dimensional geographic source coordinates matching field required")
            elif not np.isfinite(ds.lat).all() or not np.isfinite(ds.lon).all() or float(abs(ds.lat).max()) > 90 or float(abs(ds.lon).max()) > 180:
                reasons.append("Invalid geographic source coordinates")
            if ds.attrs.get("crs") != "EPSG:4326" or context.parameters.get("radius_m", 0) <= 0:
                reasons.append("Geographic CRS and positive radius_m required")
        else:
            if not context.parameters.get("model_preprocessing") or a.attrs.get("spatial_semantics") != "cell_total" or a.attrs.get("units") != "kg":
                reasons.append("Explicit model preprocessing and cell-total emissions in kg required")
            if a.dims != ("y", "x"):
                reasons.append("Emission input requires 2D cell totals")
            if self.kind == "emiproc_spatial":
                from pyproj import CRS
                try:
                    projected = CRS.from_user_input(ds.attrs.get("crs", "EPSG:4326")).is_projected
                except Exception:
                    projected = False
                if not projected:
                    reasons.append("Explicit projected area grid required; degree-area weighting not accepted")
                for axis in ("y", "x"):
                    if axis not in ds.coords or ds[axis].ndim != 1 or ds[axis].size < 2 or not np.all(np.diff(ds[axis]) > 0) or not np.allclose(np.diff(ds[axis]), np.diff(ds[axis])[0]):
                        reasons.append("Regular increasing source emission-grid axes required")
            else:
                ratios = np.asarray(context.parameters.get("ratios", []))
                if ratios.shape != (24,) or ratios.dtype.kind not in "fiu" or not np.isfinite(ratios).all() or (ratios < 0).any() or not np.isclose(ratios.sum(), 1):
                    reasons.append("Explicit 24 hourly nonnegative daily fractions summing to one required")
                if "time" not in target.coords or target.time.size != 24 or context.parameters.get("time_basis") != "UTC" or not context.parameters.get("daily_total_date"):
                    reasons.append("24 UTC hours and source daily_total_date required")
                else:
                    import pandas as pd
                    day = pd.to_datetime(context.parameters["daily_total_date"], errors="coerce")
                    if pd.isna(day) or day.tzinfo is not None or target.time.values[0] != day.to_datetime64() or day.hour or day.minute or not np.all(np.diff(target.time.values) == np.timedelta64(1, "h")):
                        reasons.append("Hourly times must match declared daily total interval")
        if self.kind != "emiproc_temporal":
            for axis in (("y", "x") if self.kind == "emiproc_spatial" else ("lat", "lon")):
                if axis not in target.coords or target[axis].ndim != 1 or target[axis].size < 2 or not np.all(np.diff(target[axis]) > 0) or not np.allclose(np.diff(target[axis]), np.diff(target[axis])[0]):
                    reasons.append("Regular increasing target axes required")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        with tempfile.TemporaryDirectory(prefix="ecodata_v5_") as directory:
            folder = Path(directory)
            source, target, output, options = [folder/n for n in ("source.nc", "target.nc", "output.nc", "options.json")]
            context.sources[0].dataset.to_netcdf(source, engine="h5netcdf")
            target_grid(context).to_netcdf(target, engine="h5netcdf")
            data = {k: v for k, v in context.parameters.items() if k != "target_grid"}
            data["variable"] = context.variable
            options.write_text(json.dumps(data), encoding="utf-8")
            subprocess.run([str(RUNTIME), str(ROOT/"tools/run_capability_v5_backend.py"), self.kind, str(source), str(target), str(output), str(options)], check=True, capture_output=True, text=True, timeout=120)
            result = xr.load_dataset(output, engine="h5netcdf")
            report = json.loads(Path(str(output)+".json").read_text(encoding="utf-8"))
        if report.get("conservation_relative_error", 0) > 1e-8:
            raise SchemaError("Emission conservation check failed; output blocked")
        return finish(context, self.spec, result, report, started)


def v5_plugins():
    return tuple(V5Plugin(kind) for kind in ("liif", "pyresample_nearest", "pyresample_bilinear", "emiproc_spatial", "emiproc_temporal"))
