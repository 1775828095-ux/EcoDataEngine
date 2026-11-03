"""Verified ForDs pressure component, not the complete ForDs/CoLM workflow."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.algorithms.open_source_stations import installed
from ecodata.core.exceptions import SchemaError

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "external/fords"
COMMIT = "c533dfa4aa65089da2dcc0035aeda536094bfe53"
FIELDS = {"elevation": "m", "elevation_reference": "m", "temperature": "K", "temperature_reference": "K"}


def source_verified():
    manifest = UPSTREAM / "source_evidence.json"
    if not manifest.exists():
        return False
    evidence = json.loads(manifest.read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in evidence["files"]}
    return evidence["commit"] == COMMIT and all(
        (UPSTREAM / path).is_file() and hashlib.sha256((UPSTREAM / path).read_bytes()).hexdigest() == entries[path]["sha256"]
        for path in ("offline/topo_factory.py", "offline/utils.py", "LICENSE"))


class ForDsPressurePlugin:
    def __init__(self):
        self.spec = AlgorithmSpec(
            "terrain.fords_pressure", "ForDs topographic pressure correction", COMMIT, "terrain_downscaling",
            "available" if source_verified() and installed("scikit-learn") and installed("netCDF4") else "dependency_missing", ("rectilinear_grid", "time_grid"), ("continuous",),
            requires_auxiliary=True, license="MIT", backend="ForDs official downscale_air_pressure",
            source_url="https://github.com/leelew/ForDs", capabilities=("terrain_pressure_adjustment",),
            preconditions=("aligned_pressure_and_temperature", "coarse_reference_elevation", "fine_elevation", "explicit_SI_units"),
            auxiliary_inputs=tuple(FIELDS), spatial_requirements=("already aligned geographic grid; no extrapolation",),
            runtime_requirements=("pinned unmodified ForDs source", "numpy", "scikit-learn", "netCDF4"),
            input_contract={"variable_standard_name": "surface_air_pressure", "units": "Pa", "auxiliary_units": FIELDS},
            output_contract={"kind": "terrain_corrected_pressure", "units": "Pa", "complete_ForDs": False},
            verification_evidence=("tests/test_algorithm_library_v2.py",),
            official_entry_point="offline/topo_factory.py: downscale_air_pressure",
            verification_status="VERIFIED" if source_verified() and installed("scikit-learn") and installed("netCDF4") else "BLOCKED", notes="Only the official pressure component is integrated; other weather variables and CoLM are not verified.")

    def compatibility(self, context):
        reasons = []
        if self.spec.status != "available":
            reasons.append("ForDs optional dependencies unavailable")
        if not source_verified():
            reasons.append("Pinned official ForDs source missing or hash mismatch")
        if context.semantic_type != "continuous" or not context.sources or not context.auxiliaries:
            return Compatibility(False, tuple(reasons + ["INPUT_REQUIREMENT_MISSING: continuous pressure and aligned auxiliary dataset"]))
        ds, aux = context.sources[0].dataset, context.auxiliaries[0]
        if context.variable not in ds:
            return Compatibility(False, ("Pressure variable missing",))
        a = ds[context.variable]
        if a.attrs.get("standard_name") != "surface_air_pressure" or a.attrs.get("units") != "Pa":
            reasons.append("Requires confirmed surface_air_pressure in Pa; use UnitEngine first")
        if ds.attrs.get("crs") != aux.attrs.get("crs") or not ds.attrs.get("crs"):
            reasons.append("Explicit identical CRS required")
        if not {"lat", "lon"}.issubset(a.dims) or any(ds[c].ndim != 1 for c in ("lat", "lon") if c in ds):
            reasons.append("Requires rectilinear lat/lon coordinates")
        if not np.isfinite(a.values).all() or (a <= 0).any():
            reasons.append("Pressure must be finite and positive")
        for name, unit in FIELDS.items():
            if name not in aux or aux[name].attrs.get("units") != unit:
                reasons.append("INPUT_REQUIREMENT_MISSING: " + name + " in " + unit)
                continue
            try:
                xr.align(a, aux[name], join="exact")
                if not set(aux[name].dims).issubset(a.dims):
                    reasons.append("Auxiliary dimensions differ: " + name)
            except ValueError:
                reasons.append("Auxiliary coordinates differ: " + name)
            if not np.isfinite(aux[name].values).all():
                reasons.append("Nonfinite auxiliary: " + name)
            if name.startswith("temperature") and (aux[name] <= 0).any():
                reasons.append("Kelvin temperature must be positive")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        ds, aux = context.sources[0].dataset, context.auxiliaries[0]
        a = ds[context.variable]
        with tempfile.TemporaryDirectory(prefix="ecodata_fords_") as directory:
            folder = Path(directory)
            np.savez(folder / "input.npz", pressure=a.values, **{k: xr.broadcast(aux[k], a)[0].transpose(*a.dims).values for k in FIELDS})
            proc = subprocess.run([sys.executable, str(ROOT / "tools/run_fords_pressure.py"), str(folder)],
                                  capture_output=True, text=True, timeout=90)
            if proc.returncode:
                raise SchemaError("Official ForDs pressure failed: " + proc.stderr[-2000:])
            values = np.load(folder / "output.npy", allow_pickle=False)
        if values.shape != a.shape or not np.isfinite(values).all() or (values <= 0).any():
            raise SchemaError("ForDs output invalid: shape, finite or positive-pressure check failed")
        output = ds.copy()
        output[context.variable] = a.copy(data=values)
        return finish(context, self.spec, output, {"backend": self.spec.backend, "commit": COMMIT,
            "official_function": "offline.topo_factory.downscale_air_pressure", "complete_ForDs": False,
            "precision_improvement_verified": False, "terrain_used": True, "interpolation_performed": False}, started)
