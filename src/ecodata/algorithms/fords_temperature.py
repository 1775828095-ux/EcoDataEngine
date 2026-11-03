"""Real terrain-dependent temperature correction, not complete ForDs/CoLM."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from time import perf_counter
import numpy as np
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.fords import source_verified, COMMIT
from ecodata.algorithms.open_source_spatial import finish, grid_target
from ecodata.algorithms.open_source_stations import installed
from ecodata.core.exceptions import SchemaError

ROOT = Path(__file__).resolve().parents[3]


def execution_verified():
    folder = ROOT / "output/algorithm_library_expansion/v4"
    manifest = folder / "execution_manifest.json"
    if not manifest.exists():
        return False
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    return any(r["algorithm"] == "terrain.fords_temperature" and r["path"] == "fords_temperature_pipeline_output.nc"
        and r["status"] in {"PASS", "WARN"} and (folder / r["path"]).is_file()
        and hashlib.sha256((folder / r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in rows)


class ForDsTemperaturePlugin:
    def __init__(self):
        available = source_verified() and installed("scikit-learn") and installed("netCDF4")
        self.spec = AlgorithmSpec("terrain.fords_temperature", "ForDs terrain-aware air temperature", COMMIT,
            "terrain_weather_downscaling", "available" if available else "dependency_missing", ("time_grid",), ("continuous",),
            requires_auxiliary=True, license="MIT", backend="official ForDs downscale_air_temperature / calc_lapse_rate",
            source_url="https://github.com/leelew/ForDs", capabilities=("terrain_auxiliary", "terrain_aware_weather_downscaling", "meteorological_downscaling", "continuous_spatial_resampling"),
            preconditions=("air_temperature K", "coarse elevation m", "fine elevation m", "same geographic CRS", "no extrapolation"),
            input_contract={"shape": "time x lat x lon", "minimum_coarse_shape": [3, 3], "target_grid": "DEM with explicit m units"},
            output_contract={"kind": "terrain_corrected_temperature", "units": "K", "complete_ForDs": False},
            auxiliary_inputs=("target_grid.elevation", "source.elevation_coarse"),
            official_entry_point="offline/topo_factory.py:downscale_air_temperature",
            verification_status="VERIFIED" if available and execution_verified() else "PARTIAL", verification_evidence=("tests/test_algorithm_library_v4.py",),
            notes="Official lapse-rate regression and correction retained. CDO file-I/O bilinear hook replaced with xarray interpolation on Windows. No complete ForDs radiation/precipitation/CoLM claim.")

    def compatibility(self, context):
        reasons = []
        if self.spec.status != "available":
            reasons.append("Official ForDs source/dependencies missing")
        if context.semantic_type != "continuous":
            reasons.append("Only continuous air temperature supported")
        try:
            source, target, axes = grid_target(context)
            a = source[context.variable]
            if axes != ("lat", "lon") or a.dims != ("time", "lat", "lon"):
                reasons.append("Explicit geographic time x lat x lon temperature required")
            if a.attrs.get("units") != "K" or a.attrs.get("standard_name") != "air_temperature":
                reasons.append("Confirmed air_temperature in K required")
            if a.dtype.kind not in "fiu" or not np.isfinite(a.values).all():
                reasons.append("Temperature contains invalid values")
            if "elevation_coarse" not in source or source.elevation_coarse.attrs.get("units") != "m":
                reasons.append("Coarse reference DEM in m required")
            elif source.elevation_coarse.dims != ("lat", "lon") or source.elevation_coarse.dtype.kind not in "fiu" or not np.isfinite(source.elevation_coarse.values).all():
                reasons.append("Coarse DEM dimensions/finite-value check failed")
            if "elevation" not in target or target.elevation.attrs.get("units") != "m":
                reasons.append("Fine target DEM in m required")
            elif target.elevation.dims != ("lat", "lon") or target.elevation.dtype.kind not in "fiu" or not np.isfinite(target.elevation.values).all():
                reasons.append("Fine DEM dimensions/finite-value check failed")
            if target.attrs.get("crs") != source.attrs.get("crs"):
                reasons.append("DEM/temperature CRS must match")
            for axis in ("lat", "lon"):
                if source[axis].size < 3 or target[axis].min() < source[axis].min() or target[axis].max() > source[axis].max():
                    reasons.append("Need >=3 coarse coordinates and target fully inside support")
        except (SchemaError, KeyError, ValueError, IndexError) as exc:
            reasons.append(str(exc))
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        source, target, _ = grid_target(context)
        with tempfile.TemporaryDirectory(prefix="ecodata_fords_temperature_") as directory:
            folder = Path(directory)
            np.savez(folder / "input.npz", temperature=source[context.variable].values, coarse_elevation=source.elevation_coarse.values,
                fine_elevation=target.elevation.values, lat=source.lat.values, lon=source.lon.values,
                target_lat=target.lat.values, target_lon=target.lon.values)
            proc = subprocess.run([sys.executable, str(ROOT / "tools/run_fords_temperature.py"), str(folder)], capture_output=True, text=True, timeout=90)
            if proc.returncode:
                raise SchemaError("Official ForDs temperature failed: " + proc.stderr[-2000:])
            with np.load(folder / "output.npz", allow_pickle=False) as data:
                values, baseline, lapse = data["values"], data["baseline"], data["coarse_lapse"]
        if values.shape != (source.time.size, target.lat.size, target.lon.size) or not np.isfinite(values).all():
            raise SchemaError("ForDs output invalid; refuse fallback from failed official calculation")
        ds = xr.Dataset({context.variable: (("time", "lat", "lon"), values),
                         "without_terrain": (("time", "lat", "lon"), baseline), "elevation": target.elevation},
                        coords={"time": source.time, "lat": target.lat, "lon": target.lon}, attrs=dict(source.attrs))
        ds[context.variable].attrs = dict(source[context.variable].attrs)
        ds.without_terrain.attrs["units"] = "K"
        return finish(context, self.spec, ds, {"commit": COMMIT, "terrain_used": True,
            "official_function": "downscale_air_temperature", "lapse_min_K_per_m": float(lapse.min()), "lapse_max_K_per_m": float(lapse.max()),
            "max_abs_terrain_adjustment_K": float(np.max(np.abs(values-baseline))), "complete_ForDs": False,
            "adapter_change": "CDO file-I/O bilinear hook -> xarray.interp only; official lapse and correction unchanged",
            "independent_accuracy_verified": False}, started)
