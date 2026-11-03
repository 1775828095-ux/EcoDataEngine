"""Fixed-task local ClimateLearn weights; not arbitrary-scale inference."""
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
FOLDER = ROOT / "output/algorithm_library_expansion/v4"
RUNTIME = ROOT / ".venv-capability-v4/Scripts/python.exe"
COMMIT = "b48fb0242acc47e365af86bfbd9dd86e9dcbd6d2"


def ready():
    if not RUNTIME.exists() or not (FOLDER / "climatelearn_model_contract.json").exists() or not (FOLDER / "climatelearn_training.json").exists():
        return False
    info = json.loads((FOLDER / "climatelearn_training.json").read_text(encoding="utf-8"))
    weights = FOLDER / "climatelearn_resnet_local.pt"
    if not weights.is_file() or info.get("checkpoint_sha256") != hashlib.sha256(weights.read_bytes()).hexdigest() or not info.get("optimizer_steps", 0):
        return False
    source = ROOT / "external/climatelearn"
    manifest = json.loads((source / "source_evidence.json").read_text(encoding="utf-8"))
    paths = {"src/climate_learn/models/hub/resnet.py", "src/climate_learn/models/hub/components/cnn_blocks.py",
             "src/climate_learn/models/module.py", "src/climate_learn/models/hub/utils.py", "LICENSE"}
    entries = {x["path"]: x for x in manifest["files"]}
    return manifest["commit"] == COMMIT and all(p in entries and (source / p).is_file() and
        hashlib.sha256((source / p).read_bytes()).hexdigest() == entries[p]["sha256"] for p in paths)


class ClimateLearnTemperaturePlugin:
    def __init__(self):
        self.spec = AlgorithmSpec("downscale.climatelearn_t2m", "ClimateLearn ResNet local ERA5 temperature model", COMMIT,
            "learned_climate_downscaling", "available" if ready() else "weights_missing", ("time_grid",), ("continuous",),
            scale_range=(4., 4.), requires_weights=True, license="MIT architecture; locally trained weights", backend="official ClimateLearn ResNet",
            source_url="https://github.com/aditya-grover/climate-learn", capabilities=("learning_based_spatial_reconstruction", "climate_temperature_downscaling", "continuous_spatial_resampling"),
            preconditions=("air_temperature", "K", "exact trained regional grid", "fixed 4x", "local trained weights"),
            input_contract={"shape": "time x 6 x 6", "variable": "air_temperature", "units": "K"},
            output_contract={"shape": "time x 24 x 24", "kind": "temperature_grid", "arbitrary_resolution_verified": False},
            runtime_requirements=(".venv-capability-v4; CPU",), verification_status="VERIFIED" if ready() else "BLOCKED",
            official_entry_point="ClimateLearn ResNet.forward; trained using official LitModule/Lightning Trainer",
            verification_evidence=("output/algorithm_library_expansion/v4/climatelearn_training.json",),
            notes="Only this regional 1deg-derived to 0.25deg ERA5 t2m task; not LAI/NPP, new regions or 30m accuracy.")

    def compatibility(self, context):
        if not ready():
            return Compatibility(False, ("Verified source, trained weights and V4 environment required",))
        if not context.sources or context.variable not in context.sources[0].dataset:
            return Compatibility(False, ("Temperature field missing",))
        ds = context.sources[0].dataset
        a = ds[context.variable]
        info = json.loads((FOLDER / "climatelearn_model_contract.json").read_text(encoding="utf-8"))
        reasons = []
        if context.semantic_type != "continuous" or a.attrs.get("standard_name") != "air_temperature" or a.attrs.get("units") != "K":
            reasons.append("This trained model supports confirmed air_temperature in K only; LAI/NPP unsupported")
        if a.dims != ("time", "lat", "lon") or a.shape[-2:] != (6, 6):
            reasons.append("Expected time x 6 x 6 input")
            return Compatibility(False, tuple(reasons))
        if a.dtype.kind not in "fiu":
            return Compatibility(False, ("Numeric temperature field required",))
        if ds.attrs.get("crs") != info["crs"]:
            reasons.append("CRS differs from trained grid")
        for coordinate, key in (("lat", "input_lat"), ("lon", "input_lon")):
            if coordinate not in ds.coords or ds[coordinate].size != len(info[key]) or not np.allclose(ds[coordinate], info[key], atol=1e-8, rtol=0):
                reasons.append("Input coordinates differ from trained region: " + coordinate)
        target = context.parameters.get("target_grid")
        for coordinate, values in (("lat", context.target_lat), ("lon", context.target_lon)):
            wanted = target[coordinate].values if isinstance(target, xr.Dataset) and coordinate in target else values
            if wanted is not None and (len(wanted) != 24 or not np.allclose(wanted, info["target_"+coordinate], rtol=0, atol=1e-8)):
                reasons.append("Fixed trained target grid only; arbitrary scale unsupported")
        if context.parameters.get("scale_factor", 4) != 4 or context.parameters.get("arbitrary_resolution", False):
            reasons.append("Only fixed 4x reconstruction is verified")
        if not np.isfinite(a.values).all() or not a.time.size:
            reasons.append("Finite, nonempty temperature input required")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        source = context.sources[0].dataset
        contract = json.loads((FOLDER / "climatelearn_model_contract.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ecodata_climatelearn_") as directory:
            folder = Path(directory)
            np.savez(folder / "input.npz", values=source[context.variable].values)
            proc = subprocess.run([str(RUNTIME), str(ROOT / "tools/infer_climatelearn_v4.py"), str(folder)], capture_output=True, text=True, timeout=120)
            if proc.returncode:
                raise SchemaError("Official ClimateLearn inference failed: " + proc.stderr[-2000:])
            values = np.load(folder / "output.npy", allow_pickle=False)
        if values.shape != (source.time.size, 24, 24) or not np.isfinite(values).all():
            raise SchemaError("ClimateLearn output shape or finite-value check failed")
        ds = xr.Dataset({context.variable: (("time", "lat", "lon"), values)},
            coords={"time": source.time, "lat": contract["target_lat"], "lon": contract["target_lon"]}, attrs=dict(source.attrs))
        ds[context.variable].attrs = dict(source[context.variable].attrs)
        return finish(context, self.spec, ds, {"official_architecture": "ClimateLearn ResNet", "commit": COMMIT,
            "weight_origin": "LOCAL_TRAINING", "fixed_scale_factor": 4, "arbitrary_resolution_verified": False,
            "original_information_scale": "derived 1deg input; ERA5 native 0.25deg reference, not independently verified 30m detail"}, started)
