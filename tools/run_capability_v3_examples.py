"""Official real USHCN smoke execution plus bounded upstream diagnostics."""
import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.schemas import EcoDataBundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion/v3"


def main():
    path = ROOT / "external/dualtimesfield/interpolation/datasets.py"
    spec = importlib.util.spec_from_file_location("official_ushcn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.USHCNDataset(root_dir=str(OUT), split="train", download=True)
    sample = dataset[0]
    times = sample["times"].numpy()[:128]
    column = int(sample["mask"][:128].sum(0).argmax())
    truth = sample["values"][:128, column].numpy()
    original_mask = sample["mask"][:128, column].numpy().astype(bool)
    span = np.flatnonzero(original_mask)
    section = slice(span[0], span[-1]+1)
    times, truth, original_mask = times[section], truth[section], original_mask[section]
    observed = original_mask.copy()
    ids = np.flatnonzero(original_mask)
    withheld = ids[2:-2:5]
    observed[withheld] = False
    values = np.where(observed, truth, np.nan)
    ds = xr.Dataset({"weather_value": ("time", values)}, coords={"time": times},
        attrs={"source": "USHCN benchmark via official DualTimesField USHCNDataset", "source_station_id": dataset.data[0][0],
               "time_origin": "Not provided by benchmark; numeric benchmark time, not invented calendar dates"})
    ds.weather_value.attrs.update(units="1", long_name=f"Official normalized USHCN benchmark channel {column}", data_kind="continuous")
    ds.time.attrs["long_name"] = "original irregular benchmark time; calendar origin unknown"
    ds.to_netcdf(OUT / "ushcn_masked_input.nc", engine="h5netcdf")
    context = AlgorithmContext((EcoDataBundle(ds),), "weather_value", "continuous", parameters={"epochs": 10, "seed": 42})
    result = UnifiedAlgorithmPipeline(create_default_registry()).run(context,
        AlgorithmPlan("ushcn_irregular_smoke", ("temporal.dualtimesfield",), 0., ("Official short fit/predict, no accuracy selection",)))
    output = OUT / "ushcn_dualtimesfield_output.nc"
    result.data.dataset.to_netcdf(output, engine="h5netcdf")
    with xr.open_dataset(output, engine="h5netcdf") as check:
        np.testing.assert_array_equal(check.time, times)
        assert np.isfinite(check.weather_value).all()
        np.testing.assert_allclose(check.weather_value.values[observed], truth[observed])
    record = {"algorithm": "temporal.dualtimesfield", "status": "VERIFIED", "commit": "6a2263fd6b7cee4f0a662406df0eb850543fe086",
        "input": "ushcn_masked_input.nc", "output": output.name, "source": "REAL USHCN benchmark, official loader train split",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "shape": dict(result.data.dataset.sizes), "time_range": [float(times.min()), float(times.max())],
        "irregular_intervals": bool(np.unique(np.diff(times)).size > 1), "original_observed": int(original_mask.sum()),
        "withheld_for_fill_check": len(withheld), "missing_after": 0, "units": "1 (official normalized benchmark channel)",
        "report": result.report, "full_environmental_accuracy_verified": False,
        "limitations": "10 epoch fit/predict only; no paper reproduction; benchmark normalized values and numeric time do not establish original NOAA units/calendar origin"}
    (OUT / "dualtimesfield_execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime = ROOT / ".venv-capability-v3/Scripts/python.exe"
    process = subprocess.run([str(runtime), "main.py", "--gpu", "-1", "--batch_size", "8", "--hidden_size", "8"],
        cwd=ROOT / "external/phydl_nwp", capture_output=True, text=True, timeout=30)
    (OUT / "phydl_execution.json").write_text(json.dumps({"command": process.args, "returncode": process.returncode,
        "stdout": process.stdout, "stderr": process.stderr, "status": "BLOCKED" if process.returncode else "PARTIAL"}, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("status", "shape", "time_range", "original_observed", "withheld_for_fill_check", "missing_after")}))


if __name__ == "__main__":
    main()
