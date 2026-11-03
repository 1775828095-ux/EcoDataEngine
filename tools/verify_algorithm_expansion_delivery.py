"""Check saved outputs and experiment integrity without rerunning algorithms."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "output/algorithm_library_expansion"


def main():
    results = []
    for directory in (FOLDER / "real_examples", FOLDER / "selected_outputs"):
        for path in sorted(directory.glob("*.nc")):
            with xr.open_dataset(path, engine="h5netcdf") as dataset:
                stats = {}
                for name, field in dataset.data_vars.items():
                    if field.dtype.kind in "iuf":
                        values = np.asarray(field.values)
                        assert not np.isinf(values).any(), str(path)
                        stats[name] = {"dtype": str(field.dtype), "finite_count": int(np.isfinite(values).sum()),
                                       "nan_count": int(np.isnan(values).sum()),
                                       "min": float(field.min(skipna=True)), "max": float(field.max(skipna=True)),
                                       "unit": field.attrs.get("units")}
                assert any(s["finite_count"] for s in stats.values()), str(path)
                if path.stem.startswith("bias.xsdba_"):
                    assert stats["PM25"]["min"] >= 0, "PM2.5 nonnegative task constraint failed: " + str(path)
                if {"lat", "lon"}.issubset(dataset.coords):
                    assert dataset.attrs.get("crs"), str(path)
                results.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dimensions": dict(dataset.sizes), "variables": stats})
    experiment = json.loads((FOLDER / "algorithm_selection_expansion_experiment.json").read_text(encoding="utf-8"))
    assert len(experiment["results"]) == 150
    assert max(item["actual_evaluations"] for item in experiment["results"]) <= 60
    for matrix in ("algorithm_capability_matrix.csv", "model_data_capability_matrix.csv"):
        assert len(pd.read_csv(FOLDER / matrix)) > 0
    report = {"readback_status": "PASS", "scope": "file readability, finite data, CRS and recorded experiment budget; not environmental accuracy",
              "netcdf_count": len(results), "files": results}
    (FOLDER / "delivery_integrity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Readable NetCDF outputs:", len(results))
    print("Bias correction ranges:", [(x["path"], x["variables"].get("PM25")) for x in results if "bias.xsdba" in x["path"]])


if __name__ == "__main__":
    main()
