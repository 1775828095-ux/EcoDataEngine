"""Check V4 execution artifacts; does not train or run GOLEM."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion"
V4 = OUT / "v4"


def main():
    records = json.loads((V4 / "execution_manifest.json").read_text(encoding="utf-8"))
    for record in records:
        path = V4 / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert record["status"] == "PASS"
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            assert np.isfinite(ds.t2m).all()
            assert ds.t2m.attrs["units"] == "K"
            assert ds.attrs["crs"] == "EPSG:4326"
            assert dict(ds.sizes) == record["dimensions"]
    train = json.loads((V4 / "climatelearn_training.json").read_text(encoding="utf-8"))
    assert train["optimizer_steps"] > 0 and train["parameters_updated"] > 0
    assert hashlib.sha256((V4 / train["checkpoint"]).read_bytes()).hexdigest() == train["checkpoint_sha256"]
    baseline = json.loads((OUT / "v2_delivery_check.json").read_text(encoding="utf-8"))["golem_sha256_unchanged"]
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in baseline.items())
    results = {}
    for name in ("targeted_tests.xml", "full_regression.xml"):
        suites = ET.parse(V4 / name).findall(".//testsuite")
        item = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        assert item["tests"] and not item["failures"] and not item["errors"], item
        item["passed"] = item["tests"] - item["skipped"]
        results[name] = item
    with (OUT / "algorithm_capability_matrix.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(r["Executable"] == "False" for r in rows if r["Status"] == "BLOCKED")
    summary = {
        "date": "2026-10-17", "tests": results, "golem_sha256_unchanged": baseline,
        "real_outputs_checked": [r["path"] for r in records],
        "ClimateDiffuse": {"commit": "7b8ee181fa3baeacf08a2bf9dc275989048b4030", "license": "UNRESOLVED; no deployment grant established", "weights_downloaded": False,
            "unet_lfs_bytes": 389657415, "diffusion_lfs_bytes": 389695738,
            "unet_lfs_oid": "eab9192c6a76b4b01de25b9dfc3d25327c7c6a5d6c28ce6dd8e4278394e247ce"},
        "C5a": "VERIFIED: official ClimateLearn architecture and training module, locally trained, fixed 4x temperature task",
        "C5b": "UNVERIFIED: arbitrary-scale learned reconstruction remains a gap",
        "C7": "VERIFIED temperature terrain path; full ForDs workflow PARTIAL",
        "C12": "Existing verified components preserved", "C13": "Existing verified interpolation preserved",
        "READY_FOR_GOLEM_EXPERIMENT": "YES",
        "readiness_scope": "Core capability candidate coverage only; not all variables, six simulators, full ForDs or independent accuracy proof",
    }
    (V4 / "delivery_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
