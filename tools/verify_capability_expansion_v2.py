"""Read-only delivery checks and a compact capability summary; no search/training."""
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion"
GOLEM_BASELINE = {
    "src/ecodata/algorithms/golem_search.py": "9294ed236fbf96fd5ac7ccb974566e07c34f95b4ee3c736f207f1218823c8f4b",
    "tools/run_golem_search.py": "8d693e84cf84b6c9a2100628d466c9163d5d12a33a0976f4bca14ad8e808994a"}


def main():
    records = json.loads((OUT / "v2_examples/execution_manifest.json").read_text(encoding="utf-8"))
    for record in records:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            assert all(np.isfinite(a.values).all() for a in ds.data_vars.values() if a.dtype.kind in "fiu")
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in GOLEM_BASELINE}
    assert hashes == GOLEM_BASELINE, "GOLEM core changed"
    with (OUT / "algorithm_capability_matrix.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = []
    for number in range(1, 31):
        code = "C" + str(number)
        matches = [r for r in rows if code in r["CapabilityCode"].split(";")]
        verified = sorted({r["Algorithm"] for r in matches if r["Status"] == "VERIFIED" and r["Executable"] == "True"})
        summary.append({"capability": code, "verified_algorithms": verified,
            "status_counts": dict(Counter(r["Status"] for r in matches)), "multiple_verified_candidates": len(verified) > 1})
    tests = {}
    for name in ("v2_tests.xml", "v2_full_regression.xml"):
        suites = ET.parse(OUT / name).findall(".//testsuite")
        totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
        tests[name] = totals
    result = {"output_files_checked": len(records), "golem_sha256_unchanged": hashes,
              "runtime_probes": {"timm_available": importlib.util.find_spec("timm") is not None,
                                 "scipy_special_sph_harm_available": hasattr(special, "sph_harm")},
              "tests": tests, "capability_summary": summary, "READY_FOR_GOLEM_EXPERIMENT": False,
              "blocking_gaps": ["C5 no verified learned arbitrary-scale environmental reconstruction", "C7 full terrain/weather workflow not verified", "C13 no executed irregular-time learning adapter"]}
    (OUT / "v2_delivery_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
