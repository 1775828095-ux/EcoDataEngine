"""Audit V3 artifacts without training or running GOLEM."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion"
V3 = OUT / "v3"


def main():
    run = json.loads((V3 / "dualtimesfield_execution.json").read_text(encoding="utf-8"))
    output = V3 / run["output"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == run["sha256"]
    with xr.open_dataset(output, engine="h5netcdf") as ds:
        assert np.isfinite(ds.weather_value).all()
        assert ds.time.size == run["shape"]["time"]
        assert ds.weather_value.attrs["units"] == "1"
    baseline = json.loads((OUT / "v2_delivery_check.json").read_text(encoding="utf-8"))["golem_sha256_unchanged"]
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in baseline.items())
    with (OUT / "algorithm_capability_matrix.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    codes = ["C5a", "C5b", "C7a", "C7b", "C7c", "C12a", "C12b", "C12c", "C13a", "C13b", "C13c"]
    summary = {code: [{k: r[k] for k in ("Algorithm", "Status", "Executable")} for r in rows if r["CapabilityCode"] == code] for code in codes}
    for r in rows:
        if r["Status"] in {"BLOCKED", "RESEARCH_CANDIDATE", "MODEL_SPECIFIC"}:
            assert r["Executable"] == "False"
    results = {}
    for name in ("targeted_tests.xml", "full_regression.xml"):
        suites = ET.parse(V3 / name).findall(".//testsuite")
        item = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        item["passed"] = item["tests"]-item["failures"]-item["errors"]-item["skipped"]
        results[name] = item
    result = {"golem_core_unchanged": True, "output_verified": run["output"], "tests": results,
        "subcapabilities": summary, "READY_FOR_GOLEM_EXPERIMENT": False,
        "remaining": ["C5 no trained climate inference executed", "C7 official Ningxia dataset absent from two official archives", "C15 generic soil-profile capability unverified"]}
    (V3 / "delivery_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tests": results, "golem_core_unchanged": True, "ready": False}))


if __name__ == "__main__":
    main()
