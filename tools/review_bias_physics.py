"""Retain this run's rejected additive outputs outside the deliverable examples."""
import json
from pathlib import Path

import xarray as xr

from ecodata.evaluation.quality_gates import ResultQualityGate

base = Path(__file__).resolve().parents[1] / "output/algorithm_library_expansion"
rejected = base / "rejected_outputs"
rejected.mkdir(exist_ok=True)
for method in ("eqm", "dqm", "qdm"):
    source = base / "real_examples" / ("bias.xsdba_" + method + ".nc")
    if not source.exists():
        continue
    with xr.open_dataset(source, engine="h5netcdf") as output:
        report = ResultQualityGate().validate(output, output, variable="PM25", semantic_type="continuous", nonnegative=True)
    if report["status"] != "BLOCK":
        continue
    target = rejected / (source.stem + "_additive_FAIL.nc")
    assert source.resolve().is_relative_to(base.resolve()) and target.resolve().is_relative_to(base.resolve())
    if target.exists():
        raise RuntimeError("Preserve existing rejected evidence; refusing to overwrite " + str(target))
    source.rename(target)
    old_json = source.with_suffix(".json")
    old_report = json.loads(old_json.read_text(encoding="utf-8")) if old_json.exists() else {}
    old_report.update(quality=report, decision="REJECTED_FROM_DELIVERABLE; preserved as failure evidence",
                      reason="Additive distribution correction generated negative PM2.5; no clipping applied",
                      output=str(target.relative_to(base)))
    (rejected / (target.stem + ".json")).write_text(json.dumps(old_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Rejected", method, report["status"])
