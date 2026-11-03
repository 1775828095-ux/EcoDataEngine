"""Independent readback of the bounded cooperation acceptance products."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


def audit(roots):
    records = []
    failures = []
    summaries = []
    for root in roots:
        summaries.extend(Path(root).rglob("summary.json"))
    for path in sorted(set(summaries)):
        summary = json.loads(path.read_text(encoding="utf-8"))
        for row in summary.get("rows", []):
            try:
                if row["status"] != "PASS":
                    raise ValueError("Execution did not pass: " + str(row.get("error")))
                product = Path(row["product"])
                if hashlib.sha256(product.read_bytes()).hexdigest() != row["product_sha256"]:
                    raise ValueError("Product SHA mismatch")
                with xr.open_dataset(product) as dataset:
                    dataset.load()
                    candidates = [name for name in dataset.data_vars if name not in {
                        "uncertainty", "fusion_uncertainty", "background_value", "residual_correction", "residual_gp_std"}]
                    if len(candidates) != 1:
                        raise ValueError("Ambiguous primary variable")
                    field = dataset[candidates[0]]
                    if not np.isfinite(field.values).all() or not field.attrs.get("units"):
                        raise ValueError("Nonfinite output or absent unit")
                    if {"background_value", "residual_correction"} <= set(dataset):
                        np.testing.assert_allclose(field, dataset.background_value + dataset.residual_correction, atol=1e-10)
                records.append(dict(summary=str(path), product=str(product), steps=row["steps"],
                                    sha256=row["product_sha256"], arithmetic_checked="residual_correction" in dataset))
            except Exception as exc:
                failures.append(dict(summary=str(path), key=row.get("key"), error=str(exc)))
    return dict(status="PASS" if records and not failures else "FAIL", products_verified=len(records),
                failures=failures, products=records, scope="synthetic public-entry cooperation acceptance; no environmental superiority claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(args.roots)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key != "products"}))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
