"""Reproduce the pinned, uncalibrated LREW native SWAT+ baseline (no fusion)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]
                        / "output/swatplus_baseline_20261026")
    parser.add_argument("--run-name", default="lrew_verified")
    args = parser.parse_args()
    root = args.root.resolve()
    provenance = json.loads((root / "lrew_provenance.json").read_text(encoding="utf-8"))
    for record in provenance["files"]:
        assert sha256(root / "lrew_source" / record["path"]) == record["sha256"]
    source = root / "lrew_extracted/plus_rev59_project/swatplus_rev59_demo"
    executable = root / "lrew_extracted/plus_rev59_win/swatplus_rev59.exe"
    run = root / "runs" / args.run_name
    if run.resolve().parent != (root / "runs").resolve():
        raise ValueError("run-name must be a single directory name")
    run.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for file in source.iterdir():
        if file.is_file() and file.name != "channel_yr.txt":
            shutil.copy2(file, run / file.name)
            manifest[file.name] = sha256(file)
    # Change reporting only. Keep weather, management, parameters and time unchanged.
    reporting = run / "print.prt"
    lines = reporting.read_text(encoding="utf-8").splitlines()
    changed = []
    for i, line in enumerate(lines):
        if line.split() and line.split()[0] == "channel":
            lines[i] = "channel                      y             y             y             n"
            changed.append({"before": line, "after": lines[i]})
    assert len(changed) == 1
    reporting.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, digest in manifest.items():
        if name != "print.prt":
            assert sha256(run / name) == digest
    assert not (run / "channel_day.txt").exists()
    started = time.perf_counter()
    with (run / "stdout.log").open("wb") as stdout, (run / "stderr.log").open("wb") as stderr:
        result = subprocess.run([str(executable)], cwd=run, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr, timeout=240, check=False)
    seconds = time.perf_counter() - started
    assert result.returncode == 0
    assert "Execution successfully completed" in (run / "stdout.log").read_text(errors="replace")
    assert (run / "stderr.log").stat().st_size == 0
    channels = pd.read_csv(run / "channel_day.txt", sep=r"\s+", skiprows=[0, 2])
    assert np.isfinite(channels.select_dtypes("number")).all().all()
    # Native routing graph: reaches 2 and 3 feed 1; reach 1 has zero out-links.
    routing = pd.read_csv(run / "channel.con", sep=r"\s+", skiprows=1)
    assert routing.loc[routing.out_tot == 0, "id"].tolist() == [1]
    assert routing.loc[routing.id.isin([2, 3]), "obj_id"].tolist() == [1, 1]
    daily = channels.loc[channels.unit == 1].copy()
    daily["date"] = pd.to_datetime(dict(year=daily.yr, month=daily.mon, day=daily.day))
    expected = pd.date_range("2003-01-01", "2012-12-31", freq="D")
    assert daily.date.tolist() == expected.tolist()
    assert (daily.flo_out >= 0).all() and daily.flo_out.nunique() > 1
    # Rev59.3 output explicitly labels flow as ha-m per reporting day, not m3/s.
    assert "ha-m" in (run / "channel_day.txt").read_text().splitlines()[2]
    daily["simulation_m3_s"] = daily.flo_out * 10000.0 / 86400.0
    sys.path.insert(0, str(root / "dependencies"))
    import rdata

    observed = rdata.read_rds(root / "lrew_source/inst/extdata/obs.rds")
    # R Date is days since 1970-01-01. Source documentation independently states coverage.
    observed["date"] = pd.to_datetime(observed.date, unit="D", origin="1970-01-01")
    assert observed.date.min() == pd.Timestamp("1968-01-01")
    assert observed.date.max() == pd.Timestamp("2012-12-31")
    assert not observed.date.duplicated().any()
    assert np.isfinite(observed.discharge).all() and (observed.discharge >= 0).all()
    observed.to_csv(run / "observed_daily_m3_s.csv", index=False)
    comparison = daily[["date", "flo_out", "simulation_m3_s"]].merge(
        observed, on="date", how="left", validate="one_to_one")
    assert len(comparison) == len(expected) and comparison.discharge.notna().all()
    comparison.to_csv(run / "outlet_daily_comparison.csv", index=False)
    error = comparison.simulation_m3_s - comparison.discharge
    metrics = {
        "n": len(comparison), "unit": "m3/s", "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs(error))), "bias": float(error.mean()),
        "pbias_percent_sim_minus_obs": float(100 * error.sum() / comparison.discharge.sum()),
        "nse": float(1 - (error**2).sum() /
                     ((comparison.discharge - comparison.discharge.mean())**2).sum()),
        "mean_simulation": float(comparison.simulation_m3_s.mean()),
        "mean_observed": float(comparison.discharge.mean()),
    }
    weather = []
    for suffix in ["pcp", "tmp", "slr", "hmd", "wnd"]:
        for file in sorted(source.glob("*." + suffix)):
            frame = pd.read_csv(file, sep=r"\s+", skiprows=3, header=None).dropna(axis=1, how="all")
            dates = pd.to_datetime(frame[0].astype(str), format="%Y") + pd.to_timedelta(frame[1] - 1, unit="D")
            mask = (dates >= "2000-01-01") & (dates <= "2012-12-31")
            values = frame.loc[mask, frame.columns[2:]].to_numpy(dtype=float)
            assert dates[mask].tolist() == pd.date_range("2000-01-01", "2012-12-31").tolist()
            weather.append({"file": file.name, "rows_simulation_period": int(mask.sum()),
                            "values": int(values.size), "minus99_values": int((values == -99).sum()),
                            "nonfinite_values": int((~np.isfinite(values)).sum()),
                            "date_min": str(dates.min().date()), "date_max": str(dates.max().date())})
    audit = {
        "model": "SWAT+ 59.3", "source_commit": provenance["commit"],
        "executable_sha256": sha256(executable), "exit_code": result.returncode,
        "wall_seconds": seconds, "simulation": ["2000-01-01", "2012-12-31"],
        "warmup_years": 3, "evaluation": [str(expected.min().date()), str(expected.max().date())],
        "hru_count": 131, "channel_count": 3, "reservoir_count": 3, "outlet_channel": 1,
        "input_hashes": manifest, "reporting_changes_only": changed,
        "channel_rows_all_reaches": len(channels), "weather": weather, "metrics": metrics,
        "channel_output_sha256": sha256(run / "channel_day.txt"),
        "no_calibration": True, "no_fusion": True,
    }
    (run / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    # Reopen written artifacts, independently of in-memory frames.
    saved = pd.read_csv(run / "outlet_daily_comparison.csv")
    assert len(saved) == 3653
    assert abs(np.sqrt(np.mean((saved.simulation_m3_s - saved.discharge)**2)) - metrics["rmse"]) < 1e-12
    print(json.dumps({"run": str(run), "exit_code": result.returncode,
                      "weather": weather, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
