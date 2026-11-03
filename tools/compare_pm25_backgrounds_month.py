"""Full-August PM2.5 background screening without neural-network training."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from compare_pm25_backgrounds import (
    SOURCE,
    download_merra2_gmi,
    file_sha256,
    metric_values,
    sample,
    variation_metrics,
)
from run_real_pm25_three_source import DATA_ROOT, GEOS, GROUND


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "output" / "pm25_background_month_screening_20261012"
MERRA_DIR = DATA_ROOT / "02b_MERRA2_GMI模拟背景场"
RAW = MERRA_DIR / "merra2_gmi_pm25_bot_201908_bth_original_units.nc"
STANDARD = MERRA_DIR / "merra2_gmi_pm25_201908_bth_ugm3.nc"
START = pd.Timestamp("2019-08-01 00:00:00")
END = pd.Timestamp("2019-09-01 00:00:00")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def fit_model(frame: pd.DataFrame, columns: list[str]) -> dict:
    train = frame[frame.split == "train"].dropna(subset=["observed", *columns])
    design = np.column_stack([train[column].to_numpy(float) for column in columns] + [np.ones(len(train))])
    target = train.observed.to_numpy(float)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ coefficients
    r2 = 1 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)
    return {
        "columns": columns,
        "coefficients": {column: float(value) for column, value in zip(columns, coefficients[:-1])},
        "intercept": float(coefficients[-1]),
        "train_r2": float(r2),
    }


def predict(frame: pd.DataFrame, model: dict) -> np.ndarray:
    result = np.full(len(frame), model["intercept"], dtype=float)
    for column, coefficient in model["coefficients"].items():
        result += coefficient * frame[column].to_numpy(float)
    return np.maximum(result, 0)


def hourly_peak_metrics(frame: pd.DataFrame, column: str) -> dict:
    peaks = frame.groupby("time")[["observed", column]].max().dropna()
    error = peaks[column] - peaks.observed
    return {
        "hourly_peak_bias": float(error.mean()),
        "hourly_peak_mae": float(np.abs(error).mean()),
        "hourly_peak_rmse": float(np.sqrt(np.mean(error**2))),
    }


def anomaly_pair_metrics(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["observed", "geos_cf", "merra2_gmi"]).copy()
    for column in ("observed", "geos_cf", "merra2_gmi"):
        valid[f"{column}_spatial_anomaly"] = valid[column] - valid.groupby("time")[column].transform("mean")
    hourly = valid.groupby("time")[["observed", "geos_cf", "merra2_gmi"]].mean()
    return {
        "raw_background_correlation": float(valid[["geos_cf", "merra2_gmi"]].corr().iloc[0, 1]),
        "background_error_correlation": float(
            np.corrcoef(valid.geos_cf - valid.observed, valid.merra2_gmi - valid.observed)[0, 1]
        ),
        "spatial_anomaly_correlation_between_backgrounds": float(
            valid[["geos_cf_spatial_anomaly", "merra2_gmi_spatial_anomaly"]].corr().iloc[0, 1]
        ),
        "hourly_mean_correlation_between_backgrounds": float(hourly[["geos_cf", "merra2_gmi"]].corr().iloc[0, 1]),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MERRA_DIR.mkdir(parents=True, exist_ok=True)
    merra = download_merra2_gmi(STANDARD, START, END, raw_path=RAW)

    with xr.open_dataset(RAW, engine="h5netcdf") as opened:
        raw_check = opened.load()
    with xr.open_dataset(STANDARD, engine="h5netcdf") as opened:
        standard_check = opened.load()
    if raw_check.pm25_bot.attrs.get("units") != "kg m-3":
        raise RuntimeError("original MERRA2-GMI unit was not preserved")
    if standard_check.pm25.attrs.get("units") != "ug m-3":
        raise RuntimeError("standardized MERRA2-GMI unit is invalid")
    if not np.isfinite(raw_check.pm25_bot).all() or not np.isfinite(standard_check.pm25).all():
        raise RuntimeError("MERRA2-GMI monthly subset contains non-finite values")
    conversion_error = float(np.max(np.abs(standard_check.pm25.values - raw_check.pm25_bot.values * 1e9)))

    ground = pd.read_parquet(GROUND)
    ground["time"] = pd.to_datetime(ground.geos_average_center_utc)
    ground = ground[
        ground.split.isin(["train", "validation"])
        & ground.time.between("2019-08-01 00:30", "2019-08-31 23:30")
        & ground.three_source_complete
    ].rename(columns={
        "pm25_as_stored": "observed",
        "longitude_as_stored": "longitude",
        "latitude_as_stored": "latitude",
    }).copy()
    if (ground.split == "test").any() or set(ground.split.unique()) != {"train", "validation"}:
        raise RuntimeError("final test stations must remain sealed")

    with xr.open_dataset(GEOS, engine="h5netcdf") as opened:
        geos = opened.pm25_rh35_gcc.isel(lev=0, drop=True).sel(
            time=slice("2019-08-01 00:30", "2019-08-31 23:30")
        ).load()
    ground["geos_cf"] = sample(geos, ground)
    ground["merra2_gmi"] = sample(merra.pm25, ground)
    ground = ground.dropna(subset=["observed", "geos_cf", "merra2_gmi"]).copy()

    models = {
        "geos_affine": fit_model(ground, ["geos_cf"]),
        "merra_affine": fit_model(ground, ["merra2_gmi"]),
        "two_background_linear": fit_model(ground, ["geos_cf", "merra2_gmi"]),
    }
    ground["geos_affine"] = predict(ground, models["geos_affine"])
    ground["merra_affine"] = predict(ground, models["merra_affine"])
    ground["two_background_linear"] = predict(ground, models["two_background_linear"])
    ground["mean_of_debiased"] = (ground.geos_affine + ground.merra_affine) / 2

    threshold = float(ground.loc[ground.split == "train", "observed"].quantile(.90))
    methods = ["geos_cf", "merra2_gmi", "geos_affine", "merra_affine", "mean_of_debiased", "two_background_linear"]
    rows = []
    for split in ("train", "validation"):
        part = ground[ground.split == split]
        for method in methods:
            row = {"split": split, "method": method, **metric_values(part.observed, part[method])}
            row.update(variation_metrics(part, method))
            row.update(hourly_peak_metrics(part, method))
            high = part[part.observed >= threshold]
            high_metrics = metric_values(high.observed, high[method])
            row.update({f"high_{key}": value for key, value in high_metrics.items() if key in {"records", "bias", "rmse", "mae"}})
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "monthly_background_metrics.csv", index=False)
    ground[["station_id", "time", "split", "observed", *methods]].to_parquet(
        OUTPUT / "monthly_station_background_samples.parquet", index=False)

    complementarity = anomaly_pair_metrics(ground[ground.split == "validation"])
    validation = metrics[metrics.split == "validation"].set_index("method")
    complementarity.update({
        "two_background_rmse_improvement_vs_best_single_affine": float(
            min(validation.loc["geos_affine", "rmse"], validation.loc["merra_affine", "rmse"])
            - validation.loc["two_background_linear", "rmse"]
        ),
        "two_background_mae_improvement_vs_best_single_affine": float(
            min(validation.loc["geos_affine", "mae"], validation.loc["merra_affine", "mae"])
            - validation.loc["two_background_linear", "mae"]
        ),
    })
    dump(OUTPUT / "complementarity.json", complementarity)

    manifest = {
        "selection_policy": "fit only on 40 train stations; compare only on 10 validation stations; final test stations sealed",
        "test_records_used": 0,
        "period": ["2019-08-01 00:30 UTC", "2019-08-31 23:30 UTC"],
        "record_counts": ground.groupby("split").size().to_dict(),
        "station_counts": ground.groupby("split").station_id.nunique().to_dict(),
        "training_peak_threshold_p90_ug_m3": threshold,
        "models": models,
        "files": {
            "merra_original": {"path": str(RAW), "bytes": RAW.stat().st_size, "sha256": file_sha256(RAW)},
            "merra_standardized": {"path": str(STANDARD), "bytes": STANDARD.stat().st_size, "sha256": file_sha256(STANDARD)},
            "geos_existing": {"path": str(GEOS), "bytes": GEOS.stat().st_size, "sha256": file_sha256(GEOS)},
        },
        "merra": {
            "source": SOURCE,
            "dims": dict(merra.sizes),
            "original_unit": raw_check.pm25_bot.attrs.get("units"),
            "standardized_unit": standard_check.pm25.attrs.get("units"),
            "unit_conversion_max_abs_error": conversion_error,
            "time_semantics": "instantaneous UTC hour; interpolated to HH:30 for comparison, not converted to a one-hour average",
            "min_ug_m3": float(standard_check.pm25.min()),
            "max_ug_m3": float(standard_check.pm25.max()),
            "nan_count": int(np.isnan(standard_check.pm25.values).sum()),
        },
        "limitations": [
            "station coordinate datum remains approximate WGS84",
            "MERRA2-GMI instantaneous and GEOS-CF/ground hourly-average semantics differ",
            "validation selection does not authorize final test evaluation",
        ],
    }
    dump(OUTPUT / "monthly_screening_manifest.json", manifest)
    print(metrics.to_string(index=False))
    print(json.dumps(complementarity, indent=2))


if __name__ == "__main__":
    main()
