"""Compare PM2.5 background fields without using the final test stations.

Downloads only a seven-day Beijing-Tianjin-Hebei subset from the anonymous
NASA MERRA2-GMI OPeNDAP service, then compares it with the existing GEOS-CF
field on the fixed train/validation station split.  It does not train or alter
any reconstruction network.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from run_real_pm25_three_source import DATA_ROOT, GEOS, GROUND


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "output" / "pm25_background_screening_20261012"
SOURCE = "https://opendap.nccs.nasa.gov/dods/merra2_gmi/inst1_2d_bot_Nx"
START = pd.Timestamp("2019-08-01 00:00:00")
END = pd.Timestamp("2019-08-08 00:00:00")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_merra2_gmi(
    path: Path,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
    raw_path: Path | None = None,
) -> xr.Dataset:
    if path.exists():
        with xr.open_dataset(path, engine="h5netcdf") as opened:
            return opened.load()

    remote = xr.open_dataset(SOURCE, decode_times=False)
    origin = pd.Timestamp("1980-01-01 01:00:00")
    first = int((start - origin) / pd.Timedelta(hours=1))
    last = int((end - origin) / pd.Timedelta(hours=1))
    lat_values = remote.lat.values
    lon_values = remote.lon.values
    lat_idx = np.where((lat_values >= 34.0) & (lat_values <= 43.5))[0]
    lon_idx = np.where((lon_values >= 111.0) & (lon_values <= 121.5))[0]
    subset = remote[["pm25_bot"]].isel(
        time=slice(first, last + 1),
        lat=slice(int(lat_idx.min()), int(lat_idx.max()) + 1),
        lon=slice(int(lon_idx.min()), int(lon_idx.max()) + 1),
    ).load()
    remote.close()
    times = pd.date_range(start, end, freq="h")
    subset = subset.assign_coords(time=times.values)
    subset.attrs.update(
        source=SOURCE,
        product="NASA MERRA2-GMI inst1_2d_bot_Nx",
        access="anonymous official NASA NCCS OPeNDAP",
        crs="EPSG:4326",
        spatial_resolution="0.5 degree latitude x 0.625 degree longitude",
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        note="aerosols are simulated; MERRA2-GMI does not constrain total AOD to observations",
    )
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_netcdf(raw_path, engine="h5netcdf")
    subset = subset.rename(pm25_bot="pm25")
    subset["pm25"] = subset.pm25 * 1e9
    subset.pm25.attrs.update(
        units="ug m-3",
        long_name="MERRA2-GMI total reconstructed surface PM2.5",
        original_variable="pm25_bot",
        original_units="kg m-3",
        conversion="ug m-3 = kg m-3 * 1e9",
        temporal_semantics="instantaneous at the labelled UTC hour",
    )
    subset.lat.attrs.update(units="degrees_north", standard_name="latitude")
    subset.lon.attrs.update(units="degrees_east", standard_name="longitude")
    subset.time.attrs.update(standard_name="time")
    subset.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as check:
        loaded = check.load()
    if not np.isfinite(loaded.pm25).all() or loaded.sizes["time"] != len(times):
        raise RuntimeError("MERRA2-GMI subset reread validation failed")
    return loaded


def sample(field: xr.DataArray, frame: pd.DataFrame) -> np.ndarray:
    return field.interp(
        time=xr.DataArray(pd.to_datetime(frame.time).values, dims="record"),
        lat=xr.DataArray(frame.latitude.to_numpy(float), dims="record"),
        lon=xr.DataArray(frame.longitude.to_numpy(float), dims="record"),
    ).values


def metric_values(observed, predicted) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    error = predicted - observed
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if len(observed) > 1 else None
    return {
        "records": int(len(error)),
        "bias": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "correlation": correlation,
        "observed_mean": float(observed.mean()),
        "predicted_mean": float(predicted.mean()),
        "observed_max": float(observed.max()),
        "predicted_max": float(predicted.max()),
        "peak_amplitude_error": float(predicted.max() - observed.max()),
    }


def fit_affine(frame: pd.DataFrame, column: str) -> dict:
    train = frame[frame.split == "train"].dropna(subset=[column, "observed"])
    x = train[column].to_numpy(float)
    y = train.observed.to_numpy(float)
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    fitted = slope * x + intercept
    r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"slope": float(slope), "intercept": float(intercept), "train_r2": float(r2)}


def variation_metrics(frame: pd.DataFrame, prediction: str) -> dict:
    valid = frame.dropna(subset=["observed", prediction]).copy()
    hourly = valid.groupby("time")[["observed", prediction]].mean()
    temporal = float(hourly.corr().iloc[0, 1])
    valid["observed_spatial_anomaly"] = valid.observed - valid.groupby("time")["observed"].transform("mean")
    valid["prediction_spatial_anomaly"] = valid[prediction] - valid.groupby("time")[prediction].transform("mean")
    spatial = float(valid[["observed_spatial_anomaly", "prediction_spatial_anomaly"]].corr().iloc[0, 1])
    return {"hourly_mean_temporal_correlation": temporal, "within_hour_spatial_anomaly_correlation": spatial}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate_path = OUTPUT / "merra2_gmi_pm25_20190801_07_bth.nc"
    merra = download_merra2_gmi(candidate_path)

    ground = pd.read_parquet(GROUND)
    ground["time"] = pd.to_datetime(ground.geos_average_center_utc)
    ground = ground[
        ground.split.isin(["train", "validation"])
        & ground.time.between("2019-08-01 00:30", "2019-08-07 23:30")
        & ground.three_source_complete
    ].rename(columns={
        "pm25_as_stored": "observed",
        "longitude_as_stored": "longitude",
        "latitude_as_stored": "latitude",
    }).copy()
    if set(ground.split.unique()) != {"train", "validation"} or (ground.split == "test").any():
        raise RuntimeError("background screening must use train and validation stations only")

    with xr.open_dataset(GEOS, engine="h5netcdf") as opened:
        geos = opened.pm25_rh35_gcc.isel(lev=0, drop=True).sel(
            time=slice("2019-08-01 00:30", "2019-08-07 23:30")
        ).load()
    ground["geos_cf"] = sample(geos, ground)
    # Ground and GEOS are one-hour means centred at HH:30. MERRA2-GMI is
    # instantaneous at HH:00, so interpolation to HH:30 is a time-semantics
    # sensitivity match, not a conversion into an hourly average.
    ground["merra2_gmi"] = sample(merra.pm25, ground)

    peak_threshold = float(ground.loc[ground.split == "train", "observed"].quantile(.90))
    results = []
    fits = {}
    for product, column in (("GEOS-CF", "geos_cf"), ("MERRA2-GMI", "merra2_gmi")):
        fit = fit_affine(ground, column)
        fits[product] = fit
        ground[f"{column}_debiased"] = np.maximum(fit["slope"] * ground[column] + fit["intercept"], 0)
        for split in ("train", "validation"):
            part = ground[ground.split == split]
            for mode, pred_col in (("raw", column), ("train_affine_debiased", f"{column}_debiased")):
                row = {"product": product, "split": split, "mode": mode, **metric_values(part.observed, part[pred_col])}
                row.update(variation_metrics(part, pred_col))
                high = part[part.observed >= peak_threshold]
                high_metrics = metric_values(high.observed, high[pred_col])
                row.update({f"high_{key}": value for key, value in high_metrics.items() if key in {"records", "bias", "rmse", "mae"}})
                results.append(row)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(OUTPUT / "background_metrics_train_validation.csv", index=False)
    ground[["station_id", "time", "split", "observed", "geos_cf", "merra2_gmi",
            "geos_cf_debiased", "merra2_gmi_debiased"]].to_csv(
        OUTPUT / "station_background_samples.csv", index=False)
    manifest = {
        "selection_policy": "train stations fit affine correction; validation stations compare products; final test stations excluded",
        "period": ["2019-08-01 00:30 UTC", "2019-08-07 23:30 UTC"],
        "peak_threshold_from_training_station_p90_ug_m3": peak_threshold,
        "affine_fits": fits,
        "candidate_file": {
            "path": str(candidate_path), "bytes": candidate_path.stat().st_size,
            "sha256": file_sha256(candidate_path), "source": SOURCE,
            "dims": dict(merra.sizes), "pm25_min": float(merra.pm25.min()), "pm25_max": float(merra.pm25.max()),
        },
        "geos_file": {"path": str(GEOS), "bytes": GEOS.stat().st_size, "sha256": file_sha256(GEOS)},
        "record_counts": ground.groupby("split").size().to_dict(),
        "station_counts": ground.groupby("split").station_id.nunique().to_dict(),
        "test_records_used": 0,
        "limitations": [
            "station coordinate datum remains an exploratory approximate-WGS84 assumption",
            "MERRA2-GMI instantaneous HH:00 values were linearly sampled at HH:30; they are not one-hour averages",
            "seven-day screening does not establish month-scale superiority",
        ],
    }
    dump(OUTPUT / "screening_manifest.json", manifest)
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    main()
