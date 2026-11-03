"""Bounded diagnostics for the existing real PM2.5 fusion experiment.

This script does not change the fusion algorithms. It independently checks the
prepared inputs, re-computes simple baselines on a fixed 48-hour window, and
runs three validation-only single-factor neural probes.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import gstools as gs
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import xarray as xr

from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig, run_neural_reconstruction
from run_real_pm25_three_source import DATA_ROOT, ERA5, GEOS, GROUND, prepare_inputs
from run_real_pm25_training_optimization import subset


PROJECT = Path(__file__).resolve().parents[1]
ABLATION = PROJECT / "output" / "pm25_ablation_generalization_20261012_parallel"
RAW_GROUND = DATA_ROOT / "01_地面监测站" / "ground_pm25_201908_60stations.csv"


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metrics(observed, predicted) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed
    return {
        "records": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "correlation": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def vector_sample(field: xr.DataArray, frame: pd.DataFrame, time_shift=0, lon_shift=0.0, lat_shift=0.0):
    dim = "record"
    return field.interp(
        time=xr.DataArray((pd.to_datetime(frame.time) + pd.Timedelta(hours=time_shift)).values, dims=dim),
        lat=xr.DataArray(frame.latitude.to_numpy() + lat_shift, dims=dim),
        lon=xr.DataArray(frame.longitude.to_numpy() + lon_shift, dims=dim),
    ).values


def input_integrity(background, stations, output: Path) -> dict:
    aligned = pd.read_parquet(GROUND)
    raw = pd.read_csv(RAW_GROUND, dtype={"station_id": str})
    raw["time_as_stored"] = pd.to_datetime(raw.time_as_stored)
    check = aligned.merge(
        raw[["station_id", "time_as_stored", "pm25_as_stored", "longitude_as_stored", "latitude_as_stored"]],
        on=["station_id", "time_as_stored"], suffixes=("_aligned", "_raw"), validate="one_to_one")
    ground_differences = {
        column: float(np.nanmax(np.abs(check[f"{column}_aligned"] - check[f"{column}_raw"])))
        for column in ("pm25_as_stored", "longitude_as_stored", "latitude_as_stored")
    }
    complete = aligned[aligned.three_source_complete].copy()
    records = complete.rename(columns={"geos_average_center_utc": "time", "longitude_as_stored": "longitude",
        "latitude_as_stored": "latitude"})
    recomputed = vector_sample(background.pm25_rh35_gcc, records)
    interpolation_error = recomputed - complete.pm25_rh35_gcc.to_numpy()
    result = {
        "raw_ground_rows": len(raw), "aligned_rows": len(aligned), "one_to_one_join_rows": len(check),
        "raw_to_aligned_max_abs_difference": ground_differences,
        "aligned_geos_recomputed_max_abs_difference": float(np.nanmax(np.abs(interpolation_error))),
        "aligned_geos_recomputed_mean_abs_difference": float(np.nanmean(np.abs(interpolation_error))),
        "aligned_geos_finite_comparison_rows": int(np.isfinite(interpolation_error).sum()),
    }
    dump(output / "input_integrity.json", result)
    return result


def distribution_and_alignment(background, stations, output: Path) -> dict:
    sampled = vector_sample(background.pm25_rh35_gcc, stations)
    observed = stations.value.to_numpy(float)
    frame = pd.DataFrame({"observed": observed, "background": sampled})
    summary = {
        name: frame[name].describe(percentiles=[.01, .05, .5, .95, .99]).to_dict()
        for name in frame.columns
    }
    summary["paired"] = metrics(observed, sampled)
    dump(output / "distribution_summary.json", summary)

    shift_rows = []
    for shift in range(-8, 9):
        prediction = vector_sample(background.pm25_rh35_gcc, stations, time_shift=shift)
        valid = np.isfinite(prediction)
        shift_rows.append({"geos_time_shift_hours": shift, **metrics(observed[valid], prediction[valid])})
    pd.DataFrame(shift_rows).to_csv(output / "time_shift_diagnostic.csv", index=False)

    spatial_rows = []
    for label, dx, dy in [
        ("stored", 0, 0), ("east_0.05deg", .05, 0), ("west_0.05deg", -.05, 0),
        ("north_0.05deg", 0, .05), ("south_0.05deg", 0, -.05),
        ("east_0.10deg", .1, 0), ("north_0.10deg", 0, .1),
    ]:
        prediction = vector_sample(background.pm25_rh35_gcc, stations, lon_shift=dx, lat_shift=dy)
        valid = np.isfinite(prediction)
        spatial_rows.append({"coordinate_case": label, "lon_shift": dx, "lat_shift": dy,
                             **metrics(observed[valid], prediction[valid])})
    pd.DataFrame(spatial_rows).to_csv(output / "coordinate_shift_diagnostic.csv", index=False)
    return summary


def fit_linear(train_bg, train_stations):
    train = train_stations[train_stations.split == "train"]
    x = vector_sample(train_bg.pm25_rh35_gcc, train)
    y = train.value.to_numpy(float)
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    prediction = np.maximum(slope * x + intercept, 0)
    ss_total = np.sum((y - y.mean())**2)
    r2 = 1 - np.sum((y - prediction)**2) / ss_total
    return float(slope), float(intercept), float(r2), metrics(y, prediction)


def small_baselines(train_bg, train_stations, future_bg, future_stations, output: Path) -> pd.DataFrame:
    slope, intercept, train_r2, train_metrics = fit_linear(train_bg, train_stations)
    train_ids = set(train_stations.loc[train_stations.split == "train", "station_id"])
    test_ids = set(future_stations.loc[future_stations.split == "test", "station_id"])
    if train_ids & test_ids:
        raise RuntimeError("Train/test station leakage")
    cosine = np.cos(np.deg2rad(40.0))
    rows = []
    kriging_failures = []
    for stamp in sorted(future_stations.time.unique()):
        live = future_stations[(future_stations.split == "train") & (future_stations.time == stamp)]
        test = future_stations[(future_stations.split == "test") & (future_stations.time == stamp)]
        if len(live) < 3 or test.empty:
            continue
        train_xy = np.column_stack([live.longitude.to_numpy() * cosine, live.latitude.to_numpy()]) * 111.2
        test_xy = np.column_stack([test.longitude.to_numpy() * cosine, test.latitude.to_numpy()]) * 111.2
        distance = cdist(test_xy, train_xy)
        weights = 1 / np.maximum(distance, .01)**2
        weights /= weights.sum(axis=1, keepdims=True)
        bg_train = vector_sample(future_bg.pm25_rh35_gcc, live)
        bg_test = vector_sample(future_bg.pm25_rh35_gcc, test)
        corrected_train = np.maximum(slope * bg_train + intercept, 0)
        corrected_test = np.maximum(slope * bg_test + intercept, 0)
        predictions = {
            "linear": corrected_test,
            "pure_idw": weights @ live.value.to_numpy(float),
            "raw_background_residual_idw": bg_test + weights @ (live.value.to_numpy(float) - bg_train),
            "linear_residual_idw": corrected_test + weights @ (live.value.to_numpy(float) - corrected_train),
        }
        try:
            values = live.value.to_numpy(float)
            bins, gamma = gs.vario_estimate((train_xy[:, 0], train_xy[:, 1]), values)
            model = gs.Exponential(dim=2)
            model.fit_variogram(bins, gamma, nugget=True)
            krige = gs.krige.Ordinary(model, cond_pos=(train_xy[:, 0], train_xy[:, 1]), cond_val=values)
            predictions["ordinary_kriging"], _ = krige((test_xy[:, 0], test_xy[:, 1]), mesh_type="unstructured")
        except Exception as exc:
            kriging_failures.append({"time": str(stamp), "error": str(exc)})
        for method, predicted in predictions.items():
            for station_id, observed, value in zip(test.station_id, test.value, predicted):
                rows.append({"time": stamp, "station_id": station_id, "method": method,
                             "observed": observed, "predicted": float(value)})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "small_baseline_predictions.csv", index=False)
    common_times = set(predictions.loc[predictions.method == "ordinary_kriging", "time"])
    comparison = predictions[predictions.time.isin(common_times)]
    result = pd.DataFrame([{"method": method, **metrics(group.observed, group.predicted)}
                           for method, group in comparison.groupby("method")]).sort_values("rmse")
    result.to_csv(output / "small_baseline_comparison.csv", index=False)
    nearest = []
    live_locations = future_stations[future_stations.split == "train"].drop_duplicates("station_id")
    test_locations = future_stations[future_stations.split == "test"].drop_duplicates("station_id")
    a = np.column_stack([live_locations.longitude * cosine, live_locations.latitude]) * 111.2
    b = np.column_stack([test_locations.longitude * cosine, test_locations.latitude]) * 111.2
    nearest = cdist(b, a).min(axis=1)
    dump(output / "small_baseline_context.json", {
        "period": [str(future_stations.time.min()), str(future_stations.time.max())],
        "train_station_count": len(train_ids), "test_station_count": len(test_ids),
        "train_test_station_overlap": sorted(train_ids & test_ids),
        "linear_fit": {"slope": slope, "intercept": intercept, "training_r2": train_r2,
                       "training_metrics": train_metrics},
        "test_to_nearest_train_station_km": {"min": float(nearest.min()), "median": float(np.median(nearest)),
                                              "max": float(nearest.max())},
        "kriging_failed_hours": kriging_failures,
        "common_evaluation_records_per_method": int(len(comparison) / max(1, comparison.method.nunique())),
    })
    return result


def neural_log_audit(output: Path) -> dict:
    runs = pd.read_csv(ABLATION / "test_runs.csv")
    neural = runs[runs.seed != -1]
    curve_files = list((ABLATION / "logs").glob("*_curve.csv"))
    curves = pd.concat([pd.read_csv(path).assign(run=path.stem.removesuffix("_curve")) for path in curve_files], ignore_index=True)
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in (ABLATION / "configs").glob("*.json")]
    result = {
        "neural_runs": len(neural), "finite_losses": bool(np.isfinite(curves.loss).all()),
        "finite_validation_rmse": bool(np.isfinite(curves.validation_rmse).all()),
        "finite_gradient_norm": bool(np.isfinite(curves.gradient_norm).all()),
        "gradient_norm_above_clip_fraction": float((curves.gradient_norm > 1.0).mean()),
        "runs_best_at_max_epoch": int((neural.best_epoch == neural.epochs_completed).sum()),
        "runs_completed_max_30": int((neural.epochs_completed == 30).sum()),
        "validation_last_minus_previous5_mean": float(np.mean([
            group.validation_rmse.iloc[-1] - group.validation_rmse.iloc[-6]
            for _, group in curves.groupby("run") if len(group) >= 6])),
        "configuration_count": len(configs),
        "coarse_weight_values": sorted(set(item["coarse_weight"] for item in configs)),
        "observation_weight_values": sorted(set(item["observation_weight"] for item in configs)),
        "spectral_weight_values": sorted(set(item["spectral_weight"] for item in configs)),
        "peak_weight_values": sorted(set(item["peak_weight"] for item in configs)),
    }
    dump(output / "neural_log_audit.json", result)
    return result


def minimal_constraint_probe(background, era, stations, output: Path) -> pd.DataFrame:
    bg, aux, frame = subset(background, era, stations, "2019-08-01 00:30", "2019-08-02 23:30")
    usable = frame[frame.split.isin(["train", "validation"])]
    base = NeuralReconstructionConfig(
        backend="observation_guided_uno", epochs=12, hidden_channels=8, n_layers=2,
        scheduler="none", early_stopping_patience=0, gradient_clip_norm=1.0,
        mixed_precision=False, variable_family="pollutant_concentration", seed=43)
    variants = {
        "current_constraints": base,
        "coarse_weight_zero": replace(base, coarse_weight=0.0),
        "spectral_weight_zero": replace(base, spectral_weight=0.0),
    }
    rows = []
    for name, config in variants.items():
        started = perf_counter()
        result = run_neural_reconstruction(bg, usable, auxiliaries=[aux], config=config)
        report = result.report
        pd.DataFrame({"epoch": range(1, len(report["loss_history"]) + 1),
                      "loss": report["loss_history"],
                      "validation_rmse": report["validation_rmse_history"]}).to_csv(
                          output / f"probe_{name}_curve.csv", index=False)
        rows.append({"variant": name, "best_validation_rmse": report["training_control"]["best_validation_rmse"],
                     "best_epoch": report["training_control"]["best_epoch"],
                     "final_validation_rmse": report["validation_rmse_history"][-1],
                     "runtime_seconds": perf_counter() - started})
    table = pd.DataFrame(rows)
    table.to_csv(output / "minimal_constraint_probe.csv", index=False)
    return table


def main() -> None:
    output = PROJECT / "output" / "pm25_root_cause_diagnosis_20261012"
    output.mkdir(parents=True, exist_ok=True)
    background, era, stations, _ = prepare_inputs("month", 24)
    integrity = input_integrity(background, stations, output)
    distributions = distribution_and_alignment(background, stations, output)
    train_bg, _, train_stations = subset(background, era, stations, "2019-08-01 00:30", "2019-08-07 23:30")
    future_bg, _, future_stations = subset(background, era, stations, "2019-08-08 00:30", "2019-08-09 23:30")
    baselines = small_baselines(train_bg, train_stations, future_bg, future_stations, output)
    logs = neural_log_audit(output)
    probes = minimal_constraint_probe(background, era, stations, output)
    dump(output / "diagnostic_run_summary.json", {
        "input_integrity": integrity, "distribution": distributions,
        "small_baselines": baselines.to_dict("records"), "neural_logs": logs,
        "constraint_probes": probes.to_dict("records"),
        "scope": "read-only input audit, 48-hour traditional baselines, validation-only 48-hour neural probes",
    })
    print(output)


if __name__ == "__main__":
    main()
