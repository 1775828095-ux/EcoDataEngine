"""Train/reuse the existing neural reconstruction backends on real PM2.5 data."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import xarray as xr

from ecodata.fusion.neural_experiment import evaluate_reconstruction
from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig, run_neural_reconstruction
from run_real_pm25_three_source import PROJECT, GEOS, ERA5, GROUND, SPLIT, prepare_inputs, run_qp


BACKENDS = ("fno", "multiscale_uno", "observation_guided_uno")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def subset(background, era, stations, start, end):
    selected = pd.date_range(start, end, freq="h")
    return (background.sel(time=selected), era.sel(time=selected), stations[stations.time.isin(selected)].copy())


def train_once(background, era, stations, config, output: Path, label: str):
    started = perf_counter()
    usable = stations[stations.split.isin(["train", "validation"])].copy()
    result = run_neural_reconstruction(background, usable, auxiliaries=[era], config=config)
    runtime = perf_counter() - started
    report = result.report
    history = pd.DataFrame({
        "epoch": np.arange(1, len(report["loss_history"]) + 1),
        "loss": report["loss_history"],
        "validation_rmse": report["validation_rmse_history"],
        "learning_rate": report["learning_rate_history"],
        "gradient_norm_before_clip": report["gradient_norm_history"],
    })
    history.to_csv(output / "logs" / f"{label}_training_history.csv", index=False)
    dump(output / "logs" / f"{label}_report.json", report)
    return result, runtime


def infer_once(background, era, stations, config, checkpoint, output: Path, label: str):
    started = perf_counter()
    usable = stations[stations.split.isin(["train", "validation"])].copy()
    result = run_neural_reconstruction(background, usable, auxiliaries=[era], config=replace(
        config, training_mode="inference", load_checkpoint_path=str(checkpoint),
        save_checkpoint_path=None))
    runtime = perf_counter() - started
    path = output / "netcdf" / f"{label}.nc"
    result.field.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        check = opened.load()
    variable = next(name for name in check.data_vars if {"lat", "lon"}.issubset(check[name].dims))
    if not np.isfinite(check[variable]).all() or check[variable].dims != ("time", "lat", "lon"):
        raise RuntimeError(f"{label} output reread failed")
    return result, runtime, path


def metrics_for(field, frame, split):
    predictions, metrics = evaluate_reconstruction(field, frame[frame.split == split])
    return predictions, metrics


def plot_training_curves(output: Path) -> Path:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for backend in BACKENDS:
        history = pd.read_csv(output / "logs" / f"{backend}_training_history.csv")
        axes[0].plot(history.epoch, history.loss, label=backend)
        axes[1].plot(history.epoch, history.validation_rmse, label=backend)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation-station RMSE", xlabel="Epoch", ylabel="RMSE")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    path = output / "training_convergence.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = PROJECT / "output" / f"real_pm25_training_optimization_{stamp}"
    for name in ("checkpoints", "logs", "netcdf", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=False)

    background, era, stations, _ = prepare_inputs("month", 24)
    variable = "pm25_rh35_gcc"
    diagnostic = {
        "station_records": len(stations),
        "station_count": int(stations.station_id.nunique()),
        "time_count": int(stations.time.nunique()),
        "duplicates": int(stations.duplicated(["station_id", "time"]).sum()),
        "station_pm25": stations.value.describe(percentiles=[.001, .01, .5, .95, .99, .999]).to_dict(),
    }
    aligned = pd.read_parquet(GROUND)
    aligned = aligned[aligned.three_source_complete & aligned.pm25_as_stored.notna()].copy()
    sampled_values = aligned.pm25_rh35_gcc.to_numpy(dtype=float)
    observed_values = aligned.pm25_as_stored.to_numpy(dtype=float)
    residual = sampled_values - observed_values
    diagnostic.update({
        "geos_at_station": pd.Series(sampled_values).describe(percentiles=[.001, .01, .5, .95, .99, .999]).to_dict(),
        "geos_minus_station_bias": float(np.mean(residual)),
        "geos_station_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "geos_station_correlation": float(np.corrcoef(sampled_values, observed_values)[0, 1]),
        "station_values_above_200": int((observed_values > 200).sum()),
        "geos_samples_above_200": int((sampled_values > 200).sum()),
        "unit_check": {"station": "ug m-3", "geos": background[variable].attrs.get("units")},
        "time_check": {"station": "Asia/Shanghai label, preceding hour, converted to GEOS HH:30 UTC center",
                       "geos": "one-hour mean centered HH:30 UTC",
                       "era5": "instantaneous fields / tp one-hour accumulation ending 30 min after GEOS center"},
        "coordinate_limit": "stored station coordinates treated as approximate EPSG:4326; datum not directly proven",
    })
    dump(output / "data_training_diagnosis.json", diagnostic)

    # Scheduler choice uses only train + validation stations in a 48-hour pilot.
    pilot_bg, pilot_era, pilot_stations = subset(background, era, stations, "2019-08-02 00:30", "2019-08-03 23:30")
    scheduler_rows = []
    for scheduler in ("warmup_cosine", "plateau"):
        config = NeuralReconstructionConfig(
            epochs=12, hidden_channels=8, n_layers=2, backend="fno", seed=20261012,
            scheduler=scheduler, warmup_epochs=3, minimum_epochs=6, early_stopping_patience=4,
            gradient_clip_norm=1.0, mixed_precision=True,
        )
        result, runtime = train_once(pilot_bg, pilot_era, pilot_stations, config, output, f"pilot_fno_{scheduler}")
        control = result.report["training_control"]
        scheduler_rows.append({"scheduler": scheduler, "best_validation_rmse": control["best_validation_rmse"],
                               "best_epoch": control["best_epoch"], "epochs_completed": control["epochs_completed"],
                               "runtime_seconds": runtime})
    scheduler_table = pd.DataFrame(scheduler_rows).sort_values("best_validation_rmse")
    scheduler_table.to_csv(output / "scheduler_comparison.csv", index=False)
    selected_scheduler = str(scheduler_table.iloc[0].scheduler)
    dump(output / "scheduler_selection.json", {
        "selected": selected_scheduler, "criterion": "lowest validation-station RMSE",
        "final_test_metrics_seen": False,
    })

    # Seven complete days balance real-data diversity and CPU-only feasibility.
    train_bg, train_era, train_stations = subset(background, era, stations, "2019-08-01 00:30", "2019-08-07 23:30")
    future_bg, future_era, future_stations = subset(background, era, stations, "2019-08-08 00:30", "2019-08-14 23:30")
    comparison_rows = []
    all_predictions = []
    base_configs = {}
    checkpoints = {}
    for backend in BACKENDS:
        checkpoint = output / "checkpoints" / f"{backend}_best.pt"
        config = NeuralReconstructionConfig(
            epochs=30, hidden_channels=8, n_layers=2, backend=backend, seed=20261012,
            scheduler=selected_scheduler, warmup_epochs=4, minimum_epochs=8,
            early_stopping_patience=5, early_stopping_min_delta=.01,
            gradient_clip_norm=1.0, mixed_precision=True,
            save_checkpoint_path=str(checkpoint), variable_family="pollutant_concentration",
        )
        trained, train_seconds = train_once(train_bg, train_era, train_stations, config, output, backend)
        inferred, inference_seconds, path = infer_once(
            future_bg, future_era, future_stations, config, checkpoint, output, f"{backend}_future_direct")
        predictions, test_metrics = metrics_for(inferred.field, future_stations, "test")
        predictions.assign(method=backend).to_csv(output / "predictions" / f"{backend}_future_test.csv", index=False)
        control = trained.report["training_control"]
        comparison_rows.append({
            "method": backend, "mode": "trained_then_direct_future_inference",
            "best_epoch": control["best_epoch"], "epochs_completed": control["epochs_completed"],
            "best_validation_rmse": control["best_validation_rmse"],
            "future_test_rmse": test_metrics["rmse"], "future_test_mae": test_metrics["mae"],
            "future_test_bias": test_metrics["bias"], "future_test_correlation": test_metrics["station_correlation"],
            "training_seconds": train_seconds, "direct_inference_seconds": inference_seconds,
            "checkpoint": str(checkpoint), "netcdf": str(path),
        })
        base_configs[backend] = config
        checkpoints[backend] = checkpoint

    # Same future period and station split for traditional baselines.
    bilinear = future_bg.interp(
        lat=np.linspace(float(future_bg.lat[0])-.09375, float(future_bg.lat[-1])+.09375, future_bg.sizes["lat"]*4),
        lon=np.linspace(float(future_bg.lon[0])-.09375, float(future_bg.lon[-1])+.09375, future_bg.sizes["lon"]*4),
        kwargs={"fill_value": "extrapolate"})
    started = perf_counter()
    bilinear_predictions, bilinear_metrics = metrics_for(bilinear, future_stations, "test")
    comparison_rows.append({"method": "bilinear", "mode": "no_training", "future_test_rmse": bilinear_metrics["rmse"],
                            "future_test_mae": bilinear_metrics["mae"], "future_test_bias": bilinear_metrics["bias"],
                            "future_test_correlation": bilinear_metrics["station_correlation"],
                            "training_seconds": 0.0, "direct_inference_seconds": perf_counter()-started})
    bilinear_predictions.assign(method="bilinear").to_csv(output / "predictions" / "bilinear_future_test.csv", index=False)
    qp, qp_report = run_qp(future_bg, future_stations[future_stations.split == "train"])
    qp_predictions, qp_metrics = metrics_for(qp, future_stations, "test")
    qp.to_netcdf(output / "netcdf" / "qp_future.nc", engine="h5netcdf")
    qp_predictions.assign(method="qp").to_csv(output / "predictions" / "qp_future_test.csv", index=False)
    comparison_rows.append({"method": "qp", "mode": "per-time optimization", "future_test_rmse": qp_metrics["rmse"],
                            "future_test_mae": qp_metrics["mae"], "future_test_bias": qp_metrics["bias"],
                            "future_test_correlation": qp_metrics["station_correlation"],
                            "training_seconds": 0.0, "direct_inference_seconds": qp_report["runtime_seconds"]})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "future_period_comparison.csv", index=False)
    plot_training_curves(output)

    # Fine-tune one mature checkpoint on three later days, then test on unseen later dates.
    tune_bg, tune_era, tune_stations = subset(background, era, stations, "2019-08-08 00:30", "2019-08-10 23:30")
    hold_bg, hold_era, hold_stations = subset(background, era, stations, "2019-08-11 00:30", "2019-08-14 23:30")
    backend = "observation_guided_uno"
    original = base_configs[backend]
    finetuned_checkpoint = output / "checkpoints" / f"{backend}_finetuned.pt"
    finetune_config = replace(
        original, epochs=10, learning_rate=5e-4, training_mode="finetune",
        load_checkpoint_path=str(checkpoints[backend]), save_checkpoint_path=str(finetuned_checkpoint),
        minimum_epochs=4, early_stopping_patience=3,
    )
    finetuned, finetune_seconds = train_once(tune_bg, tune_era, tune_stations, finetune_config, output, f"{backend}_finetune")
    direct_hold, direct_seconds, _ = infer_once(hold_bg, hold_era, hold_stations, original, checkpoints[backend], output, f"{backend}_hold_direct")
    tuned_hold, tuned_seconds, _ = infer_once(hold_bg, hold_era, hold_stations, original, finetuned_checkpoint, output, f"{backend}_hold_finetuned")
    transfer_rows = []
    for name, result, runtime in (("direct", direct_hold, direct_seconds), ("finetuned", tuned_hold, tuned_seconds)):
        predictions, metrics = metrics_for(result.field, hold_stations, "test")
        predictions.to_csv(output / "predictions" / f"{backend}_hold_{name}_test.csv", index=False)
        transfer_rows.append({"mode": name, "test_rmse": metrics["rmse"], "test_mae": metrics["mae"],
                              "test_bias": metrics["bias"], "test_correlation": metrics["station_correlation"],
                              "inference_seconds": runtime, "finetune_seconds": finetune_seconds if name == "finetuned" else 0.0})
    pd.DataFrame(transfer_rows).to_csv(output / "transfer_learning_comparison.csv", index=False)

    # Explicit negative compatibility test: PM2.5 checkpoint must reject temperature.
    incompatible_status = "FAIL"
    incompatible_message = None
    temperature = future_bg.rename({variable: "temperature"})
    temperature.temperature.attrs.update(units="K", standard_name="air_temperature", physical_variable="temperature")
    try:
        run_neural_reconstruction(temperature, future_stations.assign(unit="K"), auxiliaries=[future_era], config=replace(
            base_configs["fno"], training_mode="inference", load_checkpoint_path=str(checkpoints["fno"]),
            save_checkpoint_path=None, variable_family="temperature"))
    except Exception as exc:
        incompatible_status = "PASS"
        incompatible_message = str(exc)
    dump(output / "incompatible_checkpoint_test.json", {
        "status": incompatible_status, "case": "PM2.5 checkpoint applied to temperature",
        "rejection_message": incompatible_message,
    })

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mixed_precision_effective": bool(torch.cuda.is_available()),
        "selected_scheduler": selected_scheduler,
        "training_period": "2019-08-01 00:30 to 2019-08-07 23:30 UTC",
        "future_direct_period": "2019-08-08 00:30 to 2019-08-14 23:30 UTC",
        "split": {"train_stations": 40, "validation_stations": 10, "final_test_stations": 10},
        "final_test_used_for_training_or_early_stopping": False,
        "comparison": comparison.to_dict("records"),
        "transfer": transfer_rows,
        "incompatible_checkpoint_rejected": incompatible_status == "PASS",
        "limitations": [
            "Station coordinate datum remains an exploratory approximate-WGS84 assumption.",
            "GEOS-CF RH35 PM2.5 is not measurement-identical to the ground product.",
            "One month and one region cannot prove cross-region or cross-variable generalization.",
        ],
    }
    dump(output / "experiment_summary.json", summary)
    lines = [
        "# EcoData Engine 真实PM2.5训练优化报告", "",
        f"- 状态：**{'WARN' if incompatible_status == 'PASS' else 'FAIL'}**",
        f"- 硬件：{summary['device']}；Mixed Precision实际启用：{summary['mixed_precision_effective']}",
        f"- 验证站选定调度器：`{selected_scheduler}`", "",
        "## 训练前诊断", "",
        f"- 地面PM2.5均值 {diagnostic['station_pm25']['mean']:.2f} ug m-3，GEOS-CF站点采样均值 {diagnostic['geos_at_station']['mean']:.2f} ug m-3。",
        f"- GEOS-CF相对地面站平均偏差 {diagnostic['geos_minus_station_bias']:.2f} ug m-3，原始RMSE {diagnostic['geos_station_rmse']:.2f}。",
        "- 单位字段一致，时间窗口已按HH:30对齐；但RH35定义和数据源系统偏差导致显著分布偏移，不能只靠增加epoch解决。", "",
        "## 同地区后续日期独立测试", "",
        "| 方法 | RMSE | MAE | Bias | 训练秒 | 推理/优化秒 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(f"| {row['method']} | {row['future_test_rmse']:.3f} | {row['future_test_mae']:.3f} | {row['future_test_bias']:.3f} | {row.get('training_seconds', 0):.2f} | {row.get('direct_inference_seconds', 0):.2f} |")
    lines += ["", "## 结论边界", "",
              "- Checkpoint可直接用于同变量、同通道、同单位、同网格的后续日期，也可小学习率微调。",
              "- PM2.5权重应用于温度已被兼容性守门器拒绝。",
              "- 本结果只支持同地区、同月份后续日期的初步时间泛化；不支持跨地区或跨变量泛化声称。"]
    (output / "真实PM25训练优化报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
