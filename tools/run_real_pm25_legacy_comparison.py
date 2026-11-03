"""Compare the previous one-epoch training settings on the frozen real-data split."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig
from run_real_pm25_three_source import prepare_inputs
from run_real_pm25_training_optimization import BACKENDS, infer_once, metrics_for, subset, train_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()

    background, era, stations, _ = prepare_inputs("month", 24)
    train_bg, train_era, train_stations = subset(
        background, era, stations, "2019-08-01 00:30", "2019-08-07 23:30")
    future_bg, future_era, future_stations = subset(
        background, era, stations, "2019-08-08 00:30", "2019-08-14 23:30")

    optimized = pd.read_csv(output / "future_period_comparison.csv").set_index("method")
    rows = []
    for backend in BACKENDS:
        checkpoint = output / "checkpoints" / f"{backend}_legacy_1epoch.pt"
        config = NeuralReconstructionConfig(
            epochs=1,
            hidden_channels=8,
            n_layers=2,
            backend=backend,
            seed=20261012,
            optimizer="adam",
            scheduler="none",
            early_stopping_patience=0,
            gradient_clip_norm=None,
            mixed_precision=False,
            save_checkpoint_path=str(checkpoint),
            variable_family="pollutant_concentration",
        )
        trained, train_seconds = train_once(
            train_bg, train_era, train_stations, config, output, f"{backend}_legacy_1epoch")
        inferred, inference_seconds, netcdf = infer_once(
            future_bg,
            future_era,
            future_stations,
            config,
            checkpoint,
            output,
            f"{backend}_legacy_1epoch_future",
        )
        predictions, metrics = metrics_for(inferred.field, future_stations, "test")
        predictions.to_csv(
            output / "predictions" / f"{backend}_legacy_1epoch_future_test.csv", index=False)
        current = optimized.loc[backend]
        rows.append({
            "method": backend,
            "legacy_optimizer": "Adam",
            "legacy_epochs": 1,
            "legacy_test_rmse": metrics["rmse"],
            "optimized_test_rmse": current.future_test_rmse,
            "rmse_improvement": metrics["rmse"] - current.future_test_rmse,
            "legacy_test_mae": metrics["mae"],
            "optimized_test_mae": current.future_test_mae,
            "legacy_test_bias": metrics["bias"],
            "optimized_test_bias": current.future_test_bias,
            "legacy_training_seconds": train_seconds,
            "optimized_training_seconds": current.training_seconds,
            "legacy_inference_seconds": inference_seconds,
            "optimized_inference_seconds": current.direct_inference_seconds,
            "legacy_parameter_delta_l2": trained.report["training_control"]["parameter_delta_l2"],
            "legacy_checkpoint": str(checkpoint),
            "legacy_netcdf": str(netcdf),
        })

    table = pd.DataFrame(rows)
    table.to_csv(output / "legacy_one_epoch_comparison.csv", index=False)

    report_path = output / "真实PM25训练优化报告.md"
    lines = ["", "## 与旧版1轮训练的同条件对照", "",
             "| 方法 | 旧版1轮RMSE | 优化训练RMSE | RMSE降低 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['legacy_test_rmse']:.3f} | "
            f"{row['optimized_test_rmse']:.3f} | {row['rmse_improvement']:.3f} |")
    lines += ["", "- 对照使用完全相同的训练日期、未来测试日期、40/10/10站点划分、网络规模和随机种子。",
              "- 旧版模式使用Adam、1轮、无调度器、无早停、无梯度裁剪；优化模式使用AdamW与验证站选择的调度器。",
              "- 最终测试站只在训练方案冻结后用于这一轮评价。",
              "- 少量微调没有改善独立测试RMSE，因此不把微调描述为成功提升。"]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
