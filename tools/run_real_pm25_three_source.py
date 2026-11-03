"""Run the fixed-split Beijing-Tianjin-Hebei PM2.5 three-source experiment.

This is an experiment driver, not a new reconstruction algorithm.  It keeps the
final test stations out of training and model selection, then evaluates them
only after the validation-based choice has been written to disk.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.core.schemas import EcoDataBundle
from ecodata.fusion.neural_experiment import evaluate_reconstruction, run_reconstruction_comparison
from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig, default_target_coordinates
from ecodata.optimization.downscaling import ConstrainedDownscaler, DownscalingConfig


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\codex\数据\京津冀_PM25融合测试_201908")
GROUND = DATA_ROOT / "05_对齐后的测试数据" / "diagnostic_ground_labels_Asia_Shanghai_UTC_plus_8" / "three_source_station_hourly.parquet"
SPLIT = DATA_ROOT / "05_对齐后的测试数据" / "station_split_40_10_10.csv"
GEOS = DATA_ROOT / "02_GEOS_CF模拟背景场" / "geos_cf_pm25_201908_bth.nc"
ERA5 = DATA_ROOT / "03_ERA5气象数据" / "era5_201908_bth.nc"
METHODS = ("bilinear", "qp", "fno", "multiscale_uno", "observation_guided_uno")


def _json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _choose_smoke_window(frame: pd.DataFrame, hours: int) -> pd.DatetimeIndex:
    counts = frame.groupby(["geos_average_center_utc", "split"]).size().unstack(fill_value=0)
    all_times = pd.date_range(frame.geos_average_center_utc.min(), frame.geos_average_center_utc.max(), freq="h")
    counts = counts.reindex(all_times, fill_value=0)
    best = None
    for start in range(len(all_times) - hours + 1):
        block = counts.iloc[start:start + hours]
        # Prefer windows with broad training coverage, then independent held-out coverage.
        score = (int(block.get("train", pd.Series(0, index=block.index)).sum()),
                 int(block.get("validation", pd.Series(0, index=block.index)).sum()),
                 int(block.get("test", pd.Series(0, index=block.index)).sum()))
        if best is None or score > best[0]:
            best = (score, all_times[start:start + hours])
    if best is None or best[0][0] == 0:
        raise RuntimeError("没有可用的连续小样本时间窗口")
    return pd.DatetimeIndex(best[1])


def prepare_inputs(scope: str, smoke_hours: int):
    frame = pd.read_parquet(GROUND)
    split = pd.read_csv(SPLIT, dtype={"station_id": str})
    frame["station_id"] = frame.station_id.astype(str)
    if set(split["split"]) != {"train", "validation", "test"}:
        raise RuntimeError("站点划分不是固定的40/10/10")
    expected = split.groupby("split").station_id.nunique().to_dict()
    if expected != {"test": 10, "train": 40, "validation": 10}:
        raise RuntimeError(f"站点划分数量异常：{expected}")
    observed_map = frame[["station_id", "split"]].drop_duplicates().sort_values("station_id")
    expected_map = split[["station_id", "split"]].drop_duplicates().sort_values("station_id")
    if not observed_map.reset_index(drop=True).equals(expected_map.reset_index(drop=True)):
        raise RuntimeError("对齐数据中的站点划分与固定划分文件不一致")

    frame["geos_average_center_utc"] = pd.to_datetime(frame.geos_average_center_utc)
    frame = frame.loc[
        frame.three_source_complete
        & np.isfinite(frame.pm25_as_stored)
        & np.isfinite(frame.longitude_as_stored)
        & np.isfinite(frame.latitude_as_stored)
    ].copy()
    if scope == "smoke":
        times = _choose_smoke_window(frame, smoke_hours)
    else:
        times = pd.date_range(frame.geos_average_center_utc.min(), frame.geos_average_center_utc.max(), freq="h")
    frame = frame[frame.geos_average_center_utc.isin(times)].copy()
    stations = frame.rename(columns={
        "geos_average_center_utc": "time",
        "longitude_as_stored": "longitude",
        "latitude_as_stored": "latitude",
        "pm25_as_stored": "value",
    })[["station_id", "time", "longitude", "latitude", "value", "split"]]
    stations["unit"] = "ug m-3"
    stations["quality_weight"] = 1.0
    stations.attrs = {
        "time_assumption": "source labels treated as Asia/Shanghai (UTC+8); preceding one-hour mean",
        "coordinate_assumption": "stored coordinates treated as approximately EPSG:4326 for exploratory testing; datum is not proven",
        "qa_assumption": "no instrument QA supplied; all finite records use equal reliability",
    }

    # h5netcdf keeps the Windows Unicode path intact; netCDF4 may pass it
    # through the active ANSI code page and fail before reading the file.
    with xr.open_dataset(GEOS, engine="h5netcdf") as opened:
        background = opened[["pm25_rh35_gcc"]].sel(time=times).isel(lev=0, drop=True).load()
    background.attrs.update(crs="EPSG:4326", experiment_coordinate_status="exploratory approximate WGS84")
    background.pm25_rh35_gcc.attrs.update(data_kind="continuous", physical_variable="PM2.5")
    background.time.attrs.update(
        standard_name="time",
        temporal_semantics="GEOS-CF one-hour mean centered at HH:30 UTC",
    )
    background.lat.attrs.update(standard_name="latitude", units="degrees_north")
    background.lon.attrs.update(standard_name="longitude", units="degrees_east")
    if background.sizes["time"] != len(times):
        raise RuntimeError("选定时段不在GEOS-CF中完整覆盖")

    era_times = times + pd.Timedelta(minutes=30)
    with xr.open_dataset(ERA5, engine="h5netcdf") as opened:
        era = opened[["t2m", "d2m", "u10", "v10", "sp", "tp"]].sel(valid_time=era_times).load()
    era = era.rename(valid_time="time", latitude="lat", longitude="lon")
    era = era.assign_coords(era5_valid_time=("time", era_times.values), time=times.values)
    era.attrs.update(
        crs="EPSG:4326",
        temporal_alignment="instantaneous fields and one-hour precipitation ending 30 minutes after GEOS center",
    )
    era.lat.attrs.update(standard_name="latitude", units="degrees_north")
    era.lon.attrs.update(standard_name="longitude", units="degrees_east")
    if not all(np.isfinite(era[name]).all() for name in ("t2m", "d2m", "u10", "v10", "sp", "tp")):
        raise RuntimeError("ERA5输入包含非有限值")
    return background, era, stations, times


def run_qp(background: xr.Dataset, train: pd.DataFrame):
    variable = "pm25_rh35_gcc"
    target_lat, target_lon = default_target_coordinates(background[variable])
    fields = []
    records = []
    started = perf_counter()
    for stamp in pd.to_datetime(background.time.values):
        rows = train[train.time == stamp].copy()
        if rows.station_id.nunique() < 3:
            raise RuntimeError(f"QP时刻 {stamp} 有效训练站少于3个")
        coarse = background.sel(time=[stamp]).copy(deep=True)
        coarse[variable].attrs.update(data_kind="continuous", physical_variable="PM2.5")
        observations = rows.assign(variable="PM2.5", split="train")
        result = ConstrainedDownscaler(DownscalingConfig(
            target_lat=tuple(target_lat), target_lon=tuple(target_lon), variable=variable, adaptive=False,
        )).apply(EcoDataBundle(coarse), observations)
        fields.append(result.data.dataset)
        records.append(result.record.as_dict())
    field = xr.concat(fields, dim="time")
    field.attrs.update(background.attrs)
    field[variable].attrs.update(background[variable].attrs)
    tiny_negative_count = int((field[variable].values < 0).sum())
    field[variable].values = np.maximum(field[variable].values, 0)
    return field, {"runtime_seconds": perf_counter() - started, "per_time_records": records,
                   "numerical_negative_values_clipped": tiny_negative_count}


def _save_field(field: xr.Dataset, path: Path) -> dict:
    field.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as check:
        loaded = check.load()
    variable = "pm25_rh35_gcc"
    finite = bool(np.isfinite(loaded[variable]).all())
    nonnegative = bool((loaded[variable] >= 0).all())
    valid = finite and nonnegative and loaded[variable].dims == ("time", "lat", "lon")
    if not valid:
        raise RuntimeError(f"输出复读检查失败：{path.name}")
    return {
        "file": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size,
        "dims": dict(loaded.sizes), "finite": finite, "nonnegative": nonnegative,
        "minimum": float(loaded[variable].min()), "maximum": float(loaded[variable].max()),
    }


def _metric_columns(prefix: str, metrics: dict) -> dict:
    names = {
        "rmse": "rmse", "mae": "mae", "bias": "bias", "station_correlation": "station_correlation",
        "validation_peak_error": "peak_error", "validation_station_count": "station_count",
        "validation_record_count": "record_count",
    }
    return {f"{prefix}_{output_name}": metrics.get(source_name)
            for source_name, output_name in names.items()}


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# 京津冀2019年8月 PM2.5 真实三源算法跑通测试报告", "",
        f"- 实验状态：**{summary['overall_status']}**", f"- 范围：{summary['scope']}",
        f"- 时间：{summary['time_start']} 至 {summary['time_end']} UTC，共 {summary['hours']} 小时", "",
        "## 数据与对齐", "",
        "- 地面站标签按 Asia/Shanghai (UTC+8) 解释，代表整点前1小时平均。",
        "- GEOS-CF 使用 HH:30 UTC 中心的1小时平均。",
        "- ERA5 瞬时量使用 valid_time；tp 为截止到 valid_time 的1小时累积，与GEOS中心时间相差30分钟。",
        "- 只使用真实重叠时间，不补齐月初超出背景场的时刻。",
        "- 坐标基准尚未获得直接证据；本次仅按原始经纬度近似 EPSG:4326 进行探索性跑通。", "",
        "## 防止泄漏", "",
        "- 40个训练站用于优化；10个验证站用于方法选择。",
        "- 选中方法先写入 model_selection.json，然后才计算10个最终测试站指标。",
        "- 没有高分辨率全场真值，因此不报告、不伪造全场RMSE。", "",
        "## 算法结果", "",
        "| 方法 | 状态 | 验证RMSE | 测试RMSE | 运行秒数 |", "|---|---:|---:|---:|---:|",
    ]
    for row in summary["algorithms"]:
        lines.append(f"| {row['method']} | {row['status']} | {row.get('validation_rmse', '')} | {row.get('test_rmse', '')} | {row.get('seconds', '')} |")
    lines += ["", "## 结论与限制", "",
              f"- 基于验证站冻结选择的方法：`{summary.get('selected_method')}`。",
              "- PASS 表示程序运行并且NetCDF可独立复读；WARN表示可运行但有数据语义、收敛或精度风险；FAIL表示未完成。",
              "- 本轮是可运行性测试，不是精度证明，也不包含卫星AOT。"]
    (output / "真实数据算法跑通测试报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "month"), default="smoke")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.hours < 24 or args.hours > 48:
        raise SystemExit("--hours 必须在24到48之间")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or PROJECT / "output" / f"real_pm25_three_source_{args.scope}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "netcdf").mkdir()
    (output / "training_logs").mkdir()
    errors = [
        {
            "status": "RESOLVED",
            "issue": "netCDF4 backend could not open the Windows Unicode data path",
            "fix": "experiment input reading uses h5netcdf; data and algorithms are unchanged",
        },
        {
            "status": "RESOLVED",
            "issue": "OSQP output contained floating-point negative roundoff near -1e-21",
            "fix": "clip only negative numerical residue to zero before output validation",
        },
    ]
    started = perf_counter()

    background, era, stations, times = prepare_inputs(args.scope, args.hours)
    train_validation = stations[stations.split.isin(["train", "validation"])].copy()
    test = stations[stations.split == "test"].copy()
    config = NeuralReconstructionConfig(
        epochs=args.epochs, seed=args.seed, hidden_channels=args.hidden_channels,
        n_layers=args.layers, validation_station_count=10,
    )
    experiment = run_reconstruction_comparison(
        background, train_validation, auxiliaries=[era], evaluation_truth=None,
        config=config, methods=("bilinear", "fno", "multiscale_uno", "observation_guided_uno"),
    )
    try:
        qp_field, qp_report = run_qp(background, stations[stations.split == "train"].copy())
        qp_predictions, qp_metrics = evaluate_reconstruction(qp_field, stations[stations.split == "validation"])
        experiment["fields"]["qp"] = qp_field
        experiment["predictions"] = pd.concat([experiment["predictions"], qp_predictions.assign(method="qp")], ignore_index=True)
        qp_metrics["seconds"] = qp_report["runtime_seconds"]
        experiment["comparison"] = pd.concat([experiment["comparison"], pd.DataFrame([{"method": "qp", "status": "PASS", **qp_metrics}])], ignore_index=True)
        experiment["report"]["baselines"]["qp"] = {"metrics": qp_metrics, "diagnostics": qp_report}
    except Exception as exc:  # Keep the remaining methods auditable when this optional baseline fails.
        errors.append({"method": "qp", "error": f"{type(exc).__name__}: {exc}"})
        experiment["comparison"] = pd.concat([experiment["comparison"], pd.DataFrame([{"method": "qp", "status": "FAIL", "reason": str(exc)}])], ignore_index=True)

    output_checks = {}
    for method, field in experiment["fields"].items():
        output_checks[method] = _save_field(field, output / "netcdf" / f"{method}.nc")
        details = experiment["report"]["baselines"].get(method, {})
        history = details.get("diagnostics", {}).get("loss_history", []) if isinstance(details, dict) else []
        if history:
            pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "loss": history}).to_csv(
                output / "training_logs" / f"{method}_loss.csv", index=False)

    comparison = experiment["comparison"].copy()
    valid = comparison[(comparison.status != "FAIL") & np.isfinite(pd.to_numeric(comparison.rmse, errors="coerce"))]
    if valid.empty:
        raise RuntimeError("没有任何方法完成验证站评价")
    selected = valid.sort_values("rmse").iloc[0].method
    selection = {
        "selected_method": selected, "criterion": "lowest validation-station RMSE",
        "test_metrics_seen_during_selection": False, "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json_dump(output / "model_selection.json", selection)

    test_rows = []
    result_rows = []
    for row in comparison.to_dict("records"):
        method = row["method"]
        merged = {"method": method, "status": row.get("status", "FAIL"), "seconds": row.get("seconds")}
        if method in experiment["fields"]:
            predictions, metrics = evaluate_reconstruction(experiment["fields"][method], test)
            test_rows.append(predictions.assign(method=method))
            merged.update(_metric_columns("validation", row), **_metric_columns("test", metrics))
            # Every completed method is WARN because CRS remains an explicit exploratory assumption.
            merged["status"] = "WARN"
        else:
            merged["reason"] = row.get("reason")
        result_rows.append(merged)
    comparison_out = pd.DataFrame(result_rows)
    validation_predictions = experiment["predictions"]
    test_predictions = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    comparison_out.to_csv(output / "algorithm_comparison.csv", index=False)
    validation_predictions.to_csv(output / "validation_station_predictions.csv", index=False)
    test_predictions.to_csv(output / "final_test_station_predictions.csv", index=False)

    manifest = [{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in (GROUND, SPLIT, GEOS, ERA5)]
    parameters = {
        "scope": args.scope, "hours": len(times), "time_start": times.min(), "time_end": times.max(),
        "seed": args.seed, "neural_config": asdict(config),
        "station_split": {"train": 40, "validation": 10, "test": 10},
        "training_uses_test": False, "normalization_uses_test": False, "selection_uses_test": False,
        "coordinate_status": "exploratory: original station coordinates treated as approximately EPSG:4326",
        "full_field_truth_available": False, "full_field_rmse_reported": False,
    }
    _json_dump(output / "parameters_and_data_versions.json", {"parameters": parameters, "inputs": manifest})
    _json_dump(output / "output_reread_checks.json", output_checks)
    _json_dump(output / "error_and_fix_log.json", errors)
    _json_dump(output / "experiment_report.json", experiment["report"])
    algorithms = comparison_out.replace({np.nan: None}).to_dict("records")
    overall = "FAIL" if not any(row["status"] != "FAIL" for row in algorithms) else "WARN"
    summary = {
        "overall_status": overall, "scope": args.scope, "time_start": times.min(), "time_end": times.max(),
        "hours": len(times), "selected_method": selected, "algorithms": algorithms,
        "runtime_seconds": perf_counter() - started,
    }
    _json_dump(output / "run_summary.json", summary)
    write_report(output, summary)
    print(str(output))


if __name__ == "__main__":
    main()
