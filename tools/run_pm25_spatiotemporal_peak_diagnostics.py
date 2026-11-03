"""Reproducible traditional PM2.5 background diagnostics (Aug 2019).

This script deliberately excludes final-test stations.  Parameters are fitted
only with fixed training stations during 1--14 August; validation covers
15--31 August.  No neural model is trained or modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\codex\数据\京津冀_PM25融合测试_201908")
GROUND = DATA_ROOT / "05_对齐后的测试数据" / "diagnostic_ground_labels_Asia_Shanghai_UTC_plus_8" / "three_source_station_hourly.parquet"
SAMPLES = PROJECT / "output" / "pm25_background_month_screening_20261012" / "monthly_station_background_samples.parquet"
OUTPUT = PROJECT / "output" / "pm25_spatiotemporal_peak_diagnostics_20261012"
FIT_START = pd.Timestamp("2019-08-01 00:30")
FIT_END = pd.Timestamp("2019-08-14 23:30")
EVAL_START = pd.Timestamp("2019-08-15 00:30")
EVAL_END = pd.Timestamp("2019-08-31 14:30")
SEEDS = (17, 43, 71)
COUNTS = (40, 20, 10)
BUFFERS_KM = (5.0, 10.0, 20.0)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metrics(obs, pred) -> dict:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    err = pred - obs
    return {
        "records": int(len(obs)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "correlation": float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan,
    }


def variation_metrics(frame: pd.DataFrame) -> dict:
    hourly = frame.groupby("time")[["observed", "predicted"]].mean()
    temporal = hourly.observed.corr(hourly.predicted)
    work = frame.copy()
    work["obs_anomaly"] = work["observed"] - work.groupby("time")["observed"].transform("mean")
    work["pred_anomaly"] = work["predicted"] - work.groupby("time")["predicted"].transform("mean")
    return {
        "temporal_correlation": float(temporal),
        "spatial_anomaly_correlation": float(work.obs_anomaly.corr(work.pred_anomaly)),
    }


def fit_linear(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    clean = frame.dropna(subset=columns + ["observed"])
    x = clean[columns].to_numpy(float)
    return np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), clean.observed.to_numpy(float), rcond=None)[0]


def apply_linear(frame: pd.DataFrame, columns: list[str], beta: np.ndarray) -> np.ndarray:
    return np.maximum(frame[columns].to_numpy(float) @ beta[:-1] + beta[-1], 0.0)


def xy_km(frame: pd.DataFrame) -> np.ndarray:
    lat0 = np.deg2rad(39.5)
    return np.column_stack([frame.longitude.to_numpy(float) * 111.32 * np.cos(lat0), frame.latitude.to_numpy(float) * 110.57])


def idw_weights(source_xy: np.ndarray, target_xy: np.ndarray, power: float = 2.0) -> np.ndarray:
    d = cdist(target_xy, source_xy)
    w = 1.0 / np.maximum(d, 0.01) ** power
    return w / w.sum(axis=1, keepdims=True)


def fit_variogram_model(fit: pd.DataFrame, station_ids: list[str]) -> tuple[gs.Exponential, dict]:
    """Fit one training-period spatial-anomaly variogram per station configuration."""
    subset = fit[fit.station_id.isin(station_ids)]
    loc = subset.drop_duplicates("station_id").set_index("station_id")
    ids = [s for s in station_ids if s in loc.index]
    coords = xy_km(loc.loc[ids])
    pair_d = squareform(pdist(coords))
    upper = np.triu_indices(len(ids), 1)
    distances, semivariances = [], []
    pivot = subset.pivot_table(index="time", columns="station_id", values="observed")
    for _, row in pivot.reindex(columns=ids).iterrows():
        values = row.to_numpy(float)
        valid_pair = np.isfinite(values[upper[0]]) & np.isfinite(values[upper[1]])
        if valid_pair.any():
            distances.extend(pair_d[upper][valid_pair])
            semivariances.extend(0.5 * (values[upper[0]][valid_pair] - values[upper[1]][valid_pair]) ** 2)
    distances = np.asarray(distances)
    semivariances = np.asarray(semivariances)
    edges = np.linspace(0, np.nanmax(distances), 13)
    centres, gamma = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (distances >= lo) & (distances < hi)
        if mask.sum() >= 20:
            centres.append(float(distances[mask].mean()))
            gamma.append(float(semivariances[mask].mean()))
    model = gs.Exponential(dim=2)
    fallback = False
    try:
        model.fit_variogram(np.asarray(centres), np.asarray(gamma), nugget=True)
    except Exception:
        fallback = True
        variance = float(np.nanvar(subset.observed))
        model = gs.Exponential(dim=2, var=max(variance, 1.0), len_scale=max(float(np.nanmedian(distances)), 10.0), nugget=0.05 * max(variance, 1.0))
    return model, {"bins": centres, "gamma": gamma, "model": repr(model), "fallback": fallback}


def ordinary_kriging_weights(model: gs.Exponential, source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """Ordinary-kriging weights for a fixed geometry and GSTools covariance."""
    covariance = model.covariance(cdist(source_xy, source_xy))
    covariance.flat[:: len(source_xy) + 1] += float(model.nugget) + 1e-8
    system = np.block([[covariance, np.ones((len(source_xy), 1))], [np.ones((1, len(source_xy))), np.zeros((1, 1))]])
    rhs = np.vstack([model.covariance(cdist(source_xy, target_xy)), np.ones((1, len(target_xy)))])
    return np.linalg.solve(system, rhs)[:-1].T


def prepare_data() -> pd.DataFrame:
    samples = pd.read_parquet(SAMPLES)
    coords = pd.read_parquet(GROUND)[["station_id", "longitude_as_stored", "latitude_as_stored"]].drop_duplicates("station_id")
    coords = coords.rename(columns={"longitude_as_stored": "longitude", "latitude_as_stored": "latitude"})
    data = samples.merge(coords, on="station_id", how="left", validate="many_to_one")
    data["time"] = pd.to_datetime(data.time)
    data = data[data.time.between(FIT_START, EVAL_END) & data.split.isin(["train", "validation"])].copy()
    required = ["observed", "geos_cf", "merra2_gmi", "longitude", "latitude"]
    data = data.dropna(subset=required)
    if (data.split == "test").any() or set(data.split.unique()) != {"train", "validation"}:
        raise RuntimeError("Final-test stations must remain sealed")
    return data


def temporal_experiment(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = data[(data.split == "train") & data.time.between(FIT_START, FIT_END)]
    eva = data[(data.split == "validation") & data.time.between(EVAL_START, EVAL_END)].copy()
    models = {
        "GEOS-CF raw": (None, ["geos_cf"]),
        "MERRA2-GMI raw": (None, ["merra2_gmi"]),
        "GEOS-CF train-affine": (fit_linear(fit, ["geos_cf"]), ["geos_cf"]),
        "MERRA2-GMI train-affine": (fit_linear(fit, ["merra2_gmi"]), ["merra2_gmi"]),
        "Two-background train-linear": (fit_linear(fit, ["geos_cf", "merra2_gmi"]), ["geos_cf", "merra2_gmi"]),
    }
    rows, predictions = [], []
    for name, (beta, columns) in models.items():
        pred = eva[columns[0]].to_numpy(float) if beta is None else apply_linear(eva, columns, beta)
        part = eva[["time", "station_id", "observed"]].copy()
        part["method"], part["predicted"] = name, pred
        predictions.append(part)
        row = {"period": "2019-08-15_to_31", "method": name, **metrics(part.observed, part.predicted), **variation_metrics(part)}
        if beta is not None:
            row["coefficients"] = json.dumps(beta.tolist())
        rows.append(row)
    predictions = pd.concat(predictions, ignore_index=True)
    group_rows = []
    for method, frame in predictions.groupby("method"):
        frame = frame.copy()
        frame["day"] = frame.time.dt.strftime("%Y-%m-%d")
        frame["week"] = pd.cut(frame.time, [EVAL_START - pd.Timedelta(seconds=1), pd.Timestamp("2019-08-21 23:30"), pd.Timestamp("2019-08-28 23:30"), EVAL_END], labels=["Aug15-21", "Aug22-28", "Aug29-31"])
        for grouping in ("day", "week"):
            for label, part in frame.groupby(grouping, observed=True):
                group_rows.append({"grouping": grouping, "period": str(label), "method": method, **metrics(part.observed, part.predicted), **variation_metrics(part)})
    return pd.DataFrame(rows), pd.DataFrame(group_rows), predictions


def station_configs(data: pd.DataFrame) -> list[dict]:
    train_ids = sorted(data.loc[data.split == "train", "station_id"].unique())
    val_locs = data[data.split == "validation"].drop_duplicates("station_id")
    train_locs = data[data.split == "train"].drop_duplicates("station_id").set_index("station_id")
    distance = cdist(xy_km(train_locs.loc[train_ids]), xy_km(val_locs)).min(axis=1)
    configs = [{"name": "count40_all", "type": "count", "count": 40, "seed": 0, "station_ids": train_ids}]
    for count in (20, 10):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            configs.append({"name": f"count{count}_seed{seed}", "type": "count", "count": count, "seed": seed, "station_ids": sorted(rng.choice(train_ids, count, replace=False).tolist())})
    for buffer_km in BUFFERS_KM:
        kept = [sid for sid, d in zip(train_ids, distance) if d >= buffer_km]
        configs.append({"name": f"buffer{int(buffer_km)}km", "type": "buffer", "buffer_km": buffer_km, "count": len(kept), "seed": None, "station_ids": kept})
    return configs


def spatial_experiment(data: pd.DataFrame, configs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    fit_all = data[(data.split == "train") & data.time.between(FIT_START, FIT_END)]
    eval_all = data[data.time.between(EVAL_START, EVAL_END)]
    val = eval_all[eval_all.split == "validation"]
    metric_rows, pred_parts, status = [], [], []
    for cfg in configs:
        ids = cfg["station_ids"]
        if len(ids) < 3:
            status.append({"config": cfg["name"], "status": "SKIPPED", "reason": "fewer than 3 training stations remain"})
            continue
        fit = fit_all[fit_all.station_id.isin(ids)]
        betas = {
            "GEOS-CF train-affine": (fit_linear(fit, ["geos_cf"]), ["geos_cf"]),
            "MERRA2-GMI train-affine": (fit_linear(fit, ["merra2_gmi"]), ["merra2_gmi"]),
            "Two-background train-linear": (fit_linear(fit, ["geos_cf", "merra2_gmi"]), ["geos_cf", "merra2_gmi"]),
        }
        variogram, vario_info = fit_variogram_model(fit, ids)
        local_parts = []
        for stamp, target in val.groupby("time"):
            live = eval_all[(eval_all.split == "train") & eval_all.station_id.isin(ids) & (eval_all.time == stamp)]
            if len(live) < 3:
                continue
            source_xy, target_xy = xy_km(live), xy_km(target)
            idw = idw_weights(source_xy, target_xy)
            try:
                ok_weights = ordinary_kriging_weights(variogram, source_xy, target_xy)
            except np.linalg.LinAlgError:
                continue
            base_beta, base_cols = betas["Two-background train-linear"]
            base_target = apply_linear(target, base_cols, base_beta)
            base_live = apply_linear(live, base_cols, base_beta)
            method_pred = {
                "IDW contemporaneous": idw @ live.observed.to_numpy(float),
                "Ordinary Kriging contemporaneous": ok_weights @ live.observed.to_numpy(float),
                "GEOS-CF train-affine": apply_linear(target, betas["GEOS-CF train-affine"][1], betas["GEOS-CF train-affine"][0]),
                "MERRA2-GMI train-affine": apply_linear(target, betas["MERRA2-GMI train-affine"][1], betas["MERRA2-GMI train-affine"][0]),
                "Two-background train-linear": base_target,
                "Two-background + contemporaneous residual IDW": base_target + idw @ (live.observed.to_numpy(float) - base_live),
            }
            nearest = cdist(target_xy, source_xy).min(axis=1)
            for method, pred in method_pred.items():
                part = target[["time", "station_id", "observed"]].copy()
                part["predicted"] = np.maximum(pred, 0.0)
                part["method"], part["config"], part["nearest_train_km"] = method, cfg["name"], nearest
                local_parts.append(part)
        if not local_parts:
            status.append({"config": cfg["name"], "status": "SKIPPED", "reason": "no common evaluation hours"})
            continue
        local = pd.concat(local_parts, ignore_index=True)
        common_keys = local.groupby(["time", "station_id"]).method.nunique()
        common_keys = common_keys[common_keys == 6].index
        local = local.set_index(["time", "station_id"]).loc[common_keys].reset_index()
        pred_parts.append(local)
        for method, part in local.groupby("method"):
            metric_rows.append({
                "config": cfg["name"], "config_type": cfg["type"], "train_station_count": len(ids),
                "seed": cfg.get("seed"), "buffer_km": cfg.get("buffer_km"), "method": method,
                "mean_nearest_train_km": float(part.nearest_train_km.mean()),
                "median_nearest_train_km": float(part.nearest_train_km.median()),
                **metrics(part.observed, part.predicted), **variation_metrics(part),
            })
        status.append({"config": cfg["name"], "status": "PASS", "records_per_method": int(len(local) / 6), "variogram": vario_info})
    return pd.DataFrame(metric_rows), pd.concat(pred_parts, ignore_index=True), status


def peak_metrics(predictions: pd.DataFrame, threshold: float, source: str) -> pd.DataFrame:
    rows = []
    for keys, part in predictions.groupby((["config", "method"] if "config" in predictions else ["method"])):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"source": source, "method": keys[-1]}
        if "config" in predictions:
            row["config"] = keys[0]
        for label, subset in (("normal", part[part.observed < threshold]), ("high", part[part.observed >= threshold])):
            result = metrics(subset.observed, subset.predicted)
            row.update({f"{label}_{k}": v for k, v in result.items()})
        high = part[part.observed >= threshold]
        row["high_value_miss_rate"] = float((high.predicted < threshold).mean()) if len(high) else np.nan
        peak_rows = []
        work = part.copy()
        work["date"] = work.time.dt.date
        for _, daily in work.groupby(["station_id", "date"]):
            obs_idx = daily.observed.idxmax()
            if daily.loc[obs_idx, "observed"] < threshold:
                continue
            pred_idx = daily.predicted.idxmax()
            peak_rows.append({
                "amplitude_error_at_observed_peak": daily.loc[obs_idx, "predicted"] - daily.loc[obs_idx, "observed"],
                "time_error_hours": abs((daily.loc[pred_idx, "time"] - daily.loc[obs_idx, "time"]).total_seconds()) / 3600,
            })
        peak_frame = pd.DataFrame(peak_rows)
        row["peak_events"] = len(peak_frame)
        row["peak_amplitude_bias_at_observed_peak"] = float(peak_frame.amplitude_error_at_observed_peak.mean()) if len(peak_frame) else np.nan
        row["peak_amplitude_mae_at_observed_peak"] = float(peak_frame.amplitude_error_at_observed_peak.abs().mean()) if len(peak_frame) else np.nan
        row["mean_peak_time_error_hours"] = float(peak_frame.time_error_hours.mean()) if len(peak_frame) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(temporal_groups: pd.DataFrame, spatial: pd.DataFrame, peaks: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    weekly = temporal_groups[temporal_groups.grouping == "week"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, part in weekly.groupby("method"):
        ax.plot(part.period, part.rmse, marker="o", label=method)
    ax.set(ylabel="RMSE (ug m-3)", title="Future-period temporal generalization")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUTPUT / "01_temporal_weekly_rmse.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, part in spatial.groupby("method"):
        ax.scatter(part.mean_nearest_train_km, part.rmse, label=method, s=35)
        ordered = part.sort_values("mean_nearest_train_km")
        ax.plot(ordered.mean_nearest_train_km, ordered.rmse, alpha=.45)
    ax.set(xlabel="Mean nearest training-station distance (km)", ylabel="RMSE (ug m-3)", title="Spatial sparsity and distance sensitivity")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUTPUT / "02_sparsity_rmse_distance.png", dpi=180); plt.close(fig)

    shown = peaks[(peaks.source == "temporal") | ((peaks.source == "spatial") & (peaks.config == "count40_all"))]
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"{r.method}\n{r.get('config', '')}" for _, r in shown.iterrows()]
    x = np.arange(len(shown))
    ax.bar(x - .18, shown.normal_rmse, .36, label="Normal")
    ax.bar(x + .18, shown.high_rmse, .36, label="High pollution")
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set(ylabel="RMSE (ug m-3)", title="Normal versus high-pollution error")
    ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT / "03_peak_fidelity_rmse.png", dpi=180); plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    configs = station_configs(data)
    config_record = {
        "status": "frozen_before_evaluation", "created_by": Path(__file__).name,
        "fit_period": [str(FIT_START), str(FIT_END)], "evaluation_period": [str(EVAL_START), str(EVAL_END)],
        "station_split": {"train": 40, "validation": 10, "test": "sealed and never loaded into evaluation"},
        "seeds": list(SEEDS), "counts": list(COUNTS), "buffers_km": list(BUFFERS_KM),
        "peak_threshold_rule": "90th percentile of observations from all 40 training stations during fit period",
        "time_semantics": "ground labels treated as Asia/Shanghai interval ends; backgrounds sampled at UTC HH:30 per prior alignment",
        "coordinate_assumption": "stored station coordinates treated approximately as WGS84; provenance not fully confirmed",
        "configs": configs,
    }
    dump(OUTPUT / "experiment_config.json", config_record)

    temporal, temporal_groups, temporal_pred = temporal_experiment(data)
    spatial, spatial_pred, status = spatial_experiment(data, configs)
    fit_train = data[(data.split == "train") & data.time.between(FIT_START, FIT_END)]
    threshold = float(fit_train.observed.quantile(.90))
    peak = pd.concat([
        peak_metrics(temporal_pred, threshold, "temporal"),
        peak_metrics(spatial_pred, threshold, "spatial"),
    ], ignore_index=True)

    temporal.to_csv(OUTPUT / "temporal_generalization_summary.csv", index=False)
    temporal_groups.to_csv(OUTPUT / "temporal_generalization_daily_weekly.csv", index=False)
    temporal_pred.to_parquet(OUTPUT / "temporal_predictions.parquet", index=False)
    spatial.to_csv(OUTPUT / "spatial_sparsity_summary.csv", index=False)
    spatial_pred.to_parquet(OUTPUT / "spatial_predictions.parquet", index=False)
    peak.to_csv(OUTPUT / "peak_fidelity_summary.csv", index=False)
    dump(OUTPUT / "run_status.json", {"peak_threshold_ug_m3": threshold, "configuration_status": status})
    make_figures(temporal_groups, spatial, peak)
    print(json.dumps({"output": str(OUTPUT), "peak_threshold": threshold, "temporal_rows": len(temporal), "spatial_rows": len(spatial), "peak_rows": len(peak)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
