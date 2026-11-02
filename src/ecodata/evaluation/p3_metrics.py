from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.evaluation.metrics import fidelity_metrics


def evaluate_downscaling(
    predicted: xr.DataArray,
    truth: xr.DataArray,
    coarse: xr.DataArray,
    validation_stations: pd.DataFrame,
    factor: int,
    resolution_km: float,
) -> dict[str, float]:
    predicted, truth = xr.align(predicted, truth, join="exact")
    global_metrics = fidelity_metrics(predicted, truth)
    pred_values = predicted.values
    truth_values = truth.values
    validation = validation_stations.copy()
    validation["time"] = pd.to_datetime(validation["time"])
    validation_errors = []
    for row in validation.itertuples():
        time_index = int(np.where(predicted.time.values == np.datetime64(row.time))[0][0])
        validation_errors.append(pred_values[time_index, row.grid_y_index, row.grid_x_index] - row.value)
    validation_errors = np.asarray(validation_errors)

    pred_flat = pred_values.reshape(pred_values.shape[0], -1)
    truth_flat = truth_values.reshape(truth_values.shape[0], -1)
    pred_peak = pred_flat.max(axis=1)
    truth_peak = truth_flat.max(axis=1)
    pred_location = np.column_stack(np.unravel_index(pred_flat.argmax(axis=1), pred_values.shape[1:]))
    truth_location = np.column_stack(np.unravel_index(truth_flat.argmax(axis=1), truth_values.shape[1:]))
    location_error = np.sqrt(((pred_location - truth_location) ** 2).sum(axis=1)) * resolution_km

    reaggregated = pred_values.reshape(
        pred_values.shape[0], coarse.sizes["lat"], factor, coarse.sizes["lon"], factor
    ).mean(axis=(2, 4))
    consistency = reaggregated - coarse.values
    return {
        "mae": global_metrics.mae,
        "rmse": global_metrics.rmse,
        "validation_mae": float(np.mean(np.abs(validation_errors))),
        "validation_rmse": float(np.sqrt(np.mean(validation_errors**2))),
        "peak_amplitude_mae": float(np.mean(np.abs(pred_peak - truth_peak))),
        "peak_location_error_km": float(np.mean(location_error)),
        "coarse_consistency_mae": float(np.mean(np.abs(consistency))),
        "coarse_consistency_max": float(np.max(np.abs(consistency))),
        "negative_count": int(np.count_nonzero(pred_values < -1e-8)),
    }

