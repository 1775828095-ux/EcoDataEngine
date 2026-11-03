"""Bounded empirical portrait measurements using available stations only.

These are local cross-validation measurements, not a learned rank predictor,
not independent test scores, and not free computations.
"""
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
import numpy as np
import xarray as xr
from sklearn.model_selection import GroupKFold


def measure_candidates(evaluator, context, request, candidates, folder):
    from ecodata.algorithms.platform_search import bind_registry, platform_registry, VERSION
    from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
    from ecodata.algorithms.numerical_search import save
    started = perf_counter()
    table = context.observations
    if table is None or not table.split.eq("train").all() or "station_id" not in table:
        raise ValueError("Station probes require explicitly available train stations")
    if len(table) > 100 or table.station_id.nunique() < 8:
        raise ValueError("Station probes require 8..100 available independent stations")
    folds = list(GroupKFold(3).split(table, groups=table.station_id))
    scale = max(float(table.value.std(ddof=0)), 1e-6)
    measurements = {c["key"]: dict(prediction=[None]*len(table), failures=[], folds=[]) for c in candidates}
    attempts = 0
    for fit, hold in folds:
        local = replace(context, observations=table.iloc[fit].copy())
        registry = platform_registry(weights=request.get("weights"))
        decision = deepcopy(evaluator.decision())
        factory, external, _ = bind_registry(registry, local, decision, request)
        pipeline = UnifiedAlgorithmPipeline(registry)
        for item in candidates:
            attempts += 1
            measurement = measurements[item["key"]]
            audit = dict(fit_stations=table.iloc[fit].station_id.tolist(), held_stations=table.iloc[hold].station_id.tolist())
            try:
                output = pipeline.run_dag(item["graph"], task_id=VERSION, external=external,
                                          version=VERSION, decision=decision, factory=factory)
                ds = output["outputs"]["field"].value
                held = table.iloc[hold]
                prediction = ds[context.variable].interp(
                    lon=xr.DataArray(held.longitude.to_numpy(), dims="held"),
                    lat=xr.DataArray(held.latitude.to_numpy(), dims="held")).values
                if not np.isfinite(prediction).all():
                    raise ValueError("Incomplete probe coverage; keep failure instead of shrinking cohort")
                for index, value in zip(hold, prediction):
                    measurement["prediction"][int(index)] = float(value)
                audit["status"] = "PASS"
            except Exception as exc:
                measurement["failures"].append(type(exc).__name__ + ": " + str(exc))
                audit["status"] = "FAIL"
            measurement["folds"].append(audit)
    for measurement in measurements.values():
        if measurement["failures"] or any(x is None for x in measurement["prediction"]):
            measurement["metrics"] = None
        else:
            error = np.asarray(measurement["prediction"]) - table.value.to_numpy(float)
            measurement["metrics"] = dict(RMSE=float(np.sqrt(np.mean(error**2))),
                MAE=float(np.abs(error).mean()), abs_bias=float(abs(error.mean())))
        measurement["scale"] = scale
    report = dict(measurements=measurements, graph_fold_attempts=attempts, seconds=perf_counter()-started,
        train_station_ids=table.station_id.tolist(), training_values=table.value.tolist(),
        reference_used=False, folds=3, policy="fixed station-group folds; no retry or independent target access",
        utility="-error/(available_station_std+error); not a certified future accuracy guarantee")
    save(folder / "station_probes.json", report)
    return report
