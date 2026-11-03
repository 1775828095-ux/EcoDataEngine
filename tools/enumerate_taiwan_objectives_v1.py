"""Small-space exact enumeration and seeded order replay of real numerical outputs.

These callbacks read actual validated predictions. They are cache hits, not new
model executions, and are not a GOLEM Mutation experiment.
"""
import json
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import r2_score

from taiwan_working_contract import CONFIG, OUT, frozen_selection, save, sha


def main():
    frozen_selection()
    folder = OUT/"small_space_search"
    folder.mkdir(exist_ok=False)
    source = OUT/json.loads((OUT/"validation_evaluation_receipt.json").read_text())["output"]
    if not (source/"comparison.csv").exists():
        raise ValueError("Primary validation evaluation must be present")
    summary = pd.read_csv(source/"comparison.csv").set_index(["experiment","method"])
    methods = ["linear","gwr","uk","rf"]
    first = None
    common = None
    for method in methods:
        with xr.open_dataset(source/f"experiment3_{method}.nc") as ds:
            keys = list(zip(ds.time.values,ds.station_id.values))
            if first is None:
                first = keys
            else:
                assert keys == first
            valid = np.isfinite(ds.pm25.values)
            common = valid if common is None else common & valid
    callbacks = []
    for seed in [None,11,22,33,44,55]:
        order = methods if seed is None else list(np.random.default_rng(seed).permutation(methods))
        best = None
        for budget, method in enumerate(order,1):
            start = perf_counter()
            file = source/f"experiment3_{method}.nc"
            with xr.open_dataset(file) as ds:
                assert ds.attrs["split"] == "validation" and ds.attrs["contract_sha256"] == sha(CONFIG)
                pred, truth = ds.pm25.values[common], ds.observed.values[common]
                err = pred-truth
                rmse = float(np.sqrt(np.mean(err**2)))
                original = summary.loc[("experiment3",method)]
                np.testing.assert_allclose(rmse,original.RMSE)
                accepted = original.failures == 0
                if accepted:
                    best = rmse if best is None else min(best,rmse)
                callbacks.append(dict(mode="exhaustive" if seed is None else "random_order_cached_replay",seed=seed,
                    budget=budget,method=method,n=len(pred),RMSE=rmse,MAE=float(abs(err).mean()),Bias=float(err.mean()),
                    R2=float(r2_score(truth,pred)),quality_accepted=bool(accepted), failed_calls=int(original.failures),
                    best_validation_rmse=best,original_algorithm_seconds=float(original.seconds),
                    callback_wall_seconds=perf_counter()-start,source_nc_sha256=sha(file),
                    cache_hit=True,new_algorithm_execution_count=0,unique_configurations_seen=budget))
    pd.DataFrame(callbacks).to_csv(folder/"budget_curves.csv",index=False)
    save(folder/"verification.json",dict(actual_numerical_callbacks=len(callbacks),feasible_graphs=4,
        fixed_seeds=[11,22,33,44,55],budget_per_order=4,all_callbacks_recompute_from_real_validation_predictions=True,
        new_model_fits=0,existing_full_graph_execution_receipts="validation_run_manifest.json",
        cached_replay_not_fresh_algorithm_execution=True,A="NOT_RUN_SMALL_SPACE",B="NOT_RUN_SMALL_SPACE",C="NOT_RUN_SMALL_SPACE",
        innovation_advantage="NOT_PROVEN",reason="All four existing alternatives exactly enumerated; no artificial nodes or search-space inflation",
        test_targets_used=False))
    print("24 real-output numerical callbacks; four configurations fully enumerated; no fresh model fits or GOLEM advantage claim",flush=True)


if __name__ == "__main__":
    main()
