"""Targeted train-only spatial experiment safeguards, not accuracy claims."""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_bth_sparse_fusion import read_train_only, spatial_order, support_mask, metrics


def test_sealed_rows_not_parsed(tmp_path):
    path = tmp_path / "input.csv"
    fields = ["split", "station_id", "time_as_stored", "longitude_as_stored", "latitude_as_stored", "pm25_as_stored"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ["train", "a", "2019-08-01", 116, 39, 20])))
        for role in ["validation", "test"]:
            writer.writerow(dict(zip(fields, [role, "sealed", "DO_NOT_PARSE", "bad", "bad", "bad"])))
    frame, skipped = read_train_only(path)
    assert skipped == 2 and frame.station_id.tolist() == ["a"]


def test_station_order_coordinate_only_nested_reproducible():
    x, y = np.meshgrid(np.linspace(115, 118, 8), np.linspace(38, 40, 5))
    sites = pd.DataFrame(dict(station_id=[str(i) for i in range(40)], longitude=x.ravel(), latitude=y.ravel()))
    first = spatial_order(sites, 11)
    assert first == spatial_order(sites.sample(frac=1, random_state=4).assign(value=999), 11)
    assert len(set(first)) == 20
    assert set(first[:5]) < set(first[:10]) < set(first)


def test_geometry_rejects_extrapolation_and_insufficient_points():
    source = pd.DataFrame(dict(longitude=[115, 118, 115, 118], latitude=[38, 38, 41, 41]))
    query = pd.DataFrame(dict(longitude=[116, 125], latitude=[39, 39]))
    assert support_mask(source, query).tolist() == [True, False]
    assert not support_mask(source.iloc[:2], query).any()


def test_prediction_failure_keeps_denominator():
    result = metrics([10, 20], [10, np.nan])
    assert result["expected"] == 2 and result["predicted"] == 1
    assert result["coverage"] == .5 and result["failure_rate"] == .5


def test_fixed_real_features_and_folds():
    from run_bth_sparse_fusion import SparseRFTask, Artifact, xr
    sites = []
    for station in range(5):
        for hour in range(8):
            sites.append(dict(station_id=str(station), longitude=115+station*.1,
                latitude=39+station*.05, time=pd.Timestamp("2019-08-01")+pd.Timedelta(hours=hour),
                value=10+station+hour, geos=30+hour, split="train", unit="ug m-3"))
    task = SparseRFTask(None, xr.Dataset(), features=("geos",), train_end="2019-08-02", sealed_test_stations=("sealed",))
    state = task.fit_rf({"history": Artifact("history", pd.DataFrame(sites))}, {}, {})["state"].value
    assert state["model"].n_estimators == 50 and state["model"].max_depth == 6
    assert all(not set(f["fit_stations"]) & set(f["held_stations"]) for f in state["folds"])
    assert len(state["residuals"]) == 5
    bad = pd.DataFrame(sites).assign(station_id="sealed")
    with pytest.raises(Exception):
        task.fit_rf({"history": Artifact("history", bad)}, {}, {})


def test_paired_shared_day_bootstrap_preserves_known_difference():
    from analyze_bth_sparse_fusion import paired_uncertainty
    rows=[]
    for seed in [11,22,33]:
        for day in ["2019-08-01","2019-08-02"]:
            for method,error in [("IDW",2.),("candidate",1.)]:
                rows.append(dict(count=5,cohort="test",seed=seed,day=day,method=method,
                    n=10,squared_error=10*error**2,absolute_error=10*error))
    result=paired_uncertainty(pd.DataFrame(rows),reps=20).iloc[0]
    assert result.RMSE_improvement_pct_mean==50
    assert result.RMSE_CI_low==50 and result.RMSE_CI_high==50
    assert result.RMSE_winning_groups==3
