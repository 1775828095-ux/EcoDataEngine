"""Contract-only tests; synthetic fixtures are not environmental accuracy evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FILE = Path(__file__).resolve().parents[1] / "tools/run_taiwan_preparation.py"
spec = importlib.util.spec_from_file_location("taiwan_preparation", FILE)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


@pytest.fixture
def train():
    return pd.DataFrame(dict(station_id=["a", "b", "c", "d", "e", "f"], longitude=[120, 121, 120, 121, 120.4, 120.6],
        latitude=[23, 23, 24, 24, 23.4, 23.6], value=[10., 12., 13., 14., 11., 15.], aod=[.1,.2,.3,.4,.15,.25],
        time=pd.Timestamp("2017-03-06 09:00"), split="train", unit=task.UNIT))


def test_validation_and_test_labels_refused_by_objective(train):
    for role in ["train", "test"]:
        with pytest.raises(ValueError, match="validation labels only"):
            task.validation_metrics(np.ones(len(train)), train.assign(split=role))


def test_rf_refuses_nontraining_rows(train):
    with pytest.raises(ValueError, match="Nontraining"):
        task.rf_fit(train.assign(split="validation"))


def test_rf_real_crossfit_is_finite(train):
    state = task.rf_fit(train)
    assert np.isfinite(state["residual"]).all()
    assert set(state["fit_stations"]) == set(train.station_id)
    assert "station-disjoint" in state["crossfit"]


def test_graph_refuses_query_labels(train):
    query = train.iloc[:1].assign(station_id="heldout", split="validation")
    with pytest.raises(ValueError, match="must not contain"):
        task.execute("linear", train, query)


def test_graph_refuses_time_mismatch(train):
    query = train.iloc[:1].drop(columns=["value", "split"]).assign(station_id="heldout", time=pd.Timestamp("2017-03-07"))
    with pytest.raises(ValueError, match="time windows"):
        task.execute("linear", train, query)


def test_genuine_golem_graph_executes_linear(train):
    query = train.iloc[:2].drop(columns=["value", "split"]).assign(station_id=["h1", "h2"])
    result = task.execute("linear", train, query)
    assert np.isfinite(result["outputs"]["prediction"].value).all()
    assert len(result["trace"]) == 4
    assert result["outputs"]["quality"].value["result"]["status"] == "WARN"


def test_quality_port_returns_checked_values_and_retains_raw(train):
    query = train.iloc[:1].drop(columns=["value", "split"]).assign(station_id="heldout")
    result = task.graph_executor().adapters[("quality", "run")].execute({
        "train": task.Artifact("training", train), "query": task.Artifact("query", query),
        "prediction": task.Artifact("prediction", np.array([-3.]))}, {}, {})
    assert result["prediction"].value[0] == 0
    assert result["raw_prediction"].value[0] == -3
    assert result["quality"].value["result"]["projection_count"] == 1
