"""Contract and leakage tests; control fixtures are not experiment outcomes."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("numeric_meteo", ROOT / "tools/run_meteorology_golem_numeric.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_real_chain_orders_dependencies():
    plan = ("station.pykrige_ordinary", "spatial.bilinear")
    assert mod.chain(mod.graph_of(plan)) == plan


@pytest.mark.parametrize("plan", [("spatial.bilinear",), ("fusion.residual",),
                                  ("station.metpy_linear", "station.pykrige_ordinary")])
def test_rejects_incompatible_not_merely_algorithm_names(plan):
    with pytest.raises(ValueError):
        mod.chain(mod.graph_of(plan))


def test_rejects_disconnected_graph():
    graph = mod.OptGraph([mod.OptNode("station.metpy_linear"), mod.OptNode("spatial.nearest")])
    with pytest.raises(ValueError):
        mod.chain(graph)


def test_graph_cache_reuses_real_first_execution_only():
    obj = mod.Evaluator.__new__(mod.Evaluator)
    obj.cache, obj.executions = {}, 0
    obj.execute = lambda plan: {"score": .5, "rmse": 1., "graph": list(plan)}
    graph = mod.graph_of(("station.pykrige_ordinary",))
    assert not obj.evaluate(graph)["cache_hit"]
    assert obj.evaluate(graph)["cache_hit"]
    assert obj.executions == 1


def test_all_experiment_nodes_belong_to_unchanged_frozen_pool():
    frozen = json.loads((ROOT / "output/capability_final_acceptance/frozen_generic_pool.json").read_text(encoding="utf-8"))
    assert set(mod.STATIONS + mod.REGRIDS).issubset({v["id"] for v in frozen["algorithms"]})
    assert frozen["count"] == 34


def test_unit_datum_and_time_uncertainty_not_concealed():
    config = ROOT / "output/golem_real_meteorology_round1/experiment_config.json"
    if not config.exists():
        pytest.skip("Real-data experiment not distributed with this installation")
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["variables"]["rh"]["source_unit"] == "UNCONFIRMED"
    assert not data["variables"]["rh"]["percent_bounds_used"]
    assert not data["variables"]["pressure"]["uniform_pressure_bounds_used"]
    assert "NOT geodetic" in data["crs"]


def test_normalization_fit_excludes_all_held_out_stations_and_years():
    import pandas as pd
    folder = ROOT / "output/golem_real_meteorology_round1"
    if not (folder / "experiment_config.json").exists():
        pytest.skip("Local real-data evidence unavailable")
    config = json.loads((folder / "experiment_config.json").read_text(encoding="utf-8"))
    split = pd.read_csv(folder / "data/station_split.csv", dtype={"station": str})
    held_out = set(split.loc[split.role.ne("train"), "station"])
    for norm in config["normalisation"].values():
        assert not held_out.intersection(norm["fit_station_ids"])
        assert max(norm["fit_years"]) <= 2020


def test_pilot_has_actual_validation_feedback_and_never_opens_test():
    folder = ROOT / "output/golem_real_meteorology_round1"
    if not (folder / "pilot_results.json").exists():
        pytest.skip("Actual pilot evidence unavailable")
    runs = json.loads((folder / "pilot_results.json").read_text(encoding="utf-8"))
    assert len(runs) == 6
    for row in runs:
        assert not row["test_opened"] and row["scored_objective_calls"] <= 20
        assert row["numerical_executions"] >= 1
        assert row["best"]["n"] == 8
        assert Path(row["best"]["netcdf"]).is_file()
        assert any(item["valid"] and "rmse" in item for item in row["objective_trace"])
