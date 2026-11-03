"""Executor unit tests use small control arrays, not claimed environmental truth."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.dag import Artifact, DAGError, Node, NodeAdapter, NumericalDAGExecutor, Port
from ecodata.algorithms.pm25_dag import PM25DAGAdapters, UNIT
from ecodata.algorithms.registry import create_default_registry
from ecodata.core.schemas import EcoDataBundle
from ecodata.fusion.residual import ResidualFieldFusion, ResidualFusionConfig
from ecodata.io.readers import _frame_to_dataset


def fixture_engine():
    adapters = {
        ("left", "run"): NodeAdapter({}, {"value": Port("left")}, lambda i, p, e: {"value": e["left"]}),
        ("right", "run"): NodeAdapter({}, {"value": Port("right")}, lambda i, p, e: {"value": e["right"]}),
        ("multiply", "run"): NodeAdapter({"x": Port("left"), "y": Port("right")}, {"product": Port("number")},
            lambda i, p, e: {"product": Artifact("number", i["x"].value*i["y"].value*p.get("factor", 1))}),
    }
    return NumericalDAGExecutor(adapters)


def test_real_arithmetic_dependency_order_and_cache_invalidation():
    engine = fixture_engine()
    nodes = [Node("result", "multiply", {"x": ("a", "value"), "y": ("b", "value")}),
             Node("b", "right", parameters={"source": "right"}), Node("a", "left", parameters={"source": "left"})]
    external = {"left": Artifact("left", 3), "right": Artifact("right", 7)}
    result = engine.run(nodes, "result", external=external)
    assert result["outputs"]["product"].value == 21
    assert [x["node"] for x in result["trace"]] == ["a", "b", "result"]
    assert all(x["cache_hit"] for x in engine.run(nodes, "result", external=external)["trace"])
    external["left"].value = 4
    assert engine.run(nodes, "result", external=external)["outputs"]["product"].value == 28
    nodes[0].parameters["factor"] = 2
    assert engine.run(nodes, "result", external=external)["outputs"]["product"].value == 56


def test_xarray_fingerprints_survive_copy_and_reject_metadata_changes():
    a = Artifact("grid", xr.Dataset({"x": ("t", [1., np.nan])}, coords={"t": pd.date_range("2020", periods=2)}))
    assert a.fingerprint() == deepcopy(a).fingerprint()
    b = deepcopy(a)
    b.value.x.attrs["units"] = "K"
    assert a.fingerprint() != b.fingerprint()
    with pytest.raises(DAGError, match="metadata"):
        Port("grid", ("unit",)).validate(a)


def test_cycle_missing_edge_and_disconnected_nodes_rejected():
    engine = fixture_engine()
    with pytest.raises(DAGError, match="Incompatible"):
        engine.order([Node("a", "left"), Node("r", "multiply", {"x": ("a", "missing"), "y": ("a", "value")})], "r")
    with pytest.raises(DAGError, match="Disconnected"):
        engine.order([Node("a", "left"), Node("b", "right")], "a")
    adapters = {("pass", "run"): NodeAdapter({"x": Port("grid")}, {"y": Port("grid")}, lambda i, p, e: {"y": i["x"]})}
    with pytest.raises(DAGError, match="Cycle"):
        NumericalDAGExecutor(adapters).order([Node("a", "pass", {"x": ("b", "y")}), Node("b", "pass", {"x": ("a", "y")})], "a")


def test_official_golem_edges_resolve_names_not_parent_list_order():
    sys.path.append(str(Path(__file__).resolve().parents[1] / ".venv-golem/Lib/site-packages"))
    from golem.core.optimisers.graph import OptGraph, OptNode
    a = OptNode({"name": "left", "parameters": {"source": "left"}})
    b = OptNode({"name": "right", "parameters": {"source": "right"}})
    c = OptNode("multiply", nodes_from=[b, a])
    result = fixture_engine().run_golem(OptGraph([c]), external={"left": Artifact("left", 5), "right": Artifact("right", 6)})
    assert result["outputs"]["product"].value == 30


def test_explicit_same_kind_bindings_change_native_graph_identity():
    sys.path.append(str(Path(__file__).resolve().parents[1] / ".venv-golem/Lib/site-packages"))
    from golem.core.optimisers.graph import OptGraph, OptNode
    adapters = {
        ("a", "run"): NodeAdapter({}, {"v": Port("number")}, lambda i, p, e: {"v": e["a"]}),
        ("b", "run"): NodeAdapter({}, {"v": Port("number")}, lambda i, p, e: {"v": e["b"]}),
        ("subtract", "run"): NodeAdapter({"x": Port("number"), "y": Port("number")}, {"v": Port("number")},
            lambda i, p, e: {"v": Artifact("number", i["x"].value-i["y"].value)}),
    }
    a, b = OptNode({"name": "a", "params": {"source": "a"}}), OptNode({"name": "b", "params": {"source": "b"}})
    c = OptNode("subtract", nodes_from=[a, b])
    engine = NumericalDAGExecutor(adapters)
    with pytest.raises(DAGError, match="ambiguous"):
        engine.from_golem(OptGraph([c]))
    c.parameters = {"_port_bindings": {"x": [0, "v"], "y": [1, "v"]}}
    g = OptGraph([c])
    first_id = g.descriptive_id
    external = {"a": Artifact("number", 7), "b": Artifact("number", 2)}
    assert engine.run_golem(g, external=external)["outputs"]["v"].value == 5
    c.parameters = {"_port_bindings": {"x": [1, "v"], "y": [0, "v"]}}
    assert g.descriptive_id != first_id
    assert engine.run_golem(g, external=external)["outputs"]["v"].value == -5


def test_calibration_refuses_validation_labels_and_future_times():
    adapter = PM25DAGAdapters(None, "2019-08-15 00:30", [1, 2], [1, 2], exploratory=True)
    frame = pd.DataFrame({"split": ["validation"], "time": [pd.Timestamp("2019-08-02")], "unit": [UNIT]})
    with pytest.raises(DAGError, match="past training"):
        adapter.fit_calibration({"observations": Artifact("history_observations", frame)}, {}, {})
    frame["split"], frame["time"] = "train", pd.Timestamp("2019-08-16")
    with pytest.raises(DAGError, match="past training"):
        adapter.fit_calibration({"observations": Artifact("history_observations", frame)}, {}, {})
    with pytest.raises(DAGError, match="datum"):
        PM25DAGAdapters(None, "2019-08-15", [1, 2], [1, 2])


def test_residual_mask_does_not_hide_nearest_extrapolation():
    ds = xr.Dataset({"value": (("lat", "lon"), np.ones((4, 4)))}, coords={"lat": [0., 1., 2., 3.], "lon": [0., 1., 2., 3.]})
    ds.value.attrs.update(units=UNIT, physical_variable="pm")
    obs = pd.DataFrame({"longitude": [1., 2., 1.], "latitude": [1., 1., 2.], "value": [2., 3., 4.],
        "unit": UNIT, "variable": "pm", "time": pd.Timestamp("2020-01-01"), "station_id": ["a", "b", "c"], "qc_flag": "VALID"})
    result = ResidualFieldFusion(ResidualFusionConfig("value", "linear", "mask")).apply(EcoDataBundle(ds), EcoDataBundle(_frame_to_dataset(obs)))
    assert np.isnan(result.data.dataset.value.sel(lat=0, lon=0))
    assert float(result.data.dataset.value.sel(lat=1, lon=1)) == pytest.approx(2.)


def test_existing_pool_not_expanded():
    import json
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "output/capability_final_acceptance/frozen_generic_pool.json").read_text(encoding="utf-8"))
    assert frozen["count"] == 34
    registry = create_default_registry()
    assert all(registry.get(x["id"]) is not None for x in frozen["algorithms"])
