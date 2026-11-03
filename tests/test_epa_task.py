"""Contract/anti-leakage unit fixtures, NOT real MAIAC/ERA5 validation."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.dag import Artifact, DAGError, NumericalDAGExecutor
from ecodata.algorithms.epa_task import EPATask, UNIT


def task():
    template = xr.Dataset({"pm25": (("lat", "lon"), np.full((3, 3), np.nan))},
        coords={"lat": [33., 34., 35.], "lon": [-119., -118., -117.]}, attrs={"crs": "EPSG:4326"})
    template.pm25.attrs["units"] = UNIT
    return EPATask(None, template, features=("test_covariate",), train_end="2020-12-31")


def fixture_history():
    # Arbitrary unit-test covariate, never named or presented as actual AOD.
    return pd.DataFrame([{"station_id": str(i), "time": f"2020-01-{day:02d}", "longitude": -118.8+i*.2,
        "latitude": 33.2+(i % 2)*.4, "value": float(8+i+day), "unit": UNIT, "split": "train",
        "test_covariate": float(i*day)} for i in range(5) for day in range(1, 5)])


def test_two_routes_have_different_state_and_math_contracts():
    t = task()
    a, b = t.graph("route_a"), t.graph("route_b")
    engine = NumericalDAGExecutor(t.adapters())
    assert engine.order(a, "quality")
    assert engine.order(b, "quality")
    assert {n.operation for n in a} != {n.operation for n in b}
    assert any(n.operation == "epa.rf_fit" for n in b)
    assert not any(n.operation == "epa.rf_fit" for n in a)


def test_rf_crossfit_uses_station_groups_and_only_history():
    t = task()
    result = t.fit_rf({"history": Artifact("epa_history", fixture_history())}, {}, {})["state"].value
    for fold in result["folds"]:
        assert not set(fold["fit_stations"]) & set(fold["held_stations"])
    assert result["residuals"].station_id.nunique() == 5
    assert np.isfinite(result["residuals"].value).all()
    assert result["train_end"] == "2020-12-31"


def test_existing_dag_cache_reuses_fit_but_invalidates_changed_training():
    from ecodata.algorithms.dag import Node
    t = task()
    history = fixture_history()
    engine = NumericalDAGExecutor(t.adapters())
    nodes = [Node('history', 'epa.source.history', parameters={'source': 'history'}),
             Node('fit', 'epa.rf_fit', {'history': ('history', 'data')})]
    first = engine.run(nodes, 'fit', external={'history': Artifact('epa_history', history)})
    second = engine.run(nodes, 'fit', external={'history': Artifact('epa_history', history)})
    assert not first['trace'][-1]['cache_hit']
    assert second['trace'][-1]['cache_hit']
    changed = history.copy()
    changed.loc[0, 'value'] += 1
    third = engine.run(nodes, 'fit', external={'history': Artifact('epa_history', changed)})
    assert not third['trace'][-1]['cache_hit']


@pytest.mark.parametrize("change,match", [("test", "training-station"), ("future", "cutoff"), ("unit", "ug m-3")])
def test_bad_fit_contract_is_rejected(change, match):
    t = task()
    history = fixture_history()
    if change == "test":
        history.loc[0, "split"] = "test"
    elif change == "future":
        history.loc[0, "time"] = "2022-01-01"
    else:
        history["unit"] = "unknown"
    with pytest.raises(DAGError, match=match):
        t.fit_rf({"history": Artifact("epa_history", history)}, {}, {})


def test_objective_never_accepts_final_test_targets(tmp_path):
    t = task()
    validation = fixture_history().iloc[:1].assign(split="test")
    with pytest.raises(DAGError, match="validation only"):
        t.objective("route_a", {}, validation, tmp_path / "forbidden")
    assert t.counters == {"objective_calls": 1, "executions": 0, "failures": 1}
    assert not (tmp_path / "forbidden").exists()


def test_actual_covariate_provenance_required():
    t = task()
    observations = fixture_history().query("time == '2020-01-01'")
    with pytest.raises(DAGError, match="provenance"):
        t.estimate({"observations": Artifact("epa_train", observations),
            "covariates": Artifact("epa_covariates", xr.Dataset({"test_covariate": t.template.pm25}))},
            {"algorithm": "station.pykrige_universal"}, {})


def test_universal_masks_missing_queries_but_never_fills_satellite_gaps():
    from ecodata.algorithms.contracts import AlgorithmContext
    from ecodata.algorithms.open_source_stations import PyKrigePlugin
    from ecodata.core.schemas import EcoDataBundle
    t = task()
    observations = fixture_history().query("time == '2020-01-01'")
    field = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3),
                         dims=('lat', 'lon'), coords=t.template.coords,
                         attrs={'units': '1', 'data_kind': 'continuous'})
    field.values[-1, -1] = np.nan
    context = AlgorithmContext((EcoDataBundle(t.template),), 'pm25', 'continuous',
        observations=observations, auxiliaries=(xr.Dataset({'test_covariate': field}, attrs={'crs': 'EPSG:4326'}),),
        target_lat=tuple(t.template.lat.values), target_lon=tuple(t.template.lon.values),
        parameters={'observation_crs': 'EPSG:4326'})
    plugin = PyKrigePlugin('universal')
    assert not plugin.compatibility(context).eligible
    context.parameters['mask_missing_query_covariates'] = True
    assert plugin.compatibility(context).eligible
    output = plugin.run(context).data.dataset.pm25.values
    assert np.isnan(output[-1, -1])
    assert np.isfinite(output[1, 1])  # identical-grid lookup must not grow the NaN hole
    assert np.isfinite(output).any()


def test_nonnegative_projection_is_explicit_opt_in_and_reported():
    t = task()
    t.template = t.template.assign_coords(lat=[33.2, 33.4, 33.6], lon=[-118.8, -118.4, -118.0])
    field = t.template.copy(deep=True)
    field['pm25'] = xr.full_like(field.pm25, -1.)
    field.pm25.attrs['units'] = UNIT
    inputs = {'field': Artifact('epa_field', field),
              'observations': Artifact('epa_train', fixture_history().query("time == '2020-01-01'"))}
    with pytest.raises(DAGError):
        t.quality(inputs, {}, {})
    t.nonnegative_projection = True
    result = t.quality(inputs, {}, {})
    assert result['report'].value['status'] == 'WARN'
    assert result['report'].value['nonnegative_projection']['count'] > 0
    assert float(result['data'].value.pm25.min()) == 0.


def test_sealed_test_station_cannot_be_relabelled_train():
    t = task()
    t.sealed_test_stations = ("4",)
    with pytest.raises(DAGError, match="Sealed test"):
        t.fit_rf({"history": Artifact("epa_history", fixture_history())}, {}, {})


@pytest.mark.parametrize("change,match", [("unit", "unit"), ("time", "daily-window")])
def test_objective_checks_validation_semantics_before_execution(change, match, tmp_path):
    t = task()
    train = fixture_history().query("time == '2020-01-01'")
    validation = train.iloc[:1].assign(station_id="held_out", split="validation")
    if change == "unit":
        validation["unit"] = "unknown"
    else:
        validation["time"] = "2021-01-01"
    with pytest.raises(DAGError, match=match):
        t.objective("fusion.idw", {"stations": Artifact("epa_train", train)}, validation, tmp_path / "refused")
    assert t.counters["executions"] == 0
