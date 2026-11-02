import numpy as np
import xarray as xr

from ecodata.algorithms import AlgorithmContext, AlgorithmPlan, AlgorithmSelector, RawInputQualityGate, ResultQualityGate, create_default_registry
from ecodata.core.schemas import EcoDataBundle


def sample(values=None, *, unit="ug m-3", crs="EPSG:4326"):
    values = [[10.0, 20.0], [30.0, 40.0]] if values is None else values
    ds = xr.Dataset({"pm25": (("lat", "lon"), values)}, coords={"lat": [39.0, 40.0], "lon": [116.0, 117.0]}, attrs={"crs": crs})
    ds.pm25.attrs.update(units=unit, valid_min=0.0, spatial_semantics="cell_mean")
    return ds


def test_raw_gate_passes_valid_structure_but_does_not_claim_accuracy():
    report = RawInputQualityGate().validate(sample(), variable="pm25", intended_operation="fusion")
    assert report["status"] == "PASS"  # absence of truth is recorded, not treated as bad source quality
    assert report["decision"] == "continue"
    assert report["accuracy_assessed"] is False


def test_raw_gate_warns_missing_without_deleting_values():
    ds = sample([[10.0, np.nan], [30.0, 40.0]])
    before = ds.copy(deep=True)
    report = RawInputQualityGate().validate(ds, variable="pm25")
    assert report["status"] == "WARN"
    assert np.isnan(ds.pm25.values[0, 1])
    xr.testing.assert_identical(ds, before)


def test_raw_gate_blocks_all_missing_inf_wrong_unit_and_coordinates():
    all_missing = RawInputQualityGate().validate(sample(np.full((2, 2), np.nan)), variable="pm25")
    assert all_missing["status"] == "BLOCK"
    with_inf = RawInputQualityGate().validate(sample([[1.0, np.inf], [2.0, 3.0]]), variable="pm25")
    assert with_inf["status"] == "BLOCK"
    wrong_unit = RawInputQualityGate().validate(sample(unit="K"), variable="pm25", expected_unit="ug m-3")
    assert wrong_unit["status"] == "BLOCK"
    bad_coord = sample().assign_coords(lat=[39.0, 95.0])
    assert RawInputQualityGate().validate(bad_coord, variable="pm25", intended_operation="fusion")["status"] == "BLOCK"


def test_result_gate_blocks_physical_violation_and_failed_model_check():
    source = sample()
    negative = sample([[-1.0, 2.0], [3.0, 4.0]])
    report = ResultQualityGate().validate(source, negative, variable="pm25", nonnegative=True)
    assert report["status"] == "BLOCK"
    report = ResultQualityGate().validate(source, source, variable="pm25", model_check={"complete": False, "missing": ["wind"]})
    assert report["status"] == "BLOCK"


def test_result_gate_reports_independent_metrics_only_for_validation():
    ds = sample()
    report = ResultQualityGate().validate(ds, ds, variable="pm25", reference=ds.pm25 + 1, validation_role="validation")
    assert report["status"] == "PASS"
    assert report["metrics"]["RMSE"] == 1.0
    assert report["accuracy_assessed"] is True


def test_blocked_input_excludes_every_algorithm():
    quality = RawInputQualityGate().validate(sample(np.full((2, 2), np.nan)), variable="pm25")
    context = AlgorithmContext((EcoDataBundle(sample()),), "pm25", "continuous", input_quality=quality)
    selector = AlgorithmSelector(create_default_registry())
    assert selector.recommend(context) == ()
    assert all(not result.eligible for result in selector.screen(context).values())


def test_fallback_tries_each_candidate_once_and_records_failure():
    context = AlgorithmContext((EcoDataBundle(sample()),), "pm25", "continuous",
                               target_lat=(39.0, 39.5, 40.0), target_lon=(116.0, 116.5, 117.0))
    plans = (AlgorithmPlan("missing", ("not.registered",), 2, ("test failure",)),
             AlgorithmPlan("nearest", ("spatial.nearest",), 1, ("fallback",)))
    result = AlgorithmSelector(create_default_registry()).execute_with_fallback(context, plans, max_attempts=2)
    assert result.algorithm_id == "nearest"
    assert [item["status"] for item in result.report["fallback_attempts"]] == ["BLOCK", "PASS"]
