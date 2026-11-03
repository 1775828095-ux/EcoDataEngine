"""Preflight contract tests; fixtures are not scientific experiment results."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("golem_pm25_preflight", Path(__file__).parents[1] / "tools/audit_golem_pm25_readiness.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def profile():
    return {"duplicate_station_time_extra_rows": 0, "negative_pm25": 0,
            "invalid_coordinate_rows": 0, "split_overlap_stations": []}


def semantics():
    return {"coordinate_datum": "CONFIRMED", "crs": "EPSG:4326",
            "coordinate_evidence": "fixture metadata", "unit_supported": True, "time_supported": True}


def test_unknown_datum_blocks_even_with_plausible_coordinates():
    values = semantics()
    values["coordinate_datum"] = "UNRESOLVED"
    assert module.readiness(profile(), values)["search_permitted"] is False


def test_epsg_without_evidence_does_not_override_block():
    values = semantics()
    values["coordinate_evidence"] = None
    assert "STATION_COORDINATE_DATUM_UNCONFIRMED" in module.readiness(profile(), values)["blockers"]


def test_duplicates_and_overlap_block():
    data = profile()
    data.update(duplicate_station_time_extra_rows=1, split_overlap_stations=["fixture"])
    assert len(module.readiness(data, semantics())["blockers"]) == 2


def test_passing_preflight_is_not_accuracy_evidence():
    result = module.readiness(profile(), semantics())
    assert result["search_permitted"] and not result["prediction_accuracy_claimed"]
    assert result["status"] == "PREFLIGHT_ONLY"


def test_nonfinite_is_not_valid_observation():
    assert module.number("NaN") is None
    assert module.number("inf") is None
    assert module.number("—") is None
    assert module.number("0") == 0
