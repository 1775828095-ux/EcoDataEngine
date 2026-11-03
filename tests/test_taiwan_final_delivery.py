"""Boolean readiness contracts only; these fixtures prove no data provenance."""
import importlib.util
from pathlib import Path

import pytest

FILE = Path(__file__).resolve().parents[1] / "tools/verify_taiwan_final_delivery.py"
spec = importlib.util.spec_from_file_location("taiwan_final_delivery", FILE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("missing", ["archive_verified", "ground_time_verified", "satellite_time_verified", "coordinate_verified"])
def test_each_missing_proof_blocks_formal_freeze(missing):
    conditions = {key: True for key in ["archive_verified", "ground_time_verified", "satellite_time_verified", "coordinate_verified"]}
    conditions[missing] = False
    result = audit.formal_gate(**conditions)
    assert result["allowed"] is False
    assert len(result["blockers"]) == 1


def test_status_labels_are_not_positive_proof():
    assert not audit.formal_gate(archive_verified="UNVERIFIED", ground_time_verified="false",
        satellite_time_verified="PARTIAL", coordinate_verified="unknown")["allowed"]


def test_explicit_boolean_conditions_are_required():
    result = audit.formal_gate(archive_verified=True, ground_time_verified=True, satellite_time_verified=True, coordinate_verified=True)
    assert result["allowed"] and result["blockers"] == []
