"""Time-contract tests only; these do not prove real spatial/fusion readiness."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_golem_multisource_round2 import utc_center


@pytest.mark.parametrize("label, expected", [
    ("2019-08-01 09:00:00", datetime(2019, 8, 1, 0, 30)),
    ("2019-08-01 00:00:00", datetime(2019, 7, 31, 15, 30)),
    ("2019-08-31 23:00:00", datetime(2019, 8, 31, 14, 30)),
])
def test_preceding_hour_beijing_label_maps_to_utc_center(label, expected):
    assert utc_center(label) == expected
