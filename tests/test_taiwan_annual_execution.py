"""Summary/leakage guards; synthetic fixtures are not environmental accuracy proof."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from audit_taiwan_annual_v1 import assert_unique_truth, scores


def test_observed_column_not_groupby_boolean_property():
    frame = pd.DataFrame(dict(time=["2017-01-01T00:00Z"]*2,station_id=["v"]*2,observed=[10.,10.],predicted=[11.,12.]))
    assert_unique_truth(frame)
    result = scores(frame)
    assert result["n"] == 2 and result["MAE"] == 1.5


def test_inconsistent_targets_between_methods_block_comparison():
    frame = pd.DataFrame(dict(time=["2017-01-01T00:00Z"]*2,station_id=["v"]*2,observed=[10.,20.]))
    with pytest.raises(ValueError,match="inconsistent"):
        assert_unique_truth(frame)
