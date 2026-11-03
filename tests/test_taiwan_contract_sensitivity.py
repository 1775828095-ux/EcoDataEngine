"""Time-contract tests, not evidence for the actual author's timestamp convention."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("taiwan_sensitivity", TOOLS / "run_taiwan_contract_sensitivity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("offset,convention,expected", [
    (0, "start", "2017-01-15 08:00"), (8, "start", "2017-01-15 00:00"),
    (0, "end", "2017-01-15 07:00"), (8, "end", "2017-01-14 23:00")])
def test_hour_window_is_explicit(offset, convention, expected):
    assert audit.hour_start(pd.Timestamp("2017-01-15 08:00"), offset, convention) == pd.Timestamp(expected)


def test_unknown_conventions_are_not_silently_assumed():
    with pytest.raises(ValueError):
        audit.hour_start(pd.Timestamp("2017-01-15"), 0, "unknown")


def test_empty_windows_do_not_reuse_another_windows_rejection_label():
    empty = pd.DataFrame(columns=["station_id", "time", "value"])
    assert audit.case_key("2017-01-15 09:00", empty) != audit.case_key("2017-10-15 09:00", empty)
    assert audit.case_key("2017-01-15 09:00", empty) == audit.case_key("2017-01-15 09:00", empty)
