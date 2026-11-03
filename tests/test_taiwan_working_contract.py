"""Working-convention tests never prove the author's metadata."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from taiwan_working_contract import CONFIG, CONTRACT, PROJECTION, aod_values, frozen_selection, scan_hour, sha, timestamps, validate_ground_intervals


def test_naive_and_aware_are_same_instant_without_double_shift():
    result = timestamps(["2017-01-01 00:00", "2016-12-31T16:00:00Z", "2017-01-01T00:00:00+08:00"])
    assert result.nunique() == 1
    assert str(result.iloc[0]) == "2016-12-31 16:00:00+00:00"


def test_left_closed_right_open_and_metadata_priority():
    t, h, source = scan_hour("results_Taiwan_20170101_0030.nc", {})
    assert t.tz_convert("Asia/Taipei").hour == 0 and t.minute == 30
    assert h.tz_convert("Asia/Taipei").hour == 0
    t, h, source = scan_hour("results_Taiwan_20170101_0030.nc", {"time_coverage_start":"2017-01-01T01:00:00+08:00"})
    assert h.tz_convert("Asia/Taipei").hour == 1 and source == "OBSERVATION_METADATA"


def test_composites_are_not_fabricated_as_instants():
    with pytest.raises(ValueError, match="CROSS_HOUR"):
        scan_hour("unused", {"time_coverage_start":"2017-01-01T00:30Z", "time_coverage_end":"2017-01-01T01:30Z"})


def test_aod_qa_fill_and_scale():
    values, valid = aod_values(np.array([10, -999, 30, 9999]), np.array([0,0,2,1]),
        {"valid_range":[0,5000], "_FillValue":-999, "scale_factor":.001})
    assert valid.tolist() == [True,False,False,False]
    assert values[0] == .01


def test_same_metre_projection_roundtrip_and_frozen_assumption_status():
    x, y = PROJECTION.transform(121.,24.)
    assert x > 10000 and y > 10000
    lon, lat = Transformer.from_crs(32651,4326,always_xy=True).transform(x,y)
    np.testing.assert_allclose([lon,lat],[121.,24.],atol=1e-9)
    assert CONTRACT["author_original_definitions_verified"] is False


def test_duplicate_and_off_hour_ground_are_rejected_missing_preserved():
    frame = pd.DataFrame(dict(station_id=["a","a"],time=timestamps(["2017-01-01", "2017-01-01"])))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_ground_intervals(frame)
    frame = frame.iloc[:1].copy()
    frame["time"] = timestamps(["2017-01-01 00:30"])
    with pytest.raises(ValueError, match="aligned"):
        validate_ground_intervals(frame)
    assert pd.isna(timestamps([None]).iloc[0])


def test_existing_idw_really_uses_common_metric_distances():
    import taiwan_annual_adapter as adapter
    adapter.configure()
    table = pd.DataFrame(dict(station_id=["a","b","c"], longitude=[120.,121.,120.5],latitude=[23.,23.,24.],
        value=[10.,20.,30.],split="train",unit="ug m-3", time=pd.Timestamp("2017-01-01")))
    query = table.iloc[:1].copy().assign(station_id="q",longitude=120.4,latitude=23.4).drop(columns=["value","split"])
    source, point = adapter.task.xy(table), adapter.task.xy(query)
    weight = 1/np.linalg.norm(source-point[0],axis=1)**2
    expected = weight @ table.value / weight.sum()
    result = adapter.task.execute("idw",table,query)
    np.testing.assert_allclose(result["outputs"]["prediction"].value,[expected],rtol=1e-10)
    assert result["outputs"]["quality"].value["result"]["contract_version"] == CONTRACT["version"]


def test_test_unseal_requires_frozen_unmodified_validation_selection(tmp_path):
    with pytest.raises(FileNotFoundError):
        frozen_selection(tmp_path)
    file = tmp_path/"selection_frozen.json"
    file.write_text(json.dumps(dict(contract_sha256=sha(CONFIG),final_test_targets_used=False)))
    (tmp_path/"selection_freeze_receipt.json").write_text(json.dumps(dict(sha256=sha(file))))
    assert frozen_selection(tmp_path)["final_test_targets_used"] is False
    file.write_text("{}")
    with pytest.raises(ValueError,match="modified"):
        frozen_selection(tmp_path)
