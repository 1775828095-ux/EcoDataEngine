from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.pipeline import P1Pipeline
from ecodata.io.readers import CsvReadConfig, NetCdfReader, StationCsvReader
from ecodata.temporal.resampling import LinearTimeConfig, LinearTimeInterpolator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "p1_irregular_observations.csv"


def read_config(**overrides) -> CsvReadConfig:
    values = {
        "dataset_id": "test-irregular",
        "source_name": "test fixture",
        "time_zone": "Asia/Shanghai",
    }
    values.update(overrides)
    return CsvReadConfig(**values)


def test_complete_csv_netcdf_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "result.nc"
    result = P1Pipeline().run(
        SOURCE,
        target,
        read_config(),
        LinearTimeConfig(frequency="1h"),
    )

    assert result.report.passed
    assert all(result.report.checks.values())
    restored = result.data.dataset
    assert restored.sizes["observation"] == 4
    np.testing.assert_allclose(restored["value"].values, [20.0, 36.0, 48.0, 35.0])
    assert restored["qc_flag"].values.tolist() == ["VALID", "INTERPOLATED", "INTERPOLATED", "VALID"]
    assert set(restored["unit"].values.tolist()) == {"ug m-3"}
    assert restored.attrs["temporal_semantics"] == "instantaneous"
    assert restored.attrs["time_zone"] == "UTC"
    assert restored.attrs["source_time_zone"] == "Asia/Shanghai"
    assert restored["value"].attrs["ancillary_variables"] == "qc_flag"
    history = json.loads(restored.attrs["transform_history"])
    assert history[-1]["parameters"] == {"extrapolation": False, "frequency": "1h"}


def test_irregular_times_are_normalized_to_utc_and_interpolated() -> None:
    raw = StationCsvReader().read(SOURCE, read_config())
    result = LinearTimeInterpolator(LinearTimeConfig("1h")).apply(raw).data.dataset
    expected = pd.to_datetime(
        ["2025-12-31T16:00:00", "2025-12-31T17:00:00", "2025-12-31T18:00:00", "2025-12-31T19:00:00"]
    ).to_numpy(dtype="datetime64[ns]")
    np.testing.assert_array_equal(result["time"].values, expected)


def test_no_extrapolation_when_bounds_are_not_on_target_grid(tmp_path: Path) -> None:
    source = tmp_path / "short.csv"
    pd.DataFrame(
        [
            ["S", "2026-01-01T00:10:00Z", 30, 110, "PM2.5", 10, "ug m-3", "VALID"],
            ["S", "2026-01-01T02:50:00Z", 30, 110, "PM2.5", 30, "ug m-3", "VALID"],
        ],
        columns=["station_id", "time", "latitude", "longitude", "variable", "value", "unit", "qc_flag"],
    ).to_csv(source, index=False)
    raw = StationCsvReader().read(source, read_config())
    output = LinearTimeInterpolator(LinearTimeConfig("1h")).apply(raw).data.dataset
    expected = pd.to_datetime(["2026-01-01T01:00:00", "2026-01-01T02:00:00"]).to_numpy(dtype="datetime64[ns]")
    np.testing.assert_array_equal(output["time"].values, expected)


def test_inconsistent_units_fail_before_processing(tmp_path: Path) -> None:
    frame = pd.read_csv(SOURCE)
    frame.loc[1, "unit"] = "mg m-3"
    source = tmp_path / "mixed_units.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(SchemaError, match="inconsistent units"):
        StationCsvReader().read(source, read_config())


def test_out_of_order_observations_are_rejected_not_sorted(tmp_path: Path) -> None:
    frame = pd.read_csv(SOURCE).iloc[[0, 2, 1, 3]].reset_index(drop=True)
    source = tmp_path / "out_of_order.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(SchemaError, match="chronological"):
        StationCsvReader().read(source, read_config())


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (read_config(temporal_semantics="mean"), "instantaneous"),
        (read_config(data_kind="categorical"), "continuous"),
    ],
)
def test_invalid_semantics_reject_linear_interpolation(config, message: str) -> None:
    raw = StationCsvReader().read(SOURCE, config)
    with pytest.raises(SemanticError, match=message):
        LinearTimeInterpolator(LinearTimeConfig("1h")).apply(raw)


def test_missing_required_column_fails(tmp_path: Path) -> None:
    frame = pd.read_csv(SOURCE).drop(columns="qc_flag")
    source = tmp_path / "missing_qc.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(SchemaError, match="qc_flag"):
        StationCsvReader().read(source, read_config())


def test_netcdf_reader_preserves_key_metadata(tmp_path: Path) -> None:
    target = tmp_path / "result.nc"
    P1Pipeline().run(SOURCE, target, read_config(), LinearTimeConfig("1h"))
    restored = NetCdfReader().read(target).dataset
    assert restored.attrs["dataset_id"] == "test-irregular"
    assert restored.attrs["crs"] == "EPSG:4326"
    assert restored.attrs["Conventions"] == "CF-1.10"

