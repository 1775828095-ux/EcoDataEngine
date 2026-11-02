from __future__ import annotations

import pytest

from ecodata.interface.app import run_selected


@pytest.mark.parametrize("resolution", [2.5, 5.0])
def test_streamlit_core_operation_uses_pipeline(resolution: float) -> None:
    coarse, reference, result, metrics = run_selected("bilinear", resolution, hour=8)
    expected_size = 24 if resolution == 2.5 else 12
    assert result["value"].shape == (1, expected_size, expected_size)
    assert coarse["value"].shape == (1, 6, 6)
    assert reference["value"].shape == result["value"].shape
    assert metrics["runtime_seconds"] >= 0
    assert result.to_netcdf()
