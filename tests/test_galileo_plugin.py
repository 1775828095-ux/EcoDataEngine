from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.galileo import GalileoFeaturePlugin
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.schemas import EcoDataBundle


def _context(*, include_slope: bool = True) -> AlgorithmContext:
    values = np.arange(64, dtype=np.float32).reshape(8, 8) * 10
    variables = {"elevation": (("y", "x"), values)}
    if include_slope:
        variables["slope"] = (("y", "x"), np.gradient(values)[0].astype(np.float32))
    dataset = xr.Dataset(variables, coords={"y": np.arange(8), "x": np.arange(8)}, attrs={"crs": "EPSG:32650"})
    return AlgorithmContext(
        sources=(EcoDataBundle(dataset),),
        variable="galileo_embedding",
        semantic_type="continuous",
        parameters={"patch_size": 2, "input_resolution_m": 30, "month": 8},
    )


def test_galileo_rejects_incomplete_official_group():
    result = GalileoFeaturePlugin().compatibility(_context(include_slope=False))
    assert not result.eligible
    assert "完整" in result.reasons[0]


def test_galileo_cannot_masquerade_as_environmental_prediction():
    context = _context()
    context = AlgorithmContext(
        sources=context.sources,
        variable="elevation",
        semantic_type=context.semantic_type,
        parameters=context.parameters,
    )
    result = GalileoFeaturePlugin().compatibility(context)
    assert not result.eligible
    assert any("不能冒充" in reason for reason in result.reasons)


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "external" / "galileo" / "data" / "models" / "nano" / "encoder.pt").exists(),
    reason="official Galileo nano weights are not present",
)
def test_galileo_combines_multiple_official_groups_and_records_scale_matching():
    context = _context()
    coarse = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    s2 = xr.Dataset(
        {band: (("y", "x"), coarse + offset) for band, offset in zip(("B2", "B3", "B4"), (0.0, 0.1, 0.2))},
        coords={"y": np.linspace(0, 7, 4), "x": np.linspace(0, 7, 4)},
        attrs={"crs": "EPSG:32650"},
    )
    era5 = xr.Dataset(
        {"temperature_2m": xr.DataArray(289.0), "total_precipitation_sum": xr.DataArray(0.003)},
        attrs={"crs": "EPSG:32650"},
    )
    context = AlgorithmContext(
        sources=context.sources,
        auxiliaries=(s2, era5),
        variable=context.variable,
        semantic_type=context.semantic_type,
        parameters=context.parameters,
    )
    result = GalileoFeaturePlugin().run(context)
    assert result.report["groups"] == ["SRTM", "ERA5", "S2_RGB"]
    assert result.report["reference_group"] == "SRTM"
    assert any(item["method"] == "bilinear_feature_alignment" for item in result.report["scale_matching"])
    assert result.data.dataset.galileo_embedding.shape == (4, 4, 128)
    assert np.isfinite(result.data.dataset.galileo_embedding.values).all()


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "external" / "galileo" / "data" / "models" / "nano" / "encoder.pt").exists(),
    reason="official Galileo nano weights are not present",
)
def test_official_galileo_adapter_runs_through_registry_and_quality_gate():
    registry = create_default_registry()
    assert registry.get("features.galileo_nano").spec.license == "MIT"
    result = UnifiedAlgorithmPipeline(registry).run(
        _context(),
        AlgorithmPlan("galileo_demo", ("features.galileo_nano",), 1.0, ("official integration test",)),
    )
    output = result.data.dataset
    assert result.report["quality"]["status"] == "PASS"
    assert output.galileo_embedding.shape == (4, 4, 128)
    assert output.galileo_embedding.dtype == np.float32
    assert np.isfinite(output.galileo_embedding.values).all()
    assert output.attrs["processing_stage"] == "feature_extraction"
    assert output.attrs["official_groups"] == "SRTM"
    assert output.attrs["crs"] == "EPSG:32650"
