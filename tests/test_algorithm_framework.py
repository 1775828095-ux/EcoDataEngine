import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms import (
    AlgorithmContext,
    AlgorithmPlan,
    AlgorithmSelector,
    TargetScaleAdvisor,
    UnifiedAlgorithmPipeline,
    create_default_registry,
)
from ecodata.core.exceptions import ValidationError
from ecodata.core.schemas import EcoDataBundle


def grid(kind="continuous"):
    ds = xr.Dataset(
        {"pm25": (("lat", "lon"), [[10.0, 20.0], [30.0, 40.0]])},
        coords={"lat": [39.0, 40.0], "lon": [116.0, 117.0]},
        attrs={"crs": "EPSG:4326"},
    )
    ds.pm25.attrs.update(units="ug m-3", data_kind=kind, physical_variable="PM25", spatial_semantics="cell_mean")
    return EcoDataBundle(ds)


def observations():
    return pd.DataFrame({
        "station_id": ["A", "B", "C", "V"],
        "longitude": [116.0, 117.0, 116.0, 116.5],
        "latitude": [39.0, 39.0, 40.0, 39.5],
        "value": [12.0, 18.0, 28.0, 22.0],
        "unit": ["ug m-3"] * 4,
        "split": ["train", "train", "train", "validation"],
    })


def test_registry_distinguishes_executable_and_research_only_algorithms():
    specs = {item.id: item for item in create_default_registry().specs()}
    assert specs["spatial.bilinear"].status == "available"
    assert specs["super_resolution.zssr"].status == "research_only"
    assert specs["super_resolution.thera"].requires_weights is True


def test_scale_advisor_warns_that_grid_density_is_not_accuracy():
    context = AlgorithmContext((grid(),), "pm25", "continuous", (39.0, 39.5, 40.0), (116.0, 116.5, 117.0))
    decision = TargetScaleAdvisor().decide(context)
    assert decision.recommended_method == "spatial.bilinear"
    assert any("不等于真实空间精度" in item for item in decision.warnings)


def test_router_excludes_unimplemented_algorithms_and_recommends_existing_backend():
    registry = create_default_registry()
    context = AlgorithmContext((grid(),), "pm25", "continuous", (39.0, 39.5, 40.0), (116.0, 116.5, 117.0))
    plans = AlgorithmSelector(registry).recommend(context)
    ids = {item.id for item in plans}
    assert "spatial.bilinear" in ids
    assert "super_resolution.zssr" not in ids


def test_pipeline_regrids_and_reports_no_accuracy_without_reference():
    registry = create_default_registry()
    context = AlgorithmContext((grid(),), "pm25", "continuous", (39.0, 39.5, 40.0), (116.0, 116.5, 117.0))
    result = UnifiedAlgorithmPipeline(registry).run(context, AlgorithmPlan("bilinear", ("spatial.bilinear",), 1, ("test",)))
    assert result.data.dataset.pm25.shape == (3, 3)
    assert result.report["quality"]["status"] == "PASS"
    assert result.metrics["independent_accuracy_available"] is False


def test_idw_uses_train_rows_and_preserves_unit():
    registry = create_default_registry()
    context = AlgorithmContext((grid(),), "pm25", "continuous", observations=observations())
    result = UnifiedAlgorithmPipeline(registry).run(context, AlgorithmPlan("idw", ("fusion.idw",), 1, ("test",)))
    assert np.isfinite(result.data.dataset.pm25).all()
    assert result.data.dataset.pm25.attrs["units"] == "ug m-3"
    assert result.report["steps"][0]["report"]["training_count"] == 3


def test_final_test_reference_is_rejected_for_algorithm_selection():
    registry = create_default_registry()
    reference = grid().dataset.pm25
    context = AlgorithmContext((grid(),), "pm25", "continuous", (39.0, 40.0), (116.0, 117.0),
                               validation_reference=reference, validation_role="test")
    with pytest.raises(ValidationError, match="最终 test"):
        UnifiedAlgorithmPipeline(registry).run(context, AlgorithmPlan("bilinear", ("spatial.bilinear",), 1, ("test",)))


def test_selector_uses_validation_reference_and_never_test_role():
    registry = create_default_registry()
    reference = xr.DataArray(
        [[10.0, 15.0, 20.0], [20.0, 25.0, 30.0], [30.0, 35.0, 40.0]],
        dims=("lat", "lon"), coords={"lat": [39.0, 39.5, 40.0], "lon": [116.0, 116.5, 117.0]},
    )
    context = AlgorithmContext((grid(),), "pm25", "continuous", tuple(reference.lat.values), tuple(reference.lon.values),
                               validation_reference=reference, validation_role="validation")
    plans = (
        AlgorithmPlan("nearest", ("spatial.nearest",), 1, ("candidate",)),
        AlgorithmPlan("bilinear", ("spatial.bilinear",), 1, ("candidate",)),
    )
    selection = AlgorithmSelector(registry).select(context, plans)
    assert selection.selected.id == "bilinear"
    assert len(selection.evaluations) == 2

