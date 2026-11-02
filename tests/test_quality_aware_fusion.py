import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.quality_aware import QualityAwareFusionPlugin
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.core.schemas import EcoDataBundle


def _case(with_validation=True):
    lat = np.linspace(39, 40, 8)
    lon = np.linspace(115, 116, 8)
    truth = 35 + 12 * (lat[:, None] - 39) + 8 * (lon[None, :] - 115) + 3 * np.sin((lat[:, None] - 39) * np.pi)
    source_a = 1.18 * truth + 18
    source_b = 0.82 * truth - 5 + 2 * np.cos((lon[None, :] - 115) * np.pi)

    def bundle(values, source_id):
        ds = xr.Dataset({"pm25": (("lat", "lon"), values.astype(np.float32))}, coords={"lat": lat, "lon": lon}, attrs={"crs": "EPSG:4326", "source_id": source_id})
        ds.pm25.attrs.update(units="ug m-3", data_kind="continuous", physical_variable="PM2.5")
        return EcoDataBundle(ds)

    rows = []
    points = [(i, j) for i in range(8) for j in range(8)]
    for number, (i, j) in enumerate(points[:30]):
        split = "train" if number < 16 else "validation" if number < 24 and with_validation else "test"
        if number >= 24:
            split = "test"
        rows.append({"latitude": lat[i], "longitude": lon[j], "value": truth[i, j], "split": split})
    observations = pd.DataFrame(rows)
    context = AlgorithmContext(
        sources=(bundle(source_a, "biased_high"), bundle(source_b, "biased_low")),
        variable="pm25",
        semantic_type="continuous",
        observations=observations,
        model_constraints={"nonnegative": True},
    )
    return context


def test_quality_aware_fusion_requires_validation_reference():
    result = QualityAwareFusionPlugin().compatibility(_case(with_validation=False))
    assert not result.eligible
    assert any("验证观测" in reason for reason in result.reasons)


def test_quality_aware_fusion_corrects_bias_weights_sources_and_keeps_test_independent():
    context = _case()
    result = QualityAwareFusionPlugin().run(context)
    report = result.report
    assert report["status"] == "PASS"
    assert report["test_used_for_fitting_or_weighting"] is False
    assert abs(sum(source["weight"] for source in report["sources"]) - 1) < 1e-12
    assert report["test_fused_independent"]["count"] == 6
    raw_best = min(source["validation_raw"]["RMSE"] for source in report["sources"])
    assert report["validation_fused"]["RMSE"] < raw_best
    output = result.data.dataset
    assert output.pm25.dtype == np.float32
    assert np.isfinite(output.pm25.values).all()
    assert (output.pm25.values >= 0).all()
    assert output.fusion_uncertainty.attrs["units"] == "ug m-3"
    assert "fusion.quality_aware" in {plan.id for plan in AlgorithmSelector(create_default_registry()).recommend(context)}
