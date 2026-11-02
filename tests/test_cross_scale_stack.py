import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.cross_scale import SemanticCrossScaleStackPlugin
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.core.schemas import EcoDataBundle


def _dataset(name, values, lat, lon, *, kind="continuous", units="1", semantics="cell_mean"):
    dataset = xr.Dataset({name: (("lat", "lon"), np.asarray(values))}, coords={"lat": lat, "lon": lon}, attrs={"crs": "EPSG:4326"})
    dataset[name].attrs.update(units=units, data_kind=kind, spatial_semantics=semantics)
    return EcoDataBundle(dataset)


def test_semantic_cross_scale_stack_uses_category_fractions_and_preserves_raw_units():
    fine = np.linspace(0.0625, 0.9375, 8)
    medium = np.linspace(0.125, 0.875, 4)
    coarse = np.linspace(0.0, 1.0, 2)
    landuse = np.tile(np.array([1, 1, 2, 2, 1, 1, 2, 2]), (8, 1))
    lai = 1 + medium[:, None] + medium[None, :]
    temperature = 280 + 4 * coarse[:, None] + 2 * coarse[None, :]
    context = AlgorithmContext(
        sources=(_dataset("landuse", landuse, fine, fine, kind="categorical"),),
        auxiliaries=(
            _dataset("LAI", lai, medium, medium, units="m2 m-2").dataset,
            _dataset("temperature", temperature, coarse, coarse, units="K").dataset,
        ),
        variable="feature_stack",
        semantic_type="multidimensional_array",
        target_lat=tuple(medium),
        target_lon=tuple(medium),
    )
    result = SemanticCrossScaleStackPlugin().run(context)
    output = result.data.dataset
    fractions = output["landuse__fraction_1"] + output["landuse__fraction_2"]
    np.testing.assert_allclose(fractions, 1.0, atol=1e-6)
    assert output.LAI.attrs["units"] == "m2 m-2"
    assert output.temperature.attrs["units"] == "K"
    assert output.feature_stack.shape[0] == 4
    assert np.isfinite(output.feature_stack.values).all()
    methods = {item["variable"]: item["method"] for item in result.report["transformations"]}
    assert methods["landuse"] == "conservative category fractions"
    assert methods["LAI"] == "conservative"
    assert methods["temperature"] == "bilinear"
    assert result.report["physical_detail_claim"] is False
    assert "scale.semantic_stack" in {plan.id for plan in AlgorithmSelector(create_default_registry()).recommend(context)}
