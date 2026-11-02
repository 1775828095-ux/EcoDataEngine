from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ecodata.core.exceptions import SemanticError
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.metrics import relative_conservation_error
from ecodata.spatial.regridding import GridRegridder, RegridConfig


def grid_bundle(lat, lon, values, spatial_semantics="cell_mean", data_kind="continuous"):
    ds = xr.Dataset(
        {"value": (("lat", "lon"), np.asarray(values))},
        coords={"lat": lat, "lon": lon},
        attrs={
            "dataset_id": "spatial-synthetic",
            "crs": "EPSG:4326",
            "temporal_semantics": "instantaneous",
            "spatial_semantics": spatial_semantics,
            "data_kind": data_kind,
        },
    )
    ds["value"].attrs.update(
        {
            "units": "ug m-3",
            "physical_variable": "PM2.5",
            "data_kind": data_kind,
            "temporal_semantics": "instantaneous",
            "spatial_semantics": spatial_semantics,
        }
    )
    return EcoDataBundle(ds)


def test_bilinear_recovers_linear_truth() -> None:
    lat = np.array([0.0, 2.0])
    lon = np.array([0.0, 2.0])
    source = grid_bundle(lat, lon, 10 + 2 * lat[:, None] + 3 * lon[None, :])
    transform = GridRegridder(RegridConfig("bilinear", (0.0, 1.0, 2.0), (0.0, 1.0, 2.0)))
    output = transform.apply(source).data.dataset
    expected = 10 + 2 * output.lat.values[:, None] + 3 * output.lon.values[None, :]
    np.testing.assert_allclose(output["value"], expected, atol=1e-12)
    assert output.attrs["spatial_regridding_method"] == "bilinear"


def test_nearest_preserves_source_categories() -> None:
    source = grid_bundle([0, 2], [0, 2], [[1, 2], [3, 4]], "point", "categorical")
    output = GridRegridder(RegridConfig("nearest", (0, 1, 2), (0, 1, 2))).apply(source).data.dataset
    assert set(np.unique(output["value"])) <= {1, 2, 3, 4}


def test_conservative_regrid_preserves_area_integral() -> None:
    coordinates = [-1.5, -0.5, 0.5, 1.5]
    source = grid_bundle(coordinates, coordinates, np.arange(1.0, 17.0).reshape(4, 4))
    output = GridRegridder(
        RegridConfig("conservative", (-1.0, 1.0), (-1.0, 1.0))
    ).apply(source).data.dataset
    assert relative_conservation_error(source.dataset["value"], output["value"]) < 1e-12


def test_conservative_rejects_cell_total_without_explicit_conversion() -> None:
    source = grid_bundle([0, 1], [0, 1], [[1.0, 2.0], [3.0, 4.0]], "cell_total")
    with pytest.raises(SemanticError, match="cell_mean or flux"):
        GridRegridder(RegridConfig("conservative", (0, 0.5, 1), (0, 0.5, 1))).apply(source)
