import json

import numpy as np
import xarray as xr
import pytest

from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import export_structure
from ecodata.models import RequirementStore
from ecodata.models.scenario import execute_scenario, recognise_inputs, suggest_roles


def _input_bytes(include_boundary=True):
    shape = (2, 3)
    boundary = np.full(shape, np.nan)
    if include_boundary:
        boundary[:, 0] = [9.5, 9.0]
    ds = xr.Dataset(
        {
            "top": (("y", "x"), np.full(shape, 10.0)),
            "bottom": (("y", "x"), np.full(shape, 0.0)),
            "hydraulic_conductivity": (("y", "x"), np.full(shape, 2.5)),
            "initial_head": (("y", "x"), np.full(shape, 9.0)),
            "recharge": (("y", "x"), np.full(shape, 0.001)),
            "constant_head": (("y", "x"), boundary),
        },
        coords={"x": [500.0, 1500.0, 2500.0], "y": [500.0, 1500.0]},
        attrs={
            "structure": "grid", "crs": "EPSG:32618", "delr": 1000.0,
            "delc": 1000.0, "length_units": "METERS", "time_units": "DAYS",
            "period_length": 1.0, "nstp": 1,
        },
    )
    for name in ("top", "bottom", "initial_head", "constant_head"):
        ds[name].attrs["units"] = "m"
    for name in ("hydraulic_conductivity", "recharge"):
        ds[name].attrs["units"] = "m/day"
    return export_structure(ds, "NetCDF")[0]


def test_recognise_standardise_flopy_write_and_reload():
    profile = RequirementStore().get("modflow6")
    inputs = recognise_inputs([("groundwater_grid.nc", _input_bytes())])
    assignments = suggest_roles(profile, inputs)
    assert assignments == {"groundwater_grid": "groundwater_grid.nc"}

    result = execute_scenario(profile, inputs, assignments)
    expected = {
        "mfsim.nam", "ecodata.tdis", "ecodata.ims", "groundwater.nam",
        "groundwater.dis", "groundwater.ic", "groundwater.npf",
        "groundwater.rcha", "groundwater.chd", "groundwater.oc",
        "modflow6_manifest.json",
    }
    assert expected <= set(result.files)
    manifest = json.loads(result.files["modflow6_manifest.json"])
    assert manifest["grid"] == {"nlay": 1, "nrow": 2, "ncol": 3, "delr": 1000.0, "delc": 1000.0}
    assert manifest["boundary_cells"] == 2
    assert result.quality["native_format"]["verified"]
    assert not result.quality["completion"]["native_model_input_verified"]
    assert not manifest["validation"]["official_model_run"]


def test_missing_boundary_fails_before_native_package_output():
    profile = RequirementStore().get("modflow6")
    inputs = recognise_inputs([("groundwater_grid.nc", _input_bytes(False))])
    with pytest.raises(SchemaError, match="constant_head"):
        execute_scenario(profile, inputs, {"groundwater_grid": "groundwater_grid.nc"})
