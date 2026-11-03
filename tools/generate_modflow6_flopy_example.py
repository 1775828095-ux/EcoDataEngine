"""Generate the small, known-value FloPy integration example."""
from pathlib import Path

import numpy as np
import xarray as xr

from ecodata.io.structures import export_structure
from ecodata.models import RequirementStore
from ecodata.models.scenario import execute_scenario, recognise_inputs, suggest_roles


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "open_source_reuse" / "modflow6_flopy"


def main():
    shape = (2, 3)
    boundary = np.full(shape, np.nan)
    boundary[:, 0] = [9.5, 9.0]
    dataset = xr.Dataset({
        "top": (("y", "x"), np.full(shape, 10.0)),
        "bottom": (("y", "x"), np.zeros(shape)),
        "hydraulic_conductivity": (("y", "x"), np.full(shape, 2.5)),
        "initial_head": (("y", "x"), np.full(shape, 9.0)),
        "recharge": (("y", "x"), np.full(shape, 0.001)),
        "constant_head": (("y", "x"), boundary),
    }, coords={"x": [500.0, 1500.0, 2500.0], "y": [500.0, 1500.0]}, attrs={
        "structure": "grid", "crs": "EPSG:32618", "delr": 1000.0, "delc": 1000.0,
        "length_units": "METERS", "time_units": "DAYS", "period_length": 1.0, "nstp": 1,
    })
    for name in ("top", "bottom", "initial_head", "constant_head"):
        dataset[name].attrs["units"] = "m"
    for name in ("hydraulic_conductivity", "recharge"):
        dataset[name].attrs["units"] = "m/day"
    content = export_structure(dataset, "NetCDF")[0]
    inputs = recognise_inputs([("groundwater_grid.nc", content)])
    profile = RequirementStore().get("modflow6")
    result = execute_scenario(profile, inputs, suggest_roles(profile, inputs))
    output = CASE / "output"
    output.mkdir(parents=True, exist_ok=True)
    (CASE / "groundwater_grid.nc").write_bytes(content)
    for name, payload in result.files.items():
        (output / name).write_bytes(payload)


if __name__ == "__main__":
    main()
