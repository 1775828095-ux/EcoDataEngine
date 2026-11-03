"""Preflight only: not an executable or verified PhyDL-NWP adapter."""
from ecodata.algorithms.contracts import Compatibility

REQUIRED = {"u100": "m s-1", "v100": "m s-1", "u10": "m s-1", "v10": "m s-1", "t2m": "K",
            "msl": "Pa", "sp": "Pa", "tp": "m", "elevation": "m"}


def check_phydl_inputs(dataset):
    reasons = []
    if not dataset.attrs.get("crs"):
        reasons.append("Explicit CRS required")
    for coord in ("latitude", "longitude"):
        if coord not in dataset.coords:
            reasons.append("Missing coordinate: " + coord)
    for name, unit in REQUIRED.items():
        if name not in dataset:
            reasons.append("Missing required input: " + name)
        elif dataset[name].attrs.get("units") != unit:
            reasons.append(name + " requires standardized unit " + unit)
    # Shared xarray indexes establish alignment only when all weather and DEM
    # arrays actually use the declared geographic dimensions.
    for name in REQUIRED:
        if name in dataset and not {"latitude", "longitude"}.issubset(dataset[name].dims):
            reasons.append("Unaligned geographic dimensions: " + name)
    return Compatibility(not reasons, tuple(reasons),
        ("Preflight does not make official data/weights available or establish complete terrain-aware execution",))
