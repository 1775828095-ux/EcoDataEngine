from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

from ecodata.core.exceptions import SchemaError


@dataclass(frozen=True)
class ReconstructionHints:
    """Optional high-to-low metadata contract for future reverse reconstruction."""

    variable: str
    peak_values: tuple[float, ...]
    peak_indices: tuple[tuple[int, int], ...]
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict:
        return asdict(self)


def extract_reconstruction_hints(field: xr.DataArray, top_k: int = 3) -> ReconstructionHints:
    if not {"lat", "lon"}.issubset(field.dims) or field.ndim != 2:
        raise SchemaError("Reconstruction hints require one two-dimensional lat/lon field")
    if top_k < 1:
        raise SchemaError("top_k must be positive")
    values = np.asarray(field.values, dtype=float)
    count = min(top_k, values.size)
    flat_indices = np.argpartition(values.ravel(), -count)[-count:]
    flat_indices = flat_indices[np.argsort(values.ravel()[flat_indices])[::-1]]
    locations = tuple(tuple(map(int, np.unravel_index(index, values.shape))) for index in flat_indices)
    return ReconstructionHints(
        variable=field.name or "value",
        peak_values=tuple(float(values[index]) for index in locations),
        peak_indices=locations,
        mean=float(values.mean()),
        standard_deviation=float(values.std()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )
