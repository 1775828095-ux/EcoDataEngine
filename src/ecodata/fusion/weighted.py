from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult


@dataclass(frozen=True)
class WeightedFusionConfig:
    variable: str = "value"
    uncertainty_variable: str = "uncertainty"


class UncertaintyWeightedFusion:
    name = "fusion.uncertainty_weighted"

    def __init__(self, config: WeightedFusionConfig = WeightedFusionConfig()):
        self.config = config

    def apply(self, sources: tuple[EcoDataBundle, ...] | list[EcoDataBundle]) -> TransformResult:
        if len(sources) < 2:
            raise SchemaError("Weighted fusion requires at least two sources")
        started = perf_counter()
        first = sources[0].dataset
        variable = self.config.variable
        uncertainty = self.config.uncertainty_variable
        if variable not in first or uncertainty not in first:
            raise SchemaError("Fusion source is missing value or uncertainty")
        unit = first[variable].attrs.get("units")
        physical_variable = first[variable].attrs.get("physical_variable")
        if not physical_variable:
            raise SchemaError("Fusion variable requires a physical_variable attribute")
        aligned_values = []
        aligned_uncertainties = []
        for source in sources:
            dataset = source.dataset
            if variable not in dataset or uncertainty not in dataset:
                raise SchemaError("Fusion source is missing value or uncertainty")
            if dataset[variable].attrs.get("units") != unit:
                raise SemanticError("Fusion sources have incompatible units")
            if dataset[variable].attrs.get("physical_variable") != physical_variable:
                raise SemanticError("Fusion sources represent different physical variables")
            if first.attrs.get("crs") != dataset.attrs.get("crs"):
                from rasterio.crs import CRS
                if not first.attrs.get("crs") or not dataset.attrs.get("crs") or CRS.from_user_input(first.attrs["crs"]) != CRS.from_user_input(dataset.attrs["crs"]):
                    raise SemanticError("Fusion sources have incompatible CRS; align before fusion")
            try:
                xr.align(first[variable], dataset[variable], join="exact")
            except ValueError as exc:
                raise SchemaError("Fusion sources must have exactly matching coordinates") from exc
            if np.any(dataset[uncertainty].values <= 0):
                raise SchemaError("Fusion uncertainty must be positive")
            aligned_values.append(dataset[variable])
            aligned_uncertainties.append(dataset[uncertainty])

        values = xr.concat(aligned_values, dim="source")
        sigma = xr.concat(aligned_uncertainties, dim="source")
        weights = 1.0 / sigma**2
        weights = weights.where(np.isfinite(values) & np.isfinite(sigma) & (sigma > 0), 0)
        fused = (values * weights).sum("source") / weights.sum("source")
        fused_uncertainty = np.sqrt(1.0 / weights.sum("source"))
        output = xr.Dataset({variable: fused, uncertainty: fused_uncertainty})
        output.attrs.update(first.attrs)
        output.attrs.update(
            {"processing_stage": "fused", "fusion_method": "inverse_variance", "source_count": len(sources)}
        )
        output[variable].attrs.update(first[variable].attrs)
        output[uncertainty].attrs.update(first[uncertainty].attrs)
        histories = tuple(record for source in sources for record in source.history)
        record = TransformRecord(
            method=self.name,
            parameters={"source_count": len(sources), "variable": variable},
            input_rows=sum(int(source.dataset[variable].size) for source in sources),
            output_rows=int(output[variable].size),
            elapsed_seconds=perf_counter() - started,
        )
        bundle = EcoDataBundle(output, histories + (record,))
        return TransformResult(bundle, record)
