from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult
from ecodata.io.readers import dataset_to_frame


@dataclass(frozen=True)
class ResidualFusionConfig:
    variable: str = "value"
    interpolation: str = "linear"


class ResidualFieldFusion:
    name = "fusion.residual_field"

    def __init__(self, config: ResidualFusionConfig = ResidualFusionConfig()):
        self.config = config

    def apply(self, background: EcoDataBundle, observations: EcoDataBundle) -> TransformResult:
        started = perf_counter()
        bg = background.dataset
        variable = self.config.variable
        if variable not in bg or not {"lat", "lon"}.issubset(bg[variable].dims):
            raise SchemaError("Background variable must use lat/lon dimensions")
        if bg[variable].ndim != 2:
            raise SchemaError("P2 residual fusion accepts one two-dimensional background field")
        if self.config.interpolation not in {"linear", "nearest"}:
            raise SchemaError("Residual interpolation must be linear or nearest")

        frame = dataset_to_frame(observations.dataset)
        names = frame["variable"].unique()
        expected_name = bg[variable].attrs.get("physical_variable")
        if expected_name is not None and (len(names) != 1 or names[0] != expected_name):
            raise SemanticError("Background and observations represent different physical variables")
        background_unit = bg[variable].attrs.get("units")
        if set(frame["unit"]) != {background_unit}:
            raise SemanticError("Background and observation units are incompatible")
        lat_min, lat_max = float(bg.lat.min()), float(bg.lat.max())
        lon_min, lon_max = float(bg.lon.min()), float(bg.lon.max())
        inside = frame["latitude"].between(lat_min, lat_max) & frame["longitude"].between(lon_min, lon_max)
        frame = frame.loc[inside & frame["qc_flag"].eq("VALID")].copy()
        minimum = 3 if self.config.interpolation == "linear" else 1
        if len(frame) < minimum:
            raise SchemaError(f"Residual fusion requires at least {minimum} valid in-domain observations")

        station_lat = xr.DataArray(frame["latitude"].to_numpy(), dims="station")
        station_lon = xr.DataArray(frame["longitude"].to_numpy(), dims="station")
        predicted = bg[variable].interp(lat=station_lat, lon=station_lon, method="linear").values
        residual = frame["value"].to_numpy(dtype=float) - predicted
        target_lon, target_lat = np.meshgrid(bg["lon"].values, bg["lat"].values)
        points = np.column_stack([frame["longitude"], frame["latitude"]])
        correction = griddata(points, residual, (target_lon, target_lat), method=self.config.interpolation)
        if self.config.interpolation == "linear" and np.isnan(correction).any():
            nearest = griddata(points, residual, (target_lon, target_lat), method="nearest")
            correction = np.where(np.isnan(correction), nearest, correction)

        output = bg.copy(deep=True)
        output["background_value"] = bg[variable].copy(deep=True)
        output["residual_correction"] = (("lat", "lon"), correction)
        output[variable] = bg[variable] + output["residual_correction"]
        output.attrs.update(
            {
                "processing_stage": "fused",
                "fusion_method": "background_residual_interpolation",
                "observation_count": len(frame),
            }
        )
        output[variable].attrs.update(bg[variable].attrs)
        output["background_value"].attrs.update(bg[variable].attrs)
        output["residual_correction"].attrs.update(
            {"units": background_unit, "long_name": "interpolated observation minus background residual"}
        )
        record = TransformRecord(
            method=self.name,
            parameters={"interpolation": self.config.interpolation, "observation_count": len(frame)},
            input_rows=int(bg[variable].size + len(frame)),
            output_rows=int(output[variable].size),
            elapsed_seconds=perf_counter() - started,
        )
        history = background.history + observations.history + (record,)
        bundle = EcoDataBundle(output, history)
        return TransformResult(bundle, record)

