"""Reusable train-station affine calibration over an actual background field."""
from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_stations import station_inputs
from ecodata.algorithms.open_source_spatial import finish
from ecodata.algorithms.quality_aware import fit_affine_source
from ecodata.core.exceptions import SchemaError


def background_stations(context):
    """No evaluation labels, deletion, extrapolation, or guessed time/identity."""
    if len(context.sources) != 1 or context.semantic_type != "continuous" or context.auxiliaries:
        raise SchemaError("One continuous background and station observations required")
    source, target, axes, train, points, query, outside, dropped = station_inputs(context, 3)
    if axes != ("lat", "lon") or not CRS.from_user_input(source.attrs["crs"]).equals(CRS.from_epsg(4326)):
        raise SchemaError("Reviewed background/observation sampling requires EPSG:4326 lat/lon")
    if source[context.variable].dims != axes or not np.isfinite(source[context.variable]).all():
        raise SchemaError("Finite single 2D background required")
    if dropped or not context.observations.split.eq("train").all():
        raise SchemaError("Only finite train observations allowed; no cohort deletion")
    if "time" not in source.coords or source.time.size != 1 or "time" not in train or train.time.isna().any():
        raise SchemaError("Explicit single background/observation time required")
    stamp = pd.to_datetime(np.asarray(source.time).ravel()[0], utc=True)
    if not pd.to_datetime(train.time, utc=True).eq(stamp).all():
        raise SchemaError("Background and observation times differ")
    physical = source[context.variable].attrs.get("physical_variable")
    if not physical or "variable" not in train or not train.variable.eq(physical).all():
        raise SchemaError("Explicit matching physical_variable and observation variable required")
    tx = Transformer.from_crs(context.parameters["observation_crs"], "EPSG:4326", always_xy=True)
    lon, lat = tx.transform(train.longitude.to_numpy(float), train.latitude.to_numpy(float))
    sampled = source[context.variable].interp(lon=xr.DataArray(lon, dims="station"), lat=xr.DataArray(lat, dims="station"))
    if not np.isfinite(sampled).all():
        raise SchemaError("Background must cover every station; no extrapolation")
    return source, train, sampled.values


class StationAffineCalibrationPlugin:
    spec = AlgorithmSpec(
        id="bias.station_affine", name="Train-station affine background calibration", version="1",
        category="bias_correction", status="available", input_kinds=("rectilinear_grid", "station_table"),
        semantic_types=("continuous",), requires_observations=True, requires_training=True,
        capabilities=("bias_correction",), backend="existing fit_affine_source / NumPy least squares",
        license="project", preconditions=("train_only", "same_time", "same_variable", "same_unit"),
        input_contract={"requires_observations": True}, output_contract={"kind": "rectilinear_grid"},
        verification_evidence=("tests/test_collaboration.py",),
        notes="Fits the explicitly available train stations at the input hour; not a future forecast or guaranteed accuracy improvement.")

    def local_context(self, context):
        if not context.sources or not {"lat","lon"} <= set(context.sources[0].dataset.coords):
            raise SchemaError("Station affine calibration requires explicit lat/lon coordinates")
        source = context.sources[0].dataset
        return replace(context, target_lat=tuple(source.lat.values), target_lon=tuple(source.lon.values))

    def compatibility(self, context):
        try:
            background_stations(self.local_context(context))
        except (SchemaError, KeyError, IndexError, ValueError, TypeError) as exc:
            return Compatibility(False, (str(exc),))
        return Compatibility(True)

    def run(self, context):
        started = perf_counter()
        source, train, sampled = background_stations(self.local_context(context))
        slope, intercept = fit_affine_source(sampled, train.value.to_numpy(float))
        output = source.copy(deep=True)
        output[context.variable] = slope * source[context.variable] + intercept
        output[context.variable].attrs = dict(source[context.variable].attrs)
        # Propagate an explicitly supplied input standard deviation under y=a*x+b.
        name = context.parameters.get("uncertainty_variable", "uncertainty")
        if name in output:
            output[name] = abs(slope) * source[name]
            output[name].attrs = dict(source[name].attrs, uncertainty_scope="scaled input uncertainty only; excludes calibration parameter error")
        return finish(context, self.spec, output, dict(slope=slope, intercept=intercept,
            training_count=len(train), fit_split="train", reference_used=False), started)
