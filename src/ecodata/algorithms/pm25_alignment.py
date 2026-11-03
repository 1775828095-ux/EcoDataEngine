"""Window-aware preparation only; not permission to assume station coordinates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.dag import DAGError


INSTANT_UNITS = {"t2m": "K", "d2m": "K", "u10": "m s**-1", "v10": "m s**-1", "sp": "Pa"}


def observation_centers(labels, metadata):
    evidence = metadata["evidence_status"]
    if not (evidence["timezone"].startswith("supported:") and evidence["hour_window"].startswith("supported:")):
        raise DAGError("OBSERVATION_TIME_WINDOW_UNCONFIRMED")
    result = []
    for label in labels:
        stamp = pd.Timestamp(label)
        if pd.isna(stamp):
            raise DAGError("Unparseable station timestamp")
        # An explicit offset is already meaningful; do not apply UTC+8 twice.
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("Asia/Shanghai")
        result.append((stamp.tz_convert("UTC") - pd.Timedelta(minutes=30)).tz_localize(None))
    return pd.DatetimeIndex(result)


def formal_blockers(metadata, geos):
    evidence = metadata["evidence_status"]
    problems = []
    if (evidence.get("coordinate_datum", "").upper() != "CONFIRMED"
            or not metadata.get("confirmed_crs") or not metadata.get("coordinate_datum_evidence")):
        problems.append("STATION_COORDINATE_DATUM_UNCONFIRMED")
    if not evidence.get("pm25_unit", "").startswith("supported:"):
        problems.append("OBSERVATION_UNIT_UNCONFIRMED")
    if not all(evidence.get(k, "").startswith("supported:") for k in ("timezone", "hour_window")):
        problems.append("OBSERVATION_TIME_WINDOW_UNCONFIRMED")
    # Existing lev=72, units=millibar/long_name=altitude is not surface-layer proof.
    if not geos.attrs.get("surface_layer_evidence"):
        problems.append("GEOS_VERTICAL_LAYER_NOT_VERIFIED_AS_SURFACE")
    return problems


def require_formal_spatial_evidence(metadata, geos):
    problems = formal_blockers(metadata, geos)
    if problems:
        raise DAGError("; ".join(problems))


def align_native_covariates(geos, era, centers):
    """Preserve native grid, RH35 and missingness; no station-value sampling."""
    centers = pd.DatetimeIndex(centers).sort_values().unique()
    if centers.empty or centers.hasnans:
        raise DAGError("No valid common windows")
    if any(centers.minute != 30) or any(centers.second != 0) or any(centers.microsecond != 0):
        raise DAGError("GEOS hourly means must be queried at their real HH:30 centers")
    field = geos["pm25_rh35_gcc"]
    if field.attrs.get("units") != "ug m-3" or field.attrs.get("humidity_basis") != "RH35":
        raise DAGError("GEOS RH35 and unit contract missing")
    if geos.time.attrs.get("timezone") != "UTC" or geos.time.attrs.get("cell_methods") != "time: mean":
        raise DAGError("GEOS UTC/hourly mean metadata missing")
    if "HH:30" not in geos.attrs.get("temporal_semantics", ""):
        raise DAGError("GEOS mean center definition missing")
    for dim in ("time", "lat", "lon"):
        idx = geos[dim].to_index()
        if not idx.is_unique or not idx.is_monotonic_increasing:
            raise DAGError(f"GEOS {dim} duplicates or unsorted")
    if not era.valid_time.to_index().is_unique or not era.valid_time.to_index().is_monotonic_increasing:
        raise DAGError("ERA5 time duplicates or unsorted")
    for a, b in ((geos.lat.values, era.latitude.values), (geos.lon.values, era.longitude.values)):
        if not np.array_equal(a, b):
            raise DAGError("Native grids differ; do not silently regrid without a spatial contract")
    before, after = centers-pd.Timedelta(minutes=30), centers+pd.Timedelta(minutes=30)
    if not centers.isin(geos.time.values).all() or not before.isin(era.valid_time.values).all() or not after.isin(era.valid_time.values).all():
        raise DAGError("Missing exact GEOS window or real ERA5 bracketing endpoint")
    output = field.sel(time=centers).to_dataset()
    output.pm25_rh35_gcc.attrs["cell_methods"] = "time: mean"
    for name, unit in INSTANT_UNITS.items():
        if era[name].attrs.get("units") != unit or era[name].attrs.get("GRIB_stepType") != "instant":
            raise DAGError(f"ERA5 {name} unit or instantaneous semantics mismatch")
        midpoint = era[name].interp(valid_time=centers).rename(
            {"valid_time": "time", "latitude": "lat", "longitude": "lon"})
        # Remove unrelated static auxiliary coordinates, not their original source data.
        midpoint = midpoint.reset_coords(drop=True)
        midpoint.attrs = {**era[name].attrs, "cell_methods": "time: point",
                          "temporal_semantics": "linear estimate at UTC window center; NOT hourly mean"}
        output[name] = midpoint
    if era.tp.attrs.get("units") != "m" or era.tp.attrs.get("GRIB_stepType") != "accum" or "1 hour ending" not in era.tp.attrs.get("accumulation_comment", ""):
        raise DAGError("ERA5 precipitation one-hour accumulation contract missing")
    # [center-30min, center+30min] precipitation is stored at the END, never averaged/interpolated.
    rain = era.tp.sel(valid_time=after).rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
    rain = rain.reset_coords(drop=True).assign_coords(time=centers)
    rain.attrs = {**era.tp.attrs, "cell_methods": "time: sum",
                  "temporal_semantics": "one-hour total ending at time+30min; window aligned, NOT midpoint interpolation"}
    output["tp"] = rain
    for name in output.data_vars:
        output[name+"_valid"] = np.isfinite(output[name]).astype("uint8")
        output[name+"_valid"].attrs = {"units": "1", "flag_values": np.array([0, 1], dtype="uint8"), "flag_meanings": "missing valid"}
    output["time_bounds"] = xr.DataArray(np.column_stack([before.values, after.values]), dims=("time", "bounds"))
    output.time.attrs = {"standard_name": "time", "timezone": "UTC", "bounds": "time_bounds"}
    for name in ("time", "time_bounds"):
        output[name].encoding.update(units="seconds since 1970-01-01 00:00:00", calendar="proleptic_gregorian")
    output.attrs = {"role": "time_aligned_native_grid_covariates_NOT_fused_PM25",
                    "resolution_note": "GEOS native 0.25 degree grid; no resolution gain",
                    "station_spatial_matching": "NOT_PERFORMED",
                    "geos_vertical_layer_status": "UNVERIFIED; retain original lev and attrs"}
    return output
