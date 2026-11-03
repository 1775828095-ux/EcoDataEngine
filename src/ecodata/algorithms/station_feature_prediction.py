"""Real covariate-to-station-target estimation, not physical-field averaging."""
from time import perf_counter
import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.feature_forest import fit_station_forest
from ecodata.algorithms.open_source_spatial import finish
from ecodata.algorithms.station_calibration import background_stations
from ecodata.core.exceptions import SchemaError


class StationFeatureForest:
    def __init__(self, group):
        self.group = group
        self.spec = AlgorithmSpec(
            id="fusion.station_rf_" + group, name="Station-target feature forest: " + group,
            version="1", category="multi_source_fusion", status="available",
            input_kinds=("rectilinear_grid", "station_table", "covariate_grid"), semantic_types=("continuous",),
            requires_training=True, requires_observations=True, requires_auxiliary=True,
            compute="medium", backend="existing station-group forest / scikit-learn", license="BSD-3-Clause",
            capabilities=("multi_source_fusion", "auxiliary_variable_estimation"),
            input_contract={"requires_observations": True}, output_contract={"kind": "rectilinear_grid"},
            notes="Fits available same-hour stations only; independent covariates retain their units/definitions; not a forecast.")

    def inputs(self, context):
        from dataclasses import replace
        source, train, _ = background_stations(replace(context, auxiliaries=()))
        if len(context.auxiliaries) != 1 or not train.split.eq("train").all() or "station_id" not in train:
            raise ValueError("One explicit covariate grid and train-only station IDs required")
        fields = context.parameters.get("feature_groups", {}).get(self.group)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("Declare independent covariate names for the feature group")
        cov = context.auxiliaries[0]
        if "time" not in cov.coords or not np.array_equal(cov.time, source.time):
            raise ValueError("Covariate and observation windows differ")
        if context.parameters.get("observation_crs") != "EPSG:4326":
            raise ValueError("Feature sampling currently requires explicit EPSG:4326 observations")
        for name in fields:
            if name not in cov or cov[name].dims != ("lat", "lon") or not cov[name].attrs.get("units"):
                raise ValueError("Missing finite 2D covariate with its own unit: " + name)
        table = train.copy()
        for name in fields:
            table[name] = cov[name].interp(lon=xr.DataArray(train.longitude, dims="station"),
                                           lat=xr.DataArray(train.latitude, dims="station")).values
        if len(train) < 4 or not np.isfinite(table[list(fields)].to_numpy(float)).all():
            raise ValueError("Insufficient fitting stations or incomplete source coverage")
        return source, table, cov, fields

    def compatibility(self, context):
        try:
            self.inputs(context)
            return Compatibility(True, ())
        except (SchemaError, ValueError, KeyError, TypeError) as exc:
            return Compatibility(False, (str(exc),))

    def run(self, context):
        started = perf_counter()
        source, train, cov, fields = self.inputs(context)
        model, oof, folds = fit_station_forest(train, fields, compute_oof=False)
        lat, lon = list(context.target_lat), list(context.target_lon)
        x = np.column_stack([cov[name].interp(lat=lat, lon=lon).values.ravel() for name in fields])
        if not np.isfinite(x).all():
            raise ValueError("Incomplete query covariates; no cohort deletion")
        ds = xr.Dataset({context.variable: (("lat", "lon"), model.predict(x).reshape(len(lat), len(lon)))},
                        coords=dict(lat=lat, lon=lon, time=source.time), attrs=dict(source.attrs))
        ds[context.variable].attrs = dict(source[context.variable].attrs)
        ds.attrs["data_role"] = "station_target_estimate"
        return finish(context, self.spec, ds, dict(features=list(fields), fit_stations=train.station_id.tolist(),
            folds=folds, station_group_oof_rmse=None,
            reference_used=False, scope="same-hour retrospective spatial reconstruction"), started)
