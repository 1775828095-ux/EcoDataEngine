"""Station adapters for official PyKrige and MetPy; no local interpolation kernels."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import Delaunay

from ecodata.algorithms.contracts import AlgorithmResult, AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish, grid_target
from ecodata.core.exceptions import SchemaError


def installed(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def station_inputs(context, minimum=3):
    source, target, axes = grid_target(context)
    table = context.observations
    if table is None or not {"longitude", "latitude", "value", "split", "unit"}.issubset(table):
        raise SchemaError("INPUT_REQUIREMENT_MISSING: longitude/latitude/value/split/unit observations")
    train = table.loc[table.split.eq("train")].copy()
    if "time" in train and train.time.nunique() > 1:
        raise SchemaError("Select a single observation time; do not merge different time windows")
    original_count = len(train)
    train = train.dropna(subset=["longitude", "latitude", "value"])
    if len(train) < minimum or train.duplicated(["longitude", "latitude"]).any():
        raise SchemaError("Insufficient unique training stations or duplicate station positions")
    if not np.isfinite(train[["longitude", "latitude", "value"]].to_numpy(float)).all():
        raise SchemaError("Non-finite station values or coordinates")
    if train.unit.nunique() != 1 or train.unit.iloc[0] != source[context.variable].attrs.get("units"):
        raise SchemaError("Unit mismatch; use UnitEngine before station interpolation")
    observation_crs = context.parameters.get("observation_crs")
    if not observation_crs:
        raise SchemaError("INPUT_REQUIREMENT_MISSING: observation_crs")
    crs = CRS.from_user_input(source.attrs["crs"])
    target_x, target_y = np.meshgrid(target[axes[1]].values, target[axes[0]].values)
    metric = CRS.from_proj4(f"+proj=aeqd +lat_0={float(np.mean(target_y))} +lon_0={float(np.mean(target_x))} +datum=WGS84 +units=m") if crs.is_geographic else crs
    tx = Transformer.from_crs(crs, metric, always_xy=True)
    sx = Transformer.from_crs(observation_crs, metric, always_xy=True)
    x, y = sx.transform(train.longitude.to_numpy(float), train.latitude.to_numpy(float))
    xx, yy = tx.transform(target_x, target_y)
    points = np.column_stack([x, y])
    query = np.column_stack([xx.ravel(), yy.ravel()])
    if np.linalg.matrix_rank(points - points.mean(0)) < 2:
        raise SchemaError("Collinear station positions cannot identify a 2D field")
    outside = Delaunay(points).find_simplex(query) < 0
    return source, target, axes, train, points, query, outside, original_count - len(train)


class PyKrigePlugin:
    def __init__(self, method):
        self.method = method
        dimensional = method.endswith("3d")
        auxiliary = method == "regression"
        caps = ("observation_interpolation", "three_dimensional_interpolation") if dimensional else ("observation_interpolation",)
        if auxiliary:
            caps += ("auxiliary_variable_estimation",)
        if method == "universal":
            caps += ("external_drift_estimation",)
        self.spec = AlgorithmSpec(
            f"station.pykrige_{method}", f"PyKrige {method}", installed("pykrige") or "not installed",
            "observation_interpolation", "available" if installed("pykrige") else "dependency_missing",
            ("station_table", "target_grid"), ("continuous",), requires_observations=True,
            requires_auxiliary=auxiliary, compute="medium", license="BSD-3-Clause", backend="PyKrige",
            source_url="https://github.com/GeoStat-Framework/PyKrige", capabilities=caps,
            preconditions=("training_station_split", "explicit_crs", "same_unit", "z_coordinate" if dimensional else "noncollinear_stations"),
            input_contract={"semantic_types": ["continuous"], "requires_observations": True,
                            "spatial_kind": "stations", "requires_auxiliary": auxiliary},
            output_contract={"kind": "3d_grid" if dimensional else "grid", "uncertainty": method != "regression"},
            verification_evidence=("tests/test_algorithm_library_expansion.py",))

    def compatibility(self, context):
        reasons = []
        if self.spec.status != "available":
            reasons.append("PyKrige dependency missing")
        if context.semantic_type != "continuous":
            reasons.append("Kriging only accepts continuous physical variables")
        try:
            inputs = station_inputs(context, 4)
            if self.method.endswith("3d") and ("z" not in inputs[3] or context.parameters.get("target_z") is None or context.parameters.get("vertical_unit") != "m"):
                reasons.append("INPUT_REQUIREMENT_MISSING: z, target_z and vertical_unit=m")
            if self.method == "regression" and not context.auxiliaries:
                reasons.append("INPUT_REQUIREMENT_MISSING: continuous covariate grid")
            if context.auxiliaries and self.method in {"universal", "universal3d", "regression"}:
                covariate_arrays(context, inputs[3], inputs[1], inputs[2], allow_masked_queries=(
                    self.method == "universal" and context.parameters.get("mask_missing_query_covariates", False)))
        except (SchemaError, IndexError, ValueError) as exc:
            reasons.append(str(exc))
        return Compatibility(not reasons, tuple(reasons), ("Kriging extrapolation is an estimate; variance is model-dependent",))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        source, target, axes, train, points, query, outside, dropped = station_inputs(context, 4)
        values = train.value.to_numpy(float)
        kwargs = {"variogram_model": context.parameters.get("variogram_model", "linear"), "pseudo_inv": True}
        variance = None
        if self.method == "ordinary":
            from pykrige.ok import OrdinaryKriging
            estimator = OrdinaryKriging(points[:, 0], points[:, 1], values, **kwargs)
            prediction, variance = estimator.execute("points", query[:, 0], query[:, 1])
        elif self.method == "universal":
            from pykrige.uk import UniversalKriging
            if context.auxiliaries:
                drift, target_drift = covariate_arrays(context, train, target, axes,
                    allow_masked_queries=context.parameters.get("mask_missing_query_covariates", False))
                kwargs.update(drift_terms=["specified"], specified_drift=[drift[:, i] for i in range(drift.shape[1])])
                execute_kwargs = {"specified_drift_arrays": [target_drift[:, i] for i in range(target_drift.shape[1])]}
            else:
                kwargs.update(drift_terms=["regional_linear"])
                execute_kwargs = {}
            estimator = UniversalKriging(points[:, 0], points[:, 1], values, **kwargs)
            if context.auxiliaries and context.parameters.get("mask_missing_query_covariates", False):
                valid_query = np.isfinite(target_drift).all(axis=1)
                selected = {"specified_drift_arrays": [v[valid_query] for v in execute_kwargs["specified_drift_arrays"]]}
                prediction, variance = np.full(len(query), np.nan), np.full(len(query), np.nan)
                prediction[valid_query], variance[valid_query] = estimator.execute(
                    "points", query[valid_query, 0], query[valid_query, 1], **selected)
            else:
                prediction, variance = estimator.execute("points", query[:, 0], query[:, 1], **execute_kwargs)
        elif self.method == "regression":
            from pykrige.rk import RegressionKriging
            from sklearn.linear_model import Ridge
            drift, target_drift = covariate_arrays(context, train, target, axes)
            estimator = RegressionKriging(regression_model=Ridge(), n_closest_points=min(10, len(train)),
                                         variogram_model=kwargs["variogram_model"], verbose=False)
            estimator.fit(drift, points, values)
            prediction = estimator.predict(target_drift, query)
        else:
            from pykrige.ok3d import OrdinaryKriging3D
            from pykrige.uk3d import UniversalKriging3D
            cls = OrdinaryKriging3D if self.method == "ordinary3d" else UniversalKriging3D
            if self.method == "universal3d":
                kwargs["drift_terms"] = ["regional_linear"]
            estimator = cls(points[:, 0], points[:, 1], train.z.to_numpy(float), values, **kwargs)
            z = np.asarray(context.parameters["target_z"], float)
            prediction, variance = estimator.execute("points", np.tile(query[:, 0], len(z)),
                                                     np.tile(query[:, 1], len(z)), np.repeat(z, len(query)))
        coords = {axis: target[axis] for axis in axes}
        dims = axes
        if self.method.endswith("3d"):
            coords["z"] = context.parameters["target_z"]
            dims = ("z",) + axes
        shape = tuple(len(coords[d]) for d in dims)
        output = xr.Dataset({context.variable: (dims, np.asarray(prediction).reshape(shape))}, coords=coords)
        if variance is not None:
            output["kriging_variance"] = (dims, np.asarray(variance).reshape(shape))
        report = {"backend": "PyKrige", "version": self.spec.version, "method": self.method,
                  "training_count": len(train), "dropped_records": dropped,
                  "outside_station_hull_fraction": float(outside.mean()),
                  "test_used_for_fitting": False, "auxiliary_used": bool(context.auxiliaries)}
        return finish(context, self.spec, output, report, started)


def covariate_arrays(context, train, target, axes, *, allow_masked_queries=False):
    station, query = [], []
    converter = Transformer.from_crs(context.parameters["observation_crs"], context.sources[0].dataset.attrs["crs"], always_xy=True)
    station_x, station_y = converter.transform(train.longitude.values, train.latitude.values)
    for ds in context.auxiliaries:
        if ds.attrs.get("crs") != context.sources[0].dataset.attrs.get("crs"):
            raise SchemaError("Covariate CRS differs from target grid")
        for field in ds.data_vars.values():
            if not set(axes).issubset(field.dims):
                continue
            if field.attrs.get("data_kind") != "continuous" or not field.attrs.get("units"):
                raise SchemaError("Covariate needs confirmed continuous semantics and units")
            station.append(field.interp({axes[1]: xr.DataArray(station_x, dims="station"),
                                         axes[0]: xr.DataArray(station_y, dims="station")}).values)
            # Interpolating an already identical masked grid can propagate a
            # neighbour's NaN into an otherwise valid pixel. No resampling is
            # needed when coordinates are exactly the requested coordinates.
            same_grid = all(np.array_equal(field[a].values, target[a].values) for a in axes)
            query.append((field.transpose(*axes).values if same_grid else
                          field.interp({a: target[a] for a in axes}).values).ravel())
    if not station:
        raise SchemaError("No suitable covariate variables")
    a, b = np.column_stack(station), np.column_stack(query)
    masked = allow_masked_queries
    if (not np.isfinite(a).all() or (not masked and not np.isfinite(b).all())
            or not np.isfinite(b).all(axis=1).any()
            or np.linalg.matrix_rank(np.column_stack([np.ones(len(a)), a])) < a.shape[1] + 1):
        raise SchemaError("Covariate coverage is incomplete or training covariates are rank-deficient")
    return a, b


def station_plugins():
    return tuple(PyKrigePlugin(method) for method in ("ordinary", "universal", "regression", "ordinary3d", "universal3d")) + tuple(
        MetPyStationPlugin(method) for method in ("linear", "natural_neighbor", "barnes", "cressman")) + (MetPyVerticalPlugin(),)


class MetPyStationPlugin:
    def __init__(self, method):
        self.method = method
        self.spec = AlgorithmSpec(
            f"station.metpy_{method}", f"MetPy {method}", installed("metpy") or "not installed",
            "observation_interpolation", "available" if installed("metpy") else "dependency_missing",
            ("station_table", "target_grid"), ("continuous",), requires_observations=True,
            license="BSD-3-Clause", backend="MetPy", source_url="https://github.com/Unidata/MetPy",
            capabilities=("observation_interpolation", "irregular_spatial_interpolation"),
            preconditions=("training_station_split", "explicit_crs", "same_unit", "noncollinear_stations"),
            input_contract={"semantic_types": ["continuous"], "spatial_kind": "stations", "requires_observations": True},
            output_contract={"kind": "grid", "outside_hull": "missing" if method in {"linear", "natural_neighbor"} else "radius_dependent"},
            verification_evidence=("tests/test_algorithm_library_expansion.py",))

    def compatibility(self, context):
        reasons = []
        if self.spec.status != "available":
            reasons.append("MetPy dependency missing")
        if context.semantic_type != "continuous":
            reasons.append("Station interpolation requires continuous data")
        try:
            station_inputs(context)
        except (SchemaError, IndexError, ValueError) as exc:
            reasons.append(str(exc))
        return Compatibility(not reasons, tuple(reasons), ("Uncovered regions remain NaN; no hidden extrapolation",))

    def run(self, context):
        from metpy.interpolate import interpolate_to_points
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        source, target, axes, train, points, query, outside, dropped = station_inputs(context)
        options = {key: context.parameters[key] for key in ("minimum_neighbors", "gamma", "kappa_star", "search_radius") if key in context.parameters}
        values = interpolate_to_points(points, train.value.to_numpy(float), query, interp_type=self.method, **options)
        output = xr.Dataset({context.variable: (axes, np.asarray(values).reshape(tuple(target.sizes[a] for a in axes)))},
                            coords={axis: target[axis] for axis in axes})
        report = {"backend": "MetPy", "version": self.spec.version, "method": self.method,
                  "training_count": len(train), "dropped_records": dropped,
                  "outside_station_hull_fraction": float(outside.mean()),
                  "nan_fraction": float(np.isnan(values).mean()), "test_used_for_fitting": False}
        return finish(context, self.spec, output, report, started)


class MetPyVerticalPlugin:
    def __init__(self):
        self.spec = AlgorithmSpec(
            "vertical.metpy", "MetPy vertical interpolation", installed("metpy") or "not installed",
            "vertical_interpolation", "available" if installed("metpy") else "dependency_missing",
            ("vertical_profile",), ("continuous",), license="BSD-3-Clause", backend="MetPy",
            source_url="https://github.com/Unidata/MetPy", capabilities=("vertical_interpolation",),
            preconditions=("vertical_coordinate", "explicit_vertical_unit", "target_levels"),
            input_contract={"semantic_types": ["continuous"], "vertical": True},
            output_contract={"kind": "vertical_profile"}, verification_evidence=("tests/test_algorithm_library_expansion.py",))

    def compatibility(self, context):
        reasons = []
        if self.spec.status != "available" or context.semantic_type != "continuous":
            reasons.append("MetPy unavailable or incompatible semantics")
        if not context.sources:
            return Compatibility(False, ("Missing source",))
        ds = context.sources[0].dataset
        axis = context.parameters.get("vertical_coordinate")
        if not axis or axis not in ds.coords or not ds[axis].attrs.get("units") or context.parameters.get("target_levels") is None:
            reasons.append("INPUT_REQUIREMENT_MISSING: vertical coordinate, unit and target_levels")
        elif np.min(context.parameters["target_levels"]) < float(ds[axis].min()) or np.max(context.parameters["target_levels"]) > float(ds[axis].max()):
            reasons.append("Vertical extrapolation is not allowed")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        from metpy.interpolate import interpolate_1d, log_interpolate_1d
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        source = context.sources[0].dataset
        axis = context.parameters["vertical_coordinate"]
        field = source[context.variable].transpose(axis, ...)
        levels = np.asarray(context.parameters["target_levels"], float)
        fn = log_interpolate_1d if context.parameters.get("log_vertical", False) else interpolate_1d
        values = fn(levels, source[axis].values, field.values, axis=0)
        output = xr.Dataset({context.variable: (field.dims, values)},
                            coords={a: levels if a == axis else field[a] for a in field.dims})
        output[axis].attrs = source[axis].attrs.copy()
        return finish(context, self.spec, output, {"backend": "MetPy", "log_vertical": context.parameters.get("log_vertical", False)}, started)
