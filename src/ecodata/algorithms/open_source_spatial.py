"""Thin Registry adapters; all numerical kernels belong to the named upstreams."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import numpy as np
import xarray as xr
from pyproj import CRS

from ecodata.algorithms.contracts import AlgorithmResult, AlgorithmSpec, Compatibility
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord
from ecodata.spatial.regridding import normalize_grid_axes


def grid_target(context):
    source = context.sources[0].dataset
    axes = ("lat", "lon") if {"lat", "lon"}.issubset(source.dims) else ("y", "x")
    target = context.parameters.get("target_grid")
    if target is None:
        if context.target_lat is None or context.target_lon is None:
            raise SchemaError("INPUT_REQUIREMENT_MISSING: target_grid or target coordinates")
        target = xr.Dataset(coords={axes[0]: list(context.target_lat), axes[1]: list(context.target_lon)})
    for ds in (source, target):
        for axis in axes:
            if axis not in ds or ds[axis].ndim != 1 or ds[axis].size < 2:
                raise SchemaError("INPUT_REQUIREMENT_MISSING: two-dimensional rectilinear grid")
    source, _ = normalize_grid_axes(source, axes)
    target, _ = normalize_grid_axes(target, axes)
    if context.variable not in source or not set(axes).issubset(source[context.variable].dims):
        raise SchemaError("INPUT_REQUIREMENT_MISSING: gridded target variable")
    if not source.attrs.get("crs"):
        raise SchemaError("INPUT_REQUIREMENT_MISSING: CRS")
    if target.attrs.get("crs") and CRS.from_user_input(target.attrs["crs"]) != CRS.from_user_input(source.attrs["crs"]):
        raise SchemaError("CRS differs; reproject before regridding")
    return source, target, axes


def finish(context, spec, output, report, started):
    output.attrs = {**context.sources[0].dataset.attrs, **output.attrs}
    output.attrs.update(algorithm_backend=spec.backend, algorithm_id=spec.id)
    if context.variable in output:
        output[context.variable].attrs = {**context.sources[0].dataset[context.variable].attrs,
                                         **output[context.variable].attrs}
    for name in output.data_vars:
        if hasattr(output[name].data, "todense"):
            output[name].data = output[name].data.todense()
    record = TransformRecord(spec.id, report, context.sources[0].dataset[context.variable].size,
                             output[context.variable].size, perf_counter() - started)
    return AlgorithmResult(EcoDataBundle(output, context.sources[0].history + (record,)), spec.id, report)


class XarrayAggregationPlugin:
    def __init__(self, method):
        self.method = method
        category = method in {"most_common", "fractions"}
        caps = {"most_common": ("categorical_scale_processing", "categorical_aggregation"),
                "fractions": ("fractional_categorical_aggregation", "categorical_aggregation", "categorical_scale_processing"),
                "cubic": ("continuous_spatial_resampling",),
                "mean": ("continuous_spatial_resampling","continuous_spatial_aggregation")}.get(method, ("continuous_spatial_aggregation",))
        self.spec = AlgorithmSpec(
            f"spatial.xarray_{method}", f"xarray-regrid {method}", version("xarray-regrid"),
            "single_source_scale", "available", ("rectilinear_grid",),
            ("categorical",) if category else ("continuous",), license="Apache-2.0",
            backend="xarray-regrid", source_url="https://github.com/xarray-contrib/xarray-regrid",
            capabilities=caps, preconditions=("crs", "target_grid", "coarsening" if method != "cubic" else "grid"),
            input_contract={"source_arity":1,"semantic_types": ["categorical"] if category else ["continuous"],
                            "minimum_axis_samples": 4 if method == "cubic" else 2,
                            "spatial_statistic": method if method!="cubic" else None,
                            "direction": "aggregation" if method != "cubic" else "either"},
            output_contract={"kind": "category_fractions" if method == "fractions" else "rectilinear_grid",
                             "semantic_type": "continuous" if method == "fractions" else ("categorical" if category else "continuous")},
            verification_evidence=("tests/test_algorithm_library_expansion.py",))

    def compatibility(self, context):
        reasons = []
        if context.semantic_type not in self.spec.semantic_types:
            reasons.append("Variable semantics incompatible with this aggregation")
        try:
            source, target, axes = grid_target(context)
            if self.method == "cubic" and any(source.sizes[a] < 4 for a in axes):
                reasons.append("Cubic interpolation requires at least four source samples per axis")
            if self.method != "cubic" and any(np.median(np.diff(target[a])) < np.median(np.diff(source[a])) for a in axes):
                reasons.append("Aggregation cannot refine the source grid")
            if self.method == "fractions" and context.parameters.get("spatial_semantics", source[context.variable].attrs.get("spatial_semantics")) not in {"category", "cell_category"}:
                reasons.append("INPUT_REQUIREMENT_MISSING: confirmed cell-category support")
        except (SchemaError, IndexError, ValueError) as exc:
            reasons.append(str(exc))
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        import xarray_regrid  # noqa: F401
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        source, target, axes = grid_target(context)
        field = source[context.variable]
        report = {"backend": "xarray-regrid", "version": self.spec.version, "method": self.method,
                  "information_scale_increased": False}
        if self.method == "most_common":
            if not np.isfinite(field.values).all() or not np.equal(field.values, np.floor(field.values)).all():
                raise SchemaError("most_common requires finite integer category codes; missing categories need explicit handling")
            field = field.astype(np.int64)
            values = np.unique(field.values[np.isfinite(field.values)])
            output = field.regrid.most_common(target, values=values).to_dataset(name=context.variable)
        elif self.method == "fractions":
            # Official conservative weights applied to one-hot category indicators.
            fractions = []
            categories = np.unique(field.values[np.isfinite(field.values)])
            for category in categories:
                indicator = xr.where(field.notnull(), xr.where(field == category, 1., 0.), np.nan)
                result = indicator.regrid.conservative(target, latitude_coord="lat" if axes[0] == "lat" else None,
                                                       skipna=True, nan_threshold=0.)
                if hasattr(result.data, "todense"):
                    result.data = result.data.todense()
                fractions.append(result)
            composition = xr.concat(fractions, dim=xr.IndexVariable("category", categories))
            output = xr.Dataset({context.variable: composition})
            output[context.variable].attrs.update(units="1", data_kind="continuous", spatial_semantics="category_fraction")
            report.update(capability="fractional_categorical_aggregation", not_equal_to="most_common",
                          fraction_sum_max_error=float(abs(composition.sum("category") - 1).max()))
        elif self.method == "cubic":
            output = field.regrid.cubic(target).to_dataset(name=context.variable)
        else:
            output = field.regrid.stat(target, method=self.method, skipna=True).to_dataset(name=context.variable)
            if self.method in {"min","max","median"}:
                output[context.variable].attrs.update(spatial_semantics="cell_"+self.method,spatial_statistic=self.method)
        return finish(context, self.spec, output, report, started)


class XesmfPlugin:
    def __init__(self, method):
        from ecodata.engines.grid import GridEngine
        self.method = method
        available = GridEngine.availability()
        caps = ("continuous_spatial_resampling", "multi_grid_regridding")
        if method == "nearest":
            caps += ("categorical_scale_processing",)
        if method == "conservative":
            caps += ("spatial_conservation",)
        try:
            installed_version = version("xesmf")
        except PackageNotFoundError:
            installed_version = "not installed"
        self.spec = AlgorithmSpec(
            f"spatial.xesmf_{method}", f"xESMF {method}", installed_version, "single_source_scale",
            "available" if available["available"] else "dependency_missing", ("geographic_grid",),
            ("continuous", "categorical") if method == "nearest" else ("continuous",),
            license="MIT", backend="xESMF / ESMF", source_url="https://github.com/pangeo-data/xESMF",
            capabilities=caps, preconditions=("ESMPy", "geographic_crs", "explicit_bounds" if method == "conservative" else "grid"),
            verification_evidence=("tests/test_algorithm_library_expansion.py",))

    def compatibility(self, context):
        from ecodata.engines.grid import GridEngine
        reasons = []
        if not GridEngine.availability()["available"]:
            reasons.append("BLOCKED: ESMF/ESMPy unavailable")
        if context.semantic_type not in self.spec.semantic_types:
            reasons.append("Semantic type incompatible with xESMF method")
        if self.method == "conservative":
            target = context.parameters.get("target_grid")
            for ds in (context.sources[0].dataset if context.sources else None, target):
                if ds is None or not all(a + "_b" in ds or (a in ds and ds[a].attrs.get("bounds") in ds) for a in ("lat", "lon")):
                    reasons.append("INPUT_REQUIREMENT_MISSING: explicit source and target cell bounds")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        from ecodata.engines.grid import GridEngine
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        target = context.parameters.get("target_grid")
        if target is None:
            target = xr.Dataset(coords={"lat": list(context.target_lat), "lon": list(context.target_lon)})
        output, report = GridEngine().regrid(context.sources[0].dataset, target, self.method, backend="xesmf")
        return finish(context, self.spec, output, report, started)


def spatial_plugins():
    return tuple(XesmfPlugin(method) for method in ("bilinear", "nearest", "conservative")) + tuple(
        XarrayAggregationPlugin(method) for method in ("most_common", "fractions", "mean", "median", "min", "max", "cubic")) + (ResultValidationPlugin(),)


class ResultValidationPlugin:
    """Expose the existing result gate as an explicit terminal capability node."""
    spec = AlgorithmSpec(
        "validation.result", "Existing result quality gate", "1", "validation", "available",
        ("grid", "time_series"), ("continuous", "categorical", "accumulation", "integer_count"),
        capabilities=("nonnegative_validation", "physical_range_validation"),
        backend="EcoData existing ResultQualityGate", license="project",
        verification_evidence=("tests/test_two_stage_quality_control.py",))

    def compatibility(self, context):
        return Compatibility(bool(context.sources and context.variable in context.sources[0].dataset))

    def run(self, context):
        from ecodata.evaluation.quality_gates import ResultQualityGate
        ds = context.sources[0].dataset
        report = ResultQualityGate().validate(ds, ds, variable=context.variable,
            semantic_type=context.semantic_type, nonnegative=bool(context.model_constraints.get("nonnegative")))
        if report["status"] == "BLOCK":
            raise SchemaError("Result quality gate blocked graph execution")
        return AlgorithmResult(context.sources[0], self.spec.id, report)
