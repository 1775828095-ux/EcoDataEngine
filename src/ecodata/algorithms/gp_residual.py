"""Background plus regularised GP residuals; no automatic accuracy claims."""
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.algorithms.open_source_stations import station_inputs
from ecodata.core.exceptions import SchemaError


class GPResidualPlugin:
    spec = AlgorithmSpec(
        id="fusion.gp_residual", name="Background plus Gaussian-process residual correction",
        version=version("scikit-learn"), category="multi_source_fusion", status="available",
        input_kinds=("rectilinear_grid", "station_table"), semantic_types=("continuous",),
        requires_observations=True, compute="medium", license="BSD-3-Clause", backend="scikit-learn",
        source_url="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/_gpr.py",
        capabilities=("multi_source_fusion", "observation_constraint"),
        preconditions=("same_time", "same_physical_variable", "training_station_split", "explicit_gp_parameters"),
        input_contract={"requires_observations": True, "semantic_types": ["continuous"]},
        output_contract={"kind": "grid", "uncertainty": "conditional residual GP only; not total error"},
        verification_evidence=("tests/test_gp_residual.py",),
        notes="Fixed user/development parameters; zero-mean residual prior; no test-label fitting, no nonnegative or conservation guarantee.")

    def inputs(self, context):
        if len(context.sources) != 1 or context.auxiliaries or context.semantic_type != "continuous":
            raise SchemaError("GP residual requires one continuous background and observations; no undeclared auxiliaries")
        source, target, axes, train, points, query, outside, dropped = station_inputs(context, 3)
        if axes != ("lat", "lon") or not CRS.from_user_input(source.attrs["crs"]).equals(CRS.from_epsg(4326)):
            raise SchemaError("GP residual currently requires an EPSG:4326 lat/lon background")
        if source[context.variable].dims != ("lat", "lon") or not np.isfinite(source[context.variable]).all():
            raise SchemaError("A finite single 2D background is required")
        if dropped or not context.observations.split.eq("train").all():
            raise SchemaError("Only finite train observations may enter GP fitting; no cohort deletion")
        if "time" not in source.coords or source.time.size != 1 or "time" not in train or train.time.isna().any():
            raise SchemaError("Explicit single background/observation time required")
        stamp = pd.to_datetime(np.asarray(source.time).ravel()[0], utc=True)
        if not pd.to_datetime(train.time, utc=True).eq(stamp).all():
            raise SchemaError("Background and observation times differ")
        physical = source[context.variable].attrs.get("physical_variable")
        if not physical or "variable" not in train or not train.variable.eq(physical).all():
            raise SchemaError("Explicit matching physical_variable/observation variable required")
        params = context.parameters
        keys = ("gp_length_scale_m", "gp_signal_std", "gp_noise_std")
        if any(k not in params or isinstance(params[k], bool) or not np.isfinite(params[k]) or params[k] <= 0 for k in keys):
            raise SchemaError("Explicit positive gp_length_scale_m, gp_signal_std, gp_noise_std required")
        if len(train) > 1000:
            raise SchemaError("Dense GP limited to 1000 stations; no silent subsampling")
        tx = Transformer.from_crs(params["observation_crs"], "EPSG:4326", always_xy=True)
        lon, lat = tx.transform(train.longitude.to_numpy(float), train.latitude.to_numpy(float))
        sampled = source[context.variable].interp(lon=xr.DataArray(lon, dims="station"), lat=xr.DataArray(lat, dims="station"))
        background = source[context.variable].interp({a: target[a] for a in axes})
        if not np.isfinite(sampled).all() or not np.isfinite(background).all():
            raise SchemaError("Background must cover every station and output point; no edge extrapolation")
        return source, target, axes, train, points, query, sampled.values, background

    def compatibility(self, context):
        try:
            self.inputs(context)
        except (SchemaError, ValueError, KeyError, IndexError, TypeError) as exc:
            return Compatibility(False, (str(exc),))
        return Compatibility(True, warnings=("GP standard deviation is conditional on the fixed covariance, not calibrated total error",))

    def run(self, context):
        started = perf_counter()
        source, target, axes, train, points, query, sampled, background = self.inputs(context)
        p = context.parameters
        kernel = ConstantKernel(p["gp_signal_std"]**2, "fixed") * RBF(p["gp_length_scale_m"], "fixed")
        model = GaussianProcessRegressor(kernel=kernel, alpha=p["gp_noise_std"]**2,
                                         optimizer=None, normalize_y=False)
        model.fit(points, train.value.to_numpy(float) - sampled)
        correction, std = model.predict(query, return_std=True)
        output = target.copy().assign_coords(time=np.asarray(source.time).ravel()[0])
        output[context.variable] = background + correction.reshape(background.shape)
        output[context.variable].attrs = dict(source[context.variable].attrs)
        output["background_value"] = background
        output["residual_correction"] = (axes, correction.reshape(background.shape))
        output["residual_gp_std"] = (axes, std.reshape(background.shape))
        for name in ("residual_correction", "residual_gp_std"):
            output[name].attrs["units"] = source[context.variable].attrs["units"]
        output["residual_gp_std"].attrs["uncertainty_scope"] = "latent residual GP conditional on fixed parameters; excludes background and parameter uncertainty"
        report = dict(training_count=len(train), parameters={k: p[k] for k in ("gp_length_scale_m", "gp_signal_std", "gp_noise_std")},
                      optimizer=None, reference_used=False, accuracy_assessed=False,
                      stages=["sample_background", "observed_minus_background", "GP_fit_predict", "background_plus_residual"])
        return finish(context, self.spec, output, report, started)
