"""EPA task ports; reuse Registry kernels and gates, never GEOS RH35 semantics."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import Delaunay

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.dag import Artifact, DAGError, Node, NodeAdapter, Port, NumericalDAGExecutor
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate

UNIT = "ug m-3"


@dataclass
class EPATask:
    registry: object
    template: xr.Dataset
    features: tuple = ()
    train_end: str | None = None
    sealed_test_stations: tuple = ()
    nonnegative_projection: bool = False
    counters: dict = field(default_factory=lambda: {"objective_calls": 0, "executions": 0, "failures": 0})

    def check(self, table, *, history=False):
        columns = {"station_id", "time", "longitude", "latitude", "value", "unit", "split"}
        if not columns.issubset(table):
            raise DAGError(f"EPA table missing {sorted(columns-set(table))}")
        if not table.split.eq("train").all():
            raise DAGError("Only training-station target values may enter fitting ports")
        if set(table.station_id) & set(self.sealed_test_stations):
            raise DAGError("Sealed test station appears in fitting data")
        if table.unit.nunique() != 1 or table.unit.iloc[0] != UNIT:
            raise DAGError("EPA PM2.5 requires explicitly standardized ug m-3")
        if not np.isfinite(table[["longitude", "latitude", "value"]].to_numpy(float)).all() or (table.value < 0).any():
            raise DAGError("Invalid EPA training observation")
        if table.duplicated(["station_id", "time"]).any():
            raise DAGError("Duplicate station/date or unselected POC")
        if history and (not self.train_end or pd.to_datetime(table.time).max() > pd.Timestamp(self.train_end)):
            raise DAGError("RF history crosses declared training cutoff")
        if not history and table.time.nunique() != 1:
            raise DAGError("Spatial estimation requires one actual daily window")
        ds = xr.Dataset({"pm25": ("sample", table.value.to_numpy(float))},
            coords={"longitude": ("sample", table.longitude.to_numpy(float)),
                    "latitude": ("sample", table.latitude.to_numpy(float))}, attrs={"crs": "EPSG:4326"})
        ds.pm25.attrs.update(units=UNIT, data_kind="continuous")
        quality = RawInputQualityGate().validate(ds, variable="pm25", intended_operation="spatial", expected_unit=UNIT)
        if quality["status"] == "BLOCK":
            raise DAGError(str(quality))
        return quality

    def estimate(self, inputs, params, external):
        table = inputs["observations"].value
        raw = self.check(table)
        auxiliaries = ()
        if "covariates" in inputs:
            covariates = inputs["covariates"].value
            if not self.features or not set(self.features).issubset(covariates):
                raise DAGError("Named MAIAC/ERA5 features not present")
            if covariates.attrs.get("provenance_verified") != "true":
                raise DAGError("Covariates lack verified actual-file provenance")
            if covariates.attrs.get("epa_local_standard_date") != str(table.time.iloc[0])[:10]:
                raise DAGError("Covariate/EPA daily-window mismatch")
            auxiliaries = (covariates[list(self.features)],)
        context = AlgorithmContext((EcoDataBundle(self.template),), "pm25", "continuous",
            target_lat=tuple(self.template.lat.values), target_lon=tuple(self.template.lon.values),
            observations=table, auxiliaries=auxiliaries,
            parameters={"observation_crs": "EPSG:4326", "unit": UNIT, "variogram_model": "linear",
                        "mask_missing_query_covariates": bool(auxiliaries)})
        result = self.registry.get(params["algorithm"]).run(context)
        return {"data": Artifact("epa_field", result.data.dataset, {"raw_quality": raw, "algorithm": result.algorithm_id})}

    def fit_rf(self, inputs, params, external):
        from ecodata.algorithms.feature_forest import fit_station_forest
        table = inputs["history"].value
        self.check(table, history=True)
        if not self.features or not set(self.features).issubset(table):
            raise DAGError("RF requires matched actual AOD/ERA5 training features")
        model, oof, folds = fit_station_forest(table, self.features)
        y = table.value.to_numpy(float)
        residuals = table[["station_id", "longitude", "latitude"]].copy()
        residuals["value"] = y-oof
        residuals = residuals.groupby("station_id", as_index=False).agg(longitude=("longitude", "first"),
            latitude=("latitude", "first"), value=("value", "mean"))
        residuals = residuals.assign(time=str(self.train_end), split="train", unit=UNIT)
        return {"state": Artifact("epa_rf_state", {"model": model, "residuals": residuals, "folds": folds,
            "features": self.features, "train_end": self.train_end, "residual_definition": "mean training-period station-group OOF error"})}

    def apply_rf(self, inputs, params, external):
        state, covariates = inputs["state"].value, inputs["covariates"].value
        if covariates.attrs.get("provenance_verified") != "true":
            raise DAGError("Covariates lack verified actual-file provenance")
        date = covariates.attrs.get("epa_local_standard_date")
        if not date or pd.Timestamp(date) <= pd.Timestamp(state["train_end"]):
            raise DAGError("RF application must use a declared post-training daily window")
        same_grid = all(np.array_equal(covariates[axis].values, self.template[axis].values)
                        for axis in ("lat", "lon"))
        arrays = [(covariates[name].transpose("lat", "lon").values if same_grid else
                   covariates[name].interp(lat=self.template.lat, lon=self.template.lon).values).ravel()
                  for name in state["features"]]
        x = np.column_stack(arrays)
        valid = np.isfinite(x).all(axis=1)
        if not valid.any():
            raise DAGError("No real common AOD/ERA5 coverage")
        values = np.full(len(x), np.nan)
        values[valid] = state["model"].predict(x[valid])
        ds = self.template.copy(deep=True)
        ds["pm25"] = (("lat", "lon"), values.reshape(ds.pm25.shape))
        ds.pm25.attrs.update(units=UNIT, data_kind="continuous")
        return {"data": Artifact("epa_background", ds)}

    def residual(self, inputs, params, external):
        state, covariates = inputs["state"].value, inputs["covariates"].value
        table = inputs["observations"].value.copy()
        self.check(table)
        if pd.Timestamp(table.time.iloc[0]) <= pd.Timestamp(state["train_end"]):
            raise DAGError("Contemporaneous residuals require a post-training date; otherwise use OOF predictions")
        if covariates.attrs.get("epa_local_standard_date") != str(table.time.iloc[0])[:10]:
            raise DAGError("Residual covariate/EPA daily-window mismatch")
        features = np.column_stack([covariates[name].interp(
            lon=xr.DataArray(table.longitude.to_numpy(), dims="station"),
            lat=xr.DataArray(table.latitude.to_numpy(), dims="station")).values for name in state["features"]])
        if not np.isfinite(features).all():
            raise DAGError("Residual training stations lack actual common covariate coverage")
        # Same contemporaneous input stations as route A. The historical RF has
        # never fitted this date, so these are out-of-time, not in-sample errors.
        table["value"] = table.value.to_numpy()-state["model"].predict(features)
        context = AlgorithmContext((EcoDataBundle(self.template),), "pm25", "continuous",
            target_lat=tuple(self.template.lat.values), target_lon=tuple(self.template.lon.values), observations=table,
            parameters={"observation_crs": "EPSG:4326", "unit": UNIT})
        correction = self.registry.get("fusion.idw").run(context).data.dataset.pm25
        ds = inputs["background"].value.copy(deep=True)
        ds["background_pm25"] = ds.pm25.copy()
        ds["out_of_time_residual_correction"] = correction
        ds["out_of_time_residual_correction"].attrs.update(units=UNIT, physical_variable="signed_pm25_correction")
        ds["pm25"] = ds.pm25+correction
        ds.pm25.attrs.update(units=UNIT, data_kind="continuous")
        return {"data": Artifact("epa_field", ds)}

    def quality(self, inputs, params, external):
        ds = inputs["field"].value.copy(deep=True)
        table = inputs["observations"].value
        xy = table[["longitude", "latitude"]].to_numpy(float)
        lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
        sx, sy = transformer.transform(xy[:, 0], xy[:, 1])
        qx, qy = transformer.transform(lon.ravel(), lat.ravel())
        mask = (Delaunay(np.column_stack([sx, sy])).find_simplex(np.column_stack([qx, qy])) >= 0).reshape(lon.shape)
        for name in ds.data_vars:
            ds[name] = ds[name].where(xr.DataArray(mask, dims=("lat", "lon")))
        if np.isinf(ds.pm25.values).any():
            raise DAGError("Infinite PM2.5 prediction")
        # Satellite missingness remains masked, never filled merely for evaluation.
        ds["valid_domain"] = (("lat", "lon"), (mask & np.isfinite(ds.pm25.values)).astype("int8"))
        projection = None
        if self.nonnegative_projection:
            before = ds.pm25.values
            negative = np.isfinite(before) & (before < 0)
            projection = {"count": int(negative.sum()), "before_min": float(np.nanmin(before)),
                          "max_correction": float(-np.min(before[negative])) if negative.any() else 0.,
                          "method": "projection onto nonnegative concentration cone; no mass-conservation claim"}
            ds["pm25"] = ds.pm25.clip(min=0)
        report = ResultQualityGate().validate(self.template, ds, variable="pm25", nonnegative=True, expected_output_unit=UNIT)
        if projection:
            report["nonnegative_projection"] = projection
            ds.attrs["nonnegative_projected_pixels"] = projection["count"]
            if projection["count"]:
                report["checks"].append({"id": "output.nonnegative_projection", "status": "WARN",
                    "value": projection, "message": "Negative estimates projected to zero; inspect model validity, not an accuracy proof"})
                report["summary"]["WARN"] += 1
                if report["status"] == "PASS":
                    report.update(status="WARN", decision="confirm_required")
        if report["status"] == "BLOCK":
            raise DAGError(str(report))
        ds.attrs.update(crs="EPSG:4326", time_semantics="EPA local-standard daily mean",
                        information_scale="analysis grid, not verified high-resolution truth")
        return {"data": Artifact("epa_validated", ds), "report": Artifact("epa_quality", report)}

    def adapters(self):
        def source(kind):
            def run(inputs, params, external):
                value = external[params["source"]]
                Port(kind).validate(value)
                return {"data": value}
            return NodeAdapter({}, {"data": Port(kind)}, run)
        adapters = {(f"epa.source.{name}", "run"): source(kind) for name, kind in
                    (("stations", "epa_train"), ("history", "epa_history"), ("covariates", "epa_covariates"))}
        definitions = [
            ("epa.spatial", {"observations": "epa_train"}, {"data": "epa_field"}, self.estimate),
            ("epa.external_drift", {"observations": "epa_train", "covariates": "epa_covariates"}, {"data": "epa_field"}, self.estimate),
            ("epa.rf_fit", {"history": "epa_history"}, {"state": "epa_rf_state"}, self.fit_rf),
            ("epa.rf_apply", {"state": "epa_rf_state", "covariates": "epa_covariates"}, {"data": "epa_background"}, self.apply_rf),
            ("epa.residual", {"background": "epa_background", "state": "epa_rf_state", "observations": "epa_train", "covariates": "epa_covariates"}, {"data": "epa_field"}, self.residual),
            ("epa.quality", {"field": "epa_field", "observations": "epa_train"}, {"data": "epa_validated", "report": "epa_quality"}, self.quality)]
        for name, inputs, outputs, execute in definitions:
            adapters[(name, "run")] = NodeAdapter({k: Port(v) for k, v in inputs.items()}, {k: Port(v) for k, v in outputs.items()}, execute)
        return adapters

    def graph(self, method):
        nodes = [Node("stations", "epa.source.stations", parameters={"source": "stations"})]
        if method in ("fusion.idw", "station.pykrige_ordinary"):
            nodes += [Node("field", "epa.spatial", {"observations": ("stations", "data")}, {"algorithm": method})]
        elif method == "route_a":
            nodes += [Node("covariates", "epa.source.covariates", parameters={"source": "covariates"}),
                Node("field", "epa.external_drift", {"observations": ("stations", "data"), "covariates": ("covariates", "data")},
                     {"algorithm": "station.pykrige_universal"})]
        elif method == "route_b":
            nodes += [Node("history", "epa.source.history", parameters={"source": "history"}),
                Node("covariates", "epa.source.covariates", parameters={"source": "covariates"}),
                Node("fit", "epa.rf_fit", {"history": ("history", "data")}),
                Node("background", "epa.rf_apply", {"state": ("fit", "state"), "covariates": ("covariates", "data")}),
                Node("field", "epa.residual", {"state": ("fit", "state"), "background": ("background", "data"),
                     "observations": ("stations", "data"), "covariates": ("covariates", "data")})]
        else:
            raise DAGError("Unsupported EPA task route")
        nodes += [Node("quality", "epa.quality", {"field": ("field", "data"), "observations": ("stations", "data")})]
        return nodes

    def objective(self, method, external, validation, folder, *, golem_graph=None, executor=None):
        """No test argument; validation labels are consumed after graph execution."""
        started = perf_counter()
        self.counters["objective_calls"] += 1
        try:
            if not validation.split.eq("validation").all():
                raise DAGError("Objective accepts validation only, never test targets")
            if (not len(validation) or validation.unit.nunique() != 1 or validation.unit.iloc[0] != UNIT
                    or not np.isfinite(validation[["longitude", "latitude", "value"]].to_numpy(float)).all()):
                raise DAGError("Invalid validation values, coordinates or unit")
            if set(validation.station_id) & set(self.sealed_test_stations):
                raise DAGError("Sealed test station cannot be relabelled validation")
            observed_dates = set(pd.to_datetime(external["stations"].value.time).dt.normalize())
            if set(pd.to_datetime(validation.time).dt.normalize()) != observed_dates:
                raise DAGError("Validation/EPA input daily-window mismatch")
            forbidden = set(validation.station_id)
            for name in ("stations", "history"):
                if name in external and forbidden.intersection(external[name].value.station_id):
                    raise DAGError("Station leakage between fit and validation")
            executor = executor or NumericalDAGExecutor(self.adapters(), version="epa-task-1")
            self.counters["executions"] += 1
            result = (executor.run_golem(golem_graph, external=external) if golem_graph is not None
                      else executor.run(self.graph(method), "quality", external=external))
            ds = result["outputs"]["data"].value
            folder.mkdir(parents=True, exist_ok=False)
            # Persist real fit isolation, not only a synthetic unit-test proof.
            fit_outputs = [outputs for outputs in result["intermediates"].values()
                           if "state" in outputs and outputs["state"].kind == "epa_rf_state"]
            if fit_outputs:
                import json
                state = fit_outputs[0]["state"].value
                evidence = {"folds": state["folds"], "train_end": state["train_end"],
                            "features": state["features"], "residual_definition": state["residual_definition"],
                            "fit_station_ids": sorted(external["history"].value.station_id.unique()),
                            "fit_date_min": str(external["history"].value.time.min()),
                            "fit_date_max": str(external["history"].value.time.max()),
                            "fit_sample_count": len(external["history"].value)}
                (folder / "fit_isolation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            ds.to_netcdf(folder / "result.nc", engine="h5netcdf")
            with xr.open_dataset(folder / "result.nc", engine="h5netcdf") as reopened:
                reopened.load()
                if not np.array_equal(np.isfinite(reopened.pm25), np.isfinite(ds.pm25)):
                    raise DAGError("NetCDF finite-mask roundtrip changed")
                np.testing.assert_allclose(reopened.pm25, ds.pm25, equal_nan=True)
            predictions = ds.pm25.interp(lon=xr.DataArray(validation.longitude.to_numpy(), dims="station"),
                                        lat=xr.DataArray(validation.latitude.to_numpy(), dims="station")).values
            finite = np.isfinite(predictions) & np.isfinite(validation.value)
            if not finite.all() or not len(predictions):
                raise DAGError("Every fixed validation sample must lie inside common supported domain")
            errors = predictions-validation.value.to_numpy(float)
            validation.assign(prediction=predictions).to_csv(folder / "validation_predictions.csv", index=False)
            return {"status": "PASS", "RMSE": float(np.sqrt(np.mean(errors**2))), "MAE": float(np.abs(errors).mean()),
                "Bias": float(errors.mean()), "count": len(errors), "quality": result["outputs"]["report"].value,
                "seconds": perf_counter()-started, "trace": result["trace"], "counters": dict(self.counters)}
        except Exception as exc:
            self.counters["failures"] += 1
            exc.execution_seconds = perf_counter()-started
            exc.objective_counters = dict(self.counters)
            raise
