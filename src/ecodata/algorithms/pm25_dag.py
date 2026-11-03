"""Exploratory PM2.5 contracts for existing frozen algorithms, not new plugins."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import Delaunay

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.dag import Artifact, DAGError, NodeAdapter, Port
from ecodata.algorithms.quality_aware import fit_affine_source
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate
from ecodata.io.readers import _frame_to_dataset


TARGET = "pm25_observation"
UNIT = "ug m-3"
SEMANTIC = "PM25_station_definition"


class PM25DAGAdapters:
    def __init__(self, registry, center, lat, lon, *, exploratory=False):
        if not exploratory:
            raise DAGError("Station datum unconfirmed; explicit exploratory coordinate assumption required")
        self.registry, self.center = registry, pd.Timestamp(center)
        self.lat, self.lon = tuple(lat), tuple(lon)
        self.meta = {"center_utc": self.center.isoformat(),
            "window_start_utc": (self.center-pd.Timedelta(minutes=30)).isoformat(),
            "window_end_utc": (self.center+pd.Timedelta(minutes=30)).isoformat(),
            "crs": "EPSG:4326", "crs_evidence": "ASSUMED_NOT_GEODETICALLY_VERIFIED",
            "experiment_status": "EXPLORATORY"}

    def _artifact(self, kind, value, **metadata):
        return Artifact(kind, value, {**self.meta, **metadata})

    def _raw(self, ds, variable):
        report = RawInputQualityGate().validate(ds, variable=variable, requires_spatial=True)
        if report["status"] == "BLOCK":
            raise DAGError("Raw quality BLOCK: " + str(report["checks"]))
        return report

    def _same_window(self, inputs):
        for value in inputs.values():
            for key in ("center_utc", "window_start_utc", "window_end_utc", "crs", "crs_evidence"):
                if value.metadata.get(key) != self.meta[key]:
                    raise DAGError(f"Edge {key} differs from execution contract")

    def select_geos(self, inputs, params, external):
        raw = inputs["series"].value
        field = raw.pm25_rh35_gcc.sel(time=self.center).squeeze(drop=True).transpose("lat", "lon")
        if field.attrs.get("humidity_basis") != "RH35" or field.attrs.get("units") != UNIT:
            raise DAGError("GEOS RH35/units contract missing; no semantic relabelling")
        ds = field.to_dataset().sortby("lat").sortby("lon")
        ds.attrs.update(self.meta)
        ds.attrs["crs"] = self.meta["crs"]
        ds.pm25_rh35_gcc.attrs["data_kind"] = "continuous"
        q = self._raw(ds, "pm25_rh35_gcc")
        return {"data": self._artifact("geos_covariate", ds, unit=UNIT, semantic="PM25_RH35_model", raw_quality=q)}

    def select_meteo(self, inputs, params, external):
        raw = inputs["series"].value
        before, after = self.center-pd.Timedelta(minutes=30), self.center+pd.Timedelta(minutes=30)
        if before not in raw.valid_time.to_index() or after not in raw.valid_time.to_index():
            raise DAGError("ERA5 has no real bracketing instantaneous endpoints")
        ds = raw[["t2m"]].interp(valid_time=self.center).rename({"latitude": "lat", "longitude": "lon"})
        ds = ds.drop_vars("valid_time").sortby("lat").sortby("lon")
        if ds.t2m.attrs.get("units") != "K":
            raise DAGError("ERA5 temperature unit must be K")
        ds.t2m.attrs.update(data_kind="continuous", temporal_semantics="interpolated_instantaneous_covariate_NOT_hourly_mean")
        ds.attrs.update(self.meta)
        q = self._raw(ds, "t2m")
        return {"data": self._artifact("meteo_covariate", ds, unit="K", semantic="air_temperature", raw_quality=q)}

    def select_stations(self, inputs, params, external):
        frame = inputs["series"].value
        if not frame.split.eq("train").all():
            raise DAGError("Only input training stations may enter numerical estimation")
        selected = frame.loc[frame.time.eq(self.center)].copy()
        if selected.station_id.duplicated().any():
            raise DAGError("Duplicate input station/time")
        rows = selected.copy()
        rows = rows[np.isfinite(rows.value)]
        if len(rows) < 4 or rows.station_id.duplicated().any() or (rows.value < 0).any():
            raise DAGError("Insufficient/duplicate/negative observations")
        if rows.unit.nunique() != 1 or rows.unit.iloc[0] != UNIT:
            raise DAGError("Observation unit contract missing")
        ds = xr.Dataset.from_dataframe(selected.set_index("station_id"))
        ds.attrs.update(self.meta)
        ds.value.attrs["units"] = UNIT
        q = self._raw(ds, "value")
        return {"data": self._artifact("observations", rows, unit=UNIT, semantic=SEMANTIC,
            instrument_qa="UNKNOWN; numerical checks only, not AQI grade", raw_quality=q,
            input_rows=len(selected), dropped_missing=len(selected)-len(rows))}

    def estimate(self, inputs, params, external):
        self._same_window(inputs)
        if inputs["geos"].metadata["semantic"] != "PM25_RH35_model" or inputs["meteo"].metadata["semantic"] != "air_temperature":
            raise DAGError("Distinct covariate meanings must be retained")
        # This is a target schema only, never a fabricated concentration field.
        carrier = xr.Dataset({TARGET: (("lat", "lon"), np.full((len(self.lat), len(self.lon)), np.nan))},
            coords={"lat": list(self.lat), "lon": list(self.lon)}, attrs={**self.meta, "role": "target_schema_only"})
        carrier[TARGET].attrs.update(units=UNIT, physical_variable=SEMANTIC)
        context = AlgorithmContext((EcoDataBundle(carrier),), TARGET, "continuous", self.lat, self.lon,
            observations=inputs["observations"].value,
            auxiliaries=(inputs["geos"].value, inputs["meteo"].value),
            parameters={"observation_crs": self.meta["crs"], "variogram_model": params.get("variogram_model", "linear")})
        plugin = self.registry.get("station.pykrige_universal")
        check = plugin.compatibility(context)
        if not check.eligible:
            raise DAGError(str(check.reasons))
        result = plugin.run(context)
        result.data.dataset.attrs["role"] = "estimated_field"
        return {"data": self._artifact("analysis_field", result.data.dataset, unit=UNIT, semantic=SEMANTIC,
                algorithm_report=result.report)}

    def fit_calibration(self, inputs, params, external):
        history = inputs["observations"].value
        if not history.split.eq("train").all() or history.time.max() >= self.center:
            raise DAGError("Calibration must use past training-station observations only")
        if history.unit.nunique() != 1 or history.unit.iloc[0] != UNIT:
            raise DAGError("Calibration observation units differ")
        field = inputs["series"].value.pm25_rh35_gcc.squeeze("lev", drop=True)
        if field.attrs.get("humidity_basis") != "RH35" or field.attrs.get("units") != UNIT:
            raise DAGError("Calibration input definition must remain explicit RH35")
        point = lambda x: xr.DataArray(x, dims="pair")
        prediction = field.interp(time=point(history.time.to_numpy()), lat=point(history.latitude.to_numpy()),
                                  lon=point(history.longitude.to_numpy())).values
        reference = history.value.to_numpy(float)
        valid = np.isfinite(prediction) & np.isfinite(reference)
        if valid.sum() < 3 or np.std(prediction[valid]) == 0:
            raise DAGError("Calibration pairs insufficient or rank deficient")
        slope, intercept = fit_affine_source(prediction, reference)
        state = {"slope": slope, "intercept": intercept, "fit_count": int(valid.sum()),
            "fit_start": history.time.min().isoformat(), "fit_end": history.time.max().isoformat(),
            "fit_stations": sorted(history.station_id.unique().tolist()), "fit_split": "train",
            "input_variable": "pm25_rh35_gcc", "input_semantic": "PM25_RH35_model", "input_unit": UNIT,
            "output_semantic": SEMANTIC, "output_unit": UNIT, "crs_evidence": self.meta["crs_evidence"],
            "method": "existing_quality_aware_affine_component_NOT_physical_humidity_conversion",
            "source_fingerprints": {k: a.fingerprint() for k, a in inputs.items()},
            "validation_or_test_used": False}
        return {"state": Artifact("calibration_state", state, {"unit": UNIT, "fit_split": "train"})}

    def apply_calibration(self, inputs, params, external):
        self._same_window({"geos": inputs["geos"]})
        state, geos = inputs["state"].value, inputs["geos"].value
        if (state["input_semantic"] != inputs["geos"].metadata.get("semantic") or state["input_unit"] != UNIT
                or state["fit_split"] != "train" or state["validation_or_test_used"]
                or pd.Timestamp(state["fit_end"]) >= self.center
                or state["crs_evidence"] != self.meta["crs_evidence"]
                or state["output_semantic"] != SEMANTIC or state["output_unit"] != UNIT
                or geos.pm25_rh35_gcc.attrs.get("units") != UNIT):
            raise DAGError("Incompatible/leaking calibration state")
        field = state["slope"]*geos.pm25_rh35_gcc + state["intercept"]
        ds = field.rename(TARGET).to_dataset()
        ds[TARGET].attrs = {"units": UNIT, "physical_variable": SEMANTIC, "data_kind": "continuous",
            "calibration": "empirical_observation_equivalence_NOT_physical_RH_conversion"}
        ds.attrs.update(self.meta)
        ds.attrs["calibration_state"] = json.dumps(state, ensure_ascii=False)
        return {"data": self._artifact("calibrated_background", ds, unit=UNIT, semantic=SEMANTIC, state=state)}

    def residual(self, inputs, params, external):
        self._same_window(inputs)
        observations = inputs["observations"].value.copy()
        observations["variable"], observations["qc_flag"] = SEMANTIC, "VALID"
        # VALID denotes finite/nonnegative/in-domain schema checks, not instrument QA.
        bg = inputs["background"].value
        context = AlgorithmContext((EcoDataBundle(bg), EcoDataBundle(_frame_to_dataset(observations))), TARGET, "continuous",
            observations=observations, parameters={"interpolation": params.get("interpolation", "linear"), "outside_hull": "mask"})
        plugin = self.registry.get("fusion.residual")
        check = plugin.compatibility(context)
        if not check.eligible:
            raise DAGError(str(check.reasons))
        result = plugin.run(context)
        return {"data": self._artifact("analysis_field", result.data.dataset, unit=UNIT, semantic=SEMANTIC,
            algorithm_report=result.report, outside_hull_policy="mask")}

    def regrid(self, inputs, params, external):
        self._same_window(inputs)
        context = AlgorithmContext((EcoDataBundle(inputs["background"].value),), TARGET, "continuous", self.lat, self.lon)
        result = self.registry.get("spatial.bilinear").run(context)
        return {"data": self._artifact("calibrated_analysis_grid", result.data.dataset, unit=UNIT,
            semantic=SEMANTIC, algorithm_report=result.report)}

    def domain(self, observations):
        rows = observations
        lon, lat = np.meshgrid(self.lon, self.lat)
        source = rows[["longitude", "latitude"]].to_numpy(float)
        query = np.column_stack([lon.ravel(), lat.ravel()])
        tx = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat.mean()} +lon_0={lon.mean()} +datum=WGS84 +units=m", always_xy=True)
        sx, sy = tx.transform(source[:, 0], source[:, 1])
        qx, qy = tx.transform(query[:, 0], query[:, 1])
        metric = Delaunay(np.column_stack([sx, sy])).find_simplex(np.column_stack([qx, qy])) >= 0
        geographic = Delaunay(source).find_simplex(query) >= 0
        return (metric & geographic).reshape(lon.shape)

    def quality(self, inputs, params, external):
        self._same_window(inputs)
        ds = inputs["field"].value.copy(deep=True)
        mask = self.domain(inputs["observations"].value)
        values = ds[TARGET].values
        if not mask.any() or not np.isfinite(values[mask]).all():
            raise DAGError("No supported domain or nonfinite prediction INSIDE supported domain")
        if ds[TARGET].attrs.get("physical_variable") != SEMANTIC:
            raise DAGError("Output does not represent the station-definition target")
        for name in list(ds.data_vars):
            if set(ds[name].dims) == {"lat", "lon"}:
                ds[name] = ds[name].where(xr.DataArray(mask, dims=("lat", "lon")))
        ds["valid_domain"] = (("lat", "lon"), mask.astype(np.int8))
        ds = ds.assign_coords(time=np.datetime64(self.center))
        ds.attrs.update(self.meta)
        q = ResultQualityGate().validate(ds, ds, variable=TARGET, nonnegative=True, expected_output_unit=UNIT)
        if q["status"] == "BLOCK":
            raise DAGError("Result quality BLOCK: " + str(q["checks"]))
        q.update(exploratory=True, geodetic_accuracy_verified=False, valid_cells=int(mask.sum()),
            outside_domain_masked=int((~mask).sum()), information_scale_claim="analysis grid only; no verified high-resolution truth")
        return {"data": self._artifact("validated_field", ds, unit=UNIT, semantic=SEMANTIC),
                "report": self._artifact("quality_report", q)}

    def adapters(self):
        result = {}
        for key, kind in (("geos", "geos_series"), ("era5", "meteo_series"),
                          ("stations", "station_series"), ("history", "history_observations")):
            def source(inputs, params, external, kind=kind):
                artifact = external[params["source"]]
                if artifact.kind != kind:
                    raise DAGError("Source kind mismatch")
                return {"data": artifact}
            result[("source."+key, "run")] = NodeAdapter({}, {"data": Port(kind)}, source)
        definitions = [
            ("prepare.geos", "run", {"series": "geos_series"}, {"data": "geos_covariate"}, self.select_geos),
            ("prepare.era5", "run", {"series": "meteo_series"}, {"data": "meteo_covariate"}, self.select_meteo),
            ("prepare.stations", "run", {"series": "station_series"}, {"data": "observations"}, self.select_stations),
            ("station.pykrige_universal", "run", {"geos": "geos_covariate", "meteo": "meteo_covariate", "observations": "observations"}, {"data": "analysis_field"}, self.estimate),
            ("fusion.quality_aware", "fit", {"series": "geos_series", "observations": "history_observations"}, {"state": "calibration_state"}, self.fit_calibration),
            ("fusion.quality_aware", "apply", {"geos": "geos_covariate", "state": "calibration_state"}, {"data": "calibrated_background"}, self.apply_calibration),
            ("spatial.bilinear", "run", {"background": "calibrated_background"}, {"data": "calibrated_analysis_grid"}, self.regrid),
            ("fusion.residual", "run", {"background": "calibrated_analysis_grid", "observations": "observations"}, {"data": "analysis_field"}, self.residual),
            ("quality.result", "run", {"field": "analysis_field", "observations": "observations"}, {"data": "validated_field", "report": "quality_report"}, self.quality),
        ]
        for operation, mode, inputs, outputs, function in definitions:
            def contract(kind):
                required = ()
                if kind in {"geos_covariate", "meteo_covariate", "observations", "analysis_field", "calibrated_background",
                            "calibrated_analysis_grid", "validated_field"}:
                    required = ("crs", "crs_evidence", "center_utc", "unit", "semantic")
                elif kind == "calibration_state":
                    required = ("unit", "fit_split")
                return Port(kind, required)
            result[(operation, mode)] = NodeAdapter({k: contract(v) for k, v in inputs.items()},
                {k: contract(v) for k, v in outputs.items()}, function, mode=mode)
        return result
