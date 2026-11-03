"""Official xsdba train/adjust API with explicit training/application separation."""
from time import perf_counter
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.algorithms.open_source_stations import installed
from ecodata.core.exceptions import SchemaError


METHODS = {"eqm": "EmpiricalQuantileMapping", "dqm": "DetrendedQuantileMapping", "qdm": "QuantileDeltaMapping", "extreme": "ExtremeValues", "scaling": "Scaling"}


class XsdbaPlugin:
    def __init__(self, method):
        self.method = method
        self.spec = AlgorithmSpec(
            f"bias.xsdba_{method}", METHODS[method], installed("xsdba") or "not installed", "bias_correction",
            "available" if installed("xsdba") else "dependency_missing", ("time_series", "time_grid"),
            ("continuous", "accumulation"), requires_training=True, license="Apache-2.0", backend="xsdba",
            source_url="https://github.com/Ouranosinc/xsdba", capabilities=("extreme_value_correction",) if method == "extreme" else ("bias_correction","mean_bias_correction") if method=='scaling' else ("bias_correction", "distribution_adjustment"),
            preconditions=("training_reference", "historical_simulation", "disjoint_application_times", "explicit_units"),
            input_contract={"source_arity":1,"semantic_types": ["continuous", "accumulation"], "requires_training_pair": True},
            output_contract={"kind": "bias_corrected_field", "conservation": False},
            verification_evidence=("tests/test_raster_complements.py",) if method=='scaling' else ("tests/test_algorithm_library_v2.py",) if method == "extreme" else ("tests/test_algorithm_library_expansion.py",),
            verification_status="VERIFIED" if method in {'extreme','scaling'} and installed("xsdba") == "0.5.0" else "UNVERIFIED",
            temporal_requirements=("disjoint training/application periods", "regular samples for ExtremeValues clusters"),
            runtime_requirements=("xsdba subprocess; no main-process Pint import",),
            official_entry_point="xsdba.adjustment." + METHODS[method] + ".train/adjust",
            notes="Prior matched-grid mean adjustment; requires explicit calibration_source_id. Two samples is an interface minimum, not evidence of calibration adequacy." if method=='scaling' else "ExtremeValues is explicitly experimental upstream; tail correction is not proof of real peak accuracy." if method == "extreme" else "")

    def compatibility(self, context):
        reasons = []
        if self.spec.status != "available" or context.semantic_type not in self.spec.semantic_types:
            reasons.append("xsdba unavailable or incompatible semantic type")
        ref = context.parameters.get("training_reference")
        hist = context.parameters.get("historical_simulation")
        if not isinstance(ref, xr.DataArray) or not isinstance(hist, xr.DataArray) or not context.sources:
            return Compatibility(False, tuple(reasons + ["INPUT_REQUIREMENT_MISSING: training_reference and historical_simulation"]))
        sim = context.sources[0].dataset[context.variable]
        supports={a.attrs.get('spatial_semantics') for a in (ref,hist,sim)}-{None,'UNKNOWN'}
        if len(supports)>1:
            reasons.append('Different point/cell supports require an explicit support-estimation adapter, not same-support distribution adjustment')
        if (hist.attrs.get('distribution_transform') or sim.attrs.get('distribution_transform')) and hist.attrs.get('distribution_transform')!=sim.attrs.get('distribution_transform'):
            reasons.append('Training/application preprocessing transforms differ; refit on the same transformed source')
        bound=context.parameters.get('calibration_source_id')
        if self.method=='scaling' and not bound:reasons.append('Explicit calibration_source_id required for prior mean adjustment')
        if bound is not None:
            source=context.sources[0].dataset
            if source.attrs.get('source_id')!=bound or source.attrs.get('calibration_signature') or source.attrs.get('bias_adjustment_applied') or source.attrs.get('fusion_method'):
                reasons.append('Training pair does not describe this source/distribution')
        for array in (ref, hist, sim):
            if "time" not in array.coords or not array.attrs.get("units"):
                reasons.append("INPUT_REQUIREMENT_MISSING: time and units on all three arrays")
            elif array.time.to_index().has_duplicates or not array.time.to_index().is_monotonic_increasing:
                reasons.append("Time coordinate must be unique and increasing")
        if ref.attrs.get("split") != "train" or hist.attrs.get("split") != "train":
            reasons.append("Reference and historical simulation must be explicitly marked split=train")
        if all("time" in a.coords for a in (ref, hist, sim)):
            if not ref.time.equals(hist.time):
                reasons.append("Reference and historical simulation training times differ")
            if len(np.intersect1d(ref.time.values, sim.time.values)):
                reasons.append("Training/application time overlap is forbidden")
            if ref.time.size < (2 if self.method=='scaling' else max(20, int(context.parameters.get("nquantiles", 20)) * 2)):
                reasons.append("Insufficient samples for requested quantiles")
        if self.method=='scaling' and all('time' in a.coords for a in (ref,hist,sim)):
            try:
                kind=context.parameters.get('adjustment_kind','+')
                if kind not in {'+','*'}:raise ValueError('Scaling kind must be additive or multiplicative')
                xr.align(ref,hist,join='exact')
                if tuple(d for d in hist.dims if d!='time')!=tuple(d for d in sim.dims if d!='time'):raise ValueError('Spatial dimensions differ')
                for axis in hist.dims:
                    if axis!='time' and not hist[axis].equals(sim[axis]):raise ValueError('Training/application spatial support differs')
                if ref.attrs.get('physical_variable')!=sim.attrs.get('physical_variable') or not sim.attrs.get('physical_variable'):
                    raise ValueError('Explicit matching physical variable required')
                if hist.attrs.get('physical_variable')!=sim.attrs.get('physical_variable'):raise ValueError('Historical physical variable differs')
                if ref.time.values.max()>=sim.time.values.min():raise ValueError('Prior training must end before application')
                support=sim.notnull().any('time');counts=(ref.notnull()&hist.notnull()).sum('time')
                if bool(((counts<2)&support).any()):raise ValueError('At least two paired training observations per application cell required')
                if kind=='*' and bool(((hist.mean('time')==0)&support).any()):raise ValueError('Multiplicative scaling has zero historical mean')
            except (ValueError,KeyError) as exc:reasons.append(str(exc))
        if len({a.attrs.get("units") for a in (ref, hist, sim)}) != 1:
            reasons.append("Units differ; standardize with UnitEngine before fitting")
        if self.method == "extreme" and all("time" in a.coords for a in (ref, hist, sim)):
            if not context.parameters.get("cluster_thresh"):
                reasons.append("INPUT_REQUIREMENT_MISSING: cluster_thresh with explicit units")
            q = context.parameters.get("q_thresh", 0.95)
            if not isinstance(q, (int, float)) or not 0 < q < 1:
                reasons.append("q_thresh must lie strictly between zero and one")
            if any(not np.isfinite(a.values).all() for a in (ref, hist, sim)):
                reasons.append("Extreme fitting requires finite samples; missing-data policy must be explicit")
            if any(a.dims != ("time",) for a in (ref, hist, sim)):
                reasons.append("This adapter verifies only one-dimensional time series")
            if any(a.time.size < 365 for a in (ref, hist, sim)):
                reasons.append("Extreme fitting requires at least 365 samples per period; tail adequacy still needs review")
            for array in (ref, hist, sim):
                if "time" in array.coords and array.time.size > 1:
                    gaps = np.diff(array.time.values)
                    if not np.all(gaps == gaps[0]):
                        reasons.append("Cluster extremes require regular sampling; do not bridge missing time gaps")
            if not 0 < context.parameters.get("frac", 0.25) <= 1 or context.parameters.get("power", 1) <= 0:
                reasons.append("Extreme transition requires 0 < frac <= 1 and power > 0")
        return Compatibility(not reasons, tuple(reasons), ("Distribution correction does not establish spatial accuracy or preserve all extremes",))

    def run(self, context):
        check = self.compatibility(context)
        if not check.eligible:
            raise SchemaError("; ".join(check.reasons))
        started = perf_counter()
        ref = context.parameters["training_reference"]
        hist = context.parameters["historical_simulation"]
        sim = context.sources[0].dataset[context.variable]
        # xsdba sets Pint's shared application registry force_ndarray_like=False
        # at import. Isolate that upstream process-wide setting from pint-xarray.
        with tempfile.TemporaryDirectory(prefix="ecodata_xsdba_") as directory:
            folder = Path(directory)
            for name, array in (("ref", ref), ("hist", hist), ("sim", sim)):
                array.to_netcdf(folder / (name + ".nc"), engine="h5netcdf")
            options = {"method": METHODS[self.method], "nquantiles": int(context.parameters.get("nquantiles", 20)),
                       "kind": context.parameters.get("adjustment_kind", "+")}
            if self.method == "extreme":
                options.update(cluster_thresh=context.parameters["cluster_thresh"],
                               q_thresh=context.parameters.get("q_thresh", 0.95),
                               frac=context.parameters.get("frac", 0.25), power=context.parameters.get("power", 1))
            (folder / "options.json").write_text(json.dumps(options), encoding="utf-8")
            runner = Path(__file__).resolve().parents[3] / "tools/run_xsdba_adjustment.py"
            process = subprocess.run([sys.executable, str(runner), str(folder)], capture_output=True, text=True, timeout=180)
            if process.returncode:
                raise SchemaError("Official xsdba adjustment failed: " + process.stderr[-2500:])
            with xr.open_dataarray(folder / "corrected.nc", engine="h5netcdf") as opened:
                corrected = opened.load()
            xr.align(sim,corrected,join='exact')
            corrected=corrected.transpose(*sim.dims)
            fitted_variables = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
        if self.method == "extreme" and not np.isfinite(corrected.values).all():
            raise SchemaError("ExtremeValues produced nonfinite output: insufficient tail support or degenerate fit")
        if self.method == "extreme" and "nonnegative" in context.model_constraints and context.model_constraints["nonnegative"] and (corrected < 0).any():
            raise SchemaError("ExtremeValues violated the requested nonnegative constraint")
        output = context.sources[0].dataset.copy()
        output[context.variable] = corrected
        if context.parameters.get('calibration_source_id') is not None:output.attrs['bias_adjustment_applied']=self.spec.id
        report = {"backend": "xsdba", "version": self.spec.version, "method": METHODS[self.method],
                  "training_start": str(ref.time.values[0]), "training_end": str(ref.time.values[-1]),
                  "application_start": str(sim.time.values[0]), "application_end": str(sim.time.values[-1]),
                  "training_samples": ref.time.size, "application_samples": sim.time.size,
                  "training_application_overlap": 0, "test_used_for_fitting": False,
                  "adjustment_kind": context.parameters.get("adjustment_kind", "+"),
                  "fitted_parameter_variables": fitted_variables,
                  "profile_scope": "Prior per-cell mean adjustment; not quantile mapping or guaranteed accuracy gain" if self.method=='scaling' else 'distribution adjustment',
                  "execution_isolation": "subprocess; preserves main Pint application registry"}
        return finish(context, self.spec, output, report, started)


def bias_plugins():
    return tuple(XsdbaPlugin(method) for method in METHODS)
