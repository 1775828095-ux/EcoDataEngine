"""Small official-backend executions, with synthetic/real evidence kept separate."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.schemas import EcoDataBundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion/v2_examples"


def save(context, algorithm, label, input_kind):
    started = perf_counter()
    plan = AlgorithmPlan(label, (algorithm,), 0., ("Explicit capability verification; no GOLEM search",))
    result = UnifiedAlgorithmPipeline(create_default_registry()).run(context, plan)
    path = OUT / (label + ".nc")
    result.data.dataset.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        a = ds[context.variable].load()
        assert np.isfinite(a).all()
        assert a.attrs.get("units")
        record = {"algorithm": algorithm, "input_evidence": input_kind, "path": str(path.relative_to(ROOT)),
                  "shape": dict(a.sizes), "coordinates": list(a.coords), "units": a.attrs["units"],
                  "nan_count": int(a.isnull().sum()), "min": float(a.min()), "max": float(a.max()),
                  "runtime_seconds": perf_counter() - started, "status": "PASS", "report": result.report,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "predictive_skill_verified": False}
    (OUT / (label + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    rng = np.random.default_rng(42)
    arrays = []
    for year, scale in ((1980, 1.), (1980, 1.4), (2000, 1.6)):
        v = rng.gamma(1.5, 4 * scale, 1825)
        v[::3] = 0
        arrays.append(xr.DataArray(v, dims="time", coords={"time": pd.date_range(str(year), periods=v.size)},
                                  attrs={"units": "mm/day", "split": "train" if year == 1980 else "application"}))
    context = AlgorithmContext((EcoDataBundle(arrays[2].to_dataset(name="rain")),), "rain", "continuous",
        parameters={"training_reference": arrays[0], "historical_simulation": arrays[1], "cluster_thresh": "1 mm/day",
                    "q_thresh": .8, "nquantiles": 20, "adjustment_kind": "*"}, model_constraints={"nonnegative": True})
    records.append(save(context, "bias.xsdba_extreme", "extreme_precipitation", "SYNTHETIC daily rainfall; not real climate validation"))
    records.append(save(context, "bias.xsdba_eqm", "ordinary_eqm_precipitation", "SYNTHETIC baseline; same training/application data"))
    with xr.open_dataset(OUT / "extreme_precipitation.nc", engine="h5netcdf") as extreme, xr.open_dataset(OUT / "ordinary_eqm_precipitation.nc", engine="h5netcdf") as ordinary:
        differences = np.abs(extreme.rain.values - ordinary.rain.values)
        assert (differences > 1e-10).any(), "Extreme tail stage did not change the first-order adjustment"
        (OUT / "extreme_vs_eqm.json").write_text(json.dumps({
            "changed_values": int((differences > 1e-10).sum()), "max_absolute_change": float(differences.max()),
            "meaning": "Confirms a distinct second-order official ExtremeValues stage, not an accuracy claim",
            "independent_extreme_accuracy_verified": False}, indent=2), encoding="utf-8")
    # Existing example is GEOS-CF, despite its historical era5_sample filename.
    with xr.open_dataset(ROOT / "examples/engineering_case_air_quality/meteorology/era5_sample.nc", engine="h5netcdf") as opened:
        hourly = opened.temperature.load()
        source_attributes = dict(opened.attrs)
    assert hourly.attrs["source_url"].startswith("https://opendap.nccs.nasa.gov/")
    daily = hourly.resample(time="1D").mean(keep_attrs=True)
    assert (hourly.resample(time="1D").count() == 24).all()
    daily.attrs.update(temporal_semantics="interval_mean", preprocessing="24 hourly means per UTC day")
    daily_ds = daily.to_dataset(name="temperature").assign_attrs(source_attributes)
    for op in ("mean", "min", "max"):
        context = AlgorithmContext((EcoDataBundle(daily_ds),), "temperature", "continuous",
            parameters={"statistic": op, "target_frequency": "MS"})
        records.append(save(context, "temporal.xclim_statistics", "real_temperature_" + op,
                            "REAL NASA GEOS-CF; 2 days only, incomplete month explicitly not a monthly climatology"))
    coords = {"lat": [35., 36.], "lon": [110., 111., 112.]}
    pressure = xr.Dataset({"pressure": (("lat", "lon"), np.full((2, 3), 100000.))}, coords=coords, attrs={"crs": "EPSG:4326"})
    pressure.pressure.attrs.update(standard_name="surface_air_pressure", units="Pa")
    aux = xr.Dataset({k: (("lat", "lon"), np.full((2, 3), v)) for k, v in
        {"temperature": 280., "temperature_reference": 280., "elevation_reference": 0., "elevation": 1000.}.items()},
        coords=coords, attrs={"crs": "EPSG:4326"})
    for k in aux:
        aux[k].attrs["units"] = "m" if k.startswith("elevation") else "K"
    context = AlgorithmContext((EcoDataBundle(pressure),), "pressure", "continuous", auxiliaries=(aux,))
    records.append(save(context, "terrain.fords_pressure", "fords_pressure", "SYNTHETIC aligned temperature/DEM/pressure; official pressure component only"))
    (OUT / "execution_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("algorithm", "shape", "status", "runtime_seconds")} for r in records]))


if __name__ == "__main__":
    main()
