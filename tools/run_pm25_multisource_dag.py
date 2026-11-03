"""Execute actual GOLEM parent graphs against existing real sources; exploratory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / ".venv-golem/Lib/site-packages"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xarray as xr
from golem.core.optimisers.graph import OptGraph, OptNode
from ecodata.algorithms.dag import Artifact, NumericalDAGExecutor
from ecodata.algorithms.pm25_dag import PM25DAGAdapters, TARGET, UNIT
from ecodata.algorithms.registry import create_default_registry


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph(path):
    def node(name, parents=(), **parameters):
        return OptNode({"name": name, "params": parameters}, nodes_from=list(parents))
    geos = node("source.geos", source="geos")
    stations = node("source.stations", source="stations")
    g = node("prepare.geos", [geos])
    obs = node("prepare.stations", [stations])
    if path == "external_drift":
        meteo = node("prepare.era5", [node("source.era5", source="era5")])
        estimate = node("station.pykrige_universal", [g, meteo, obs])
    elif path == "calibrated_residual":
        history = node("source.history", source="history")
        state = node("fusion.quality_aware", [geos, history], mode="fit")
        corrected = node("fusion.quality_aware", [g, state], mode="apply")
        regridded = node("spatial.bilinear", [corrected])
        estimate = node("fusion.residual", [regridded, obs], interpolation="linear")
    else:
        raise ValueError(path)
    return OptGraph([node("quality.result", [estimate, obs])])


def graph_record(g):
    nodes = list(g.nodes)
    return [{"operation": n.name, "parameters": n.content.get("params", n.content.get("parameters", {})),
             "parents": [nodes.index(p) for p in n.nodes_from]} for n in nodes]


def prepare(data):
    audit = json.loads((ROOT / "output/golem_multisource_round2_preparation/preparation_audit.json").read_text(encoding="utf-8"))
    stations_path = Path(audit["station"]["source"]["path"])
    raw = pd.read_csv(stations_path)
    frame = raw.rename(columns={"longitude_as_stored": "longitude", "latitude_as_stored": "latitude",
                                "pm25_as_stored": "value"}).copy()
    frame["time"] = pd.to_datetime(frame.time_as_stored)-pd.Timedelta(hours=8, minutes=30)
    frame["unit"] = UNIT
    train = frame[frame.split.eq("train")].copy()
    history = train[pd.to_datetime(train.time_as_stored).between("2019-08-02", "2019-08-14 23:00:00")].copy()
    live_times = pd.to_datetime(["2019-08-15 00:30:00", "2019-08-15 12:30:00"])
    live = train[train.time.isin(live_times)].copy()
    geos_path = data / "background_geos_cf/subset/geos_cf_pm25_201908_bth.nc"
    era_path = data / "meteorology_era5/anonymous_public/era5_201908_bth.nc"
    with xr.open_dataset(geos_path, engine="h5netcdf") as opened:
        geos = opened.load()
    with xr.open_dataset(era_path, engine="h5netcdf") as opened:
        era = opened.load()
    external = {"geos": Artifact("geos_series", geos), "era5": Artifact("meteo_series", era),
                "stations": Artifact("station_series", live), "history": Artifact("history_observations", history)}
    sources = [{"path": str(p), "sha256": sha(p)} for p in (stations_path, geos_path, era_path)]
    # Test target values never passed to evaluator or artifacts.
    validation = frame[frame.split.eq("validation")].copy()
    return external, validation, sources


def evaluate(result, adapter, validation, directory):
    ds = result["outputs"]["data"].value
    path = directory / "result.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as reopened:
        xr.testing.assert_allclose(ds, reopened.load())
        for name in ("units", "physical_variable"):
            if reopened[TARGET].attrs.get(name) != ds[TARGET].attrs.get(name):
                raise ValueError(f"NetCDF target metadata lost: {name}")
        for name in ("crs", "crs_evidence", "window_start_utc", "window_end_utc"):
            if reopened.attrs.get(name) != ds.attrs.get(name):
                raise ValueError(f"NetCDF spatial/time evidence lost: {name}")
    intermediate_file = directory / "intermediates.nc"
    written = False
    groups = []
    for node_id, ports in result["intermediates"].items():
        for port, artifact in ports.items():
            if artifact.kind in {"geos_covariate", "meteo_covariate", "calibrated_background", "calibrated_analysis_grid", "analysis_field"}:
                group = f"{node_id}/{port}"
                artifact.value.to_netcdf(intermediate_file, engine="h5netcdf", group=group, mode="a" if written else "w")
                with xr.open_dataset(intermediate_file, engine="h5netcdf", group=group) as reopened:
                    xr.testing.assert_allclose(artifact.value, reopened.load())
                written = True
                groups.append({"group": group, "kind": artifact.kind})
    target = validation[validation.time.eq(adapter.center) & np.isfinite(validation.value)].copy()
    x = xr.DataArray(target.longitude.to_numpy(), dims="point")
    y = xr.DataArray(target.latitude.to_numpy(), dims="point")
    # Eligibility comes from geometry only; never from candidate error or prediction availability.
    support = ds.valid_domain.interp(lat=y, lon=x).values
    eligible = np.isfinite(support) & (support >= 1-1e-10)
    target = target.loc[eligible].copy()
    if target.empty:
        raise ValueError("No validation sites with fully supported grid neighbourhood")
    predicted = ds[TARGET].interp(lat=xr.DataArray(target.latitude.to_numpy(), dims="point"),
                                 lon=xr.DataArray(target.longitude.to_numpy(), dims="point")).values
    if not np.isfinite(predicted).all():
        raise ValueError("Candidate missing predictions for predeclared geometry-supported targets")
    target["prediction"] = predicted
    target[["station_id", "time", "longitude", "latitude", "value", "prediction"]].to_csv(directory / "validation_predictions.csv", index=False)
    delta = predicted-target.value.to_numpy()
    metrics = {"count": len(target), "RMSE": float(np.sqrt(np.mean(delta**2))),
        "MAE": float(np.mean(np.abs(delta))), "Bias": float(np.mean(delta)),
        "supported_station_ids": target.station_id.tolist(), "geometry_excluded": int((~eligible).sum())}
    dump(directory / "execution_trace.json", {"trace": result["trace"], "quality": result["outputs"]["report"].value,
        "metrics": metrics, "netcdf_readback": "PASS", "output_sha256": sha(path),
        "intermediate_groups": groups, "intermediate_note": "Ungated intermediate estimates are not final deliverable predictions"})
    # Save transmitted training state separately; it can be audited without parsing NetCDF attrs.
    states = [a.value for ports in result["intermediates"].values() for a in ports.values() if a.kind == "calibration_state"]
    if states:
        dump(directory / "calibration_state.json", states)
    return metrics


def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "execution_results.json").exists():
        raise FileExistsError("Use a new run directory, do not overwrite evidence")
    external, validation, sources = prepare(data)
    registry = create_default_registry()
    frozen_path = ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_ids = {a["id"] for a in frozen["algorithms"]}
    code = [ROOT / "src/ecodata/algorithms/dag.py", ROOT / "src/ecodata/algorithms/pm25_dag.py",
            ROOT / "src/ecodata/algorithms/quality_aware.py", ROOT / "src/ecodata/fusion/residual.py",
            ROOT / "src/ecodata/algorithms/plugins.py"]
    version = hashlib.sha256("".join(sha(p) for p in code).encode()).hexdigest()
    manifest = {"status": "EXPLORATORY", "assumption": "Original station coordinates treated as WGS84 only for engineering test",
        "crs_verified": False, "independent_full_field_truth": False, "test_metrics_computed": False,
        "sources": sources, "frozen_pool_sha256": sha(frozen_path), "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
        "validation_centers_utc": ["2019-08-15 00:30:00", "2019-08-15 12:30:00"],
        "fit_window_beijing": ["2019-08-02", "2019-08-14 23:00:00"],
        "target_grid_note": "0.1 degree analysis representation; not verified 0.1 degree physical detail"}
    dump(out / "run_manifest.json", manifest)
    results = []
    for index, center in enumerate(manifest["validation_centers_utc"]):
        adapter = PM25DAGAdapters(registry, center, np.arange(36.5, 40.6, .1), np.arange(114.4, 118.4, .1), exploratory=True)
        executor = NumericalDAGExecutor(adapter.adapters(), version=version+center)
        for name in ("external_drift", "calibrated_residual"):
            folder = out / f"slice{index}_{name}"
            folder.mkdir(exist_ok=True)
            g = graph(name)
            assert all(n.name in frozen_ids or n.name.startswith(("source.", "prepare.", "quality.")) for n in g.nodes)
            dump(folder / "golem_graph.json", graph_record(g))
            started = perf_counter()
            try:
                result = executor.run_golem(g, external=external)
                metrics = evaluate(result, adapter, validation, folder)
                # Replay is executed by the same DAG engine, without re-fitting algorithms.
                replay = executor.run_golem(g, external=external)
                xr.testing.assert_identical(result["outputs"]["data"].value, replay["outputs"]["data"].value)
                results.append({"path": name, "center": center, "status": "WARN_EXPLORATORY", "metrics": metrics,
                    "seconds": perf_counter()-started, "all_replay_nodes_cached": all(t["cache_hit"] for t in replay["trace"]),
                    "trace": str(folder / "execution_trace.json")})
            except Exception as exc:
                failure = {"path": name, "center": center, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                           "trace": getattr(exc, "trace", [])}
                dump(folder / "failure.json", failure)
                results.append(failure)
                print(failure["error"], flush=True)
            dump(out / "execution_results.json", results)
    print(json.dumps([{k: v for k, v in r.items() if k != "trace"} for r in results], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
