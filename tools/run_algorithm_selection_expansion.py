"""Five tasks, two library sizes, fixed seeds; execute returned chain candidates."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.golem_search import GolemCapabilitySearch, GolemSearchConfig
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.algorithms.semantic_requirements import RequirementMapper, SemanticFeatureExtractor
from ecodata.core.schemas import EcoDataBundle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/algorithm_library_expansion"


def tasks():
    rng = np.random.default_rng(20261016)
    axis = np.linspace(0, 1, 8)
    values = 2 + axis[:, None] + axis[None, :]
    source = xr.Dataset({"LAI": (("lat", "lon"), values,
        {"units": "1", "data_kind": "continuous", "spatial_semantics": "cell_mean"})},
        coords={"lat": 39 + axis * .0315, "lon": 116 + axis * .0405}, attrs={"crs": "EPSG:4326"})
    ctx = AlgorithmContext((EcoDataBundle(source),), "LAI", "continuous",
        tuple(np.linspace(float(source.lat.min()), float(source.lat.max()), 118)),
        tuple(np.linspace(float(source.lon.min()), float(source.lon.max()), 118)))
    yield "A_LAI_500m_to_30m", ctx, {"spatial_resolution_km": .03}, {"spatial_resolution_km": .5}, "synthetic; no high-resolution truth"

    y, x = np.indices((100, 100))
    land = xr.Dataset({"landuse": (("y", "x"), 1 + (x >= 45) + (y >= 70),
        {"units": "1", "data_kind": "categorical", "spatial_semantics": "category"})},
        coords={"y": 15 + np.arange(100) * 30., "x": 15 + np.arange(100) * 30.}, attrs={"crs": "EPSG:32650"})
    ctx = AlgorithmContext((EcoDataBundle(land),), "landuse", "categorical", parameters={
        "target_grid": xr.Dataset(coords={"y": [500., 1500., 2500.], "x": [500., 1500., 2500.]})})
    yield "B_landuse_30m_to_1km", ctx, {"spatial_resolution_km": 1}, {"spatial_resolution_km": .03}, "synthetic category aggregation"

    data = ROOT.parent / "数据/京津冀_PM25融合测试_201908"
    table = pd.read_csv(data / "01_地面监测站/ground_pm25_201908_60stations.csv")
    table = table.loc[table.time_as_stored.eq(table.time_as_stored.min())].rename(columns={
        "longitude_as_stored": "longitude", "latitude_as_stored": "latitude", "pm25_as_stored": "value"})
    table = table.dropna(subset=["value"])
    table["unit"] = "ug/m3"
    with xr.open_dataset(data / "02_GEOS_CF模拟背景场/geos_cf_pm25_201908_bth.nc", engine="h5netcdf") as opened:
        source = opened.isel(time=0, lev=0, drop=True).rename({"pm25_rh35_gcc": "PM25"}).load()
    source.PM25.attrs.update(units="ug/m3", data_kind="continuous", spatial_semantics="cell_mean")
    source.attrs["crs"] = "EPSG:4326"
    lat = tuple(39.8 + np.arange(21) / 111.32)
    lon = tuple(116.1 + np.arange(21) / (111.32 * np.cos(np.deg2rad(39.9))))
    ctx = AlgorithmContext((EcoDataBundle(source),), "PM25", "continuous", lat, lon, observations=table,
        parameters={"observation_crs": "EPSG:4326", "task_operation": "station_to_grid"})
    yield "C_station_PM25_to_1km", ctx, {"operation": "station_to_grid"}, {}, "real 2019 BTH stations; original coordinates approximately WGS84, datum unconfirmed; grid metadata only from GEOS"

    source = xr.Dataset({"temperature": (("lat", "lon"), values + 280,
        {"units": "K", "data_kind": "continuous", "spatial_semantics": "cell_mean"})},
        coords={"lat": 39 + axis * .063, "lon": 116 + axis * .081}, attrs={"crs": "EPSG:4326"})
    target_lat = tuple(np.linspace(float(source.lat.min()), float(source.lat.max()), 234))
    target_lon = tuple(np.linspace(float(source.lon.min()), float(source.lon.max()), 234))
    dem = xr.Dataset({"elevation": (("lat", "lon"), rng.uniform(1, 100, (234, 234)), {"units": "m", "data_kind": "continuous"})},
                     coords={"lat": list(target_lat), "lon": list(target_lon)}, attrs={"crs": "EPSG:4326"})
    ctx = AlgorithmContext((EcoDataBundle(source),), "temperature", "continuous", target_lat, target_lon, auxiliaries=(dem,))
    yield "D_temperature_with_DEM", ctx, {"spatial_resolution_km": .03, "topographic_auxiliary": True, "domain": "weather"}, {"spatial_resolution_km": 1}, "synthetic approximate 1km-to-30m grid; no terrain-informed accuracy claim"

    lat = np.linspace(39, 40, 8)
    lon = np.linspace(116, 117, 8)
    truth = 4 + axis[:, None] + axis[None, :]
    grids = []
    for offset in (2, -1):
        ds = xr.Dataset({"precipitation": (("lat", "lon"), truth + offset,
            {"units": "mm/h", "physical_variable": "precipitation", "data_kind": "continuous", "spatial_semantics": "cell_mean"})},
            coords={"lat": lat, "lon": lon}, attrs={"crs": "EPSG:4326"})
        grids.append(EcoDataBundle(ds))
    observations = pd.DataFrame({"longitude": rng.uniform(116.05, 116.95, 26), "latitude": rng.uniform(39.05, 39.95, 26),
                                 "split": ["train"] * 14 + ["validation"] * 8 + ["test"] * 4, "unit": "mm/h"})
    observations["value"] = 4 + observations.longitude - 116 + observations.latitude - 39 + rng.normal(0, .1, 26)
    ctx = AlgorithmContext(tuple(grids), "precipitation", "continuous", observations=observations,
        parameters={"observation_crs": "EPSG:4326"})
    yield "E_multisource_precipitation", ctx, {"source_count": 2, "quality_difference": True}, {}, "synthetic; checks fusion logic, not real precipitation accuracy"


def chain_steps(graph):
    children = {item["index"]: 0 for item in graph}
    for item in graph:
        if len(item["parents"]) > 1:
            raise ValueError("Multi-parent graph needs a DAG executor; existing Pipeline executes chains")
        for parent in item["parents"]:
            children[parent] += 1
    if sum(not item["parents"] for item in graph) != 1 or any(n > 1 for n in children.values()):
        raise ValueError("Non-chain graph; no implicit flattening into a Pipeline")
    ordered, seen = [], set()
    while len(seen) < len(graph):
        ready = [item for item in graph if item["index"] not in seen and set(item["parents"]).issubset(seen)]
        if not ready:
            raise ValueError("Cycle in output graph")
        for item in ready:
            ordered.append(item["algorithm"])
            seen.add(item["index"])
    return tuple(ordered)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Re-execute saved graphs without repeating search")
    args = parser.parse_args()
    previous = json.loads((OUTPUT / "algorithm_selection_expansion_experiment.json").read_text(encoding="utf-8")) if args.replay else None
    registry = create_default_registry()
    initial = json.loads((OUTPUT / "initial_registry_audit.json").read_text(encoding="utf-8"))
    baseline = tuple(item["id"] for item in initial["registry"]) + ("validation.result",)
    expanded = tuple(spec.id for spec in registry.specs())
    config = GolemSearchConfig(seeds=(11,) if args.smoke else (11, 22, 33, 44, 55), population=8,
                               generations=3 if args.smoke else 8, max_evaluations=20 if args.smoke else 60,
                               timeout_seconds=30)
    experiments = []
    rows = []
    for name, context, target, confirmed, scope in tasks():
        semantics = SemanticFeatureExtractor().extract(context.sources[0].dataset, context.variable, confirmed=confirmed)
        requirements = RequirementMapper().map(semantics, target)
        for library, ids in (("baseline", baseline), ("expanded", expanded)):
            if previous:
                search = next(item["search"] for item in previous["experiments"] if item["task"] == name and item["library"] == library)
            else:
                search = GolemCapabilitySearch(registry).run(requirements, config, context=context, algorithm_ids=ids)
            for run in search["results"]:
                key = f"{name}_{library}_{run['mode']}_{run['seed']}"
                executed = {"status": "NOT_QUALIFIED"}
                if run.get("best_metrics", {}).get("qualified"):
                    try:
                        steps = chain_steps(run["best_graph_structure"])
                        result = UnifiedAlgorithmPipeline(registry).run(context, AlgorithmPlan(key, steps, 0, ("GOLEM candidate execution check",)))
                        path = OUTPUT / "selected_outputs" / (key + ".nc")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        result.data.dataset.to_netcdf(path, engine="h5netcdf")
                        with xr.open_dataset(path, engine="h5netcdf") as restored:
                            assert context.variable in restored
                            if context.target_lat is not None and context.target_lon is not None:
                                np.testing.assert_allclose(restored.lat, context.target_lat)
                                np.testing.assert_allclose(restored.lon, context.target_lon)
                        executed = {"status": result.report["status"], "output": str(path.relative_to(ROOT)), "steps": steps,
                                    "quality": result.report["quality"], "independent_accuracy": result.metrics.get("independent_accuracy_available", False)}
                    except Exception as exc:
                        executed = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
                run["execution"] = executed
                rows.append({"task": name, "library": library, "mode": run["mode"], "seed": run["seed"],
                    **{key: run.get(key) for key in ("mutation_proposals", "actual_evaluations", "invalid_mutation_proposals", "structural_invalid_mutation_proposals", "legal_mutation_ratio",
                        "invalid_evaluations", "invalid_candidates", "qualified_graphs", "unique_evaluated_graphs", "first_qualified_seconds", "runtime_seconds")},
                    "qualified_best": run.get("best_metrics", {}).get("qualified", False), "execution_status": executed["status"]})
            experiments.append({"task": name, "library": library, "scope": scope, "semantics": semantics.as_dict(),
                                "requirements": requirements.as_dict(), "candidate_count": len(ids), "search": search})
            print(name, library, pd.DataFrame(rows).tail(len(config.seeds) * 3).groupby("mode").qualified_best.sum().to_dict(), flush=True)
            (OUTPUT / "algorithm_selection_expansion_experiment.json").write_text(json.dumps({
                "config": config.__dict__, "score_scope": "capability coverage, explicit compatibility, estimated cost and graph complexity; not field accuracy",
                "experiments": experiments, "results": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "selection_results.csv", index=False)
    frame.groupby(["task", "library", "mode"]).agg({"mutation_proposals": "mean", "actual_evaluations": "mean",
        "legal_mutation_ratio": "mean", "qualified_graphs": "mean", "unique_evaluated_graphs": "mean", "qualified_best": "sum",
        "first_qualified_seconds": "mean", "runtime_seconds": "mean"}).to_csv(OUTPUT / "selection_summary.csv")


if __name__ == "__main__":
    main()
