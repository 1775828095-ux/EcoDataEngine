"""Real input: semantic mapping -> GOLEM multi-node graph -> existing Pipeline."""
import json
from pathlib import Path

import numpy as np
from pyproj import CRS

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.golem_search import GolemCapabilitySearch, GolemSearchConfig
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor, RequirementMapper
from ecodata.core.schemas import EcoDataBundle
from ecodata.recognition import recognise_file
from run_algorithm_selection_expansion import chain_steps


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "output/algorithm_library_expansion"
    source = root / "examples/engineering_case_air_quality/background/pm25_background.nc"
    ds, info = recognise_file(source.name, source.read_bytes())
    ds = ds.isel(time=0, drop=True).drop_vars("time_bnds", errors="ignore")
    context = AlgorithmContext((EcoDataBundle(ds),), "PM25", "continuous",
        tuple(np.linspace(float(ds.lat.min()), float(ds.lat.max()), 5)),
        tuple(np.linspace(float(ds.lon.min()), float(ds.lon.max()), 5)), model_constraints={"nonnegative": True})
    geod = CRS.from_user_input(ds.attrs["crs"]).get_geod()
    dx = abs(geod.inv(float(ds.lon[0]), float(ds.lat.mean()), float(ds.lon[1]), float(ds.lat.mean()))[2]) / 1000
    dy = abs(geod.inv(float(ds.lon.mean()), float(ds.lat[0]), float(ds.lon.mean()), float(ds.lat[1]))[2]) / 1000
    source_resolution = max(dx, dy)
    semantics = SemanticFeatureExtractor().extract(ds, "PM25", confirmed={"nonnegative": True, "spatial_resolution_km": source_resolution})
    required = RequirementMapper().map(semantics, {"spatial_resolution_km": float(source_resolution) / 2})
    registry = create_default_registry()
    search = GolemCapabilitySearch(registry).run(required,
        GolemSearchConfig(population=8, generations=8, max_evaluations=60, timeout_seconds=20), context=context)
    for run in search["results"]:
        if run["best_metrics"]["qualified"]:
            steps = chain_steps(run["best_graph_structure"])
            result = UnifiedAlgorithmPipeline(registry).run(context,
                AlgorithmPlan("real_composition", steps, 0, ("required continuous scale processing plus nonnegative validation",)))
            path = folder / "selected_outputs" / f"real_composition_{run['mode']}_{run['seed']}.nc"
            result.data.dataset.to_netcdf(path, engine="h5netcdf")
            run["execution"] = {"status": result.report["status"], "steps": steps,
                                "quality": result.report["quality"], "output": str(path.relative_to(root))}
    path = folder / "algorithm_selection_expansion_experiment.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["real_composition_case"] = {"source": str(source.relative_to(root)), "recognized_format": info.format,
        "scope": "real numerical execution and quality gates; no high-resolution truth or accuracy claim",
        "requirements": required.as_dict(), "resolution_evidence": {"backend": "pyproj.Geod", "dx_km": dx, "dy_km": dy}, "search": search}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print([(r["mode"], r["seed"], r["best_metrics"]["qualified"], r.get("execution", {}).get("steps")) for r in search["results"]])


if __name__ == "__main__":
    main()
