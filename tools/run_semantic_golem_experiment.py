"""Reproduce A/B/C GOLEM comparison with fixed seeds and one mixed capability task."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xarray as xr

from ecodata.algorithms.golem_search import GolemCapabilitySearch, GolemSearchConfig
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.semantic_requirements import RequirementMapper, SemanticFeatureExtractor


root = Path(__file__).resolve().parents[1]
output = root / "output" / "semantic_golem_experiment"
output.mkdir(parents=True, exist_ok=True)
dataset = xr.Dataset({"PM25": (("lat", "lon"), [[21.0, 33.0], [45.0, 60.0]])},
                     coords={"lat": [39.0, 39.1], "lon": [116.0, 116.1]},
                     attrs={"crs": "EPSG:4326", "structure": "grid", "source_name": "small reproducible capability fixture"})
dataset.PM25.attrs.update(units="ug m-3", data_kind="continuous", spatial_semantics="cell_mean", valid_min=0.0)
semantics = SemanticFeatureExtractor().extract(dataset, "PM25", confirmed={
    "spatial_resolution_km": 10.0, "temporal_semantics": "instantaneous"})
requirements = RequirementMapper().map(semantics, {
    "spatial_resolution_km": 5.0, "source_count": 2, "quality_difference": True})
config = GolemSearchConfig(seeds=(11, 22, 33, 44, 55), population=8, generations=4,
                           timeout_seconds=30, alpha=1.0, beta=2.0, gamma=0.25, exploration=0.1)
continuous_result = GolemCapabilitySearch(create_default_registry()).run(requirements, config)
landuse = xr.Dataset({"landuse": (("lat", "lon"), [[1, 2], [2, 3]])}, coords=dataset.coords,
                     attrs={"crs": "EPSG:4326", "structure": "grid", "source_name": "categorical fixture"})
landuse.landuse.attrs.update(units="1", data_kind="categorical", spatial_semantics="category")
land_semantics = SemanticFeatureExtractor().extract(landuse, "landuse", confirmed={
    "spatial_resolution_km": 0.03, "temporal_semantics": "instantaneous"})
land_requirements = RequirementMapper().map(land_semantics, {"spatial_resolution_km": 1.0, "source_count": 1})
categorical_result = GolemCapabilitySearch(create_default_registry()).run(land_requirements, config)
rows = []
for scenario, result in (("continuous_multisource", continuous_result), ("categorical_scale", categorical_result)):
  for item in result["results"]:
    rows.append({"scenario": scenario, **{key: item[key] for key in ("mode", "seed", "mutation_proposals", "actual_evaluations",
                 "invalid_candidates", "qualified_graphs", "unique_evaluated_graphs", "runtime_seconds")},
                 "qualified_best": item["best_metrics"]["qualified"], "best_cost": item["best_metrics"]["cost"],
                 "best_complexity": item["best_metrics"]["complexity"]})
frame = pd.DataFrame(rows)
summary = frame.groupby(["scenario", "mode"]).agg({"mutation_proposals": "mean", "actual_evaluations": "mean",
    "invalid_candidates": "mean", "qualified_graphs": "mean", "unique_evaluated_graphs": "mean",
    "runtime_seconds": "mean", "qualified_best": "sum", "best_cost": "mean", "best_complexity": "mean"}).reset_index()
payload = {"experiment": "A default GOLEM; B default GOLEM plus semantic constraint; C GOLEM plus semantic-guided mutation",
           "data_scope": "small reproducible capability-search fixture; not an environmental accuracy experiment",
           "semantics": {"continuous_multisource": semantics.as_dict(), "categorical_scale": land_semantics.as_dict()},
           "requirements": {"continuous_multisource": requirements.as_dict(), "categorical_scale": land_requirements.as_dict()},
           "config": config.__dict__, "golem": {"continuous_multisource": continuous_result, "categorical_scale": categorical_result},
           "summary": summary.to_dict(orient="records")}
(output / "semantic_golem_experiment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
frame.to_csv(output / "abc_results.csv", index=False, encoding="utf-8-sig")
summary.to_csv(output / "abc_summary.csv", index=False, encoding="utf-8-sig")
print(summary.to_string(index=False))
