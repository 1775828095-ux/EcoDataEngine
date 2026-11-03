from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from ecodata.algorithms.capabilities import capability_catalog, coverage
from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility
from ecodata.algorithms.golem_search import GolemCapabilitySearch, GolemSearchConfig
from ecodata.algorithms.registry import AlgorithmRegistry, create_default_registry
from ecodata.algorithms.semantic_requirements import RequirementMapper, SemanticFeatureExtractor


def grid(name="value", kind="continuous", unit="1", spatial="cell_mean"):
    ds = xr.Dataset({name: (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
                    coords={"lat": [39.0, 39.01], "lon": [116.0, 116.01]},
                    attrs={"crs": "EPSG:4326", "structure": "grid"})
    ds[name].attrs.update(data_kind=kind, units=unit, spatial_semantics=spatial)
    return ds


def requirement(ds, variable, target, confirmed=None):
    semantics = SemanticFeatureExtractor().extract(ds, variable, confirmed=confirmed)
    return semantics, RequirementMapper().map(semantics, target)


def test_lai_landuse_and_precipitation_trigger_different_capabilities():
    lai = grid("LAI"); lai.LAI.attrs["valid_min"] = 0.0
    _, lai_req = requirement(lai, "LAI", {"spatial_resolution_km": 0.03},
                             {"spatial_resolution_km": 0.5, "temporal_semantics": "instantaneous"})
    land = grid("landuse", "categorical", "1", "category")
    _, land_req = requirement(land, "landuse", {"spatial_resolution_km": 1.0},
                              {"spatial_resolution_km": 0.03, "temporal_semantics": "instantaneous"})
    rain = grid("rain", "continuous", "mm", "cell_mean").expand_dims(time=np.array(["2020-01-01T01", "2020-01-01T02"], dtype="datetime64[h]"))
    _, rain_req = requirement(rain, "rain", {"temporal_resolution": "1D"},
                              {"temporal_resolution": "1h", "temporal_semantics": "accumulation"})
    assert "continuous_spatial_resampling" in lai_req.required
    assert "categorical_scale_processing" in land_req.required
    assert "category_preservation" in land_req.hard_constraints
    assert "temporal_accumulation" in rain_req.required
    assert "interval_conservation" in rain_req.hard_constraints


def test_multisource_unknown_and_conflict_are_explicit():
    ds = grid("pm25", "continuous", "")
    semantics, req = requirement(ds, "pm25", {"source_count": 2, "quality_difference": True},
                                 {"variable_class": "categorical", "spatial_resolution_km": 1.0})
    assert semantics.variable_class.status == "CONFLICT"
    assert {"multi_source_fusion", "quality_aware_fusion"}.issubset(req.required)
    assert "unit" in req.unknowns
    assert "no_semantic_conflict" in req.hard_constraints


def test_new_capability_plugin_participates_without_optimizer_changes():
    class Plugin:
        spec = AlgorithmSpec("test.capability", "Test", "1", "test", "available", ("grid",), ("continuous",),
                             capabilities=("brand_new_capability",), compute="low")
        def compatibility(self, context): return Compatibility(True)
        def run(self, context): raise NotImplementedError
    registry = AlgorithmRegistry(); registry.register(Plugin())
    record = capability_catalog(registry)[0]
    assert record.capabilities == ("brand_new_capability",)


def test_official_golem_backend_runs_all_three_modes_and_semantic_mutation():
    if not Path(".venv-golem/Scripts/python.exe").is_file():
        return
    ds = grid("pm25"); ds.pm25.attrs["valid_min"] = 0.0
    _, req = requirement(ds, "pm25", {"source_count": 2, "quality_difference": True},
                         {"spatial_resolution_km": 1.0, "temporal_semantics": "instantaneous"})
    result = GolemCapabilitySearch(create_default_registry()).run(
        req, GolemSearchConfig(seeds=(7,), population=5, generations=2, timeout_seconds=20))
    assert result["backend"] == "thegolem"
    assert {item["mode"] for item in result["results"]} == {"A", "B", "C"}
    c = next(item for item in result["results"] if item["mode"] == "C")
    assert c["mutation_proposals"] > 0
    assert c["semantic_mutation_trace"]
