from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.open_source_spatial import XarrayAggregationPlugin, XesmfPlugin
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.open_source_stations import PyKrigePlugin, MetPyStationPlugin, MetPyVerticalPlugin
from ecodata.algorithms.open_source_bias import XsdbaPlugin


def category_context():
    ds = xr.Dataset({"landuse": (("y", "x"), np.tile([1, 1, 2, 2], (4, 1)))},
                    coords={"y": [15., 45., 75., 105.], "x": [15., 45., 75., 105.]},
                    attrs={"crs": "EPSG:32650"})
    ds.landuse.attrs.update(units="1", data_kind="categorical", spatial_semantics="category")
    return AlgorithmContext((EcoDataBundle(ds),), "landuse", "categorical",
                            parameters={"target_grid": xr.Dataset(coords={"y": [30., 90.], "x": [30., 90.]})})


def test_most_common_and_fractions_are_distinct_real_upstream_operations():
    context = category_context()
    mode = XarrayAggregationPlugin("most_common").run(context).data.dataset.landuse
    np.testing.assert_array_equal(mode, [[1, 2], [1, 2]])
    fractions = XarrayAggregationPlugin("fractions").run(context)
    np.testing.assert_allclose(fractions.data.dataset.landuse.sum("category"), 1)
    assert fractions.report["not_equal_to"] == "most_common"


@pytest.mark.parametrize("method", ["mean", "median", "min", "max"])
def test_upstream_statistics_preserve_units_and_are_not_categorical(method):
    context = category_context()
    plugin = XarrayAggregationPlugin(method)
    assert not plugin.compatibility(context).eligible
    context = replace(context, semantic_type="continuous")
    result = plugin.run(context)
    assert result.data.dataset.landuse.attrs["units"] == "1"
    assert np.isfinite(result.data.dataset.landuse).all()


def test_category_aggregation_rejects_refinement_and_unknown_support():
    context = category_context()
    finer = xr.Dataset(coords={"y": [15., 30., 45.], "x": [15., 30., 45.]})
    assert not XarrayAggregationPlugin("most_common").compatibility(replace(context, parameters={"target_grid": finer})).eligible
    context.sources[0].dataset.landuse.attrs.pop("spatial_semantics")
    assert not XarrayAggregationPlugin("fractions").compatibility(context).eligible


def test_xesmf_conservative_never_infers_missing_bounds():
    context = category_context()
    reasons = XesmfPlugin("conservative").compatibility(context).reasons
    assert any("INPUT_REQUIREMENT_MISSING" in message for message in reasons)
    with pytest.raises(SchemaError, match="INPUT_REQUIREMENT_MISSING"):
        XesmfPlugin("conservative").run(context)


def station_context():
    import pandas as pd
    rng = np.random.default_rng(112)
    context = replace(category_context(), variable="landuse", semantic_type="continuous")
    observations = pd.DataFrame({"longitude": rng.uniform(15, 105, 15), "latitude": rng.uniform(15, 105, 15),
                                 "value": rng.uniform(5, 20, 15), "split": ["train"] * 12 + ["validation"] * 3, "unit": "1"})
    return replace(context, observations=observations, parameters={**context.parameters, "observation_crs": "EPSG:32650"})


@pytest.mark.parametrize("method", ["ordinary", "universal"])
def test_pykrige_excludes_heldout_values_and_outputs_variance(method):
    context = station_context()
    first = PyKrigePlugin(method).run(context)
    table = context.observations.copy()
    table.loc[table.split.eq("validation"), "value"] += 10000
    second = PyKrigePlugin(method).run(replace(context, observations=table))
    xr.testing.assert_equal(first.data.dataset, second.data.dataset)
    assert "kriging_variance" in first.data.dataset
    assert first.report["training_count"] == 12


def test_pykrige_missing_units_duplicate_coordinates_and_z_are_blocked():
    context = station_context()
    assert not PyKrigePlugin("ordinary3d").compatibility(context).eligible
    table = context.observations.copy()
    table.loc[1, ["longitude", "latitude"]] = table.loc[0, ["longitude", "latitude"]].values
    assert not PyKrigePlugin("ordinary").compatibility(replace(context, observations=table)).eligible
    table = context.observations.drop(columns="unit")
    assert not PyKrigePlugin("ordinary").compatibility(replace(context, observations=table)).eligible


@pytest.mark.parametrize("method", ["linear", "natural_neighbor", "barnes", "cressman"])
def test_metpy_interpolation_reports_uncovered_pixels_without_silent_fill(method):
    context = station_context()
    output = MetPyStationPlugin(method).run(context)
    assert output.report["backend"] == "MetPy"
    assert output.report["training_count"] == 12
    if method in {"linear", "natural_neighbor"}:
        assert output.report["nan_fraction"] > 0
    assert np.isfinite(output.data.dataset.landuse.values).any()


def test_metpy_vertical_preserves_levels_and_rejects_extrapolation():
    ds = xr.Dataset({"temperature": ("pressure", [270., 280., 290.], {"units": "K"})},
                    coords={"pressure": ("pressure", [700., 850., 1000.], {"units": "hPa"})})
    context = AlgorithmContext((EcoDataBundle(ds),), "temperature", "continuous",
        parameters={"vertical_coordinate": "pressure", "target_levels": [775., 925.]})
    output = MetPyVerticalPlugin().run(context).data.dataset
    np.testing.assert_allclose(output.temperature, [275., 285.])
    assert output.pressure.attrs["units"] == "hPa"
    assert not MetPyVerticalPlugin().compatibility(replace(context, parameters={"vertical_coordinate": "pressure", "target_levels": [600.]})).eligible


def bias_context():
    import pandas as pd
    time = pd.date_range("2019-08-01", periods=120, freq="h")
    ref = xr.DataArray(280 + np.sin(np.arange(80) / 6), dims="time", coords={"time": time[:80]}, attrs={"units": "K", "split": "train"})
    hist = (ref + 20).assign_attrs(ref.attrs)
    sim = xr.DataArray(300 + np.sin(np.arange(80, 120) / 6), dims="time", coords={"time": time[80:]}, attrs={"units": "K"})
    return AlgorithmContext((EcoDataBundle(sim.to_dataset(name="temperature")),), "temperature", "continuous",
                            parameters={"training_reference": ref, "historical_simulation": hist, "nquantiles": 10})


@pytest.mark.parametrize("method", ["eqm", "dqm", "qdm"])
def test_xsdba_official_adjustment_and_disjoint_application(method):
    from ecodata.engines.units import UnitEngine, registry
    context = bias_context()
    before = registry.force_ndarray_like
    result = XsdbaPlugin(method).run(context)
    assert registry.force_ndarray_like == before
    converted = UnitEngine().convert(xr.DataArray([1.], attrs={"units": "mg/L"}), "ug/L")
    assert converted.item() == pytest.approx(1000)
    assert np.isfinite(result.data.dataset.temperature).all()
    assert float(result.data.dataset.temperature.mean()) < 285
    assert result.report["training_application_overlap"] == 0
    parameters = {**context.parameters, "training_reference": context.sources[0].dataset.temperature.assign_attrs(units="K", split="train"),
                  "historical_simulation": context.sources[0].dataset.temperature.assign_attrs(units="K", split="train")}
    assert not XsdbaPlugin(method).compatibility(replace(context, parameters=parameters)).eligible


def test_declared_category_to_fraction_transition_passes_existing_quality_gate():
    from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
    from ecodata.algorithms.registry import create_default_registry
    from ecodata.algorithms.router import AlgorithmPlan
    context = category_context()
    result = UnifiedAlgorithmPipeline(create_default_registry()).run(context,
        AlgorithmPlan("fractions", ("spatial.xarray_fractions",), 0, ("explicit structure conversion",)))
    assert result.report["quality"]["status"] == "PASS"
    assert result.data.dataset.landuse.attrs["data_kind"] == "continuous"


def test_conservative_cell_mean_integral_is_preserved_by_real_backend():
    from ecodata.engines.grid import GridEngine
    ctx = category_context()
    ds = ctx.sources[0].dataset
    # Rename projected coordinates only for the legacy GridEngine contract.
    source = ds.rename({"y": "lat", "x": "lon"})
    target = ctx.parameters["target_grid"].rename({"y": "lat", "x": "lon"})
    output, evidence = GridEngine().regrid(source, target, "conservative", backend="fallback")
    assert evidence["backend"] == "xarray-regrid"
    assert float(output.landuse.sum()) * 60**2 == pytest.approx(float(source.landuse.sum()) * 30**2)


def test_golem_context_excludes_incompatible_capabilities_before_guided_mutation():
    from pathlib import Path
    from ecodata.algorithms.golem_search import GolemCapabilitySearch, GolemSearchConfig
    from ecodata.algorithms.registry import create_default_registry
    from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor, RequirementMapper
    if not Path(".venv-golem/Scripts/python.exe").is_file():
        pytest.skip("Optional official GOLEM environment unavailable")
    ctx = category_context()
    semantics = SemanticFeatureExtractor().extract(ctx.sources[0].dataset, ctx.variable,
        confirmed={"spatial_resolution_km": .03, "temporal_semantics": "instantaneous"})
    req = RequirementMapper().map(semantics, {"spatial_resolution_km": 1.0})
    assert "categorical_aggregation" in req.required
    result = GolemCapabilitySearch(create_default_registry()).run(req,
        GolemSearchConfig(seeds=(3,), population=5, generations=2, timeout_seconds=15, max_evaluations=20), context=ctx)
    legal = {item["algorithm_id"] for item in result["candidate_compatibility"] if item["eligible"] and item["executable"]}
    c = next(item for item in result["results"] if item["mode"] == "C")
    assert all(trace["candidate"] in legal for trace in c["semantic_mutation_trace"])
    assert c["actual_evaluations"] <= 20
    assert c["best_metrics"]["qualified"]
    assert "super_resolution.thera" not in legal


def test_existing_idw_honors_requested_target_grid():
    from ecodata.algorithms.registry import create_default_registry
    import pandas as pd
    ds = xr.Dataset({"value": (("lat", "lon"), [[1., 2.], [3., 4.]], {"units": "1"})},
                    coords={"lat": [39., 40.], "lon": [116., 117.]}, attrs={"crs": "EPSG:4326"})
    obs = pd.DataFrame({"longitude": [116., 117., 116.5], "latitude": [39., 40., 39.7],
                        "value": [1., 2., 3.], "unit": "1", "split": "train"})
    context = AlgorithmContext((EcoDataBundle(ds),), "value", "continuous", (39.2, 39.4, 39.6), (116.2, 116.4, 116.6), observations=obs)
    output = create_default_registry().get("fusion.idw").run(context).data.dataset
    np.testing.assert_allclose(output.lat, context.target_lat)
    np.testing.assert_allclose(output.lon, context.target_lon)


def test_regrid_search_compatibility_checks_confirmed_spatial_support():
    from ecodata.algorithms.registry import create_default_registry
    ds = xr.Dataset({"value": (("lat", "lon"), [[1., 2.], [3., 4.]], {"units": "1", "data_kind": "continuous"})},
                    coords={"lat": [39., 40.], "lon": [116., 117.]}, attrs={"crs": "EPSG:4326"})
    context = AlgorithmContext((EcoDataBundle(ds),), "value", "continuous", (39., 39.5, 40.), (116., 116.5, 117.))
    registry = create_default_registry()
    assert registry.get("spatial.bilinear").compatibility(context).eligible
    assert not registry.get("spatial.conservative").compatibility(context).eligible
    ds.value.attrs["spatial_semantics"] = "cell_mean"
    assert registry.get("spatial.conservative").compatibility(context).eligible
