"""Synthetic mechanism checks, not evidence of environmental accuracy."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.gp_residual import GPResidualPlugin
from ecodata.core.schemas import EcoDataBundle
from ecodata.core.exceptions import SchemaError


def case():
    axes = np.linspace(30, 32, 5), np.linspace(110, 112, 5)
    ds = xr.Dataset({"v": (("lat", "lon"), np.full((5, 5), 10.))},
                    coords={"lat": axes[0], "lon": axes[1], "time": np.datetime64("2020-01-01")},
                    attrs={"crs": "EPSG:4326"})
    ds.v.attrs.update(units="ug m-3", physical_variable="PM2.5", data_kind="continuous", spatial_semantics="point")
    obs = pd.DataFrame(dict(longitude=[110.1, 110.3, 110.1, 110.3], latitude=[30.1, 30.1, 30.3, 30.3],
                            value=[12., 12., 12., 12.], split="train", unit="ug m-3", variable="PM2.5", time=pd.Timestamp("2020-01-01")))
    return AlgorithmContext((EcoDataBundle(ds),), "v", "continuous", tuple(axes[0]), tuple(axes[1]),
                            observations=obs, parameters=dict(observation_crs="EPSG:4326", gp_length_scale_m=20000.,
                                                             gp_signal_std=2., gp_noise_std=.2))


def test_correction_preserves_background_and_decays_without_station_support():
    ctx = case(); plugin = GPResidualPlugin()
    assert plugin.compatibility(ctx).eligible
    before = ctx.sources[0].dataset.copy(deep=True)
    result = plugin.run(ctx)
    out = result.data.dataset
    assert out.time.values == ctx.sources[0].dataset.time.values
    np.testing.assert_allclose(out.v, out.background_value + out.residual_correction)
    assert out.residual_correction.values[0, 0] > 1
    assert abs(out.residual_correction.values[-1, -1]) < 1e-8
    assert out.residual_gp_std.values[-1, -1] > out.residual_gp_std.values[0, 0]
    xr.testing.assert_identical(ctx.sources[0].dataset, before)
    assert result.report["reference_used"] is False


def test_matches_explicit_gp_linear_system():
    ctx = case(); plugin = GPResidualPlugin()
    _, _, _, train, points, query, sampled, _ = plugin.inputs(ctx)
    distance = lambda x,y: ((x[:,None,:]-y[None,:,:])**2).sum(-1)
    kernel = lambda x,y: 4*np.exp(-distance(x,y)/(2*20000**2))
    expected = kernel(query,points) @ np.linalg.solve(kernel(points,points)+.04*np.eye(len(points)), train.value.to_numpy()-sampled)
    np.testing.assert_allclose(plugin.run(ctx).data.dataset.residual_correction.values.ravel(), expected, atol=1e-10)


@pytest.mark.parametrize("change", ["split", "time", "unit", "variable", "missing_parameter", "outside"])
def test_invalid_resources_rejected(change):
    ctx = case(); obs = ctx.observations.copy(); params = dict(ctx.parameters)
    if change == "split": obs.loc[0, "split"] = "test"
    if change == "time": obs.loc[0, "time"] = pd.Timestamp("2020-01-02")
    if change == "unit": obs["unit"] = "mg m-3"
    if change == "variable": obs["variable"] = "O3"
    if change == "missing_parameter": params.pop("gp_noise_std")
    if change == "outside": obs.loc[0,"longitude"] = 113.
    ctx = replace(ctx, observations=obs, parameters=params)
    assert not GPResidualPlugin().compatibility(ctx).eligible
    with pytest.raises((SchemaError, ValueError)): GPResidualPlugin().run(ctx)


def test_validation_reference_cannot_affect_prediction():
    ctx = case(); a = GPResidualPlugin().run(ctx).data.dataset
    b = GPResidualPlugin().run(replace(ctx, validation_reference=xr.full_like(ctx.sources[0].dataset.v,9999))).data.dataset
    xr.testing.assert_identical(a,b)


def test_public_profile_recommends_and_public_executor_runs_same_graph(tmp_path):
    from ecodata.algorithms.platform_search import submit_search, make_context, platform_registry, bind_registry
    from ecodata.algorithms.router import AlgorithmSelector
    from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
    from golem.core.optimisers.graph import OptGraph
    from ecodata.algorithms.platform_search import SearchInput
    from test_profile_matching import fixture
    ctx = case(); ds = ctx.sources[0].dataset
    profile_ds,_,profile_request,_ = fixture()
    variable = profile_request["variable"]
    ds = ds.rename(v=variable)
    ds[variable].attrs["units"] = profile_request["unit"]
    ds[variable].attrs["physical_variable"] = profile_ds[variable].attrs["physical_variable"]
    obs = ctx.observations.copy()
    obs["unit"] = profile_request["unit"]
    obs["variable"] = ds[variable].attrs["physical_variable"]
    table = xr.Dataset.from_dataframe(obs.rename_axis("row"))
    request = dict(task="multisource", variable=variable, unit=profile_request["unit"], semantic_type="continuous",
                   target_lat=list(ctx.target_lat), target_lon=list(ctx.target_lon), parameters=ctx.parameters,
                   model_profile=profile_request["model_profile"],
                   profile_options=dict(candidate_pool=dict(algorithms=["fusion.gp_residual"]), max_steps=1))
    inputs = [SearchInput("background.nc","primary",ds.to_netcdf()), SearchInput("stations.nc","observations",table.to_netcdf())]
    result = submit_search(inputs,request,None,tmp_path/"recommend",mode="Profile")
    assert result["counts"]["real_executions"] == 0
    assert any(n["operation"] == "fusion.gp_residual" for n in result["recommendations"][0]["graph"])
    from ecodata.algorithms.platform_search import DatasetReference
    executed = submit_search(inputs,request,DatasetReference(ds[variable],variable),tmp_path/"execute",mode="Profile")
    assert executed["best"]["file_readback"] == "PASS"
    actual,_,_ = make_context([("primary",ds),("observations",table)],request)
    registry = platform_registry()
    decision = AlgorithmSelector(registry).decide(actual,dict(operation="fusion"))
    factory, external, _ = bind_registry(registry,actual,decision,request)
    graphs = [OptGraph([factory.instantiate(recipe)]) for _,recipe in factory.recipes("validated_field")]
    graph = next(g for g in graphs if any(n.name == "fusion.gp_residual" for n in g.nodes))
    from ecodata.algorithms.platform_search import VERSION
    out = UnifiedAlgorithmPipeline(registry).run_dag(graph,task_id=VERSION,external=external,version=VERSION,decision=decision,factory=factory)
    assert out
