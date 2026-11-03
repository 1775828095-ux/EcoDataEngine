from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.capabilities import describe_algorithm
from ecodata.algorithms.climatelearn import ClimateLearnTemperaturePlugin, ready
from ecodata.algorithms.fords_temperature import ForDsTemperaturePlugin
from ecodata.core.schemas import EcoDataBundle

FOLDER = Path(__file__).resolve().parents[1] / "output/algorithm_library_expansion/v4"


def temperature_context():
    if not ready():
        pytest.skip("optional trained ClimateLearn V4 task not prepared")
    with xr.open_dataset(FOLDER / "era5_bth_t2m_coarse.nc", engine="h5netcdf") as ds:
        source = ds.isel(time=slice(288, 289)).load()
    return AlgorithmContext((EcoDataBundle(source),), "t2m", "continuous")


@pytest.mark.parametrize("problem", ["lai", "npp", "units", "scale", "arbitrary", "grid", "nan", "crs"])
def test_local_learned_model_rejects_unsupported_inputs(problem):
    context = temperature_context()
    source = context.sources[0].dataset
    if problem in {"lai", "npp"}:
        source.t2m.attrs["standard_name"] = problem
    elif problem == "units":
        source.t2m.attrs["units"] = "degC"
    elif problem == "scale":
        context.parameters["scale_factor"] = 8
    elif problem == "arbitrary":
        context.parameters["arbitrary_resolution"] = True
    elif problem == "grid":
        context = replace(context, target_lat=tuple(np.linspace(35, 40, 30)))
    elif problem == "nan":
        source.t2m.values[0, 0, 0] = np.nan
    else:
        source.attrs["crs"] = "EPSG:3857"
    assert not ClimateLearnTemperaturePlugin().compatibility(context).eligible


def test_actual_trained_official_model_inference_not_random_initialization():
    context = temperature_context()
    record = json.loads((FOLDER / "climatelearn_training.json").read_text(encoding="utf-8"))
    assert record["optimizer_steps"] == 75 and record["parameters_updated"] > 0
    assert record["weight_origin"] == "LOCAL_TRAINING"
    result = ClimateLearnTemperaturePlugin().run(context)
    assert result.data.dataset.t2m.shape == (1, 24, 24)
    assert result.data.dataset.t2m.attrs["units"] == "K"
    assert np.isfinite(result.data.dataset.t2m).all()
    assert not result.report["arbitrary_resolution_verified"]


def terrain_context():
    if not (FOLDER / "fords_real_coarse_input.nc").exists():
        pytest.skip("optional real California terrain case not prepared")
    with xr.open_dataset(FOLDER / "fords_real_coarse_input.nc", engine="h5netcdf") as ds:
        source = ds.load()
    with xr.open_dataset(FOLDER / "fords_real_target_dem.nc", engine="h5netcdf") as ds:
        target = ds.load()
    return AlgorithmContext((EcoDataBundle(source),), "t2m", "continuous", auxiliaries=(target,), parameters={"target_grid": target})


@pytest.mark.parametrize("problem", ["dem", "coarse_dem", "temperature", "unit", "crs", "coverage", "nan"])
def test_terrain_model_requires_real_compatible_inputs(problem):
    context = terrain_context()
    source, target = context.sources[0].dataset, context.parameters["target_grid"]
    if problem == "dem":
        context.parameters["target_grid"] = target.drop_vars("elevation")
    elif problem == "coarse_dem":
        context = replace(context, sources=(EcoDataBundle(source.drop_vars("elevation_coarse")),))
    elif problem == "temperature":
        source.t2m.attrs["standard_name"] = "leaf_area_index"
    elif problem == "unit":
        target.elevation.attrs["units"] = "unknown"
    elif problem == "crs":
        target.attrs["crs"] = "EPSG:3857"
    elif problem == "coverage":
        context.parameters["target_grid"] = target.assign_coords(lat=target.lat+10)
    else:
        target.elevation.values[0, 0] = np.nan
    assert not ForDsTemperaturePlugin().compatibility(context).eligible


def test_real_official_terrain_output_responds_to_dem_without_claiming_accuracy():
    context = terrain_context()
    result = ForDsTemperaturePlugin().run(context)
    ds = result.data.dataset
    assert ds.t2m.shape == (2, 21, 21)
    assert np.max(np.abs(ds.t2m.values-ds.without_terrain.values)) > 0
    assert not result.report["complete_ForDs"]
    assert not result.report["independent_accuracy_verified"]
    sensitivity = json.loads((FOLDER / "terrain_sensitivity.json").read_text(encoding="utf-8"))
    assert max(abs(sensitivity["temperature_response_min_K"]), abs(sensitivity["temperature_response_max_K"])) > 0


def test_new_plugins_use_existing_registry_and_keep_blocked_models_out():
    registry = create_default_registry()
    for name in ("downscale.climatelearn_t2m", "terrain.fords_temperature"):
        plugin = registry.get(name)
        assert plugin.spec.capabilities
        if plugin.spec.status == "available":
            assert describe_algorithm(plugin.spec).executable
    for name in ("research.climatediffuse", "research.phydl_nwp", "super_resolution.thera"):
        assert not describe_algorithm(registry.get(name).spec).executable


def test_two_dimensional_unrelated_field_does_not_crash_algorithm_selection():
    from ecodata.algorithms.router import AlgorithmSelector
    ds = xr.Dataset({"pm25": (("lat", "lon"), np.ones((2, 2)))}, coords={"lat": [35., 36.], "lon": [115., 116.]}, attrs={"crs": "EPSG:4326"})
    ds.pm25.attrs.update(units="ug m-3", data_kind="continuous", spatial_semantics="cell_mean")
    context = AlgorithmContext((EcoDataBundle(ds),), "pm25", "continuous")
    assert not ClimateLearnTemperaturePlugin().compatibility(context).eligible
    AlgorithmSelector(create_default_registry()).screen(context)
