from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.dual_times_field import DualTimesFieldPlugin, source_ready
from ecodata.algorithms.phydl_contract import REQUIRED, check_phydl_inputs
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.algorithms.capabilities import describe_algorithm
from ecodata.core.schemas import EcoDataBundle


def irregular_context():
    ds = xr.Dataset({"value": ("time", [1., 2., np.nan, 3., 4., 5.])}, coords={"time": [0., 1., 1.5, 4., 7., 10.]})
    ds.value.attrs["units"] = "1"
    return AlgorithmContext((EcoDataBundle(ds),), "value", "continuous", parameters={"epochs": 2})


@pytest.mark.parametrize("problem", ["duplicate", "descending", "inf_time", "string_time", "inf_value", "few_observations", "forecast", "no_units", "epochs"])
def test_dualtime_preflight_rejects_unsupported_input(problem):
    context = irregular_context()
    ds = context.sources[0].dataset
    if problem == "duplicate":
        ds = ds.assign_coords(time=[0., 1., 1., 4., 7., 10.])
    elif problem == "descending":
        ds = ds.isel(time=slice(None, None, -1))
    elif problem == "inf_time":
        ds = ds.assign_coords(time=[0., 1., 1.5, 4., 7., np.inf])
    elif problem == "string_time":
        ds = ds.assign_coords(time=list("abcdef"))
    elif problem == "inf_value":
        ds.value.values[1] = np.inf
    elif problem == "few_observations":
        ds.value.values[:4] = np.nan
    elif problem == "forecast":
        context.parameters["target_times"] = [0., 11.]
    elif problem == "no_units":
        ds.value.attrs.clear()
    else:
        context.parameters["epochs"] = 0
    context = replace(context, sources=(EcoDataBundle(ds),))
    assert not DualTimesFieldPlugin().compatibility(context).eligible


def test_official_dualtime_fit_predict_pipeline_mask_and_alignment(tmp_path):
    if not source_ready():
        pytest.skip("optional official DTF environment not prepared")
    context = irregular_context()
    result = UnifiedAlgorithmPipeline(create_default_registry()).run(context,
        AlgorithmPlan("test_irregular", ("temporal.dualtimesfield",), 0., ("official fit/predict smoke",)))
    out = result.data.dataset
    assert np.isfinite(out.value).all()
    np.testing.assert_array_equal(out.time, context.sources[0].dataset.time)
    np.testing.assert_allclose(out.value.values[[0, 1, 3, 4, 5]], [1, 2, 3, 4, 5])
    assert out.observed_mask.values[2] == 0
    assert not result.report["steps"][0]["report"]["hidden_values_used_for_training"]
    out.to_netcdf(tmp_path / "result.nc", engine="h5netcdf")
    with xr.open_dataset(tmp_path / "result.nc", engine="h5netcdf") as reread:
        xr.testing.assert_equal(out, reread)


def test_queries_outside_observed_not_just_input_support_are_rejected():
    context = irregular_context()
    context.sources[0].dataset.value.values[0] = np.nan
    assert not DualTimesFieldPlugin().compatibility(context).eligible


def weather_inputs():
    ds = xr.Dataset({k: (("latitude", "longitude"), np.ones((2, 2))) for k in REQUIRED},
                    coords={"latitude": [35., 36.], "longitude": [105., 106.]}, attrs={"crs": "EPSG:4326"})
    for k, unit in REQUIRED.items():
        ds[k].attrs["units"] = unit
    return ds


@pytest.mark.parametrize("missing", list(REQUIRED) + ["latitude", "longitude"])
def test_phydl_requires_dem_weather_and_geography(missing):
    assert not check_phydl_inputs(weather_inputs().drop_vars(missing)).eligible


def test_phydl_input_contract_is_not_execution_verification():
    assert check_phydl_inputs(weather_inputs()).eligible
    ds = weather_inputs()
    ds.t2m.attrs["units"] = "degC"
    assert not check_phydl_inputs(ds).eligible
    spec = create_default_registry().get("research.phydl_nwp").spec
    assert not describe_algorithm(spec).executable


@pytest.mark.parametrize("algorithm", ["research.consistency_climate", "research.dsfno", "research.geofar", "super_resolution.thera"])
def test_c5_candidates_do_not_pretend_arbitrary_scale_is_verified(algorithm):
    plugin = create_default_registry().get(algorithm)
    assert not describe_algorithm(plugin.spec).executable
    context = irregular_context()
    context.parameters.update(target_scale_factor=33.3, arbitrary_resolution=True)
    assert not plugin.compatibility(context).eligible
    assert not plugin.spec.output_contract.get("arbitrary_resolution_verified", False)


def test_status_and_model_adapter_separation():
    registry = create_default_registry()
    for spec in registry.specs():
        if spec.verification_status in {"BLOCKED", "RESEARCH_CANDIDATE"}:
            assert not describe_algorithm(spec).executable
        assert spec.category != "MODEL_SPECIFIC"
    assert registry.get("terrain.fords_pressure").spec.capabilities == ("terrain_pressure_adjustment",)
    assert "irregular_time_interpolation" in registry.get("temporal.dualtimesfield").spec.capabilities


def test_official_phydl_archives_and_real_uschn_evidence():
    folder = Path(__file__).resolve().parents[1] / "output/algorithm_library_expansion/v3"
    if not (folder / "source_manifest.json").exists():
        pytest.skip("optional official downloads not prepared")
    r = json.loads((folder / "source_manifest.json").read_text(encoding="utf-8"))
    assert not r["phydl_archive"]["data_members"]
    assert not r["phydl_latest_archive"]["data_members"]
    run = json.loads((folder / "phydl_execution.json").read_text(encoding="utf-8"))
    assert run["returncode"] != 0 and "ningxia_real_train.npy" in run["stderr"]
