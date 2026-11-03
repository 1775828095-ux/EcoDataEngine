import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.expansion_v5 import OUT, ROOT, V5Plugin, RUNTIME
from ecodata.algorithms.soil_spline import SoilSplinePlugin
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.schemas import EcoDataBundle


def climate():
    if not (OUT/"liif_training.json").exists():
        pytest.skip("Optional trained V5 LIIF artifacts absent")
    with xr.open_dataset(ROOT/"output/algorithm_library_expansion/v4/era5_bth_t2m_coarse.nc", engine="h5netcdf") as d:
        source = d.isel(time=slice(288, 289)).load()
    contract = json.loads((OUT/"liif_training.json").read_text())
    lat0, lat1, lon0, lon1 = contract["bounds"]
    target = xr.Dataset(coords={"lat": np.linspace(lat0+.25, lat1-.25, 12), "lon": np.linspace(lon0+.25, lon1-.25, 12)}, attrs={"crs": "EPSG:4326"})
    return AlgorithmContext((EcoDataBundle(source),), "t2m", "continuous", parameters={"target_grid": target})


def test_real_training_and_multiple_query_sizes():
    c = climate()
    info = json.loads((OUT/"liif_training.json").read_text())
    assert info["optimizer_steps"] == 150 and info["updated_parameters"] > 0
    assert info["weight_origin"] == "LOCAL_TRAINING" and not info["independent_truth"]
    assert info["normalization_fit"] == "training only"
    for scale, size in (("1_5", 9), ("2", 12), ("3", 18), ("4", 24)):
        with xr.open_dataset(OUT/f"liif_{scale}x.nc", engine="h5netcdf") as ds:
            assert ds.value.shape == (48, size, size)
            assert np.isfinite(ds.value).all() and ds.value.attrs["units"] == "K"


@pytest.mark.parametrize("problem", ["lai", "unit", "shape", "region", "target_extent", "target_size", "nan", "crs"])
def test_liif_rejects_incompatible_inputs(problem):
    c = climate()
    ds = c.sources[0].dataset.copy(deep=True)
    target = c.parameters["target_grid"].copy(deep=True)
    if problem == "lai": ds.t2m.attrs["standard_name"] = "leaf_area_index"
    if problem == "unit": ds.t2m.attrs["units"] = "degC"
    if problem == "shape": ds = ds.isel(time=0, drop=True)
    if problem == "region": ds = ds.assign_coords(lat=ds.lat+1)
    if problem == "target_extent": target = target.assign_coords(lon=target.lon+1)
    if problem == "target_size": target = target.isel(lon=slice(0, 6))
    if problem == "nan": ds.t2m.values.flat[0] = np.nan
    if problem == "crs": ds.attrs["crs"] = "EPSG:3857"
    c = replace(c, sources=(EcoDataBundle(ds),), parameters={"target_grid": target})
    assert not V5Plugin("liif").compatibility(c).eligible


def test_liif_actual_pipeline():
    c = climate()
    result = UnifiedAlgorithmPipeline(create_default_registry()).run(c, AlgorithmPlan("liif_test", ("v5.liif",), 0., ("test",)))
    assert result.report["status"] in {"PASS", "WARN"}
    assert result.data.dataset.t2m.shape == (1,12,12)


def test_liif_enters_existing_selector_for_both_grid_contracts():
    from ecodata.algorithms.router import AlgorithmSelector
    from ecodata.algorithms.scale import spatial_scale_factor
    c=climate()
    assert spatial_scale_factor(c)==2
    assert "v5.liif" in {p.id for p in AlgorithmSelector(create_default_registry()).recommend(c)}
    grid=c.parameters["target_grid"]
    c=replace(c,target_lat=tuple(grid.lat.values),target_lon=tuple(grid.lon.values),parameters={})
    assert V5Plugin("liif").compatibility(c).eligible
    assert "v5.liif" in {p.id for p in AlgorithmSelector(create_default_registry()).recommend(c)}


def emissions():
    ds = xr.Dataset({"value": (("y","x"), np.ones((2,2)))}, coords={"y": [.5,1.5],"x": [.5,1.5]}, attrs={"crs":"EPSG:6933"})
    ds.value.attrs.update(units="kg",spatial_semantics="cell_total")
    target = xr.Dataset(coords={"y":[.25,.75,1.25,1.75],"x":[.25,.75,1.25,1.75]}, attrs={"crs":"EPSG:6933"})
    return AlgorithmContext((EcoDataBundle(ds),), "value", "cell_total", parameters={"target_grid":target,"model_preprocessing":True})


@pytest.mark.parametrize("problem", ["opt_in", "density", "geographic", "unit"])
def test_emissions_cannot_be_unconditional_search_node(problem):
    c=emissions(); ds=c.sources[0].dataset
    if problem=="opt_in": c=replace(c,parameters={"target_grid":c.parameters["target_grid"]})
    if problem=="density": ds.value.attrs["spatial_semantics"]="density"
    if problem=="geographic": ds.attrs["crs"]="EPSG:4326"
    if problem=="unit": ds.value.attrs["units"]="kg/m2"
    assert not V5Plugin("emiproc_spatial").compatibility(c).eligible


def test_model_preprocessors_excluded_from_golem_catalog():
    from ecodata.algorithms.capabilities import capability_catalog
    records={r.algorithm_id:r for r in capability_catalog(create_default_registry())}
    assert not records["v5.emiproc_spatial"].executable
    assert not records["v5.emiproc_temporal"].executable
    assert records["v5.liif"].executable


def test_official_emission_numerical_checks_not_model_native_claim():
    rows=json.loads((OUT/"execution_manifest.json").read_text(encoding="utf-8"))
    for row in rows:
        if row["algorithm"].startswith("v5.emiproc"):
            report=row["quality"]["steps"][0]["report"]
            assert report["conservation_relative_error"] < 1e-8
            assert "synthetic" in row["data_kind"]
            assert not report["independent_accuracy_proven"]
    with xr.open_dataset(OUT/"emiproc_temporal.nc", engine="h5netcdf") as ds:
        assert ds.value.attrs["temporal_semantics"]=="accumulation"
        assert ds.value.attrs["time_label"]=="interval_start"
        assert ds.time.attrs["timezone"]=="UTC"
        assert np.all(ds.time_bounds.values[:,1]-ds.time_bounds.values[:,0]==np.timedelta64(1,"h"))


def soil():
    if not (OUT/"real_honcut_profile.csv").exists(): pytest.skip("Optional real soil example absent")
    data=pd.read_csv(OUT/"real_honcut_profile.csv")
    ds=xr.Dataset({"value":("horizon",data.field_ph),"upper":("horizon",data.top),"lower":("horizon",data.bottom)})
    ds.value.attrs["units"]="pH"; ds.upper.attrs["units"]=ds.lower.attrs["units"]="cm"
    return AlgorithmContext((EcoDataBundle(ds),),"value","continuous",parameters={"depth_boundaries_cm":[0,5,15,30,60],"physical_range":[0,14],"allow_missing_horizons":True})


@pytest.mark.parametrize("problem",["missing_confirmation","overlap","unit","extrapolation","ph_mass"])
def test_soil_rejects_unsafe_conversion(problem):
    c=soil(); ds=c.sources[0].dataset; p=dict(c.parameters)
    if problem=="missing_confirmation":p["allow_missing_horizons"]=False
    if problem=="overlap":
        values=ds.upper.values.copy();values[1]=0;ds.upper.data=values
    if problem=="unit":ds.upper.attrs["units"]="mm"
    if problem=="extrapolation":p["depth_boundaries_cm"]=[0,500]
    if problem=="ph_mass":p["conserve_depth_integral"]=True
    assert not SoilSplinePlugin().compatibility(replace(c,parameters=p)).eligible


def test_soil_real_output_and_confirmed_missing_records():
    c=soil()
    result=UnifiedAlgorithmPipeline(create_default_registry()).run(c,AlgorithmPlan("soil",("soil.mpspline2",),0.,("test",)))
    ds=result.data.dataset
    assert np.isfinite(ds.value).all() and ds.value.attrs["units"]=="pH"
    assert result.report["steps"][0]["report"]["dropped_missing_horizons"]==2


def test_soil_depth_integral_for_applicable_linear_property():
    c=soil(); ds=c.sources[0].dataset.isel(horizon=slice(0,7)).copy(deep=True)
    ds.value.values[:]=10.; ds.value.attrs["units"]="g/kg"
    p={"depth_boundaries_cm":[0,10,30,60,122],"physical_range":[0,100],"conserve_depth_integral":True}
    result=SoilSplinePlugin().run(replace(c,sources=(EcoDataBundle(ds),),parameters=p))
    assert result.report["conservation_relative_error"]<1e-8


def test_pyresample_real_coordinates_and_coverage():
    for method in ("nearest","bilinear"):
        with xr.open_dataset(OUT/f"pyresample_{method}.nc", engine="h5netcdf") as ds:
            assert ds.value.shape==(20,20)
            assert np.isfinite(ds.value).mean()>.95
            assert ds.attrs["crs"]=="EPSG:4326"


def test_blocked_plugins_still_refuse_execution():
    c=climate()
    for spec in create_default_registry().specs():
        if spec.status not in {"available"}:
            assert not create_default_registry().get(spec.id).compatibility(c).eligible
