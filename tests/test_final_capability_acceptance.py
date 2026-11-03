import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.official_preprocessing import get_preprocessor

ROOT=Path(__file__).parents[1]
OUT=ROOT/"output/capability_final_acceptance"


def test_epa_s2s_reproduces_published_mapping_data():
    for prefix in ("gspro","gscnv"):
        generated=next((OUT/"s2s_run/output").glob(prefix+".CB6R5_AE7_CRITERIA_VOC.CMAQ.*.txt"))
        published=OUT/"official_s2s/output"/(prefix+".CB6R5_AE7_CRITERIA_VOC.CMAQ.2026-02-19.txt")
        a=pd.read_csv(generated,sep=r"\s+",comment="#",header=None,dtype={0:str})
        b=pd.read_csv(published,sep=r"\s+",comment="#",header=None,dtype={0:str})
        pd.testing.assert_frame_equal(a,b,check_dtype=False)
    execution=json.loads((OUT/"s2s_execution.json").read_text())
    assert execution["status"]=="EXECUTED" and not execution["official_core_modified"]


def test_chemical_mass_and_molecular_weight_are_real_not_renaming():
    with xr.open_dataset(OUT/"chemical_speciation.nc",engine="h5netcdf") as ds:
        assert ds.sizes["species"]==30
        assert "PAR" in ds.species and "TOL" in ds.species and "NMOG" not in ds.species
        np.testing.assert_allclose(ds.amount,ds.mass*1000/ds.molecular_weight)
        assert float(abs(ds.mass.sum()-1))<1e-6
        assert ds.mass.attrs["units"]=="kg" and ds.amount.attrs["units"]=="mol"
    report=json.loads((OUT/"chemical_speciation.nc.validation.json").read_text(encoding="utf-8"))
    assert report["official_entry_point"]=="emiproc.speciation.speciate"
    assert not report["renormalized"] and not report["MODEL_NATIVE_VERIFIED"]
    assert report["quality_control"]["mass"]["output"]["status"]=="PASS"


@pytest.mark.parametrize("problem",["mechanism","unit","profile"])
def test_chemical_adapter_rejects_unverified_contract(tmp_path,problem):
    cfg={"mechanism":"CB6R5_AE7","profile_code":"0000","input_pollutant":"TOG","unit":"kg","mass_basis_description":"reference basis"}
    cfg[{"mechanism":"mechanism","unit":"unit","profile":"profile_code"}[problem]]="unsupported"
    with pytest.raises(SchemaError):get_preprocessor("epa_s2s_emiproc").generate("unused","unused",cfg,tmp_path/"bad.nc")
    assert not (tmp_path/"bad.nc").exists()


@pytest.mark.parametrize("kind",["icon","bcon"])
def test_actual_pnc_adapter_and_independent_read(tmp_path,kind):
    cfg={"profile_kind":"static_CB05_AERO5_reference","grid_name":"36US3","VGTYP":7,"VGTOP":5000.,"SDATE":2011222,"STIME":0}
    report=get_preprocessor("pnc_cmaq_profile").generate(kind,OUT/"pnc_official_reference"/(kind+"_profile"),OUT/"pnc_official_reference/GRIDDESC",cfg,tmp_path/(kind+".nc"))
    from netCDF4 import Dataset
    with Dataset(tmp_path/(kind+".nc")) as ds:
        assert ds.FTYPE==(1 if kind=="icon" else 2)
        assert ds.NCOLS==172 and ds.NROWS==148
        assert len(ds.dimensions["LAY"])==6 and len(ds.VGLVLS)==7
        assert ds.TSTEP==0
        assert (ds.variables["TFLAG"][0,:,0]==2011222).all()
        assert (ds.variables["TFLAG"][0,:,1]==0).all()
        if kind=="bcon":assert len(ds.dimensions["PERIM"])==644
        for name in report["species"]:
            assert np.isfinite(ds.variables[name][:]).all()
    assert report["independent_reader"]=="netCDF4.Dataset" and not report["MODEL_NATIVE_VERIFIED"]
    assert all(r["output"]["status"]=="PASS" for r in report["quality_control"].values())


def test_static_profile_does_not_claim_dynamic_or_modern_mechanism(tmp_path):
    adapter=get_preprocessor("pnc_cmaq_profile")
    with pytest.raises(SchemaError):adapter.generate("icon","unused","unused",{"profile_kind":"dynamic_CB6"},tmp_path/"bad.nc")


def test_edgar_real_inventory_mass_time_and_quality():
    report=json.loads((OUT/"real_emissions_validation.json").read_text(encoding="utf-8"))
    assert report["source_is_actual_inventory"] and report["spatial_relative_error"]<1e-8
    assert report["hourly_ratio_sum"]==1.
    np.testing.assert_allclose(report["hourly_total_kg"],report["input_daily_equivalent_kg"])
    with xr.open_dataset(OUT/"edgar_hourly.nc",engine="h5netcdf") as h,xr.open_dataset(OUT/"edgar_spatial.nc",engine="h5netcdf") as s:
        np.testing.assert_allclose(h.mass.sum("time"),s.mass)
        assert h.sizes["time"]==24 and h.mass.attrs["units"]=="kg"
        assert np.all(h.time_bounds.values[:,1]-h.time_bounds.values[:,0]==np.timedelta64(1,"h"))
        assert "not actual daily" in h.attrs["time_support"]


def test_no_unconfirmed_time_profile_assumption(tmp_path):
    with pytest.raises(SchemaError):get_preprocessor("emiproc_edgar_subset").generate("unused",{},tmp_path)


def test_regular_conservation_independent_spherical_area_assertion():
    original=xr.load_dataset(ROOT/"output/algorithm_library_expansion/v4/era5_bth_t2m_reference.nc",engine="h5netcdf").t2m.isel(time=0,lat=slice(0,4),lon=slice(0,4))
    target=xr.load_dataset(OUT/"regular_conservative_real_era5.nc",engine="h5netcdf").value
    def area(field):
        lat=field.lat.values;lon=field.lon.values
        dy=(lat[1]-lat[0])/2;dx=(lon[1]-lon[0])/2
        weights=(np.sin(np.deg2rad(lat+dy))-np.sin(np.deg2rad(lat-dy)))[:,None]*np.deg2rad(2*dx)
        return float(np.sum(field.values*weights))
    assert abs(area(original)-area(target))/area(original)<1e-10


def test_rate_integral_and_coarse_consistency():
    d=xr.load_dataset(OUT/"interval_rate_control.nc",engine="h5netcdf")
    assert float(d.value.sum())*14400==25*14400
    qp=xr.load_dataset(OUT/"qp_reaggregation_control.nc",engine="h5netcdf")
    source=xr.load_dataset(ROOT/"examples/datasets/p3_pm25_coarse_10km_48h.nc",engine="h5netcdf").isel(time=slice(8,9))
    np.testing.assert_allclose(qp.value.coarsen(lat=4,lon=4).mean(),source.value,atol=1e-5)


def test_physical_violation_is_really_blocked():
    report=json.loads((OUT/"physical_violation_expected_block.json").read_text(encoding="utf-8"))
    assert report["status"]=="BLOCK"
    assert any(c["id"]=="output.nonnegative" and c["status"]=="BLOCK" for c in report["checks"])


def test_time_duplicate_is_rejected():
    from ecodata.core.schemas import EcoDataBundle
    from ecodata.temporal.semantic import SemanticTimeConfig,SemanticTimeResampler
    ds=xr.Dataset({"value":("time",[1.,2.])},coords={"time":pd.to_datetime(["2019-08-01","2019-08-01"])},attrs={"temporal_semantics":"instantaneous","data_kind":"continuous"})
    ds.value.attrs.update(units="K",temporal_semantics="instantaneous",data_kind="continuous")
    with pytest.raises(SchemaError):SemanticTimeResampler(SemanticTimeConfig("30min")).apply(EcoDataBundle(ds))


def test_missing_time_values_warn_without_silent_data_repair():
    from ecodata.evaluation.quality_gates import RawInputQualityGate
    ds=xr.Dataset({"value":("time",[1.,np.nan,3.])},coords={"time":pd.date_range("2019-08-01",periods=3,freq="h")})
    ds.value.attrs["units"]="K"
    report=RawInputQualityGate().validate(ds,variable="value",requires_spatial=False)
    assert report["status"]=="WARN" and np.isnan(ds.value[1])


def test_actual_fusion_uses_train_only_and_final_test_is_sealed():
    report=json.loads((OUT/"fusion_execution_manifest.json").read_text(encoding="utf-8"))
    assert report["training_stations"]==40 and report["validation_stations"]==10
    assert not report["final_test_used"]
    for row in report["records"]:
        assert row["report"]["status"] in {"PASS","WARN"}
        with xr.open_dataset(OUT/row["output"],engine="h5netcdf") as ds:assert np.isfinite(ds.pm25).all()
        if row["algorithm"]=="fusion.quality_aware":
            r=row["report"]["steps"][0]["report"]
            assert not r["test_used_for_fitting_or_weighting"]
            assert all(s["bias_correction"]["fit_split"]=="train" for s in r["sources"])
            assert abs(sum(s["weight"] for s in r["sources"])-1)<1e-12


def test_actual_galileo_auxiliary_information_changes_features():
    r=json.loads((OUT/"auxiliary_input_sensitivity.json").read_text())
    assert r["feature_change_max"]>0 and r["counterfactual_only"] and not r["physical_prediction_claimed"]
    with xr.open_dataset(OUT/"galileo_real_terrain.nc",engine="h5netcdf") as ds:assert np.isfinite(ds.galileo_embedding).all()
