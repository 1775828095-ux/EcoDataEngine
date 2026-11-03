"""Contract fixtures, not environmental accuracy or invented fusion results."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.dag import DAGError
from ecodata.algorithms.pm25_alignment import (INSTANT_UNITS, align_native_covariates,
    formal_blockers, observation_centers, require_formal_spatial_evidence)


def metadata():
    return {"evidence_status": {"timezone": "supported: UTC+8", "hour_window": "supported: preceding hour",
             "pm25_unit": "supported: ug m-3", "coordinate_datum": "unresolved"}}


def sources():
    time = pd.date_range("2019-08-01", periods=3, freq="h")
    geos = xr.Dataset({"pm25_rh35_gcc": (("time", "lev", "lat", "lon"), np.full((2,1,2,2), 20.))},
        coords={"time": time[:-1]+pd.Timedelta(minutes=30), "lev": [72.], "lat": [36.,37.], "lon": [115.,116.]},
        attrs={"temporal_semantics": "hourly mean centered at HH:30 UTC"})
    geos.time.attrs.update(timezone="UTC", cell_methods="time: mean")
    geos.pm25_rh35_gcc.attrs.update(units="ug m-3", humidity_basis="RH35")
    vals = np.arange(3.)[:,None,None]*np.ones((3,2,2))
    era = xr.Dataset({n:(("valid_time","latitude","longitude"), vals.copy()) for n in [*INSTANT_UNITS, "tp"]},
        coords={"valid_time":time, "latitude":[36.,37.], "longitude":[115.,116.]})
    for n,u in INSTANT_UNITS.items():
        era[n].attrs.update(units=u, GRIB_stepType="instant")
    era.tp.attrs.update(units="m", GRIB_stepType="accum", accumulation_comment="over 1 hour ending at valid_time")
    return geos,era


def test_explicit_zone_does_not_receive_second_offset():
    labels=["2019-08-01 09:00", "2019-08-01T01:00:00Z", "2019-08-01T09:00:00+08:00"]
    result=observation_centers(labels,metadata())
    assert (result==pd.Timestamp("2019-08-01 00:30")).all()


def test_missing_time_evidence_blocks():
    m=metadata();m["evidence_status"]["timezone"]="unknown"
    with pytest.raises(DAGError,match="TIME_WINDOW"):
        observation_centers(["2019-08-01 09:00"],m)


def test_wrong_mean_center_is_not_silently_interpolated():
    g,e=sources()
    with pytest.raises(DAGError,match="HH:30"):
        align_native_covariates(g,e,[pd.Timestamp("2019-08-01 00:00")])


def test_accumulation_selected_at_end_not_interpolated():
    g,e=sources();out=align_native_covariates(g,e,g.time.values)
    np.testing.assert_array_equal(out.tp[:,0,0],[1.,2.])
    np.testing.assert_array_equal(out.t2m[:,0,0],[.5,1.5])
    assert out.time_bounds.values[0,0]==e.valid_time.values[0]
    assert out.time_bounds.values[0,1]==e.valid_time.values[1]
    assert out.attrs["station_spatial_matching"]=="NOT_PERFORMED"
    assert out.sizes["lat"]==g.sizes["lat"] and out.sizes["lon"]==g.sizes["lon"]
    assert out.pm25_rh35_gcc.attrs["humidity_basis"]=="RH35"
    assert out.pm25_rh35_gcc.attrs["cell_methods"]=="time: mean"
    assert out.tp.attrs["cell_methods"]=="time: sum"
    assert out.t2m.attrs["cell_methods"]=="time: point"
    assert "cell_methods" not in out.time.attrs
    assert "pm25_observation" not in out


@pytest.mark.parametrize("kind",["no_endpoint","wrong_unit","wrong_rain_semantics","grid_change","duplicate_time","ambient_relabel"])
def test_invalid_edge_contract_is_rejected(kind):
    g,e=sources()
    if kind=="no_endpoint":e=e.isel(valid_time=slice(0,2))
    if kind=="wrong_unit":e.t2m.attrs["units"]="Celsius"
    if kind=="wrong_rain_semantics":e.tp.attrs["accumulation_comment"]="24 hour cumulative"
    if kind=="grid_change":e=e.assign_coords(longitude=[115.1,116.1])
    if kind=="duplicate_time":e=e.assign_coords(valid_time=[e.valid_time.values[0]]*3)
    if kind=="ambient_relabel":g.pm25_rh35_gcc.attrs["humidity_basis"]="ambient"
    with pytest.raises(DAGError):align_native_covariates(g,e,g.time.values)


def test_mask_is_retained_and_never_filled():
    g,e=sources();g.pm25_rh35_gcc.values[0,0,0,0]=np.nan
    out=align_native_covariates(g,e,g.time.values)
    assert np.isnan(out.pm25_rh35_gcc.values[0,0,0,0])
    assert out.pm25_rh35_gcc_valid.values[0,0,0,0]==0


def test_formal_gate_requires_evidence_not_numeric_plausibility():
    g,_=sources();m=metadata()
    assert "STATION_COORDINATE_DATUM_UNCONFIRMED" in formal_blockers(m,g)
    with pytest.raises(DAGError,match="DATUM"):
        require_formal_spatial_evidence(m,g)
    # Only a fixture demonstrating both independently necessary prerequisites.
    confirmed=deepcopy(m)
    confirmed["evidence_status"]["coordinate_datum"]="CONFIRMED"
    confirmed.update(confirmed_crs="EPSG:4326", coordinate_datum_evidence="fixture source declaration")
    with pytest.raises(DAGError,match="VERTICAL_LAYER"):
        require_formal_spatial_evidence(confirmed,g)
    g.attrs["surface_layer_evidence"]="fixture explicit layer declaration"
    assert formal_blockers(confirmed,g)==[]


def test_netcdf_roundtrip_preserves_bounds_units_masks_and_layer(tmp_path):
    g,e=sources();out=align_native_covariates(g,e,g.time.values)
    path=tmp_path/"covariates.nc"
    out.to_netcdf(path,engine="h5netcdf")
    with xr.open_dataset(path,engine="h5netcdf") as reopened:
        xr.testing.assert_equal(out,reopened.load())
        assert reopened.time.encoding["units"]==reopened.time_bounds.encoding["units"]
