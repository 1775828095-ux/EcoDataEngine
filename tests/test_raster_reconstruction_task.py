import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from ecodata.algorithms.raster_reconstruction_task import (degradation,overlap_weights,
    CoarseObservationProjection,NonnegativeRaster,select_raster_validation,LocalLIIFRaster)
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.core.schemas import EcoDataBundle


def tile(values):
    ds=xr.Dataset({"aod":(("lat","lon"),values)},coords={"lat":np.arange(len(values))*.01+30,"lon":np.arange(len(values))*.01+110},attrs={"crs":"EPSG:4326"})
    ds.aod.attrs.update(units="1",data_kind="continuous")
    return ds


@pytest.mark.parametrize("n",[4,6,8])
def test_integer_and_fractional_degradation_constant_and_support(n):
    ds=tile(np.ones((12,12))*.6);coarse,a=degradation(ds,n)
    np.testing.assert_allclose(coarse.aod,.6)
    np.testing.assert_allclose(np.asarray(a.sum(1)),1)
    assert a.shape==(n*n,144)


def test_qp_enforces_declared_observation_not_unjustified_total():
    ds=tile(np.arange(144).reshape(12,12)/144);coarse,a=degradation(ds,8)
    estimate=ds.copy(deep=True);estimate.aod.values[:]=.3
    context=AlgorithmContext((EcoDataBundle(estimate),),"aod","continuous",tuple(ds.lat.values),tuple(ds.lon.values),
        parameters={"coarse_reference":coarse,"degradation_operator":a})
    actual=CoarseObservationProjection().run(context)
    np.testing.assert_allclose(a@actual.data.dataset.aod.values.ravel(),coarse.aod.values.ravel(),atol=1e-5)
    assert actual.data.dataset.aod.min()>=0
    assert actual.report["physical_total_conservation_asserted"] is False


def records():
    return [dict(stage="validation",coarse_size=8,tile=t,plan=plan,RMSE=loss,seconds=1.,
        coverage=1.,failure=False,evaluated_pixels=100) for t in ["a","b"] for plan,loss in [("simple+nonnegative",.1),("simple+consistency",.2)]]


def test_selector_chooses_simple_when_combination_worse():
    choice=select_raster_validation(records(),2)
    assert choice["8"]["automatic"]=="simple+nonnegative"


def test_selector_refuses_test_or_unequal_reference_cohort():
    data=records();data[0]["stage"]="test"
    with pytest.raises(ValueError,match="validation"):select_raster_validation(data,2)
    data=records();data[0]["evaluated_pixels"]=99
    with pytest.raises(ValueError,match="mask"):select_raster_validation(data,2)


def test_projection_requires_operator_and_variable_units():
    ds=tile(np.ones((12,12)));context=AlgorithmContext((EcoDataBundle(ds),),"aod","continuous",tuple(ds.lat.values),tuple(ds.lon.values))
    assert not CoarseObservationProjection().compatibility(context).eligible
    ds.aod.attrs["units"]="K"
    assert not NonnegativeRaster().compatibility(context).eligible


def test_missing_task_weight_not_replaced_by_temperature_weights(tmp_path):
    ds=tile(np.ones((12,12)));context=AlgorithmContext((EcoDataBundle(ds),),"aod","continuous",tuple(ds.lat.values),tuple(ds.lon.values))
    check=LocalLIIFRaster(tmp_path/"not_aod_weights.pt").compatibility(context)
    assert not check.eligible
    assert any("checkpoint" in message for message in check.reasons)
