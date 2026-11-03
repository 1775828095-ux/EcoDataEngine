"""Metric and information-boundary checks before the real pilot."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import pytest
import xarray as xr
from run_spatial_pilot import AreaReference, area_weights, split, matrix
from ecodata.algorithms.raster_reconstruction_task import degradation


def test_weighted_metric_has_correct_units_and_keeps_cohort():
    ds=xr.Dataset({'t2m':(('lat','lon'),np.ones((4,4))*280)},coords={'lat':np.arange(30.,34.),'lon':np.arange(110.,114.)},attrs={'crs':'EPSG:4326'})
    ds.t2m.attrs.update(units='K',data_kind='continuous')
    coarse,operator=degradation(ds,2,'t2m')
    metric=AreaReference(ds.t2m,coarse,operator,279,'fit')
    predicted=ds.copy(deep=True);predicted.t2m.values+=2
    result=metric(predicted)
    for key in ['objective','RMSE','MAE','Bias','full_RMSE','boundary_RMSE','high_temperature_RMSE','coarse_RMSE']:
        assert result[key]==pytest.approx(2)
    assert result['gradient_RMSE_K_per_km']==pytest.approx(0)
    assert result['denominator']==4
    assert np.all(area_weights(ds.lat,ds.lon)>0)
    predicted.t2m.values[0,0]=np.nan
    with pytest.raises(ValueError,match='Incomplete'):metric(predicted)


def test_temporal_groups_and_gaps_are_frozen():
    assert [day for day in range(1,15) if split(day) is None]==[6,10]
    assert [sum(split(day)==stage for day in range(1,15))*4 for stage in ['fit','internal_holdout','validation','audit']]==[16,4,12,16]


def test_audit_evaluation_is_blocked_before_choice_freeze(tmp_path):
    with pytest.raises(AssertionError):matrix(tmp_path,dict(id='dummy',split='audit'))
