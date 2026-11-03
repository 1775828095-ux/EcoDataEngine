"""Synthetic contract checks only; not environmental accuracy evidence."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from run_fixed_validation_enumeration import METHODS, qualify, summarize


def inputs():
    train = pd.DataFrame({'longitude': [-122,-122,-121,-120,-120,-120,-121,-122],
        'latitude': [36,37,38,38,37,36,36,38], 'aod': np.arange(8.),
        'temperature_2m': np.arange(8.)**2})
    validation = pd.DataFrame({'station_id': ['held'], 'time': ['2022-05-01'],
        'longitude': [-121.], 'latitude': [37.], 'value': [10.]})
    cov = xr.Dataset({'aod': (('lat','lon'), np.ones((5,5))),
                      'temperature_2m': (('lat','lon'), np.ones((5,5))*20)},
        coords={'lat': [36.,36.5,37.,37.5,38.], 'lon': [-122.,-121.5,-121.,-120.5,-120.]})
    return train, validation, cov


def test_cohort_qualification_cannot_use_validation_target():
    train, val, cov = inputs()
    before = qualify(train, val, cov)
    val['value'] = 1e9
    pd.testing.assert_frame_equal(before, qualify(train, val, cov))
    assert before.reason.iloc[0] == 'QUALIFIED'


def test_covariate_hole_rejects_stencil_without_filling():
    train, val, cov = inputs()
    cov.aod.values[2,2] = np.nan
    assert qualify(train, val, cov).reason.iloc[0] == 'INCOMPLETE_COMMON_READBACK_STENCIL'
    assert np.isnan(cov.aod.values[2,2])


def test_low_station_count_is_a_common_exclusion():
    train, val, cov = inputs()
    assert qualify(train.iloc[:7], val, cov).reason.iloc[0] == 'INPUT_STATIONS_LT_8'


def test_incomplete_method_cannot_rank_on_its_successful_subset():
    rows, runs = [], []
    for method in METHODS:
        for i in range(2):
            fail = method == 'route_b' and i == 1
            rows.append(dict(method=method, status='FAIL' if fail else 'PASS',
                             prediction=np.nan if fail else 9., value=10.))
            runs.append(dict(method=method, status='FAIL' if fail else 'PASS', seconds=.1,
                             valid_pixels=0 if fail else 10, grid_cells=10))
    stats = {r['method']: r for r in summarize(pd.DataFrame(rows), runs, 2, 4)}
    assert stats['route_b']['RMSE'] is None
    assert not stats['route_b']['rank_eligible']
    assert stats['route_b']['cohort_coverage'] == .5
    assert stats['route_b']['failure_rate'] == .5
    assert stats['route_b']['grid_coverage'] == .5
    assert stats['fusion.idw']['rank_eligible']
    assert stats['fusion.idw']['raw_validation_coverage'] == .5


def test_no_samples_does_not_create_fake_zero_rmse():
    frame = pd.DataFrame(columns=['method','status','prediction','value'])
    assert all(r['RMSE'] is None and not r['rank_eligible'] for r in summarize(frame, [], 0, 51))
