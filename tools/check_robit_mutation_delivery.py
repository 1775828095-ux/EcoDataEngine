"""Independent NumPy test-metric and daily-support readback for all 75 files."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
from run_robit_era5_mutation import OUT,OLD,load,freeze
from ecodata.algorithms.numerical_search import save


def main():
    freeze();metrics={(r['case'],r['method']):r for r in load(OUT/'test_metrics.json')};checked=[]
    for item in load(OUT/'predictions_frozen.json'):
        path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        with xr.open_dataset(path) as f:ds=f.load()
        with xr.open_dataset(OLD/'cases'/item['case']/'test_labels.nc') as f:ref=f.precipitation.load()
        with xr.open_dataset(OLD/'cases'/item['case']/'station_masked.nc') as f:primary=f.precipitation.load()
        assert np.isnan(primary.sel(time=ref.time)).all()
        np.testing.assert_array_equal(ds.time,primary.time)
        np.testing.assert_array_equal(ds.time_bounds.values[:,0],ds.time.values)
        assert np.all(ds.time_bounds.values[:,1]-ds.time_bounds.values[:,0]==np.timedelta64(1,'D'))
        visible=np.isfinite(primary.values)
        np.testing.assert_array_equal(ds.precipitation.values[visible],primary.values[visible])
        values=ds.precipitation.sel(time=ref.time).values.ravel();truth=ref.values.ravel()
        years=pd.DatetimeIndex(ref.time.values).year
        annual=[float(np.sqrt(np.mean((values[years==y]-truth[years==y])**2))) for y in np.unique(years)]
        expected=metrics[item['case'],item['method']]
        np.testing.assert_allclose(np.mean(annual),expected['objective'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean((values-truth)**2)),expected['RMSE'],rtol=1e-12,atol=1e-12)
        checked.append(dict(case=item['case'],method=item['method'],hidden_days=len(values),independent_objective=float(np.mean(annual))))
    save(OUT/'delivery_numerical_readback.json',dict(status='PASS',files=len(checked),test_metric_independently_recomputed=True,
         daily_interval_support_checked=True,visible_values_preserved=True,rows=checked))
    print('DELIVERY NUMERICAL READBACK PASS',len(checked))


if __name__=='__main__':main()
