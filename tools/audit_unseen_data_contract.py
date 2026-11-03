"""Independent raw flag/unit/day-boundary and final interval checks."""
from pathlib import Path
import calendar,hashlib
import numpy as np
import pandas as pd
import xarray as xr
from run_unseen_portrait_response import ROOT,OUT,DATA,STATIONS,VAR,load
from ecodata.algorithms.numerical_search import save


def main():
    checks=[]
    for sid in STATIONS:
        folder=DATA/sid;meta=load(folder/'metadata.json');receipt=load(folder/'era5_receipt.json')
        assert hashlib.sha256((folder/'gauge.dly').read_bytes()).hexdigest()==meta['gauge_sha256']
        assert hashlib.sha256((folder/'era5_hourly.json').read_bytes()).hexdigest()==meta['era5_sha256']==receipt['sha256']
        raw={}
        for line in (folder/'gauge.dly').read_text().splitlines():
            if line[17:21]!='PRCP':continue
            year,month=int(line[11:15]),int(line[15:17])
            for day in range(1,calendar.monthrange(year,month)[1]+1):
                block=line[21+8*(day-1):29+8*(day-1)];value=int(block[:5])
                raw[pd.Timestamp(year,month,day)]=(value/10 if value>=0 and block[6]==' ' and block[5] in ' BDFGT' else np.nan)
        gauge=pd.read_csv(folder/'gauge.csv',parse_dates=['date'])
        np.testing.assert_allclose([raw[d] for d in gauge.date],gauge.precipitation,equal_nan=True)
        data=load(folder/'era5_hourly.json');times=pd.DatetimeIndex(data['hourly']['time']);values=np.array(data['hourly']['precipitation'])
        daily=pd.read_csv(folder/'era5_daily_9am.csv',parse_dates=['time'])
        positions=times.get_indexer(daily.time);assert (positions>=23).all()
        independent=np.array([values[i-23:i+1].sum() for i in positions])
        assert all(times[i]-times[i-23]==pd.Timedelta(hours=23) for i in positions)
        np.testing.assert_allclose(independent,daily.precipitation,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(daily.time,gauge.date+pd.Timedelta(hours=1))
        checks.append(dict(station=sid,days=len(gauge),raw_flags_units_and_24_hour_sums='PASS'))
    predictions=0
    for entry in load(OUT/'predictions_frozen.json'):
        config=load(OUT/'cases'/entry['case']/'config.json')
        with xr.open_dataset(config['inputs'][0]['path']) as f:primary=f.load()
        with xr.open_dataset(entry['path']) as f:output=f.load()
        np.testing.assert_array_equal(output.time,primary.time)
        np.testing.assert_array_equal(output.lat,primary.lat);np.testing.assert_array_equal(output.lon,primary.lon)
        np.testing.assert_array_equal(output.time_bounds,primary.time_bounds)
        for attr in ['units','physical_variable','spatial_semantics','temporal_semantics']:
            assert output[VAR].attrs[attr]==primary[VAR].attrs[attr]
        predictions+=1
    assert predictions==108
    save(OUT/'independent_data_contract.json',dict(status='PASS',raw_checks=checks,prediction_intervals=predictions,
       independent_method='Raw fixed-width PRCP parsing; direct previous-24-hour slicing, not producer resampling helper'))
    print('INDEPENDENT DATA CONTRACT PASS',len(checks),predictions,flush=True)


if __name__=='__main__':main()
