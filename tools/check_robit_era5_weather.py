"""Read back all exported weather columns against frozen products/originals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from run_robit_era5_fusion import OUT,raw_weather,VAR


def main():
    raw=raw_weather();raw=raw[raw.time.dt.year>=2008].copy();checks=[]
    for item in json.loads((OUT/'weather_handoff/index.json').read_text()):
        folder=Path(item['folder'])
        with xr.open_dataset(OUT/'predictions'/item['case']/(item['method']+'.nc')) as f:
            rain=f[VAR].sel(time=pd.DatetimeIndex(raw.time)).values[:,0,0]
        fields=[('station.pcp',[rain]),('station.tem',[raw.air_temperature_max,raw.air_temperature_min]),
            ('station.slr',[raw.solar_radiation]),('station.hmd',[raw.relative_humidity]),('station.wnd',[raw.wind_speed])]
        for name,values in fields:
            frame=pd.read_csv(folder/name,sep=r'\s+',skiprows=3,header=None)
            np.testing.assert_array_equal(frame[0],raw.time.dt.year)
            np.testing.assert_array_equal(frame[1],raw.time.dt.dayofyear)
            for i,value in enumerate(values,2):np.testing.assert_allclose(frame[i],value,rtol=0,atol=.00051)
        assert not (folder/'file.cio').exists()
        checks.append(dict(case=item['case'],method=item['method'],days=len(raw),five_weather_types='PASS',
            observed_other_variables_unchanged=True,no_model_control_overwrite=True))
    (OUT/'weather_readback.json').write_text(json.dumps(dict(status='PASS',packages=len(checks),checks=checks),indent=2),encoding='utf-8')
    print('ALL WEATHER COLUMNS VERIFIED',len(checks))


if __name__=='__main__':main()
