"""Safe component defaults. Scientific values are never invented by an LLM."""
from copy import deepcopy
import hashlib,io
import numpy as np
import pandas as pd
import xarray as xr


def complete_configuration(model, operation, raw, configuration):
    config=deepcopy(configuration);completed=[]
    if model!='swat_plus' or operation not in {'weather','precipitation'}:
        return config,completed
    with xr.open_dataset(io.BytesIO(raw)) as opened:ds=opened.load()
    def put(target,key,value,basis):
        if key not in target and value is not None:
            target[key]=value
            completed.append(dict(field=('station.' if target is station else '')+key,
                value=value,status='AUTO_COMPLETED',basis=basis,observed_value=False))
    station=config.setdefault('station',{})
    put(config,'project_name','uploaded_weather','Nonphysical output label only')
    put(station,'name',str(ds.attrs.get('station_id','station_'+hashlib.sha256(raw).hexdigest()[:10])),
        'Source station_id when declared, otherwise stable content-derived output label')
    for key,aliases in [('latitude',('lat','latitude')),('longitude',('lon','longitude')),('elevation',('elevation','elevation_m'))]:
        values=[float(ds[n].values.reshape(-1)[0]) for n in aliases if n in ds.coords and ds[n].size==1]
        if key in ds.attrs:values.append(float(ds.attrs[key]))
        if values and not np.allclose(values,values[0],rtol=0,atol=1e-9):
            raise ValueError('Conflicting station metadata: '+key)
        if values:
            if key in station and not np.isclose(station[key],values[0],rtol=0,atol=1e-9):
                raise ValueError('Declared station conflicts with source metadata: '+key)
            put(station,key,values[0],'Actual source coordinate/attribute; no default geographic position')
    if 'time' in ds.coords:
        times=pd.DatetimeIndex(ds.time.values)
        if len(times) and not times.hasnans:
            put(config,'start',str(times.min().date()),'Actual source time coverage')
            put(config,'end',str(times.max().date()),'Actual source time coverage')
    return config,completed
