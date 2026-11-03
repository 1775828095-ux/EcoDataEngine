"""Select only real 2013 development feature extremes, never audit outcomes."""
import collections
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.algorithms.raster_reconstruction_task import degradation
import run_spatial_next as r


def acquire():
    target=r.OUT/'feature_extreme_acquisition.json'
    if target.exists():return r.read(target)
    old={t['id'] for t in r.prepare(True)};chosen=collections.defaultdict(list)
    with xr.open_dataset(r.SOURCE) as raw:
        source=raw.sel(time=slice('2013-01-01','2013-12-31T23:59')).load()
        dates=pd.DatetimeIndex(source.time.values)
        for region,start in [('west',0),('east',29)]:
            fine=source.isel(lat=slice(0,24),lon=slice(start,start+24)).sortby('lat')
            sample=fine.isel(time=0,drop=True)[['air']].reset_coords(drop=True)
            for scale in [2,3]:
                coarse,operator=degradation(sample,24//scale,'air')
                arrays=(operator@fine.air.values.reshape(len(dates),-1).T).T.reshape(len(dates),24//scale,24//scale)
                rows=[]
                for array in arrays:
                    ds=xr.Dataset({'air':(('lat','lon'),array)},coords={'lat':coarse.lat,'lon':coarse.lon})
                    features=SemanticFeatureExtractor.raster_features(ds,'air',24)
                    spread=float(np.ptp(array));diff=np.r_[abs(np.diff(array,axis=0)).ravel(),abs(np.diff(array,axis=1)).ravel()]
                    features.update(minimum=float(array.min()),maximum=float(array.max()),variation=float(diff.mean()/spread) if spread else 0.)
                    rows.append(features)
                table=pd.DataFrame(rows)
                for feature in table:
                    for index in [table[feature].idxmin(),table[feature].idxmax()]:
                        stamp=dates[index].strftime('%Y%m%dT%H');token=f'{stamp}_{region}_x{scale}'
                        if token not in old:chosen[(stamp,region,scale)].append(feature)
    result=dict(scope='2013 source-only feature coverage; no reconstruction labels or 2014 input/reference inspected',
        candidates_scanned_per_context=len(dates),new_tasks=[dict(timestamp=k[0],region=k[1],scale=k[2],features=sorted(set(v))) for k,v in chosen.items()],
        support_gate_changed=False)
    r.save(target,result);return result


if __name__=='__main__':
    data=acquire();extra={(x['timestamp'],x['region'],x['scale']) for x in data['new_tasks']}
    tasks=r.prepare(True,extra,'manifest_extremes.json')
    print('Acquired tasks',len(extra),'manifest tasks',len(tasks),flush=True)
    for task in tasks:
        if task['stage']=='fit':r.matrix(task)
