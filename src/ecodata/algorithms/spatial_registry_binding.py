"""Bind existing station plugins to the spatial evaluator's retained-data port.

No held observations enter these contexts. Existing plugins receive ordinary
rectilinear target geometry; only the requested paired locations are returned.
"""
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.core.schemas import EcoDataBundle


def operations(registry):
    return {spec.id:spec for spec in registry.specs()
            if spec.status=='available' and spec.category=='observation_interpolation'
            and spec.requires_observations}


def execute(plugin, values, cohort, request):
    xy=cohort['allxy'];station=cohort['sxy']
    lat=np.unique(xy[:,1]);lon=np.unique(xy[:,0])
    iy=np.searchsorted(lat,xy[:,1]);ix=np.searchsorted(lon,xy[:,0])
    # Empty target cells carry no synthetic observations or background values.
    template=xr.Dataset({request['variable']:(('lat','lon'),np.full((len(lat),len(lon)),np.nan))},
        coords=dict(lat=lat,lon=lon),attrs=dict(crs='EPSG:4326'))
    template[request['variable']].attrs=dict(units=request['unit'],data_kind='continuous',spatial_semantics='point')
    result=[];reports=[]
    for row in values:
        observations=pd.DataFrame(dict(longitude=station[:,0],latitude=station[:,1],value=row,
            unit=request['unit'],split='train'))
        context=AlgorithmContext((EcoDataBundle(template),),request['variable'],'continuous',
            target_lat=tuple(lat),target_lon=tuple(lon),observations=observations,
            parameters=dict(observation_crs='EPSG:4326'),model_constraints=dict(unit=request['unit']))
        check=plugin.compatibility(context)
        if not check.eligible:raise ValueError('Actual retained station contract: '+str(check.reasons))
        product=plugin.run(context)
        output=product.data.dataset[request['variable']].transpose('lat','lon')
        if not np.array_equal(output.lat,lat) or not np.array_equal(output.lon,lon):
            raise ValueError('Station plugin changed query coordinates')
        result.append(output.values[iy,ix]);reports.append(product.report)
    return np.asarray(result),dict(algorithm=plugin.spec.id,frame_calls=len(result),reports=reports,
        observation_role='actual retained-observation parent only',reference_used=False)
