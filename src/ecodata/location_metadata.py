"""Recognise explicit global location metadata without inventing field support."""
import json
import math
from ecodata.core.exceptions import SchemaError


def declared_location(dataset):
    result = {'source': 'dataset_global_attributes', 'values': {}, 'attributes': {},
              'scope': 'dataset_location_not_necessarily_all_fields', 'promoted': False}
    for coordinate, aliases, limit in [('latitude', {'latitude','lat','station_latitude'},90),
                                        ('longitude', {'longitude','lon','station_longitude'},180)]:
        found=[]
        for name,value in dataset.attrs.items():
            if name.casefold() not in aliases:
                continue
            try:
                number=float(value)
            except (TypeError,ValueError):
                raise SchemaError('位置属性必须是单个有效数值：'+name)
            if not math.isfinite(number) or abs(number)>limit:
                raise SchemaError('位置属性超出有效范围：'+name)
            found.append((name,number))
        if found:
            if any(abs(number-found[0][1])>1e-9 for _,number in found):
                raise SchemaError('位置属性存在冲突，不能自动选择：'+coordinate)
            result['values'][coordinate]=found[0][1]
            result['attributes'][coordinate]=[name for name,_ in found]
    result['complete']=len(result['values'])==2
    return result


def recognise_location(dataset):
    """Promote only an explicitly declared common point for all science fields."""
    location=declared_location(dataset)
    if not location['complete']:
        return dataset
    science=[value for value in dataset.data_vars.values() if value.dtype.kind in 'iuf']
    explicit_point=bool(science) and all(value.attrs.get('spatial_semantics')=='point_observation'
        and set(value.dims).issubset({'time'}) for value in science)
    present=any(name.casefold() in {'lat','latitude','lon','longitude','x','y'} for name in dataset.variables)
    if not explicit_point or present:
        return dataset
    result=dataset.assign_coords(lat=location['values']['latitude'], lon=location['values']['longitude'])
    result['lat'].attrs.update(standard_name='latitude',units='degrees_north')
    result['lon'].attrs.update(standard_name='longitude',units='degrees_east')
    location.update(promoted=True,scope='all_science_fields_explicit_point_observation')
    result.attrs['location_identification']=json.dumps(location,ensure_ascii=False)
    return result
