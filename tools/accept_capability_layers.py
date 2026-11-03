"""Fixed input-only matrix. Both layers call the public service with identical data.

Declared layer confirms independent, observable facts before the SAME mapper;
it may not override the file. This tests consumers, not human metadata inference.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,xarray as xr,pandas as pd
from ecodata.algorithms.platform_search import SearchInput,EngineeringMetric,submit_search
from ecodata.algorithms.numerical_search import save
OUT=ROOT/'output/capability_matching_20261021'
BUDGET=dict(objective_limit=4,proposal_limit=40,callback_limit=90,wall_seconds=15)


def cases():
    result=[]
    old=ROOT/'output/local_problem_20261021/inputs'
    with xr.open_dataset(old/'aligned_primary.nc',engine='h5netcdf') as ds:target=ds.load()
    for case in ['aligned','mismatch']:
        inputs=[SearchInput(f'{case}_{role}.nc',role,(old/f'{case}_{role}.nc').read_bytes()) for role in ['primary','background']]
        req=dict(task='multisource',variable='aod',unit='1',semantic_type='continuous',nonnegative=True,target_lat=target.lat.values.tolist(),target_lon=target.lon.values.tolist(),
            working_assumptions=['Derived resolutions of one LGHAP field; equal engineering weights, not independent sensors'])
        facts=[dict(source_id='role:'+role+':aod',feature='grid_change',value=(role=='primary' and case=='mismatch'),unit='1',status='CONFIRMED',confirmed=True,evidence='fixed coordinate comparison before graph search') for role in ['primary','background']]
        result.append((case,inputs,req,facts,'real LGHAP product-derived functional test'))
    old=ROOT/'output/platform_integration_20261021/inputs'
    for case in ['temporal','temporal_instant']:
        name=case+'.nc'
        with xr.open_dataset(old/name,engine='h5netcdf') as ds:
            variable='tp' if case=='temporal' else 't2m';unit=ds[variable].attrs['units'];semantics=ds[variable].attrs['temporal_semantics']
        if case=='temporal':req=json.loads((old/'temporal_request.json').read_text(encoding='utf-8'))
        else:req=dict(task='temporal',variable=variable,unit=unit,semantic_type='continuous',temporal_semantics=semantics,parameters=dict(target_time_frequency='6h'))
        facts=[dict(source_id='role:primary:'+variable,feature='temporal_semantics',value=semantics,unit='1',status='CONFIRMED',confirmed=True,evidence='actual ERA5 variable metadata and preserved interval contract')]
        result.append((case,[SearchInput(name,'primary',(old/name).read_bytes())],req,facts,'real ERA5 existing small input'))
    # Synthetic category is a boundary test, never environmental validation.
    ds=xr.Dataset({'land':(('lat','lon'),np.arange(16).reshape(4,4)%3)},coords={'lat':np.arange(4.)+30,'lon':np.arange(4.)+110},attrs={'crs':'EPSG:4326'})
    ds.land.attrs.update(units='1',data_kind='categorical',spatial_semantics='cell_category')
    path=OUT/'inputs/category_functional.nc'
    if not path.exists():ds.to_netcdf(path,engine='h5netcdf')
    req=dict(task='spatial',variable='land',unit='1',semantic_type='categorical',nonnegative=True,target_lat=[30.5,32.5],target_lon=[110.5,112.5])
    facts=[dict(source_id='role:primary:land',feature='variable_class',value='categorical',unit='1',status='CONFIRMED',confirmed=True,evidence='explicit synthetic category fixture')]
    result.append(('category',[SearchInput(path.name,'primary',path.read_bytes())],req,facts,'SYNTHETIC boundary only'))
    times=pd.date_range('2020-01-01 01:00',periods=24,freq='h').values
    mean=xr.Dataset({'temperature':('time',280.+np.sin(np.arange(24)*np.pi/12)),
        'time_bounds':(('time','bounds'),np.column_stack((times-np.timedelta64(1,'h'),times)))},coords={'time':times},
        attrs={'provenance':'SYNTHETIC interval means with known 1-hour support; not ERA5 or instrument observations'})
    mean.time.attrs['bounds']='time_bounds';mean.temperature.attrs.update(units='K',data_kind='continuous',temporal_semantics='interval_mean')
    path=OUT/'inputs/interval_mean_functional.nc'
    if not path.exists():mean.to_netcdf(path,engine='h5netcdf')
    req=dict(task='temporal',variable='temperature',unit='K',semantic_type='continuous',temporal_semantics='interval_mean',parameters=dict(target_time_frequency='6h'))
    facts=[dict(source_id='role:primary:temperature',feature='temporal_semantics',value='interval_mean',unit='1',status='CONFIRMED',confirmed=True,evidence='fixed synthetic interval-mean generator definition')]
    result.append(('interval_mean',[SearchInput(path.name,'primary',path.read_bytes())],req,facts,'SYNTHETIC boundary only'))
    return result


def run(only=None,version=''):
    items=[row for row in cases() if only is None or row[0]==only]
    suffix=('_'+only if only else '')+('_'+version if version else '')
    save(OUT/('layer_protocol'+suffix+'.json'),dict(budget=BUDGET,seed=11,cases=[dict(case=c,request=r,declarations=d,scope=s) for c,i,r,d,s in items],
        comparison='same actual input/semantics; declared facts confirmed and consumed vs extracted facts; never algorithm answers'))
    rows=[]
    for case,inputs,request,facts,scope in items:
        for layer in ['declared','recognized']:
            req=dict(request)
            if layer=='declared':req['feature_declarations']=facts
            folder=OUT/'layers'/(case+'_'+layer+('_'+version if version else ''))
            try:
                result=submit_search(inputs,req,EngineeringMetric(req['variable']),folder,budget=dict(BUDGET),mode='C-local',seed=11)
                rows.append(dict(case=case,layer=layer,scope=scope,passed=bool(result['best']),counts=result['counts'],objective=result['best']['objective'] if result['best'] else None,folder=str(folder)))
            except Exception as exc:rows.append(dict(case=case,layer=layer,scope=scope,passed=False,error=str(exc),folder=str(folder)))
    save(OUT/('layer_runs'+suffix+'.json'),rows)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--only');p.add_argument('--version',default='');a=p.parse_args();run(a.only,a.version)
