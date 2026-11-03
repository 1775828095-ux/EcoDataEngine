"""Fixed bounded real-data public clients; no candidate policy or final graph."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,xarray as xr
from ecodata.algorithms.platform_search import SearchInput,EngineeringMetric,DatasetReference,submit_search
from ecodata.algorithms.numerical_search import save
from ecodata.algorithms.raster_reconstruction_task import degradation
from ecodata.interface.search_page import process_search_upload
OUT=ROOT/'output/local_problem_20261021'
OLD=ROOT/'output/platform_integration_20261021/inputs'
BUDGET=dict(objective_limit=5,proposal_limit=48,callback_limit=100,wall_seconds=20)

def inputs_for(case):
    if case in {'aligned','mismatch'}:
        source=ROOT/'output/lghap_reconstruction_20261020_v3/tiles/train/BTH_20190101.nc'
        with xr.open_dataset(source,engine='h5netcdf') as ds:fine=ds.isel(lat=slice(0,16),lon=slice(0,16)).load()
        coarse,_=degradation(fine,8)
        inputs=[]
        for role,field in [('primary',fine if case=='mismatch' else coarse),('background',coarse)]:
            ds=field.copy(deep=True);ds.attrs['crs']='EPSG:4326'
            ds.aod.attrs.update(units='1',data_kind='continuous',spatial_semantics='cell_mean',physical_variable='aod',physical_definition='total_column_AOD')
            ds['uncertainty']=xr.ones_like(ds.aod);ds.uncertainty.attrs.update(units='1',meaning='explicit equal constant engineering prior; not measured sensor uncertainty')
            path=OUT/'inputs'/f'{case}_{role}.nc';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():ds.to_netcdf(path,engine='h5netcdf')
            inputs.append(SearchInput(path.name,role,path.read_bytes()))
        request=dict(task='multisource',variable='aod',unit='1',semantic_type='continuous',nonnegative=True,target_lat=coarse.lat.values.tolist(),target_lon=coarse.lon.values.tolist(),
            working_assumptions=['Two resolutions of SAME LGHAP product, not independent observations','Equal constant engineering weights, no scientific uncertainty claim'])
        evaluation=EngineeringMetric('aod')
    elif case=='spatial':
        source=ROOT/'output/problem_directed_20261021/inputs'
        request=json.loads((source/'spatial_request.json').read_text(encoding='utf-8'))
        inputs=[SearchInput('spatial_area_mean.nc','primary',(source/'spatial_area_mean.nc').read_bytes())]
        evaluation=EngineeringMetric('aod')
    elif case=='temporal':
        request=json.loads((OLD/'temporal_request.json').read_text(encoding='utf-8'))
        inputs=[SearchInput('temporal.nc','primary',(OLD/'temporal.nc').read_bytes())];evaluation=EngineeringMetric('tp')
    else:
        request=json.loads((OLD/'multisource_request.json').read_text(encoding='utf-8'))
        request['working_assumptions']=['Frozen Taiwan working time and coordinate convention; not independently confirmed']
        inputs=[SearchInput(name+'.csv',name,(OLD/(name+'.csv')).read_bytes()) for name in ['observations','query','covariates']]
        with xr.open_dataset(OLD/'validation_reference.nc',engine='h5netcdf') as ds:evaluation=DatasetReference(ds.pm25.load(),'pm25')
    return inputs,request,evaluation

def run(case,client,version='v5'):
    inputs,request,evaluation=inputs_for(case)
    if client=='page':
        # Cosmetic names and input order differ deliberately; source roles and
        # data bytes do not. Scientific decisions must remain identical.
        inputs=[SearchInput('renamed_'+x.filename,x.role,x.content) for x in reversed(inputs)]
    folder=OUT/'runs'/(case+'_'+client+'_'+version)
    if folder.exists():raise ValueError('Preserve saved acceptance; choose explicit new version before rerun')
    save(folder/'protocol.json',dict(case=case,client=client,budget=BUDGET,seed=11,mode='C-local',request=request,evaluation=evaluation.protocol,
        inputs=[dict(role=x.role,sha256=hashlib.sha256(x.content).hexdigest()) for x in inputs],scope='Engineering acceptance only, no efficacy or independent product accuracy claim'))
    function=process_search_upload if client=='page' else submit_search
    result=function(inputs,request,evaluation,folder,budget=dict(BUDGET),seed=11,mode='C-local')
    if not result.get('best'):raise RuntimeError('No passing graph; retained actual failure in '+str(folder))
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['aligned','mismatch','spatial','temporal','multisource']);parser.add_argument('--client',default='toolkit',choices=['toolkit','page']);parser.add_argument('--version',default='v5')
    args=parser.parse_args();run(args.case,args.client,args.version)
