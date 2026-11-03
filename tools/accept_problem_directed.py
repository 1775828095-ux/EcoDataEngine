"""Bounded real-input clients of the public service; no selection policy here."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import xarray as xr
from ecodata.algorithms.platform_search import SearchInput,DatasetReference,EngineeringMetric,submit_search
from ecodata.algorithms.raster_reconstruction_task import degradation
from ecodata.algorithms.numerical_search import save
from ecodata.interface.search_page import process_search_upload

OUT=ROOT/'output/problem_directed_20261021'
OLD=ROOT/'output/platform_integration_20261021/inputs'
BUDGET=dict(objective_limit=8,proposal_limit=64,callback_limit=150,wall_seconds=30)

def run(task,client_name):
    folder=OUT/'inputs';folder.mkdir(exist_ok=True,parents=True)
    if task=='spatial':
        source=ROOT/'output/lghap_reconstruction_20261020_v3/tiles/train/BTH_20190101.nc'
        with xr.open_dataset(source,engine='h5netcdf') as ds:fine=ds.isel(lat=slice(0,16),lon=slice(0,16)).load()
        coarse,operator=degradation(fine,8)
        coarse.aod.attrs.update(units='1',data_kind='continuous',spatial_semantics='cell_mean')
        coarse.attrs['crs']='EPSG:4326'
        path=folder/'spatial_area_mean.nc'
        if not path.exists():coarse.to_netcdf(path,engine='h5netcdf')
        request=dict(task='spatial',variable='aod',unit='1',semantic_type='continuous',spatial_semantics='cell_mean',nonnegative=True,
            target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),requirements=dict(try_coarse_consistency=True))
        inputs=[SearchInput(path.name,'primary',path.read_bytes())]
        evaluation=EngineeringMetric('aod','coarse_consistency',operator=operator,coarse=coarse.aod.values.ravel())
    elif task=='temporal':
        request=json.loads((OLD/'temporal_request.json').read_text(encoding='utf-8'))
        inputs=[SearchInput('temporal.nc','primary',(OLD/'temporal.nc').read_bytes())]
        evaluation=EngineeringMetric('tp')
    else:
        request=json.loads((OLD/'multisource_request.json').read_text(encoding='utf-8'))
        request['working_assumptions']=['Author datum unverified; frozen EPSG:4326 working convention','Taipei time working convention; retrospective validation']
        inputs=[SearchInput(name+'.csv',name,(OLD/(name+'.csv')).read_bytes()) for name in ['observations','query','covariates']]
        with xr.open_dataset(OLD/'validation_reference.nc',engine='h5netcdf') as ds:evaluation=DatasetReference(ds.pm25.load(),'pm25')
    destination=OUT/'runs'/(task+'_'+client_name+'_bounded_v4')
    if (destination/'summary.json').exists():raise ValueError('Preserve existing run; do not overwrite evidence')
    save(folder/(task+'_request.json'),request)
    save(destination/'protocol.json',dict(seed=11,budget=BUDGET,mode='C-directed',client=client_name,request=request,
        inputs=[dict(role=x.role,sha256=hashlib.sha256(x.content).hexdigest()) for x in inputs],evaluation=evaluation.protocol,
        scope='Bounded engineering acceptance, not comparative efficacy research'))
    client=process_search_upload if client_name=='page' else submit_search
    result=client(inputs,request,evaluation,destination,budget=dict(BUDGET),seed=11,mode='C-directed')
    if not result.get('best'):raise RuntimeError('No quality-passing real graph; inspect preserved evidence')
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['spatial','temporal','multisource']);parser.add_argument('--client',choices=['toolkit','page'],default='toolkit')
    args=parser.parse_args();run(args.task,args.client)
