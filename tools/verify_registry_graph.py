"""Bounded real-data client; all recommendation/graph/search logic is public.

Product reconstruction is an engineering check, not independent environmental
accuracy evidence. No algorithm winner or graph is supplied to the recommender.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.platform_search import (SearchInput,SearchValidation,DatasetReference,
                                               recommend_and_submit)
from ecodata.desktop import service
from ecodata.core.credentials import load


ROOT=Path('output/registry_graph_20261127')
BUDGET=dict(objective_limit=12,total_call_limit=30,proposal_limit=80,wall_seconds=40)
SEED=11


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def item(name,role,ds):
    ds=ds.copy(deep=True)
    for variable in ds.variables:ds[variable].encoding={}
    return SearchInput(name,role,bytes(ds.to_netcdf(engine='h5netcdf')))


def base(task,variable,unit):
    return dict(task=task,variable=variable,unit=unit,semantic_type='continuous',
        physical_variable='air_temperature' if variable=='temperature' else 'precipitation_amount',
        temporal_semantics='instantaneous',soft_guidance=False,nonnegative=True,
        parameters=dict(reference_support='fixed_finite_reference',require_complete_output=True),
        application_contract=dict(id='bounded_registry_integration',version='1',field=variable,unit=unit,
            semantic_type='continuous',valid_range=None,issues=[],
            evidence='Existing real product reconstruction; engineering acceptance, no independent accuracy claim'))


def labels(reference,variable):
    reference=reference.copy(deep=True);reference.attrs['split']='search_validation'
    return SearchValidation(reference,variable)


def main():
    folder=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'real-v2');folder.mkdir()
    protocol=dict(seed=SEED,budget=BUDGET,
        selection='DS receives the same full Registry contracts and visible features; no expected algorithm or graph',
        cases=['NCEP first 6 time steps, 6x6 at 40N/110W, 2x2 block means to native grid',
               'NCEP first 32 6-hour steps at 40N/110W, every other interior timestamp removed',
               'First existing WA station, first 90 days of 2018: first 45 gauge days visible; next 22 validation; last 23 confirmation'],
        claims='Engineering integration only; exposed historical data, no innovation superiority experiment',
        source_immutability=True,scope='No full Beijing rerun or enumeration')
    replay=Path(sys.argv[2]) if len(sys.argv)>2 else None
    if replay:protocol['replayed_online_proposals']=str(replay)
    save(folder/'protocol.json',protocol)
    key=load()
    if not key:raise ValueError('No locally stored DeepSeek credential')
    source=Path('data/ncep_xarray/air_temperature.nc')
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    with xr.open_dataset(source) as raw:
        raw=raw.assign_coords(lon=((raw.lon+180)%360)-180).sortby(['lat','lon'])
        iy=int(np.argmin(abs(raw.lat.values-40)));ix=int(np.argmin(abs(raw.lon.values+110)))
        region=raw.isel(lat=slice(iy-3,iy+3),lon=slice(ix-3,ix+3),time=slice(0,6)).load().rename(air='temperature')
        point=raw.isel(lat=slice(iy,iy+1),lon=slice(ix,ix+1),time=slice(0,32)).load().rename(air='temperature')
    attrs=dict(units='K',data_kind='continuous',spatial_semantics='cell_mean',temporal_semantics='instantaneous',
               physical_variable='air_temperature',physical_definition='NCEP sigma 0.995 temperature')
    for ds in (region,point):
        ds.attrs=dict(crs='EPSG:4326',source_id='NCEP-public-reanalysis')
        ds.temperature.attrs=attrs.copy()
    coarse=region.coarsen(lat=2,lon=2,boundary='exact').mean(keep_attrs=True)
    q=base('spatial','temperature','K');q.update(target_lat=region.lat.values.tolist(),target_lon=region.lon.values.tolist(),target_times=[str(t) for t in region.time.values])
    cases=[('spatial',[item('coarse.nc','primary',coarse)],q,labels(region.temperature,'temperature'),None)]
    temporal=point.copy(deep=True);hidden=np.arange(1,31,2)
    temporal.temperature.values[hidden]=np.nan
    reference=point.temperature.copy(deep=True);reference.values[:]=np.nan;reference.values[hidden]=point.temperature.values[hidden]
    q=base('temporal','temperature','K');q.update(target_times=[str(t) for t in point.time.values])
    cases.append(('temporal',[item('visible.nc','primary',temporal)],q,labels(reference,'temperature'),None))
    station=next(p for p in sorted(Path('data/wa_precipitation_2014_2024').iterdir()) if p.is_dir())
    metadata=json.loads((station/'metadata.json').read_text(encoding='utf8'))
    clock=pd.date_range('2018-01-01 01:00',periods=90)
    g=pd.read_csv(station/'gauge.csv');g['time']=pd.to_datetime(g.date)+pd.Timedelta(hours=1)
    g=g.set_index('time').reindex(clock)
    e=pd.read_csv(station/'era5_daily_9am.csv');e['time']=pd.to_datetime(e.time)
    values=e.set_index('time').reindex(clock).precipitation.to_numpy(float)
    truth=g.precipitation.where(g.valid.eq(True)).to_numpy(float)
    def rain(values,identity):
        d=xr.Dataset(dict(precipitation=(('time','lat','lon'),values[:,None,None])),
            coords=dict(time=clock,lat=[metadata['latitude']],lon=[metadata['longitude']]),
            attrs=dict(crs='EPSG:4326',source_id=identity))
        d.precipitation.attrs=dict(units='mm',data_kind='continuous',physical_variable='precipitation_amount',
            physical_definition=metadata['day_rule'],spatial_semantics='point',temporal_semantics='accumulation')
        d['time_bounds']=(('time','bounds'),np.column_stack([clock.values-np.timedelta64(1,'D'),clock.values]))
        return d
    gauge=rain(truth,'gauge:'+metadata['id']);visible=gauge.copy(deep=True);visible.precipitation.values[45:]=np.nan
    q=base('multisource','precipitation','mm');q.update(temporal_semantics='accumulation',
        target_times=[str(t) for t in clock],target_lat=gauge.lat.values.tolist(),target_lon=gauge.lon.values.tolist())
    cases.append(('multisource',[item('gauge.nc','primary',visible),item('ERA5.nc','background',rain(values,'ERA5'))],q,
        labels(gauge.precipitation.isel(time=slice(45,67)),'precipitation'),
        DatasetReference(gauge.precipitation.isel(time=slice(67,None)),'precipitation',role='independent_validation')))
    results=[]
    for name,inputs,request,validation,evaluation in cases:
        local=folder/name;local.mkdir();receipts=[];started=perf_counter()
        def provider(payload):
            if replay:
                prior=json.loads((replay/name/'recommendation_request.json').read_text(encoding='utf8'))
                canonical=lambda value:json.dumps(value,sort_keys=True,ensure_ascii=False,default=str)
                if canonical(prior)!=canonical(payload):raise ValueError('Proposal replay request fingerprint differs')
                return json.loads((replay/name/'recommendation_response.json').read_text(encoding='utf8'))
            approved=service.preview(payload)
            return service.send(approved,approved.digest,api_key=key,on_receipt=receipts.append,
                on_progress=lambda message:print(name,message,flush=True))
        try:
            result=recommend_and_submit(inputs,request,evaluation,local,provider=provider,budget=BUDGET,seed=SEED,validation=validation)
            row=dict(case=name,status=result['status'],best=result.get('best'),product=result.get('product'),
                     recommendations=result.get('recommendation_validation'),cost=result.get('total_cost'))
            if result.get('product'):
                with xr.open_dataset(result['product']) as output:
                    row['readback']=dict(dimensions=dict(output.sizes),finite=int(np.isfinite(output[request['variable']]).sum()))
        except Exception as exc:
            import traceback
            row=dict(case=name,status='FAILED',error=str(exc),traceback=traceback.format_exc())
        row['seconds']=perf_counter()-started
        save(local/'api_receipts.json',receipts);results.append(row);save(folder/'results.json',results)
        print(name,row['status'],row.get('error',''),flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original
    save(folder/'input_integrity.json',dict(ncep_sha256=original,unchanged=True))


if __name__=='__main__':main()
