"""Versioned public-source subsets; no target-effect calculation or platform edits."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,sys,time
from threading import Lock
import requests,fsspec
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds,Window
import xarray as xr
import planetary_computer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/basin_ecology_2013_2017'
YEARS=list(range(2013,2018))
STAC='https://planetarycomputer.microsoft.com/api/stac/v1'
NETCDF_LOCK=Lock()
SIGN_LOCK=Lock()
BBOX=gpd.read_file(OUT/'watershed.geojson').total_bounds+np.array([-.02,-.02,.02,.02])


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def request(method,url,**kwargs):
    for attempt in range(3):
        try:
            response=requests.request(method,url,timeout=90,**kwargs)
            response.raise_for_status();return response
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2*(attempt+1))


def catalog(collection,year=None):
    folder=OUT/'catalog';path=folder/(collection+('_'+str(year) if year else '')+'.json')
    if path.exists():return json.loads(path.read_text())
    body=dict(collections=[collection],bbox=BBOX.tolist(),limit=100)
    if year:body['datetime']=f'{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z'
    items=[]
    while True:
        data=request('POST',STAC+'/search',json=body).json();items+=data['features']
        link=next((x for x in data['links'] if x['rel']=='next'),None)
        if link is None:break
        body=link['body']
    save(path,items);return items


def crop(item,key,group):
    path=OUT/'raw_subsets'/group/item['id']/(key+'.tif')
    if path.exists() and path.with_suffix('.receipt.json').exists():return path
    asset=item['assets'][key]
    # The official SDK caches one SAS token per storage container. Do not
    # individually call the signing service for every raster band.
    with SIGN_LOCK:signed=planetary_computer.sign(asset['href'])
    path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.Env(GDAL_HTTP_TIMEOUT='90',GDAL_HTTP_MAX_RETRY='2',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
        with rasterio.open(signed) as source:
            bounds=transform_bounds('EPSG:4326',source.crs,*BBOX,densify_pts=21)
            w=from_bounds(*bounds,transform=source.transform)
            # Expand to native pixel edges, then intersect the real source.
            left,top=int(np.floor(w.col_off)),int(np.floor(w.row_off))
            right,bottom=int(np.ceil(w.col_off+w.width)),int(np.ceil(w.row_off+w.height))
            window=Window(left,top,right-left,bottom-top).intersection(Window(0,0,source.width,source.height))
            values=source.read(1,window=window)
            profile=source.profile.copy();profile.update(driver='GTiff',height=values.shape[0],width=values.shape[1],
                count=1,transform=source.window_transform(window),compress='deflate')
            for name in ['blockxsize','blockysize','tiled']:profile.pop(name,None)
            with rasterio.open(path,'w',**profile) as target:target.write(values,1);target.update_tags(**source.tags())
            metadata=dict(crs=str(source.crs),transform=list(source.window_transform(window)),nodata=source.nodata,
                native_shape=[source.height,source.width],subset_shape=list(values.shape),dtype=str(values.dtype))
    save(path.with_suffix('.receipt.json'),dict(source=asset['href'],item=item['id'],asset=key,
        stac_properties=item['properties'],raster_bands=asset.get('raster:bands'),subset=metadata,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),retrieved_utc=datetime.now(timezone.utc).isoformat(),
        raw_native_window=True,resampled=False))
    return path


def satellite():
    jobs=[];decisions=[]
    for year in YEARS:
        landsat=[i for i in catalog('landsat-c2-l2',year) if i['properties'].get('platform')=='landsat-8'
                 and i['properties'].get('landsat:collection_category')=='T1'
                 and 5<=int(i['properties']['datetime'][5:7])<=10]
        # Quality-only selection; no reconstruction or test scores are available.
        landsat=sorted(landsat,key=lambda i:(i['properties']['eo:cloud_cover'],i['id']))[:12]
        decisions.append(dict(year=year,landsat_selected=[i['id'] for i in landsat],selection='12 lowest scene-cloud T1 Landsat-8 May-October; retain pixel QA separately'))
        for item in landsat:
            for key in ['red','nir08','qa_pixel','qa_radsat']:jobs.append((item,key,'landsat'))
        for collection,keys in [('modis-15A2H-061',['Lai_500m','FparLai_QC']),('modis-17A3HGF-061',['Npp_500m','Npp_QC_500m'])]:
            items=[i for i in catalog(collection,year) if i['properties'].get('platform')=='terra']
            if '15A2H' in collection:
                items=[i for i in items if 5<=int(i['properties']['start_datetime'][5:7])<=10]
            for item in items:
                for key in keys:jobs.append((item,key,'lai' if '15A2H' in collection else 'npp'))
        print('CATALOG',year,len(landsat),flush=True)
    for item in catalog('nasadem'):
        keys=[k for k,v in item['assets'].items() if 'data' in v.get('roles',[]) and k not in ['tilejson']]
        key='elevation' if 'elevation' in item['assets'] else keys[0]
        jobs.append((item,key,'dem'))
    save(OUT/'satellite_selection.json',dict(decisions=decisions,bbox=BBOX.tolist(),planned_native_windows=len(jobs),effect_scores_used=False))
    failures=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={pool.submit(crop,*job):job for job in jobs}
        for index,future in enumerate(as_completed(pending),1):
            job=pending[future]
            try:future.result()
            except Exception as exc:failures.append(dict(item=job[0]['id'],asset=job[1],error=str(exc)))
            if index%25==0:print('NATIVE WINDOWS',index,'/',len(jobs),'FAIL',len(failures),flush=True)
    save(OUT/'satellite_failures.json',failures)
    if failures:raise RuntimeError(f'{len(failures)} native subset failures; retained, not silently replaced')
    print('SATELLITE SUBSETS COMPLETE',len(jobs),flush=True)


def clcd():
    record=8176941
    for year in YEARS:
        filename=f'CLCD_v01_{year}_albert_province.zip';url=f'https://zenodo.org/api/records/{record}/files/{filename}/content'
        dest=OUT/'raw_subsets/clcd'/f'anhui_{year}.tif'
        if dest.exists():continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        with fsspec.open(url,'rb',block_size=1024*1024).open() as source:
            with zipfile.ZipFile(source) as archive:
                name=next(n for n in archive.namelist() if n.endswith('_anhui.tif'));info=archive.getinfo(name)
                with archive.open(name) as handle,dest.open('wb') as target:shutil.copyfileobj(handle,target,1024*1024)
        save(dest.with_suffix('.receipt.json'),dict(url=url,record=record,member=name,member_crc32=info.CRC,
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),categorical=True,units='1',native_resolution_m=30))
        print('CLCD ANHUI',year,dest.stat().st_size,flush=True)


def climate_one(record,prefix,start,end):
    name=f'{prefix}_{start}_{end}.zip';url=f'https://zenodo.org/api/records/{record}/files/{name}/content'
    folder=OUT/'raw_subsets/climate';folder.mkdir(parents=True,exist_ok=True)
    target=folder/(name.replace('.zip','_basin.nc'))
    if target.exists():return
    temporary=folder/(name.replace('.zip','_temporary.nc'))
    archive_path=folder/name
    if not archive_path.exists():
        response=request('GET',url,stream=True);partial=archive_path.with_suffix('.zip.part');written=0
        with partial.open('wb') as dest:
            for block in response.iter_content(1024*1024):
                if block:dest.write(block);written+=len(block)
                if written and written%(64*1024*1024)<1024*1024:print('DOWNLOAD',name,written,flush=True)
        with zipfile.ZipFile(partial) as archive:assert archive.testzip() is None
        partial.replace(archive_path)
    digest=hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        member=next(n for n in archive.namelist() if n.endswith('.nc'));info=archive.getinfo(member)
        with archive.open(member) as handle,temporary.open('wb') as dest:
            while block:=handle.read(4*1024*1024):digest.update(block);dest.write(block)
    with NETCDF_LOCK:
        with xr.open_dataset(temporary) as ds:
            lat=next(x for x in ds.coords if x.lower() in {'lat','latitude'});lon=next(x for x in ds.coords if x.lower() in {'lon','longitude'})
            subset=ds.isel({lat:np.flatnonzero((ds[lat]>=BBOX[1]-.02)&(ds[lat]<=BBOX[3]+.02)),
                lon:np.flatnonzero((ds[lon]>=BBOX[0]-.02)&(ds[lon]<=BBOX[2]+.02))}).load()
            subset.to_netcdf(target,engine='h5netcdf')
            metadata=dict(dataset_attrs=dict(ds.attrs),variables={k:dict(v.attrs) for k,v in ds.data_vars.items()},dimensions=dict(ds.sizes))
    save(target.with_suffix('.receipt.json'),dict(url=url,member=member,member_crc32=info.CRC,
        original_uncompressed_sha256=digest.hexdigest(),raw_subset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        units_not_manually_converted=True,metadata=metadata))
    # Only this newly generated staging file is removed; reproducible raw subset remains.
    assert temporary.resolve().parent==(OUT/'raw_subsets/climate').resolve() and temporary.name.endswith('_temporary.nc')
    temporary.unlink()
    print('CLIMATE',name,dict(subset.sizes),flush=True)


def climate():
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(climate_one,record,prefix,start,end) for record,prefix in [(3114194,'pre'),(3185722,'tmp')]
                 for start,end in [(2012,2014),(2015,2017)]]
        for f in futures:f.result()


if __name__=='__main__':{'satellite':satellite,'clcd':clcd,'climate':climate}[sys.argv[1]]()
