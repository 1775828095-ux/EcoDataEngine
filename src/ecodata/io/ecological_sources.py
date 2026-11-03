"""Public, metadata-driven ingestion of the seven ecological data modalities.

This common measurement preparation is separate from reconstruction search:
class fractions and physical units are contracts, never accuracy rewards.
"""
from pathlib import Path
import json,hashlib,warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.warp import reproject,transform_bounds,Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds,Window
from rasterio.features import geometry_mask


load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def read(path,bbox=None):
    with rasterio.open(path) as f:
        if bbox is None:array=f.read(1);transform=f.transform
        else:
            window=from_bounds(*transform_bounds('EPSG:4326',f.crs,*bbox,densify_pts=21),transform=f.transform)
            left,top=int(np.floor(window.col_off)),int(np.floor(window.row_off))
            right,bottom=int(np.ceil(window.col_off+window.width)),int(np.ceil(window.row_off+window.height))
            window=Window(left,top,right-left,bottom-top).intersection(Window(0,0,f.width,f.height))
            array=f.read(1,window=window);transform=f.window_transform(window)
        return array,dict(transform=transform,crs=f.crs,nodata=f.nodata)


def warp(array,source,grid,method='average'):
    output=np.full(grid['shape'],np.nan,np.float32)
    reproject(np.asarray(array,np.float32),output,src_transform=source['transform'],src_crs=source['crs'],
        src_nodata=np.nan,dst_transform=grid['transform'],dst_crs=grid['crs'],dst_nodata=np.nan,
        resampling=getattr(Resampling,method))
    return output


def mosaic(layers,grid,method='average'):
    stack=np.stack([warp(a,p,grid,method) for a,p in layers])
    count=np.isfinite(stack).sum(0);total=np.nansum(stack,axis=0)
    return np.divide(total,count,out=np.full(total.shape,np.nan),where=count>0)


def native_tif(path,array,grid,unit):
    path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',count=1,dtype='float32',height=array.shape[0],width=array.shape[1],
        transform=grid['transform'],crs=grid['crs'],nodata=np.nan,compress='deflate') as f:
        f.write(array.astype('float32'),1);f.update_tags(units=unit)


def latest(paths):
    chosen={}
    for path in paths:
        receipt=load(path.with_suffix('.receipt.json'));p=receipt['stac_properties']
        key=(p.get('start_datetime'),p.get('modis:horizontal-tile'),p.get('modis:vertical-tile'))
        if key not in chosen or path.parent.name>chosen[key].parent.name:chosen[key]=path
    return list(chosen.values())


def prepare(directory,destination):
    """Recognize typed source products, enforce QA/units, preserve raw lineage."""
    root=Path(directory);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    if (out/'standardized.nc').exists():raise ValueError('Prepared ecological inputs are immutable; use a new destination')
    raw=root/'raw_subsets';years=range(2013,2018);basin=gpd.read_file(root/'watershed.geojson')
    bounds=basin.total_bounds+np.array([-.02,-.02,.02,.02]);records=[]
    first=next((raw/'climate').glob('pre_2012_2014_basin.nc'))
    with xr.open_dataset(first) as ds:
        lat=np.sort(ds.lat.values);lon=np.sort(ds.lon.values)
    dx=float(np.median(np.diff(lon)));dy=float(np.median(np.diff(lat)))
    grid=dict(shape=(len(lat),len(lon)),crs='EPSG:4326',transform=from_origin(lon[0]-dx/2,lat[-1]+dy/2,dx,dy))
    inside=geometry_mask(basin.geometry,grid['shape'],grid['transform'],invert=True)
    selection=load(root/'satellite_selection.json')['decisions'];ndvi={};coverage={}
    sample=next((raw/'landsat').glob('*/red.tif'));_,profile=read(sample)
    ext=from_bounds(*transform_bounds('EPSG:4326',profile['crs'],*bounds,densify_pts=21),transform=profile['transform'])
    col,row=np.floor([ext.col_off,ext.row_off]).astype(int);width,height=np.ceil([ext.col_off+ext.width-col,ext.row_off+ext.height-row]).astype(int)
    native_grid=dict(shape=(height,width),crs=profile['crs'],transform=rasterio.windows.transform(Window(col,row,width,height),profile['transform']))
    native_inside=geometry_mask(basin.to_crs(profile['crs']).geometry,native_grid['shape'],native_grid['transform'],invert=True)
    for decision in selection:
        y=decision['year'];arrays=[]
        for item in decision['landsat_selected']:
            folder=raw/'landsat'/item;red,p=read(folder/'red.tif');nir,q=read(folder/'nir08.tif')
            qa,_=read(folder/'qa_pixel.tif');saturation,_=read(folder/'qa_radsat.tif')
            rr=load(folder/'red.receipt.json')['raster_bands'][0];nr=load(folder/'nir08.receipt.json')['raster_bands'][0]
            valid=(qa&63)==0;valid&=(saturation==0)&(red!=rr['nodata'])&(nir!=nr['nodata'])
            red=red*rr['scale']+rr['offset'];nir=nir*nr['scale']+nr['offset']
            valid&=(red>=0)&(nir>=0)&((red+nir)>0)
            values=np.divide(nir-red,nir+red,out=np.full(red.shape,np.nan),where=valid)
            values[~valid]=np.nan;arrays.append(warp(values,p,native_grid,'nearest'))
        stack=np.stack(arrays);count=np.isfinite(stack).sum(0)
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);ndvi[y]=np.nanmedian(stack,axis=0)
        ndvi[y][count<2]=np.nan;coverage[y]=count
        print('PLATFORM INTAKE NDVI',y,'valid',float(np.isfinite(ndvi[y][native_inside]).mean()),flush=True)
    training=np.concatenate([ndvi[y][native_inside&np.isfinite(ndvi[y])] for y in (2013,2014)])
    soil,vegetation=np.quantile(training,[.05,.95]);assert vegetation>soil
    fvc=[];fvc_coverage=[]
    for y in years:
        values=np.clip((ndvi[y]-soil)/(vegetation-soil),0,1)
        native_tif(out/'native_30m'/f'fvc_proxy_{y}.tif',values,native_grid,'1')
        fraction=warp(np.isfinite(values).astype('float32'),native_grid,grid)
        result=warp(values,native_grid,grid);result[fraction<.8]=np.nan
        fvc.append(result);fvc_coverage.append(fraction)
    records.append(dict(modality='FVC',source='Landsat-8 Collection 2 L2 T1 red/nir08',native_resolution='30m',
        method='QA bits0-5 clear and no radiometric saturation; >=2 visible observations; May-Oct NDVI median; dimidiate proxy',
        endmembers=dict(soil=float(soil),vegetation=float(vegetation),fit_years=[2013,2014],quantiles=[.05,.95]),
        output_unit='1',aggregation='GDAL area-weighted average of valid native pixels; >=80% native support',
        ground_truth_status='Derived Landsat FVC proxy, not field-validated true vegetation cover'))
    dem_layers=[]
    for path in (raw/'dem').glob('*/elevation.tif'):
        a,p=read(path);a=a.astype('float32');a[a==p['nodata']]=np.nan;dem_layers.append((a,p))
    dem=mosaic(dem_layers,native_grid,'bilinear')
    gy,gx=np.gradient(dem,30.,30.);slope=np.degrees(np.arctan(np.hypot(gx,gy)))
    native_tif(out/'native_30m/dem.tif',dem,native_grid,'m');native_tif(out/'native_30m/slope.tif',slope,native_grid,'degree')
    dem_grid=warp(dem,native_grid,grid);slope_grid=warp(slope,native_grid,grid)
    records.append(dict(modality='DEM/slope',source='NASADEM, SRTM-derived',native_resolution='1 arc-second (~30m)',
        unit=['m','degree'],method='Elevation on native 30m metric grid; slope before area aggregation; source information not sharpened'))
    data=dict(fvc=np.stack(fvc),fvc_native_support=np.stack(fvc_coverage),dem=np.repeat(dem_grid[None],5,0),slope=np.repeat(slope_grid[None],5,0))
    units=dict(fvc='1',fvc_native_support='1',dem='m',slope='degree')
    classes=range(1,10)
    for category in classes:data['land_fraction_'+str(category)]=[];units['land_fraction_'+str(category)]='1'
    for y in years:
        a,p=read(raw/'clcd'/f'anhui_{y}.tif',bounds);valid=a!=p['nodata']
        for category in classes:
            indicator=np.where(valid,(a==category).astype('float32'),np.nan)
            data['land_fraction_'+str(category)].append(warp(indicator,p,grid))
    for category in classes:data['land_fraction_'+str(category)]=np.stack(data['land_fraction_'+str(category)])
    total=sum(data['land_fraction_'+str(c)] for c in classes)
    np.testing.assert_allclose(total[np.isfinite(total)],1.,atol=2e-6)
    records.append(dict(modality='land cover',source='CLCD v1.0.2 Anhui annual maps',native_resolution='30m',unit='1',
        method='Native class indicators aggregated to area fractions; class codes never averaged',classes=list(classes)))
    lai=[];npp=[];npp_qc=[]
    for y in years:
        paths=[p for p in latest((raw/'lai').glob('*/Lai_500m.tif')) if int(load(p.with_suffix('.receipt.json'))['stac_properties']['start_datetime'][:4])==y]
        by_date={}
        for path in paths:
            receipt=load(path.with_suffix('.receipt.json'));date=receipt['stac_properties']['start_datetime']
            a,p=read(path);qa,_=read(path.with_name('FparLai_QC.tif'))
            valid=(a<=100)&(((qa>>5)&7)<=1)&(((qa>>3)&3)==0)
            value=np.where(valid,a*receipt['raster_bands'][0]['scale'],np.nan)
            by_date.setdefault(date,[]).append((value,p))
        samples=np.stack([mosaic(layers,grid) for layers in by_date.values()]);count=np.isfinite(samples).sum(0)
        mean=np.divide(np.nansum(samples,axis=0),count,out=np.full(grid['shape'],np.nan),where=count>0)
        lai.append(mean)
        paths=[p for p in latest((raw/'npp').glob('*/Npp_500m.tif')) if int(load(p.with_suffix('.receipt.json'))['stac_properties']['start_datetime'][:4])==y]
        layers=[];quality=[]
        for path in paths:
            receipt=load(path.with_suffix('.receipt.json'));a,p=read(path);qc,_=read(path.with_name('Npp_QC_500m.tif'))
            valid=(a>=-30000)&(a<=32700)&(qc<=100)
            layers.append((np.where(valid,a*receipt['raster_bands'][0]['scale'],np.nan),p))
            quality.append((np.where(valid,qc,np.nan),p))
        npp.append(mosaic(layers,grid));npp_qc.append(mosaic(quality,grid))
        print('PLATFORM INTAKE MODIS',y,'LAI dates',len(by_date),'NPP native tiles',len(paths),flush=True)
    data.update(lai=np.stack(lai),npp=np.stack(npp),npp_gapfill_percent=np.stack(npp_qc));units.update(lai='m2/m2',npp='kg C m-2 yr-1',npp_gapfill_percent='%')
    records+= [dict(modality='LAI',source='Terra MOD15A2H V061, 500m',unit='m2/m2',method='SCF_QC 0/1 and clear cloud state; May-Oct mean of valid 8-day observations'),
        dict(modality='NPP',source='Terra MOD17A3HGF V061, 500m',unit='kg C m-2 yr-1',method='Annual density mean; fill codes excluded; gap-fill percentage retained, not misread as bit flags',
             dependency='NPP product uses MODIS LAI/FPAR; not independent of LAI')]
    for prefix,var,unit in [('pre','precipitation','mm/May-Oct'),('tmp','air_temperature','degree_Celsius')]:
        chunks=[]
        for start,end in [(2012,2014),(2015,2017)]:
            with xr.open_dataset(raw/'climate'/f'{prefix}_{start}_{end}_basin.nc') as ds:
                assert np.array_equal(ds.time.values,np.arange(1,37))
                a=ds[prefix].assign_coords(time=pd.date_range(f'{start}-01-01',periods=36,freq='MS')).sortby('lat').sortby('lon').load()
                expected='0.1 mm' if prefix=='pre' else '0.1 degree centigrade';assert a.attrs['unit']==expected
                chunks.append(a/10.)
        all_years=xr.concat(chunks,dim='time');values=[]
        for y in years:
            a=all_years.sel(time=slice(f'{y}-05-01',f'{y}-10-31'))
            value=a.sum('time',skipna=False) if prefix=='pre' else a.weighted(a.time.dt.days_in_month).mean('time',skipna=False)
            assert np.array_equal(value.lat,lat) and np.array_equal(value.lon,lon)
            values.append(value.values[::-1])
        data[var]=np.stack(values);units[var]=unit
        records.append(dict(modality=var,source='Peng et al. China monthly 1km v2019',unit=unit,
            native_unit=expected,conversion='multiply by 0.1, from source unit attribute',method='May-Oct sum' if prefix=='pre' else 'May-Oct day-weighted monthly mean',
            time_index='Source indices1..36 assigned by immutable three-year archive dates; not inferred from numeric epoch'))
    coords=dict(time=pd.to_datetime([f'{y}-07-01' for y in years]),lat=lat,lon=lon)
    standardized=xr.Dataset({name:(('time','lat','lon'),np.asarray(values,dtype='float32')[:,::-1]) for name,values in data.items()},coords=coords,
        attrs=dict(crs='EPSG:4326',source_id='basin-ecological-inputs',nominal_grid='Native climate 1/120-degree (~1km); grid spacing is not a new information resolution',
          basin_outlet_hybas_id=4120774820,basin_area_km2=1056.6,season='May-October; NPP is full-year ancillary context, not a forecast'))
    for name in standardized:
        standardized[name].attrs=dict(units=units[name],data_kind='continuous',spatial_semantics='cell_mean',
            feature_role='target' if name=='fvc' else 'quality' if name=='fvc_native_support' else 'covariate')
    standardized.fvc.attrs.update(physical_variable='fvc',physical_definition='Landsat-derived training-endmember dimidiate FVC proxy, May-Oct median NDVI',
         temporal_semantics='mean',valid_min=0.,valid_max=1.)
    standardized['basin_mask']=(('lat','lon'),inside[::-1].astype('uint8'));standardized.basin_mask.attrs.update(units='1',data_kind='categorical',feature_role='mask')
    standardized.to_netcdf(out/'standardized.nc',engine='h5netcdf')
    common=np.isfinite(standardized.fvc).all('time')&standardized.basin_mask.astype(bool)
    qa=dict(years=list(years),basin_cells=int(inside.sum()),common_finite_FVC_cells=int(common.sum()),
        fields={name:dict(units=standardized[name].attrs.get('units'),finite_fraction=float(np.isfinite(standardized[name]).mean()),
          minimum=float(standardized[name].min()),maximum=float(standardized[name].max())) for name in standardized},
        data_effect_not_evaluated=True,preparation_records=records)
    (out/'quality_and_units.json').write_text(json.dumps(qa,indent=2,default=str),encoding='utf-8')
    (out/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file() and not p.name.endswith(('.zip','.part','_temporary.nc'))},indent=2),encoding='utf-8')
    print('PLATFORM PREPARATION COMPLETE',qa['basin_cells'],qa['common_finite_FVC_cells'],flush=True)
    return out/'standardized.nc'
