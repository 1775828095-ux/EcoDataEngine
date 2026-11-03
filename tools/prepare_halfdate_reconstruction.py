"""QA-only native-detail patches and sealed half-date cohorts; no graph choices."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from scipy.signal import convolve2d
from ecodata.io.ecological_sources import read,warp,mosaic,latest
from ecodata.algorithms.numerical_search import save

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/basin_ecology_2013_2017'
OUT=ROOT/'output/halfdate_reconstruction_20261028'
PREP=OUT/'data'
ATTRS=dict(units='1',physical_variable='ndvi',physical_definition='normalized difference of near-infrared and red surface reflectance',
           spatial_semantics='cell_mean',temporal_semantics='representative_observation',semantic_type='continuous',data_kind='continuous')


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def landsat(grid,years=None):
    by_date={}
    for p in (DATA/'raw_subsets/landsat').glob('*/red.tif'):
        receipt=load(p.with_suffix('.receipt.json'));date=receipt['stac_properties']['datetime'][:10]
        if years is not None and int(date[:4]) not in years:continue
        red,profile=read(p);nir,_=read(p.with_name('nir08.tif'));qa,_=read(p.with_name('qa_pixel.tif'));sat,_=read(p.with_name('qa_radsat.tif'))
        r=red*.0000275-.2;n=nir*.0000275-.2
        valid=((qa&63)==0)&(sat==0)&(r>=0)&(n>=0)&(r+n>0)
        v=np.divide(n-r,n+r,out=np.zeros_like(r),where=r+n!=0)
        by_date.setdefault(date,[]).append(warp(np.where(valid,v,np.nan),profile,grid,method='nearest'))
    dates=sorted(by_date)
    def merge(layers):
        a=np.stack(layers);count=np.isfinite(a).sum(0)
        return np.divide(np.nansum(a,axis=0),count,out=np.full(a.shape[1:],np.nan,dtype='float32'),where=count>0)
    values=np.stack([merge(by_date[d]) for d in dates])
    return dates,values


def modis(grid,years=None):
    by_date={}
    for p in latest((DATA/'raw_subsets/modis_reflectance').glob('*/sur_refl_b01.tif')):
        receipt=load(p.with_suffix('.receipt.json'));date=receipt['stac_properties']['start_datetime'][:10]
        if years is not None and int(date[:4]) not in years:continue
        r,profile=read(p);n,_=read(p.with_name('sur_refl_b02.tif'))
        qa,_=read(p.with_name('sur_refl_qc_250m.tif'));state,_=read(p.with_name('sur_refl_state_250m.tif'))
        valid=(r>=0)&(n>=0)&(r<=10000)&(n<=10000)&(r+n>0)
        valid&=((qa&3)<=1)&(((qa>>4)&15)==0)&(((qa>>8)&15)==0)
        valid&=((state&3)==0)&(((state>>2)&1)==0)&(((state>>10)&1)==0)&(((state>>12)&1)==0)&(((state>>13)&1)==0)
        valid&=(((state>>8)&3)==0)&(((state>>15)&1)==0)
        value=np.divide(n.astype(float)-r,n.astype(float)+r,out=np.zeros_like(r,dtype=float),where=r+n!=0)
        by_date.setdefault(date,[]).append((np.where(valid,value,np.nan),profile))
    dates=sorted(by_date)
    return dates,np.stack([mosaic(by_date[d],grid,method='nearest') for d in dates])


def dataset(values,dates,lat,lon,source):
    ds=xr.Dataset(dict(ndvi=(('time','lat','lon'),values.astype('float32'))),coords=dict(time=pd.to_datetime(dates),lat=lat,lon=lon),
        attrs=dict(crs='EPSG:4326',source_id=source))
    ds.ndvi.attrs=dict(ATTRS);return ds


def hide_indices(n,kind):
    count=n//2
    if n<4:raise ValueError('Need at least four quality-qualified dates per year')
    if kind=='alternating':return np.linspace(1,n-2,count).round().astype(int).tolist()
    if kind=='contiguous':
        start=max(1,(n-count)//2);return list(range(start,start+count))
    raise ValueError(kind)


def main():
    if (OUT/'prepared.json').exists():raise ValueError('Immutable cohort already prepared')
    if not (DATA/'halfdate_acquisition.json').exists() or load(DATA/'halfdate_acquisition.json')['failures']:
        raise ValueError('Independent coarse data incomplete')
    if not (DATA/'halfdate_fine_acquisition.json').exists() or load(DATA/'halfdate_fine_acquisition.json')['failures']:
        raise ValueError('Full fine acquisition is not yet complete')
    if any(load(DATA/n).get('months')!='full_year' for n in ['halfdate_acquisition.json','halfdate_fine_acquisition.json']):
        raise ValueError('Full-year sources required after cloudy-year intake audit')
    if any(load(DATA/n).get('through_year')!=2018 for n in ['halfdate_acquisition.json','halfdate_fine_acquisition.json']):
        raise ValueError('Fresh 2018 inputs not yet complete')
    PREP.mkdir(parents=True,exist_ok=True)
    basin=gpd.read_file(DATA/'watershed.geojson');xmin,ymin,xmax,ymax=basin.total_bounds
    step=.0003
    lat=np.arange(np.floor(ymin/step)*step,ymax+step,step)
    lon=np.arange(np.floor(xmin/step)*step,xmax+step,step)
    grid=dict(shape=(len(lat),len(lon)),crs='EPSG:4326',transform=from_origin(lon[0]-step/2,lat[-1]+step/2,step,step))
    dates,values=landsat(grid,years={2013,2014});values=values[:,::-1]
    inside=geometry_mask(basin.geometry,grid['shape'],grid['transform'],invert=True)[::-1]
    # Patch placement depends on training-year data QA and geometry only.
    training=np.array([int(d[:4])<2015 for d in dates])
    count=np.isfinite(values[training]).sum(0)
    good=inside&(count>=4)
    # Require actual independent-source training coverage, not a nominal STAC
    # bounding box; unavailable sensor footprint is not a fusion experiment.
    spacing=.0024
    support_grid=dict(shape=(int(np.ceil((ymax-ymin)/spacing))+3,int(np.ceil((xmax-xmin)/spacing))+3),
        crs='EPSG:4326',transform=from_origin(xmin-spacing,ymax+spacing,spacing,spacing))
    _,coarse_training=modis(support_grid,years={2013,2014})
    support=np.isfinite(coarse_training).any(0).astype('float32')
    good&=warp(support,support_grid,grid,method='nearest')[::-1]>.5
    size=48
    starts=np.argwhere(convolve2d(good.astype(int),np.ones((size,size),int),mode='valid')>=size*size*.98)
    if not len(starts):raise ValueError('No training-QA patch of 48x48 fine pixels')
    centre=np.mean(np.argwhere(inside),axis=0)-size/2
    chosen=[starts[np.argmin(((starts-centre)**2).sum(1))]]
    for _ in range(2):
        possible=[s for s in starts if all(abs(s[0]-p[0])>=size*3 or abs(s[1]-p[1])>=size*3 for p in chosen)]
        if not possible:raise ValueError('Insufficient separated patches')
        chosen.append(max(possible,key=lambda s:min(float(((s-p)**2).sum()) for p in chosen)))
    cases=[];qa=[]
    for number,(y,x) in enumerate(chosen):
        flat=lat[y:y+size];flon=lon[x:x+size]
        fine_grid=dict(shape=(size,size),crs='EPSG:4326',transform=from_origin(flon[0]-step/2,flat[-1]+step/2,step,step))
        dates,fine_values=landsat(fine_grid)
        fine=dataset(fine_values[:,::-1],dates,flat,flon,'landsat_visible')
        # Coarse grid keeps approximately 250m support; never pre-upsample it to fine.
        delta=.0024
        clat=np.arange(float(fine.lat.min())-.01,float(fine.lat.max())+.012,delta)
        clon=np.arange(float(fine.lon.min())-.01,float(fine.lon.max())+.012,delta)
        coarse_grid=dict(shape=(len(clat),len(clon)),crs='EPSG:4326',transform=from_origin(clon[0]-delta/2,clat[-1]+delta/2,delta,delta))
        cd,cv=modis(coarse_grid);coarse=dataset(cv[:,::-1],cd,clat,clon,'modis_terra')
        coarse.attrs['observation_support']='MOD09Q1 V061 native nominal250m best observation in each8day interval; label=start, unknown within-interval day'
        fine.attrs['observation_support']='Landsat8 native30m QA-clear instantaneous acquisitions; nearest reprojection to .0003degree (~29-33m) grid'
        # Date inclusion uses QA coverage, never an error or reference-dependent method score.
        coverage=np.isfinite(fine.ndvi).mean(('lat','lon')).values
        selected={}
        for year in range(2013,2019):
            candidates=np.where((fine.time.dt.year.values==year)&(coverage>=.30))[0]
            selected[year]=candidates
        qa.append(dict(patch=number,origin=[int(y),int(x)],size=size,date_coverage=dict(zip(dates,coverage.tolist())),
                       selected_dates={str(k):[dates[i] for i in v] for k,v in selected.items()}))
        save(OUT/'quality_inventory_in_progress.json',qa)
        if any(len(selected[year])<4 for year in [2015,2016,2018]):
            raise ValueError('Insufficient clear dates; retain intake and revise QA protocol before effects: '+str(number))
        fine.to_netcdf(PREP/f'patch{number}_all_reference.nc',engine='h5netcdf')
        coarse.to_netcdf(PREP/f'patch{number}_coarse.nc',engine='h5netcdf')
        for pattern in ['alternating','contiguous']:
            case=f'patch{number}_{pattern}';folder=OUT/'cases'/case;folder.mkdir(parents=True,exist_ok=True)
            visible=[];hidden=[];phases={}
            for year in [2015,2016,2018]:
                ix=selected[year];h=ix[hide_indices(len(ix),pattern)];v=np.array([i for i in ix if i not in set(h)])
                hidden.extend(h.tolist());visible.extend(v.tolist())
                phase={2015:'development',2016:'validation',2018:'test'}[year]
                labels=fine.ndvi.isel(time=h).copy();labels.attrs=dict(ATTRS,split='test' if phase=='test' else 'search_validation')
                labels=labels.where(xr.DataArray(inside[y:y+size,x:x+size],dims=('lat','lon')))
                labels.to_dataset().to_netcdf(folder/(phase+'_labels.nc'),engine='h5netcdf')
                phases[phase]=dict(qualified=len(ix),hidden_dates=[dates[i] for i in h],visible_dates=[dates[i] for i in v],
                                   hidden_fraction=len(h)/len(ix),reference_pixels=int(np.isfinite(labels).sum()))
            visible_fine=fine.isel(time=sorted(visible));visible_fine.to_netcdf(folder/'visible_fine.nc',engine='h5netcdf')
            coarse.sel(time=coarse.time.dt.year.isin([2015,2016,2018])).to_netcdf(folder/'visible_coarse.nc',engine='h5netcdf')
            targets=fine.time.values[sorted(set(visible+hidden))]
            # No training label, annual FVC, CLCD or other Landsat derivative is passed.
            base=load(ROOT/'output/basin_ecology_2013_2017/cases/block0/config.json')['request']
            contract=dict(base['application_contract']);contract.update(id='ndvi-hidden-date',field='ndvi',valid_range=[-1.,1.],
                 evidence='NDVI ratio bounds and valid reconstruction support, no downstream model sensitivity',
                 issues=[base['application_contract']['issues'][0]])
            request=dict(task='multisource',variable='ndvi',unit='1',semantic_type='continuous',nonnegative=False,
                protect_observed_source='landsat_visible',
                target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),target_times=[str(t) for t in targets],
                parameters=dict(grid_gap_recovery=True,raster_target_times=[str(t) for t in targets],reference_support='fixed_finite_reference'),
                application_contract=contract,scoring=base['scoring'],response_scoring=dict(enabled=True))
            inputs=[dict(path=str(folder/'visible_fine.nc'),role='primary'),dict(path=str(folder/'visible_coarse.nc'),role='background')]
            save(folder/'config.json',dict(inputs=inputs,request=request));cases.append(dict(case=case,patch=number,pattern=pattern,phases=phases))
            for phase,year in [('development',2015),('validation',2016),('test',2018)]:
                files=[]
                for name,ds,role in [('fine',visible_fine,'primary'),('coarse',coarse,'background')]:
                    path=folder/f'{phase}_{name}.nc';ds.sel(time=slice(str(year),str(year))).to_netcdf(path,engine='h5netcdf')
                    files.append(dict(path=str(path),role=role))
                wanted=[str(t) for t in targets if pd.Timestamp(t).year==year]
                local=dict(request,target_times=wanted,parameters=dict(request['parameters'],raster_target_times=wanted))
                save(folder/f'config_{phase}.json',dict(inputs=files,request=local))
    save(OUT/'quality_inventory.json',qa)
    save(OUT/'prepared.json',dict(cases=cases,variable='NDVI',train=[2013,2014],development=2015,validation=2016,test=2018,
        fine_grid_step_degrees=step,coarse_grid_step_degrees=.0024,patch_selection='Training-year pixel QA and basin geometry only',
        candidate_pool='No handwritten full graphs; public registered algorithms',seeds=[11,29,47],test_scored=False,
        optional_experiment='High-change dates are a post-selection secondary stratum, never chosen to tune graphs',
        independence='Different sensors; MODIS8day composite timing and spectral response differ from Landsat'))
    print(json.dumps(load(OUT/'prepared.json'),indent=2),flush=True)


if __name__=='__main__':main()
