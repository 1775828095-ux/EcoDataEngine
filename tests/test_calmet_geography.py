import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from ecodata.models.adapters.calmet_geography import export_geography
from ecodata.core.exceptions import SchemaError


def case(alternate=False):
    nx,ny=(3,2) if not alternate else (2,3)
    values={'landuse':(np.arange(nx*ny)%2+10).reshape(ny,nx),'elevation':np.arange(nx*ny).reshape(ny,nx)+100.,
            'roughness':np.full((ny,nx),.2),'albedo':np.full((ny,nx),.15),'bowen':np.full((ny,nx),.5),
            'soil_heat_flux':np.full((ny,nx),.25),'anthropogenic_heat':np.full((ny,nx),10.),'leaf_area_index':np.full((ny,nx),2.)}
    units=['1','m','m','1','1','1','W/m^2','1']
    ds=xr.Dataset({k:(('y','x'),v.copy(),{'units':u}) for (k,v),u in zip(values.items(),units)},coords={'x':500500+np.arange(nx)*1000,'y':1000500+np.arange(ny)*1000})
    ds.x.attrs['units']='m';ds.y.attrs['units']='m'
    cfg=dict(mode='geography',target_version='CALMET-5.8.5-GEO.DAT-2.0',coordinate_location='cell_center',elevation_vertical_reference='MSL',
        source={'kind':'synthetic','description':'Synthetic eight-parameter UTM grid, not a real geographical case'},missing_values=[-9999],
        crs='EPSG:32631',coordinates=dict(x='x',y='y'),fields={k:dict(variable=k) for k in values},landuse_categories=[10,11],water_category_range=[50,59])
    if alternate:
        ds=ds.assign_coords(x=ds.x/1000,y=ds.y/1000);ds.x.attrs['units']='km';ds.y.attrs['units']='km'
        ds['elevation']=ds.elevation/1000;ds.elevation.attrs['units']='km'
        ds['albedo']=ds.albedo*100;ds.albedo.attrs['units']='%'
        ds=ds.isel(x=slice(None,None,-1)).rename({'x':'east','y':'north','elevation':'height'}).transpose('east','north')
        cfg['coordinates']=dict(x='east',y='north');cfg['fields']['elevation']['variable']='height';cfg['crs']='EPSG:32755'
    return ds,cfg,values


def parse(raw):
    """Separate stdlib reader following official READGE field layout."""
    lines=iter(raw.decode('ascii').splitlines());header=next(lines)
    assert header[:16].strip()=='GEO.DAT' and header[16:32].strip()=='2.0'
    for _ in range(int(next(lines))):next(lines)
    assert next(lines).strip()=='UTM';zone=next(lines)
    assert next(lines)[:8].strip()=='WGS-84'
    grid=next(lines);nx,ny=int(grid[:8]),int(grid[8:16]);geometry=[float(grid[i:i+12]) for i in range(16,64,12)]
    assert next(lines).strip()=='KM' and next(lines)=='1'
    nlu,lo,hi=map(int,next(lines).split());cats=list(map(int,next(lines).split()));assert nlu==len(cats)
    def arr():return np.array([[float(v) for v in next(lines).split()] for _ in range(ny)])
    data={'landuse':arr()};assert float(next(lines))==1;data['elevation']=arr()
    for name in ('roughness','albedo','bowen','soil_heat_flux','anthropogenic_heat','leaf_area_index'):
        assert next(lines)=='2';data[name]=arr()
    assert list(lines)==[] and all(a.shape==(ny,nx) for a in data.values())
    return (int(zone[:4]),zone[4:].strip()),geometry,data


@pytest.mark.parametrize('alternate',[False,True])
def test_independent_parse_all_grids_and_units(alternate):
    ds,cfg,expected=case(alternate);files,report=export_geography(ds,cfg)
    zone,geometry,data=parse(files['GEO.DAT']);assert zone==((55,'S') if alternate else (31,'N'))
    assert geometry==[500,1000,1,1]
    for field in expected:np.testing.assert_allclose(data[field],expected[field][::-1],rtol=0,atol=1e-12)
    assert report['complete_model_input'] is False


@pytest.mark.parametrize('error',['missing_field','missing_cell','nan','marker','units','non_square','category','crs','submetre','time'])
def test_reject_missing_or_ambiguous(error):
    ds,cfg,_=case()
    if error=='missing_field':ds=ds.drop_vars('bowen')
    if error=='missing_cell':ds=ds.isel(x=[0,2]);ds.x.attrs['units']='m' # spacing now disagrees with y
    if error=='nan':ds.roughness.values[0,0]=np.nan
    if error=='marker':ds.elevation.values[0,0]=-9999
    if error=='units':cfg['fields']['elevation']['source_units']='km'
    if error=='non_square':ds=ds.assign_coords(x=[500500,502500,504500]);ds.x.attrs['units']='m'
    if error=='category':cfg['landuse_categories']=[10]
    if error=='crs':cfg['crs']='EPSG:4326'
    if error=='submetre':ds=ds.assign_coords(x=ds.x+.25);ds.x.attrs['units']='m'
    if error=='time':ds['roughness']=ds.roughness.expand_dims(time=[0])
    with pytest.raises(SchemaError):export_geography(ds,cfg)


def save_examples(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    for alternate in (False,True):
        ds,cfg,_=case(alternate);name='synthetic_south_km' if alternate else 'synthetic_north_m'
        ds.to_netcdf(path/f'A_{name}.nc',engine='scipy')
        (path/f'config_{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
