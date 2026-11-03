import io,json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from scipy.io import netcdf_file
from ecodata.models.adapters.cmaq_boundary import export_boundary
from ecodata.core.exceptions import SchemaError


def case(alternate=False):
    nx,ny,nl=(3,2,2) if not alternate else (2,3,3)
    grid=dict(GDTYP=1,P_ALP=0,P_BET=0,P_GAM=0,XCENT=0,YCENT=0,XORIG=110,YORIG=30,XCELL=.1,YCELL=.1,NCOLS=nx,NROWS=ny,GDNAM='TEST_BOUNDARY')
    if alternate:grid.update(GDTYP=2,P_ALP=30,P_BET=60,P_GAM=-97,XCENT=-97,YCENT=40,XORIG=-12000,YORIG=-18000,XCELL=12000,YCELL=12000)
    # Input is unordered row-major external cells, not encoded perimeter order.
    cells=[(i,j) for j in range(ny+2) for i in range(nx+2) if i in (0,nx+1) or j in (0,ny+1)][::-1]
    bc,br=np.array(cells).T;n=len(cells)
    vals=np.array([[[100*(k+1)+10*j+i+t for i,j in cells] for k in range(nl)] for t in range(2)],dtype=float)
    unit='m' if alternate else 'degree';vertical=dict(VGTYP=7,VGTOP=10000.,VGLVLS=[1,.7,.4,0] if alternate else [1,.5,0])
    ds=xr.Dataset({'ozone':(('time','layer','perim'),vals,{'units':'ppbV'}),'sulfate':(('time','layer','perim'),vals/100,{'units':'ug/m^3'}),
        'bc':('perim',bc),'br':('perim',br),'px':('perim',grid['XORIG']+(bc-.5)*grid['XCELL'],{'units':unit}),
        'py':('perim',grid['YORIG']+(br-.5)*grid['YCELL'],{'units':unit})},
        coords={'time':np.array(['2024-02-29T23','2024-03-01T00'],dtype='datetime64[h]'),'layer':np.arange(1,nl+1),'perim':np.arange(n),
                'x':grid['XORIG']+(np.arange(nx)+.5)*grid['XCELL'],'y':grid['YORIG']+(np.arange(ny)+.5)*grid['YCELL']},attrs=vertical)
    ds.x.attrs['units']=unit;ds.y.attrs['units']=unit
    cfg=dict(mode='boundary_concentrations',version='CMAQ-5.5',mechanism='cb6r3_ae7_aq',grid=grid,vertical=vertical,
        axes=dict(time='time',layer='layer',perimeter='perim',x='x',y='y'),time_mode='hourly',source_timezone='UTC',coordinate_units=unit,
        perimeter_mapping=dict(column='bc',row='br',x='px',y='py'),fields={'O3':dict(source='ozone',type='gas',source_units='ppbV'),'ASO4J':dict(source='sulfate',type='aerosol_mass',source_units='ug/m^3')},
        required_species=['O3','ASO4J'],missing_values=[-9999],source=dict(kind='synthetic',description='Synthetic external perimeter, no real boundary observations',license='CC0 synthetic values'))
    if alternate:
        cfg.update(source_crs='+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +R=6370000 +units=m',earth_radius_m=6370000)
        ds['ozone']=ds.ozone*.001;ds.ozone.attrs['units']='ppmV';cfg['fields']['O3']['source_units']='ppmV'
        ds=ds.rename({'time':'valid','layer':'z','perim':'edge','ozone':'gas'}).transpose('edge','valid','z','y','x',missing_dims='ignore')
        cfg['axes'].update(time='valid',layer='z',perimeter='edge');cfg['fields']['O3']['source']='gas'
    return ds,cfg


@pytest.mark.parametrize('alternate',[False,True])
def test_independent_perimeter_values_and_metadata(alternate,tmp_path):
    ds,cfg=case(alternate);files,report=export_boundary(ds,cfg)
    # Explicitly transcribed from the official CMAS BDY.jpg S/E/N/W diagram.
    cells=([(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3),(0,3),(1,3),(2,3),(3,3),(0,0),(0,1),(0,2)] if not alternate else
           [(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(0,4),(1,4),(2,4),(0,0),(0,1),(0,2),(0,3)])
    nl=3 if alternate else 2
    expected=np.array([[[100*(k+1)+10*j+i+t for i,j in cells] for k in range(nl)] for t in range(2)])
    with netcdf_file(io.BytesIO(files['boundary.CMAQ55.nc']),mmap=False) as nc:
        assert nc.FTYPE==2 and nc.NTHIK==1 and nc.NLAYS==nl and nc.dimensions['PERIM']==14
        assert nc.variables['O3'].dimensions==('TSTEP','LAY','PERIM')
        np.testing.assert_allclose(nc.variables['O3'][:],expected*.001,rtol=1e-7,atol=1e-9)
        np.testing.assert_allclose(nc.variables['ASO4J'][:],expected/100,rtol=1e-7,atol=1e-9)
        np.testing.assert_array_equal(nc.variables['TFLAG'][:,0],[[2024060,230000],[2024061,0]])
        np.testing.assert_allclose(nc.VGLVLS,cfg['vertical']['VGLVLS'],rtol=1e-7)
        assert nc.variables['O3'].units.strip()==b'ppmV'
    from PseudoNetCDF import pncopen
    target=tmp_path/'boundary.nc';target.write_bytes(files['boundary.CMAQ55.nc'])
    f=pncopen(str(target),format='ioapi')
    try:assert f.FTYPE==2 and f.variables['O3'].shape==(2,nl,14)
    finally:f.close()
    assert report['model_run_verified'] is False


@pytest.mark.parametrize('error',['missing','gap','units','perimeter','location','vertical','layers','negative','nan','species','required'])
def test_boundary_rejects_inconsistent_input(error):
    ds,cfg=case()
    if error=='missing':ds=ds.drop_vars('ozone')
    if error=='gap':ds=ds.assign_coords(time=np.array(['2024-02-29T23','2024-03-01T01'],dtype='datetime64[h]'))
    if error=='units':cfg['fields']['O3']['source_units']='ppmV'
    if error=='perimeter':ds=ds.isel(perim=slice(0,-1))
    if error=='location':ds.px.values[0]+=.1
    if error=='vertical':ds.attrs['VGTOP']=5000
    if error=='layers':ds=ds.assign_coords(layer=[2,1])
    if error=='negative':ds.ozone.values[0,0,0]=-1
    if error=='nan':ds.ozone.values[0,0,0]=np.nan
    if error=='species':cfg['fields']['NOx']=cfg['fields'].pop('O3')
    if error=='required':cfg['required_species'].append('NO2')
    with pytest.raises(SchemaError):export_boundary(ds,cfg)


def save_examples(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    for alternate in (False,True):
        ds,cfg=case(alternate);name='synthetic_LCC' if alternate else 'synthetic_LATLON'
        ds.to_netcdf(path/f'A_{name}.nc',engine='scipy')
        (path/f'config_{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
