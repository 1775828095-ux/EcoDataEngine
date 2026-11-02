"""Synthetic coordinate/unit cases; scipy Fortran reader is independent of writer."""
import copy
import json
import struct
import tempfile
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from scipy.io import FortranFile
from ecodata.models.adapters.wps_fields import export_pressure
from ecodata.core.exceptions import SchemaError


def case(alternate=False):
    nx,ny=(3,2) if not alternate else (2,3)
    unit,scale=('km',1) if alternate else ('m',1000)
    x=np.arange(nx)*12*scale;y=np.arange(ny)*9*scale
    t=np.array(['2024-02-29T00','2024-02-29T06'],dtype='datetime64[h]')
    a=np.arange(2*ny*nx).reshape(2,ny,nx)+15.
    ds=xr.Dataset({'skin':(('time','y','x'),a,{'units':'degC'})},coords={'time':t,'y':y,'x':x})
    ds.x.attrs['units']=unit;ds.y.attrs['units']=unit
    cfg=dict(mode='pressure_fields',version='WPS-4.6-intermediate-v5-fields',time_basis='UTC',
        crs='+proj=lcc +lat_0=40 +lon_0=-100 +lat_1=30 +lat_2=60 +R=6370000 +units=m +no_defs',
        projection=dict(type='lambert',earth_radius_m=6370000,latitude_origin=40,standard_longitude=-100,true_latitudes=[30,60]),
        coordinates=dict(time='time',x='x',y='y'),interval_seconds=21600,fields={'SKINTEMP':{'variable':'skin'}})
    if alternate:
        ds=ds.isel(x=slice(None,None,-1),y=slice(None,None,-1)).rename({'time':'date','x':'east','y':'north','skin':'temperature'}).transpose('east','date','north')
        cfg['coordinates']=dict(time='date',x='east',y='north');cfg['fields']['SKINTEMP']['variable']='temperature'
    return ds,cfg,a


def parse(raw):
    with tempfile.TemporaryFile() as fp:
        fp.write(raw);fp.seek(0)
        f=FortranFile(fp,header_dtype='>u4')
        assert f.read_record('>i4').tolist()==[5]
        h=f.read_record('u1').tobytes()
        level,nx,ny,iproj=struct.unpack('>fiii',h[140:])
        assert iproj==3 and level==200100 and h[60:69].strip()==b'SKINTEMP'
        assert h[69:94].strip()==b'K'
        grid=f.read_record('u1').tobytes();assert grid[:8]==b'SWCORNER'
        assert f.read_record('>i4').tolist()==[0]
        a=f.read_record('>f4').reshape(ny,nx)
        assert fp.tell()==len(raw)
        return h[:24].decode().strip(),struct.unpack('>8f',grid[8:]),a


@pytest.mark.parametrize('alternate',[False,True])
def test_lambert_independent_records_units_and_values(alternate):
    ds,cfg,expected=case(alternate);files,report=export_pressure(ds,cfg)
    assert report['projection']['iproj']==3
    for i,hour in enumerate(('00','06')):
        date,grid,actual=parse(files[f'ATM_2024-02-29_{hour}.wps'])
        assert date==f'2024-02-29_{hour}:00:00'
        # Analytic origin at x=y=0, explicit 12/9 km spacing, no converter-derived oracle.
        np.testing.assert_allclose(grid,[40,-100,12,9,-100,30,60,6370],rtol=0,atol=1e-5)
        np.testing.assert_allclose(actual,expected[i]+273.15,rtol=0,atol=3e-5)


@pytest.mark.parametrize('error',['ellipsoid','units','time','field','nan','projection','spacing'])
def test_lambert_rejects_inconsistent_data(error):
    ds,cfg,_=case()
    if error=='ellipsoid':cfg['crs']=cfg['crs'].replace('+R=6370000','+datum=WGS84')
    if error=='units':cfg['coordinate_units']={'x':'km'}
    if error=='time':cfg['interval_seconds']=3600
    if error=='field':cfg['fields']['SKINTEMP']['variable']='missing'
    if error=='nan':ds.skin.values[0,0,0]=np.nan
    if error=='projection':cfg['projection']['type']='polar'
    if error=='spacing':ds=ds.assign_coords(x=[0,12000,25000]);ds.x.attrs['units']='m'
    with pytest.raises(SchemaError):export_pressure(ds,cfg)


def save_examples(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for alternate in (False,True):
        ds,cfg,_=case(alternate);name='synthetic_mapped_km' if alternate else 'synthetic_m'
        # scipy serialization supports Unicode paths and requires no extra native dependency.
        ds.to_netcdf(destination/f'A_{name}.nc',engine='scipy')
        (destination/f'config_{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
