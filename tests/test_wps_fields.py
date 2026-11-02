"""Tests use a reader independent of the converter's framing helpers."""
import io
import struct
import tempfile

import numpy as np
import pytest
import xarray as xr
from scipy.io import FortranFile

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.wps_fields import export_pressure


def read_records(payload):
    stream = tempfile.TemporaryFile(dir='.')
    stream.write(payload)
    stream.seek(0)
    file = FortranFile(stream, header_dtype=np.dtype('>u4'))
    rows = []
    while stream.tell() < len(payload):
        assert file.read_record('>i4').tolist() == [5]
        header = file.read_record('u1').tobytes()
        hdate = header[:24].decode('ascii').strip()
        label = header[60:69].decode('ascii').strip()
        units = header[69:94].decode('ascii').strip()
        level, nx, ny, projection = struct.unpack('>fiii', header[140:156])
        assert projection == 0
        grid = file.read_record('u1').tobytes()
        assert grid[:8] == b'SWCORNER'
        grid_values = struct.unpack('>5f', grid[8:])
        flag = file.read_record('>i4')[0]
        values = file.read_record('>f4').reshape(ny, nx)
        rows.append(dict(date=hdate, field=label, units=units, level=level,
                         grid=grid_values, grid_relative=bool(flag), values=values))
    file.close()
    return rows


def case():
    coords = {'time': np.array(['2024-02-29T00','2024-02-29T06'], dtype='datetime64[h]'),
              'pressure': [850,500], 'latitude': [40,35], 'longitude': [110,105]}
    values = np.arange(16).reshape(2,2,2,2)+250.
    ds = xr.Dataset({'temp': (('time','pressure','latitude','longitude'), values)}, coords=coords)
    ds.temp.attrs['units']='K'
    ds.pressure.attrs['units']='hPa'
    ds.latitude.attrs['units']='degrees_north'
    ds.longitude.attrs['units']='degrees_east'
    cfg = {'mode':'pressure_fields','version':'WPS-4.6-intermediate-v5-fields','time_basis':'UTC',
           'crs':'EPSG:4326','coordinates':dict(time='time',pressure='pressure',latitude='latitude',longitude='longitude'),
           'interval_seconds':21600,'fields':{'TT':{'variable':'temp'}},'prefix':'ATM'}
    return ds,cfg


def test_independent_parse_pressure_date_grid_and_number():
    ds,cfg=case();files,report=export_pressure(ds,cfg)
    assert report['records_per_time']==2
    assert report['native_model_input_verified'] is False
    rows=read_records(files['ATM_2024-02-29_06.wps'])
    assert len(rows)==2
    assert rows[0]['date']=='2024-02-29_06:00:00'
    assert rows[0]['units']=='K' and rows[0]['level']==85000
    assert rows[0]['grid'][:4]==(35,105,5,5)
    np.testing.assert_array_equal(rows[0]['values'],ds.temp.values[1,0,::-1,::-1])


@pytest.mark.parametrize('source,unit,expected', [(20,'degC',293.15),(0,'degrees_Celsius',273.15)])
def test_temperature_unit_conversion(source,unit,expected):
    ds,cfg=case();ds.temp[:]=source;ds.temp.attrs['units']=unit
    files,_=export_pressure(ds,cfg)
    rows=read_records(files['ATM_2024-02-29_00.wps'])
    assert np.max(abs(rows[0]['values'].astype('f8')-expected))<3e-5


def test_winds_earth_relative_and_component_values():
    ds,cfg=case();ds['east']=ds.temp.copy();ds.east[:]=1;ds.east.attrs['units']='m/s'
    ds['north']=ds.temp.copy();ds.north[:]=-2;ds.north.attrs['units']='m/s'
    cfg['fields']={'UU':{'variable':'east'},'VV':{'variable':'north'}}
    cfg['wind_reference']='earth_relative'
    files,_=export_pressure(ds,cfg);rows=read_records(files['ATM_2024-02-29_00.wps'])
    assert all(not row['grid_relative'] for row in rows)
    assert rows[0]['values'].min()==1 and rows[-1]['values'].max()==-2


def test_surface_pressure_and_mask():
    ds,cfg=case();surface=ds.temp.isel(pressure=0,drop=True)
    ds['p']=surface.copy();ds.p[:]=1000;ds.p.attrs['units']='hPa'
    ds['land']=surface.copy();ds.land[:]=1;ds.land.attrs['units']='1'
    cfg['fields']={'PSFC':{'variable':'p'},'LANDSEA':{'variable':'land'}}
    cfg['landsea_convention']='land=1,water=0'
    files,_=export_pressure(ds,cfg);rows=read_records(files['ATM_2024-02-29_00.wps'])
    assert rows[0]['level']==200100 and rows[0]['units']=='Pa'
    assert rows[0]['values'].min()==100000
    assert rows[1]['values'].min()==1


@pytest.mark.parametrize('change,error', [
    ('nan','NaN'),('missing','源变量不存在'),('time_gap','时间间隔'),('units','单位'),
    ('grid','等间距'),('pressure','等压层'),('wind_unpaired','成对'),
    ('wind_unknown','wind_reference'),('field','不支持'),('required','未映射'),
    ('duplicate_time','重复'),('ambiguous_dimension','维度'),('half_hour','整点'),
    ('source_sentinel','missing_value'),('mask_unknown','landsea_convention')])
def test_rejects_ambiguous_or_invalid_input(change,error):
    ds,cfg=case()
    if change=='nan':ds.temp.values[0,0,0,0]=np.nan
    elif change=='missing':cfg['fields']['TT']['variable']='absent'
    elif change=='time_gap':cfg['interval_seconds']=3600
    elif change=='units':ds.temp.attrs['units']=''
    elif change=='grid':ds=ds.assign_coords(latitude=[35,35])
    elif change=='pressure':ds=ds.assign_coords(pressure=[500,500]);ds.pressure.attrs['units']='hPa'
    elif change=='wind_unpaired':cfg['fields']={'UU':{'variable':'temp'}}
    elif change=='wind_unknown':cfg['fields']={'UU':{'variable':'temp'},'VV':{'variable':'temp'}}
    elif change=='field':cfg['fields']={'SOILX':{'variable':'temp'}}
    elif change=='required':cfg['required_fields']=['TT','RH']
    elif change=='duplicate_time':ds=ds.assign_coords(time=np.array(['2024-02-29T00','2024-02-29T00'],dtype='datetime64[h]'))
    elif change=='ambiguous_dimension':ds['temp']=ds.temp.expand_dims(extra=[0])
    elif change=='half_hour':ds=ds.assign_coords(time=ds.time+np.timedelta64(30,'m'))
    elif change=='source_sentinel':ds.temp.attrs['missing_value']=250
    elif change=='mask_unknown':cfg['fields']={'LANDSEA':{'variable':'temp'}}
    with pytest.raises(SchemaError,match=error):export_pressure(ds,cfg)


def test_percent_and_fraction_relative_humidity():
    ds,cfg=case();ds['humidity']=ds.temp.copy();ds.humidity[:]=.5;ds.humidity.attrs['units']='1'
    cfg['fields']={'RH':{'variable':'humidity'}}
    files,_=export_pressure(ds,cfg)
    assert read_records(files['ATM_2024-02-29_00.wps'])[0]['values'].min()==50


def test_field_subset_is_explicitly_incomplete():
    ds,cfg=case();_,report=export_pressure(ds,cfg)
    assert 'RH' in report['missing_pressure_fields_for_common_WRF_chain']
    assert report['complete_model_input'] is False



def test_mean_sea_level_pressure_special_wps_level():
    ds,cfg=case();ds['slp']=ds.temp.isel(pressure=0,drop=True).copy()
    ds.slp[:]=1013.25;ds.slp.attrs['units']='hPa'
    cfg['fields']={'PMSL':{'variable':'slp'}}
    files,_=export_pressure(ds,cfg)
    row=read_records(files['ATM_2024-02-29_00.wps'])[0]
    assert row['level']==201300 and row['units']=='Pa'
    assert row['values'].min()==101325


def test_fractional_land_mask_is_rejected_not_rounded():
    ds,cfg=case();ds['mask']=ds.temp.isel(pressure=0,drop=True).copy()
    ds['mask'][:]=.4;ds['mask'].attrs['units']='1'
    cfg['fields']={'LANDSEA':{'variable':'mask'}};cfg['landsea_convention']='land=1,water=0'
    with pytest.raises(SchemaError,match='0/1'):export_pressure(ds,cfg)


@pytest.mark.parametrize('conflict',['temperature','pressure','latitude'])
def test_source_units_cannot_be_overridden(conflict):
    ds,cfg=case()
    if conflict=='temperature':cfg['fields']['TT']['source_units']='degC'
    elif conflict=='pressure':cfg['coordinate_units']={'pressure':'Pa'}
    else:cfg['coordinate_units']={'latitude':'radian'}
    with pytest.raises(SchemaError,match='冲突'):export_pressure(ds,cfg)
