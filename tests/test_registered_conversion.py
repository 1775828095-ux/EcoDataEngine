import io,zipfile
import numpy as np
import pytest
from netCDF4 import Dataset
from ecodata.interface.registered_conversion import accept_netcdf,member,compatible_netcdf


def archive(raw):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:z.writestr('input.nc',raw)
    return buffer.getvalue()


def make(path,value=2,origin=0,units='ppm'):
    with Dataset(path,'w') as data:
        data.createDimension('x',2);data.XORIG=origin
        v=data.createVariable('O3','f4',('x',));v.units=units;v[:]=[value,3]
    return path.read_bytes()


def test_current_bytes_retained(tmp_path):
    ref=tmp_path/'reference.nc';raw=make(ref)
    current,report=accept_netcdf(archive(raw),'input.nc',ref)
    assert current==raw and report['reference']==str(ref)


@pytest.mark.parametrize('kwargs',[{'value':7},{'origin':1},{'units':'ppb'}])
def test_mismatch_rejected_without_fixture_fallback(tmp_path,kwargs):
    ref=tmp_path/'reference.nc';make(ref)
    raw=make(tmp_path/'other.nc',**kwargs)
    with pytest.raises(ValueError):accept_netcdf(archive(raw),'input.nc',ref)


def test_missing_member_rejected():
    with pytest.raises(ValueError):member(archive(b'not executed'),'wrong.nc')


def test_structural_contract_accepts_new_values_without_changing_source(tmp_path):
    ref=tmp_path/'reference.nc';original=make(ref)
    raw=make(tmp_path/'new.nc',value=7)
    result=compatible_netcdf(raw,ref)
    assert result['changed_fields']==['O3'] and ref.read_bytes()==original


@pytest.mark.parametrize('kwargs',[{'origin':1},{'units':'ppb'},{'value':float('nan')},{'value':-1}])
def test_structural_contract_rejects_grid_units_nan_negative(tmp_path,kwargs):
    ref=tmp_path/'reference.nc';make(ref)
    raw=make(tmp_path/'new.nc',**kwargs)
    with pytest.raises(ValueError):compatible_netcdf(raw,ref)
