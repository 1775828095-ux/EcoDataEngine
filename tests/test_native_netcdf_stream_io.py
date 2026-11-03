"""Backend compatibility only, not physical-model validation."""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integrations/existing_air'))
from existing_air_converters.netcdf_io import load_dataset


@pytest.mark.parametrize('engine,format', [('h5netcdf',None), ('scipy','NETCDF3_CLASSIC'),
                                          ('scipy','NETCDF3_64BIT')])
def test_unicode_paths_preserve_decoded_values_coordinates_and_metadata(tmp_path, engine, format):
    path=tmp_path/'中文目录'/'气象.nc'
    path.parent.mkdir()
    original=xr.Dataset({'temperature': ('time', [273.15, np.nan, 291.25],
                        {'units':'K','standard_name':'air_temperature'})},
                        coords={'time':np.array(['2024-01-01','2024-01-02','2024-01-03'],dtype='datetime64[ns]')},
                        attrs={'source':'explicit test fixture'})
    kwargs={'engine':engine}
    if format:kwargs['format']=format
    original.to_netcdf(path, encoding={'temperature':{'dtype':'int16','scale_factor':.01,
                                                     'add_offset':273.15,'_FillValue':-32768}}, **kwargs)
    before=path.read_bytes()
    with path.open('rb') as stream:
        with xr.open_dataset(stream,engine=engine) as expected:
            xr.testing.assert_identical(load_dataset(path),expected.load())
    assert path.read_bytes()==before
    # Returned arrays remain readable after all file handles are closed.
    assert np.isclose(load_dataset(path).temperature.values[-1],291.25)


@pytest.mark.parametrize('payload',[b'not netcdf',b'CDF\x05unsupported',b'\x89HDF\r\n\x1a\nbroken'])
def test_bad_or_unsupported_files_fail_without_rewrite(tmp_path,payload):
    path=tmp_path/'invalid.nc';path.write_bytes(payload)
    with pytest.raises((ValueError,OSError)):
        load_dataset(path)
    assert path.read_bytes()==payload
