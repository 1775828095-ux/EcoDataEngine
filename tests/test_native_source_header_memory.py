"""Native inputs must not allocate numerical arrays during source discovery."""
from contextlib import ExitStack
from io import BytesIO
import numpy as np
import pytest
import xarray as xr
from ecodata.models.project_binding import ProjectFiles
from ecodata.models.engineering.preparation import source_fields


@pytest.mark.parametrize('declared', [False, True])
def test_native_header_does_not_open_arrays(monkeypatch, declared):
    ds=xr.Dataset({'A': ('x', np.arange(8.))}, attrs={'IOAPI_VERSION': 'test', 'description': 'x'*90000})
    upload=BytesIO(ds.to_netcdf(engine='scipy'));upload.name='native.nc'
    def forbidden(*args, **kwargs):
        raise AssertionError('Native arrays opened during independent-source discovery')
    monkeypatch.setattr(xr, 'open_dataset', forbidden)
    with ExitStack() as stack:
        files=ProjectFiles([upload], stack)
        if declared:
            with pytest.raises(ValueError, match='原生工程文件不能同时声明为独立新来源'):
                source_fields(files, {'native.nc': {'role': 'source'}})
        else:
            assert source_fields(files, {}) == []


def test_component_role_probe_does_not_open_classic_arrays(monkeypatch, tmp_path):
    from ecodata.models.component_service import prepare_components
    ds=xr.Dataset({'A': ('x', np.arange(8.))}, attrs={'IOAPI_VERSION': 'test', 'description': 'x'*90000})
    upload=BytesIO(ds.to_netcdf(engine='scipy'));upload.name='native.nc'
    def forbidden(*args, **kwargs):
        raise AssertionError('Arrays opened merely to discover processed_product role')
    monkeypatch.setattr(xr, 'open_dataset', forbidden)
    assert prepare_components([upload], 'cmaq', tmp_path) is None
