"""Path-independent reads using xarray's upstream h5netcdf/scipy backends.

https://github.com/h5netcdf/h5netcdf (BSD-3-Clause)
https://github.com/pydata/xarray (Apache-2.0)
No source rewriting, guessed metadata, resampling or backend-error fallback.
"""
from pathlib import Path


def load_dataset(path):
    """Load and close a supported NetCDF through a Python-opened binary stream.

    The C NetCDF library never receives a Windows Unicode filename. Choose by
    format signature, not extension. Invalid data and unsupported CDF5 fail.
    """
    import xarray as xr

    path = Path(path)
    with path.open('rb') as stream:
        signature = stream.read(8)
        stream.seek(0)
        if signature == b'\x89HDF\r\n\x1a\n':
            engine = 'h5netcdf'
        elif signature[:4] in (b'CDF\x01', b'CDF\x02'):
            engine = 'scipy'
        else:
            raise ValueError('未支持的NetCDF文件签名（不猜测或替换数据）：' + str(path))
        with xr.open_dataset(stream, engine=engine) as dataset:
            return dataset.load()
