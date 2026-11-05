"""Read local NetCDF products without passing Unicode paths to the C backend.

Uses the same signature/stream strategy as the existing converter's netcdf_io.
No data rewriting, inferred metadata or retry with different scientific options.
"""
from pathlib import Path

import xarray as xr


def load_dataset(path, **decode_options):
    """Return an eagerly loaded dataset after closing both file handles.

    NetCDF4/HDF5 uses h5netcdf; classic and 64-bit-offset NetCDF3 use scipy.
    Choose by bytes, not suffix. Unsupported formats and damaged files fail.
    Python opens the path, so compiled backends never receive its filename.
    """
    path = Path(path)
    with path.open('rb') as stream:
        signature = stream.read(8)
        stream.seek(0)
        if signature == b'\x89HDF\r\n\x1a\n':
            engine = 'h5netcdf'
        elif signature[:4] in (b'CDF\x01', b'CDF\x02'):
            engine = 'scipy'
        else:
            raise ValueError('Unsupported NetCDF file signature: ' + str(path))
        with xr.open_dataset(stream, engine=engine, **decode_options) as dataset:
            return dataset.load()
