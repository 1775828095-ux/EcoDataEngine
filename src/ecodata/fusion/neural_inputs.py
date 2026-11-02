"""Read real experiment grids through existing readers; require explicit metadata."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import read_structure, spatial_axes, raster_plane


def read_experiment_stations(name, content, *, value_column='value', unit=None):
    import io
    frame = pd.read_csv(io.BytesIO(content), dtype={'station_id': str})
    if value_column not in frame:
        raise SchemaError(f'没有观测列 {value_column}，请明确映射实际数值字段')
    if value_column != 'value' and 'value' in frame:
        raise SchemaError('value已经存在，不能用另一个字段覆盖')
    frame = frame.rename(columns={value_column: 'value'})
    frame['value'] = pd.to_numeric(frame.value, errors='raise')
    if unit:
        frame['unit'] = unit
    frame.attrs = {'source_filename': Path(name).name, 'value_column_mapping': {value_column: 'value'},
                   'unit_confirmation': unit}
    return frame


def read_experiment_grid(name, content, *, variable=None, unit=None, timestamp=None, crs=None, categorical=False):
    ds = read_structure(name, content)
    x, y = spatial_axes(ds)
    candidates = [key for key, field in ds.data_vars.items() if {x, y}.issubset(field.dims) and field.dtype.kind in 'biuf']
    if variable is None:
        if len(candidates) != 1:
            raise SchemaError(f'存在多个/没有空间变量，请明确选择：{candidates}')
        variable = candidates[0]
    if variable not in candidates:
        raise SchemaError(f'变量 {variable} 不是二维数值空间场')
    bounds = [ds[axis].attrs.get('bounds') for axis in (x, y, 'time') if axis in ds.coords]
    ds = ds[[variable, *[name for name in bounds if name in ds.data_vars]]].copy()
    field = ds[variable]
    if set(field.dims) - {x, y, 'time'}:
        raise SchemaError('存在band/level等额外维度，请先明确选择，不能猜测波段日期')
    unit = unit or field.attrs.get('units')
    source_crs = crs or ds.attrs.get('crs')
    if not unit or unit == 'unknown':
        raise SchemaError('缺少明确单位，请确认后输入；分类土地利用可明确设为1')
    if not source_crs:
        raise SchemaError('缺少CRS，请确认后输入，不能从坐标范围猜测')
    ds[variable].attrs['units'] = unit
    if categorical:
        ds[variable].attrs['data_kind'] = 'categorical'
    original = {'filename': Path(name).name, 'crs': str(source_crs), 'unit': unit,
                'variable': variable, 'timestamp_confirmation': timestamp}
    source_crs = CRS.from_user_input(source_crs)
    if source_crs != CRS.from_epsg(4326):
        time_bounds_name = ds.time.attrs.get('bounds') if 'time' in ds.coords else None
        time_bounds = ds[time_bounds_name].copy() if time_bounds_name in ds.data_vars else None
        # Use GDAL/rasterio, not relabel projected x/y as longitude/latitude.
        planes = []
        count = ds.sizes.get('time', 1)
        for index in range(count):
            plane_ds = ds.isel(time=index, drop=True) if 'time' in ds.dims else ds
            plane, transform, _ = raster_plane(plane_ds, variable)
            from rasterio.transform import array_bounds
            bounds = array_bounds(*plane.shape, transform)
            target_transform, width, height = calculate_default_transform(source_crs, 'EPSG:4326', plane.shape[1], plane.shape[0], *bounds)
            if width * height * count > 2_000_000:
                raise SchemaError('重投影超过200万像元，请先裁剪数据')
            output = np.full((height, width), np.nan)
            reproject(plane, output, src_transform=transform, src_crs=source_crs,
                      dst_transform=target_transform, dst_crs='EPSG:4326', src_nodata=np.nan,
                      dst_nodata=np.nan, resampling=Resampling.nearest if categorical else Resampling.bilinear)
            planes.append(output)
        coords = {'lon': target_transform.c + (np.arange(width)+.5)*target_transform.a,
                  'lat': target_transform.f + (np.arange(height)+.5)*target_transform.e}
        dims, values = ('lat', 'lon'), planes[0]
        if 'time' in ds.dims:
            coords['time'] = ds.time
            dims, values = ('time', 'lat', 'lon'), np.stack(planes)
        attrs = dict(ds[variable].attrs)
        ds = xr.Dataset({variable: (dims, values)}, coords=coords, attrs=dict(ds.attrs))
        ds[variable].attrs = attrs
        if time_bounds is not None:
            ds[time_bounds_name] = time_bounds
    else:
        ds = ds.rename({old: new for old, new in ((x, 'lon'), (y, 'lat')) if old != new})
    ds = ds.sortby('lat').sortby('lon')
    if timestamp is not None and 'time' not in ds.dims:
        ds = ds.expand_dims(time=[np.datetime64(timestamp)])
    ds.attrs.update(crs='EPSG:4326', spatial_x='lon', spatial_y='lat',
                    experiment_input=json.dumps(original, ensure_ascii=False))
    ds.lat.attrs.update(standard_name='latitude', units='degrees_north')
    ds.lon.attrs.update(standard_name='longitude', units='degrees_east')
    return ds
