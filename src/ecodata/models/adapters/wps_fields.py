"""Mapped geographic or spherical Lambert atmospheric fields to WPS 4.6 intermediate v5.

Uses the existing SST record framing. No meteorological variables are derived.
"""
import json
import re
import struct

import numpy as np
import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.wps_sst import _record, _text

# Names and units match WPS v4.6.0 METGRID.TBL.ARW / Vtable.GFS.
FIELDS = {
    'TT': ('K', 'pressure', 'Air temperature'),
    'UU': ('m s-1', 'pressure', 'Zonal wind'),
    'VV': ('m s-1', 'pressure', 'Meridional wind'),
    'RH': ('%', 'pressure', 'Relative humidity'),
    'GHT': ('m', 'pressure', 'Geopotential height'),
    'PSFC': ('Pa', 'surface', 'Surface pressure'),
    'PMSL': ('Pa', 'surface', 'Mean sea level pressure'),
    'SKINTEMP': ('K', 'surface', 'Skin temperature'),
    'SOILHGT': ('m', 'surface', 'Terrain height'),
    'LANDSEA': ('1', 'surface', 'Land mask: land=1 water=0'),
    'SST': ('K', 'surface', 'Sea surface temperature'),
    'SNOW': ('kg m-2', 'surface', 'Snow water equivalent'),
    'SNOWH': ('m', 'surface', 'Snow depth'),
}
SOIL_DEPTHS = [(0,10),(10,40),(40,100),(100,200),(10,200)]
for top,bottom in SOIL_DEPTHS:
    FIELDS[f'ST{top:03d}{bottom:03d}'] = ('K','soil','Soil temperature')
    FIELDS[f'SM{top:03d}{bottom:03d}'] = ('fraction','soil','Volumetric soil moisture')

ALIASES = {'fraction':'1', 'degK': 'K', 'm/s': 'm s-1', 'mb': 'hPa', 'millibar': 'hPa',
           'degrees_north': 'degree', 'degrees_east': 'degree'}


def _units(value):
    return ALIASES.get(value, value)


def _check_unit_declaration(actual, declared, label):
    if actual and declared:
        scale, offset = unit_transform(_units(actual), _units(declared))
        if not np.isclose(scale,1.,rtol=0,atol=1e-12) or not np.isclose(offset,0.,rtol=0,atol=1e-12):
            raise SchemaError(f'{label}配置单位与数据已有units冲突，不能覆盖真实单位')


def _axis(dataset, name, role):
    if name not in dataset.coords or dataset[name].dims != (name,):
        raise SchemaError(f'{role}必须是同名的一维坐标：{name}')
    values = np.asarray(dataset[name].values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise SchemaError(f'{role}需至少2个有效坐标')
    delta = np.diff(values)
    if not (np.all(delta > 0) or np.all(delta < 0)) or not np.allclose(delta, delta[0], rtol=0, atol=1e-6):
        raise SchemaError(f'{role}需严格单调等间距；不自动重网格')
    return values


def export_pressure(dataset, config):
    """Return native-format candidates; external metgrid acceptance is separate."""
    if config.get('mode') != 'pressure_fields' or config.get('version') != 'WPS-4.6-intermediate-v5-fields':
        raise SchemaError('大气场模式需mode=pressure_fields及version=WPS-4.6-intermediate-v5-fields')
    if config.get('time_basis') != 'UTC':
        raise SchemaError('需明确time_basis=UTC')
    projected=config.get('projection') is not None
    if not projected and config.get('crs')!='EPSG:4326':raise SchemaError('经纬度网格需crs=EPSG:4326')
    axes = config.get('coordinates', {})
    horizontal=('y','x') if projected else ('latitude','longitude')
    if any(not isinstance(axes.get(k), str) or not axes[k] for k in ('time', *horizontal)):
        raise SchemaError('coordinates须映射time及latitude/longitude或投影y/x；不猜测轴')
    tim, lat, lon = (axes[k] for k in ('time', *horizontal))
    if tim not in dataset.coords or dataset[tim].dims != (tim,):
        raise SchemaError('time必须为同名一维坐标')
    if dataset[tim].dtype.kind != 'M':
        raise SchemaError('仅接受已解码公历datetime64时间，拒绝未说明历法/数值时间')
    times = pd.DatetimeIndex(dataset[tim].values)
    if len(times) == 0 or times.isna().any() or times.has_duplicates or not times.is_monotonic_increasing:
        raise SchemaError('时间缺失、重复、乱序或无记录')
    if any(t.minute or t.second or t.microsecond or t.nanosecond for t in times):
        raise SchemaError('WPS文件名仅接受UTC整点；不移动时间')
    interval = config.get('interval_seconds')
    if len(times) > 1:
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise SchemaError('序列须显式提供正整数interval_seconds')
        if not np.all(np.diff(times.values) == np.timedelta64(interval, 's')):
            raise SchemaError('时间间隔与interval_seconds不一致或缺时刻；不补值')
    coordinate_units = config.get('coordinate_units', {})
    if projected:
        from ecodata.models.adapters.wps_projection import lambert_record
        projection,ny,nx,projection_info=lambert_record(dataset,config,_axis,_check_unit_declaration)
        iproj=3
    else:
        latitude = _axis(dataset, lat, '纬度')
        longitude = _axis(dataset, lon, '经度')
        if latitude.min() < -90 or latitude.max() > 90 or longitude.min() < -180 or longitude.max() > 360 or np.ptp(longitude) >= 360:
            raise SchemaError('经纬度范围不合法')
        coordinate_units = config.get('coordinate_units', {})
        for name, role in ((lat, 'latitude'), (lon, 'longitude')):
            declared = coordinate_units.get(role, dataset[name].attrs.get('units', ''))
            _check_unit_declaration(dataset[name].attrs.get('units',''), declared, role)
            if _units(declared) not in {'degree', 'degrees'}:
                raise SchemaError(f'{role}单位需明确为角度')
        iproj=0
        projection_info={'iproj':0,'type':'regular_latlon'}
    source = config.get('map_source', 'EcoData mapped atmospheric data')
    prefix = config.get('prefix', 'ATM')
    if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,15}', prefix):
        raise SchemaError('prefix须是1..16位ASCII字母、数字或下划线，首位字母')
    mappings = config.get('fields')
    if not isinstance(mappings, dict) or not mappings:
        raise SchemaError('fields须将WPS标签映射到源变量配置')
    unknown = sorted(set(mappings)-set(FIELDS))
    if unknown:
        raise SchemaError(f'不支持的WPS字段{unknown}；仅支持已列明字段和深度；不推算其他层或变量')
    required = config.get('required_fields', [])
    if not isinstance(required, list) or any(k not in FIELDS for k in required):
        raise SchemaError('required_fields需为支持字段标签列表')
    missing = sorted(set(required)-set(mappings))
    if missing:
        raise SchemaError(f'配置要求的字段未映射：{missing}')
    if bool('UU' in mappings) != bool('VV' in mappings):
        raise SchemaError('风场UU与VV必须成对提供')
    wind_frame = config.get('wind_reference')
    if 'UU' in mappings and wind_frame != 'earth_relative':
        raise SchemaError('本入口仅支持明确wind_reference=earth_relative；网格相对风需先正确旋转，不能直接改标志')
    prepared = []
    level_sets = {}
    transformations = {}
    for label, mapping in mappings.items():
        if not isinstance(mapping, dict) or set(mapping)-({'variable','source_units','source_depth_cm','soil_moisture_basis'} if FIELDS[label][1]=='soil' else {'variable','source_units'}) or not mapping.get('variable'):
            raise SchemaError(f'{label}配置仅接受variable和可选source_units')
        variable = mapping['variable']
        if variable not in dataset.data_vars:
            raise SchemaError(f'{label}源变量不存在：{variable}')
        array = dataset[variable]
        if array.dtype.kind not in 'iuf':
            raise SchemaError(f'{label}必须为实数')
        target_units, kind, description = FIELDS[label]
        source_units = mapping.get('source_units', array.attrs.get('units', ''))
        _check_unit_declaration(array.attrs.get('units',''), source_units, label)
        if not source_units:
            raise SchemaError(f'{label}单位不明；必须配置source_units或保留源units属性')
        if label == 'LANDSEA' and config.get('landsea_convention') != 'land=1,water=0':
            raise SchemaError('LANDSEA需显式landsea_convention=land=1,water=0；不反转掩码')
        if kind=='soil':
            expected_depth=[int(label[2:5]),int(label[5:8])]
            if mapping.get('source_depth_cm')!=expected_depth:
                raise SchemaError(f'{label}需source_depth_cm={expected_depth}，不能把不同深度层改名')
            if 'soil_depth_cm' in array.attrs and list(array.attrs['soil_depth_cm'])!=expected_depth:
                raise SchemaError(f'{label}源深度元数据与配置冲突')
            if label.startswith('SM') and (mapping.get('soil_moisture_basis')!='volumetric' or _units(source_units) not in {'1','%','percent','m3/m3','m^3/m^3','m3 m-3'}):
                raise SchemaError(f'{label}只接受明确volumetric含水率；不把kg/kg当体积比')
        scale, offset = unit_transform(_units(source_units), _units(target_units))
        pressure_axis = axes.get('pressure')
        dims = (tim, pressure_axis, lat, lon) if kind == 'pressure' else (tim, lat, lon)
        if set(array.dims) != set(dims) or len(array.dims) != len(dims):
            raise SchemaError(f'{label}维度须为{dims}，实际{array.dims}；不自动选层或广播')
        array = array.sortby(lat).sortby(lon).transpose(*dims).astype('float64')
        values = np.asarray(array.values)*scale+offset
        if not np.isfinite(values).all():
            raise SchemaError(f'{label}包含NaN/无效/缺测；不插值、不补造')
        for attr in ('_FillValue', 'missing_value'):
            for marker in np.atleast_1d(dataset[variable].attrs.get(attr, [])):
                if np.any(dataset[variable].values == marker):
                    raise SchemaError(f'{label}包含源定义的{attr}缺测标记')
        if kind=='soil' and (values.min() < (0 if label.startswith('SM') else 100) or values.max() > (1 if label.startswith('SM') else 400)):
            raise SchemaError(f'{label}土壤数值超出预设物理检查范围')
        bounds = {'TT': (100, 400), 'RH': (0, 100), 'PSFC': (1000, 120000),
                  'PMSL': (1000, 120000), 'SST': (250, 330), 'SKINTEMP': (100, 400),
                  'UU': (-500, 500), 'VV': (-500, 500), 'SNOW': (0, 1e6), 'SNOWH': (0, 1000)}
        if label in bounds and (values.min() < bounds[label][0] or values.max() > bounds[label][1]):
            raise SchemaError(f'{label}超出预先规定的工程检查范围{bounds[label]} {target_units}')
        if label == 'LANDSEA' and not np.isin(values, [0., 1.]).all():
            raise SchemaError('LANDSEA只接受0/1分类值')
        if kind == 'pressure':
            if pressure_axis not in dataset.coords or dataset[pressure_axis].dims != (pressure_axis,):
                raise SchemaError('需明确一维pressure坐标')
            pressure_unit = coordinate_units.get('pressure', dataset[pressure_axis].attrs.get('units', ''))
            _check_unit_declaration(dataset[pressure_axis].attrs.get('units',''), pressure_unit, 'pressure')
            pscale, poffset = unit_transform(_units(pressure_unit), 'Pa')
            levels = np.asarray(dataset[pressure_axis].values, dtype=float)*pscale+poffset
            if not np.isfinite(levels).all() or len(set(levels)) != len(levels) or not np.all((levels > 0)&(levels <= 110000)):
                raise SchemaError('等压层需唯一有效0..110000Pa；不把sigma/模型层当气压')
            if len(levels)>1 and not (np.all(np.diff(levels)>0) or np.all(np.diff(levels)<0)):
                raise SchemaError('等压层必须单调')
            level_sets[label] = levels.tolist()
        else:
            # Sea-level pressure uses the special WPS level 201300, not surface 200100.
            levels = [201300. if label == 'PMSL' else 200100.]
            values = values[:, None, :, :]
        if ((values!=0)&(values.astype('float32')==0)).any():
            raise SchemaError(f'{label}float32下溢，不能静默丢为0')
        if not np.isfinite(values.astype('float32')).all():
            raise SchemaError(f'{label}不能安全编码为float32')
        prepared.append((label, target_units, description, levels, values))
        transformations[label] = {'source_variable': variable, 'source_units': source_units,
                                  'target_units': target_units, 'scale': scale, 'offset': offset}
    if 'UU' in mappings and level_sets['UU'] != level_sets['VV']:
        raise SchemaError('UU/VV等压层不一致')
    if not projected:
        radius = config.get('earth_radius_km', 6367.47)
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not np.isfinite(radius) or not 6300 <= radius <= 6400:
            raise SchemaError('earth_radius_km需明确为6300..6400之间的有限数值（km）')
        latitude = np.sort(latitude)
        longitude = np.sort(longitude)
        ny, nx = len(latitude), len(longitude)
        projection = _text('SWCORNER', 8)+struct.pack('>5f', latitude[0], longitude[0], latitude[1]-latitude[0], longitude[1]-longitude[0], radius)
        projection_info['earth_radius_km'] = radius
        projection_info['earth_radius_source'] = 'configuration' if 'earth_radius_km' in config else 'legacy default; verify against target project'
    files, float_errors = {}, {}
    for i, t in enumerate(times):
        chunks = []
        for label, units, desc, levels, values in prepared:
            for j, level in enumerate(levels):
                slab = values[i, j]
                header = (_text(t.strftime('%Y-%m-%d_%H:%M:%S'), 24)+struct.pack('>f', 0.)+
                          _text(source, 32)+_text(label, 9)+_text(units, 25)+_text(desc, 46)+
                          struct.pack('>fiii', level, nx, ny, iproj))
                # WPS calls this is_wind_grid_rel: false means earth-relative.
                chunks.extend(_record(payload) for payload in (struct.pack('>i', 5), header, projection,
                              struct.pack('>i', 0), slab.astype('>f4').tobytes(order='C')))
                error = float(np.max(np.abs(slab-slab.astype('f4').astype('f8'))))
                float_errors[label] = max(float_errors.get(label, 0.), error)
        files[f'{prefix}_{t:%Y-%m-%d_%H}.wps'] = b''.join(chunks)
    files['README_WPS_atmospheric_fields.txt'] = (
        f'WPS 4.6 intermediate v5 / IPROJ={iproj}.\n'
        f'On WPS host rename {prefix}_YYYY-MM-DD_HH.wps to {prefix}:YYYY-MM-DD_HH.\n'
        f'Append {prefix} to existing metgrid fg_name and retain all required sources.\n'
        'Dates, original levels and valid physical values are preserved; no time/vertical/spatial interpolation.\n'
        'Earth-relative winds remain earth-relative (is_wind_grid_rel=false).\n'
        'These mapped fields are not a complete WRF input chain. Required terrain/geogrid,\n'
        'land use, compatible METGRID.TBL, surface/soil fields not supplied in this conversion and real.exe configuration\n'
        'must still be supplied. Availability depends on selected physics and domain.\n'
        'Official metgrid, real.exe and wrf.exe execution is not established by file export.\n'
    ).encode('ascii')
    report = {'scope': f'WPS4.6 intermediate-v5 mapped pressure/surface/soil fields; IPROJ={iproj}',
              'projection':projection_info,
              'complete_model_input': False, 'native_model_input_verified': False,
              'fields': list(mappings), 'pressure_levels_Pa': level_sets, 'shape_y_x': [ny, nx],
              'timestamps_UTC': [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in times],
              'interval_seconds': interval, 'records_per_time': sum(len(row[3]) for row in prepared),
              'time_processing': 'none; source timestamps retained', 'wind_reference': wind_frame,
              'unit_transformations': transformations, 'float32_max_absolute_error': float_errors,
              'engineering_bounds_source': 'Provisional pre-conversion engineering checks; not WPS acceptance thresholds',
              'missing_pressure_fields_for_common_WRF_chain': sorted(set(['TT','UU','VV','RH','GHT'])-set(mappings)),
              'missing_surface_fields_for_common_WRF_chain': sorted(set(['PSFC','SKINTEMP','SOILHGT','LANDSEA'])-set(mappings)),
              'missing_soil_layers_for_common_WRF_chain': sorted({f'{p}{a:03d}{z:03d}' for a,z in SOIL_DEPTHS[:4] for p in ('ST','SM')}-set(mappings)),
              'unimplemented_required_components': [
                  'geogrid/geo_em terrain and land use', 'METGRID.TBL + metgrid.exe',
                  'real.exe initialization, boundary configuration, wrf.exe'],
              'format_sources': ['https://github.com/wrf-model/WPS/blob/v4.6.0/ungrib/src/output.F',
                  'https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/read_met_module.F',
                  'https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/METGRID.TBL.ARW']}
    return files, report

