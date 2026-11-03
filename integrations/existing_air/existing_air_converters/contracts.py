"""Requirements derived from the same pinned configuration used by the writer."""
from pathlib import Path
import hashlib
import json


def requirements(model):
    bindings = json.loads(Path(__file__).with_name('bindings.json').read_text(encoding='utf8'))
    binding = bindings['models'][model]
    config_path = Path(binding['configuration'])
    if not config_path.is_file():
        raise ValueError('缺少转换配置：'+str(config_path))
    with config_path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != binding['configuration_sha256']:
            raise ValueError('转换配置哈希变化，拒绝用未知配置匹配数据')
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    result = dict(schema='ecodata.converter.requirements.v1', model_id=model,
        scope=binding['scope'], fields=[], coordinates={}, role='emissions',
        dependencies=[dict(name='configuration', path=str(config_path), sha256=binding['configuration_sha256']),
                      dict(name='engineering', path=binding['engineering'], sha256=binding.get('engineering_sha256')),
                      dict(name='legacy_python', path=bindings['legacy_python']),
                      dict(name='legacy_source', path=bindings['legacy_source'])],
        metadata={}, reference_fields=[], required_references=[],
        deferred_checks=['原生 writer 校验维度、时标、网格及内部引用', '工程副本核对；本次不启动原生模型'])
    def field(name, unit, aliases=()):
        result['fields'].append(dict(name=name, unit=unit, aliases=list(aliases), data_type='continuous', required=True))
    def axis(name, **rule):
        result['coordinates'][name] = rule
        result['reference_fields'].append(name)
    if model == 'wrf':
        result.update(role='meteorological_analysis', crs=cfg['crs'], spatial_gate=True)
        aliases = {'air':['air_temperature'], 'uwnd':['eastward_wind'],
                   'vwnd':['northward_wind'], 'hgt':['geopotential_height']}
        units = {'TT':'K', 'UU':'m/s', 'VV':'m/s', 'GHT':'m'}
        for native in cfg['required_fields']:
            name = cfg['fields'][native]['variable']; field(name, units[native], aliases.get(name, []))
        coords = cfg['coordinates']
        axis(coords['time'], datetime=True, interval_seconds=cfg['interval_seconds'], dimension=True)
        for name in ('pressure','latitude','longitude'):
            axis(coords[name], unit=cfg['coordinate_units'][name], dimension=True)
        result['metadata'] = dict(wind_reference=cfg['wind_reference'], time_basis=cfg['time_basis'])
    elif model == 'camx':
        result['spatial_gate'] = False  # Native row/col indices: separate explicit contract below.
        for species in cfg['species']:
            field(species['column'], species['unit'])
        grid=cfg['grid']; base=cfg['row_col_base']
        axis(cfg['fields']['time'], datetime=True, interval_seconds=3600, dimension=True)
        axis(cfg['fields']['row'], size=grid['nrows'], start=base, step=1, dimension=True)
        axis(cfg['fields']['col'], size=grid['ncols'], start=base, step=1, dimension=True)
        result['metadata'] = dict(time_basis=cfg['time_basis'], grid=grid)
    elif model == 'cmaq':
        result.update(crs=cfg['source_crs'], spatial_gate=False)
        for species, item in cfg['fields'].items():
            field(item['source'], item['source_units'], [species])
        grid=cfg['grid']; axes=cfg['axes']
        axis(axes['time'], datetime=True, interval_seconds=3600, dimension=True)
        axis(axes['x'], size=grid['NCOLS'], unit=cfg['coordinate_units'],
             start=grid['XORIG']+grid['XCELL']/2, step=grid['XCELL'], dimension=True)
        axis(axes['y'], size=grid['NROWS'], unit=cfg['coordinate_units'],
             start=grid['YORIG']+grid['YCELL']/2, step=grid['YCELL'], dimension=True)
        result['metadata'] = dict(source_timezone=cfg['source_timezone'])
    elif model == 'calpuff':
        result.update(role='receptors', crs=cfg['source_crs'], spatial_gate=True)
        for name in ('elevation','height_agl'):
            field(cfg['fields'][name], cfg['units'][name])
        for name in ('x','y'):
            axis(cfg['fields'][name], unit=cfg['units'][name])
        identifier=cfg['fields']['id']
        result['reference_fields'].append(identifier)
        result['required_references'] = [identifier]
        result['metadata'] = dict(elevation_vertical_reference=cfg['elevation_vertical_reference'])
    return result
