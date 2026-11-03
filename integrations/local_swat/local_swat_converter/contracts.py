"""The station/period contract of the bound SWAT+ engineering copy."""
from pathlib import Path
import json


def requirements():
    cfg=json.loads(Path(__file__).with_name('binding.json').read_text(encoding='utf8'))
    return dict(schema='ecodata.converter.requirements.v1', model_id='swat_plus',
        role='weather', scope=cfg['scope'], crs='EPSG:4326', spatial_gate=True,
        fields=[dict(name=name, unit=unit, aliases=aliases, required=True, data_type='continuous')
                for name,unit,aliases in [('precipitation','mm/day',['pcp','prcp']),
                    ('air_temperature_max','degC',['tmax']),('air_temperature_min','degC',['tmin'])]],
        coordinates={'time':dict(datetime=True, dimension=True, interval_seconds=86400,
                                   start=cfg['start'], end=cfg['end']),
                     'lat':dict(size=1, value=cfg['latitude']),
                     'lon':dict(size=1, value=cfg['longitude'])},
        coordinate_aliases={'latitude':'lat','longitude':'lon'},
        reference_fields=['lat','lon'], required_references=[], metadata={},
        dependencies=[dict(name='engineering', path=cfg['project_zip'], sha256=cfg['project_sha256'])],
        deferred_checks=['站点天气引用、逐日文本独立读回、原工程哈希；原生运行与转换分开记录'])
