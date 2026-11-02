"""Adapters call upstream parsers, not duplicated RDB/ISD character parsers."""
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle


def _bundle(frame, attrs):
    from ecodata.io.structures import table_dataset
    ds = table_dataset(frame)
    ds.attrs.update(attrs)
    return EcoDataBundle(ds)


class USGSRDBReader:
    def read(self, source):
        return self.read_bytes(Path(source).read_bytes())

    def read_bytes(self, content):
        from dataretrieval.rdb import read_rdb
        text = content.decode('utf-8-sig')
        frame = read_rdb(text, dtypes={'site_no':str})
        if frame.empty:
            raise SchemaError('USGS RDB 没有数据记录')
        metadata = '\n'.join(x for x in text.splitlines() if x.startswith('#'))
        attrs = dict(reader_backend='dataretrieval.rdb.read_rdb', source_metadata=metadata,
                     source_sha256=hashlib.sha256(content).hexdigest())
        if 'datetime' not in frame:
            ds = _bundle(frame, dict(attrs, data_role='quality_auxiliary')).dataset
            if 'site_no' in ds:ds['site_no'].attrs['standard_name']='station_id'
            return EcoDataBundle(ds)
        measures = [c for c in frame if re.fullmatch(r'\d+_\d{5}(?:_\d{5})?',c)]
        if not measures:
            raise SchemaError('USGS RDB 未找到参数代码列')
        rows=[]
        for name in measures:
            code=name.split('_')[1]
            values=pd.to_numeric(frame[name],errors='coerce')
            # Only documented parameter 00060 has a verified unit here.
            part=pd.DataFrame(dict(station_id=frame.site_no,parameter_code=code,datetime=frame.datetime,
                                   timezone=frame.get('tz_cd','unknown_local_day'),value=values,
                                   unit='ft^3/s' if code=='00060' and 'cubic feet per second' in text else '',
                                   quality_code=frame.get(name+'_cd',''),source_parameter_column=name,
                                   raw_value=frame[name].astype(str)))
            rows.append(part)
        result=pd.concat(rows,ignore_index=True)
        result['time']=pd.to_datetime(result.datetime,errors='raise')
        daily='tz_cd' not in frame
        daily_mean=daily and all(name.endswith('_00003') for name in measures)
        if not daily:
            offsets={'EST':-5,'EDT':-4,'UTC':0,'GMT':0}
            if not result.timezone.isin(offsets).all():
                raise SchemaError('USGS 存在尚未支持的原始时区；禁止猜测时间偏移')
            result['time']=result.time-pd.to_timedelta(result.timezone.map(offsets),unit='h')
        ds=_bundle(result,dict(attrs,time_zone='local_calendar_day_unconfirmed' if daily else 'UTC',
                              temporal_semantics=('mean' if daily_mean else 'unknown') if daily else 'instantaneous',
                              time_semantics_note='daily statistic retained in source_parameter_column; no artificial UTC midnight' if daily else 'converted using explicit tz_cd; source datetime/timezone retained',
                              data_role='tributary_inflow' if set(result.parameter_code)=={'00060'} else 'unclassified')).dataset
        units=set(result.unit)-{''}
        if len(units)==1:ds['value'].attrs.update(units=next(iter(units)),standard_name='water_volume_flow_rate' if set(result.parameter_code)=={'00060'} else '',ancillary_variables='quality_code')
        return EcoDataBundle(ds)


class NOAAISDReader:
    def read(self, source):
        return self.read_bytes(Path(source).read_bytes())

    def read_bytes(self, content):
        from ish_parser import ish_report
        from ish_parser.ish_report import ish_reportException
        raw=gzip.decompress(content) if content.startswith(b'\x1f\x8b') else content
        rows=[]; failures=[]
        mapping={'temperature':('air_temperature','degC'),'wind_speed':('wind_speed','m/s'),
                 'wind_direction':('wind_direction','degree'),'pressure':('sea_level_pressure','hPa'),
                 'dew_point':('dew_point','degC'),'cloud_ceiling':('sky_ceiling','m')}
        for i,line in enumerate(raw.decode('ascii').splitlines()):
            if not line:continue
            report=ish_report()
            try:report.loads(line)
            except (Exception, ish_reportException) as exc:
                failures.append(dict(line=i+1,error=str(exc)))
                continue
            row=dict(station=report.weather_station+report.wban,time=report.datetime.replace(tzinfo=None),
                     latitude=report.latitude,longitude=report.longitude,raw_record=line,
                     raw_additional=json.dumps(report._additional,default=str),raw_remarks=json.dumps(report._remarks,default=str))
            for name,(attribute,unit) in mapping.items():
                observation=getattr(report,attribute,None)
                if observation is not None:
                    row[name]=observation.get_numeric()
                    row[name+'_quality_code']=str(observation._obs_quality)
            for key,extra in report._additional.items():
                if key.startswith('AA') and hasattr(extra,'precipitation'):
                    row['precipitation_'+key]=extra.precipitation['depth'].get_numeric()
                    row['precipitation_'+key+'_hours']=extra.precipitation['hours']
                if key=='GF1' and hasattr(extra,'sky_condition_observation'):
                    cloud=extra.sky_condition_observation.get('total_coverage')
                    if cloud is not None:row['cloud_coverage']=cloud.get_numeric()
            rows.append(row)
        if not rows:raise SchemaError('NOAA ISD 没有可读取的原始记录')
        ds=_bundle(pd.DataFrame(rows),dict(reader_backend='ish-parser 0.0.25',data_role='forcing',time_zone='UTC',
                 crs='EPSG:4326',temporal_semantics='instantaneous',source_sha256=hashlib.sha256(content).hexdigest(),
                 reader_evidence=json.dumps(dict(source_records=len(raw.splitlines()),parsed_records=len(rows),failed_records=failures),ensure_ascii=False),
                 reader_status='WARN' if failures else 'PASS')).dataset
        ds.attrs['additional_quality_status']='not independently decoded; original AA/GF sections retained in raw_record; no invented quality flags'
        for name,(_,unit) in mapping.items():
            if name in ds:ds[name].attrs.update(units=unit,ancillary_variables=name+'_quality_code')
        for name in ds:
            if name.startswith('precipitation_AA') and not name.endswith('_quality_code'):
                ds[name].attrs.update(units='h' if name.endswith('_hours') else 'mm',temporal_semantics='duration' if name.endswith('_hours') else 'accumulation')
        if 'cloud_coverage' in ds:ds.cloud_coverage.attrs['units']='okta'
        return EcoDataBundle(ds)
