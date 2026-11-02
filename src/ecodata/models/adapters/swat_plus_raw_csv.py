"""Explicit CSV selection/alignment; never modify source bytes or fill values."""
import csv, io, hashlib, json
import pandas as pd
import numpy as np
import xarray as xr
from ecodata.models.adapters.swat_plus_upload import export_uploaded_weather


def read_csv(raw):
    text=raw.decode('utf-8-sig')
    rows=list(csv.reader(io.StringIO(text)))
    if not rows or len(set(rows[0]))!=len(rows[0]) or any(len(r)!=len(rows[0]) for r in rows[1:]):
        raise ValueError('CSV空表、重复字段或记录列数错误')
    return pd.DataFrame(rows[1:],columns=rows[0])


def convert_csv(rain_raw,temp_raw,selection,config):
    frames=[]; sources=[]
    dates=pd.date_range(config['start'],config['end'],freq='D')
    if not len(dates):raise ValueError('起止日期倒置')
    for role,raw in [('rain',rain_raw),('temperature',temp_raw)]:
        spec=selection[role];frame=read_csv(raw); total=len(frame)
        if len(set(spec['values'].values()))!=len(spec['values']):raise ValueError('最高温和最低温不能选择同一字段')
        frame=frame.loc[frame[spec['object_field']]==spec['object_value']].copy()
        if frame.empty:raise ValueError(role+'没有所选对象')
        parsed=pd.to_datetime(frame[spec['date_field']],format='%Y-%m-%d',errors='raise')
        frame.index=parsed
        frame=frame.loc[((parsed>=dates[0]) & (parsed<=dates[-1])).to_numpy()].sort_index()
        if frame.index.has_duplicates:raise ValueError(role+'所选对象存在重复日期')
        if not frame.index.equals(dates):raise ValueError(role+'所选时段日期缺失；不会补值')
        for field in spec['values'].values():
            v=pd.to_numeric(frame[field],errors='raise')
            if not np.isfinite(v).all():raise ValueError(role+'存在缺测或非有限数值')
        frames.append(frame)
        sources.append(dict(role=role,sha256=hashlib.sha256(raw).hexdigest(),source_rows=total,selected_rows=len(frame),selection=spec))
    if selection['rain']['unit']!='mm/day' or selection['temperature']['unit']!='degC':raise ValueError('当前需明确日降水mm/day和温度degC，不隐式换算')
    variables={}
    for role,frame in zip(('rain','temperature'),frames):
        for target,field in selection[role]['values'].items():
            variables[target]=('time',pd.to_numeric(frame[field]).to_numpy(dtype=float),{'units':'mm/day' if target=='precipitation' else 'degC'})
    ds=xr.Dataset(variables,coords={'time':dates},attrs={'time_standard':'source daily date labels; no timezone conversion'})
    nc=ds.to_netcdf(engine='scipy')
    files,report=export_uploaded_weather(nc,config)
    audit=dict(sources=sources,selection=selection,config=config,policy='Explicit object/date filtering and exact daily one-to-one alignment; no filling, averaging, interpolation or source modification',selected_source_records={role:frame.reset_index(names='selected_date').to_dict(orient='records') for role,frame in zip(('rain','temperature'),frames)})
    files['raw_csv_provenance.json']=json.dumps(audit,ensure_ascii=False,indent=2,default=str).encode()
    report['raw_csv_sources']=sources
    report['raw_csv_policy']=audit['policy']
    files['conversion_report.json']=json.dumps(report,ensure_ascii=False,indent=2).encode()
    return files,report,nc
