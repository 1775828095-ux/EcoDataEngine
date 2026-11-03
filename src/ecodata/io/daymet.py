"""Documented Daymet single-pixel precipitation CSV, without invented leap days."""
import io
import re
import json
from hashlib import sha256
import numpy as np
import pandas as pd
import xarray as xr


def read_daymet_precipitation(content):
    text=content.decode('utf-8-sig');lines=text.splitlines()
    match=re.fullmatch(r'Latitude:\s*([-+\d.]+)\s+Longitude:\s*([-+\d.]+)\s*',lines[0])
    if not match:raise ValueError('Daymet query coordinates absent')
    lat,lon=map(float,match.groups())
    if not -90<=lat<=90 or not -180<=lon<=180:raise ValueError('Invalid Daymet query coordinates')
    header=next((i for i,l in enumerate(lines) if l.startswith('year,yday,')),None)
    if header is None or not any('Daymet' in l for l in lines[:header]):raise ValueError('Daymet format evidence absent')
    frame=pd.read_csv(io.StringIO(text),skiprows=header)
    if list(frame.columns)!=['year','yday','prcp (mm/day)']:
        raise ValueError('This Daymet reader supports precipitation-only CSV; other variables need their own physical contracts')
    if frame.isna().any().any() or not np.isfinite(frame.values).all():raise ValueError('Invalid Daymet numeric records')
    if not np.equal(frame[['year','yday']],np.floor(frame[['year','yday']])).all().all():raise ValueError('Integer Daymet year/day required')
    years=sorted(frame.year.astype(int).unique())
    if not years or years!=list(range(years[0],years[-1]+1)):raise ValueError('Consecutive Daymet years required for a daily window')
    for _,part in frame.groupby('year',sort=False):
        if part.yday.astype(int).tolist()!=list(range(1,366)):raise ValueError('Each Daymet year must contain exactly ordered days1..365')
    dates=pd.to_datetime(frame.year.astype(int).astype(str)+'-01-01')+pd.to_timedelta(frame.yday-1,unit='D')
    if (frame['prcp (mm/day)']<0).any():raise ValueError('Negative Daymet precipitation')
    full=pd.date_range(f'{years[0]}-01-01',f'{years[-1]}-12-31')
    values=pd.Series(frame['prcp (mm/day)'].values,index=dates).reindex(full).to_numpy()
    citation=next((l for l in lines[:header] if l.startswith('How to cite:')),None)
    if citation is None:raise ValueError('Daymet product/version citation required to bind training and application sources')
    ds=xr.Dataset(dict(precipitation=(('time','lat','lon'),values[:,None,None])),
        coords=dict(time=full,lat=[lat],lon=[lon]),attrs=dict(crs='EPSG:4326',
            source_id=f'daymet_query:{lat:.8f}:{lon:.8f}:'+sha256(citation.encode()).hexdigest()[:12],
            geometry_kind='extracted_cell_series',pixel_selection='nearest_native_cell',
            extraction_latitude=lat,extraction_longitude=lon,
            extraction_evidence='https://daymet.ornl.gov/single-pixel-tool-guide',
            coordinate_role='extraction query; native cell centre/footprint not inferred',
            calendar_evidence='Daymet365 records/year; leap day retained, Dec31 absent in leap years',
            natural_missing_dates=json.dumps(full[~np.isfinite(values)].astype(str).tolist()),
            raw_daymet_header=json.dumps(lines[:header]),source_citation=citation))
    ds.precipitation.attrs=dict(units='mm/day',physical_variable='precipitation_amount',
        physical_definition='Daily precipitation water-equivalent depth',spatial_semantics='grid_estimate',
        temporal_semantics='accumulation',data_kind='continuous',native_resolution_m=1000)
    ds['time_bounds']=(('time','bounds'),np.stack([full.values,(full+pd.Timedelta(days=1)).values],axis=1))
    return ds
