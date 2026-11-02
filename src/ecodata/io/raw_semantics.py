"""Explicit aliases and source evidence only; no value-range unit guesses."""
import json
from pathlib import Path

import pandas as pd
import xarray as xr


def enrich_raw_semantics(dataset):
    config=json.loads(Path(__file__).with_name('monitoring_semantics.json').read_text(encoding='utf8'))
    ds=dataset.copy()
    aliases={alias.casefold():name for name,names in config['variables'].items() for alias in names}
    for name in list(ds.data_vars):
        source_name=str(ds[name].attrs.get('source_field_name',name))
        physical=aliases.get(name.casefold()) or aliases.get(source_name.casefold())
        for rule in config.get('unit_qualified_variables',[]):
            if source_name in rule['headers'] and ds[name].attrs.get('units') in rule['units']:physical=rule['physical_variable']
        if not physical:continue
        ds[name].attrs['physical_variable']=physical
        unit=ds[name].attrs.get('units','')
        if unit in {'C','°C'}:ds[name].attrs.update(units='degC',source_units=unit)
        if unit=='mg/l':ds[name].attrs.update(units='mg/L',source_units=unit)
        if unit in {'ppb-N','ppb-P'}:
            ds[name].attrs.update(units='ug/L',source_units=unit,mass_basis='N' if unit.endswith('N') else 'P',unit_evidence='explicit aqueous concentration header; elemental mass basis retained')
        if ds[name].dtype.kind in 'OUS':
            raw=pd.Series(ds[name].values.ravel()).astype(str).str.strip()
            numeric=pd.to_numeric(raw,errors='coerce')
            if numeric.notna().any() and not numeric[raw.ne('')].notna().all():
                key=name+'__numeric'
                if key in ds:continue
                ds[key]=xr.DataArray(numeric.to_numpy(dtype=float).reshape(ds[name].shape),dims=ds[name].dims,attrs=dict(ds[name].attrs,source_field_name=name,ancillary_variables=name,interpretation='only exact numeric measurements; censored/nonreportable values remain NaN; original tokens retained'))
                ds[name].attrs.update(data_kind='quality_metadata',physical_variable='raw_'+physical)
    available=list(ds.data_vars)
    if 'time' not in ds:
        date=next((x for x in available if x.casefold() in {'datetime','date/time','datetimestamp','collection date','date time'}),None)
        if date:
            source=pd.Series(ds[date].values.ravel())
            parsed=pd.to_datetime(source,format='mixed',errors='coerce')
            if date=='Collection Date' and 'Collection Time' in ds:
                clocks=pd.Series(ds['Collection Time'].values).map(lambda value: value.strftime('%H:%M:%S') if hasattr(value,'strftime') else str(value))
                parsed=pd.to_datetime(source.astype(str).str.split().str[0]+' '+clocks,format='mixed',errors='coerce')
            if parsed.notna().all():
                ds['time']=xr.DataArray(parsed.to_numpy(dtype='datetime64[ns]'),dims=ds[date].dims,attrs=dict(source_field_name=date))
                ds.attrs.setdefault('time_zone','unknown; requires confirmation')
    physicals={ds[x].attrs.get('physical_variable','') for x in ds}
    if physicals & set(config['variables']):
        ds.attrs.setdefault('data_role','water_quality_observation')
        ds.attrs['role_evidence']='explicit water-quality header aliases; observation purpose not inferred'
    return ds
