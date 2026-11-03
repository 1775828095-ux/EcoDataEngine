"""Local September availability check, including contents not just filenames."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from run_real_pm25_three_source import SPLIT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not (a.output/'audit.json').exists():
        raise RuntimeError('Finish ablations and seed testing before September preparation')
    listing=subprocess.run(['rg','--files',r'D:\codex','-g','*.nc','-g','*.nc4','-g','*.parquet',
        '-g','*201909*.csv','-g','*2019-09*.csv','-g','*201909*.grib*'],capture_output=True,text=True,encoding='utf-8',check=True).stdout.splitlines()
    # Exclude generated experiments and environment fixtures from real input availability.
    paths=[Path(s) for s in listing if '/output/' not in s.replace(chr(92),'/').lower()
           and '/.venv/' not in s.replace(chr(92),'/').lower()]
    station_ids=set(pd.read_csv(SPLIT,dtype={'station_id':str}).station_id)
    records=[];available={'station':[],'geos':[],'era5':[]}
    for path in paths:
        record={'file':str(path),'inspection':'metadata only','september_records':0,'kind':'other'}
        try:
            if path.suffix.lower() in ('.nc','.nc4'):
                with xr.open_dataset(path,engine='h5netcdf') as ds:
                    variables=set(ds.data_vars)
                    if 'pm25_rh35_gcc' in variables:
                        record['kind']='geos'
                    elif {'t2m','d2m','u10','v10','sp','tp'}.issubset(variables):
                        record['kind']='era5'
                    axis=next((name for name in ['time','valid_time'] if name in ds.coords),None)
                    record['variables']=','.join(sorted(variables))
                    if axis:
                        time=pd.to_datetime(ds[axis].values)
                        if len(time):
                            record.update(time_min=str(time.min()),time_max=str(time.max()),
                                september_records=int(((time>='2019-09-01') & (time<'2019-10-01')).sum()))
                    record['units']=json.dumps({v:ds[v].attrs.get('units') for v in variables},ensure_ascii=False)
            elif path.suffix.lower()=='.parquet':
                frame=pd.read_parquet(path)
                axis=next((name for name in ['geos_average_center_utc','time','timestamp'] if name in frame),None)
                if axis:
                    time=pd.to_datetime(frame[axis])
                    count=int(((time>='2019-09-01') & (time<'2019-10-01')).sum())
                    record.update(time_min=str(time.min()),time_max=str(time.max()),september_records=count)
                if 'station_id' in frame:
                    present=station_ids.intersection(frame.station_id.astype(str))
                    record.update(kind='station',matched_station_ids=len(present))
            elif path.suffix.lower()=='.csv':
                header=pd.read_csv(path,nrows=0)
                matched=station_ids.intersection(set(header.columns.astype(str)))
                record.update(kind='station' if matched else 'city_or_unmatched_csv',matched_station_ids=len(matched),
                    inspection='header check against fixed 60 station IDs',september_records=1 if matched else 0)
            else:
                record['inspection']='format needs separate parsing; no usable matched source established'
            if record['kind'] in available and record['september_records']:
                available[record['kind']].append(str(path))
        except Exception as e:
            record['inspection']='unreadable: '+str(e)
        records.append(record)
    pd.DataFrame(records).to_csv(a.output/'september_data_inventory.csv',index=False,encoding='utf-8-sig')
    complete=all(available.values())
    description='检查了本地候选NetCDF/Parquet的内部时间范围，以及9月CSV表头是否包含固定60站的ID。'
    if not complete:
        description+='本地9月城市平均空气质量CSV存在，但没有固定60站的9月观测；所找到的GEOS-CF和ERA5真实区域输入仅覆盖8月。9月兼容三源输入不齐，不能开展跨月份直接推理或微调。'
    status={'status':'INPUTS_NEED_COMPATIBILITY_CHECK' if complete else 'BLOCKED_MISSING_SEPTEMBER_INPUTS',
        'description':description,'available_candidates':available,
        'search_scope':r'D:\codex (real input files; generated experiment outputs excluded)',
        'cross_month_direct_executed':False,'cross_month_finetune_executed':False,
        'city_data_used_as_station_proxy':False,'inspected_files':len(records)}
    (a.output/'september_data_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(status)


if __name__=='__main__':main()
