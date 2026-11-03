"""Acquire an explicitly selected ERA5 product; no station values are uploaded."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import requests
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/robit_era5_1990_2013'


def fetch(zone):
    OUT.mkdir(parents=True,exist_ok=True);slug='local_utc3' if zone=='Africa/Addis_Ababa' else 'utc'
    target=OUT/(slug+'.json')
    params=dict(latitude=11.6,longitude=37.36,start_date='1990-01-01',end_date='2013-12-31',
        daily='precipitation_sum',timezone=zone,models='era5',cell_selection='nearest')
    if target.exists():payload=json.loads(target.read_text(encoding='utf-8'))
    else:
        response=requests.get('https://archive-api.open-meteo.com/v1/archive',params=params,timeout=(15,120))
        if response.status_code!=200:
            (OUT/(slug+'_failure.txt')).write_text(response.text,encoding='utf-8');response.raise_for_status()
        payload=response.json();target.write_bytes(response.content)
        (OUT/(slug+'_receipt.json')).write_text(json.dumps(dict(url=response.url,
            retrieved_utc=datetime.now(timezone.utc).isoformat(),parameters=params,status=response.status_code,
            sha256=hashlib.sha256(response.content).hexdigest()),indent=2),encoding='utf-8')
    assert payload['daily_units']['precipitation_sum']=='mm'
    frame=pd.DataFrame(dict(time=pd.to_datetime(payload['daily']['time']),precipitation=payload['daily']['precipitation_sum']))
    assert len(frame)==8766 and np.isfinite(frame.precipitation).all() and (frame.precipitation>=0).all()
    assert frame.time.equals(pd.Series(pd.date_range('1990-01-01','2013-12-31'),name='time'))
    frame.to_csv(OUT/(slug+'.csv'),index=False)
    print(slug,len(frame),'grid',payload['latitude'],payload['longitude'],'utc_offset',payload['utc_offset_seconds'],flush=True)


if __name__=='__main__':
    fetch('Africa/Addis_Ababa');fetch('GMT')
