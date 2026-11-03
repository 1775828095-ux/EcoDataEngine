"""New GHCN gauges + ERA5 hourly, aligned to Western Australia 9am days."""
from pathlib import Path
from datetime import datetime,timezone
import calendar,hashlib,json
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
INVENTORY=ROOT/'data/new_gauge_inventory_20261028'
OUT=ROOT/'data/wa_precipitation_2014_2024'
STATIONS=['ASN00009021','ASN00003003','ASN00012038']


def gauge(path):
    records=[]
    for line in path.read_text().splitlines():
        if line[17:21]!='PRCP' or not 2014<=int(line[11:15])<=2024:continue
        year,month=int(line[11:15]),int(line[15:17])
        for day in range(calendar.monthrange(year,month)[1]):
            cell=line[21+day*8:29+day*8];raw=int(cell[:5]);m,q,s=cell[5:8]
            valid=raw>=0 and q==' ' and m in ' BDFGT'
            records.append(dict(date=pd.Timestamp(year,month,day+1),precipitation=raw/10 if valid else np.nan,
                          raw_value=raw,mflag=m,qflag=q,sflag=s,valid=valid))
    frame=pd.DataFrame(records).set_index('date')
    assert frame.index.is_unique
    return frame.reindex(pd.date_range('2014-01-01','2024-12-31',name='date'))


def main():
    OUT.mkdir(exist_ok=True)
    metadata={line[:11]:line for line in (INVENTORY/'ghcnd-stations.txt').read_text().splitlines() if line[:11] in STATIONS}
    decisions=[]
    for path in (INVENTORY/'stations').glob('ET*.dly'):
        frame=gauge(path);coverage=frame.precipitation.notna().groupby(frame.index.year).mean()
        decisions.append(dict(id=path.stem,annual_valid_fraction=coverage.to_dict(),excluded=True,reason='Fails prespecified >=80% per-year coverage; never scored for algorithm selection'))
    for sid in STATIONS:
        folder=OUT/sid;folder.mkdir(exist_ok=True)
        original=INVENTORY/'stations'/(sid+'.dly')
        frame=gauge(original)
        coverage=frame.precipitation.notna().groupby(frame.index.year).mean()
        assert coverage.min()>=.8,(sid,coverage.to_dict())
        (folder/'gauge.dly').write_bytes(original.read_bytes())
        frame.to_csv(folder/'gauge.csv')
        line=metadata[sid];latitude=float(line[12:20]);longitude=float(line[21:30]);elevation=float(line[31:37])
        params=dict(latitude=latitude,longitude=longitude,start_date='2013-12-31',end_date='2024-12-31',
                    hourly='precipitation',models='era5',timezone='GMT',cell_selection='nearest')
        path=folder/'era5_hourly.json'
        if not path.exists():
            response=requests.get('https://archive-api.open-meteo.com/v1/archive',params=params,timeout=120)
            response.raise_for_status();path.write_bytes(response.content)
            (folder/'era5_receipt.json').write_text(json.dumps(dict(url=response.url,parameters=params,status=response.status_code,
                sha256=hashlib.sha256(response.content).hexdigest(),retrieved_utc=datetime.now(timezone.utc).isoformat()),indent=2))
        data=json.loads(path.read_text());assert data['hourly_units']['precipitation']=='mm'
        hourly=pd.Series(data['hourly']['precipitation'],index=pd.DatetimeIndex(data['hourly']['time']),dtype=float)
        assert hourly.index.is_unique and np.isfinite(hourly).all() and (hourly>=0).all()
        daily=hourly.resample('24h',origin='epoch',offset='1h',closed='right',label='right').sum(min_count=24)
        ends=frame.index+pd.Timedelta(hours=1);values=daily.reindex(ends)
        assert values.notna().all() and len(values)==4018
        pd.DataFrame(dict(time=ends,precipitation=values.to_numpy())).to_csv(folder/'era5_daily_9am.csv',index=False)
        meta=dict(id=sid,name=line[41:71].strip(),latitude=latitude,longitude=longitude,elevation=elevation,
            native_era5_latitude=data['latitude'],native_era5_longitude=data['longitude'],
            day_rule='24h ending 09:00 Australia/Perth = 01:00 UTC on gauge date; hourly values are preceding-hour sums',
            source_independence='Different production chain; not certified statistical independence or independent ground truth',
            annual_valid_fraction=coverage.to_dict(),gauge_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            era5_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_performance_inspected=False)
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
        print(sid,'VALID COVERAGE',float(coverage.min()),'ERA5 DAILY',len(values),flush=True)
    (OUT/'inventory_decisions.json').write_text(json.dumps(dict(selected=STATIONS,excluded=decisions,
        qa='PRCP tenths mm; nonnegative, blank QFLAG, MFLAG blank/B/D/F/G/T; no assumed zero/partial-day substitution',
        identifier_correction='ASN00001238 returned HTTP404; actual Broome identifier verified in GHCN station metadata as ASN00003003'),indent=2))


if __name__=='__main__':main()
