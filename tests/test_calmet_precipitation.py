import json,copy
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np,pandas as pd,pytest
from ecodata.models.adapters.calmet_precipitation import export_precipitation
from ecodata.core.exceptions import SchemaError
BASE=Path(__file__).parents[1]/'examples/ai_round1/calpuff/precipitation'
def load(real=True):
    a,c=('A_NOAA_USCRN_20210701.csv','config_NOAA.json') if real else ('A_synthetic_two_station.csv','config_synthetic.json')
    return pd.read_csv(BASE/a),json.loads((BASE/c).read_text(encoding='utf8'))
def independent(data):
    lines=data.decode().splitlines();assert lines[0][:16].strip()=='PRECIP.DAT' and lines[0][16:32].strip()=='2.0'
    n=int(lines[1]);i=2+n;assert lines[i]=='NONE    ';header=list(map(int,lines[i+1].split()));ids=list(map(int,lines[i+2].split()));assert len(ids)==header[-1]
    rows=[list(map(float,l.split())) for l in lines[i+3:]];assert all(len(r)==3+len(ids) for r in rows)
    return header,ids,np.array(rows)

def test_real_against_noaa_raw_not_writer():
    frame,cfg=load();files,report=export_precipitation(frame,cfg);header,ids,rows=independent(files['PRECIP.DAT'])
    raw=[l.split() for l in (BASE/'raw_USCRN_20210701.txt').read_text().splitlines()]
    assert len(rows)==24 and ids==[int(raw[0][0])] and header[6]==7
    for output,original in zip(rows,raw):
        t=datetime.strptime(original[1]+original[2],'%Y%m%d%H%M')-timedelta(hours=7)
        assert output[:3].tolist()==[t.year,int(t.strftime('%j')),t.hour]
        assert output[3]==float(original[12])
    assert rows[:,3].sum()==sum(float(r[12]) for r in raw)
    assert report['time_processing']=='none' and report['time_label']=='hour_end'

def test_different_rate_units_columns_stations_leap_day():
    frame,cfg=load(False);files,_=export_precipitation(frame,cfg);header,ids,rows=independent(files['PRECIP.DAT'])
    assert ids==[92,91] and header[6]==-8
    np.testing.assert_allclose(rows,[[2024,60,23,3,0],[2024,61,0,1,2]],rtol=0,atol=1e-12)

@pytest.mark.parametrize('problem',['missing','duplicate','unknown_station','nan','negative','marker','bool','unit','no_timezone','fraction_hour','time_gap','quantity','daily','station_order','version','underflow'])
def test_reject_without_fill(problem):
    f,c=load(False);v=c['fields']['precipitation'];t=c['fields']['time'];s=c['fields']['station']
    if problem=='missing':f=f.iloc[1:]
    elif problem=='duplicate':f=pd.concat([f,f.iloc[[0]]])
    elif problem=='unknown_station':f.loc[0,s]=999
    elif problem=='nan':f.loc[0,v]=np.nan
    elif problem=='negative':f.loc[0,v]=-1
    elif problem=='marker':f.loc[0,v]=-9999
    elif problem=='bool':f[v]=True
    elif problem=='unit':c['source_units']='kg'
    elif problem=='no_timezone':f[t]=f[t].str.replace('Z','')
    elif problem=='fraction_hour':f.loc[0,t]='2024-02-29T16:30:00Z'
    elif problem=='time_gap':f.loc[f[t].str.contains('16:'),t]='2024-02-29T17:00:00Z'
    elif problem=='quantity':c['quantity']='instantaneous'
    elif problem=='daily':c.update(quantity='one_hour_amount',source_units='mm',accumulation_seconds=86400)
    elif problem=='station_order':c['station_ids']=[91,91]
    elif problem=='version':c['version']='CALMET-7'
    elif problem=='underflow':f[v]=1e-60
    with pytest.raises(SchemaError):export_precipitation(f,c)
