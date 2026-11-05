from pathlib import Path
from decimal import Decimal
from datetime import datetime,timedelta
from unittest.mock import patch
import io,json,zipfile
import numpy as np,pandas as pd,pytest,xarray as xr
from streamlit.testing.v1 import AppTest
from ecodata.round1 import convert
from ecodata.models.adapters.swat_plus import SWATPlusWriter
from ecodata.io.structures import read_structure,export_structure
from ecodata.core.exceptions import SchemaError

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'examples/ai_round1/swat_plus/power_2021'

def load():
    return (CASE/'A_power_2021.nc').read_bytes(),json.loads((CASE/'config_ui.json').read_text())

def independent_read(payload,width):
    # Independent of writer/adapter: SWAT+ 62.0.0 list-directed records.
    lines=payload.decode('utf-8').splitlines()
    assert lines[1].split()==['NBYR','TSTEP','LAT','LONG','ELEV']
    meta=list(map(Decimal,lines[2].split()));assert len(meta)==5 and meta[:2]==[1,0]
    rows=[l.split() for l in lines[3:]]
    assert len(rows)==365 and all(len(r)==width for r in rows)
    dates=[(datetime(int(r[0]),1,1)+timedelta(days=int(r[1])-1)).strftime('%Y%m%d') for r in rows]
    assert dates==pd.date_range('2021-01-01','2021-12-31').strftime('%Y%m%d').tolist()
    return meta,dates,[[Decimal(v) for v in r[2:]] for r in rows]

def test_public_power_values_and_references_independently():
    data,cfg=load();files,report=convert('swat_plus',data,cfg)
    raw=json.loads((CASE/'NASA_POWER_2021_original.json').read_text(),parse_float=Decimal)
    mapping={'pcp':['PRECTOTCORR'],'tem':['T2M_MAX','T2M_MIN'],'slr':['ALLSKY_SFC_SW_DWN'],'hmd':['RH2M'],'wnd':['WS2M']}
    for kind,codes in mapping.items():
        meta,dates,values=independent_read(files['station.'+kind],len(codes)+2)
        assert meta[2:]==[Decimal(str(cfg['station'][n])) for n in ['latitude','longitude','elevation']]
        for date,row in zip(dates,values):
            for code,v in zip(codes,row):
                expected=Decimal(str(raw['properties']['parameter'][code][date]))
                if code=='RH2M':expected/=100
                # Predeclared engineering threshold, not a model-run threshold.
                assert abs(v-expected)<=Decimal('1e-12'),(date,code,v,expected)
        index='tmp' if kind=='tem' else kind
        assert files[index+'.cli'].decode().splitlines()==[index+'.cli','FILENAME','station.'+kind]
    refs=files['weather-sta.cli'].decode().splitlines()[2].split()
    assert refs==['POWER_site','null','station.pcp','station.tem','station.slr','station.hmd','station.wnd','null','null']
    assert 'file.cio' not in files
    assert report['index_references_verified'] is True
    assert 'weather-wgn.cli' not in files and report['model_run_verified'] is False
    assert report['wind_height_m_from_source']==2 and report['time_standard_from_source']=='LST'

@pytest.mark.parametrize('field',['solar_radiation','relative_humidity','wind_speed'])
@pytest.mark.parametrize('problem',['missing','nan','negative'])
def test_optional_fields_reject_missing_and_invalid(field,problem):
    data,cfg=load();ds=read_structure('weather.nc',data)
    if problem=='missing':ds=ds.drop_vars(field)
    else:ds[field].values[0]=np.nan if problem=='nan' else -99
    bad,_,_=export_structure(ds,'NetCDF')
    with pytest.raises(SchemaError,match=field):convert('swat_plus',bad,cfg)

def test_units_ranges_dates_station_version_and_daylight_power_rejected():
    data,cfg=load()
    for field,unit,match in [('solar_radiation','W/m2','积分时段'),('relative_humidity','g/kg','单位需'),('wind_speed','knots','单位需')]:
        ds=read_structure('weather.nc',data);ds[field].attrs['units']=unit
        bad,_,_=export_structure(ds,'NetCDF')
        with pytest.raises(SchemaError,match=match):convert('swat_plus',bad,cfg)
    ds=read_structure('weather.nc',data);ds.relative_humidity.values[0]=101
    bad,_,_=export_structure(ds,'NetCDF')
    with pytest.raises(SchemaError,match='允许范围'):convert('swat_plus',bad,cfg)
    ds=read_structure('weather.nc',data).isel(time=slice(1,None));bad,_,_=export_structure(ds,'NetCDF')
    with pytest.raises(SchemaError,match='连续逐日'):convert('swat_plus',bad,cfg)
    with pytest.raises(SchemaError,match='62.0.0'):convert('swat_plus',data,{**cfg,'target_model_version':'61'})
    with pytest.raises(SchemaError,match='不一致'):convert('swat_plus',data,{**cfg,'station':{**cfg['station'],'latitude':0}})

def test_synthetic_unit_conversion_and_leap_day():
    ds=xr.Dataset(coords={'time':pd.date_range('2020-02-28','2020-03-01')})
    for name,v,u in [('precipitation',[0,1,2],'mm/day'),('air_temperature_max',[10,11,12],'degC'),('air_temperature_min',[0,1,2],'degC'),('solar_radiation',[0,1,2],'kWh/m2/day'),('relative_humidity',[0,.5,1],'1'),('wind_speed',[0,3.6,7.2],'km/h')]:
        ds[name]=xr.DataArray(v,dims='time',attrs={'units':u})
    data,_,_=export_structure(ds,'NetCDF');_,cfg=load();cfg.update(start='2020-02-28',end='2020-03-01')
    cfg['station'].update(latitude=39.67912345,longitude=-77.23456789,elevation=191.25)
    files,_=convert('swat_plus',data,cfg)
    for kind in ['pcp','tem','slr','hmd','wnd']:
        meta=list(map(Decimal,files['station.'+kind].decode().splitlines()[2].split()))
        assert meta[2:]==[Decimal(str(cfg['station'][n])) for n in ['latitude','longitude','elevation']]
    for kind,expected in [('slr',[0,3.6,7.2]),('hmd',[0,.5,1]),('wnd',[0,1,2])]:
        rows=np.loadtxt(io.StringIO(files['station.'+kind].decode()),skiprows=3)
        np.testing.assert_array_equal(rows[:,1],[59,60,61]);np.testing.assert_allclose(rows[:,2],expected,rtol=0,atol=1e-12)

def test_extended_real_ui_calls_writer_and_downloads_cli_bytes(tmp_path):
    good,cfg=load();expected,_=convert('swat_plus',good,cfg)
    folder=tmp_path/'inputs';folder.mkdir();bundle=folder/'request.zip'
    def submit(data,configuration):
        declaration=dict(schema='ecodata.model-task.v1',model_id='swat_plus',operation='weather',input='weather.nc',configuration=configuration)
        with zipfile.ZipFile(bundle,'w') as z:
            z.writestr('weather.nc',data);z.writestr('task.json',json.dumps(declaration))
    submit(good,cfg)
    original=SWATPlusWriter.write
    with patch.object(SWATPlusWriter,'write',autospec=True,side_effect=original) as spy:
        app=AppTest.from_file(str(ROOT/'src/ecodata/interface/app.py'),default_timeout=30).run()
        app.text_input[0].set_value(str(folder)).run()
        app.button[0].click().run()
        assert not app.exception and not app.error and spy.call_count==1
        result=app.session_state['workbench_result']['data']
        assert result['manifest']['status']=='COMPONENT_EXPORTED'
        with zipfile.ZipFile(result['archive']) as z:
            assert z.testzip() is None
            for name,payload in expected.items():assert z.read('components/'+name)==payload
        assert app.get('download_button')
        ds=read_structure('weather.nc',good).drop_vars('wind_speed')
        submit(export_structure(ds,'NetCDF')[0],cfg);app.run()
        assert not app.get('download_button')
        app.button[0].click().run()
        assert any('缺少字段：wind_speed' in x.value for x in app.error) and spy.call_count==1
        submit(good,{**cfg,'target_model_version':'61'});app.run()
        app.button[0].click().run()
        assert any('62.0.0' in x.value for x in app.error) and spy.call_count==1
        assert not app.get('download_button')
