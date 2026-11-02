import copy
import csv
import io
import json
import zipfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from ecodata.core.exceptions import SchemaError
from ecodata.round1 import convert

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'examples/ai_round1/swat_plus/wgn'


def load():
    return (CASE/'A_official_decoded_monthly.csv').read_bytes(), json.loads((CASE/'config.json').read_text(encoding='utf-8'))


def independent_read(content):
    # Positions from cli_wgnread.f90, not from the Python writer or its constants.
    lines=content.decode('ascii').splitlines()
    assert len(lines)>=15 and (len(lines)-1)%14==0
    stations={}
    for i in range(1,len(lines),14):
        header=lines[i].split(); assert len(header)==5 and header[0] not in stations
        assert len(lines[i+1].split())==14
        records=[list(map(Decimal,line.split())) for line in lines[i+2:i+14]]
        assert len(records)==12 and all(len(r)==14 for r in records)
        stations[header[0]]={'meta':list(map(Decimal,header[1:])),'months':records}
    return stations


def mutate(content, action):
    rows=list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    headers=list(rows[0]);action(rows,headers)
    out=io.StringIO(); w=csv.DictWriter(out,fieldnames=headers);w.writeheader();w.writerows(rows)
    return out.getvalue().encode()


def test_official_format_reference_months_and_168_values():
    data,cfg=load(); files,report=convert('swat_plus',data,cfg)
    actual=independent_read(files['weather-wgn.cli'])
    reference=independent_read((CASE/'B2_official_weather-wgn.cli').read_bytes())
    assert actual.keys()==reference.keys()=={'wgn001'}
    assert actual['wgn001']['meta']==reference['wgn001']['meta']
    errors=[abs(a-b) for row,ref in zip(actual['wgn001']['months'],reference['wgn001']['months']) for a,b in zip(row,ref)]
    assert len(errors)==168 and max(errors)<=Decimal('1e-12')
    assert report['units']==dict(zip(('tmpmx','tmpmn','tmpstdmx','tmpstdmn','pcpmm','pcpstd','pcpskw','pr_wd','pr_ww','pcpd','rainhmx','solarav','dewpt','windav'),('degC','degC','degC','degC','mm','mm/day','1','1','1','day','mm','MJ/m2/day','degC','m/s')))
    assert report['source']['kind']=='format_reference'
    assert report['station_reference_check']['status']=='not_supplied'
    assert not report['random_weather_generation'] and not report['model_run_verified']
    assert 'file.cio' not in files and 'weather-sta.cli' not in files


@pytest.mark.parametrize('problem', ['missing_month','duplicate_month','missing_field','nan','blank','marker','wrong_station','negative_std','probability','wet_days','wrong_temperature','extra_field'])
def test_invalid_statistics_rejected_with_specific_error(problem):
    data,cfg=load()
    def action(rows,headers):
        if problem=='missing_month': rows.pop(1)
        elif problem=='duplicate_month':rows[1]['month']='1'
        elif problem=='missing_field':
            headers.remove('RAINHMX')
            for r in rows:r.pop('RAINHMX')
        elif problem=='nan':rows[0]['PCPMM']='NaN'
        elif problem=='blank':rows[0]['DEWPT']=''
        elif problem=='marker':rows[0]['DEWPT']='-9999';cfg['missing_values']=[-9999]
        elif problem=='wrong_station':rows[0]['station']='other'
        elif problem=='negative_std':rows[0]['TMPSTDMX']='-1'
        elif problem=='probability':rows[0]['PR_WD']='1.1'
        elif problem=='wet_days':rows[0]['PCPD']='32'
        elif problem=='wrong_temperature':rows[0]['TMPMX']='-90'
        else:headers.append('unmapped');[r.update(unmapped='1') for r in rows]
    bad=mutate(data,action)
    with pytest.raises(SchemaError) as exc:convert('swat_plus',bad,cfg)
    assert str(exc.value)


@pytest.mark.parametrize('problem',['unit','station_unit','one_year','rain_years','blank_source','invalid_source_kind','duplicate_station','version','unsafe_name'])
def test_invalid_config_rejected(problem):
    data,cfg=load()
    if problem=='unit':cfg['units']['solarav']='W/m2'
    elif problem=='station_unit':cfg['station_units']['elevation']='km'
    elif problem=='one_year':cfg['source'].update(kind='climate_statistics',period_start_year=2021,period_end_year=2021)
    elif problem=='rain_years':cfg['stations'][0]['rain_years']=0
    elif problem=='blank_source':cfg['source']['url']=''
    elif problem=='invalid_source_kind':cfg['source']['kind']=[]
    elif problem=='duplicate_station':cfg['stations'].append(copy.deepcopy(cfg['stations'][0]))
    elif problem=='version':cfg['target_model_version']='61'
    else:cfg['stations'][0]['id']='wgn001/../../x'
    with pytest.raises(SchemaError):convert('swat_plus',data,cfg)


def test_mapping_reordering_units_and_valid_negative_temperatures():
    data,cfg=load()
    old=independent_read(convert('swat_plus',data,cfg)[0]['weather-wgn.cli'])
    def action(rows,headers):
        rows.reverse()
        for r in rows:
            r['TMPMX']=str(float(r['TMPMX'])+273.15)
            r['TMPSTDMX']=str(float(r['TMPSTDMX'])*1.8)
            r['PR_WD']=str(float(r['PR_WD'])*100)
            r['WINDAV']=str(float(r['WINDAV'])*3.6)
            r['custom_solar']=r.pop('SOLARAV')
        headers[headers.index('SOLARAV')]='custom_solar'
    new=mutate(data,action)
    cfg['units'].update(tmpmx='K',tmpstdmx='degF',pr_wd='%',windav='km/h')
    cfg['fields']['solarav']='custom_solar'
    actual=independent_read(convert('swat_plus',new,cfg)[0]['weather-wgn.cli'])
    for a,b in zip(actual['wgn001']['months'],old['wgn001']['months']):
        assert all(abs(x-y)<=Decimal('1e-12') for x,y in zip(a,b))
    assert actual['wgn001']['months'][0][1]<0


def test_explicit_wgn_reference_validation_without_rewriting_index():
    data,cfg=load()
    # Synthetic linkage only, not a complete official weather project.
    index=b'synthetic reference\nname wgn pcp tmp slr hmd wnd pet atmo\nweather1 wgn001 station.pcp station.tem sim sim sim null null\n'
    files,r=convert('swat_plus',data,cfg,weather_station_index=index)
    assert r['station_reference_check']['links']==[{'weather_station':'weather1','wgn_station':'wgn001'}]
    assert 'weather-sta.cli' not in files and 'file.cio' not in files
    for ref in (b'null',b'absent',b'wgn_001'):
        with pytest.raises(SchemaError,match='未对应本次统计站点'):
            convert('swat_plus',data,cfg,weather_station_index=index.replace(b'wgn001',ref))


def test_two_synthetic_station_blocks_do_not_mix_months():
    data,cfg=load();cfg['source']['kind']='synthetic_test'
    cfg['source']['description']='Synthetic duplicate station for block-order regression, not independent real climatology'
    cfg['stations'].append({**cfg['stations'][0], 'id':'synthetic2','latitude':40.0})
    def action(rows,headers):
        rows.extend([{**r,'station':'synthetic2','WINDAV':str(float(r['WINDAV'])+1)} for r in list(rows)])
        rows.reverse()
    files,r=convert('swat_plus',mutate(data,action),cfg)
    parsed=independent_read(files['weather-wgn.cli'])
    assert list(parsed)==['wgn001','synthetic2']
    assert parsed['synthetic2']['meta'][0]==Decimal('40')
    for a,b in zip(parsed['wgn001']['months'],parsed['synthetic2']['months']):
        assert a[:13]==b[:13] and abs(b[13]-a[13]-1)<=Decimal('1e-12')
    assert r['source']['kind']=='synthetic_test'


@pytest.mark.parametrize('unit,value', [('1','0.75'),('%','75')])
def test_native_humidity_fraction_slot_requires_explicit_units(unit,value):
    data,cfg=load();cfg['source']['kind']='synthetic_test';cfg['units']['dewpt']=unit
    data=mutate(data,lambda rows,headers:[r.update(DEWPT=value) for r in rows])
    files,report=convert('swat_plus',data,cfg)
    records=independent_read(files['weather-wgn.cli'])['wgn001']['months']
    assert all(row[12]==Decimal('0.75') for row in records)
    assert report['units']['dewpt']=='1'
    assert report['humidity_semantics']=={'wgn001':'relative_humidity_fraction'}
    assert report['unit_conversions']['dewpt']['output']=='1'


def test_ambiguous_all_month_dewpoint_not_silently_read_as_humidity():
    data,cfg=load()
    data=mutate(data,lambda rows,headers:[r.update(DEWPT='0.5') for r in rows])
    with pytest.raises(SchemaError,match='SWAT\\+62会按相对湿度分数读取'):
        convert('swat_plus',data,cfg)


@pytest.mark.parametrize('value', ['-0.01','1.01'])
def test_humidity_slot_out_of_range_is_rejected(value):
    data,cfg=load();cfg['units']['dewpt']='1'
    data=mutate(data,lambda rows,headers:rows[0].update(DEWPT=value))
    with pytest.raises(SchemaError,match='相对湿度声明'):
        convert('swat_plus',data,cfg)


def test_real_ui_wgn_download_and_rejection():
    data,cfg=load();mutable={'data':data,'cfg':json.dumps(cfg).encode()}
    def upload(label,**kwargs):
        if label.startswith('可选'):return None
        b=mutable['data'] if label=='导入源数据' else mutable['cfg']
        return SimpleNamespace(getvalue=lambda:b)
    from ecodata.models.adapters.swat_plus_wgn import export_wgn
    with patch('streamlit.file_uploader',side_effect=upload),patch('ecodata.models.adapters.swat_plus_wgn.export_wgn',wraps=export_wgn) as writer,patch('streamlit.download_button',wraps=st.download_button) as download:
        app=AppTest.from_string('from ecodata.interface.round1_page import render_round1_page\nrender_round1_page()',default_timeout=30).run()
        app.selectbox[0].set_value('swat_plus').run();app.radio[0].set_value('WGN月统计').run()
        app.button[0].click().run()
        assert not app.error and not app.exception and writer.call_count==1
        files,_=app.session_state['round1_result']
        archive=next(c.args[1] for c in download.call_args_list if c.args[0]=='下载全部转换文件')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert set(z.namelist())==set(files) and 'file.cio' not in z.namelist()
            assert z.read('weather-wgn.cli')==files['weather-wgn.cli']
            independent_read(z.read('weather-wgn.cli'))
        mutable['data']=mutate(data,lambda rows,headers:rows.pop())
        app.run();assert not app.get('download_button')
        app.button[0].click().run();assert '缺月份' in app.error[0].value
