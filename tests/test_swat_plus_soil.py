import copy,io,json,shlex,zipfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError

BASE=Path(__file__).parents[1]/'examples/ai_round1/swat_plus/soil'

def load():
    return pd.read_csv(BASE/'A_official_valid_subset.csv',keep_default_na=False),json.loads((BASE/'config.json').read_text())

def raw(df):return df.to_csv(index=False).encode()

def parse(content):
    # Independent shlex/Decimal parser, no imports from soil writer.
    lines=content.decode('ascii').splitlines()[2:];i=0;result={}
    while i<len(lines):
        h=shlex.split(lines[i]);i+=1
        if not h:continue
        assert len(h)>=7
        n=int(h[1]);layers=[]
        for j in range(n):
            values=[Decimal(x) for x in lines[i].split()];i+=1
            assert len(values)==14;layers.append(values)
        result[h[0]]=([h[2],*[Decimal(x) for x in h[3:6]],' '.join(h[6:])],layers)
    return result

def test_official_13_soils_52_layers_exact():
    df,cfg=load();files,r=convert('swat_plus',raw(df),cfg)
    assert parse(files['soils.sol'])=={k:v for k,v in parse((BASE/'B2_official_soils.sol').read_bytes()).items() if k!='soil_05'}
    assert len(r['soils'])==13 and r['layer_records']==52
    assert 'file.cio' not in files and 'weather-wgn.cli' not in files

@pytest.mark.parametrize('problem',['column','missing','nan','inf','depth','duplicate_layer','metadata','texture','id','percent','negative','unit','version','depth_basis','mapping','source','missing_config','hydgrp'])
def test_reject(problem):
    df,cfg=load()
    if problem=='column':df=df.drop(columns='k')
    elif problem=='missing':df.loc[0,'k']=-9999
    elif problem=='nan':df.loc[0,'k']=float('nan')
    elif problem=='inf':df.loc[0,'k']=float('inf')
    elif problem=='depth':df.loc[1,'depth']=1
    elif problem=='duplicate_layer':df.loc[1,'layer']=1
    elif problem=='metadata':df.loc[1,'hyd_grp']='C'
    elif problem=='texture':df['texture']='very long texture label exceeding 16'
    elif problem=='id':df['soil']='../../x'
    elif problem=='percent':df.loc[0,'clay']=90
    elif problem=='negative':df.loc[0,'bd']=-1
    elif problem=='unit':cfg['units']['bd']='m'
    elif problem=='version':cfg['target_model_version']='61'
    elif problem=='depth_basis':cfg['depth_basis']='thickness'
    elif problem=='mapping':cfg['fields']['k']='bd'
    elif problem=='source':cfg['source']={}
    elif problem=='missing_config':cfg.pop('missing_values')
    elif problem=='hydgrp':df['hyd_grp']='A/D'
    with pytest.raises(SchemaError):convert('swat_plus',raw(df),cfg)

@pytest.mark.parametrize('fmt',['csv','xlsx','parquet'])
def test_multiple_layers_mapping_units_formats(fmt):
    df,cfg=load();df=df.iloc[:7].copy() # first soil4 layers, second soil3 layers
    cfg['source']={'kind':'synthetic_test','description':'Engineered format/unit regression, not real soil observations'}
    expected=parse(convert('swat_plus',raw(df),cfg)[0]['soils.sol'])
    df['depth']/=10;df['zmx']/=1000;df['bd']*=1000;df['k']/=3600000
    df['ec']*=1
    for k in ['cbn','clay','silt','sand','rock','cal']:df[k]/=100
    cfg['units'].update(depth='cm',zmx='m',bd='kg/m3',k='m/s',ec='mS/cm')
    for k in ['cbn','clay','silt','sand','rock','cal']:cfg['units'][k]='1'
    df=df.iloc[::-1].rename(columns={'k':'custom_K'});cfg['fields']['k']='custom_K';cfg['source_format']=fmt
    b=io.BytesIO()
    if fmt=='csv':content=raw(df)
    elif fmt=='xlsx':df.to_excel(b,index=False);content=b.getvalue()
    else:df.to_parquet(b,index=False);content=b.getvalue()
    actual=parse(convert('swat_plus',content,cfg)[0]['soils.sol'])
    assert actual.keys()==expected.keys()
    for name in expected:
        assert actual[name][0]==expected[name][0]
        for row,reference in zip(actual[name][1],expected[name][1]):
            assert all(abs(a-b)<=Decimal('1e-10') for a,b in zip(row,reference))


def test_ui_real_writer_download_and_missing_column_error():
    df,cfg=load();mutable={'data':raw(df),'cfg':json.dumps(cfg).encode()}
    def upload(label,**kwargs):
        if label.startswith('可选'):return None
        return SimpleNamespace(getvalue=lambda:mutable['data' if label=='导入源数据' else 'cfg'])
    from ecodata.models.adapters.swat_plus_soil import export_soils
    with patch('streamlit.file_uploader',side_effect=upload),patch('ecodata.models.adapters.swat_plus_soil.export_soils',wraps=export_soils) as writer,patch('streamlit.download_button',wraps=st.download_button) as download:
        app=AppTest.from_string('from ecodata.interface.round1_page import render_round1_page\nrender_round1_page()',default_timeout=30).run()
        app.selectbox[0].set_value('swat_plus').run();app.radio[0].set_value('分层土壤').run();app.button[0].click().run()
        assert not app.error and not app.exception and writer.call_count==1
        files,_=app.session_state['round1_result']
        archive=next(c.args[1] for c in download.call_args_list if c.args[0]=='下载全部转换文件')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert z.read('soils.sol')==files['soils.sol'] and 'file.cio' not in z.namelist()
            parse(z.read('soils.sol'))
        mutable['data']=raw(df.drop(columns='k'));app.run();app.button[0].click().run()
        assert '缺列' in app.error[0].value and 'round1_result' not in app.session_state


def test_original_official_inconsistent_texture_is_rejected():
    df,cfg=load()
    df=pd.read_csv(BASE/'A_official_decoded_layers.csv',keep_default_na=False)
    with pytest.raises(SchemaError,match='soil_05.*100%'):
        convert('swat_plus',raw(df),cfg)
