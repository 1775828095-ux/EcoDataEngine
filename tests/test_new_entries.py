import io,json,zipfile,subprocess,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from ecodata.round1 import convert
BASE=Path(__file__).parents[1]/'examples/ai_round1'
CASES=[
 ('calpuff','常量点源排放','calpuff/points/A_synthetic_mapped_WGS84.zip','calpuff/points/config_mapped_WGS84.json','CALPUFF.INP'),
 ('swat_plus','固定日历施肥日程','swat_plus/management/A_synthetic_mapped.zip','swat_plus/management/config_mapped.json','management.sch'),
 ('camx','逐小时边界浓度','camx/boundary/A_synthetic_LCC.csv','camx/boundary/config_LCC.json','boundary.CAMx7.40.nc'),
 ('wrf','已有WPS场一致组包','wrf/input_package/A_synthetic_lambert.zip','wrf/input_package/config_lambert.json','COMBINED_2024-02-29_00.wps'),
 ('swat_plus','土地利用空间关联','swat_plus/spatial_landuse/A_synthetic_mapped.zip','swat_plus/spatial_landuse/config_mapped.json','hru-data.hru'),
 ('calpuff','离散受体接入','calpuff/receptors/A_synthetic_mapped_WGS84.zip','calpuff/receptors/config_synthetic_mapped_WGS84.json','CALPUFF.INP'),
 ('wrf','地理栅格瓦片','wrf/geography/A_synthetic_terrain_mapped.nc','wrf/geography/config_synthetic_terrain_mapped.json','geogrid_HGT_M.zip'),
 ('wrf','地理栅格瓦片','wrf/geography/A_synthetic_landuse_standard.nc','wrf/geography/config_synthetic_landuse_standard.json','geogrid_LANDUSEF.zip'),
 ('camx','三维初始浓度','camx/initial/A_synthetic_LCC.csv','camx/initial/config_synthetic_LCC.json','initial.CAMx7.40.nc'),
 ('cmaq','显式外圈边界浓度','cmaq/boundary/A_synthetic_LCC.nc','cmaq/boundary/config_synthetic_LCC.json','boundary.CMAQ55.nc'),
 ('calpuff','地理参数网格','calpuff/geography/A_synthetic_south_km.nc','calpuff/geography/config_synthetic_south_km.json','GEO.DAT'),
 ('wrf','多变量气象场','wrf/lambert/A_synthetic_mapped_km.nc','wrf/lambert/config_synthetic_mapped_km.json','ATM_2024-02-29_00.wps'),
 ('swat_plus','显式对象连接','swat_plus/connections/A_official.zip','swat_plus/connections/config_official.json','hru.con'),
 ('calpuff','高空探测','calpuff/upper_air/A_NOAA_IGRA_72357_20260101.csv','calpuff/upper_air/config_NOAA_IGRA.json','UP1.DAT'),
 ('calpuff','独立小时降水','calpuff/precipitation/A_NOAA_USCRN_20210701.csv','calpuff/precipitation/config_NOAA.json','PRECIP.DAT'),
 ('swat_plus','HRU属性与地形参数','swat_plus/hru/A_official_HRU_bundle.zip','swat_plus/hru/config_official.json','hru-data.hru'),
 ('swat_plus','土地利用配置数据库','swat_plus/landuse/A_official_vegetated_bundle.zip','swat_plus/landuse/config_official.json','landuse.lum'),
 ('swat_plus','含水层参数数据库','swat_plus/aquifer/A_synthetic_mapped_bundle.zip','swat_plus/aquifer/config_synthetic.json','aquifer.aqu'),
 ('wrf','多变量气象场','wrf/pressure/A_ncep_pressure.nc','wrf/pressure/config.json','ATM_2024-01-01_00.wps'),
 ('camx','网格排放','camx/emissions/A_synthetic_LATLON_hourly.csv','camx/emissions/config_LATLON.json','emissions.CAMx7.40.nc'),
 ('cmaq','BCON静态边界剖面','cmaq/bcon_profile/A_synthetic_wide.csv','cmaq/bcon_profile/config_synthetic_wide.json','BC_PROFILE_CMAQ55.csv'),
 ('cmaq','网格排放','cmaq/emissions/A_EDGAR_CO_2014_Paris.nc','cmaq/emissions/config_EDGAR.json','emissions.CMAQ55.nc')]

@pytest.mark.parametrize('model,choice,a,c,expected',CASES)
def test_cli_and_ui_share_real_conversion(model,choice,a,c,expected,tmp_path):
    data=(BASE/a).read_bytes();config=(BASE/c).read_bytes();files,_=convert(model,data,json.loads(config))
    assert expected in files and 'conversion_report.json' in files
    def upload(label,**kwargs):
        if label.startswith('可选'):return None
        return SimpleNamespace(getvalue=lambda:data if label=='导入源数据' else config)
    with patch('streamlit.file_uploader',side_effect=upload),patch('streamlit.download_button',wraps=st.download_button) as downloads:
        app=AppTest.from_string('from ecodata.interface.round1_page import render_round1_page\nrender_round1_page()',default_timeout=30).run()
        app.selectbox[0].set_value(model).run();app.radio[0].set_value(choice).run();app.button[0].click().run()
        assert not app.exception and not app.error
        actual,_=app.session_state['round1_result'];assert set(actual)==set(files)
        # CMAQ timestamps in global CDATE can differ by seconds, so compare meaningful arrays separately in adapter tests.
        if model!='cmaq':assert actual[expected]==files[expected]
        archive=next(c.args[1] for c in downloads.call_args_list if c.args[0]=='下载全部转换文件')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:assert z.read(expected)==actual[expected]
        # Deliberate mismatch: clear cached downloads and reject before attempting old converter.
        wrong=json.loads(config);wrong['mode']='wrong_mode'
    wrong_bytes=json.dumps(wrong).encode()
    with patch('streamlit.file_uploader',side_effect=lambda label,**kwargs:None if label.startswith('可选') else SimpleNamespace(getvalue=lambda:data if label=='导入源数据' else wrong_bytes)):
        app.run();app.button[0].click().run();assert '不一致' in app.error[0].value
    out=tmp_path/model
    result=subprocess.run([__import__('sys').executable,'-B','-X','utf8','-m','ecodata.round1','--model',model,'--input',str(BASE/a),'--config',str(BASE/c),'--output',str(out)],capture_output=True,text=True,encoding='utf8',timeout=40)
    assert result.returncode==0,result.stderr;assert (out/expected).exists()


@pytest.mark.parametrize('model,mode',[('calpuff','gridded_emissions'),('swat_plus','unknown'),('wrf',[]),('cmaq','bcon'),('camx','meteorology')])
def test_unknown_mode_never_falls_back_to_another_format(model,mode):
    from ecodata.core.exceptions import SchemaError
    with pytest.raises(SchemaError,match='不支持转换'):
        convert(model,b'',{'mode':mode})
