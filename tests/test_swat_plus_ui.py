from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.round1 import convert
from ecodata.models.adapters.swat_plus import SWATPlusWriter
from ecodata.io.structures import read_structure,export_structure
from ecodata.core.exceptions import SchemaError
ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'examples/ai_round1/swat_plus'

def test_upload_matches_existing_reference():
    files,r=convert('swat_plus',(CASE/'daymet_2021_complete.nc').read_bytes(),json.loads((CASE/'config_ui.json').read_text()))
    assert 'file.cio' not in files
    assert r['records']==365 and r['writer']=='SWATPlusWriter'
    assert {'pcp.cli','tmp.cli','weather-sta.cli'}<=set(files)

def test_real_ui_calls_writer_and_reports_errors():
    good=(CASE/'daymet_2021_complete.nc').read_bytes()
    cfg=(CASE/'config_ui.json').read_bytes()
    missing,_,_=export_structure(read_structure('weather.nc',good).drop_vars('air_temperature_min'),'NetCDF')
    mutable={'data':good,'config':cfg}
    def upload(label,**kwargs):
        if label.startswith('上传已有流域工程'): return None
        data=mutable['data'] if label=='导入源数据' else mutable['config']
        return SimpleNamespace(name='weather.nc' if label=='导入源数据' else 'config.json',getvalue=lambda:data)
    original=SWATPlusWriter.write
    with patch('streamlit.file_uploader',side_effect=upload),patch.object(SWATPlusWriter,'write',autospec=True,side_effect=original) as spy:
        app=AppTest.from_file(str(ROOT/'src/ecodata/interface/app.py'),default_timeout=30).run()
        app.sidebar.radio[0].set_value('首轮专项转换').run()
        next(s for s in app.selectbox if s.label=='专项模型').set_value('swat_plus').run()
        next(b for b in app.button if b.label=='执行专项转换').click().run()
        assert not app.exception and not app.error and spy.call_count==1
        files,r=app.session_state['round1_result']
        assert 'file.cio' not in files
        assert len(app.get('download_button'))==len(files)+1
        pcp=np.loadtxt(files['station.pcp'].decode().splitlines(),skiprows=3)
        tem=np.loadtxt(files['station.tem'].decode().splitlines(),skiprows=3)
        assert pcp.shape==(365,3) and tem.shape==(365,4)
        np.testing.assert_array_equal(pcp[:,1],np.arange(1,366))
        assert pcp[0,2]==22.08 and list(tem[0,2:])==[1.65,-4.94]
        mutable['data']=missing;app.run()
        assert not app.get('download_button')
        next(b for b in app.button if b.label=='执行专项转换').click().run()
        assert '缺少字段：air_temperature_min' in app.error[0].value and spy.call_count==1
        mutable['data']=good;mutable['config']=b'{}';app.run()
        next(b for b in app.button if b.label=='执行专项转换').click().run()
        assert '配置需' in app.error[0].value and not app.get('download_button') and spy.call_count==1

def test_wrong_units_not_relabelled():
    good=(CASE/'daymet_2021_complete.nc').read_bytes();ds=read_structure('weather.nc',good)
    ds.precipitation.attrs['units']='m';bad,_,_=export_structure(ds,'NetCDF')
    with pytest.raises(SchemaError,match='单位需mm/day'):convert('swat_plus',bad,json.loads((CASE/'config_ui.json').read_text()))
