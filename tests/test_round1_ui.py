from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/'examples/ai_round1'

def test_scoped_page_runs_same_converter_and_clears_stale_downloads():
    mutable={'data':(CASES/'calpuff/A_synthetic_complete.csv').read_bytes()}
    cfg=(CASES/'calpuff/config_synthetic.json').read_bytes()
    def upload(label,**kwargs):
        if label=='导入源数据':return SimpleNamespace(name='surface.csv',getvalue=lambda:mutable['data'])
        if label=='导入字段、单位与版本配置JSON':return SimpleNamespace(name='config.json',getvalue=lambda:cfg)
        return None
    with patch('streamlit.file_uploader',side_effect=upload):
        app=AppTest.from_file(str(ROOT/'src/ecodata/interface/app.py'),default_timeout=30).run()
        app.sidebar.radio[0].set_value('首轮专项转换').run()
        next(b for b in app.button if b.label=='执行专项转换').click().run()
        assert not app.exception and not app.error
        assert len(app.get('download_button'))==3
        files,report=app.session_state['round1_result']
        assert 'SURF.DAT' in files and report['hours']==4 and not report['complete_model_input']
        mutable['data']=mutable['data'].replace(b'20,50',b'9999,50')
        app.run()
        assert not app.get('download_button')
        next(b for b in app.button if b.label=='执行专项转换').click().run()
        assert app.error and not app.get('download_button')
