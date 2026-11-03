"""Actual Streamlit form submission; only upload transport is a test fixture."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import json,sys
import xarray as xr
from streamlit.testing.v1 import AppTest
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/autonomous_search_closure_20261029'
CASE=ROOT/'output/halfdate_reconstruction_20261028/cases/patch2_contiguous/config_test.json'


def run():
    folder=OUT/'page_acceptance';folder.mkdir(exist_ok=True)
    app=folder/'app.py'
    app.write_text('''from pathlib import Path
import json
import streamlit as st
from ecodata.interface.search_page import render_search_page
cfg=json.loads(Path('''+repr(str(CASE))+''').read_text(encoding='utf-8'))
class Upload:
    def __init__(self,item):self.name=Path(item['path']).name;self.payload=Path(item['path']).read_bytes()
    def getvalue(self):return self.payload
original=st.file_uploader
def transport(*args,**kwargs):
    return [Upload(item) for item in cfg['inputs']] if kwargs.get('key')=='unified_inputs' else None
st.file_uploader=transport
try:render_search_page()
finally:st.file_uploader=original
''',encoding='utf-8')
    guard=dict(active=True);reads=[]
    def audit(event,args):
        if event=='open' and guard['active'] and isinstance(args[0],(str,bytes,os.PathLike)):
            if Path(os.fsdecode(args[0])).name in {'test_labels.nc','validation_labels.nc','development_labels.nc'}:
                reads.append(str(args[0]));raise AssertionError('No external reference in page/API decisions')
    sys.addaudithook(audit)
    try:
        cfg=json.loads(CASE.read_text(encoding='utf-8'));q=cfg['request']
        at=AppTest.from_file(str(app),default_timeout=600).run()
        assert not at.exception
        at.text_input(key='unified_variable').set_value('ndvi');at.text_input(key='unified_unit').set_value('1')
        at.selectbox(key='unified_role_0').set_value('primary');at.selectbox(key='unified_role_1').set_value('background')
        at.text_area(key='unified_target').set_value(json.dumps(q));at.checkbox(key='unified_quality_confirm').check()
        at.number_input(key='unified_seed').set_value(11)
        at.button(key='unified_submit').click().run(timeout=600)
        assert not at.exception,at.exception
        result,page_folder=at.session_state['unified_result'];assert result['status']=='PASS',result
        assert at.success and Path(result['product']).exists()
        items=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes(),{}) for i in cfg['inputs']]
        request=dict(q,purpose='same_variable',allow_training=False,allow_existing_weights=False)
        direct=api.process_current_input(items,request,folder/'direct_api',seed=11)
        assert direct['status']=='PASS'
        assert result['best']['key']==direct['best']['key']
        assert result['stop_reason']==direct['stop_reason']
        with xr.open_dataset(result['product']) as a,xr.open_dataset(direct['product']) as b:xr.testing.assert_equal(a.load(),b.load())
        assert not reads
        save(folder/'acceptance.json',dict(status='PASS',page_folder=page_folder,api_folder=str(folder/'direct_api'),
            selected_key=result['best']['key'],stop_reason=result['stop_reason'],numeric_products_equal=True,
            checkpoint_available=result['checkpoint_available'],external_reference_reads=reads,seed=11,
            transport_scope='Streamlit AppTest real controls and submit event; file upload bytes injected because AppTest lacks native uploader support',
            search_backend_mocked=False,manual_deployment_used=False))
        print('PAGE SUBMIT AND API NUMERIC OUTPUT MATCH; NO EXTERNAL REFERENCE OR MANUAL DEPLOYMENT',flush=True)
    finally:guard['active']=False


if __name__=='__main__':run()
