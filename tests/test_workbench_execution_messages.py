"""Presentation-only checks. Cached manifests do not simulate scientific runs."""
from pathlib import Path
import json,zipfile
import pytest
from ecodata.interface.workbench_page import fusion_state,stage_strip,input_identity


def local_manifest():
    return dict(status='PROJECT_UPDATED',native_writeback_verified=True,native_model_executed=False,
        preparation=dict(tasks=[dict(status='LOCAL_RESIDUAL_EXECUTED',processed_observations=3)]))


def test_fusion_labels_require_actual_processing_receipt():
    m=local_manifest()
    assert fusion_state(m)=='LOCAL_RESIDUAL_EXECUTED'
    assert '局部观测融合已完成' in stage_strip(m['status'],fusion_state(m))
    m['preparation']['tasks'][0]['processed_observations']=0
    assert fusion_state(m) is None
    assert '局部观测融合已完成' not in stage_strip(m['status'],fusion_state(m))
    m=local_manifest();m['preparation']['tasks'].append(dict(status='NO_COLLOCATED_SUPPORT'))
    assert fusion_state(m)=='PARTIAL_LOCAL_RESIDUAL'
    m=dict(preparation=dict(fusion_decision=dict(status='SINGLE_SOURCE_SELECTED')))
    assert '最终采用单源' in stage_strip('PROJECT_UPDATED',fusion_state(m))


@pytest.mark.parametrize('mode',['local','writeback_only','unknown_run','awaiting','failed'])
def test_cached_page_renders_actual_stage_without_placeholder(tmp_path,mode):
    from streamlit.testing.v1 import AppTest
    manifest=local_manifest()
    if mode=='writeback_only':manifest['preparation']={}
    if mode=='unknown_run':manifest.pop('native_model_executed')
    if mode in {'awaiting','failed'}:
        manifest['status']='AWAITING_CONVERTER' if mode=='awaiting' else 'CONVERSION_FAILED'
        manifest['preparation']={}
    product=tmp_path/'download.zip'
    with zipfile.ZipFile(product,'w') as z:z.writestr('handoff.json',json.dumps(manifest))
    app=AppTest.from_file(Path(__file__).parents[1]/'src/ecodata/interface/app.py')
    app.session_state['workbench_result']=dict(identity=input_identity([],'SWAT+'),model='SWAT+',
        data=dict(manifest=manifest,archive=str(product)))
    app.run(timeout=30)
    assert not app.exception
    text='\n'.join(x.value for kind in ['success','caption','markdown','info'] for x in app.get(kind))
    assert '接口已预留' not in text and '接入口已预留' not in text and '队友的格式转化模块尚未接入' not in text
    assert app.get('download_button')
    if mode=='local':
        assert '局部融合' in text and '原生写回及回读校验已通过' in text and '模型模拟尚未运行' in text
    elif mode=='writeback_only':assert '局部融合' not in text and '写回完成不等于多源融合' in text
    elif mode=='unknown_run':assert '模型运行状态未确认' in text
    else:assert '局部融合' not in text and '本次原生转化尚未完成' in text
