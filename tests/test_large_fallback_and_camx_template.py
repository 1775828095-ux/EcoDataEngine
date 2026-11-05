import io,json,zipfile
from pathlib import Path
import pytest
from ecodata.interface import engineering_upload as intake
from ecodata.models.project_inventory import prepare_project_inventory


def test_large_zip_never_calls_getvalue_in_generic_fallback(tmp_path,monkeypatch):
    path=tmp_path/'data.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('data.csv',b'x'*100)
    class StreamingOnly:
        name=path.name
        def __init__(self):self.path=path
        def getvalue(self):raise AssertionError('Whole archive loaded')
    monkeypatch.setattr(intake,'MAX_EAGER_SCAN_BYTES',32)
    with pytest.raises(ValueError,match='未匹配可用的流式工程适配器'):
        intake.scan_model_uploads([StreamingOnly()],include_all=True)


def test_small_local_zip_is_read_from_path_without_whole_archive(tmp_path):
    path=tmp_path/'data.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('data.csv',b'time,value\n')
    class StreamingOnly:
        name=path.name
        def __init__(self):self.path=path
        def getvalue(self):raise AssertionError('Whole archive loaded')
    assert intake.scan_model_uploads([StreamingOnly()],include_all=True).files==[('data.csv',b'time,value\n')]


def test_camx_job_template_is_inspected_without_executing_or_guessing(tmp_path):
    path=tmp_path/'case.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('run.job',"cat << END > CAMx.in\n &CAMx_Control\n Chemistry_Parameters='chem.dat',\n Met2D_Grid(1)='$MET/day',\n /\nEND\n")
        z.writestr('chem.dat',b'original')
    result=prepare_project_inventory([intake.LocalUpload(path.name,path)],'camx',tmp_path)
    assert result['manifest']['status']=='ENGINEERING_INSPECTED'
    doc=json.loads((tmp_path/'project_inventory.json').read_text(encoding='utf-8'))
    row=doc['projects'][0]
    assert not row['commands_executed'] and not row['native_control_generated']
    assert {r['status'] for r in row['bindings']}=={'BOUND','UNKNOWN'}
