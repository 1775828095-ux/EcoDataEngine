"""Functional fixtures only; no claims about environmental accuracy."""
import zipfile
import pytest
from ecodata.models.final_check import check_model_output
from ecodata.models import final_native_check as native
from test_wrf_external_project import project
from test_project_binding import case


def artifact(tmp_path,files):
    archive=tmp_path/'product.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,raw in files.items():z.writestr(name,raw.encode() if isinstance(raw,str) else raw)
    return dict(archive=str(archive),manifest={})


def test_wrf_reuses_actual_native_validator(tmp_path):
    result=artifact(tmp_path,project())
    report=check_model_output(result,'wrf',tmp_path/'gate')
    assert report['format']['status']=='PASS'
    assert report['project']['validation']['native_structure_candidate']
    assert report['native']['status']=='NOT_RUN'
    assert 'WRFDA' in report['native']['reason']


def test_invalid_wrf_dimensions_cannot_pass(tmp_path):
    files=project();files['user/run/namelist.input']=files['user/run/namelist.input'].replace(b'e_we=4',b'e_we=8')
    report=check_model_output(artifact(tmp_path,files),'wrf',tmp_path/'gate')
    assert report['format']['status']=='FAIL' and report['native']['status']=='NOT_RUN'


@pytest.mark.parametrize('model',['camx','cmaq','calpuff','wrf'])
def test_report_only_is_not_native_product(tmp_path,model):
    report=check_model_output(artifact(tmp_path,{'report.json':'{}'}),model,tmp_path/'gate')
    assert report['project']['status']=='FAIL'
    assert report['native']['status']=='NOT_RUN'
    assert not report['native'].get('model_read_confirmed')


def test_camx_no_tflag_not_marked_pass(tmp_path):
    report=check_model_output(artifact(tmp_path,case()),'camx',tmp_path/'gate')
    assert report['project']['status']=='PASS'
    assert report['format']['status']=='FAIL'
    assert any('TFLAG' in error for error in report['format']['errors'])


def test_registered_program_missing_is_explicit(monkeypatch):
    monkeypatch.setattr(native,'_runtime',lambda model:dict(engine='Z:/absent-model.exe',sha256='0'*64))
    result=native._probe('camx')
    assert result['status']=='NOT_RUN' and '未安装' in result['reason']


def test_program_hash_mismatch_never_executes(tmp_path,monkeypatch):
    p=tmp_path/'not-model.exe';p.write_bytes(b'not a model')
    monkeypatch.setattr(native,'_runtime',lambda model:dict(engine=str(p),sha256='0'*64))
    result=native._probe('cmaq')
    assert result['status']=='NOT_RUN' and 'SHA256' in result['reason']


def test_shared_gate_dispatches_four_models(tmp_path,monkeypatch):
    calls=[]
    def check(archive,model,folder,**kwargs):
        calls.append(model)
        return dict(format=dict(status='UNKNOWN',errors=[]),project=dict(status='UNKNOWN'),native=dict(status='NOT_RUN',reason='test'))
    monkeypatch.setattr(native,'check_other_model',check)
    result=artifact(tmp_path,{'data.txt':'test'})
    for model in ['wrf','cmaq','camx','calpuff']:check_model_output(result,model,tmp_path/model)
    assert calls==['wrf','cmaq','camx','calpuff']


def test_old_ready_flag_cannot_override_final_failure():
    from ecodata.desktop.output_review import describe_output
    result=dict(manifest=dict(native_ready=True,final_check=dict(format=dict(status='FAIL',errors=['bad time']),
        project=dict(status='PASS'),native=dict(status='NOT_RUN'))))
    check=describe_output(result,dict(id='camx'))
    assert not check['native_ready']
    assert any('bad time' in item for item in check['notes'])
