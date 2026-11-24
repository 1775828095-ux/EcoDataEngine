"""Synthetic boundary checks; actual native acceptance has separate real evidence."""
from pathlib import Path
import json,zipfile
import pytest
from ecodata.models.final_check import check_model_output,attach_final_check
from ecodata.interface import swat_runner
from ecodata.core.exceptions import SchemaError
from test_swat_weather_project import project,weather


def bundle(tmp_path,changes=None):
    files=project();values=weather()
    files['file.cio']=b'synthetic rev.62.0.0\nsimulation time.sim null null\nclimate weather-sta.cli null null pcp.cli tmp.cli null null null null\n'
    files['old.pcp']=values['station.pcp'];files['old.tmp']=values['station.tem']
    files['weather-sta.cli']=files['weather-sta.cli'].replace(b'sim sim sim',b'null null null')
    files.update(changes or {})
    path=tmp_path/'converted.zip'
    with zipfile.ZipFile(path,'w') as archive:
        for name,raw in files.items():archive.writestr('case/TxtInOut/'+name,raw)
    return dict(archive=str(path),original_archive=path,manifest={},archive_members=[],source_formats=[])


def unavailable(*args,**kwargs):raise SchemaError('registered engine unavailable')


def test_format_pass_does_not_claim_model_read(tmp_path):
    result=bundle(tmp_path)
    gate=check_model_output(result,'swat_plus',tmp_path/'check',run_native=False)
    assert gate['format']['status']=='PASS' and gate['project']['status']=='PASS'
    assert gate['native']['status']=='NOT_RUN' and not gate['native'].get('model_read_confirmed')


def test_project_failure_is_separate_from_weather_format(tmp_path):
    result=bundle(tmp_path)
    with zipfile.ZipFile(result['archive'],'a') as archive:
        archive.writestr('case/TxtInOut/a.bsn','synthetic')
        archive.writestr('case/TxtInOut/b.bsn','synthetic')
    with zipfile.ZipFile(result['archive']) as archive:data={n:archive.read(n) for n in archive.namelist()}
    data['case/TxtInOut/file.cio']+=b'basin a.bsn b.bsn\n'
    with zipfile.ZipFile(result['archive'],'w') as archive:
        for name,raw in data.items():archive.writestr(name,raw)
    gate=check_model_output(result,'swat_plus',tmp_path/'check')
    assert gate['format']['status']=='PASS' and gate['project']['status']=='FAIL'
    assert gate['native']['status']=='NOT_RUN'


def test_duplicate_weather_date_fails_before_native_execution(tmp_path,monkeypatch):
    raw=weather()['station.pcp'].replace(b'2021 2 0',b'2021 1 0')
    result=bundle(tmp_path,{'old.pcp':raw})
    monkeypatch.setattr(swat_runner,'start_job',lambda *a,**k:pytest.fail('invalid data must not execute'))
    gate=check_model_output(result,'swat_plus',tmp_path/'check')
    assert gate['format']['status']=='FAIL' and gate['native']['status']=='NOT_RUN'


def test_missing_or_untrusted_engine_is_not_format_failure(tmp_path,monkeypatch):
    result=bundle(tmp_path)
    monkeypatch.setattr(swat_runner,'engine_path',unavailable)
    gate=check_model_output(result,'swat_plus',tmp_path/'check')
    assert gate['format']['status']=='PASS' and gate['native']['status']=='NOT_RUN'
    assert 'unavailable' in gate['native']['reason']


def test_ai_cannot_promote_unrun_model_and_public_review_is_cached(tmp_path,monkeypatch):
    from ecodata.desktop import service,output_review
    from ecodata.models.requirements import RequirementStore
    result=bundle(tmp_path)
    monkeypatch.setattr(swat_runner,'engine_path',unavailable)
    attach_final_check(result,'swat_plus',tmp_path/'check')
    calls=[]
    def reply(*args,**kwargs):
        calls.append(1)
        return dict(verdict='ready',conversion_benefit='synthetic advice',missing_data=[],missing_information=[])
    monkeypatch.setattr(service,'send',reply)
    review,error=service.review_model_output(result,RequirementStore().get('swat_plus'))
    assert error is None and review['verdict']=='needs_verification' and review['original_verdict']=='ready'
    assert result['manifest']['final_check']['native']['status']=='NOT_RUN'
    service.review_model_output(result,RequirementStore().get('swat_plus'));assert len(calls)==1
    check=output_review.describe_output(result,RequirementStore().get('swat_plus'))
    text=output_review.availability_text(check,review)
    assert '原生格式检查 · 通过' in text and '模型实际试读 · 未运行' in text
    with zipfile.ZipFile(result['archive']) as archive:
        assert json.loads(archive.read(result['manifest']['final_check']['evidence_prefix']+'/complete.json'))['ai_review']['status']=='COMPLETED'


def test_no_reference_to_output_can_overwrite_original_input(tmp_path):
    result=bundle(tmp_path);before=Path(result['archive']).read_bytes()
    with pytest.raises(ValueError,match='输出目录之外'):
        attach_final_check(result,'swat_plus',tmp_path/'other-output'/'check')
    assert Path(result['archive']).read_bytes()==before


def test_public_preparation_invokes_the_same_final_gate(tmp_path,monkeypatch):
    from ecodata.algorithms import automatic_preparation
    from ecodata.interface.engineering_upload import LocalUpload
    output=tmp_path/'output';output.mkdir()
    result=bundle(output)
    source=tmp_path/'input.txt';source.write_text('synthetic source, read-only')
    monkeypatch.setattr(swat_runner,'engine_path',unavailable)
    monkeypatch.setattr(automatic_preparation,'_prepare_uploads',lambda *a,**k:result)
    actual=automatic_preparation.prepare_uploads([LocalUpload(source.name,source)],'swat_plus',output)
    assert actual['manifest']['final_check']['format']['status']=='PASS'
    assert actual['manifest']['final_check']['native']['status']=='NOT_RUN'
    assert source.read_text()=='synthetic source, read-only'


def test_assumptions_are_recognized_by_content_not_package_filename(tmp_path):
    result=bundle(tmp_path)
    with zipfile.ZipFile(result['archive'],'a') as archive:
        archive.writestr('arbitrary-scientific-declaration.json',json.dumps(dict(status='WORKING_ASSUMPTION_ONLY',rule='test convention')))
    report=check_model_output(result,'swat_plus',tmp_path/'check',run_native=False)
    assert report['scientific_definitions']['status']=='WORKING_ASSUMPTIONS'
    assert report['scientific_definitions']['assumptions'][0]['source_document']=='arbitrary-scientific-declaration.json'


def test_reprocessing_preserves_old_receipts_without_duplicate_zip_members(tmp_path,monkeypatch):
    result=bundle(tmp_path);monkeypatch.setattr(swat_runner,'engine_path',unavailable)
    first=attach_final_check(result,'swat_plus',tmp_path/'check1')['evidence_prefix']
    second=attach_final_check(result,'swat_plus',tmp_path/'check2')['evidence_prefix']
    assert first!=second
    with zipfile.ZipFile(result['archive']) as archive:
        names=archive.namelist();assert len(names)==len(set(names))
        assert first+'/final_check.json' in names and second+'/final_check.json' in names


def test_other_models_keep_existing_native_execution_evidence(tmp_path):
    result=bundle(tmp_path);result['manifest']['native_model_executed']=True
    report=check_model_output(result,'camx',tmp_path/'check')
    assert report['native']['status']=='NOT_RUN' and report['native']['previous_native_execution']
    assert result['manifest']['native_model_executed'] is True


def test_unreferenced_historical_outputs_are_not_loaded_for_native_inputs(tmp_path):
    result=bundle(tmp_path)
    with zipfile.ZipFile(result['archive'],'a') as archive:
        archive.writestr('case/TxtInOut/unrelated_previous_output.txt',b'old output'*10000)
    report=check_model_output(result,'swat_plus',tmp_path/'check',run_native=False)
    assert report['format']['status']=='PASS'
    excluded=report['input_dependency_scope']['excluded']
    assert any(row['name']=='unrelated_previous_output.txt' for row in excluded)
    assert report['input_dependency_scope']['loaded_bytes']<10000


def test_uploaded_runtime_cannot_be_injected_through_a_control_reference(tmp_path,monkeypatch):
    result=bundle(tmp_path)
    with zipfile.ZipFile(result['archive']) as archive:files={n:archive.read(n) for n in archive.namelist()}
    files['case/TxtInOut/file.cio']+=b'other bad.dll\n'
    files['case/TxtInOut/bad.dll']=b'not a trusted library'
    with zipfile.ZipFile(result['archive'],'w') as archive:
        for name,raw in files.items():archive.writestr(name,raw)
    monkeypatch.setattr(swat_runner,'start_job',lambda *a,**k:pytest.fail('uploaded library cannot execute'))
    report=check_model_output(result,'swat_plus',tmp_path/'check')
    assert report['format']['status']=='FAIL' and report['native']['status']=='NOT_RUN'
