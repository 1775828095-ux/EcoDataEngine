"""Recovery after module reload: status evidence, not scientific model fixtures."""
import json
import pytest
from ecodata.interface import swat_build as build,swat_runner as runner
from ecodata.core.exceptions import SchemaError


def test_completed_build_recovers_after_memory_cleared(tmp_path,monkeypatch):
    monkeypatch.setattr(build,'RUNS',tmp_path)
    monkeypatch.setattr(build,'JOBS',{})
    ident='a'*32;directory=tmp_path/ident;directory.mkdir()
    record=dict(id=ident,directory=str(directory),state='ready',stage='已生成工程，尚未运行模型')
    build._save_status(record)
    assert build.status(ident)==record
    record['state']='failed';build._save_status(record)
    assert build.status(ident)['state']=='failed'


def test_build_rejects_mismatched_persisted_location(tmp_path,monkeypatch):
    monkeypatch.setattr(build,'RUNS',tmp_path);monkeypatch.setattr(build,'JOBS',{})
    ident='b'*32;directory=tmp_path/ident;directory.mkdir()
    (directory/'build_status.json').write_text(json.dumps(dict(id=ident,directory=str(tmp_path/'elsewhere'))))
    with pytest.raises(ValueError,match='位置'):build.status(ident)


def test_native_status_reads_disk_without_claiming_process_control(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'_JOBS',{})
    ident='c'*32;directory=tmp_path/'runs/swat_plus'/ident;directory.mkdir(parents=True)
    record=dict(id=ident,directory=str(directory),state='running')
    target=directory/'runner_status.json';target.write_text(json.dumps(record))
    assert runner.job_status(ident)['state']=='running'
    assert runner.job_status(ident)['process_control_available'] is False
    record['state']='succeeded';target.write_text(json.dumps(record))
    assert runner.job_status(ident)['state']=='succeeded'


@pytest.mark.parametrize('ident',['../escape','x'*32,None])
def test_recovered_ids_are_not_arbitrary_paths(ident,monkeypatch):
    monkeypatch.setattr(runner,'_JOBS',{});monkeypatch.setattr(build,'JOBS',{})
    with pytest.raises((ValueError,SchemaError)):build.status(ident)
    with pytest.raises((ValueError,SchemaError)):runner.job_status(ident)
