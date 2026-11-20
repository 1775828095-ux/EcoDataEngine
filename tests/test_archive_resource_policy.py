"""Deployment limits and disk guards cannot be controlled by uploaded data."""
import io,json,tarfile,zipfile
from types import SimpleNamespace
import pytest
from ecodata.interface.engineering_upload import LocalUpload
from ecodata.models import archive_inputs as module


@pytest.mark.parametrize('value',['0','-1','unlimited','1.5'])
def test_invalid_deployment_limit_rejected(monkeypatch,value):
    monkeypatch.setenv('ECODATA_ARCHIVE_LIMIT_GIB',value)
    with pytest.raises(ValueError,match='正整数'):module.deployment_limit()


def test_no_fixed_default_limit_and_explicit_server_override(monkeypatch):
    monkeypatch.delenv('ECODATA_ARCHIVE_LIMIT_GIB',raising=False)
    assert module.deployment_limit() is None
    monkeypatch.setenv('ECODATA_ARCHIVE_LIMIT_GIB','32')
    assert module.deployment_limit()==32*1024**3


def test_disk_guard_prevents_payload_write(monkeypatch,tmp_path):
    monkeypatch.setattr(module.psutil,'disk_usage',lambda _:SimpleNamespace(free=512*1024**2))
    reader=module._Reader(tmp_path)
    with pytest.raises(ValueError,match='磁盘空间不足'):reader.copy(io.BytesIO(b'x'),tmp_path/'never',1)
    assert not (tmp_path/'never').exists()


def test_tar_index_fails_before_payload_and_records_reason(monkeypatch,tmp_path):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as archive:
        info=tarfile.TarInfo('large.dat');info.size=2048
        archive.addfile(info,io.BytesIO(b'x'*2048))
    original=tmp_path/'input.zip'
    with zipfile.ZipFile(original,'w') as z:z.writestr('case.tgz',raw.getvalue())
    monkeypatch.setattr(module,'deployment_limit',lambda:1024)
    with pytest.raises(ValueError,match='累计展开'):
        module.normalize_archives([LocalUpload(original.name,original)],tmp_path)
    failure=json.loads((tmp_path/'archive_intake_failure.json').read_text(encoding='utf-8'))
    assert failure['archive_indexes'][0]['declared_regular_bytes']==2048
    assert not failure['model_executed']
    assert len(list((tmp_path/'archive_intake').iterdir()))==1
