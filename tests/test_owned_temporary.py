"""Filesystem lifecycle tests, not evidence of environmental prediction accuracy."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ecodata.core import temporary


@pytest.fixture
def owned_root(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    return temporary.scratch_root()


def abandon():
    script="""from ecodata.core.temporary import TemporaryDirectory
from pathlib import Path
import os
t=TemporaryDirectory(prefix='test-crash-')
(Path(t.name)/'scratch.bin').write_bytes(b'x'*8192)
print(t.name,flush=True)
os._exit(0)
"""
    return Path(subprocess.check_output([sys.executable,'-c',script],text=True).strip())


def test_normal_scope_and_exception_remove_only_their_files(owned_root,tmp_path):
    original=tmp_path/'original.nc';original.write_bytes(b'original')
    with pytest.raises(RuntimeError):
        with temporary.TemporaryDirectory() as folder:
            path=Path(folder);(path/'work').write_bytes(b'temporary')
            raise RuntimeError('test failure')
    assert not path.exists() and original.read_bytes()==b'original'


def test_real_crash_next_start_reclaims_but_preserves_active_and_unknown(owned_root):
    dead=abandon()
    with temporary.TemporaryDirectory(prefix='active-') as live:
        unknown=owned_root/'unknown';unknown.mkdir();(unknown/'user').write_bytes(b'keep')
        report=temporary.sweep_abandoned()
        assert not dead.exists() and Path(live).exists() and unknown.exists()
        assert report['released_bytes']>=8192 and len(report['removed'])==1
        assert report['retained'] and report['failures']


def test_pid_reuse_not_mistaken_for_live_owner(owned_root):
    dead=abandon();marker=dead/temporary.MARKER
    data=json.loads(marker.read_text());data['pid']=os.getpid();data['created']=1
    marker.write_text(json.dumps(data))
    assert temporary.sweep_abandoned()['removed'] and not dead.exists()


def test_locked_file_failure_is_retained_and_retried(owned_root,monkeypatch):
    dead=abandon();original=temporary._remove
    monkeypatch.setattr(temporary,'_remove',lambda p:(_ for _ in ()).throw(PermissionError('occupied')))
    report=temporary.sweep_abandoned()
    assert report['failures'] and dead.exists() and not report['removed']
    monkeypatch.setattr(temporary,'_remove',original)
    assert temporary.sweep_abandoned()['removed'] and not dead.exists()


def test_invalid_marker_and_link_are_not_deletion_authority(owned_root,monkeypatch):
    dead=abandon();marker=dead/temporary.MARKER
    data=json.loads(marker.read_text());data['directory']='someone-else'
    marker.write_text(json.dumps(data))
    assert temporary.sweep_abandoned()['failures'] and dead.exists()
    data['directory']=dead.name;marker.write_text(json.dumps(data))
    linked=dead/'external';linked.write_bytes(b'protected')
    check=temporary._linked
    monkeypatch.setattr(temporary,'_linked',lambda p:p==linked or check(p))
    assert temporary.sweep_abandoned()['failures'] and linked.exists()
    monkeypatch.setattr(temporary,'_linked',check)
    temporary.sweep_abandoned()


def test_uncertain_process_status_retains_directory(owned_root,monkeypatch):
    dead=abandon();original=temporary.psutil.Process
    monkeypatch.setattr(temporary.psutil,'Process',lambda pid:(_ for _ in ()).throw(temporary.psutil.AccessDenied(pid)))
    assert temporary.sweep_abandoned()['failures'] and dead.exists()
    monkeypatch.setattr(temporary.psutil,'Process',original)
    temporary.sweep_abandoned()
