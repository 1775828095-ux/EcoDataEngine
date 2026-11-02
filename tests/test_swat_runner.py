from datetime import date
import io
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.interface import swat_runner as r


def files():
    # Synthetic minimal routing fixture. It is NOT a runnable SWAT watershed.
    return {"file.cio": b"synthetic IO fixture\nsimulation time.sim print.prt null\nclimate null null null pcp.cli tmp.cli null null null\n",
            "time.sim": b"synthetic\nday_start yrc_start day_end yrc_end step\n0 2020 0 2020 0\n",
            "print.prt": b"fixture\n", "pcp.cli": b"fixture\nfilename\nstation.pcp\n",
            "tmp.cli": b"fixture\nfilename\nstation.tmp\n",
            "station.pcp": b"fixture\n", "station.tmp": b"fixture\n"}


def zipped(content, prefix=""):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, data in content.items():
            archive.writestr(prefix + name, data)
    return out.getvalue()


@pytest.mark.parametrize("prefix", ["", "Case/TxtInOut/"])
def test_project_layout(prefix):
    data = r.read_project(zipped(files(), prefix))
    assert data == files()
    assert not r.check_project(data)["errors"]


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "CON", "file.", "run.exe", "run.ps1"])
def test_unsafe_upload(name):
    with pytest.raises(SchemaError):
        r.read_project(zipped({**files(), name: b"x"}))


def test_weather_only_and_ambiguous():
    with pytest.raises(SchemaError, match="file.cio"):
        r.read_project(zipped({"station.pcp": b"x"}))
    with pytest.raises(SchemaError, match="file.cio"):
        r.read_project(zipped({"a/file.cio": b"x", "b/file.cio": b"y"}))


def test_reference_and_date_errors():
    data = files()
    data.pop("station.pcp")
    assert any("station.pcp" in x for x in r.check_project(data)["errors"])
    data = files()
    data["file.cio"] += b"hru ../outside\n"
    assert any("外部路径" in x for x in r.check_project(data)["errors"])
    data = files()
    data["time.sim"] = b"title\nfields\n366 2021 0 2021 0\n"
    assert r.check_project(data)["errors"]


def test_dates_are_only_changed_in_copy():
    source = files()
    converted = r.set_dates(source, date(2020, 2, 29), date(2020, 3, 1))
    assert converted["time.sim"].splitlines()[2] == b"60 2020 61 2020 0"
    assert source["time.sim"].splitlines()[2] == b"0 2020 0 2020 0"
    assert all(converted[k] == v for k, v in source.items() if k != "time.sim")
    with pytest.raises(SchemaError):
        r.set_dates(source, date(2021, 1, 2), date(2021, 1, 1))


def test_legacy_basin_record_cannot_shift_weather_references():
    data = files()
    data.update({'codes.bsn': b'fixture', 'parameters.bsn': b'fixture'})
    data['file.cio'] += b'basin codes.bsn parameters.bsn\n'
    assert any('basin' in message and '3' in message for message in r.check_project(data)['errors'])
    with pytest.raises(SchemaError, match='basin'):
        r.start_job(data, Path('unused_schema_failure'))
    data['file.cio'] = data['file.cio'].replace(b'basin codes.bsn parameters.bsn\n', b'basin codes.bsn parameters.bsn null\n')
    assert not r.check_project(data)['errors']


def test_active_carbon_file_still_requires_real_file():
    data = files()
    data.update({'codes.bsn': b'fixture', 'parameters.bsn': b'fixture'})
    data['file.cio'] += b'basin codes.bsn parameters.bsn carbon.bsn\n'
    assert any('carbon.bsn' in message for message in r.check_project(data)['errors'])


def test_rev6057_exact_engine_and_no_silent_migration():
    data = files()
    data['file.cio'] = data['file.cio'].replace(b'synthetic IO fixture', b'written for SWAT+ rev.60.5.7')
    data.update({'codes.bsn': b'fixture', 'parameters.bsn': b'fixture'})
    data['file.cio'] += b'basin codes.bsn parameters.bsn\n'
    original = dict(data)
    assert not r.check_project(data, 'official60')['errors']
    assert any('版本不匹配' in x for x in r.check_project(data, 'editor62')['errors'])
    assert data == original
    data['file.cio'] = data['file.cio'].replace(b'60.5.7', b'60.5.6')
    assert any('补丁版本' in x for x in r.check_project(data, 'official60')['errors'])


def test_declared_legacy_version_and_engine_mismatch(tmp_path):
    data = files()
    data['file.cio'] = data['file.cio'].replace(b'synthetic IO fixture', b'written for SWAT+ rev.61.0.1')
    data.update({'codes.bsn': b'fixture', 'parameters.bsn': b'fixture'})
    data['file.cio'] += b'basin codes.bsn parameters.bsn\n'
    original = dict(data)
    assert not r.check_project(data, engine_id='official61')['errors']
    assert r.check_project(data)['target_version'] == '61'
    assert any('版本不匹配' in x for x in r.check_project(data, engine_id='editor62')['errors'])
    with pytest.raises(SchemaError, match='版本不匹配'):
        r.start_job(data, tmp_path, engine_id='editor62')
    assert data == original
    data['file.cio'] = data['file.cio'].replace(b'rev.61.0.1', b'rev.62.0.0')
    assert any('版本不匹配' in x for x in r.check_project(data, engine_id='official61')['errors'])


def test_unregistered_declared_version_is_rejected():
    data = files()
    data['file.cio'] = data['file.cio'].replace(b'synthetic IO fixture', b'written for SWAT+ rev.59.0.0')
    assert any('仅登记60.5.7/61/62' in x for x in r.check_project(data)['errors'])


def test_legacy_generated_weather_does_not_require_unused_indexes():
    data = files()
    data['file.cio'] = b'written for SWAT+ rev.61.0.1\nsimulation time.sim print.prt null\nclimate weather-sta.cli weather-wgn.cli null pcp.cli tmp.cli null hmd.cli wnd.cli null\n'
    data['weather-sta.cli'] = b'title\nname wgn pcp tmp slr hmd wnd pet atmo_dep\nstation wgn station.pcp station.tmp null sim sim null null\n'
    data['weather-wgn.cli'] = ('fixture\nwgn 1 2 3 10\n' + ' '.join('field'+str(i) for i in range(14))+'\n'+(' '.join(['1']*14)+'\n')*12).encode()
    assert not r.check_project(data, engine_id='official61')['errors']
    assert any('WGN' in x for x in r.check_project(data)['warnings'])
    data['weather-sta.cli'] = data['weather-sta.cli'].replace(b'sim sim', b'missing.hmd sim')
    assert any('hmd.cli' in x for x in r.check_project(data)['errors'])


def wait(ident):
    until = time.monotonic() + 8
    while r.job_status(ident)["state"] == "running" and time.monotonic() < until:
        time.sleep(.02)
    return r.job_status(ident)


@pytest.mark.parametrize("exitcode,success,expected", [(0, True, "succeeded"), (2, False, "failed"), (0, False, "failed"), (2, True, "failed")])
def test_job_status_and_stale_success(tmp_path, exitcode, success, expected):
    trusted = tmp_path / 'trusted.exe'
    trusted.write_bytes(b'synthetic executable fixture; never executed')
    class Process:
        pid = 123
        def __init__(self, args, **kw):
            assert kw["shell"] is False
            assert not (kw["cwd"] / "success.fin").exists()
            kw["stdout"].write(b"synthetic process test; NOT SWAT\n")
            if success:
                (kw["cwd"] / "success.fin").write_text("fresh marker")
        def wait(self, timeout=None): return exitcode
    data = {**files(), "success.fin": b"old success"}
    with patch.object(r, "engine_path", return_value=trusted), patch.object(r.subprocess, "Popen", Process):
        ident = r.start_job(data, tmp_path)
        status = wait(ident)
    assert status["state"] == expected
    assert status["returncode"] == exitcode
    with zipfile.ZipFile(io.BytesIO(r.result_zip(ident))) as z:
        assert "runner.log" in z.namelist() and "runner_status.json" in z.namelist()
        assert "station.pcp" not in z.namelist()  # Unchanged inputs aren't presented as output.
        assert json.loads(z.read("runner_status.json"))["state"] == expected


def test_timeout(tmp_path):
    trusted = tmp_path / 'trusted.exe'
    trusted.write_bytes(b'synthetic executable fixture; never executed')
    class Process:
        pid = 123
        killed = False
        def __init__(self, *a, **kw): pass
        def wait(self, timeout=None):
            if timeout is not None: raise subprocess.TimeoutExpired("fixture", timeout)
            return -1
        def kill(self): self.killed = True
    with patch.object(r, "engine_path", return_value=trusted), patch.object(r.subprocess, "Popen", Process):
        ident = r.start_job(files(), tmp_path, timeout=1)
        assert wait(ident)["state"] == "timeout"


def test_stop_and_concurrency(tmp_path):
    trusted = tmp_path / 'trusted.exe'
    trusted.write_bytes(b'synthetic executable fixture; never executed')
    stopped = threading.Event()
    class Process:
        pid = 123
        def __init__(self, *a, **kw): pass
        def wait(self, timeout=None): stopped.wait(5); return -1
        def poll(self): return None
        def terminate(self): stopped.set()
    with patch.object(r, "engine_path", return_value=trusted), patch.object(r.subprocess, "Popen", Process):
        ident = r.start_job(files(), tmp_path)
        with pytest.raises(SchemaError, match="已有"):
            r.start_job(files(), tmp_path)
        r.stop_job(ident)
        assert wait(ident)["state"] == "stopped"


def test_registered_engine_validation(tmp_path, monkeypatch):
    import hashlib
    trusted = tmp_path / 'official-fixture.exe'
    trusted.write_bytes(b'synthetic hash fixture; not a real model')
    monkeypatch.setattr(r, 'EDITOR_ENGINE', trusted)
    monkeypatch.setattr(r, 'EDITOR_ENGINE_SHA256', hashlib.sha256(trusted.read_bytes()).hexdigest())
    assert r.engine_path('editor62') == trusted
    with pytest.raises(SchemaError, match='未知'):
        r.engine_path('uploaded.exe')
    trusted.write_bytes(b'changed')
    with pytest.raises(SchemaError, match='校验失败'):
        r.engine_path('editor62')


def test_selected_engine_is_recorded(tmp_path):
    import hashlib
    trusted = tmp_path / 'official-fixture.exe'
    trusted.write_bytes(b'synthetic manifest fixture')
    with patch.object(r, 'engine_path', return_value=trusted) as lookup, patch.object(r, '_work'):
        ident = r.start_job(files(), tmp_path / 'jobs', engine_id='editor62')
        manifest = json.loads((Path(r.job_status(ident)['directory']) / 'runner_manifest.json').read_text(encoding='utf-8'))
        lookup.assert_called_once_with('editor62')
        assert manifest['engine_id'] == 'editor62'
        assert manifest['engine_sha256'] == hashlib.sha256(trusted.read_bytes()).hexdigest()
        with r._LOCK:
            r._JOBS[ident]['state'] = 'stopped'


def test_page_upload_error_and_start(tmp_path):
    mutable = {"payload": zipped(files())}
    # The optional weather upload is independent of the complete project.
    # Do not accidentally return the project ZIP for both upload controls.
    upload = lambda *a, **kw: (SimpleNamespace(getvalue=lambda: mutable["payload"])
                              if kw.get('key') == 'swat_project_upload' else None)
    source = "from ecodata.interface.swat_run_page import render_swat_run_page\nrender_swat_run_page()"
    with patch("streamlit.file_uploader", side_effect=upload), patch.object(r, "recent_jobs", return_value=[]), patch.object(r, "engine_path", return_value=Path("trusted.exe")), patch.object(r, "start_job", return_value="ui-test") as start, patch("ecodata.interface.swat_run_page._render_job") as render:
        app = AppTest.from_string(source, default_timeout=20).run()
        assert not app.exception and not app.error
        app.checkbox[0].check().run()
        next(b for b in app.button if b.label == "启动 SWAT+").click().run()
        assert start.call_count == 1 and render.call_args.args[0] == "ui-test"
        assert b"1 2020 366 2020 0" in start.call_args.args[0]["time.sim"]
        incomplete = files(); incomplete.pop("station.pcp")
        mutable["payload"] = zipped(incomplete)
        app.run()
        assert any("station.pcp" in e.value for e in app.error)
        assert not [b for b in app.button if b.label == "启动 SWAT+"]
