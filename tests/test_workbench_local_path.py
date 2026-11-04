from io import BytesIO
from pathlib import Path
import zipfile

import pytest

from ecodata.interface.engineering_upload import local_path_uploads, scan_model_uploads


@pytest.mark.parametrize("suffix", [".csv", ".zip"])
def test_local_file_uses_same_bytes_and_archive_scan_as_upload(tmp_path, suffix):
    path = tmp_path / ("本地数据" + suffix)
    payload = b"time,value\n2020-01-01,1\n"
    if suffix == ".zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("source/data.csv", payload)
    else:
        path.write_bytes(payload)
    local = local_path_uploads(f' "{path}" ', include_all=True)
    manual = BytesIO(path.read_bytes())
    manual.name = path.name
    assert len(local) == 1
    assert local[0].getvalue() == manual.getvalue()
    assert scan_model_uploads(local).files == scan_model_uploads([manual]).files


def test_local_folder_keeps_relative_source_paths(tmp_path):
    data = tmp_path / "来源" / "data.csv"
    data.parent.mkdir()
    data.write_bytes(b"x,y\n1,2\n")
    assert [u.name for u in local_path_uploads(tmp_path, include_all=True)] == ["来源/data.csv"]
    with pytest.raises(ValueError, match="完整路径"):
        local_path_uploads(tmp_path / "missing.zip")


def test_ui_reads_zip_path_without_manual_upload_and_clears_stale_result(tmp_path):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "本地数据.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.csv", "time,value\n2020-01-01,1\n")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/ecodata/interface/app.py").run(timeout=30)
    app.text_input[0].set_value(str(path)).run()
    assert not app.exception and not app.error
    assert not app.button[0].disabled
    assert any("1" in item.value and "自动读取" in item.value for item in app.caption)
    app.session_state["workbench_result"] = dict(identity="old-result")
    app.text_input[0].set_value(str(tmp_path / "missing.zip")).run()
    assert app.button[0].disabled and app.error
    assert "workbench_result" not in app.session_state
