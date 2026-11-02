import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.interface.downloads import package_results
from ecodata.io.structures import export_structure, read_structure, table_dataset
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("formats", [("csv", "csv"), ("csv", "nc"), ("nc", "nc")])
def test_multiple_inputs_generate_downloadable_complete_zip(formats):
    uploads = []
    for index, suffix in enumerate(formats):
        frame = pd.DataFrame({"station_id": ["001", "002"], "value": [1.1 + index, 2.2 + index]})
        if suffix == "csv":
            content = frame.to_csv(index=False).encode("utf-8")
        else:
            dataset = table_dataset(frame)
            dataset.value.attrs["units"] = "mg/L"
            content = export_structure(dataset, "NetCDF")[0]
        uploads.append(SimpleNamespace(name=f"source_{index}.{suffix}", getvalue=lambda content=content: content))
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(item for item in app.button if item.label == "执行数据转换").click().run()
    assert not app.exception
    assert not app.error
    downloads = app.get("download_button")
    assert any(item.label == "下载全部结果(zip)" for item in downloads)
    with zipfile.ZipFile(io.BytesIO(app.session_state["task_zip"])) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        outputs = [name for name in names if name.endswith(".nc")]
        assert len(outputs) == 2
        assert len(names) == 13
        for index in (1, 2):
            for report in ("before_profile.json", "after_profile.json", "validation_report.json"):
                assert f"{index:02d}_{report}" in names
        assert json.loads(archive.read("failures.json")) == []
        assert all(archive.getinfo(name).file_size > 0 for name in names)
        assert len(json.loads(archive.read("conversion_report.json"))) == 2
        assert len(json.loads(archive.read("data_recognition_report.json"))) == 2
        for index, name in enumerate(outputs):
            restored = read_structure(name, archive.read(name))
            np.testing.assert_allclose(restored.value, [1.1 + index, 2.2 + index])
            assert restored.station_id.values.tolist() == ["001", "002"]


@pytest.mark.parametrize("invalid", [b"", None])
def test_package_rejects_missing_or_empty_result(invalid):
    with pytest.raises(SchemaError, match="未生成或为空"):
        package_results([("one.nc", b"one"), ("two.nc", invalid)], {}, {})


def test_package_rejects_duplicate_filenames():
    with pytest.raises(SchemaError, match="重名"):
        package_results([("same.nc", b"one"), ("same.nc", b"two")], {}, {})


def test_single_output_retains_original_downloads():
    upload = SimpleNamespace(name="source.csv", getvalue=lambda: b"station_id,value\n001,0.11\n")
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(item for item in app.button if item.label == "执行数据转换").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.get("download_button")) == 5
    assert not any(item.label == "下载全部结果(zip)" for item in app.get("download_button"))


def test_failed_batch_preserves_successes_and_reports_failures():
    uploads = [SimpleNamespace(name="good.csv", getvalue=lambda: b"value\n0.11\n"),
               SimpleNamespace(name="missing.csv", getvalue=lambda: b"other\n0.22\n")]
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(item for item in app.button if item.label == "执行数据转换").click().run()
    assert not app.exception
    assert not app.error
    assert app.warning
    with zipfile.ZipFile(io.BytesIO(app.session_state["task_zip"])) as archive:
        failures = json.loads(archive.read("failures.json"))
        assert failures[0]["输入文件"] == "missing.csv"
        assert len([name for name in archive.namelist() if name.endswith(".nc")]) == 1
