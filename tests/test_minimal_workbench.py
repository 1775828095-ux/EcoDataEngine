from io import BytesIO
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.models import conversion_bridge


def upload(index=0, missing=False, unit="degC"):
    lat = np.linspace(30, 31, 4)
    lon = np.linspace(110, 111, 4)
    values = 20 + np.arange(160).reshape(10, 4, 4) * .01 + index
    if missing:
        values[3:5, 1, 1] = np.nan
    ds = xr.Dataset(dict(air_temperature_max=(("time", "lat", "lon"), values)),
        coords=dict(time=pd.date_range("2000-01-01", periods=10), lat=lat, lon=lon),
        attrs=dict(crs="EPSG:4326", source_id=f"source{index}"))
    ds.air_temperature_max.attrs.update(units=unit, physical_variable="air_temperature_max",
        physical_definition="daily maximum air temperature",
        spatial_semantics="grid_estimate", temporal_semantics="instantaneous", data_kind="continuous")
    stream = BytesIO(ds.to_netcdf())
    stream.name = f"source{index}.nc"
    return stream


def test_complete_upload_reuses_units_and_prepares_portable_handoff(tmp_path, monkeypatch):
    # This case tests converter absence, independently of installed plugins.
    monkeypatch.setattr(conversion_bridge, 'entry_points', lambda **kw: [])
    result = prepare_uploads([upload()], "swat_plus", tmp_path)
    assert result["manifest"]["status"] == "AWAITING_CONVERTER"
    assert not result["manifest"]["native_ready"]
    assert result["manifest"]["preparation"]["tasks"][0]["result"]["status"] == "UNCHANGED_COMPLETE_INPUT"
    with zipfile.ZipFile(result["archive"]) as bundle:
        manifest = json.loads(bundle.read("handoff.json"))
        assert bundle.testzip() is None
        for product in manifest["products"]:
            assert product["path"] in bundle.namelist()


def test_unknown_unit_cannot_be_invented(tmp_path):
    with pytest.raises(Exception, match="单位"):
        prepare_uploads([upload(unit="unknown")], "swat_plus", tmp_path)


def test_wrong_model_does_not_silently_relabel_quantity(tmp_path):
    with pytest.raises(ValueError, match="未识别"):
        prepare_uploads([upload()], "wasp", tmp_path)


def test_converter_seam_calls_registered_converter_and_checks_files(tmp_path, monkeypatch):
    class Plugin:
        def load(self):
            def convert(manifest, target):
                assert json.loads(manifest.read_text(encoding="utf-8"))["model_id"] == "swat_plus"
                (target / "test-output.txt").write_text("test fixture, not a native simulator claim")
                return dict(status="PASS", files=["test-output.txt"], validation=dict(test_fixture=True))
            return convert
    monkeypatch.setattr(conversion_bridge, "entry_points", lambda **kw: [Plugin()])
    result = prepare_uploads([upload()], "swat_plus", tmp_path)
    assert result["manifest"]["status"] == "CONVERTED"


def test_invalid_converter_never_becomes_ready(tmp_path, monkeypatch):
    class Plugin:
        def load(self):
            return lambda *_: dict(status="PASS", files=["../escape.txt"], validation=dict(test=True))
    monkeypatch.setattr(conversion_bridge, "entry_points", lambda **kw: [Plugin()])
    result = prepare_uploads([upload()], "swat_plus", tmp_path)
    assert result["manifest"]["status"] == "CONVERSION_FAILED"
    assert not result["manifest"]["native_ready"]


def test_real_multisource_public_random_search(tmp_path, monkeypatch):
    from ecodata.algorithms import platform_search
    monkeypatch.setattr(platform_search, "CURRENT_INPUT_BUDGET", dict(objective_limit=100,
        total_call_limit=160, proposal_limit=100, wall_seconds=35, probe_call_limit=30))
    result = prepare_uploads([upload(), upload(1)], "swat_plus", tmp_path)
    task = result["manifest"]["preparation"]["tasks"][0]["result"]
    assert task["status"] == "PASS"
    assert task["optimizer"]["backend"] == "random"
    assert task["total_cost"]["phases"]["deploy"] > 0
    assert task["confirmation"]["used_for_selection"] is False


def test_ui_has_one_model_selector_no_developer_navigation():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/ecodata/interface/app.py").run(timeout=30)
    assert not app.exception
    assert len(app.selectbox) == 1
    assert len(app.radio) == 0
    assert len(app.slider) == 0
    assert len(app.number_input) == 0
    assert app.button[0].disabled
