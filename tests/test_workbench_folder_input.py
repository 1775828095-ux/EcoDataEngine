from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.interface.engineering_upload import local_folder_uploads, scan_model_uploads


def source_file(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "最高温.nc"
    ds = xr.Dataset(dict(air_temperature_max=(("time", "lat", "lon"), np.full((3, 2, 2), 20.))),
        coords=dict(time=pd.date_range("2000-01-01", periods=3), lat=[30., 31.], lon=[110., 111.]),
        attrs=dict(crs="EPSG:4326", source_id="observed"))
    ds.air_temperature_max.attrs.update(units="degC", physical_variable="air_temperature_max",
        physical_definition="daily maximum air temperature", spatial_semantics="grid_estimate",
        temporal_semantics="instantaneous", data_kind="continuous")
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def test_folder_recurses_and_preserves_relative_names(tmp_path):
    source_file(tmp_path / "站点甲")
    (tmp_path / "说明.md").write_text("not data", encoding="utf-8")
    source_file(tmp_path / "references")
    files = local_folder_uploads(f'"{tmp_path}"')
    assert [f.name for f in files] == ["站点甲/最高温.nc"]
    assert len(scan_model_uploads(files).files) == 1


@pytest.mark.parametrize("kind", ["missing", "empty", "file"])
def test_invalid_folder_is_actionable(tmp_path, kind):
    path = tmp_path / kind
    if kind == "empty":
        path.mkdir()
    if kind == "file":
        path.write_text("not a directory")
    with pytest.raises(ValueError, match="文件夹"):
        local_folder_uploads(path)


def test_folder_and_zip_use_same_preparation_pipeline(tmp_path, monkeypatch):
    from ecodata.models import conversion_bridge
    monkeypatch.setattr(conversion_bridge, 'entry_points', lambda **kw: [])
    source = source_file(tmp_path / "输入" / "嵌套")
    zip_data = BytesIO()
    with zipfile.ZipFile(zip_data, "w") as archive:
        archive.writestr("嵌套/最高温.nc", source.read_bytes())
    zip_data.name = "数据.zip"
    folder_result = prepare_uploads(local_folder_uploads(tmp_path / "输入"), "swat_plus", tmp_path / "folder-result")
    zip_result = prepare_uploads([zip_data], "swat_plus", tmp_path / "zip-result")
    for result in [folder_result, zip_result]:
        assert result["manifest"]["status"] == "AWAITING_CONVERTER"
        with zipfile.ZipFile(result["archive"]) as bundle:
            assert bundle.testzip() is None
    with xr.open_dataset(tmp_path / "folder-result/field-1/product.nc", engine="h5netcdf") as a, \
         xr.open_dataset(tmp_path / "zip-result/field-1/product.nc", engine="h5netcdf") as b:
        xr.testing.assert_equal(a.air_temperature_max, b.air_temperature_max)


def test_folder_drives_real_ui_and_stale_result_is_cleared(tmp_path, monkeypatch):
    from ecodata.models import conversion_bridge
    monkeypatch.setattr(conversion_bridge, 'entry_points', lambda **kw: [])
    from streamlit.testing.v1 import AppTest
    source_file(tmp_path / "用户数据")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/ecodata/interface/app.py").run(timeout=30)
    app.text_input[0].set_value(str(tmp_path / "用户数据")).run()
    assert not app.exception and not app.button[0].disabled
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["workbench_result"]["data"]["manifest"]["status"] == "AWAITING_CONVERTER"
    app.text_input[0].set_value(str(tmp_path / "不存在")).run()
    assert app.button[0].disabled and app.error
    assert "workbench_result" not in app.session_state
