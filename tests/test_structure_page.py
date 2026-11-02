from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_upload_csv_to_grid_netcdf_through_page():
    content = b"longitude,latitude,value,station_id,qc_flag\n110,30,1,001,VALID\n111,30,2,002,VALID\n110,31,3,003,VALID\n111,31,4,004,VALID\n"
    upload = SimpleNamespace(name="grid.csv", getvalue=lambda: content)
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(x for x in app.text_input if x.label == "CRS").set_value("EPSG:4326")
        next(x for x in app.text_input if x.label == "value").set_value("ug m-3").run()
        next(x for x in app.button if x.label == "执行数据转换").click().run()
        assert not app.exception
        assert not app.error
        result = app.session_state["task_result"]
        assert result.output.value.shape == (2, 2)
        assert result.output.station_id.values[0, 0] == "001"
        assert len(app.get("download_button")) == 5
        next(x for x in app.selectbox if x.label == "输出格式").set_value("GeoTIFF").run()
        assert not app.get("download_button")
        next(x for x in app.button if x.label == "执行数据转换").click().run()
        assert not app.error
        assert not app.exception
        assert app.session_state["task_result"].export("GeoTIFF")[1].endswith(".tif")


def test_non_spatial_table_does_not_show_spatial_controls():
    content = "断面名称,监测时间,自动监测站编码,总磷,总氮\n站点A,2021-02-01 00:00:00,A001,0.1,2.0\n".encode("utf-8-sig")
    upload = SimpleNamespace(name="water.csv", getvalue=lambda: content)
    with patch("streamlit.file_uploader", return_value=[upload]):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert not [item for item in app.selectbox if item.label == "数据结构转换"]
    assert not [item for item in app.text_input if item.label == "CRS"]
    assert not [item for item in app.number_input if item.label.startswith("源空间分辨率")]


def test_qp_algorithm_and_iteration_controls_are_visible_for_valid_uploads():
    data = ROOT / "examples" / "datasets"
    coarse = data / "p3_pm25_coarse_10km_48h.nc"
    stations = data / "p3_pm25_train_stations.csv"
    uploads = [
        SimpleNamespace(name=coarse.name, getvalue=coarse.read_bytes),
        SimpleNamespace(name=stations.name, getvalue=stations.read_bytes),
    ]
    with patch("streamlit.file_uploader", return_value=uploads):
        app = AppTest.from_file(str(ROOT / "src/ecodata/interface/app.py"), default_timeout=30).run()
        next(x for x in app.number_input if x.label.startswith("目标空间分辨率")).set_value(2.5).run()
        algorithm = next(x for x in app.selectbox if x.label == "空间转换算法")
        assert any("可信峰值约束" in option for option in algorithm.options)
        algorithm.set_value("自适应平滑＋可信峰值约束 QP").run()
    assert not app.exception
    assert next(x for x in app.selectbox if x.label == "训练监测站文件").value == 0
    assert next(x for x in app.selectbox if x.label == "优化求解器").options == ["OSQP", "CLARABEL", "SCS"]
    assert next(x for x in app.number_input if x.label == "最大迭代次数").value == 20000
    assert next(x for x in app.selectbox if x.label == "初始高分辨率场").value == "bilinear"
