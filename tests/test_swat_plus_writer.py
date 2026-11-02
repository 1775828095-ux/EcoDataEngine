from pathlib import Path

import pytest

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters import get_adapter
from ecodata.models.adapters.swat_plus import SWATPlusAdapter, SWATPlusWriter


def weather_config():
    return {
        "project_name": "test",
        "start": "2020-01-01",
        "end": "2020-01-03",
        "station": {"name": "sta1", "latitude": 39.0, "longitude": -77.0, "elevation": 100.0},
        "weather": [
            {"time": "2020-01-01", "precipitation": 1.0, "air_temperature_max": 10.0, "air_temperature_min": -2.0},
            {"time": "2020-01-02", "precipitation": 0.0, "air_temperature_max": 11.0, "air_temperature_min": -1.0},
            {"time": "2020-01-03", "precipitation": 2.5, "air_temperature_max": 9.0, "air_temperature_min": -3.0},
        ],
    }


def test_writer_generates_weather_and_candidate_index_without_inventing_topology():
    files = SWATPlusWriter().write(weather_config())
    expected = {"time.sim", "station.pcp", "station.tem", "pcp.cli", "tmp.cli", "weather-sta.cli", "file.cio"}
    assert expected <= set(files)
    assert "object.cnt" not in files
    assert "hru-data.hru" not in files
    assert "2020 1 1.0" in files["station.pcp"].decode()
    assert "2020\t1\t10.0\t-2.0" in files["station.tem"].decode()
    assert "native_model_input_verified" in files["validation_report.json"].decode()


def test_writer_refuses_weather_outside_configured_time():
    config = weather_config()
    config["weather"][0]["time"] = "2019-12-31"
    with pytest.raises(SchemaError, match="连续逐日"):
        SWATPlusWriter().write(config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("hourly", "小时数据"),
        ("duplicate", "重复"),
        ("out_of_order", "乱序"),
        ("missing_day", "连续逐日"),
    ],
)
def test_writer_rejects_hourly_duplicate_disordered_or_missing_days(change, message):
    config = weather_config()
    if change == "hourly":
        config["weather"] = [
            {"time": "2020-01-01 00:00", "precipitation": 0.0, "air_temperature_max": 1.0, "air_temperature_min": 0.0},
            {"time": "2020-01-01 01:00", "precipitation": 1.0, "air_temperature_max": 2.0, "air_temperature_min": 1.0},
        ]
    elif change == "duplicate":
        config["weather"][1] = dict(config["weather"][0])
    elif change == "out_of_order":
        config["weather"] = [config["weather"][1], config["weather"][0], config["weather"][2]]
    elif change == "missing_day":
        config["weather"] = [config["weather"][0], config["weather"][2]]
    with pytest.raises(SchemaError, match=message):
        SWATPlusWriter().write(config)


def test_writer_uses_official_template_for_file_cio(tmp_path):
    template = tmp_path / "b2"
    template.mkdir()
    (template / "file.cio").write_text("file.cio: AMES\names.pcp\names.tem\n", encoding="utf-8")
    files = SWATPlusWriter().write(weather_config(), template)
    text = files["file.cio"].decode()
    assert "file.cio: test" in text
    assert "station.pcp" in text and "station.tem" in text


def test_adapter_is_registered_and_never_claims_native_acceptance():
    class Schema:
        preparation = {"adapter": "swat_plus_text"}

    schema = Schema()
    adapter = get_adapter(schema)
    assert isinstance(adapter, SWATPlusAdapter)
    assert adapter.native_validation(None, schema)["verified"] is False


def test_save_package_does_not_overwrite_existing_files(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SchemaError, match="非空"):
        SWATPlusAdapter().save_package(output, weather_config())
