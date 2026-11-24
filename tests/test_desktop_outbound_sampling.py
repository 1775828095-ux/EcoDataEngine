import json
from pathlib import Path

import pytest

from ecodata.desktop import sample_values
from ecodata.desktop.file_tools import inspect


def test_netcdf_outbound_samples_cover_multiple_time_windows(tmp_path):
    import numpy as np
    from netCDF4 import Dataset

    path = tmp_path / "rain.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 240)
        dataset.createDimension("y", 4)
        dataset.createDimension("x", 4)
        rain = dataset.createVariable("rain", "f4", ("time", "y", "x"))
        rain[:] = np.broadcast_to(np.arange(240)[:, None, None], (240, 4, 4))

    row = {"name": path.name, "format": "NetCDF", "ref": (path, "", None, None),
           "bytes": path.stat().st_size}
    sample = sample_values.field_sample(row, {"name": "rain"}, outbound=True)
    assert sample[:8] == [0, 74, 149, 224, 1, 75, 150, 225]
    assert len(sample) == 64
    assert len(sample_values.export_sample(sample)["perturbed_samples"]) == 16


@pytest.mark.parametrize("archive, original_reader", [("zip", False), ("", True)])
def test_outbound_skips_large_reads_that_would_materialize_the_whole_file(
    monkeypatch, archive, original_reader
):
    row = {"name": "rain.nc", "format": "NetCDF", "ref": (Path("source"), archive, None, None),
           "bytes": sample_values.MAX_SAMPLE_MATERIALIZATION + 1, "original_reader": original_reader}
    monkeypatch.setattr(sample_values, "local_member", lambda _: pytest.fail("whole-file read"))
    assert sample_values.field_sample(row, {"name": "rain"}, outbound=True) is None


def test_file_tool_sends_csv_header_and_samples_only_eligible_columns(tmp_path):
    path = tmp_path / "stations.csv"
    path.write_text("station_id,rain\n123456,42\n234567,43\n", encoding="utf-8")
    header = inspect(path, "read_header", {"path": ""})
    assert header["header_lines"] == ["station_id,rain"]
    with pytest.raises(ValueError, match="read_header"):
        inspect(path, "read_text", {"path": ""})

    row = header["row"]
    station_id, rain = row["fields"]
    assert sample_values.field_sample(row, station_id, outbound=True) is None
    assert sample_values.field_sample(row, rain, outbound=True) == [42, 43]


def test_default_desktop_flow_sends_values_only_after_field_request(tmp_path, monkeypatch):
    from ecodata.desktop import discovery, service
    from ecodata.desktop.file_agent import run, scan_index

    path = tmp_path / "rain.csv"
    path.write_text("rain\n42\n43\n", encoding="utf-8")
    scan, _ = scan_index(path)
    summary, _ = discovery.discover(scan, "swat_plus")
    turns = iter([
        {"role": "assistant", "tool_calls": [
            {"id": "header", "function": {"name": "read_header", "arguments": '{"path":""}'}}
        ]},
        {"role": "assistant", "content": "{}"},
    ])

    def tool_turn(messages, *args):
        for message in messages:
            if message["role"] == "tool":
                result = json.loads(message["content"])["result"]
                assert result["header_lines"] == ["rain"]
        return next(turns), {}

    requests = []

    def send(request, *args, **kwargs):
        payload = json.loads(request.summary)
        if request.stage == "file_classification":
            return {"classifications": [{"file": file["id"], "kind": "data"} for file in payload["files"]]}
        requests.append(payload)
        if len(requests) == 1:
            assert "local_check_results" not in payload
            fid = payload["sources"][0]["fields"][0]["id"]
            return {"decisions": [], "questions": [{"check": "numeric_samples", "fields": [fid]}]}
        values = payload["local_check_results"]["results"][0]["perturbed_samples"]
        assert len(values) == 2 and 42 < values[0] < 43 and 43 < values[1] < 44
        return {"decisions": [], "questions": []}

    monkeypatch.setattr(service, "tool_turn", tool_turn)
    monkeypatch.setattr(service, "send", send)
    reply, *_ = run(scan, summary, {}, False, "local-test", lambda _: None, lambda _: None)
    assert len(requests) == 2
    assert reply["local_checks"][0]["status"] == "checked"
def test_directional_tools_keep_observations_local(tmp_path):
    import json
    from ecodata.desktop.file_agent import FileTools

    source = tmp_path / "rain.pcp"
    source.write_text("year,day,rain (mm)\n2020,12,87654.321\n", encoding="utf-8")
    report = tmp_path / "README.md"
    report.write_text("# Engineering setup\n\nUse UTC for daily rainfall.\n\n"
                      "| time | rain |\n|---|---|\n| 2020-01-12 | 87654.321 |\n\n"
                      "Example measurement 65432.109 mm and .635309mm. 示例降水53647.901毫米。\n", encoding="utf-8")
    tools = FileTools(tmp_path)
    header = tools.call("read_header", {"path": source.name})["result"]
    assert header["header_lines"] == ["year,day,rain (mm)"]
    assert "ref" not in header["row"] and "87654.321" not in json.dumps(header, default=str)
    assert tools.call("read_text", {"path": source.name})["ok"] is False
    document = tools.call("read_text", {"path": report.name})["result"]
    assert "Use UTC for daily rainfall." in document["text"]
    text = json.dumps(document, default=str)
    assert all(original not in text for original in (
        "87654.321", "65432.109", ".635309", "53647.901", "2020-01-12", "raw_base64"))
    assert "87654.321" in tools.documents[report.name]


def test_on_demand_numeric_request_needs_a_sampling_partition(tmp_path):
    from ecodata.desktop.sampling_plan import export_field

    path = tmp_path / "rain.csv"
    path.write_text("rain\n12.5\n", encoding="utf-8")
    row = dict(format="CSV", original_reader="legacy", ref=(path, "", None, None))
    sample = export_field(row, {"name": "rain"}, require_partition=True)
    assert sample["basis"] == "holdout_partition_unavailable" and sample["perturbed_samples"] == []
