import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import xarray as xr

from ecodata.desktop import fallback_evidence, release_gate, sampling_plan
from ecodata.desktop.scan import inventory


def source(path, kind, fields):
    return dict(name=path.name, format=kind, ref=(path, "", None, None),
                bytes=path.stat().st_size, fields=fields)


def scan(path, rows):
    return {"path": path, "signature": tuple(inventory(path)), "rows": rows}


def test_precipitation_gate_integrates_heldout_actual_intervals_and_blocks_missing_bounds(tmp_path):
    path = tmp_path / "rain.nc"
    time = np.arange(30)
    ds = xr.Dataset(
        {"rain": ("time", np.ones(30), {"units": "mm/h"}),
         "time_bounds": (("time", "edge"), np.column_stack([time, time + 1]))},
        coords={"time": ("time", time, {"units": "hours since 2020-01-01", "bounds": "time_bounds"})})
    ds.to_netcdf(path)
    field = {"name": "rain", "unit": "mm/h"}
    inputs = scan(path, [source(path, "NetCDF", [field])])
    recipe = dict(task="C1:align_time", steps=["engineering.precipitation_adaptation"])
    fields = [(0, "D1.F1", field)]
    recipe["local_holdout_review"] = sampling_plan.recommendation_review(inputs, recipe, {}, fields)
    assert recipe["local_holdout_review"][0]["precipitation_trial"]["operation"] == "rate_to_interval_amount"
    assert release_gate.assess(inputs, recipe, {}, fields)["status"] == "released"
    recipe["local_holdout_review"][0].pop("precipitation_trial")
    recipe["execution_release"] = release_gate.assess(inputs, recipe, {}, fields)
    assert recipe["execution_release"]["status"] == "pending"
    assert not fallback_evidence.eligible(recipe)
    with pytest.raises(ValueError, match="未放行"):
        release_gate.require([recipe])


def test_runtime_rejects_duplicate_station_time_across_table_chunks(tmp_path):
    path = tmp_path / "rain.csv"
    with path.open("w", encoding="utf-8") as stream:
        stream.write("station_id,time,rain\n")
        for day in range(1, 4097):
            from datetime import date, timedelta
            clock = date(2000, 1, 1) + timedelta(days=day)
            stream.write(f"A,{clock},42\n")
        stream.write("A,2000-01-02,43\n")
    field = dict(name="rain", unit="mm")
    inputs = scan(path, [source(path, "CSV", [field])])
    with pytest.raises(ValueError, match="重复时刻"):
        release_gate.runtime(inputs, {"fields": [(0, "rain")]}, tmp_path / "scratch")


def test_full_fallback_build_then_approved_send_never_includes_identifiers(tmp_path, monkeypatch):
    path = tmp_path / "rain.csv"
    path.write_text("station_id,time,rain,land_cover\nPRIVATE,2025-01-01,42,4\n"
                    "PRIVATE,2025-01-02,,9\nPRIVATE,2025-01-03,44,3\n", encoding="utf-8")
    fields = [dict(name="rain", unit="mm"), dict(name="station_id", unit="m"), dict(name="land_cover", unit="m")]
    inputs = scan(path, [source(path, "CSV", fields)])
    recommendation = dict(execution_fields=[(0, field["name"]) for field in fields],
                          execution_release=dict(status="pending", blockers=[
                              {"field": "D1.F3", "reason": "insufficient_sample_coverage"}]))
    assert fallback_evidence.eligible(recommendation)
    authorized = fallback_evidence.approved_fields(inputs, recommendation)
    assert authorized == [(0, fields[0])]
    folder = tmp_path / "out"
    folder.mkdir()
    archive = folder / "evidence.zip"
    monkeypatch.setattr(fallback_evidence, "PART_BYTES", 16)
    manifest = fallback_evidence.build(inputs, authorized, archive)
    assert manifest[0]["records"] == 3
    assert len(manifest[0]["parts"]) > 1
    with ZipFile(archive) as data:
        all_rows = b"".join(data.read(part) for part in manifest[0]["parts"])
    assert b"PRIVATE" not in all_rows and b"2025-" not in all_rows
    assert b"2,\r\n" in all_rows
    sent = []

    def transport(messages, tools, key, progress):
        sent.append(json.loads(messages[1]["content"]))
        return {"content": "{}"}, {"id": str(len(sent))}

    with pytest.raises(ValueError, match="已变化"):
        fallback_evidence.transmit(archive, manifest, "not-approved", api_key="test", send=transport)
    assert not sent
    approved = fallback_evidence.digest_file(archive)
    result = fallback_evidence.transmit(archive, manifest, approved, api_key="test", send=transport)
    assert result["received"] == result["total"] == len(sent)
    assert b"".join(item["csv"].encode() for item in sent) == all_rows
    assert not result["receipts"].read_text(encoding="utf-8").count("PRIVATE")


def test_full_fallback_asks_deepseek_to_resolve_split_votes(tmp_path, monkeypatch):
    from ecodata.desktop import independent_review

    path = tmp_path / "rain.csv"
    path.write_text("rain\n42\n43\n", encoding="utf-8")
    field = dict(name="rain", unit="mm")
    inputs = scan(path, [source(path, "CSV", [field])])
    archive = tmp_path / "evidence.zip"
    monkeypatch.setattr(fallback_evidence, "PART_BYTES", 16)
    manifest = fallback_evidence.build(inputs, [(0, field)], archive)
    options = [dict(id=f"R{i}", name=f"候选{i}", steps=["spatial.blend"], description="融合") for i in (1, 2)]
    total = len(manifest[0]["parts"])
    assert total >= 2
    parts = iter(["R1", "R2", *([None] * (total - 2))])

    def transport(messages, *_):
        payload = json.loads(messages[1]["content"])
        if "csv" in payload:
            return {"content": json.dumps(dict(recipe=next(parts), reason="分片理由"))}, {}
        assert payload["votes"] == {"R1": 1, "R2": 1}
        return {"content": json.dumps(dict(recipe="R2", reason="全量汇总理由"))}, {}

    result = fallback_evidence.transmit(archive, manifest, fallback_evidence.digest_file(archive),
                                        api_key="test", recipes=options, send=transport)
    assert result["received"] == total and result["candidate"] == "R2"
    candidate = independent_review.start(dict(local_recipe_options=options), result)
    assert "全量汇总理由" in candidate["reason"] and candidate["execution_release"]["status"] == "pending"


def test_zip_engineering_missing_member_is_treated_as_evidence_gap():
    recommendation = {"steps": ["engineering.port_binding"]}
    checked = {"document_evidence_checks": [{"document": "bundle.zip!/README.md", "status": "pending"}]}
    decision = release_gate.assess(checked, recommendation, {}, [])
    assert decision["status"] == "pending"
    assert not fallback_evidence.eligible({"execution_release": decision})


def test_geotiff_fallback_keeps_layout_and_nodata_mask(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "rain.tif"
    with rasterio.open(path, "w", driver="GTiff", count=1, width=2, height=2, dtype="float32",
                       nodata=-99, crs="EPSG:4326", transform=from_origin(5, 40, 0.01, 0.01)) as ds:
        ds.write(np.array([[10, -99], [20, 30]], dtype="float32"), 1)
    field = dict(name="band1", band=1, unit="mm")
    row = source(path, "GeoTIFF", [field])
    local = release_gate.raster_review(row, field)
    assert local == {"valid_cells": 3, "missing_cells": 1}
    values = list(fallback_evidence.numerical_rows(row, field))
    assert values[1] == [0, 1, ""]
    assert values[0][:2] == [0, 0] and 10 < values[0][2] < 11
    assert fallback_evidence.metadata(row, field)["crs"] == "EPSG:4326"


def test_document_observation_table_extraction_keeps_original_units_and_dates(tmp_path):
    from docx import Document
    from ecodata.desktop.document_evidence import observation_table

    path = tmp_path / "observations.docx"
    document = Document()
    table = document.add_table(rows=2, cols=2)
    for row, values in zip(table.rows, [("time", "rain (mm)"), ("2025-01-01", "2.25")]):
        for cell, value in zip(row.cells, values):
            cell.text = value
    document.save(path)
    assert observation_table(path, 1) == [["time", "rain (mm)"], ["2025-01-01", "2.25"]]


def test_netcdf_supplement_reads_unused_window_outside_holdout(tmp_path):
    path = tmp_path / "series.nc"
    time = np.arange(100)
    xr.Dataset({"rain": ("time", time.astype(float), {"units": "mm"})},
               coords={"time": time}).to_netcdf(path)
    field = dict(name="rain", unit="mm")
    row = source(path, "NetCDF", [field])
    recommendation = sampling_plan.sample(row, field)
    extra, basis = fallback_evidence.uncovered_window(row, field, recommendation)
    assert extra and basis == "uncovered_netcdf_window"
    assert max(extra) < recommendation["partition"]["holdout_start"]
    assert set(extra).isdisjoint(recommendation["values"])
