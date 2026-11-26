import hashlib
from zipfile import ZipFile

import numpy as np
import pytest
import xarray as xr

from ecodata.desktop import document_evidence, file_tools, sampling_plan


def row(path, kind):
    return dict(name=path.name, format=kind, ref=(path, "", None, None), bytes=path.stat().st_size)


def test_netcdf_reserves_later_time_and_separate_space(tmp_path):
    path = tmp_path / "rain.nc"
    time = np.arange(100)
    values = (time[:, None] + 1000 * np.arange(10)).astype(float)
    ds = xr.Dataset(
        {"rain": (("time", "y"), values, {"units": "mm"}),
         "bounds": (("time", "edge"), np.column_stack([time, time + 1]))},
        coords={"time": ("time", time, {"units": "days since 2000-01-01", "bounds": "bounds"})},
    )
    ds.to_netcdf(path)
    source, field = row(path, "NetCDF"), {"name": "rain"}
    sent = sampling_plan.sample(source, field)
    later = sampling_plan.sample(source, field, holdout=True)
    area = sampling_plan.sample(source, field, holdout=True, spatial=True)
    assert max(v % 1000 for v in sent["values"]) < 80 <= min(v % 1000 for v in later["values"])
    assert max(v // 1000 for v in sent["values"]) < 8 <= min(v // 1000 for v in area["values"])
    assert later["windows"][0]["time"]["intervals_valid"]
    assert set(sent["values"]).isdisjoint(later["values"])
    assert set(sent["values"]).isdisjoint(area["values"])
    payload = sampling_plan.export_field(source, field)
    widths = payload["sampling"]["windows"][0]["time"]["interval_widths"]
    assert widths and set(widths) == {1}
    assert "first" not in payload["sampling"]["windows"][0]["time"]
    private = dict(time_coverage=[1, 100], resolution=dict(time_seconds=86400), nested=[{
        "sampling_interval_seconds": 86400, "unit": "mm"}])
    assert sampling_plan.recommendation_metadata(private) == {"resolution": {}, "nested": [{"unit": "mm"}]}
    assert private["time_coverage"] == [1, 100]
    source["fields"] = [field]
    with xr.open_dataset(path) as original:
        assert original.rain.attrs["units"] == "mm"
    assert document_evidence.cross_check(dict(rows=[source], engineering_docs=[
        ("README.md", "rain.nc 按 UTC 日累计")]))[-1]["status"] == "checked"


def test_netcdf_sampling_does_not_build_eager_coordinate_indexes(tmp_path):
    from inspect import signature

    if "create_default_indexes" not in signature(xr.open_dataset).parameters:
        pytest.skip("xarray does not support lazy dimension indexes")
    path = tmp_path / "clock.nc"
    xr.Dataset(coords={"time": np.arange(100)}).to_netcdf(path)
    with sampling_plan.open_netcdf_for_sampling(path) as dataset:
        assert not dataset.indexes
        assert dataset.time.isel(time=slice(98, 100)).values.tolist() == [98, 99]


def test_raster_recommendation_and_local_blocks_do_not_touch(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path / "grid.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(5, 40, 0.01, 0.01), nodata=-99) as dataset:
        dataset.write(np.arange(400, dtype="float32").reshape(20, 20), 1)
    source, field = row(path, "GeoTIFF"), dict(name="band1", band=1)
    sent = sampling_plan.sample(source, field)
    held = sampling_plan.sample(source, field, holdout=True)
    assert set(sent["values"]).isdisjoint(held["values"])
    trial = sampling_plan.raster_trial(source, field, held, (0.02, 0.02), "nearest")
    assert trial["valid_cells"] == 4


def test_csv_holdout_uses_independent_station_and_real_interpolation(tmp_path):
    path = tmp_path / "stations.csv"
    held = next(str(i) for i in range(100) if hashlib.sha256(str(i).encode()).digest()[0] % 4 == 0)
    sent = next(str(i) for i in range(100) if hashlib.sha256(str(i).encode()).digest()[0] % 4 != 0)
    lines = ["station_id,time,rain"]
    for station, shift in [(sent, 0), (held, 100)]:
        lines.extend(f"{station},2025-01-01T0{i}:00:00,{i+shift}" for i in range(7))
    path.write_text("\n".join(lines), encoding="utf-8")
    source, field = row(path, "CSV"), dict(name="rain", unit="mm")
    assert max(sampling_plan.sample(source, field)["values"]) < 100
    assert min(sampling_plan.sample(source, field, holdout=True)["values"]) == 100
    result = sampling_plan.recommendation_review(
        dict(rows=[source]), dict(steps=["temporal.wide_table_interpolation"]),
        {}, [(0, "D1.F3", field)])[0]
    assert result["status"] == "checked"
    assert result["metrics"]["candidate"]["rmse"] == 0
    assert result["metrics"]["baseline"]["mae"] == 1


def test_unit_and_format_conversion_runs_on_local_heldout_rows(tmp_path):
    path = tmp_path / "rain.csv"
    path.write_text("rain\n" + "\n".join(str(i) for i in range(32)), encoding="utf-8")
    source, field = row(path, "CSV"), dict(name="rain", unit="mm")
    report = sampling_plan.recommendation_review(dict(rows=[source]),
        dict(steps=["units.harmonize", "format.parquet"]), {"target_unit": "cm"}, [(0, "D1.F1", field)])[0]
    assert report["status"] == "checked"
    assert report["count"] == 8
    assert "format_roundtrip_executed_on_holdout" in report["checks"]
    assert "metrics" not in report


def test_excel_locked_sheet_and_header_conflicts(tmp_path):
    from openpyxl import Workbook
    path = tmp_path / "rain.xlsx"
    book = Workbook()
    book.active.title = "sample"
    book.active.append(["rain"])
    book.active.append([2])
    review = book.create_sheet("review")
    review.append(["rain"])
    review.append([20])
    book.create_sheet("_ecodata_metadata").sheet_state = "hidden"
    book.save(path)
    source, field = row(path, "Excel"), dict(name="rain", sheet="sample", unit="mm")
    assert sampling_plan.sample(source, field)["values"] == [2]
    assert sampling_plan.sample(source, field, holdout=True)["values"] == [20]
    source["fields"] = [dict(name="rain", sheet="review", unit="cm")]
    assert sampling_plan.sample(source, field, holdout=True)["basis"] == "heldout_sheet_unit_conflict"


def test_read_markdown_docx_and_pdf_with_locations(tmp_path, monkeypatch):
    from docx import Document
    from pypdf import PdfWriter

    markdown = tmp_path / "README.md"
    markdown.write_text("# Scheme\n\nUTC daily precipitation is stored in rain.nc.\n", encoding="utf-8")
    md = file_tools.inspect(markdown, "read_text", {"path": ""})
    assert md["evidence"][0]["location"].startswith("Scheme")
    assert md["evidence"][0]["references"] == ["rain.nc"]

    docx = tmp_path / "README.docx"
    document = Document()
    document.add_heading("Forcing", 1)
    document.add_paragraph("Units for rain.nc: mm")
    document.save(docx)
    result = file_tools.inspect(docx, "read_text", {"path": ""})
    assert result["evidence"][0]["location"].startswith("Forcing")

    pdf = tmp_path / "README.pdf"
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.write(pdf)
    monkeypatch.setattr(document_evidence, "ocr", lambda *_: "")
    unknown = file_tools.inspect(pdf, "read_text", {"path": ""})
    assert unknown["pending"] and not unknown["evidence"]
    monkeypatch.setattr(document_evidence, "ocr", lambda *_: "Units for rain.nc: mm")
    known = file_tools.inspect(pdf, "read_text", {"path": ""})
    assert known["evidence"][0]["location"].startswith("第 1 页")


def test_pdf_walks_nested_outline_then_remaining_pages_without_repeats(tmp_path, monkeypatch):
    from pypdf import PdfWriter

    path = tmp_path / "outline.pdf"
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(200, 200)
    chapter = writer.add_outline_item("Appendix", 0)
    writer.add_outline_item("Variable units", 2, parent=chapter)
    writer.write(path)
    monkeypatch.setattr(document_evidence, "ocr", lambda *_: "Units for rain.nc: mm")

    first = document_evidence.extract(path, ".pdf", 0)
    second = document_evidence.extract(path, ".pdf", first["next_offset"])
    pages = [entry["location"].split(" · ")[0] for entry in first["evidence"] + second["evidence"]]
    assert pages == ["第 3 页", "第 1 页", "第 2 页", "第 4 页"]


def test_zip_csv_header_does_not_materialize_member(tmp_path, monkeypatch):
    path = tmp_path / "mixed.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("tables/rain.csv", "rain\n42\n")
    monkeypatch.setattr(file_tools, "materialize_member", lambda *_: pytest.fail("member should stream"))
    result = file_tools.inspect(path, "read_header", {"path": "tables/rain.csv"})
    assert result["row"]["name"] == "mixed.zip!/tables/rain.csv"
    assert result["header_lines"] == ["rain"]
    assert result["row"]["bytes"] == len(b"rain\n42\n")
    assert sampling_plan.sample(result["row"], {"name": "rain"})["values"] == [42]


def test_time_frequency_change_is_sampled_before_holdout(tmp_path):
    path = tmp_path / "mixed.nc"
    time = np.r_[np.arange(24), 24 + 24 * np.arange(60)]
    xr.Dataset({"rain": ("time", time.astype(float))}, coords={
        "time": ("time", time, {"units": "hours since 2000-01-01"})}).to_netcdf(path)
    sampled = sampling_plan.sample(row(path, "NetCDF"), {"name": "rain"})
    assert sampled["frequency_scope"] == "partition_time_coordinates"
    assert {window["interval"] for window in sampled["windows"]} == {1, 24}
    assert all(window["stop_index"] <= sampled["partition"]["holdout_start"] for window in sampled["windows"])


def test_indexed_csv_reserves_global_latest_month_and_seeks_past_prefix(tmp_path):
    from ecodata.desktop.csv_index import build

    path = tmp_path / "rain.csv"
    content = 'time,rain,notes\n' + '2025-01-01,1,"quoted\nrecord"\n' * 4200
    path.write_text(content + "2025-02-01,2,ok\n2025-03-01,3,ok\n", encoding="utf-8")
    source, field = row(path, "CSV"), {"name": "rain"}
    old = sampling_plan.sample(source, field)
    assert old["partition"]["coverage"] == "bounded_prefix"
    assert set(old["values"]) == {1}
    unindexed = sampling_plan.sample(source, field, holdout=True, future=True)
    assert unindexed["partition"]["coverage"] == "streamed_file"
    assert unindexed["values"] == [3]
    build(path)
    sent = sampling_plan.sample(source, field)
    held = sampling_plan.sample(source, field, holdout=True)
    assert sent["partition"] == {"held_month": "2025-03", "coverage": "indexed_file"}
    assert set(sent["values"]) == {1, 2}
    assert held["values"] == [3]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("2025-04-01,4,ok\n")
    assert sampling_plan.sample(row(path, "CSV"), field)["partition"]["coverage"] == "bounded_prefix"


def test_future_holdout_ignores_station_hash_and_keeps_stations_separate(tmp_path):
    path = tmp_path / "stations.csv"
    path.write_text("station_id,time,rain\n" + "\n".join(
        f"{station},2025-{month:02d}-{day:02d},{100*month+day}"
        for station in ("1", "2", "3") for month in (1, 2) for day in range(1, 5)),
        encoding="utf-8")
    source, field = row(path, "CSV"), {"name": "rain"}
    sent = sampling_plan.sample(source, field)
    future = sampling_plan.sample(source, field, future=True, holdout=True)
    assert all(value < 200 for value in sent["values"])
    assert all(value > 200 for value in future["values"])
    assert len(future["observation_groups"]) == 3


def test_explicit_reference_raster_scores_both_methods_and_is_not_sent(tmp_path):
    import json
    import rasterio
    from rasterio.transform import from_origin
    from ecodata.desktop.file_agent import FileTools
    from ecodata.desktop.raster_reference import MANIFEST, references

    path = tmp_path / "rain.tif"
    truth = tmp_path / "ground.tif"
    for raster, bump in [(path, 0), (truth, 1)]:
        with rasterio.open(raster, "w", driver="GTiff", width=20, height=20, count=1, dtype="float32",
                           crs="EPSG:4326", transform=from_origin(5, 40, 0.01, 0.01), nodata=-99) as output:
            output.write(np.arange(400, dtype="float32").reshape(20, 20) + bump, 1)
            output.set_band_unit(1, "mm")
    (tmp_path / MANIFEST).write_text(json.dumps({"rain.tif": {"reference": "ground.tif"}}), encoding="utf-8")
    tools = FileTools(tmp_path)
    entries = tools.call("list_directory", {"path": ""})["result"]["entries"]
    assert [entry["name"] for entry in entries] == ["rain.tif"]
    with pytest.raises(ValueError, match="本地"):
        tools.call("read_header", {"path": "ground.tif"})
    source = row(path, "GeoTIFF")
    report = sampling_plan.recommendation_review(dict(rows=[source], local_raster_references=references(tmp_path)),
        {"steps": ["raster_task.spatial.bilinear"]}, {}, [(0, "D1.F1", {"name": "band1", "band": 1})])[0]
    assert report["status"] == "checked"
    assert report["metrics"]["n"] == 16
    assert report["metrics"]["candidate"]["rmse"] == report["metrics"]["baseline"]["rmse"] == 1


def test_real_image_only_pdf_ocr(tmp_path):
    import os
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    from pypdf import PdfReader

    font = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts/msyh.ttc"
    if os.name != "nt" or not font.is_file():
        pytest.skip("Windows OCR fixture requires the system Chinese font")
    path = tmp_path / "scan.pdf"
    image = Image.new("RGB", (1300, 360), "white")
    ImageDraw.Draw(image).text((50, 120), "单位 UTC rain.nc", fill="black",
                               font=ImageFont.truetype(str(font), 72))
    image.save(path)
    assert not PdfReader(path).pages[0].extract_text()
    result = document_evidence.extract(path, ".pdf", 0)
    assert result["evidence"] and "rain.nc" in result["evidence"][0]["references"]


def test_zip_relative_reference_checks_unread_member(tmp_path):
    path = tmp_path / "inside.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("docs/README.md", "UTC references ../tables/rain.csv")
        archive.writestr("tables/rain.csv", "rain\n1\n")
    checks = document_evidence.cross_check(dict(path=path, rows=[], engineering_docs=[
        ("inside.zip!/docs/README.md", "UTC references ../tables/rain.csv")]))
    assert checks == [{"document": "inside.zip!/docs/README.md", "reference": "../tables/rain.csv",
                       "check": "member_reference", "status": "checked"}]


def test_netcdf_budget_covers_distinct_cadences_and_files(tmp_path):
    from ecodata.desktop.sampling_strata import choose, strata

    scan = {"rows": [{"format": "NetCDF", "declared_time_resolution": cadence}
                     for cadence in ("PT1H", "PT1H", "P1D")]}
    original = {fid: (index, fid, {"name": "rain", "unit": "mm"}) for fid, index in
                [("D1.F1", 0), ("D1.F2", 0), ("D2.F1", 1), ("D3.F1", 2)]}
    info = strata(scan, original)
    assert choose(list(original), info, 3) == ["D1.F1", "D3.F1", "D2.F1"]
    assert choose(["D1.F1"], info, 3) == ["D1.F1"]


def test_complete_netcdf_coordinate_review_catches_between_window_change(tmp_path, monkeypatch):
    path = tmp_path / "long-clock.nc"
    samples = [0, 1, 2, 3, 5, 7, 9, 11, 11, 13, 15]
    xr.Dataset({"rain": ("time", np.ones(len(samples)))}, coords={
        "time": ("time", samples, {"units": "hours since 2020-01-01"})}).to_netcdf(path)
    monkeypatch.setattr(sampling_plan, "TIME_COORD_LIMIT", 4)
    source, field = row(path, "NetCDF"), {"name": "rain"}
    result = sampling_plan.recommendation_review(dict(rows=[source]), {"steps": []}, {},
        [(0, "D1.F1", field)])[0]
    assert result["status"] == "failed"
    clock = result["time_coordinate_audit"]
    assert clock["scope"] == "all_time_coordinates"
    assert clock["coordinate_count"] == len(samples)
    assert clock["change_examples"] == [4]
    assert clock["nonincreasing"] == 1


def test_csv_station_reservoir_uses_rows_beyond_recommendation_prefix(tmp_path):
    from ecodata.desktop.csv_index import build
    from ecodata.desktop.sampling_plan import station_reserved

    path = tmp_path / "stations.csv"
    held = [str(value) for value in range(100) if station_reserved(str(value))][:2]
    sent = next(str(value) for value in range(100) if not station_reserved(str(value)))
    with path.open("w", encoding="utf-8") as stream:
        stream.write("station_id,rain\n")
        for index in range(5000):
            for station, offset in [(sent, -10000), (held[0], 0), (held[1], 10000)]:
                stream.write(f"{station},{index + offset}\n")
    source, field = row(path, "CSV"), {"name": "rain"}
    assert max(sampling_plan.sample(source, field)["values"]) < 0
    local = sampling_plan.sample(source, field, holdout=True)
    assert local["partition"]["selection"] == "stratified_uniform_reservoir"
    assert local["partition"]["eligible_records"] == 10000
    assert sum(value < 10000 for value in local["values"]) == 16
    assert sum(value >= 10000 for value in local["values"]) == 16
    assert max(value % 10000 for value in local["values"]) > 1366
    build(path)
    indexed = sampling_plan.sample(source, field, holdout=True)
    assert indexed["partition"]["coverage"] == "indexed_file"
    assert indexed["partition"]["eligible_records"] == 10000
    assert set(indexed["values"]) == set(local["values"])


def test_conservative_geotiff_uses_declared_cell_means_and_actual_reference(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from ecodata.desktop.raster_reference import compare

    source_path, truth_path = tmp_path / "fine.tif", tmp_path / "truth.tif"
    fine = np.broadcast_to(np.arange(20, dtype="float32"), (20, 20))
    coarse = fine.reshape(10, 2, 10, 2).mean(axis=(1, 3))
    for path, values, spacing in [(source_path, fine, 0.01), (truth_path, coarse, 0.02)]:
        height, width = values.shape
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(5, 40, spacing, spacing)) as writer:
            writer.write(values, 1)
            writer.set_band_unit(1, "mm")
    source = row(source_path, "GeoTIFF")
    field = {"name": "band1", "band": 1}
    local = sampling_plan.sample(source, field, holdout=True)
    reference = {"path": truth_path, "band": 1, "baseline": "nearest"}
    assert compare(source, field, local, reference, "conservative") is None
    reference["spatial_semantics"] = "cell_mean"
    scored = compare(source, field, local, reference, "conservative")
    assert scored["candidate_method"] == "conservative"
    assert scored["n"] == 4
    assert scored["candidate"]["rmse"] < 0.001
    assert scored["baseline"]["rmse"] > 0.1


def test_zip_follows_transitive_dependencies_without_reading_numeric_member(tmp_path):
    path = tmp_path / "inside.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("docs/README.md", "See details.txt")
        archive.writestr("docs/details.txt", "Read ../data/forcing.csv")
        archive.writestr("data/forcing.csv", "rain\n999\n")
    checks = document_evidence.cross_check(dict(path=path, rows=[], engineering_docs=[
        ("inside.zip!/docs/README.md", "See details.txt")]))
    assert [(result["reference"], result["status"]) for result in checks] == [
        ("details.txt", "checked"), ("../data/forcing.csv", "checked")]
