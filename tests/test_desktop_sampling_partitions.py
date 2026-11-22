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
    source["fields"] = [field]
    with xr.open_dataset(path) as original:
        assert original.rain.attrs["units"] == "mm"
    assert document_evidence.cross_check(dict(rows=[source], engineering_docs=[
        ("README.md", "rain.nc 按 UTC 日累计")]))[-1]["status"] == "checked"


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
