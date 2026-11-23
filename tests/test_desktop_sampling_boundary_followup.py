import io
import json
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
import pytest
import xarray as xr

from ecodata.desktop import deepseek_files, fallback_evidence, native_dependencies, precipitation_delivery, release_gate
from ecodata.desktop.scan import inventory


def source(path, kind, fields):
    return dict(name=path.name, format=kind, ref=(path, "", None, None), bytes=path.stat().st_size, fields=fields)


@pytest.mark.parametrize("kind", ["CSV", "Excel"])
def test_mixed_units_are_converted_and_read_back_in_order(tmp_path, kind):
    import pandas as pd
    from ecodata.desktop.pair_processing import conversion_task, run_conversion
    from ecodata.io.structures import read_structure

    path = tmp_path / ("data.xlsx" if kind == "Excel" else "data.csv")
    frame = pd.DataFrame({"rain": [10, 2, 30], "rain_unit": ["mm", "cm", "mm"]})
    field = dict(name="rain", unit="mm")
    if kind == "Excel":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame({"rain": [999]}).to_excel(writer, sheet_name="wrong", index=False)
            frame.to_excel(writer, sheet_name="observations", index=False)
        field["sheet"] = "observations"
    else:
        frame.to_csv(path, index=False)
    row = source(path, kind, [field])
    reviewed = release_gate.table_review(row, field, tmp_path, target_unit="mm")
    assert reviewed["unit_segment_count"] == 3
    scan = {"rows": [row]}
    task = conversion_task(scan, [(0, field)], "convert_units", {"target_unit": "mm"})
    task["recipes"] = [task["recipe_steps"][0]]
    bundle = run_conversion(scan, "C1", task, tmp_path, None, None)
    with read_structure(*bundle["files"][0]) as out:
        assert out["rain"].values.tolist() == [10, 20, 30]
        assert out["rain_effective_unit"].values.tolist() == ["mm"] * 3
        assert out["rain_unit"].values.tolist() == ["mm", "cm", "mm"]


def test_mixed_frequency_exports_exact_intervals_or_complete_utc_days(tmp_path):
    path = tmp_path / "mixed.nc"
    widths = np.array([1] * 12 + [2] * 6)
    edges = np.r_[0, widths.cumsum()]
    xr.Dataset({"rain": ("time", np.ones(len(widths)), {"units": "mm/h", "cell_methods": "time: mean"}),
                "time_bounds": (("time", "edge"), np.column_stack([edges[:-1], edges[1:]]))},
               coords={"time": ("time", edges[:-1], {"units": "hours since 2021-01-01",
                                                      "bounds": "time_bounds"})}).to_netcdf(path)
    field = dict(name="rain", unit="mm/h")
    row = source(path, "NetCDF", [field])
    original = fallback_evidence.digest_file(path)
    for daily in False, True:
        output = tmp_path / f"result-{daily}.nc"
        precipitation_delivery.export(row, field, output, daily=daily)
        with xr.open_dataset(output, decode_times=False) as ds:
            assert ds["rain"].attrs["units"] == "mm"
            assert ds["rain"].values.tolist() == ([24.0] if daily else widths.tolist())
            assert ds[ds.time.attrs["bounds"]].values[-1, -1] == 24
    assert fallback_evidence.digest_file(path) == original


def test_precipitation_rejects_duplicate_time_without_replacing_output(tmp_path):
    path = tmp_path / "duplicate.nc"
    xr.Dataset(
        {"rain": ("time", [1.0, 1.0], {"units": "mm/h", "cell_methods": "time: mean"}),
         "time_bounds": (("time", "edge"), [[0, 1], [1, 2]])},
        coords={"time": ("time", [1, 1], {"units": "hours since 2021-01-01", "bounds": "time_bounds"})},
    ).to_netcdf(path)
    field = dict(name="rain", unit="mm/h")
    output = tmp_path / "existing.nc"
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match="重复"):
        precipitation_delivery.export(source(path, "NetCDF", [field]), field, output)
    assert output.read_bytes() == b"existing"


def test_mixed_unit_holdout_integrates_actual_row_representations(tmp_path):
    from ecodata.desktop.sampling_plan import recommendation_review

    path = tmp_path / "mixed.csv"
    path.write_text("rain,rain_unit\n" + "1,cm\n10,mm\n" * 20, encoding="utf-8")
    field = dict(name="rain", unit="mm")
    row = source(path, "CSV", [field])
    reports = recommendation_review({"rows": [row]}, {"steps": ["units.harmonize", "format.parquet"]},
                                   {"target_unit": "mm"}, [(0, "D1.F1", field)])
    assert reports[0]["status"] == "checked"
    assert reports[0]["observed_unit_segments"] == ["cm", "mm"]


def test_tiff_upload_uses_original_bytes_and_document_purpose():
    data = b"II*\x00synthetic-tiff"

    def opener(request, timeout):
        payload = b"".join(request.data)
        assert request.full_url == "https://api.deepseek.com/files"
        assert int(request.get_header("Content-length")) == len(payload)
        assert payload.index(b"rich_layout_recognition") < payload.index(data)
        assert b"expires_after[seconds]" in payload
        return io.BytesIO(json.dumps(dict(id="file-api-test", bytes=len(data), expires_at=123)).encode())

    receipt = deepseek_files.upload_tiff(io.BytesIO(data), len(data), api_key="synthetic", open_request=opener)
    assert receipt["id"] == "file-api-test"


def test_geo_category_tags_are_detected_before_outbound_selection(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from ecodata.desktop.file_tools import inspect

    path = tmp_path / "classes.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=2, dtype="uint8",
                       transform=from_origin(0, 1, 1, 1), crs="EPSG:4326") as dataset:
        dataset.write(np.ones((2, 2, 2), dtype=np.uint8))
        dataset.set_band_unit(1, "mm")
        dataset.set_band_unit(2, "m")
        dataset.update_tags(2, data_kind="categorical", flag_values="1 2")
    row = inspect(path, "read_header", {"path": ""})["row"]
    selected = fallback_evidence.approved_fields({"rows": [row]},
        {"execution_fields": [(0, "band_1", "D1.F1"), (0, "band_2", "D1.F2")]})
    assert [field["band"] for _, field in selected] == [1]
    assert [field["band"] for _, field in fallback_evidence.classification_fields({"rows": [row]}, selected)] == [2]


def test_tiff_binary_is_only_sent_after_optional_authorization(tmp_path):
    path = tmp_path / "review.zip"
    tiff = b"actual native TIFF bytes"
    entry = dict(id="F1", parts=["F1/part.csv"], native_upload=[{"member": "F1/copy.tif", "bytes": len(tiff)}])
    with ZipFile(path, "w") as file:
        file.writestr("F1/part.csv", "0,0,9\n")
        file.writestr("F1/copy.tif", tiff)
        file.writestr("manifest.json", json.dumps([entry]))
    file_ids = []

    def upload(stream, size, *, api_key):
        assert stream.read() == tiff and size == len(tiff)
        return {"id": "file-api-example"}

    def mock(messages, *args):
        content = messages[1]["content"]
        if isinstance(content, list):
            file_ids.append(content[0]["file_id"])
        return {"content": "{}"}, {"id": "mock"}

    digest = fallback_evidence.digest_file(path)
    fallback_evidence.transmit(path, [entry], digest, api_key="synthetic", send=mock,
                               upload=lambda *args, **kw: pytest.fail("native upload lacks consent"))
    assert not file_ids
    second = tmp_path / "approved.zip"
    second.write_bytes(path.read_bytes())
    fallback_evidence.transmit(second, [entry], digest, api_key="synthetic", send=mock,
                               native_approved=True, upload=upload)
    assert file_ids == ["file-api-example"]


def test_native_zip_parser_finds_implicit_required_wrf_domains(tmp_path):
    archive = tmp_path / "project.zip"
    with ZipFile(archive, "w") as package:
        package.writestr("RUN/namelist.input", "&time_control\n interval_seconds = 3600,\n/\n&domains\n max_dom = 2,\n/\n")
        package.writestr("RUN/wrfinput_d01", b"header stub")
        package.writestr("RUN/wrfbdy_d01", b"header stub")
    checks = native_dependencies.review({"path": archive}, "wrf")
    assert any(item["status"] == "pending" and "wrfinput_d02" in item["reference"] for item in checks)
    assert any(item.get("data", "").endswith("wrfinput_d01") for item in checks)
    assessment = release_gate.assess({"path": archive, "document_evidence_checks": checks},
                                     {"steps": ["engineering.port_binding"]}, {}, [])
    assert assessment["status"] == "pending"
    assert any(member["member"].endswith("wrfinput_d01") for member in assessment["members"])


def test_pdf_original_page_preview_is_rendered_locally(tmp_path):
    from pypdf import PdfWriter
    from ecodata.desktop.document_evidence import render_original_page
    path = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(72, 72)
    writer.write(path)
    assert render_original_page(path, 1).startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="没有"):
        render_original_page(path, 2)


def test_connection_check_sends_only_generated_tiff_and_fixed_csv():
    from rasterio.io import MemoryFile
    from ecodata.desktop import deepseek_self_check

    def upload(stream, size, *, api_key):
        data = stream.read()
        assert len(data) == size and api_key == "test-key"
        with MemoryFile(data) as memory, memory.open() as image:
            assert image.shape == (128, 128)
            assert set(np.unique(image.read(1))) == {0, 255}
        return {"id": "test-attachment", "expires_at": 123}

    def response(messages, tools, key, progress):
        assert messages[1]["content"][0] == {"type": "file", "file_id": "test-attachment"}
        assert messages[1]["content"][1]["text"].endswith("value\n1\n2")
        assert not tools and key == "test-key"
        return {"content": '{"synthetic_sum": 3}'}, {"id": "test-chat"}

    result = deepseek_self_check.check("test-key", upload=upload, send=response)
    assert result["status"] == "passed" and result["chat_id"] == "test-chat"
    with pytest.raises(ValueError, match="校验没有通过"):
        deepseek_self_check.check("test-key", upload=upload,
                                  send=lambda *_: ({"content": '{"synthetic_sum": 4}'}, {}))
    with pytest.raises(ValueError, match="API 密钥"):
        deepseek_self_check.check("", upload=upload, send=response)


def test_runtime_manifest_verifies_members_without_bypassing_native_gate(tmp_path):
    archive = tmp_path / "project.zip"
    data = b"test meteorology"
    with ZipFile(archive, "w") as package:
        package.writestr("run.csh", "# CMAQ run\nsetenv CTM_MET inputs/met.nc\n")
        package.writestr("inputs/met.nc", data)
    checks = native_dependencies.review({"path": archive}, "cmaq")
    assert any(item.get("runtime_manifest_eligible") for item in checks)
    scan = dict(path=archive, signature=tuple(inventory(archive)), document_evidence_checks=checks)
    binding = dict(path="inputs/met.nc", sha256=sha256(data).hexdigest())
    manifest = dict(schema="ecodata.native-runtime-bindings.v1", model="cmaq",
                    archive_sha256=fallback_evidence.digest_file(archive),
                    controls=[dict(path="run.csh", files=[binding])])
    updates = native_dependencies.verify_runtime_manifest(scan, "cmaq", manifest)
    assert [item["check"] for item in updates] == ["verified_runtime_member", "verified_runtime_bindings"]
    assert updates[0]["verified_sha256"] == binding["sha256"]
    scan["document_evidence_checks"] = [*checks, *updates]
    assessment = release_gate.assess(scan, {"steps": ["engineering.port_binding"]}, {}, [])
    assert assessment["status"] == "pending"
    with pytest.raises(ValueError, match="模型或 ZIP"):
        native_dependencies.verify_runtime_manifest(scan, "wrf", manifest)
    binding["sha256"] = sha256(b"changed").hexdigest()
    with pytest.raises(ValueError, match="内容与清单不匹配"):
        native_dependencies.verify_runtime_manifest(scan, "cmaq", manifest)
    binding["path"] = "../outside"
    with pytest.raises(ValueError, match="路径不明确"):
        native_dependencies.verify_runtime_manifest(scan, "cmaq", manifest)
