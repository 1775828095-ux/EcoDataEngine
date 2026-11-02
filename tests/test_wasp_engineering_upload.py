import io
import zipfile
from types import SimpleNamespace
from pathlib import Path

from ecodata.interface.engineering_upload import scan_model_uploads, scan_wasp_uploads


def _upload(name, content):
    return SimpleNamespace(name=name, getvalue=lambda: content)


def test_directory_upload_groups_five_inputs_and_ignores_references():
    uploads = [_upload(f"raw_source_data/{group}/data.csv", group.encode())
               for group in ["usgs", "nbc", "noaa", "ri_dem", "wwtp"]]
    uploads.append(_upload("raw_source_data/references/official.csv", b"reference"))
    scan = scan_wasp_uploads(uploads)
    assert scan.group_counts == {"nbc": 1, "noaa": 1, "ri_dem": 1, "usgs": 1, "wwtp": 1}
    assert len(scan.files) == 5
    assert scan.ignored[0]["file"].endswith("official.csv")


def test_zip_preserves_different_paths_even_when_content_matches_and_rejects_traversal():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("raw/usgs/flow.rdb", b"flow")
        archive.writestr("raw/nbc/same.txt", b"flow")
        archive.writestr("raw/noaa/weather.gz", b"weather")
        archive.writestr("../escape.csv", b"bad")
        archive.writestr("raw/usgs/readme.pdf", b"pdf")
    scan = scan_wasp_uploads([_upload("engineering.zip", buffer.getvalue())])
    assert [name for name, _ in scan.files] == ["raw/usgs/flow.rdb", "raw/nbc/same.txt", "raw/noaa/weather.gz"]
    assert not scan.duplicates
    assert {item["file"] for item in scan.ignored} == {"../escape.csv", "raw/usgs/readme.pdf"}


def test_generic_model_folder_and_zip_use_the_same_ingestion_path():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("meteorology/forcing.nc", b"netcdf")
        archive.writestr("references/note.txt", b"not model input")
    scan = scan_model_uploads([
        _upload("terrain/dem.tif", b"tiff"),
        _upload("model_inputs.zip", buffer.getvalue()),
    ])
    assert [name for name, _ in scan.files] == ["terrain/dem.tif", "meteorology/forcing.nc"]
    assert scan.accepted[0]["group"] == "unclassified"
    assert scan.ignored == [{"file": "references/note.txt", "reason": "参考资料目录不作为平台输入：references"}]


def test_directory_mode_does_not_expand_archives_already_extracted_beside_them():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("station.txt", b"same data")
    scan = scan_model_uploads([
        _upload("ri_dem/archive.zip", buffer.getvalue()),
        _upload("ri_dem/archive/station.txt", b"same data"),
    ], classify_wasp=True, expand_archives=False)
    assert scan.files == [("ri_dem/archive/station.txt", b"same data")]
    assert scan.ignored == [{"file": "ri_dem/archive.zip", "reason": "暂不读取 .zip 文件"}]
    assert not scan.duplicates


def test_only_same_path_and_content_is_a_duplicate_browser_submission():
    scan = scan_model_uploads([
        _upload("usgs/flow.rdb", b"flow"),
        _upload("usgs/flow.rdb", b"flow"),
        _upload("backup/flow.rdb", b"flow"),
    ], classify_wasp=True, expand_archives=False)
    assert [name for name, _ in scan.files] == ["usgs/flow.rdb", "backup/flow.rdb"]
    assert scan.duplicates[0]["reason"] == "同一路径和内容被浏览器重复提交"


def test_directory_uploader_does_not_reject_project_sidecar_files_in_browser():
    source = (Path(__file__).resolve().parents[1] / "src/ecodata/interface/model_page.py").read_text(encoding="utf-8")
    call = source.split('uploads = st.file_uploader("选择模型工程总文件夹"', 1)[1].split(")", 1)[0]
    assert "type=" not in call
    assert 'accept_multiple_files="directory"' in call
