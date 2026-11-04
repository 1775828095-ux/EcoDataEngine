from io import BytesIO
import zipfile
import pytest
from existing_air_converters.wrf_bundle_pipeline import prepare


def bundle(name, payload=b"description"):
    stream=BytesIO()
    with zipfile.ZipFile(stream,"w") as z:
        z.writestr(name,payload)
    stream.name="input.zip"
    return stream


def test_wrf_adapter_no_longer_imports_deleted_constants(tmp_path):
    assert prepare([bundle("readme.txt")],tmp_path) is None


def test_nonarchive_task_can_fall_through_without_import_failure(tmp_path):
    stream=BytesIO(b"x,y\n1,2");stream.name="input.csv"
    assert prepare([stream],tmp_path) is None


def test_path_safety_remains_enabled(tmp_path):
    with pytest.raises(ValueError,match="不安全"):
        prepare([bundle("../outside.txt")],tmp_path)
