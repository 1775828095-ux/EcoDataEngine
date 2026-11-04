from io import BytesIO
import zipfile
import pytest
from existing_air_converters.wrf_bundle_pipeline import prepare


def bundle(name='readme.txt'):
    b=BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr(name,b'documentation')
    b.name='project.zip'
    return b


def test_wrf_parser_no_longer_imports_deleted_size_constants(tmp_path):
    assert prepare([bundle()],tmp_path) is None


def test_path_safety_survives_removal(tmp_path):
    with pytest.raises(ValueError,match='不安全'):
        prepare([bundle('../escape.txt')],tmp_path)
