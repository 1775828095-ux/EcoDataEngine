import io,json,zipfile
import pytest
from ecodata.interface.camx_project_zip import unpack,SCHEMA


def payload(members=None,inputs=None):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('author/camx_project.json',json.dumps(dict(schema=SCHEMA,control='original.INP',inputs=inputs or {'Initial_Conditions':{'path':'inputs/ic.nc'}})))
        z.writestr('author/original.INP',"&CAMx_Control\n Initial_Conditions = '/old/computer/ic.nc',\n/\n")
        for n,v in (members or {}).items():z.writestr(n,v)
    return b.getvalue()


def test_missing_member_does_not_fall_back_to_old_computer(tmp_path):
    with pytest.raises(ValueError,match='缺少文件.*|工程包缺少文件'):unpack(payload(),tmp_path)
    assert not list(tmp_path.iterdir())


def test_archive_path_escape_rejected_before_extraction(tmp_path):
    with pytest.raises(ValueError):unpack(payload({'author/../../outside':b'bad'}),tmp_path)
    assert not list(tmp_path.iterdir())


def test_input_mapping_must_cover_exact_active_references(tmp_path):
    with pytest.raises(ValueError,match='逐项对应'):unpack(payload(inputs={'Other':{'path':'x'}}),tmp_path)


def test_declared_input_hash_cannot_be_bypassed(tmp_path):
    with pytest.raises(ValueError,match='校验值'):
        unpack(payload({'author/inputs/ic.nc':b'fixture'},inputs={'Initial_Conditions':{'path':'inputs/ic.nc','sha256':'wrong'}}),tmp_path)
