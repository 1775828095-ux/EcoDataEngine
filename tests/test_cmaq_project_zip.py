import io,json,zipfile,hashlib
import pytest
from ecodata.interface.cmaq_project_zip import unpack,SCHEMA

def package(extra=None,**changes):
    m={'schema':SCHEMA,'environment':{'MET_CRO_3D':'inputs/met.nc'},'inputs':{'MET_CRO_3D':{'path':'inputs/met.nc'}}};m.update(changes)
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('project/cmaq_project.json',json.dumps(m))
        z.writestr('project/inputs/met.nc',b'original test bytes')
        for name,data in (extra or {}).items():z.writestr(name,data)
    return b.getvalue()

def test_nested_package_resolves_without_old_machine_paths(tmp_path):
    b=package();r=unpack(b,tmp_path)
    from pathlib import Path
    p=Path(r['inputs']['MET_CRO_3D']['path'])
    assert p.read_bytes()==b'original test bytes'
    assert r['inputs']['MET_CRO_3D']['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert r['environment']['MET_CRO_3D'].startswith('/mnt/')
    assert unpack(b,tmp_path)==r

def test_lists_missing_file_without_host_fallback(tmp_path):
    with pytest.raises(ValueError,match='缺少文件.*|工程包缺少文件'):
        unpack(package(environment={'A':'missing.nc'},inputs={'A':{'path':'missing.nc'}}),tmp_path)
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize('name',['../outside','/outside','C:/outside','project/../outside','project/con.txt'])
def test_unsafe_member_never_extracted(tmp_path,name):
    with pytest.raises(ValueError):unpack(package({name:b'bad'}),tmp_path)
    assert not list(tmp_path.iterdir())

def test_declared_hash_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError,match='校验值'):
        unpack(package(inputs={'MET_CRO_3D':{'path':'inputs/met.nc','sha256':'bad'}}),tmp_path)

def test_ambiguous_configuration_rejected(tmp_path):
    with pytest.raises(ValueError,match='只能包含一份'):
        unpack(package({'second/cmaq_project.json':b'{}'}),tmp_path)

def test_modified_cached_file_rejected(tmp_path):
    b=package();r=unpack(b,tmp_path)
    from pathlib import Path
    Path(r['inputs']['MET_CRO_3D']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='缓存文件被修改'):unpack(b,tmp_path)

def test_declared_absent_reference_never_uses_author_machine(tmp_path):
    r=unpack(package(environment={'MET_CRO_3D':'inputs/met.nc','INIT_SENS_1':'/old/absent.nc'},unavailable_inputs={'INIT_SENS_1':'/old/absent.nc'}),tmp_path)
    assert r['environment']['INIT_SENS_1']==r['unavailable_inputs']['INIT_SENS_1']
    assert '/old/absent.nc' not in r['environment']['INIT_SENS_1']
    assert '/unavailable/INIT_SENS_1' in r['environment']['INIT_SENS_1']
