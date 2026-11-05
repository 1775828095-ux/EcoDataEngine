import io
import json
import tarfile
import zipfile
from pathlib import Path
import pytest
from ecodata.interface.engineering_upload import LocalUpload
from ecodata.models.archive_inputs import normalize_archives


def zipped(files):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        for n,v in files.items():z.writestr(n,v)
    return b.getvalue()


def upload(tmp_path,raw):
    p=tmp_path/'input.zip';p.write_bytes(raw);return LocalUpload(p.name,p)


def test_flat_zip_preserves_original_contract(tmp_path):
    original=upload(tmp_path,zipped({'file.cio':b'original'}))
    result,audit=normalize_archives([original],tmp_path)
    assert result==[original] and audit is None


def test_nested_project_keeps_bytes_paths_receipts_and_collapses_equal_copy(tmp_path):
    raw=zipped({'data/project.zip':zipped({'engine/file.cio':b'original'}),
                'data/copy.zip':zipped({'engine/file.cio':b'original'}),
                'data/source.nc.receipt.json':b'{"time_standard":"UTC"}'})
    original=upload(tmp_path,raw);result,audit=normalize_archives([original],tmp_path)
    with zipfile.ZipFile(result[0].path) as z:
        assert set(z.namelist())=={'data/engine/file.cio','data/source.nc.receipt.json'}
        assert z.read('data/engine/file.cio')==b'original'
    assert sum(i['duplicate'] for i in audit['resources'])==1
    assert original.getvalue()==raw


def test_conflicting_project_copy_never_picks_a_winner(tmp_path):
    original=upload(tmp_path,zipped({'a.zip':zipped({'file.cio':b'a'}),'b.zip':zipped({'file.cio':b'b'})}))
    with pytest.raises(ValueError,match='冲突'):normalize_archives([original],tmp_path)


@pytest.mark.parametrize('name',['../outside','/absolute','C:/escape'])
def test_nested_path_traversal_rejected(tmp_path,name):
    original=upload(tmp_path,zipped({'inner.zip':zipped({name:b'data'})}))
    with pytest.raises(ValueError,match='不安全'):normalize_archives([original],tmp_path)


def test_tar_internal_links_only_become_verified_bytes(tmp_path):
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        v=b'background';i=tarfile.TarInfo('rc/field');i.size=len(v);t.addfile(i,io.BytesIO(v))
        i=tarfile.TarInfo('rc/alias');i.type=tarfile.SYMTYPE;i.linkname='field';t.addfile(i)
    original=upload(tmp_path,zipped({'case.tar.gz':b.getvalue()}));result,_=normalize_archives([original],tmp_path)
    with zipfile.ZipFile(result[0].path) as z:assert z.read('rc/field')==z.read('rc/alias')==v


def test_posix_timestamp_stays_virtual_without_renaming_raw_member(tmp_path):
    from ecodata.models.project_binding import safe_name
    raw=zipped({'case.zip':zipped({'ob/obs_12:00:00':b'observation'})})
    result,_=normalize_archives([upload(tmp_path,raw)],tmp_path)
    with zipfile.ZipFile(result[0].path) as z:assert z.read('ob/obs_12:00:00')==b'observation'
    with pytest.raises(ValueError):safe_name('ob/obs_12:00:00')


def test_existing_pure_grib_tar_dependency_contract_is_preserved(tmp_path):
    b=io.BytesIO();raw=b'GRIB\0\0\0\2'+b'original forcing'
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        i=tarfile.TarInfo('field.grib');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
    payload=b.getvalue();original=upload(tmp_path,zipped({'forcing.tar.gz':payload}))
    result,_=normalize_archives([original],tmp_path)
    with zipfile.ZipFile(result[0].path) as z:assert z.read('forcing.tar.gz')==payload


def test_project_archive_dependency_is_preserved_not_unpacked_away(tmp_path):
    dependency=zipped({'picture.png':b'image'})
    original=upload(tmp_path,zipped({'project.zip':zipped({'project/file.cio':b'control','project/attachments.zip':dependency})}))
    result,_=normalize_archives([original],tmp_path)
    with zipfile.ZipFile(result[0].path) as z:
        assert z.read('project/attachments.zip')==dependency
        assert 'project/picture.png' not in z.namelist()


def test_7z_streaming_is_used_and_preserves_member_bytes(tmp_path):
    import py7zr
    p=tmp_path/'inner.7z'
    with py7zr.SevenZipFile(p,'w') as z:z.writestr(b'weather','project/file.cio')
    original=upload(tmp_path,zipped({'inner.7z':p.read_bytes()}));result,_=normalize_archives([original],tmp_path)
    with zipfile.ZipFile(result[0].path) as z:assert z.read('project/file.cio')==b'weather'


def test_wrfda_recognition_does_not_claim_assimilation(tmp_path):
    from ecodata.models.wrfda_inputs import prepare_wrfda_inputs
    u=upload(tmp_path,zipped({'namelist.input.3dvar':b'&wrfvar1 var4d=.false. /\n&wrfvar18 analysis_date="2008-02-05_12:00:00" /',
        'be/be.dat':b'error','rc/wrfinput_d01':b'background','ob/obs_gts':b'obs'}))
    result=prepare_wrfda_inputs([u],tmp_path)
    assert result['manifest']['status']=='ENGINEERING_INSPECTED'
    assert not result['manifest']['native_ready']
    d=json.loads((tmp_path/'required_information.json').read_text(encoding='utf-8'))
    assert d[0]['code']=='WRFDA_BACKEND_NOT_CONNECTED'


def test_unrelated_humidity_is_reported_without_hiding_target_precipitation(tmp_path):
    from contextlib import ExitStack
    from ecodata.models.project_binding import ProjectFiles
    from ecodata.models.engineering.ports import weather_ports
    from test_unified_engineering_ports import pack
    files,_,_,_,_=pack('swat_plus','precipitation','mm','accumulation')
    files['project/weather-sta.cli']=files['project/weather-sta.cli'].replace(b'null null null\n',b'null wet.hmd null\n')
    files['project/file.cio']=files['project/file.cio'].replace(b'tmp.cli null null null null',b'tmp.cli null hmd.cli null null')
    files['project/hmd.cli']=b'title\nfilename\nwet.hmd\n'
    files['project/wet.hmd']=b'title\nNBYR TSTEP LAT LON ELEV\n1 0 0 0 1\n2020 1 84\n2020 2 90\n'
    original=upload(tmp_path,zipped(files))
    with ExitStack() as stack:
        project=ProjectFiles([original],stack)
        ports,unknown=weather_ports(project,requested_quantities={'precipitation'})
        assert any(p.quantity=='precipitation' for p in ports)
        assert any(i['code']=='NON_TARGET_WEATHER_INVALID' and '0至1' in i['reason'] for i in unknown)
        with pytest.raises(ValueError,match='0至1'):weather_ports(project,requested_quantities={'relative_humidity'})
