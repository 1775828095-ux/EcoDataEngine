"""Generated tiny archives: cleanup safety and real shared entry, not skill evidence."""
import io,json,hashlib,tarfile,zipfile
from pathlib import Path
import pytest
from ecodata.interface.engineering_upload import LocalUpload
from ecodata.models.archive_inputs import normalized_archive_scope


def nested(tmp_path):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode="w:gz") as t:
        info=tarfile.TarInfo("field.txt");info.size=4;t.addfile(info,io.BytesIO(b"data"))
    path=tmp_path/"original.zip"
    with zipfile.ZipFile(path,"w") as z:z.writestr("inner.tgz",raw.getvalue())
    return path


@pytest.mark.parametrize("exit_kind",["success","failure","cancel"])
def test_owned_scratch_released_on_every_cooperative_exit(tmp_path,exit_kind):
    original=nested(tmp_path);digest=hashlib.sha256(original.read_bytes()).hexdigest();folder=tmp_path/"run"
    class Cancelled(BaseException):pass
    try:
        with normalized_archive_scope([LocalUpload(original.name,original)],folder) as (active,audit,cleanup):
            assert Path(active[0].path).exists() and audit['indexed_members']==1
            assert audit['staged_bytes']==0
            (folder/"final-product.nc").write_bytes(b"keep product")
            (folder/"search.log").write_text("keep log")
            if exit_kind=="failure":raise ValueError("real operation failed")
            if exit_kind=="cancel":raise Cancelled()
    except (ValueError,Cancelled):
        if exit_kind=="success":raise
    assert not (folder/"archive_intake").exists()
    assert not (folder/"expanded_inputs.zip").exists()
    assert cleanup['status']=='COMPLETED' and cleanup['released_bytes']>0
    assert (folder/"final-product.nc").read_bytes()==b"keep product"
    assert (folder/"search.log").exists()
    assert hashlib.sha256(original.read_bytes()).hexdigest()==digest


def test_preexisting_or_input_named_like_scratch_is_not_removed(tmp_path):
    folder=tmp_path/"run";folder.mkdir()
    original=folder/"expanded_inputs.zip"
    with zipfile.ZipFile(original,"w") as z:z.writestr("field.txt",b"data")
    (folder/"archive_intake").mkdir();(folder/"archive_intake/user.txt").write_bytes(b"user")
    with normalized_archive_scope([LocalUpload(original.name,original)],folder) as (_,__,cleanup):pass
    assert original.exists() and (folder/"archive_intake/user.txt").exists()
    assert cleanup['released_bytes']==0


def test_product_that_reuses_normalized_archive_is_protected(tmp_path):
    original=nested(tmp_path);folder=tmp_path/"run";keep=[]
    with normalized_archive_scope([LocalUpload(original.name,original)],folder,keep=keep) as (active,_,cleanup):
        keep.append(active[0].path)
    assert Path(keep[0]).exists() and not (folder/"archive_intake").exists()
    assert cleanup['selective_resources']['temporary_directory_absent']


def test_normalization_error_still_cleans_partial_intake(tmp_path):
    source=tmp_path/"broken.zip"
    with zipfile.ZipFile(source,"w") as z:z.writestr("broken.tgz",b"not a tar archive")
    with pytest.raises(Exception):
        with normalized_archive_scope([LocalUpload(source.name,source)],tmp_path/"run"):pass
    assert not (tmp_path/"run/archive_intake").exists()
    assert source.exists()


def test_public_preparation_exports_product_then_releases_intake(tmp_path):
    from ecodata.algorithms.automatic_preparation import prepare_uploads
    root=Path(__file__).parents[1]/'examples/ai_round1/camx/emissions'
    request=dict(schema='ecodata.model-task.v1',model_id='camx',operation='emissions',input='data.csv',
                 configuration=json.loads((root/'config_LATLON.json').read_text()))
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as t:
        for name,data in [('request.json',json.dumps(request).encode()),('data.csv',(root/'A_synthetic_LATLON_hourly.csv').read_bytes())]:
            info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
    source=tmp_path/'upload.zip'
    with zipfile.ZipFile(source,'w') as z:z.writestr('package.tgz',raw.getvalue())
    folder=tmp_path/'result';result=prepare_uploads([LocalUpload(source.name,source)],'camx',folder)
    assert result['manifest']['status']=='COMPONENT_EXPORTED'
    assert result['temporary_cleanup']['released_bytes']>0
    assert not (folder/'archive_intake').exists() and not (folder/'expanded_inputs.zip').exists()
    with zipfile.ZipFile(result['archive']) as z:
        assert z.testzip() is None
        assert json.loads(z.read('_ecodata/temporary_cleanup.json'))['status']=='COMPLETED'
        assert any(n.endswith('.nc') for n in z.namelist())


def test_cleanup_error_is_recorded_without_replacing_processing_failure(tmp_path,monkeypatch):
    import shutil
    original=nested(tmp_path);folder=tmp_path/'run'
    with pytest.raises(ValueError,match='original processing failure'):
        with normalized_archive_scope([LocalUpload(original.name,original)],folder) as (_,__,cleanup):
            monkeypatch.setattr(shutil,'rmtree',lambda *a,**k:(_ for _ in ()).throw(PermissionError('locked temporary file')))
            raise ValueError('original processing failure')
    assert cleanup['status']=='PARTIAL' and cleanup['failures']
    assert original.exists()
