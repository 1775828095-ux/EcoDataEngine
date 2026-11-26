"""Synthetic archive boundary checks; no environmental accuracy claims."""
import io,tarfile,zipfile
from contextlib import ExitStack
from pathlib import Path
import pytest
from ecodata.interface.engineering_upload import LocalUpload
from ecodata.models.archive_inputs import normalized_archive_scope
from ecodata.models.project_binding import ProjectFiles


def package(tmp_path,bad=False):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as tar:
        for name,data in [('control.txt',b'known'),('../escape' if bad else 'unneeded.bin',b'x'*2000000)]:
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    path=tmp_path/'original.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('inputs.tgz',raw.getvalue());z.writestr('unneeded.csv',b'y'*2000000)
    return [LocalUpload(path.name,path)]


@pytest.mark.parametrize('fail',[False,True])
def test_headers_only_then_selected_member_and_cleanup(tmp_path,fail):
    uploads=package(tmp_path);before=Path(uploads[0].path).read_bytes()
    audit=None
    try:
        with normalized_archive_scope(uploads,tmp_path/'work') as (active,audit,cleanup):
            assert active==uploads
            with ExitStack() as stack:
                files=ProjectFiles(active,stack)
                assert files.prefix('control.txt',5)==b'known'
                assert files.prefix('unneeded.csv',8)==b'y'*8
                assert audit['staged_bytes']==0
                with files.files['control.txt'][0]() as stream:assert stream.read()==b'known'
                assert audit['staged_bytes']==5
                assert [x['path'] for x in audit['selected_members']]==['control.txt']
                assert not (tmp_path/'work/expanded_inputs.zip').exists()
                if fail:raise RuntimeError('consumer failure')
    except RuntimeError:
        assert fail
    assert not Path(audit['temporary_directory']).exists()
    assert Path(uploads[0].path).read_bytes()==before


def test_unsafe_nested_name_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='不安全'):
        with normalized_archive_scope(package(tmp_path,True),tmp_path/'work'):pass


def test_native_readback_uses_bounded_blocks(tmp_path,monkeypatch):
    import numpy as np
    from netCDF4 import Dataset
    from existing_air_converters import native_netcdf_update as writer
    from ecodata.models.native_meteorology import write_native_copy
    original=tmp_path/'native.nc';target=tmp_path/'updated.nc'
    with Dataset(original,'w') as ds:
        for name,size in [('TSTEP',2),('LAY',1),('ROW',1100),('COL',1100)]:ds.createDimension(name,size)
        for name in ['t2','untouched']:
            v=ds.createVariable(name,'f4',('TSTEP','LAY','ROW','COL'));v[:]=280.
    blocks=[];chunks=writer._chunks
    def tracked(shape,itemsize,*args,**kwargs):
        for block in chunks(shape,itemsize,*args,**kwargs):
            size=int(np.prod([s.stop-s.start for s in block]))*itemsize
            blocks.append(size);assert size<=4*1024**2
            yield block
    monkeypatch.setattr(writer,'_chunks',tracked)
    result=write_native_copy(original,target,'t2',np.array([[290.]]),[1],[1209999])
    assert result['changed_values']==1 and len(blocks)>2
    with Dataset(original) as a,Dataset(target) as b:
        assert a['t2'][1,0,1099,1099]==280 and b['t2'][1,0,1099,1099]==290
    original.unlink();target.unlink()


def test_written_sequential_controls_rebind_without_fake_restart(tmp_path):
    from test_camx_template_schedule import analyze,SCRIPT
    from ecodata.models.project_binding import bind_project
    parsed=analyze(SCRIPT)
    path=tmp_path/'derivative.zip'
    with zipfile.ZipFile(path,'w') as z:
        for i,run in enumerate(parsed['runs']):z.writestr(f'project/runfiles/run_{i}.in',run['rendered_control'])
        for name in ['chem','photo','ozone','ic','boundary.20160610','boundary.20160611']:
            z.writestr('project/met/'+name,b'original')
    binding=bind_project([LocalUpload(path.name,path)],'camx')
    assert binding['execution_order']==[0,1]
    assert binding['runtime_dependencies']=={'0':[],'1':[0]}
    assert sum(b['status']=='GENERATED_DEPENDENCY' for b in binding['bindings'])==1
    assert not binding['native_model_executed']
