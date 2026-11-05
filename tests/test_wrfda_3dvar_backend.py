"""Synthetic binding/quality guards; real solver evidence is recorded separately."""
from pathlib import Path
from contextlib import ExitStack
import io,json,zipfile,hashlib
import numpy as np
import pytest
import xarray as xr
from ecodata.models.project_binding import ProjectFiles
from ecodata.models.wrfda_3dvar import plan,validate
from ecodata.interface.engineering_upload import LocalUpload


def bundle(tmp_path,radiance=False,confirmed=False):
    ds=xr.Dataset({n:(('Time','y','x'),np.ones((1,2,2))) for n in ['T','U','V','QVAPOR','P','PB','PH','PHB','XLAT','XLONG']})
    ds['Times']=('Time',np.array([b'2020-01-01_12:00:00'],dtype='S19'))
    ds.attrs['TITLE']='OUTPUT FROM REAL_EM V3.9'
    control='&wrfvar1 var4d=.false. /\n&wrfvar18 analysis_date="2020-01-01_12:00:00" /\n&wrfvar21 time_window_min="2020-01-01_11:00:00" /\n&wrfvar22 time_window_max="2020-01-01_13:00:00" /\n'
    if radiance:control+='&wrfvar4 use_amsuaobs=.true., use_synopobs=.true. /\n&wrfvar14 rtminit_nsensor=1, use_varbc=.true. /\n'
    obs=b'TOTAL = 1\nINFO_FMT = x\nEACH_FMT = x\nFM-12 SYNOP  2020-01-01_12:00:00 station\n'
    p=tmp_path/'inputs.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('namelist.input',control);z.writestr('rc/wrfinput_d01',ds.to_netcdf(engine='scipy'))
        z.writestr('be/be.dat',b'original error bytes');z.writestr('ob/something.3DVAR',obs)
        if confirmed:z.writestr('task.json',json.dumps(dict(schema='ecodata.wrfda-task.v1',mode='3dvar',observation_scope='conventional',confirmed=True)))
    controls=[dict(resource='namelist.input',var4d=False,sha256=hashlib.sha256(control.encode()).hexdigest())]
    return p,controls


def test_matching_and_compatibility_do_not_modify_original_bytes(tmp_path):
    p,controls=bundle(tmp_path);original=p.read_bytes()
    with ExitStack() as stack:
        files=ProjectFiles([LocalUpload(p.name,p)],stack)
        request,report=plan(files,controls,tmp_path/'request')
    assert p.read_bytes()==original
    assert report['observation_records_declared']==1
    assert len(report['configuration_changes'])==1
    assert report['configuration_changes'][0]['field']=='force_use_old_data'
    assert request['be.dat'].read_bytes()==b'original error bytes'


def test_radiance_cannot_be_silently_disabled(tmp_path):
    p,c=bundle(tmp_path,radiance=True)
    with ExitStack() as stack,pytest.raises(ValueError,match='卫星辐射'):
        plan(ProjectFiles([LocalUpload(p.name,p)],stack),c,tmp_path/'request')


def test_explicit_scope_retains_conventional_observations(tmp_path):
    import f90nml
    p,c=bundle(tmp_path,radiance=True,confirmed=True)
    with ExitStack() as stack:
        request,report=plan(ProjectFiles([LocalUpload(p.name,p)],stack),c,tmp_path/'request')
    n=f90nml.read(request['namelist.input'])
    assert n['wrfvar4']['use_synopobs'] is True
    assert n['wrfvar4']['use_amsuaobs'] is False
    assert n['wrfvar14']['rtminit_nsensor']==0


def test_exit_zero_does_not_mean_native_success(tmp_path):
    with pytest.raises(ValueError,match='未成功生成分析场'):
        validate({},tmp_path,dict(returncode=0,timed_out=False))


def test_fixed_width_multiword_platform_is_not_lost(tmp_path):
    p,c=bundle(tmp_path)
    with zipfile.ZipFile(p) as z:members={n:z.read(n) for n in z.namelist()}
    members['ob/something.3DVAR']=members['ob/something.3DVAR'].replace(b'FM-12 SYNOP  ',b'FM-36 TEMP S ')
    with zipfile.ZipFile(p,'w') as z:
        for n,v in members.items():z.writestr(n,v)
    with ExitStack() as stack:
        _,report=plan(ProjectFiles([LocalUpload(p.name,p)],stack),c,tmp_path/'request')
    assert report['observation_records_declared']==1


def test_archive_absolute_link_relocates_only_to_unique_internal_target(tmp_path):
    import tarfile
    from ecodata.models.archive_inputs import normalize_archives
    p=tmp_path/'a.tar.gz'
    with tarfile.open(p,'w:gz') as t:
        raw=b'original';i=tarfile.TarInfo('case/ob/records');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
        i=tarfile.TarInfo('case/ob/alias');i.type=tarfile.SYMTYPE;i.linkname='/old/machine/case/ob/records';t.addfile(i)
    uploads,_=normalize_archives([LocalUpload(p.name,p)],tmp_path)
    with zipfile.ZipFile(uploads[0].path) as z:assert z.read('case/ob/alias')==b'original'
