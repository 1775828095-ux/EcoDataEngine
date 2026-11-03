"""Real writer + existing project tests; metadata fixture is NOT UI acceptance.

The production input cannot pass the frozen upstream spatial gate. This test
adds coordinates only within the test fixture to exercise the downstream code.
It must never be reported as end-to-end platform/native-run evidence.
"""
from pathlib import Path
import hashlib,io,json,zipfile
import pytest
import xarray as xr
import local_swat_converter as plugin

SOURCE=Path('C:/Users/ASUS/Documents/ChatGPT/测试重/Hubbard已有工程替换资料_20261031/01_平台上传/A_HQ_1964_1968.nc')


def fixture_manifest(tmp_path,monkeypatch,mutation=None):
    cfg=plugin._binding();cfg['run_native']=False
    monkeypatch.setattr(plugin,'_binding',lambda:cfg)
    ds=xr.open_dataset(io.BytesIO(SOURCE.read_bytes())).load()
    ds=ds.assign_coords(lon=cfg['longitude'],lat=cfg['latitude']);ds.attrs['crs']='EPSG:4326'
    if mutation:ds=mutation(ds)
    p=tmp_path/'product.nc';ds.to_netcdf(p,engine='h5netcdf');products=[];tasks=[]
    # One product may carry all fields; tasks explicitly name each used field.
    for field in ds.data_vars:
        products.append(dict(path='product.nc',sha256=plugin.sha(p)))
        tasks.append(dict(field=field))
    m=dict(protocol='ecodata.converter.v1',model_id='swat_plus',products=products,preparation=dict(tasks=tasks))
    path=tmp_path/'handoff.json';path.write_text(json.dumps(m),encoding='utf8')
    return path,cfg


def test_real_writer_and_project_connection_fixture_only(tmp_path,monkeypatch):
    m,cfg=fixture_manifest(tmp_path,monkeypatch)
    result=plugin.convert(m,tmp_path/'converted')
    assert result['status']=='PASS'
    assert result['validation']['native']['status']=='NOT_RUN'
    assert plugin.sha(cfg['project_zip'])==cfg['project_sha256']
    with zipfile.ZipFile(tmp_path/'converted/SWATplus-current-project.zip') as z:
        assert z.testzip() is None and 'file.cio' in z.namelist()


@pytest.mark.parametrize('kind',['missing_project','missing_variable','unit','time','space','hash','escape'])
def test_failure_is_explicit_and_does_not_change_original(tmp_path,monkeypatch,kind):
    def mutation(ds):
        if kind=='missing_variable':return ds.drop_vars('precipitation')
        if kind=='unit':ds.precipitation.attrs['units']='unknown'
        if kind=='time':return ds.isel(time=slice(1,None))
        if kind=='space':ds=ds.assign_coords(lon=0.)
        return ds
    m,cfg=fixture_manifest(tmp_path,monkeypatch,mutation)
    original=cfg['project_sha256'];source=cfg['project_zip']
    if kind=='missing_project':cfg['project_zip']=str(tmp_path/'missing.zip')
    if kind in {'hash','escape'}:
        d=json.loads(m.read_text());d['products'][0].update({'sha256':'bad'} if kind=='hash' else {'path':'../escape.nc'});m.write_text(json.dumps(d))
    with pytest.raises(ValueError):plugin.convert(m,tmp_path/'converted')
    assert plugin.sha(source)==original
    assert not (tmp_path/'converted/validation.json').exists()
