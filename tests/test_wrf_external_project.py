"""Small synthetic structural fixtures; these are not meteorological runs."""
import io
import json
import zipfile
import pytest
from netCDF4 import Dataset
from ecodata.interface.wrf_external_project import inspect_project,read_archive

NML=b'''&time_control
 start_year=2018, start_month=1, start_day=1, start_hour=0, start_minute=0, start_second=0,
 end_year=2018, end_month=1, end_day=1, end_hour=6, end_minute=0, end_second=0,
/
&domains
 max_dom=1,e_we=4,e_sn=4,e_vert=4,
/
'''

def nc(times,wrong_shape=False,boundary_next=None):
    d=Dataset('fixture','w',memory=10000,format='NETCDF3_CLASSIC')
    for key,n in [('Time',len(times)),('DateStrLen',19),('west_east',2 if wrong_shape else 3),('south_north',3),('bottom_top',3)]:d.createDimension(key,n)
    d.createVariable('Times','S1',('Time','DateStrLen'))[:]=[list(t) for t in times]
    if boundary_next is not None:
        for key,values in [('thisbdytime',times),('nextbdytime',boundary_next)]:
            d.createVariable('md___'+key+'e_x_t_d_o_m_a_i_n_m_e_t_a_data_','S1',('Time','DateStrLen'))[:]=[list(t) for t in values]
    d.TITLE='Synthetic structure fixture, not simulation data'
    d.MAP_PROJ=1
    return bytes(d.close())

def pack(files):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        for n,b in files.items():z.writestr(n,b)
    return buf.getvalue()

def project():
    return {'user/run/namelist.input':NML,'user/run/wrfinput_d01':nc(['2018-01-01_00:00:00']),
            'user/run/wrfbdy_d01':nc(['2018-01-01_00:00:00'],boundary_next=['2018-01-01_06:00:00'])}

def test_native_inputs_do_not_imply_pressure_replacement():
    report=inspect_project(pack(project()))
    assert report['native_structure_candidate']
    assert len(report['replacement_missing'])==3
    assert not report['replacement_checked'] and not report['native_model_executed']
    json.dumps(report)

@pytest.mark.parametrize('name',['../namelist.input','/namelist.input','C:/namelist.input','a\\namelist.input'])
def test_unsafe_paths_rejected(name):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        info=zipfile.ZipInfo('fixture');info.filename=name
        z.writestr(info,NML)
    with pytest.raises(ValueError,match='不安全'):read_archive(buf.getvalue())

def test_multiple_namelists_need_explicit_scope():
    data=project();data['user/real/namelist.input']=NML
    with pytest.raises(ValueError,match='恰好一个'):inspect_project(pack(data))

@pytest.mark.parametrize('change,needle',[
    ('missing','缺少与namelist'),('short_boundary','边界时段'),
    ('shape','维度'),('start','起始时间'),('nested','嵌套域')])
def test_scientific_contract_mismatches(change,needle):
    data=project()
    if change=='missing':del data['user/run/wrfbdy_d01']
    if change=='short_boundary':data['user/run/wrfbdy_d01']=nc(['2018-01-01_00:00:00'],boundary_next=['2018-01-01_03:00:00'])
    if change=='shape':data['user/run/wrfinput_d01']=nc(['2018-01-01_00:00:00'],True)
    if change=='start':data['user/run/wrfinput_d01']=nc(['2018-01-01_01:00:00'])
    if change=='nested':data['user/run/namelist.input']=NML.replace(b'max_dom=1',b'max_dom=2')
    report=inspect_project(pack(data))
    assert not report['native_structure_candidate']
    assert any(needle in e for e in report['errors'])

def test_uploaded_program_is_never_executed_or_counted_as_dependency():
    data=project();data['user/run/wrf.exe']=b'not an executable'
    report=inspect_project(pack(data))
    assert report['ignored_executable_files']==['user/run/wrf.exe']
    assert not report['native_model_executed']

def test_duplicate_zip_members_rejected():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('A',b'a');z.writestr('a',b'b')
    with pytest.raises(ValueError,match='重复'):read_archive(buf.getvalue())

def test_boundary_times_without_interval_metadata_are_not_guessed():
    data=project()
    data['user/run/wrfbdy_d01']=nc(['2018-01-01_00:00:00','2018-01-01_06:00:00'])
    report=inspect_project(pack(data))
    assert not report['native_structure_candidate']
    assert any('thisbdytime/nextbdytime' in e for e in report['errors'])

def test_external_page_never_runs_without_explicit_inputs_and_action():
    from streamlit.testing.v1 import AppTest
    from unittest.mock import patch
    from ecodata.interface import wrf_project
    page=AppTest.from_string('from ecodata.interface.wrf_project import render; render()').run()
    with patch.object(wrf_project,'start_external') as native:
        page.radio(key='wrf_custom_route').set_value('接入用户上传的外部工程（试验）').run()
    native.assert_not_called()
    assert not page.exception
    assert any('绝不覆盖成模板' in w.value for w in page.warning)

def test_explicit_namelist_selection_supports_two_real_directories():
    data=project();data['user/real/namelist.input']=NML
    report=inspect_project(pack(data),namelist_path='user/run/namelist.input')
    assert report['namelist']=='user/run/namelist.input'
    assert report['native_structure_candidate']

def test_scientific_settings_rejected_instead_of_overwritten(monkeypatch):
    from ecodata.interface import wrf_external_project as p
    reference=NML+b"&physics\n physics_suite='CONUS'\n/\n"
    changed=reference.replace(b"'CONUS'",b"'TROPICAL'")
    files={'run/namelist.input':changed,'real/namelist.input':reference}
    monkeypatch.setattr(p,'inspect_project',lambda *a,**k:{'replacement_checked':True})
    original=dict(files)
    with pytest.raises(ValueError,match='科学或IO设置'):
        p.validate_run_contract(pack(files),b'unused','run/namelist.input','real/namelist.input',reference,reference)
    assert files==original

def test_platform_export_preserves_supplied_file_bytes(tmp_path,monkeypatch):
    from ecodata.interface import wrf_project as p
    folder=tmp_path/('a'*32);folder.mkdir()
    (folder/'status.json').write_text('{"state":"succeeded"}')
    names=['wrf/namelist.input','real/namelist.input','real/wrfinput_d01','real/wrfbdy_d01','wps/namelist.wps','wps/geo_em.d01.nc','original_wps/time.wps']
    for name in names:
        path=folder/name;path.parent.mkdir(exist_ok=True);path.write_bytes(('fixture '+name).encode())
    monkeypatch.setattr(p,'RUNS',tmp_path)
    with zipfile.ZipFile(io.BytesIO(p.export_project('a'*32))) as z:
        assert z.read('wrf/wrfinput_d01')==(folder/'real/wrfinput_d01').read_bytes()
        assert z.read('wrf/namelist.input')==(folder/'wrf/namelist.input').read_bytes()
        assert not json.loads(z.read('platform_export.json'))['scientific_data_modified']
