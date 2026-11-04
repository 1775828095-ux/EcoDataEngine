"""Synthetic boundary/codec tests, not environmental accuracy evidence."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.models.native_meteorology import read_observations
from ecodata.models.wrf_meteorology import sample_daily_sst


def upload(name, raw):
    f=BytesIO(raw);f.name=name;return f


def wrf_fixture():
    x,y=np.meshgrid([0.,.5,1.],[0.,.5,1.])
    ds=xr.Dataset(dict(SST=(('Time','south_north','west_east'),np.full((1,3,3),280,dtype='f4'),dict(units='K')),
        XLAT=(('Time','south_north','west_east'),y[None]),XLONG=(('Time','south_north','west_east'),x[None]),
        LANDMASK=(('Time','south_north','west_east'),np.array([[[1,0,0],[0,0,0],[0,0,0]]])),
        SEAICE=(('Time','south_north','west_east'),np.array([[[0,1,0],[0,0,0],[0,0,0]]])),
        Times=('Time',np.array(['2023-03-31_00:00:00'],dtype='S19'))),
        attrs={'TITLE':'OUTPUT FROM REAL_EM synthetic boundary test','WEST-EAST_GRID_DIMENSION':4,'SOUTH-NORTH_GRID_DIMENSION':4,'START_DATE':'2023-03-31_00:00:00'})
    analysis=xr.Dataset(dict(sst=(('time','lat','lon'),np.full((1,2,2),22.),dict(units='Celsius',valid_min=-300,valid_max=4500))),
        coords=dict(time=pd.to_datetime(['2023-03-31T12:00']),lat=('lat',[0.,1.],dict(units='degrees_north')),lon=('lon',[0.,1.],dict(units='degrees_east'))),
        attrs=dict(title='NOAA/NCEI Optimum Interpolation synthetic schema test',time_coverage_start='2023-03-31T00:00:00Z',time_coverage_end='2023-03-31T23:59:59Z'))
    b=BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('project/namelist.input','&time_control start_year=2023, start_month=3, start_day=31, start_hour=0 /\n&domains max_dom=1,e_we=4,e_sn=4 /')
        z.writestr('project/wrfinput_d01',ds.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC'))
        z.writestr('project/wrfbdy_d01',b'unchanged boundary fixture')
        z.writestr('analysis.nc',analysis.to_netcdf(engine='scipy'))
    return b.getvalue()


def test_public_wrf_upload_continues_to_actual_native_writeback(tmp_path):
    source=wrf_fixture();u=upload('unrelated-name.zip',source)
    r=prepare_uploads([u],'wrf',tmp_path)
    assert r['manifest']['status']=='PROJECT_UPDATED'
    assert not r['manifest']['native_ready']
    with zipfile.ZipFile(r['archive']) as z:
        with xr.open_dataset(BytesIO(z.read('project/wrfinput_d01'))) as d:
            a=d.SST.values
            assert a[0,0,0]==280 and a[0,0,1]==280  # land, sea ice preserved
            np.testing.assert_allclose(a[0,1:],295.15,atol=1e-4)
        assert z.read('project/wrfbdy_d01')==b'unchanged boundary fixture'
        plan=json.loads(z.read('_ecodata/wrf_meteorology_plan.json'))
        assert plan['tasks'][0]['processed_cells']==7
        assert plan['tasks'][0]['accuracy_gain']=='NOT_EVALUATED'
    assert u.getvalue()==source


def test_daily_sst_does_not_borrow_other_day_or_resolve_conflicts():
    d=xr.Dataset(dict(sst=(('time','lat','lon'),np.full((1,2,2),20.),dict(units='degC'))),
        coords=dict(time=pd.to_datetime(['2023-03-31T12:00']),lat=[0.,1.],lon=[0.,1.]))
    a=('a',d,pd.Timestamp('2023-03-31'),pd.Timestamp('2023-03-31T23:59:59'))
    values,used=sample_daily_sst([a],pd.to_datetime(['2023-04-01']),np.array([[.5]]),np.array([[.5]]))
    assert np.isnan(values).all() and not used
    other=d.copy(deep=True);other.sst.values[:]=25
    with pytest.raises(ValueError,match='重叠值冲突'):
        sample_daily_sst([a,('b',other,a[2],a[3])],pd.to_datetime(['2023-03-31']),np.array([[.5]]),np.array([[.5]]))


def test_noaa_raw_units_qa_and_coordinate_changes():
    header='STATION,DATE,TMP,REPORT_TYPE,LATITUDE,LONGITUDE,ELEVATION\n'
    raw=header+'A,2023-03-31T00:00:00,"+0228,1",FM-12,35,-90,10\nA,2023-03-31T00:00:00,"+0206,1",SHEF,35,-90,10\n'
    obs,report=read_observations([upload('anything.csv',raw.encode())])
    assert len(obs)==1 and obs.value.iloc[0]==pytest.approx(295.95)
    assert report['qa_rejected']==1
    changed=raw+'A,2023-03-31T01:00:00,"+0230,1",FM-12,36,-90,10\n'
    with pytest.raises(ValueError,match='元数据重复'):
        read_observations([upload('anything.csv',changed.encode())])


def test_calpuff_conversion_contract_has_no_solver_dependency():
    from existing_air_converters.calpuff_native_bundle import contract,ROUTE
    m=dict(preparation=dict(route=ROUTE,station_ids=[1],station_binding_sha256='x'))
    assert contract(m)['dependencies']==[]
    m['preparation']['run_models']=True
    assert {x['name'] for x in contract(m)['dependencies']}=={'CALMET','CALPUFF'}


def test_calpuff_real_codec_export_uses_current_product_without_subprocess(tmp_path,monkeypatch):
    from existing_air_converters import calpuff_native_bundle as m
    ds=xr.Dataset({f:(('time','station_id'),np.full((2,1),v),dict(units=u)) for f,u,v in
        zip(m.FIELDS,m.UNITS,[2.,90.,5.,3.,280.,50.,1000.,0.])},
        coords=dict(time=pd.date_range('2023-01-01',periods=2,freq='h'),station_id=[1]),attrs=dict(station_binding_sha256='x'))
    source=tmp_path/'source.nc';ds.to_netcdf(source)
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('kept.txt','unchanged')
    plan={}
    for kind in ['CALMET','CALPUFF']:
        control=tmp_path/(kind+'.inp');control.write_text('synthetic control; not solver-validated')
        plan[kind]=dict(control_path=control.name,inputs=[],values=dict(LCFILES=['T']))
    manifest=dict(model_id='calpuff',preparation=dict(route=m.ROUTE,station_ids=[1],station_binding_sha256='x',
        timezone_west_hours=0,source_archive=str(archive),source_archive_sha256=m.digest(archive),
        handoff_dependencies=[],native_plan=plan),products=[dict(path=source.name,field=f,sha256=m.digest(source)) for f in m.FIELDS])
    path=tmp_path/'handoff.json';path.write_text(json.dumps(manifest),encoding='utf8')
    def forbidden(*a,**k):raise AssertionError('conversion must not invoke a solver or legacy process')
    monkeypatch.setattr(m.subprocess,'run',forbidden)
    out=tmp_path/'out';r=m.convert(path,out)
    assert r['status']=='PASS' and not r['native_ready']
    actual,times,ids,zone=m.read_surface((out/'SURF.DAT').read_bytes())
    np.testing.assert_allclose(actual[:,:,4],ds.temperature.values)
    assert r['validation']['native_model_executed'] is False


def test_swat_unknown_database_not_treated_as_engineering(tmp_path):
    import sqlite3
    from ecodata.models.swat_editor_project import _config
    path=tmp_path/'metadata.sqlite'
    with sqlite3.connect(path) as c:c.execute('create table metadata (key text)')
    assert _config(path) is None


def test_official_cio_header_is_recognised_without_rewriting(tmp_path):
    from ecodata.models.project_inventory import _swat
    from ecodata.models.project_binding import ProjectFiles
    from contextlib import ExitStack
    names='simulation basin climate connect channel reservoir routing_unit hru exco recall dr aquifer herd water_rights link hydrology structural hru_parm_db ops lum chg init soils decision_table regions'.split()
    raw='file.cio: written by SWAT+ editor v4.0.0 for SWAT+ 62.0.0\n'+'\n'.join(n+' null null null null null null null null null' for n in names)
    with ExitStack() as stack:
        rows=_swat(ProjectFiles([upload('project/file.cio',raw.encode())],stack))
    assert not any('声明或完整' in x for x in rows[0]['unknown'])

def test_wrf_time_storage_and_invalid_clock():
    from ecodata.models.wrf_meteorology import wrf_times
    labels=['2023-03-31_00:00:00','2023-03-31_01:00:00']
    whole=np.array(labels,dtype='S19');chars=np.array([list(x) for x in labels],dtype='S1')
    assert wrf_times(whole).equals(wrf_times(chars))
    with pytest.raises(ValueError,match='唯一递增'):
        wrf_times(np.array([labels[0],labels[0]],dtype='S19'))


def test_one_station_is_recorded_not_spread_over_wrf_domain(tmp_path):
    source=wrf_fixture();b=BytesIO()
    with zipfile.ZipFile(BytesIO(source)) as old,zipfile.ZipFile(b,'w') as z:
        for name in old.namelist():
            if name=='analysis.nc':continue
            data=old.read(name)
            if name.endswith('wrfinput_d01'):
                with xr.open_dataset(BytesIO(data)) as d:
                    d=d.load();d['T2']=d.SST.copy(deep=True);data=d.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC')
            z.writestr(name,data)
        z.writestr('obs.csv','STATION,DATE,TMP,REPORT_TYPE,LATITUDE,LONGITUDE,ELEVATION\nA,2023-03-31T00:00:00,"+0228,1",FM-12,0.5,0.5,10\n')
    r=prepare_uploads([upload('project.zip',b.getvalue())],'wrf',tmp_path)
    assert r['manifest']['status']=='ENGINEERING_BOUND'
    tasks=r['manifest']['preparation']['processing_plan']['tasks']
    assert tasks[0]['variable']=='T2' and tasks[0]['status']=='NOT_APPLICABLE'
    assert tasks[0]['accepted_stations']==1


def test_swat_bad_reference_is_rejected_before_official_export(tmp_path):
    import sqlite3
    from ecodata.interface.engineering_upload import LocalUpload
    from ecodata.models.swat_editor_project import prepare_editor_project
    p=tmp_path/'project.sqlite'
    with sqlite3.connect(p) as c:
        c.execute('create table project_config(project_name text,gis_type text,gis_version text,reference_db text)')
        c.execute("insert into project_config values ('basin','qgis','4.0.1','../../outside.sqlite')")
    out=tmp_path/'run';out.mkdir()
    with pytest.raises(ValueError,match='不安全|缺失|越界'):
        prepare_editor_project([LocalUpload(p.name,p)],'swat_plus',out)

def test_wgn_requires_real_monthly_support_not_empty_table(tmp_path):
    import sqlite3
    from ecodata.models.swat_editor_project import validate_wgn_database
    p=tmp_path/'wgn.sqlite'
    with sqlite3.connect(p) as c:
        c.execute('create table wgn(id integer,name text,lat real,lon real,elev real,rain_yrs integer)')
        fields='tmp_max_ave tmp_min_ave tmp_max_sd tmp_min_sd pcp_ave pcp_sd pcp_skew wet_dry wet_wet pcp_days pcp_hhr slr_ave dew_ave wnd_ave'.split()
        c.execute('create table wgn_mon(wgn_id integer, month integer,'+','.join(n+' real' for n in fields)+')')
    with pytest.raises(ValueError,match='为空'):
        validate_wgn_database(p,'wgn')
    with sqlite3.connect(p) as c:
        c.execute("insert into wgn values(1,'synthetic-schema',11,37,100,20)")
        c.executemany('insert into wgn_mon values('+','.join('?'*16)+')',[(1,n,*([.1]*14)) for n in range(1,12)])
    with pytest.raises(ValueError,match='12个月'):
        validate_wgn_database(p,'wgn')
    with sqlite3.connect(p) as c:c.execute('insert into wgn_mon values('+','.join('?'*16)+')',(1,12,*([.1]*14)))
    validate_wgn_database(p,'wgn')  # Schema only; no environmental score or native run.
    with pytest.raises(ValueError,match='不安全'):
        validate_wgn_database(p,'wgn;drop table wgn')

def test_wrf_page_and_toolkit_use_same_actual_processing(tmp_path):
    from streamlit.testing.v1 import AppTest
    folder=tmp_path/'input';folder.mkdir();(folder/'project.zip').write_bytes(wrf_fixture())
    direct=prepare_uploads([upload('project.zip',wrf_fixture())],'wrf',tmp_path/'direct')
    app=AppTest.from_file(Path(__file__).parents[1]/'src/ecodata/interface/app.py').run(timeout=30)
    app.selectbox[0].select('WRF').run();app.text_input[0].set_value(str(folder)).run()
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    page=app.session_state['workbench_result']['data']
    assert page['manifest']['status']==direct['manifest']['status']=='PROJECT_UPDATED'
    assert app.get('download_button')[0].label=='下载更新工程与核验记录'
    with zipfile.ZipFile(page['archive']) as a,zipfile.ZipFile(direct['archive']) as b:
        assert a.read('project/wrfinput_d01')==b.read('project/wrfinput_d01')
        assert json.loads(a.read('_ecodata/wrf_meteorology_plan.json'))==json.loads(b.read('_ecodata/wrf_meteorology_plan.json'))

def test_wgn_csv_pairing_uses_fields_and_foreign_keys_not_filenames():
    from contextlib import ExitStack
    from pathlib import PurePosixPath
    from ecodata.models.project_binding import ProjectFiles
    from ecodata.models.swat_editor_project import find_wgn_csv,WGN_FIELDS
    stat=b'id,name,lat,lon,elev,rain_yrs\n1,synthetic_contract,11,37,100,10\n'
    month=('id,wgn_id,month,'+','.join(WGN_FIELDS)+'\n'+'\n'.join(','.join([str(n),'1',str(n),*(['0.1']*14)]) for n in range(1,13))+'\n').encode()
    a=upload('basin/arbitrary-a.csv',stat);b=upload('basin/arbitrary-b.csv',month)
    unrelated=upload('other/stat.csv',stat)
    with ExitStack() as stack:
        r=find_wgn_csv(ProjectFiles([a,b,unrelated],stack),PurePosixPath('basin'))
    with ExitStack() as stack:
        s=find_wgn_csv(ProjectFiles([upload('basin/renamed-m.csv',month),upload('basin/renamed-s.csv',stat)],stack),PurePosixPath('basin'))
    assert r['station_rows']==s['station_rows'] and r['month_rows']==s['month_rows']
    assert len(r['month_rows'])==12
    bad=month.replace(b'1,1,1,',b'1,2,1,',1)
    with ExitStack() as stack,pytest.raises(ValueError,match='外键'):
        find_wgn_csv(ProjectFiles([upload(a.name,stat),upload(b.name,bad)],stack),PurePosixPath('basin'))
    missing=b'\n'.join(month.splitlines()[:-1])+b'\n'
    with ExitStack() as stack,pytest.raises(ValueError,match='12个月'):
        find_wgn_csv(ProjectFiles([upload(a.name,stat),upload(b.name,missing)],stack),PurePosixPath('basin'))
