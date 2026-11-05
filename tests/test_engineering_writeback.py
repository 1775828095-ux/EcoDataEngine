"""Synthetic mechanism boundaries, not model accuracy experiments."""
from io import BytesIO
from pathlib import Path
import json
import zipfile
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.models.engineering.native import extract_native_field
from ecodata.engines.units import UnitEngine


def upload(name,content):
    f=BytesIO(content);f.name=name;return f


def sources(model='camx',shift=0):
    ds=xr.Dataset({'T':(('time','y','x'),np.full((2,2,3),280.),dict(units='K',standard_name='air_temperature'))},
                  coords=dict(time=('time',[0,1],dict(units='hours since 2021-01-01')),x=('x',np.arange(3.)+shift,dict(axis='X',units='m')),y=('y',[0.,1.],dict(axis='Y',units='m'))),attrs=dict(crs='EPSG:32650'))
    if model=='wrf':
        ds.attrs['TITLE']='OUTPUT FROM REAL_EM synthetic boundary test'
        ds['Times']=('time',np.array(['2021-01-01_00:00:00','2021-01-01_01:00:00'],dtype='S19'))
    else:
        ds['TFLAG']=(('time','VAR','DATE_TIME'),np.array([[[2021001,0]],[[2021001,10000]]]))
        ds.attrs.update({'CAMx_NAME':'2D METEOROLOGY'} if model=='camx' else {'IOAPI_VERSION':'synthetic header boundary test'})
    native=upload('project/state.nc',ds.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC'))
    p=extract_native_field([native],model,native.name,'T')
    p['T']=UnitEngine().convert(p['T'],'degC')
    p['T'].values[0,0,0]+=2  # explicit synthetic input change, no hidden truth
    product=upload('processed.nc',p.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC'))
    return native,product


@pytest.mark.parametrize('model',['camx','cmaq','wrf'])
def test_common_auto_binding_and_actual_native_writeback(tmp_path,model):
    native,product=sources(model)
    original=native.getvalue()
    r=prepare_uploads([product,native,upload('project/keep.txt',b'unrelated')],model,tmp_path)
    assert r['manifest']['status']=='PROJECT_UPDATED' and not r['manifest']['native_ready']
    assert r['manifest']['preparation']['tasks'][0]['validation']['result_quality']['accuracy_assessed'] is False
    with zipfile.ZipFile(r['archive']) as z:
        assert z.testzip() is None and z.read('project/keep.txt')==b'unrelated'
        with xr.open_dataset(BytesIO(z.read('project/state.nc'))) as ds:
            assert ds.T.values[0,0,0]==pytest.approx(282)
            assert np.count_nonzero(ds.T.values!=280)==1
        c=json.loads(z.read(r['manifest']['context_archive_path']))
        assert c['model_name']==model and c['domains'][0]['resource']==native.name
    assert native.getvalue()==original


def test_changed_coordinates_not_hidden_by_equal_shape(tmp_path):
    native,product=sources()
    with xr.open_dataset(BytesIO(product.getvalue()),decode_times=False) as d:p=d.load()
    p['x']=p.x+100
    with pytest.raises(ValueError,match='网格/时间/坐标'):
        prepare_uploads([native,upload('processed.nc',p.to_netcdf())],'camx',tmp_path)


def test_ambiguous_targets_and_wrong_model_fail(tmp_path):
    native,product=sources()
    with pytest.raises(ValueError,match='匹配数量'):
        prepare_uploads([native,upload('another.nc',native.getvalue()),product],'camx',tmp_path/'a')
    with pytest.raises(ValueError,match='目标模型'):
        prepare_uploads([native,product],'cmaq',tmp_path/'b')


def test_nested_task_declaration_does_not_rebase_absolute_resource(tmp_path):
    native,product=sources()
    req=dict(schema='ecodata.model-task.v1',model_id='camx',operation='update_project',input='../processed.nc',configuration={})
    r=prepare_uploads([native,product,upload('meta/request.json',json.dumps(req).encode())],'camx',tmp_path)
    assert r['manifest']['status']=='PROJECT_UPDATED'


def test_same_shape_different_project_not_accepted(tmp_path):
    _,p=sources();other,_=sources(shift=50)
    with pytest.raises(ValueError,match='匹配数量'):
        prepare_uploads([other,p],'camx',tmp_path)


def test_declared_nodata_preserved_and_all_missing_blocked(tmp_path):
    ds=xr.Dataset({'T':('x',[280.,np.nan],dict(units='K'))},coords=dict(x=[0,1]))
    ds.attrs['CAMx_NAME']='synthetic container boundary test'
    ds['TFLAG']=(('t','VAR','DATE_TIME'),np.array([[[2021001,0]]]))
    raw=ds.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC',encoding={'T':{'_FillValue':-9999.}})
    native=upload('state.nc',raw);p=extract_native_field([native],'camx',native.name,'T')
    p['T'].values[0]=281
    r=prepare_uploads([native,upload('product.nc',p.to_netcdf())],'camx',tmp_path/'valid')
    with zipfile.ZipFile(r['archive']) as z,xr.open_dataset(BytesIO(z.read('state.nc'))) as actual:
        np.testing.assert_array_equal(actual.T.values,[281,np.nan])
    p['T'].values[:]=np.nan
    with pytest.raises(ValueError,match='输入质量门'):
        prepare_uploads([native,upload('product.nc',p.to_netcdf())],'camx',tmp_path/'blocked')


def test_changed_time_and_bad_units_fail(tmp_path):
    native,product=sources()
    with xr.open_dataset(BytesIO(product.getvalue()),decode_times=False) as d:p=d.load()
    p['time']=p.time+1
    with pytest.raises(ValueError,match='网格/时间/坐标'):
        prepare_uploads([native,upload('processed.nc',p.to_netcdf())],'camx',tmp_path/'time')
    with xr.open_dataset(BytesIO(product.getvalue()),decode_times=False) as d:p=d.load()
    p['T'].attrs['units']='kg'
    with pytest.raises(Exception):
        prepare_uploads([native,upload('processed.nc',p.to_netcdf())],'camx',tmp_path/'units')


def test_real_swat_lon_header_alias_has_same_meaning():
    from ecodata.models.swat_weather_project import _weather_metadata
    source=b'title\nNBYR TSTEP LAT LONG ELEV\n1 0 24.99 -105.89 2490\n'
    assert _weather_metadata(source)==_weather_metadata(source.replace(b'LONG',b'lon'))


def test_generic_netcdf_is_not_called_a_native_model_file():
    ds=xr.Dataset({'T':('x',[280.],dict(units='K'))})
    native=upload('wrfinput_d01',ds.to_netcdf())
    with pytest.raises(ValueError,match='模型类型未知'):
        extract_native_field([native],'wrf',native.name,'T')


@pytest.mark.parametrize('condition',['category','range','bitmask'])
def test_native_declared_value_contracts_are_enforced(tmp_path,condition):
    attrs=dict(units='1')
    if condition=='category':attrs['flag_values']=[1.,2.]
    elif condition=='range':attrs['valid_range']=[0.,2.]
    else:attrs['flag_masks']=[1,2]
    ds=xr.Dataset({'value':('x',[1.,2.],attrs),'TFLAG':(('t','VAR','DATE_TIME'),[[[2021001,0]]])},attrs=dict(CAMx_NAME='synthetic boundary test'))
    native=upload('state.nc',ds.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC'))
    p=extract_native_field([native],'camx',native.name,'value');p['value'].values[0]=3
    with pytest.raises(ValueError,match='类别|有效范围|位掩码'):
        prepare_uploads([native,upload('product.nc',p.to_netcdf())],'camx',tmp_path)


def test_equal_units_do_not_allow_changed_statistical_meaning(tmp_path):
    native,product=sources()
    with xr.open_dataset(BytesIO(product.getvalue()),decode_times=False) as ds:p=ds.load()
    p['T'].attrs['cell_methods']='time: sum'
    with pytest.raises(ValueError,match='统计/物理语义'):
        prepare_uploads([native,upload('product.nc',p.to_netcdf())],'camx',tmp_path)


def test_support_hash_is_independent_of_numpy_display_settings():
    from ecodata.models.engineering.native import support_signature
    ds=xr.Dataset({'T':('x',[280.,281.],dict(units='K'))},coords=dict(x=[0,1]),
                  attrs=dict(grid_coefficients=np.array([1.23456789,4.76543210])))
    a=support_signature(ds,'T')
    with np.printoptions(precision=2):assert support_signature(ds,'T')==a


@pytest.mark.parametrize('version',[1,99])
def test_legacy_fingerprint_compatibility_and_unknown_version(tmp_path,version):
    from ecodata.models.engineering.native import support_signature
    native,product=sources()
    with xr.open_dataset(BytesIO(product.getvalue()),decode_times=False) as ds:p=ds.load()
    with xr.open_dataset(BytesIO(native.getvalue()),decode_times=False) as ds:legacy=support_signature(ds,'T',legacy=True)
    p.attrs.update(native_support_sha256=legacy,native_support_version=version)
    inputs=[native,upload('product.nc',p.to_netcdf())]
    if version==1:assert prepare_uploads(inputs,'camx',tmp_path)['manifest']['status']=='PROJECT_UPDATED'
    else:
        with pytest.raises(ValueError,match='指纹版本'):prepare_uploads(inputs,'camx',tmp_path)


def test_page_calls_same_writeback_service_and_downloads(tmp_path):
    from streamlit.testing.v1 import AppTest
    native,product=sources()
    folder=tmp_path/'input';(folder/'project').mkdir(parents=True)
    (folder/native.name).write_bytes(native.getvalue());(folder/product.name).write_bytes(product.getvalue())
    app=AppTest.from_file(Path(__file__).parents[1]/'src/ecodata/interface/app.py').run(timeout=30)
    app.selectbox[0].select('CAMx').run();app.text_input[0].set_value(str(folder)).run()
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.get('download_button')[0].label=='下载更新工程与核验记录'
    assert app.session_state['workbench_result']['data']['manifest']['native_writeback_verified']


def test_context_is_consumed_not_only_logged(tmp_path):
    from contextlib import ExitStack
    from ecodata.models.project_binding import ProjectFiles
    from ecodata.models.engineering.native import bind_native_product,finish_project
    native,product=sources()
    with ExitStack() as stack:
        cfg,context=bind_native_product(ProjectFiles([native,product],stack),'camx',product.getvalue(),{})
    cfg['target']='unrelated.nc'
    with pytest.raises(ValueError,match='统一工程上下文不一致'):
        finish_project([native,product],'camx',tmp_path,'update_project',{'native_updated.nc':native.getvalue()},cfg,{},context=context)


def test_known_control_rejects_unused_resource_without_borrowing_role(tmp_path):
    native,product=sources('wrf')
    with xr.open_dataset(BytesIO(native.getvalue()),decode_times=False) as ds:other=ds.load()
    other['T'].values[:]=281
    control=upload('project/namelist.input',b'&time_control\n restart=.false.,\n/\n&domains\n max_dom=1,\n/')
    active=upload('project/wrfinput_d01',other.to_netcdf())
    boundary=upload('project/wrfbdy_d01',b'synthetic dependency placeholder, not native validation')
    with pytest.raises(ValueError,match='已验证的工程输入端口'):
        prepare_uploads([native,product,control,active,boundary],'wrf',tmp_path)


@pytest.mark.parametrize('conflict',['grid','time'])
def test_control_requirements_actually_reject_conflicts(tmp_path,conflict):
    native,_=sources('wrf')
    with xr.open_dataset(BytesIO(native.getvalue()),decode_times=False) as ds:p=ds.load()
    p.attrs.update(START_DATE='2021-01-01_00:00:00',**{'WEST-EAST_GRID_DIMENSION':4,'SOUTH-NORTH_GRID_DIMENSION':3})
    native=upload('wrfinput_d01',p.to_netcdf())
    product=extract_native_field([native],'wrf',native.name,'T')
    day=2 if conflict=='time' else 1;nx=5 if conflict=='grid' else 4
    control=f'&time_control\n restart=.false.,start_year=2021,start_month=1,start_day={day},\n/\n&domains\n max_dom=1,e_we={nx},e_sn=3,\n/'
    with pytest.raises(ValueError,match='网格声明冲突|起始时间冲突'):
        prepare_uploads([native,upload('product.nc',product.to_netcdf()),upload('namelist.input',control.encode()),upload('wrfbdy_d01',b'synthetic dependency slot')],'wrf',tmp_path)


def test_swat_native_record_identity_path_uses_existing_writer(tmp_path):
    from test_swat_inputs_project import project
    from ecodata.models.engineering.native import finish_project
    original=project();blob=BytesIO()
    with zipfile.ZipFile(blob,'w') as z:
        for name,raw in original.items():z.writestr('region/'+name,raw)
    raw=b'converted\nid name attribute\n1 hru1 3.0\n2 hru2 5.0\n'
    f=upload('project.zip',blob.getvalue())
    r=finish_project([f],'swat_plus',tmp_path,'hru_attributes',{'hru-data.hru':raw},{},{'format':'validated by upstream writer'})
    with zipfile.ZipFile(r['archive']) as z:
        assert z.read('region/hru-data.hru')==raw
        assert z.read('region/file.cio')==original['file.cio']
    assert r['manifest']['status']=='PROJECT_UPDATED'

def test_nested_project_collection_does_not_use_legacy_local_case(tmp_path):
    inner=BytesIO()
    with zipfile.ZipFile(inner,'w') as z:z.writestr('namelist.input',b'&domains max_dom=1 /')
    outer=BytesIO()
    with zipfile.ZipFile(outer,'w') as z:
        z.writestr('WRF/project.zip',inner.getvalue())
        z.writestr('WRF/observations.csv','time,value\n')
    with pytest.raises(ValueError,match="WRF控制读取失败.*time_control"):
        prepare_uploads([upload('collection.zip',outer.getvalue())],'wrf',tmp_path)
