"""Synthetic contract tests. Never environmental accuracy or solver evidence."""
from io import BytesIO
from pathlib import Path
import json,zipfile
import numpy as np,pandas as pd,xarray as xr,pytest
from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.models.engineering.ports import canonical
from ecodata.models.adapters.calmet_surface import export_surface


@pytest.fixture(autouse=True)
def bounded_search(monkeypatch):
 import ecodata.algorithms.platform_search as service
 monkeypatch.setattr(service,"CURRENT_INPUT_BUDGET",dict(objective_limit=40,proposal_limit=100,total_call_limit=200,wall_seconds=8))


def assert_public_outcome(result):
 manifest=result["manifest"]
 if manifest["status"]=="ENGINEERING_INSPECTED":
  assert not manifest["native_writeback_verified"]
  assert manifest["preparation"]["source_admission"]["required_information"]
  return False
 assert manifest["status"]=="PROJECT_UPDATED"
 tasks=manifest["preparation"]["tasks"]
 assert tasks and all(t["execution"]["route"]=="platform_search.process_current_input" for t in tasks)
 assert all(t["execution"]["result"]["selected_graph"] for t in tasks)
 return True


def upload(name,raw):
 b=BytesIO(raw);b.name=name;return b


def sources(quantity,unit,temporal,lon,lat,values,station=False,sigma=None):
 if station:
  d=xr.Dataset({'arbitrary_name':(('time','station_id'),values,dict(units=unit,physical_variable=quantity,temporal_semantics=temporal,data_kind='continuous'))},coords=dict(time=pd.date_range('2020-01-01',periods=2,freq='h'),station_id=[1]))
 else:
  d=canonical(values,pd.date_range('2020-01-01',periods=2,freq='h'),lon,lat,quantity,unit,temporal).rename({quantity:'arbitrary_name'})
 d.attrs.update(engineering_role='source',source_id='source-1',crs='EPSG:4326',time_basis='UTC')
 if sigma is not None:
  d['sigma']=(d.arbitrary_name.dims,np.full(d.arbitrary_name.shape,sigma),dict(units=unit,uncertainty_type='standard_deviation'))
  d.arbitrary_name.attrs['ancillary_variables']='sigma'
 return d


def pack(model,quantity,unit,temporal,*,value=280.):
 files={};lon,lat=np.meshgrid([0.,1.],[0.,1.]);times=pd.date_range('2020-01-01',periods=2,freq='h')
 if model in {'wrf','camx','cmaq'}:
  ds=xr.Dataset({'custom_field':(('time','y','x'),np.full((2,2,2),value,dtype='f4'),dict(units=unit,physical_variable=quantity,temporal_semantics=temporal))},
   coords=dict(time=times,lon=(('y','x'),lon),lat=(('y','x'),lat)))
  if model=='wrf':ds.attrs['TITLE']='OUTPUT FROM REAL_EM synthetic';ds['Times']=('time',np.array([t.strftime('%Y-%m-%d_%H:%M:%S') for t in times],dtype='S19'))
  else:
   ds['TFLAG']=(('time','VAR','DATE_TIME'),np.array([[[2020001,0]],[[2020001,10000]]]))
   ds.attrs.update(dict(CAMx_NAME='synthetic input') if model=='camx' else dict(IOAPI_VERSION='synthetic input'))
  ds.attrs['time_basis']='UTC'
  name='project/unusual.nc';files[name]=ds.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC')
  return files,name,lon,lat,(2,4)
 if model=='swat_plus':
  files={'project/file.cio':b'title\nsimulation time.sim null\nclimate weather-sta.cli null null pcp.cli tmp.cli null null null null\n',
   'project/time.sim':b'title\nday_start yrc_start day_end yrc_end step\n1 2020 2 2020 0\n',
   'project/weather-sta.cli':b'title\nname wgn pcp tmp slr hmd wnd\nS null rain.pcp temp.tem null null null\n',
   'project/pcp.cli':b'title\nfilename\nrain.pcp\n','project/tmp.cli':b'title\nfilename\ntemp.tem\n',
   'project/rain.pcp':b'title\nNBYR TSTEP LAT LON ELEV\n1 0 0 0 1\n2020 1 1\n2020 2 2\n',
   'project/temp.tem':b'title\nNBYR TSTEP LAT LON ELEV\n1 0 0 0 1\n2020 1 20 10\n2020 2 21 11\n'}
  return files,'project/rain.pcp',np.array([0.]),np.array([0.]),(2,1)
 frame=pd.DataFrame([dict(time=t.isoformat()+'Z',station_id=1,wind_speed=2,wind_direction=90,ceiling=5,opaque_cover=3,temperature=280,relative_humidity=50,station_pressure=1000,precip_code=0) for t in times])
 from existing_air_converters.calpuff_native_bundle import FIELDS,UNITS
 cfg=dict(version='CALMET-5.8.5-SURF-2.0',timezone_west_hours=0,time_label='hour_end',station_ids=[1],missing_policy='preserve',columns={f:f for f in FIELDS},units=dict(zip(FIELDS,UNITS)))
 output,_=export_surface(frame,cfg);name='project/surface.dat'
 return {name:output['SURF.DAT']},name,None,None,(2,1)


def request(model,quantity='air_temperature',multi=False):
 unit='K';temporal='instantaneous'
 if model=='swat_plus':quantity='precipitation';unit='mm';temporal='accumulation'
 if model=='calpuff':temporal='hour_end_observation'
 files,name,lon,lat,shape=pack(model,quantity,unit,temporal)
 a=sources(quantity,unit,temporal,lon,lat,np.full(shape,282. if unit=='K' else 3.),station=model=='calpuff',sigma=1. if multi else None)
 if model=='swat_plus':
  a=a.assign_coords(time=pd.to_datetime(['2020-01-01','2020-01-02']))
  a.arbitrary_name.attrs['interval_seconds']=86400
  a.attrs['time_basis']='model_calendar_day'
 files['source.nc']=a.to_netcdf(engine='scipy')
 if multi:
  b=a.copy(deep=True);b.arbitrary_name.values[:]+=2;b.attrs['source_id']='source-2';files['source_2.nc']=b.to_netcdf(engine='scipy')
 files['semantics.json']=json.dumps(dict(schema='ecodata.engineering-semantics.v1',model_id=model,resources=[dict(resource=name,role='model_input',variables={},confirmed=True)])).encode()
 blob=BytesIO()
 with zipfile.ZipFile(blob,'w') as z:
  for k,v in files.items():z.writestr(k,v)
 return upload('no-case-name.zip',blob.getvalue())


@pytest.mark.parametrize('model',['wrf','camx','cmaq','swat_plus','calpuff'])
def test_five_models_same_public_port_processor(tmp_path,model):
 r=prepare_uploads([request(model)],model,tmp_path)
 assert_public_outcome(r)
 assert not r['manifest']['native_ready'] and not r['manifest']['native_model_executed']


@pytest.mark.parametrize('model',['wrf','camx','cmaq','swat_plus','calpuff'])
def test_five_models_real_fusion_kernel_consumed(tmp_path,model):
 r=prepare_uploads([request(model,multi=True)],model,tmp_path)
 if assert_public_outcome(r):
  assert r['manifest']['preparation']['tasks'][0]['updated_cells']>0
  assert r['manifest']['preparation']['tasks'][0]['output_quality']['accuracy_assessed'] is False


def mutate_request(model,change):
 original=request(model)
 with zipfile.ZipFile(original) as z:files={n:z.read(n) for n in z.namelist()}
 change(files)
 blob=BytesIO()
 with zipfile.ZipFile(blob,'w') as z:
  for n,v in reversed(list(files.items())):z.writestr(n,v)
 return upload('renamed.zip',blob.getvalue())


def change_source(files,**attrs):
 with xr.open_dataset(BytesIO(files['source.nc'])) as opened:d=opened.load()
 d.arbitrary_name.attrs.update(attrs)
 files['source.nc']=d.to_netcdf(engine='scipy')


@pytest.mark.parametrize('model',['wrf','camx','cmaq','swat_plus','calpuff'])
def test_rename_order_and_unrelated_members_preserved(tmp_path,model):
 r=prepare_uploads([mutate_request(model,lambda f:f.update({'do-not-change.bin':b'original-user-bytes'}))],model,tmp_path)
 if assert_public_outcome(r):
  with zipfile.ZipFile(r['archive']) as z:assert z.read('do-not-change.bin')==b'original-user-bytes'


@pytest.mark.parametrize('model',['wrf','camx','cmaq','swat_plus','calpuff'])
@pytest.mark.parametrize('attrs,reason',[
 ({'units':'kg'},'单位|unit|convert|换算|转换|维度'),
 ({'temporal_semantics':'unknown_interval'},'时间统计'),
 ({'time_basis':'UNCONFIRMED'},'时间基准'),
 ({'physical_definition':'different physical definition'},'物理定义')])
def test_five_models_same_scientific_rejection(tmp_path,model,attrs,reason):
 from ecodata.core.exceptions import SchemaError
 with pytest.raises((ValueError,SchemaError),match=reason):
  prepare_uploads([mutate_request(model,lambda f:change_source(f,**attrs))],model,tmp_path)


def test_target_declaration_cannot_overwrite_actual_metadata(tmp_path):
 def alter(files):
  doc=json.loads(files['semantics.json']);doc['resources'][0]['variables']={'custom_field':{'temporal_semantics':'accumulation'}}
  files['semantics.json']=json.dumps(doc).encode()
 with pytest.raises(ValueError,match='元数据冲突'):
  prepare_uploads([mutate_request('wrf',alter)],'wrf',tmp_path)


def test_daily_interval_length_is_not_just_matching_date(tmp_path):
 with pytest.raises(ValueError,match='区间长度'):
  prepare_uploads([mutate_request('swat_plus',lambda f:change_source(f,interval_seconds=3600))],'swat_plus',tmp_path)


def test_new_quantity_not_temperature_specific(tmp_path):
 files,name,lon,lat,shape=pack('wrf','air_pressure','Pa','instantaneous',value=100000)
 a=sources('air_pressure','hPa','instantaneous',lon,lat,np.full(shape,1005.))
 files['source.nc']=a.to_netcdf(engine='scipy')
 files['semantics.json']=json.dumps(dict(schema='ecodata.engineering-semantics.v1',model_id='wrf',resources=[dict(resource=name,role='model_input',variables={},confirmed=True)])).encode()
 blob=BytesIO()
 with zipfile.ZipFile(blob,'w') as z:
  for n,v in files.items():z.writestr(n,v)
 r=prepare_uploads([upload('pressure.zip',blob.getvalue())],'wrf',tmp_path)
 with zipfile.ZipFile(r['archive']) as z,xr.open_dataset(BytesIO(z.read(name))) as output:
  np.testing.assert_allclose(output.custom_field,100500.)


def test_no_uncertainty_does_not_invent_equal_weights(tmp_path):
 original=request('wrf',multi=True)
 with zipfile.ZipFile(original) as z:files={n:z.read(n) for n in z.namelist()}
 for name in ['source.nc','source_2.nc']:
  with xr.open_dataset(BytesIO(files[name])) as d:a=d.load().drop_vars('sigma')
  del a.arbitrary_name.attrs['ancillary_variables'];files[name]=a.to_netcdf(engine='scipy')
 stream=BytesIO()
 with zipfile.ZipFile(stream,'w') as z:
  for n,v in files.items():z.writestr(n,v)
 result=prepare_uploads([upload('unweighted.zip',stream.getvalue())],'wrf',tmp_path)
 assert assert_public_outcome(result)


def test_uncertainty_not_point_interpolated_without_covariance():
 from ecodata.models.engineering.preparation import _source_on_port
 from ecodata.models.engineering.ports import InputPort
 d=sources('air_temperature','K','instantaneous',[0.,1.],[0.,1.],np.ones((2,2))*280,sigma=2.)
 target=canonical(np.ones((2,1))*280,d.time.values,[.5],[.5],'air_temperature','K','instantaneous');target.attrs['time_basis']='UTC'
 p=InputPort('native','T','air_temperature','K','instantaneous','netcdf',target,{})
 with pytest.raises(ValueError,match='协方差'):
  _source_on_port(dict(dataset=d,field='arbitrary_name',source_id='a'),p)


def test_public_native_guard_and_format_units():
 from ecodata.models.native_source_inspection import check_native_values
 from ecodata.engines.units import UnitEngine
 with pytest.raises(ValueError,match='flag_values'):check_native_values(dict(flag_values=[1,2]),np.array([3]))
 with pytest.raises(ValueError,match='有效范围'):check_native_values(dict(valid_range=[0,1]),np.array([2]))
 assert UnitEngine().affine('hundred_ft','m')==pytest.approx((30.48,0))
 assert UnitEngine().affine('tenths','1')==pytest.approx((.1,0))
 assert UnitEngine().affine('fraction','%')==pytest.approx((100,0))


@pytest.mark.parametrize('model,label',[('wrf','WRF'),('camx','CAMx'),('cmaq','CMAQ'),('swat_plus','SWAT+'),('calpuff','CALPUFF')])
def test_new_common_route_page_and_toolkit(tmp_path,model,label):
 from streamlit.testing.v1 import AppTest
 folder=tmp_path/'input';folder.mkdir();(folder/'request.zip').write_bytes(request(model,multi=True).getvalue())
 direct=prepare_uploads([request(model,multi=True)],model,tmp_path/'direct')
 app=AppTest.from_file(Path(__file__).parents[1]/'src/ecodata/interface/app.py').run(timeout=30)
 app.selectbox[0].select(label).run();app.text_input[0].set_value(str(folder.resolve())).run()
 app.button[0].click().run(timeout=30)
 assert not app.exception and not app.error
 page=app.session_state['workbench_result']['data']
 assert page['manifest']['status']==direct['manifest']['status']
 assert_public_outcome(page)
 if page['manifest']['status']=='PROJECT_UPDATED':assert page['manifest']['changes']==direct['manifest']['changes']


def test_shared_decision_is_a_real_fusion_consumer(tmp_path,monkeypatch):
 import ecodata.algorithms.platform_search as service
 original=service.process_current_input;calls=[]
 def observe(inputs,request,folder,**kwargs):
  calls.append(request);return original(inputs,request,folder,**kwargs)
 monkeypatch.setattr(service,'process_current_input',observe)
 result=prepare_uploads([request('wrf',multi=True)],'wrf',tmp_path)
 assert assert_public_outcome(result) and calls
 assert all(r['mutation_control']=='random' for r in calls)


def test_native_vocabulary_does_not_assume_wind_rotation_or_specific_humidity():
 from ecodata.models.engineering.ports import NATIVE_QUANTITIES
 assert NATIVE_QUANTITIES['wrf']['Q2']=='humidity_mixing_ratio'
 for model in ['wrf','camx']:
  assert not any(n in NATIVE_QUANTITIES[model] for n in ['U10','V10','u10','v10'])
