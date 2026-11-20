"""Synthetic mechanism boundaries; not environmental accuracy evidence."""
import numpy as np,pandas as pd,pytest,xarray as xr
from ecodata.models.engineering.collocation import process_collocations
from ecodata.models.engineering.schedule import dependency_order,materialize_controls
from ecodata.models.project_binding import bind_project
from test_camx_template_schedule import SCRIPT,analyze
from io import BytesIO
import zipfile


def fixture():
 times=pd.date_range('2022-02-01',periods=3,freq='h');x=np.arange(3.);y=np.arange(3.);lon,lat=np.meshgrid(x,y)
 bg=np.array([280+lon+2*lat+i for i in range(3)])
 obs=pd.DataFrame(dict(time=[times[0],times[1]+pd.Timedelta(minutes=30)],station=['one','one'],longitude=[.2,.2],latitude=[.3,.3],value=[300.,310.]))
 return times,x,y,lon,lat,bg,obs


def run(bg,obs):
 t,x,y,lon,lat,_,_=fixture()
 return process_collocations(bg,t,x,y,obs,lambda x,y:(x,y),lon,lat,variable='air_temperature',unit='K')


def test_single_station_local_path_consumes_real_upstream_and_preserves_support():
 *_,bg,obs=fixture();product,report=run(bg,obs)
 values,ti,cells,out=product
 assert ti.tolist()==[0] and cells.tolist()==[0]
 assert values[0,0]==pytest.approx(299.2)
 assert report['processed_observations']==1 and report['rejected'][0]['reason'].startswith('not an exact')
 assert not report['accuracy_assessed'] and not report['extrapolation']
 assert any(t['operation']=='fusion.residual' and len(t['edges'])==2 for t in report['trace'])
 changed=bg.copy();changed[0,0,0]+=10
 other,_=run(changed,obs);assert other[0][0,0]!=values[0,0]
 obs=obs.copy();obs.loc[0,'value']+=5
 other,_=run(bg,obs);assert other[0][0,0]==pytest.approx(values[0,0]+5)


def test_outside_missing_and_collision_are_not_silently_fused():
 *_,bg,obs=fixture();obs=obs.iloc[:1].copy();obs.longitude=100
 p,r=run(bg,obs);assert p is None and r['status']=='NO_COLLOCATED_SUPPORT'
 *_,bg,obs=fixture();obs=obs.iloc[:1].copy();bg[0,:,:]=np.nan
 p,r=run(bg,obs);assert p is None
 *_,bg,obs=fixture();obs=obs.iloc[:1].copy();extra=obs.copy();extra.station='other'
 with pytest.raises(ValueError,match='聚合契约'):run(bg,pd.concat([obs,extra]))


def test_temporal_unknown_and_nonincreasing_are_rejected():
 t,x,y,lon,lat,bg,obs=fixture()
 with pytest.raises(ValueError,match='非递增'):process_collocations(bg,t[::-1],x,y,obs,lambda x,y:(x,y),lon,lat,variable='air_temperature',unit='K')


def test_buffer_support_is_preserved():
 t,x,y,lon,lat,bg,obs=fixture();mask=np.ones((3,3),dtype=bool);mask[0]=False
 p,r=process_collocations(bg,t,x,y,obs,lambda x,y:(x,y),lon,lat,variable='air_temperature',unit='K',support_mask=mask)
 assert p is None and 'buffer preserved' in r['rejected'][0]['reason']


def test_native_buffer_coordinates_and_restricted_writeback(tmp_path):
 from netCDF4 import Dataset
 from ecodata.models.native_meteorology import native_axes,write_native_copy
 source=tmp_path/'original.nc';target=tmp_path/'updated.nc'
 with Dataset(source,'w') as d:
  for n,s in [('TSTEP',2),('LAY',1),('ROW',5),('COL',6)]:d.createDimension(n,s)
  d.setncatts(dict(NCOLS=4,NROWS=3,NCOLS_BUF=6,NROWS_BUF=5,XORIG=100.,YORIG=200.,XORIG_BUF=90.,YORIG_BUF=190.,XCELL=10.,YCELL=10.))
  for n,s,o in [('X',6,90.),('Y',5,190.)]:
   v=d.createVariable(n,'f4',('COL' if n=='X' else 'ROW',));v.units='km';v[:]=(o+(np.arange(s)+.5)*10)/1000
  for n in ['t2','other']:
   v=d.createVariable(n,'f4',('TSTEP','LAY','ROW','COL'));v[:]=280+np.arange(60).reshape(2,1,5,6)
 with Dataset(source) as d:
  x,y=native_axes(d);assert len(x)==6 and x[0]==95 and len(y)==5
 r=write_native_copy(source,target,'t2',np.array([[300.]]),np.array([0]),np.array([7]))
 assert r['changed_values']==1 and r['other_variables_and_metadata_equal']
 with Dataset(source,'r+') as d:d.XORIG_BUF=80.
 with Dataset(source) as d:
  with pytest.raises(ValueError,match='缓冲'):native_axes(d)


def bundle(script=SCRIPT):
 data=BytesIO();data.name='anything.zip'
 with zipfile.ZipFile(data,'w') as z:
  z.writestr('project/runfiles/worker.job',script)
  for name in ['chem','photo','ozone','ic','boundary.20160610','boundary.20160611']:z.writestr('project/met/'+name,b'original')
 return data


def test_template_schedule_binds_generated_dependencies_and_writes_controls(tmp_path):
 b=bind_project([bundle()],'camx')
 assert b['execution_order']==[0,1] and b['runtime_dependencies']['1']==[0]
 mapping={r['path']:'inputs/'+str(i) for i,r in enumerate(b['resources'])}
 receipts=materialize_controls(b,mapping,tmp_path)
 assert len(receipts)==2 and all(not r['model_executed'] for r in receipts)
 import f90nml
 second=f90nml.read(tmp_path/'CAMx_run_1.in')['camx_control']
 assert second['master_grid_restart']=='outputs/run_0/CAMx.inst'
 assert second['restart'] is True


def test_ambiguity_and_invalid_runtime_state_are_explicit():
 a=bundle();b=bundle(SCRIPT.replace('restart=.', 'restart=.'))
 # Equivalent executable variants group; scientific alternatives do not.
 b=BytesIO();b.name='other.zip'
 with zipfile.ZipFile(b,'w') as z:z.writestr('project/runfiles/other.job',SCRIPT.replace('end_date_hour=2016,6,${CAL},2400','end_date_hour=2016,6,${CAL},1200'))
 with pytest.raises(ValueError,match='科学配置不同'):bind_project([a,b],'camx')
 with pytest.raises(ValueError,match='生产者'):dependency_order([dict(start=[2020,1,1,0])],[dict(status='GENERATED_DEPENDENCY',run_index=0,producer_required=dict(run_start=[2020,1,1,0]))])


def test_public_multigrid_plain_control_and_local_writeback(tmp_path):
 """Small synthetic engineering fixture, unrelated names/date/grid sizes."""
 from netCDF4 import Dataset
 from pyproj import CRS,Transformer
 from test_project_binding import case,uploaded
 from ecodata.algorithms.automatic_preparation import prepare_uploads
 files=case(root='another_engineering_tree')
 for i,cell,origin in [(0,36000.,0.),(1,12000.,48000.)]:
  p=tmp_path/f'native_{i}.nc'
  with Dataset(p,'w',format='NETCDF3_CLASSIC') as d:
   for n,s in [('TSTEP',2),('DATE-TIME',2),('VAR',1),('LAY',1),('ROW',4),('COL',4)]:d.createDimension(n,s)
   d.setncatts(dict(NCOLS=4,NROWS=4,XORIG=origin,YORIG=origin,XCELL=cell,YCELL=cell,
       GDTYP=2,CPROJ=2,ITZON=0,P_ALP=33.,P_BET=45.,XCENT=-97.,YCENT=40.,CAMx_NAME='2D METEOROLOGY'))
   for name,dim in [('X','COL'),('Y','ROW')]:
    v=d.createVariable(name,'f4',(dim,));v.units='km';v[:]=(origin+(np.arange(4)+.5)*cell)/1000
   d.createVariable('TFLAG','i4',('TSTEP','VAR','DATE-TIME'))[:]=np.array([[[2022032,0]],[[2022032,10000]]])
   v=d.createVariable('t2','f4',('TSTEP','LAY','ROW','COL'));v.units='K';v.long_name='temperature at 2 m';v[:]=280.
   v=d.createVariable('untouched','f4',('ROW','COL'));v[:]=123.
  files[f'another_engineering_tree/inputs/met{i}.nc']=p.read_bytes()
 files.pop('another_engineering_tree/inputs/met.nc')
 control=files['another_engineering_tree/control.any'].replace("Met2D_Grid(1)='inputs/met.nc',", "Met2D_Grid(1)='inputs/met0.nc', Met2D_Grid(2)='inputs/met1.nc',\n Master_Grid_Columns=4, Master_Grid_Rows=4, Nest_Meshing_Factor(1)=3,\n Start_Date_Hour=2022,2,1,0, End_Date_Hour=2022,2,1,100,")
 files['another_engineering_tree/control.any']=control
 crs=CRS.from_proj4('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +R=6370000 +units=m')
 lon,lat=Transformer.from_crs(crs,4326,always_xy=True).transform(60000,60000)
 files['observation.csv']=f'station,time_utc,temperature_C,temperature_quality,report_type,latitude,longitude,elevation_m\nstation,2022-02-01T00:00:00Z,20,1,FM-15,{lat},{lon},100\n'
 result=prepare_uploads([uploaded(files)],'camx',tmp_path/'result')
 assert result['manifest']['native_writeback_verified'] and not result['manifest']['native_model_executed']
 checks=result['manifest']['validation'];assert len(checks)==2 and all(c['changed_values']==1 for c in checks)
 assert all(c['other_variables_and_metadata_equal'] for c in checks)
 with zipfile.ZipFile(result['archive']) as z:assert z.testzip() is None
