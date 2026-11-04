"""Bounded declaration/port tests. No shell process is invoked."""
from io import BytesIO
from contextlib import ExitStack
import numpy as np,pandas as pd,pytest,xarray as xr
from ecodata.models.project_binding import ProjectFiles,bind_project
from ecodata.models.control_flow import ShellDeclarations
from ecodata.models.native_reference import reference_profile

def upload(name,text):
 b=BytesIO(text.encode() if isinstance(text,str) else text);b.name=name;return b

def analyze(text,extra=()):
 with ExitStack() as stack:
  files=ProjectFiles([upload('run.csh',text),*extra],stack)
  return ShellDeclarations(files,loop_limit=5).analyze('run.csh')

def test_dates_conditions_finite_loop_and_arithmetic():
 r=analyze('''set START = "2020-12-31"
set STOP = `date -ud "2021-01-01" +%Y%j`
set DAY = $START
set J = `date -ud "$DAY" +%Y%j`
set FIRST = TRUE
while ($J <= $STOP)
 if ($FIRST == TRUE) then
  setenv PORT cold
 else
  setenv PORT restart
 endif
 setenv CTM_STDATE $J
 mpirun $EXEC
 set FIRST = false
 set DAY = `date -ud "${DAY}+1days" +%Y-%m-%d`
 set J = `date -ud "$DAY" +%Y%j`
end
''')
 assert [(f['values']['CTM_STDATE'].text,f['values']['PORT'].text) for f in r['frames']]==[('2020366','cold'),('2021001','restart')]
 assert r['commands_executed'] is False

def test_unknown_branch_cannot_choose_one_input_and_commands_do_not_run(tmp_path):
 marker=tmp_path/'never-created'
 r=analyze(f'''if ($UNKNOWN == Y) then
 setenv MET_CRO_2D A
else
 setenv MET_CRO_2D B
endif
set X = `python -c "open('{marker}','w').write('bad')"`
mpirun $EXEC
''')
 assert r['frames'][0]['values']['MET_CRO_2D'].unknown
 assert r['frames'][0]['values']['X'].unknown and not marker.exists()
 assert not analyze('exit 1\nsetenv MET_CRO_2D bad\nmpirun $EXEC')['frames']

def test_source_is_parsed_without_host_environment_or_execution(monkeypatch):
 monkeypatch.setenv('HOST_SECRET','not-an-input')
 r=analyze('source ./settings.csh\nsetenv X $ROOT/file\nsetenv SECRET $HOST_SECRET',
  [upload('settings.csh','set ROOT = resources\n')])
 assert r['frames'][0]['values']['X'].text=='resources/file'
 assert r['frames'][0]['values']['SECRET'].unknown
 assert r['includes'][0]['status']=='PARSED_NOT_EXECUTED'

def test_unbounded_loop_and_generated_script_body_not_accepted():
 with pytest.raises(ValueError,match='无可验证进展|超限'):analyze('set A = 1\nwhile ($A == 1)\necho no\nend')
 r=analyze('cat >generated.csh <<EOF\nsetenv MET_CRO_2D pretend\nEOF\nset A = safe')
 assert 'MET_CRO_2D' not in r['frames'][0]['values']

def bundle(root='data',start='2021-01-01',end='2021-01-01'):
 text=f'''set FIRST = TRUE
set START = {start}
set END = {end}
set DAY = $START
set J = `date -ud "$DAY" +%Y%j`
set STOP = `date -ud "$END" +%Y%j`
while ($J <= $STOP)
 set DATE = `date -ud "$DAY" +%Y%m%d`
 setenv MET_CRO_2D $ROOT/{root}/met_$DATE.nc
 setenv GRID_CRO_2D $ROOT/{root}/grid.nc
 setenv CTM_STDATE $J
 mpirun $EXEC
 set DAY = `date -ud "${{DAY}}+1days" +%Y-%m-%d`
 set J = `date -ud "$DAY" +%Y%j`
end
'''
 return [upload('arbitrary-control.csh',text),upload(root+'/met_20210101.nc',b'actual payload'),upload(root+'/grid.nc',b'grid payload')]

def test_generic_paths_days_and_unavailable_control_filtering():
 for root in ['alpha','unrelated/nested']:
  inputs=bundle(root)
  b=bind_project(inputs,'cmaq')
  assert b['bindings'][0]['target']==root+'/met_20210101.nc'
  two=bundle(root,end='2021-01-02')[0];two.name='another-run.csh'
  b=bind_project([*inputs,two],'cmaq')
  assert b['control']['path']=='arbitrary-control.csh' and b['rejected_controls']
  assert not b['native_model_executed']

def test_ambiguous_science_configs_and_unknown_date_are_rejected():
 inputs=bundle();other=upload('second.csh',inputs[0].getvalue().replace(b'set FIRST = TRUE',b'set FIRST = FALSE'))
 with pytest.raises(ValueError,match='多个不同'):bind_project([*inputs,other],'cmaq')
 bad=upload('bad.csh',inputs[0].getvalue().replace(b'set START = 2021-01-01',b'set START = $UNDECLARED'))
 with pytest.raises(ValueError,match='未知|核心|绑定'):bind_project([bad,*inputs[1:]],'cmaq')

def test_source_profile_qa_and_scale_independent_of_filename_or_date():
 raw=np.array([[10,20],[30,255]],dtype='u1')
 ds=xr.Dataset(dict(Lai_500m=(('y','x'),np.where(raw<=100,raw*.1,np.nan),dict(units='1',source_scale_factor=.1)),
  Lai_500m_raw=(('y','x'),raw),FparLai_QC_raw=(('y','x'),np.array([[0,1],[0,0]],dtype='u1')),
  FparExtra_QC_raw=(('y','x'),np.zeros((2,2),dtype='u1'))),
  attrs=dict(product='MCD15A3H',time_coverage_start='2020-02-28T00:00Z',time_coverage_end='2020-03-03T00:00Z'))
 p=reference_profile(ds);assert p['valid'].tolist()==[[True,False],[True,False]]
 ds.Lai_500m.values[0,0]=99
 with pytest.raises(ValueError,match='尺度'):reference_profile(ds)

def test_native_big_endian_attribute_roundtrip(tmp_path,monkeypatch):
 from ecodata.models.engineering.native import extract_native_field,support_signature
 from ecodata.models import native_source_inspection
 from ecodata.interface.engineering_upload import LocalUpload
 monkeypatch.setattr(native_source_inspection,'native_container_evidence',lambda *args: {})
 ds=xr.Dataset({'LAI':(('TSTEP','LAY','ROW','COL'),np.ones((1,1,2,2)),dict(units='1')),
   'TFLAG':(('TSTEP','VAR','DATE-TIME'),np.array([[[2021001,0]]],dtype='i4'))},attrs=dict(VGLVLS=np.array([1.,.9975],dtype='>f4')))
 source=tmp_path/'source.nc';ds.to_netcdf(source,engine='scipy',format='NETCDF3_CLASSIC')
 product=extract_native_field([LocalUpload(source.name,source)],'cmaq',source.name,'LAI')
 target=tmp_path/'processed.nc';product.to_netcdf(target)
 with source.open('rb') as stream,xr.open_dataset(stream,decode_times=False) as a,xr.open_dataset(target,decode_times=False) as b:
  np.testing.assert_array_equal(a.attrs['VGLVLS'],b.attrs['VGLVLS'])
  assert support_signature(a,'LAI')==support_signature(b,'LAI')

def test_unknown_invocation_is_not_a_definite_runtime_frame():
 r=analyze('setenv PORT known\nif ($UNKNOWN == yes) then\n mpirun $EXEC\nendif\n')
 assert not r['frames']
