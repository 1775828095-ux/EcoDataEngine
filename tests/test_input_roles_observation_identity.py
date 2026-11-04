"""Public intake behavior tests; synthetic fixtures are not accuracy evidence."""
from io import BytesIO
import hashlib,json,zipfile
import pandas as pd
import pytest
from ecodata.models.native_meteorology import read_observations

def upload(name,raw):
 b=BytesIO(raw);b.name=name;return b

def noaa(value='+0200',time='2021-01-01T00:00:00',report='FM-12'):
 return ('STATION,DATE,TMP,REPORT_TYPE,LATITUDE,LONGITUDE,ELEVATION\n'
  f'A,{time},"{value},1",{report},35,-90,10\n').encode()

def test_raw_subset_decoded_aliases_and_timezone_order():
 decoded=b'station,time_utc,temperature_C,temperature_quality,report_type\nA,2020-12-31T19:00:00-05:00,20,1,FM-12\n'
 inputs=[upload('raw.csv',noaa()),upload('subset.csv',noaa()),upload('decoded.csv',decoded)]
 first,report=read_observations(inputs,times=pd.to_datetime(['2021-01-01']))
 second,other=read_observations(list(reversed(inputs)),times=pd.to_datetime(['2021-01-01']))
 assert len(first)==1 and first.value.iloc[0]==pytest.approx(293.15)
 assert report['reconciliation']['collapsed_rows']==2
 assert len(first.source_provenance.iloc[0])==3
 assert first.record_id.tolist()==second.record_id.tolist()
 assert report['reconciliation']==other['reconciliation']

def test_relevant_conflicts_rejected_but_unrelated_time_not_used():
 inputs=[upload('a.csv',noaa()),upload('b.csv',noaa('+0210'))]
 with pytest.raises(ValueError,match='冲突|重复'):
  read_observations(inputs,times=pd.to_datetime(['2021-01-01']))
 empty,report=read_observations(inputs,times=pd.to_datetime(['2021-01-02']))
 assert empty.empty and report['outside_time_support']==2
 with pytest.raises(ValueError,match='冲突|重复'):read_observations(inputs)

def test_equal_temperature_different_report_is_not_same_record():
 with pytest.raises(ValueError,match='冲突|重复'):
  read_observations([upload('a.csv',noaa()),upload('b.csv',noaa(report='FM-15'))])


def test_provider_identity_is_not_inferred_from_value_or_filename():
 from ecodata.models.source_catalog import reconcile_records
 frame=pd.DataFrame([dict(provider=p,station='A',time='T',value=v,source_sha256=p,source_member='same.csv',source_row=2)
                     for p,v in [('network-1',20.),('network-2',21.)]])
 result,report=reconcile_records(frame,['provider','station','time'],['value'])
 assert len(result)==2 and report['collapsed_rows']==0
 frame.loc[0,'provider']=None
 with pytest.raises(ValueError,match='未知'):reconcile_records(frame,['provider','station','time'],['value'])


def test_utc_support_storage_units_and_invalid_support():
 from numpy import array
 for unit in ['us','ns']:
  result,_=read_observations([upload('a.csv',noaa())],times=array(['2021-01-01'],dtype='datetime64['+unit+']'))
  assert len(result)==1
 with pytest.raises(ValueError,match='重复'):
  read_observations([upload('a.csv',noaa())],times=pd.to_datetime(['2021-01-01','2021-01-01']))

def software(reference=True,science=False,changed=False,hidden_science=False):
 commit='a'*40;folder='some-project-'+commit
 stream=BytesIO()
 with zipfile.ZipFile(stream,'w') as z:
  z.writestr(folder+'/README.md','source tree');z.writestr(folder+'/license.md','license')
  z.writestr(folder+'/src/math.f90','end')
  link=zipfile.ZipInfo(folder+'/src/alias.f90');link.create_system=3;link.external_attr=0o120777<<16
  z.writestr(link,'math.f90')
  if science:z.writestr(folder+'/run/input.nc',b'CDF scientific input fixture')
  if hidden_science:z.writestr(folder+'/run/renamed-field',b'CDF scientific input fixture')
 raw=stream.getvalue()
 manifest=[dict(file='old-filename.zip',url='https://codeload.github.com/owner/some-project/zip/'+commit,
  bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())]
 if changed:manifest[0]['sha256']='0'*64
 inputs=[upload('completely-renamed.zip',raw),upload('real.csv',b'time,value\n0,1\n')]
 if reference:inputs.append(upload('download-audit.json',json.dumps(manifest).encode()))
 return inputs

def test_source_archive_quarantine_is_content_and_checksum_based(tmp_path):
 from ecodata.models.source_catalog import partition_uploads
 inputs=software();active,audit=partition_uploads(inputs,tmp_path)
 assert 'completely-renamed.zip' not in [u.name for u in active]
 assert audit['references'][0]['role']=='software_reference'
 assert audit['references'][0]['symbolic_links']==1
 assert any(u.name=='real.csv' for u in active)
 assert audit['references'][0]['extracted'] is False

@pytest.mark.parametrize('kwargs',[dict(reference=False),dict(changed=True),dict(science=True),dict(hidden_science=True)])
def test_unproven_or_mixed_archives_keep_strict_preflight(tmp_path,kwargs):
 from ecodata.models.source_catalog import partition_uploads
 from ecodata.models.archive_preflight import inspect_upload_archives
 active,audit=partition_uploads(software(**kwargs),tmp_path)
 assert any(u.name.endswith('.zip') for u in active) and not audit['references']
 with pytest.raises(ValueError,match='不安全'):
  inspect_upload_archives(active,object(),tmp_path,native_required=False)

def test_partition_is_the_public_entry_boundary(tmp_path,monkeypatch):
 from ecodata.algorithms import automatic_preparation as public
 def downstream(uploads,model_id,folder,progress=None,task=None):
  assert all(not u.name.endswith('.zip') for u in uploads)
  out=folder/'result.zip'
  with zipfile.ZipFile(out,'w') as z:z.writestr('actual-product.txt','test dispatcher only')
  return dict(manifest=dict(status='BOUNDARY_TEST_ONLY'),archive=str(out))
 monkeypatch.setattr(public,'_prepare_uploads',downstream)
 result=public.prepare_uploads(software(),'any-model',tmp_path)
 with zipfile.ZipFile(result['archive']) as z:
  assert any('input_roles' in n for n in z.namelist())
 assert result['input_roles']['references']
