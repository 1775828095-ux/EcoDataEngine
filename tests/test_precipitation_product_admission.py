"""Synthetic mechanism tests; no product authenticity or accuracy claims."""
from contextlib import ExitStack
from io import BytesIO
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.models.project_binding import ProjectFiles
from ecodata.models.engineering.preparation import source_fields,write_ports
from ecodata.models.engineering.ports import weather_ports
from ecodata.models.swat_weather_project import _records
from test_unified_engineering_ports import pack,upload


def product(family):
    attrs=dict(units='mm/day')
    global_attrs={}
    if family=='power':
        attrs.update(standard_name='Precipitation_Corrected',definition='MERRA-2 bias corrected total precipitation')
        global_attrs=dict(source='Prediction Of Worldwide Energy Resource (POWER)',institution='NASA Langley Research Center (LaRC)')
    else:
        attrs.update(standard_name='lwe_precipitation_rate')
        global_attrs=dict(title='NOAA CMORPH precipitation',naming_authority='gov.noaa.ncei',institution='DOC/NOAA',
            time_coverage_start='2020-01-01T00:00:00Z',time_coverage_end='2020-01-01T23:59:59Z',time_coverage_resolution='P1D')
    ds=xr.Dataset({'arbitrary':(('time','lat','lon'),np.ones((1,2,2)),attrs)},
        coords=dict(time=pd.date_range('2020-01-01',periods=1),lat=[0.,1.],lon=[0.,1.]),attrs=global_attrs)
    ds.lat.attrs['units']='degrees_north';ds.lon.attrs['units']='degrees_east'
    if family=='cmorph':
        ds['edges']=(('time','nv'),np.array([['2020-01-01T00:00:00','2020-01-01T23:59:59']],dtype='datetime64[ns]'))
        ds.time.attrs['bounds']='edges'
        ds.time.encoding['units']='seconds since 1970-01-01'
        ds.edges.encoding['units']='seconds since 1970-01-01'
    return ds


@pytest.mark.parametrize('family',['power','cmorph'])
def test_product_identity_from_metadata_not_filename(family):
    ds=product(family);audit=[]
    with ExitStack() as stack:
        files=ProjectFiles([upload('irrelevant-name.nc',ds.to_netcdf(engine='scipy'))],stack)
        sources=source_fields(files,{},audit)
    assert len(sources)==1 and sources[0]['quantity']=='precipitation'
    assert audit[0]['status']=='IDENTIFIED_PRODUCT' and audit[0]['used'] is False
    if family=='power':assert any(x['code']=='RATE_ACCUMULATION_CONTRACT_REQUIRED' for x in sources[0]['blockers'])
    else:assert audit[0]['fields'][0]['interval_seconds']==86400
    assert 'source_id' not in ds.attrs and 'temporal_semantics' not in ds.arbitrary.attrs
    assert ds.lon.values.tolist()==[0.,1.]


def test_filename_cannot_manufacture_product_identity():
    ds=product('cmorph');ds.attrs['institution']='unknown'
    audit=[]
    with ExitStack() as stack:
        files=ProjectFiles([upload('CMORPH_OFFICIAL.nc',ds.to_netcdf(engine='scipy'))],stack)
        assert not source_fields(files,{},audit)
    assert audit[0]['status']=='UNRECOGNIZED_SOURCE'


@pytest.mark.parametrize('defect,code',[
    ('unit','PRODUCT_UNIT_UNSUPPORTED'),('bounds','TIME_SUPPORT_UNCONFIRMED'),
    ('negative','INVALID_PRECIPITATION'),('qa','QA_RULE_UNCONFIRMED')])
def test_identified_product_still_requires_scientific_contract(defect,code):
    from ecodata.models.engineering.precipitation_products import inspect_product
    ds=product('cmorph')
    if defect=='unit':ds.arbitrary.attrs['units']='kg'
    elif defect=='bounds':ds.edges.values[0,1]=np.datetime64('2020-01-01T12:00:00')
    elif defect=='negative':ds.arbitrary.values[:]=-1
    else:
        ds['quality']=xr.zeros_like(ds.arbitrary)
        ds.quality.attrs['flag_values']=[0,1]
        ds.arbitrary.attrs['ancillary_variables']='quality'
    report=inspect_product(ds)
    assert code in {b['code'] for b in report['fields'][0]['blockers']}


def test_day_shards_are_not_distinct_sources():
    ds=product('cmorph')
    with ExitStack() as stack:
        files=ProjectFiles([upload(n,ds.to_netcdf(engine='scipy')) for n in ['one.nc','two.nc']],stack)
        sources=source_fields(files,{})
    assert len(sources)==2 and len({s['source_id'] for s in sources})==1


@pytest.mark.parametrize('family',['power','cmorph'])
def test_public_route_reports_every_source_without_inventing_admission(tmp_path,family):
    files,_,_,_,_=pack('swat_plus','precipitation','mm/day','accumulation')
    files['unknown.nc']=xr.Dataset({'x':('t',[1.])}).to_netcdf(engine='scipy')
    files['renamed.nc']=product(family).to_netcdf(engine='scipy')
    blob=BytesIO()
    with zipfile.ZipFile(blob,'w') as z:
        for n,v in files.items():z.writestr(n,v)
    result=prepare_uploads([upload('project.zip',blob.getvalue())],'swat_plus',tmp_path)
    m=result['manifest'];a=m['preparation']['source_admission']
    assert m['status']=='ENGINEERING_INSPECTED' and not m['native_model_executed']
    assert a['external_resources_identified']==1 and len(a['sources'])==2
    identified=next(s for s in a['sources'] if s['status']=='IDENTIFIED_PRODUCT')
    assert 'MODEL_DAY_UNCONFIRMED' in {r['code'] for r in identified['blockers']}
    if family=='power':assert 'RATE_ACCUMULATION_CONTRACT_REQUIRED' in {r['code'] for r in identified['blockers']}
    with zipfile.ZipFile(result['archive']) as z:
        assert z.testzip() is None
        assert {'source_admission.json','processing_status.txt','handoff.json'}<=set(z.namelist())
        actual=json.loads(z.read('source_admission.json'))
        assert actual['stages']==dict(engineering_inspected=True,fusion_executed=False,native_writeback=False,native_model_executed=False)


def weather_bytes(kind,values):
    return ('title\nNBYR TSTEP LAT LON ELEV\n1 0 0 0 1\n2020 1 '+values+'\n').encode()


@pytest.mark.parametrize('flag',[-97,-99,-100])
def test_native_missing_is_opt_in(flag):
    raw=weather_bytes('pcp',str(flag))
    with pytest.raises(ValueError,match='缺测'):_records(raw,'pcp')
    assert len(_records(raw,'pcp',allow_native_missing=True))==1
    with pytest.raises(ValueError,match='负值'):_records(weather_bytes('pcp','-1'),'pcp',allow_native_missing=True)


def test_temperature_missing_not_compared_as_real_minimum():
    raw=weather_bytes('tmp','-99 5')
    assert len(_records(raw,'tmp',allow_native_missing=True))==1
    with pytest.raises(ValueError,match='缺测'):_records(raw,'tmp')
    assert len(_records(weather_bytes('tmp','-2 -5'),'tmp'))==1


def test_partial_writeback_preserves_missing_dates_and_nontarget_bytes(tmp_path):
    files,name,_,_,_=pack('swat_plus','precipitation','mm/day','accumulation')
    files[name]=files[name].replace(b'2020 1 1',b'2020 1 -99')
    blob=BytesIO()
    with zipfile.ZipFile(blob,'w') as z:
        for n,v in files.items():z.writestr(n,v)
    with ExitStack() as stack:
        project=ProjectFiles([upload('project.zip',blob.getvalue())],stack)
        ports,issues=weather_ports(project)
        p=next(p for p in ports if p.quantity=='precipitation')
        assert np.isnan(p.data.precipitation.values[0,0]) and p.binding['native_missing_cells']==1
        assert any(i['code']=='WGN_DEPENDENCY_CHECK' for i in issues)
        result=write_ports(project,'swat_plus',[(p,np.array([[np.nan],[5.]]))],[],tmp_path)
    with zipfile.ZipFile(result['archive']) as z:
        updated=z.read(name)
        assert b'2020 1 -99\n' in updated and b'2020 2 5\n' in updated
        assert _records(updated,'pcp',allow_native_missing=True)==_records(files[name],'pcp',allow_native_missing=True)
        for n,v in files.items():
            if n!=name:assert z.read(n)==v


def test_writer_rejects_new_missing_flag(tmp_path):
    files,name,_,_,_=pack('swat_plus','precipitation','mm/day','accumulation')
    with ExitStack() as stack:
        project=ProjectFiles([upload(n,v) for n,v in files.items()],stack)
        p=next(p for p in weather_ports(project)[0] if p.quantity=='precipitation')
        with pytest.raises(ValueError,match='新的原生缺测'):
            write_ports(project,'swat_plus',[(p,np.full((2,1),-99.))],[],tmp_path)
