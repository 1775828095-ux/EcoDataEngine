"""Adaptation/feedback tests, not environmental accuracy evidence."""
from contextlib import ExitStack
import hashlib,json
from io import BytesIO
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zipfile
from ecodata.engines.units import UnitEngine
from ecodata.core.exceptions import SchemaError
from ecodata.models.engineering.precipitation_adaptation import adapt_products,acquisition_receipts,_longitude
from ecodata.models.engineering.preparation import source_fields
from ecodata.models.project_binding import ProjectFiles
from ecodata.algorithms.automatic_preparation import prepare_uploads
from test_unified_engineering_ports import upload,pack,request
from test_precipitation_product_admission import product


def test_rate_integral_uses_explicit_duration_and_keeps_missing():
    a=xr.DataArray([2.,np.nan],dims='time',coords={'time':pd.date_range('2020-01-01',periods=2)},attrs={'units':'mm/h'})
    out=UnitEngine().integrate_rate(a,[86400.,86400.])
    assert out.values[0]==48 and np.isnan(out.values[1]) and out.attrs['units']=='mm'
    assert a.attrs['units']=='mm/h' and a.values[0]==2
    with pytest.raises(SchemaError):UnitEngine().integrate_rate(a,[86400.,0.])


def test_regular_interval_reuses_xclim_when_available():
    pytest.importorskip('xclim')
    a=xr.DataArray([1.,2.,3.],dims='time',coords={'time':pd.date_range('2020-01-01',periods=3)},attrs={'units':'mm/h'})
    out=UnitEngine().integrate_rate(a,[86400.]*3)
    np.testing.assert_allclose(out,[24,48,72])
    assert json.loads(out.attrs['interval_integration'])['backend']=='xclim.core.units.rate2amount'


def test_longitude_conversion_carries_cells_and_bounds():
    d=xr.Dataset({'v':('lon',[4.,7.]),'edges':(('lon','nv'),[[249.5,250.5],[9.5,10.5]])},coords={'lon':[250.,10.]})
    d.lon.attrs.update(bounds='edges',units='degrees_east')
    out=_longitude(d)
    np.testing.assert_array_equal(out.lon,[-110.,10.])
    np.testing.assert_array_equal(out.v,[4.,7.])
    np.testing.assert_array_equal(out.edges,[[-110.5,-109.5],[9.5,10.5]])
    np.testing.assert_array_equal(d.lon,[250.,10.])


def test_product_shards_are_joined_not_fused(tmp_path):
    uploads=[]
    for i in range(3):
        d=product('cmorph');day=pd.Timestamp('2020-01-01')+pd.Timedelta(days=i)
        d=d.assign_coords(time=[day]);d.edges.values[:]+=np.timedelta64(i,'D')
        d.time.attrs['bounds']='edges'
        d.attrs.update(time_coverage_start=day.strftime('%Y-%m-%dT00:00:00Z'),time_coverage_end=day.strftime('%Y-%m-%dT23:59:59Z'))
        uploads.append(upload(f'arbitrary_{i}.nc',d.to_netcdf(engine='scipy')))
    with ExitStack() as stack:
        f=ProjectFiles(uploads,stack);audit=[];s=source_fields(f,{},audit)
        normal,reports=adapt_products(s,audit,tmp_path/'adapt',[])
    assert len(normal)==1 and len(reports)==1 and reports[0]['fusion_executed'] is False
    assert normal[0]['dataset'].sizes['time']==3
    assert normal[0]['dataset'].precipitation.attrs['units']=='mm'
    assert len(normal[0]['members'])==3


def test_sha_bound_receipt_resolves_power_time_without_modifying_data(tmp_path):
    d=product('power');d.time.attrs['delta']='0000-00-01 00:00:00'
    raw=d.to_netcdf(engine='scipy');digest=hashlib.sha256(raw).hexdigest()
    url='https://power.larc.nasa.gov/api/temporal/daily/regional?parameters=arbitrary&start=20200101&end=20200101&time-standard=UTC'
    receipts=[upload('any_receipt.json',json.dumps(dict(sha256=digest,url=url)).encode())]
    with ExitStack() as stack:
        files=ProjectFiles([upload('unknown.nc',raw),*receipts],stack);audit=[]
        s=source_fields(files,{},audit);normal,reports=adapt_products(s,audit,tmp_path/'adapt',acquisition_receipts(files))
    assert normal[0]['dataset'].attrs['time_basis']=='UTC'
    assert not audit[0]['blockers']
    assert 'time_basis' not in d.attrs


def test_forged_receipt_cannot_resolve_missing_basis(tmp_path):
    d=product('power');d.time.attrs['delta']='0000-00-01 00:00:00'
    raw=d.to_netcdf(engine='scipy')
    with ExitStack() as stack:
        files=ProjectFiles([upload('data.nc',raw),upload('r.json',json.dumps(dict(sha256='not-this-file',url='https://power.larc.nasa.gov/api/temporal/daily/regional?time-standard=UTC')).encode())],stack)
        audit=[];normal,_=adapt_products(source_fields(files,{},audit),audit,tmp_path/'adapt',acquisition_receipts(files))
    assert normal[0]['dataset'].attrs['time_basis']=='UNKNOWN'
    assert audit[0]['blockers'][0]['code']=='SOURCE_DAY_UNCONFIRMED'


def test_public_feedback_and_actual_adaptation_download(tmp_path):
    files,_,_,_,_=pack('swat_plus','precipitation','mm','accumulation')
    files['product.nc']=product('cmorph').to_netcdf(engine='scipy')
    blob=BytesIO()
    with zipfile.ZipFile(blob,'w') as z:
        for n,v in files.items():z.writestr(n,v)
    result=prepare_uploads([upload('bundle.zip',blob.getvalue())],'swat_plus',tmp_path/'run')
    assessment=result['manifest']['preparation']['source_admission']
    assert {'MODEL_DAY_UNCONFIRMED'}<={r['code'] for r in assessment['required_information']}
    assert len(assessment['adaptation'])==1
    with zipfile.ZipFile(result['archive']) as z:
        assert 'required_information.json' in z.namelist()
        assert 'adaptation/product_0.nc' in z.namelist()
        with xr.open_dataset(BytesIO(z.read('adaptation/product_0.nc'))) as ds:
            assert ds.precipitation.attrs['units']=='mm'
    assert result['manifest']['native_writeback_verified'] is False


def test_native_day_confirmation_consumed_without_guessing():
    from ecodata.models.engineering.ports import weather_ports
    files,name,_,_,_=pack('swat_plus','precipitation','mm','accumulation')
    with ExitStack() as stack:
        f=ProjectFiles([upload(n,v) for n,v in files.items()],stack)
        original=weather_ports(f)[0][0]
        confirmed=weather_ports(f,{name:dict(variables={'precipitation':{'time_basis':'UTC'}})})[0][0]
    assert original.data.attrs['time_basis']=='model_calendar_day'
    assert confirmed.data.attrs['time_basis']=='UTC' and confirmed.unit=='mm'


def test_same_lst_abbreviation_does_not_prove_same_solar_day(tmp_path):
    files,name,_,_,_=pack('swat_plus','precipitation','mm','accumulation')
    d=product('power');d.time.attrs['delta']='0000-00-01 00:00:00';d.arbitrary.attrs['time_basis']='LST'
    files['source.nc']=d.to_netcdf(engine='scipy')
    files['semantics.json']=json.dumps(dict(schema='ecodata.engineering-semantics.v1',model_id='swat_plus',resources=[
        dict(resource=name,role='model_input',confirmed=True,variables={'precipitation':dict(time_basis='LST')})])).encode()
    result=prepare_uploads([upload(n,v) for n,v in files.items()],'swat_plus',tmp_path)
    assert result['manifest']['status']=='ENGINEERING_INSPECTED'
    codes={x['code'] for x in result['manifest']['preparation']['source_admission']['required_information']}
    assert 'LOCAL_DAY_SUPPORT_UNCONFIRMED' in codes


def test_existing_verified_fusion_still_executes_and_writes(tmp_path):
    r=prepare_uploads([request('swat_plus',multi=True)],'swat_plus',tmp_path)
    assert r['manifest']['status']=='ENGINEERING_INSPECTED'
    assert r['manifest']['preparation']['source_admission']['required_information'][0]['code']=='VALIDATION_EVIDENCE_INSUFFICIENT'


def test_two_evidenced_product_sources_fuse_and_partially_write_native_project(tmp_path):
    """Synthetic complete contracts; prove real kernels/writer, not accuracy."""
    files,name,_,_,_=pack('swat_plus','precipitation','mm','accumulation')
    for label,value in [('a',2.),('b',6.)]:
        ds=product('cmorph');ds.attrs['product_version']=label
        ds.arbitrary.values[:]=value
        ds['sigma']=xr.ones_like(ds.arbitrary)
        ds.sigma.attrs.update(units='mm/day',uncertainty_type='standard_deviation')
        ds.arbitrary.attrs['ancillary_variables']='sigma'
        files[label+'.nc']=ds.to_netcdf(engine='scipy')
    files['semantics.json']=json.dumps(dict(schema='ecodata.engineering-semantics.v1',model_id='swat_plus',resources=[
        dict(resource=name,role='model_input',confirmed=True,variables={'precipitation':dict(time_basis='UTC')})])).encode()
    blob=BytesIO()
    with zipfile.ZipFile(blob,'w') as z:
        for n,v in files.items():z.writestr(n,v)
    result=prepare_uploads([upload('neutral.zip',blob.getvalue())],'swat_plus',tmp_path/'中文输出')
    assert result['manifest']['status']=='ENGINEERING_INSPECTED'
    assert not result['manifest']['native_writeback_verified']
    codes={r['code'] for r in result['manifest']['preparation']['source_admission']['required_information']}
    assert codes=={'VALIDATION_EVIDENCE_INSUFFICIENT'}
