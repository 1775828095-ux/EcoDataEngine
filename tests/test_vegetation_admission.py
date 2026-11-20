"""Identify satellite metadata without silently making it hourly model forcing."""
from io import BytesIO
import json
import numpy as np
import xarray as xr
import pytest
from ecodata.models.engineering.precipitation_products import inspect_product
from ecodata.algorithms.automatic_preparation import prepare_uploads


def product():
    return xr.Dataset({'Lai_500m':(('y','x'),np.ones((2,2)),dict(long_name='Leaf area index',units='1')),
        'FparLai_QC_raw':(('y','x'),np.array([[0,1],[255,0]],dtype='uint8'))},
        attrs=dict(product='MCD15A3H',collection='UNKNOWN_NOT_RETURNED_BY_SERVICE',
            source_url='https://modisrest.ornl.gov/rst/api/v1/MCD15A3H/subset'))


def test_modis_is_identified_and_blocked_without_changing_values():
    ds=product();before=ds.copy(deep=True);report=inspect_product(ds)
    assert report['product']=='MODIS_MCD15A3H'
    codes={r['code'] for r in report['fields'][0]['blockers']}
    assert {'PRODUCT_VERSION_UNCONFIRMED','QUALITY_POLICY_UNCONFIRMED','TARGET_TIME_SUPPORT_UNCONFIRMED'}<=codes
    assert not report['quality_flags']['FparLai_QC_raw']['filter_applied']
    xr.testing.assert_identical(ds,before)


@pytest.mark.parametrize('url',['https://example.com/MCD15A3H','https://modisrest.ornl.gov:wrong/rst/api/v1/MCD15A3H/subset'])
def test_product_name_without_source_evidence_does_not_authorize(url):
    ds=product();ds.attrs['source_url']=url
    assert inspect_product(ds) is None


def test_public_input_returns_specific_required_information_without_fake_fusion(tmp_path):
    upload=BytesIO(product().to_netcdf(engine='h5netcdf'));upload.name='unfamiliar.nc'
    result=prepare_uploads([upload],'cmaq',tmp_path)
    manifest=result['manifest'];admission=manifest['preparation']['source_admission']
    assert manifest['status']=='ENGINEERING_INSPECTED' and not manifest['native_ready']
    assert admission['external_resources_identified']==1
    assert not admission['stages']['fusion_executed']
    assert 'TARGET_LAI_DEFINITION_UNCONFIRMED' in {r['code'] for r in admission['required_information']}


def test_precipitation_adapter_does_not_interpret_lai_as_precipitation(tmp_path):
    from ecodata.models.engineering.precipitation_adaptation import adapt_products
    source=dict(product='MODIS_MCD15A3H',source_id='modis',quantity='leaf_area_index')
    sources,reports=adapt_products([source],[],tmp_path/'adapt',[])
    assert sources==[source] and reports==[]


def test_existing_bound_reference_route_is_not_preempted(tmp_path,monkeypatch):
    from ecodata.models.engineering import preparation
    upload=BytesIO(product().to_netcdf(engine='h5netcdf'));upload.name='satellite.nc'
    calls=[]
    def binding(uploads,model):
        calls.append(model)
        return {'control':{'path':'run.csh'},'bindings':[]}
    monkeypatch.setattr(preparation,'bind_project',binding)
    assert preparation.prepare_engineering_inputs([upload],'cmaq',tmp_path) is None
    assert calls==['cmaq']
