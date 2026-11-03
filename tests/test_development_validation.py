import hashlib
import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms import development_validation as module
from ecodata.algorithms.attribute_registry import digest
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.platform_search import parse_inputs
from ecodata.core.schemas import EcoDataBundle


def dataset(year,count=13,step=12,sensor='landsat-8',source='historical_fine',n=3):
    times=pd.date_range(f'{year}-04-01',periods=count,freq=f'{step}D')
    ds=xr.Dataset(dict(ndvi=(('time','lat','lon'),np.full((count,n,n),.3))),
        coords=dict(time=times,lat=40+np.arange(n)*.001,lon=-92+np.arange(n)*.001),
        attrs=dict(sensor=sensor,source_id=source,crs='EPSG:4326',evidence_role='development'))
    ds.ndvi.attrs=dict(units='1',physical_variable='ndvi',physical_definition='fixture red/NIR NDVI',
        semantic_type='continuous',spatial_semantics='cell_mean',temporal_semantics='representative_observation')
    return ds


def current():
    fine=dataset(2024,count=10,step=16,source='current_fine')
    coarse=dataset(2024,count=20,step=8,sensor='terra-modis',source='current_coarse',n=2)
    visible=fine.isel(time=[0,2,4,6,9])
    context=AlgorithmContext((EcoDataBundle(visible),EcoDataBundle(coarse)),'ndvi','continuous',
        tuple(fine.lat.values),tuple(fine.lon.values),parameters={},model_constraints=dict(unit='1'))
    request=dict(task='multisource',variable='ndvi',unit='1',protect_observed_source='current_fine',
        target_times=[str(t) for t in fine.time.values],target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),parameters={})
    return request,context


def publish(tmp_path,years=(2020,2021),count=13,mutate=None):
    cases=[]
    for year in years:
        sources=[]
        for role,sensor,n in [('primary','landsat-8',3),('background','terra-modis',2)]:
            ds=dataset(year,count=count,source=f'{role}_{year}',sensor=sensor,n=n)
            if mutate is not None:mutate(ds,role,year)
            path=tmp_path/f'{role}_{year}.nc';ds.to_netcdf(path,engine='h5netcdf')
            sources.append(dict(role=role,path=str(path),sensor=sensor,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        cases.append(dict(id='case'+str(year),year=year,sources=sources))
    registry=dict(evidence_role='development',evidence='Generated unit fixtures, no external data',cases=cases)
    path=tmp_path/'published.json'
    write_registry(path,registry)
    return path


def write_registry(path,registry):
    path.write_text(json.dumps(dict(schema_version=1,registry=registry,sha256=digest(registry))),encoding='utf-8')


def test_historical_folds_match_anchor_hidden_counts_without_label_in_inputs(tmp_path):
    path=publish(tmp_path);q,ctx=current()
    folds,receipt=module.history_folds(q,ctx,{},path)
    assert receipt['status']=='MATCHED_DEVELOPMENT_SIMULATIONS' and len(folds)==2
    for fold in folds:
        parsed,_=parse_inputs(fold['inputs']);fine=parsed[0][1];coarse=parsed[1][1]
        reference=fold['validation'].reference;protocol=fold['protocol']
        assert fine.sizes['time']==reference.sizes['time']==5
        assert coarse.sizes['time']==13
        assert not set(fine.time.values)&set(reference.time.values)
        assert len(fold['request']['target_times'])==10
        assert protocol['retained_dates']==protocol['hidden_dates']==5
        assert protocol['visible_indices']==[0,2,4,6,9] and protocol['hidden_indices']==[1,3,5,7,8]
        assert reference.attrs['split']=='search_validation' and fold['confirmation'] is None
        assert fold['request']['protect_observed_source']==fine.attrs['source_id']
        assert list(fine.lat.values)==fold['request']['target_lat']
        assert len(fold['id'])==64 and protocol['selection_uses_values'] and not protocol['selection_uses_hidden_fine_values'] and not protocol['external_test_used']
    repeat,_=module.history_folds(q,ctx,{},path)
    assert [f['id'] for f in folds]==[f['id'] for f in repeat]


def test_changing_only_hidden_truth_does_not_change_chosen_dates_or_numeric_inputs(tmp_path):
    path=publish(tmp_path,years=(2020,));q,ctx=current()
    before,_=module.history_folds(q,ctx,{},path);a=before[0]
    envelope=json.loads(path.read_text(encoding='utf-8'));item=envelope['registry']['cases'][0]['sources'][0]
    source=Path(item['path'])
    with xr.open_dataset(source) as ds:full=ds.load()
    full.ndvi.loc[dict(time=a['validation'].reference.time)]=.7
    full.to_netcdf(source,engine='h5netcdf');item['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    write_registry(path,envelope['registry'])
    after,_=module.history_folds(q,ctx,{},path);b=after[0]
    assert a['protocol']['target_times']==b['protocol']['target_times']
    assert a['protocol']['input_sha256']==b['protocol']['input_sha256']
    assert not np.array_equal(a['validation'].reference.values,b['validation'].reference.values)


def test_absent_registry_is_explicit_unknown(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'DEFAULT_REGISTRY',tmp_path/'not_published.json');q,ctx=current()
    folds,receipt=module.history_folds(q,ctx,{})
    assert folds==[] and receipt['status']=='UNKNOWN'


def test_same_year_is_skipped_before_reading_any_reference_file(tmp_path):
    q,ctx=current();path=tmp_path/'published.json'
    registry=dict(evidence_role='development',evidence='Fixture same-year guard',cases=[dict(id='same_year',year=2024,
        sources=[dict(role=role,path='must_not_be_opened.nc',sensor=sensor,sha256='0'*64)
                 for role,sensor in [('primary','landsat-8'),('background','terra-modis')]])])
    write_registry(path,registry)
    folds,receipt=module.history_folds(q,ctx,{},path)
    assert not folds and 'Current target year excluded' in receipt['cases'][0]['reason']


@pytest.mark.parametrize('kind',['sensor','units','physical_definition'])
def test_unmatched_physical_or_sensor_context_remains_unknown(tmp_path,kind):
    def mutate(ds,role,year):
        if kind=='units':ds.ndvi.attrs['units']='different unit'
        if kind=='physical_definition':ds.ndvi.attrs['physical_definition']='different physical field'
    path=publish(tmp_path,mutate=mutate);q,ctx=current()
    if kind=='sensor':ctx.sources[0].dataset.attrs['sensor']='landsat-9'
    folds,receipt=module.history_folds(q,ctx,{},path)
    assert not folds and receipt['status']=='UNKNOWN'


@pytest.mark.parametrize('kind',['file_role','field_split'])
def test_test_resources_are_rejected_even_with_valid_file_hash(tmp_path,kind):
    def mutate(ds,role,year):
        if role=='primary':
            if kind=='file_role':ds.attrs['evidence_role']='test'
            else:ds.ndvi.attrs['split']='test'
    path=publish(tmp_path,mutate=mutate);q,ctx=current()
    with pytest.raises(ValueError,match='test|Test'):
        module.history_folds(q,ctx,{},path)


@pytest.mark.parametrize('kind',['registry','resource'])
def test_sha_integrity_is_required(tmp_path,kind):
    path=publish(tmp_path);q,ctx=current();envelope=json.loads(path.read_text(encoding='utf-8'))
    if kind=='registry':
        envelope['sha256']='0'*64;path.write_text(json.dumps(envelope),encoding='utf-8')
    else:
        envelope['registry']['cases'][0]['sources'][0]['sha256']='0'*64
        write_registry(path,envelope['registry'])
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        module.history_folds(q,ctx,{},path)


def test_insufficient_history_is_not_fabricated_into_a_matching_case(tmp_path):
    path=publish(tmp_path,count=8);q,ctx=current()
    folds,receipt=module.history_folds(q,ctx,{},path)
    assert not folds and all('Insufficient historical dates' in r['reason'] for r in receipt['cases'])


def test_natural_missing_values_do_not_become_truth(tmp_path):
    def mutate(ds,role,year):
        if role=='primary':ds.ndvi.values[:,0,0]=np.nan
    path=publish(tmp_path,mutate=mutate);q,ctx=current()
    folds,_=module.history_folds(q,ctx,{},path)
    assert len(folds)==2
    assert all(np.isnan(f['validation'].reference.values[:,0,0]).all() for f in folds)


def test_current_fitted_state_is_not_reused_for_historical_folds(tmp_path):
    path=publish(tmp_path);q,ctx=current();ctx=replace(ctx,parameters=dict(fitted_state=xr.Dataset()))
    folds,receipt=module.history_folds(q,ctx,{},path)
    assert not folds and 'independent historical-fold provenance' in receipt['reason']


def test_calendar_assignment_is_monotonic_and_ties_choose_earlier():
    result=module._ordered_dates(np.array([10,20]),np.array([9,11,19,21]))
    np.testing.assert_array_equal(result,[0,2])


from pathlib import Path
