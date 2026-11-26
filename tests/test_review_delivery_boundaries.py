"""Functional status/delivery checks, not environmental accuracy evidence."""
import json
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ecodata.algorithms import platform_search as api


def test_incomplete_search_cannot_recommend_even_when_measured_error_improves(monkeypatch,tmp_path):
    clock=pd.date_range('2000-01-01',periods=12)
    source=xr.Dataset(dict(t=(('time','lat','lon'),np.full((12,1,1),10.))),
        coords=dict(time=clock,lat=[30.],lon=[110.]),attrs=dict(crs='EPSG:4326'))
    source.t.attrs=dict(units='K',physical_variable='air_temperature',spatial_semantics='point',temporal_semantics='instantaneous')
    truth=xr.zeros_like(source.t)
    label=truth.isel(time=slice(0,6)).copy(deep=True);label.attrs['split']='search_validation'
    held=truth.isel(time=slice(6,12))
    folder=tmp_path/'run'

    def backend(inputs,request,evaluation,folder,**options):
        folder.mkdir()
        product=source.copy(deep=True);product.t.values[:]=0
        path=folder/'product.nc';product.to_netcdf(path)
        (folder/'selection_freeze.json').write_text(json.dumps(dict(key='candidate',source_baseline_key='s0')))
        return dict(status='MUTATION_NOT_EXECUTED',product=str(path),
            best=dict(key='candidate',metrics=options['validation'](product)),
            source_baselines=[dict(key='s0',status='ELIGIBLE_OUTPUT',metrics=options['validation'](source))],
            independent_evaluation=evaluation(product))

    monkeypatch.setattr(api,'_submit_search',backend)
    result=api.submit_search([api.SearchInput('source.nc','primary',source.to_netcdf())],
        dict(task='temporal',variable='t',unit='K',semantic_type='continuous'),
        api.DatasetReference(held,'t',role='independent_validation'),folder,
        validation=api.SearchValidation(label,'t'))
    gate=result['recommendation_review']
    assert gate['status']=='SEARCH_INCOMPLETE' and not gate['recommended']
    assert gate['search_status']=='MUTATION_NOT_EXECUTED'
    assert result['delivery_status']=='BASELINE_RETAINED'
    with xr.open_dataset(result['product']) as delivered:np.testing.assert_array_equal(delivered.t,source.t)
    with xr.open_dataset(result['candidate_product']) as candidate:assert float(candidate.t.max())==0


def interval_pairs():
    clock=pd.date_range('2000-01-01',periods=24)
    return pd.DataFrame(dict(station='functional_fixture',grid_time=clock,background=1.+np.arange(24),
        observed=3.+np.arange(24),longitude=110.,latitude=30.,unit='mm',temporal_semantics='accumulation',
        interval_start=clock-pd.Timedelta(days=1),interval_end=clock))


def test_explicit_interval_contract_preserves_bounds_and_rejects_ambiguity():
    from ecodata.algorithms.paired_station_search import prepare
    frame=interval_pairs()
    outputs=[]
    for storage in ['us','ns']:
        current=frame.copy()
        for col in ['grid_time','interval_start','interval_end']:current[col]=current[col].astype('datetime64['+storage+']')
        data,state,validation,held,protocol=prepare(current,'precipitation_amount','mm',temporal_semantics='accumulation')
        assert data.precipitation_amount.attrs['temporal_semantics']=='accumulation'
        assert (data.time_bounds.values[:,1]-data.time_bounds.values[:,0]==np.timedelta64(1,'D')).all()
        assert protocol['intervals_verified'] and validation.attrs['temporal_semantics']=='accumulation'
        outputs.append(data)
    xr.testing.assert_allclose(*outputs)
    with pytest.raises(ValueError,match='explicit interval'):prepare(frame.drop(columns=['interval_start']),'p','mm',temporal_semantics='accumulation')
    with pytest.raises(ValueError,match='semantics conflict'):prepare(frame,'p','mm')
    with pytest.raises(ValueError,match='supported paired'):prepare(frame,'p','mm',temporal_semantics='UNKNOWN')
    frame.loc[5,'interval_start']-=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='equal positive non-overlapping'):prepare(frame,'p','mm',temporal_semantics='accumulation')


def test_public_accumulation_calibration_executes_without_instantaneous_relabel(tmp_path):
    result=api.submit_search([api.SearchInput('functional.csv','paired_observations',interval_pairs().to_csv(index=False).encode())],
        dict(task='temporal',adapter='paired_station_calibration',variable='precipitation_amount',unit='mm',
            temporal_semantics='accumulation',nonnegative=True),None,tmp_path/'run',
        budget=dict(objective_limit=80,total_call_limit=120,proposal_limit=256,wall_seconds=30),seed=11)
    station=result['stations'][0]
    assert result['status']=='PASS',result
    assert station['recommendation_review']['recommended']
    assert station['independent_evaluation']['RMSE']==0
    with xr.open_dataset(station['product']) as ds:
        assert ds.precipitation_amount.attrs['temporal_semantics']=='accumulation'
        assert 'time_bounds' in ds and np.isfinite(ds.precipitation_amount).all()
