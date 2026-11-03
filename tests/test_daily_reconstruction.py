"""Leakage, observed-value protection and real public-entry regression checks."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.algorithms.daily_reconstruction import DailyRecovery,gap_lengths
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.models import portrait_policy as policy


def context():
    time=pd.date_range('2002-01-01',periods=50)
    values=np.array([0.,10.,0.,20.,0.]*10);values[[4,5,18,19,20,21]]=np.nan
    ds=xr.Dataset({'precipitation':(('time','lat','lon'),values[:,None,None]),
        'time_bounds':(('time','bounds'),np.stack([time.values,(time+pd.Timedelta(days=1)).values],axis=1))},
        coords=dict(time=time,lat=[11.6],lon=[37.36]),attrs=dict(crs='EPSG:4326',source_id='station'))
    ds.precipitation.attrs.update(units='mm/day',physical_variable='precipitation_amount',
        spatial_semantics='point',temporal_semantics='accumulation',data_kind='continuous')
    state=xr.Dataset(dict(month_mean=('month',np.arange(12.)+1),scale=5.),coords=dict(month=np.arange(1,13)),attrs=dict(training_end='2001-12-31'))
    return AlgorithmContext((EcoDataBundle(ds),),'precipitation','continuous',(11.6,),(37.36,),parameters=dict(fitted_state=state))


@pytest.mark.parametrize('method',['linear','nearest','ffill','month_mean'])
def test_prior_only_and_observation_protection(method):
    ctx=context();ds=ctx.sources[0].dataset;visible=np.isfinite(ds.precipitation)
    result=DailyRecovery(method).run(ctx).data.dataset
    assert np.isfinite(result.precipitation).all()
    np.testing.assert_array_equal(result.precipitation.values[visible],ds.precipitation.values[visible])
    xr.testing.assert_identical(result.time_bounds,ds.time_bounds)


def test_partial_then_fallback_is_a_real_two_kernel_combination():
    ctx=context();first=DailyRecovery('linear').run(replace(ctx,parameters=dict(ctx.parameters,max_gap_days=2))).data.dataset
    assert np.isfinite(first.precipitation.values[4:6]).all()
    assert np.isnan(first.precipitation.values[18:22]).all()
    second=DailyRecovery('month_mean').run(replace(ctx,sources=(EcoDataBundle(first),))).data.dataset
    assert np.isfinite(second.precipitation).all()
    np.testing.assert_allclose(second.precipitation.values[18:22],1.)
    np.testing.assert_allclose(gap_lengths(ctx.sources[0].dataset.precipitation.values[:,0,0])[[4,18]],[2,4])


def test_future_state_and_irregular_clock_rejected():
    ctx=context();state=ctx.parameters['fitted_state'].copy();state.attrs['training_end']='2002-01-01'
    assert not DailyRecovery('month_mean').compatibility(replace(ctx,parameters=dict(fitted_state=state))).eligible
    ds=ctx.sources[0].dataset.isel(time=[0,2,3])
    assert not DailyRecovery('linear').compatibility(replace(ctx,sources=(EcoDataBundle(ds),))).eligible


def test_wrong_physical_meaning_and_unbracketed_linear_are_not_silently_fixed():
    ctx=context();ds=ctx.sources[0].dataset.copy(deep=True)
    ds.precipitation.attrs['units']='mm/hour'
    assert not DailyRecovery('linear').compatibility(replace(ctx,sources=(EcoDataBundle(ds),))).eligible
    ds.precipitation.attrs['units']='mm/day';ds.precipitation.values[-1]=np.nan
    actual=DailyRecovery('linear').run(replace(ctx,sources=(EcoDataBundle(ds),))).data.dataset
    assert np.isnan(actual.precipitation.values[-1]).all()


def test_public_three_portrait_service_executes_and_records_probes(tmp_path,monkeypatch):
    monkeypatch.setattr(ar,'ROOT',tmp_path/'attributes');monkeypatch.setattr(policy,'ROOT',tmp_path/'models')
    attrs=deepcopy(ar.load()['attributes'])
    for metric in ('missing_fraction','negative_fraction','expected_error'):
        attrs.append(dict(id='daily_'+metric,definition=metric,tables=['data','algorithm','model'],unit='fraction',
            role='scoring',scoring_enabled=True,hard_allowed=False,provider='state_metric',property=metric,
            activation='constant_one',evidence='Engineering acceptance fixture',history=['test']))
    api.publish_attribute_registry('daily-test',attrs,evidence='Fixture')
    issues=[dict(attribute='daily_'+m,metric=m,acceptable=0,bad=1,importance=1,evidence='Fixture input requirements') for m in ('missing_fraction','negative_fraction','expected_error')]
    api.publish_model_portrait('daily-test','swat_plus',[dict(id='daily',task='temporal',role='weather',field='precipitation',issues=issues,coupled_constraints=[],evidence='Fixture')],evidence='Fixture')
    ctx=context();ds=ctx.sources[0].dataset
    choices=[[dict(algorithm='reconstruction.daily_'+m,parameters={})] for m in ['linear','nearest','month_mean']]
    req=dict(task='temporal',variable='precipitation',unit='mm/day',semantic_type='continuous',nonnegative=True,
        spatial_semantics='point',temporal_semantics='accumulation',target_lat=[11.6],target_lon=[37.36],
        target_times=[str(pd.Timestamp(t)) for t in ds.time.values],parameters={},
        model_profile=dict(id='swat_plus',role='weather',field='precipitation'),
        scoring=dict(registry_version='daily-test'),portrait_policy=dict(version='daily-test',scenario='daily'),
        composition=dict(source_stages={'station':dict(plans=choices,evidence='Three declared methods')},post_stages=[],
            parallel_scoring=dict(weights=dict(data=.5,model=.5,collaboration=0),evidence='Frozen fixture priorities'),
            search=dict(backend='alns',survivors=1,generations=4,population=2)))
    missing=np.isnan(ds.precipitation);reference=xr.full_like(ds.precipitation,np.nan).where(~missing,7.)
    reference.attrs.update(units='mm/day',split='search_validation')
    inputs=[api.SearchInput('input.nc','primary',ds.to_netcdf()),api.SearchInput('state.nc','state',ctx.parameters['fitted_state'].to_netcdf())]
    result=api.submit_search(inputs,req,None,tmp_path/'run',mode='Profile',validation=api.SearchValidation(reference,'precipitation'),budget=dict(objective_limit=3,wall_seconds=60),seed=11)
    assert result['screening']['complete'] and result['parallel_scoring']['probe_graph_executions']==3
    assert result['counts']['failed_observed']==0
    assert result['native_optimizer']['package']=='alns'
    assert result['independent_test_used_for_selection'] is False
    assert result['parallel_scoring']['unique_score_intervals']>1
    assert all(row['trace'] and row['file_readback']=='PASS' for row in result['rows'])
