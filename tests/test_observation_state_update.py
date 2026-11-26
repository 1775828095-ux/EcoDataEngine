"""Functional access/state binding tests, not environmental accuracy evidence."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dataclasses import replace
from ecodata.fusion.observation_state import update_state
from ecodata.algorithms.quality_aware import fit_ratio_frames
from ecodata.algorithms.platform_search import SearchInput,submit_search


def fixture():
    clock=pd.date_range('2000-01-01',periods=24)
    frame=pd.DataFrame(dict(station='TARGET',grid_time=clock,background=11.+np.arange(24),
        observed=13.+np.arange(24),longitude=110.,latitude=30.,unit='K'))
    from ecodata.algorithms.paired_station_search import prepare
    source,prior,_,_,_=prepare(frame,'air_temperature','K')
    source.attrs['background_product']='declared_product'
    parts=[]
    for sid,x in [('A',109.),('B',111.)]:
        parts.append(pd.DataFrame(dict(station_id=sid,time=clock,value=15.+np.arange(24),background=11.+np.arange(24),
            unit='K',background_unit='K',background_product='declared_product',variable='air_temperature',
            longitude=x,latitude=30.,split='train',qc_flag='VALID',temporal_semantics='instantaneous',crs='EPSG:4326')))
    return source,prior,pd.concat(parts,ignore_index=True)


POLICY=dict(mode='contemporaneous_analysis',evidence='Explicit synthetic other-location access')


def test_epoch_update_uses_only_legal_same_period_and_missing_falls_back():
    source,prior,obs=fixture()
    state,report=update_state(prior,source,obs,'air_temperature',POLICY)
    np.testing.assert_allclose(state.affine_intercept,4.)
    assert report['residual_updated_epochs']==12 and not report['reference_used']
    changed=obs.copy();changed.loc[changed.time==source.time.values[-1],'value']+=20
    other,_=update_state(prior,source,changed,'air_temperature',POLICY)
    np.testing.assert_array_equal(state.affine_intercept[:-1],other.affine_intercept[:-1])
    np.testing.assert_allclose(other.affine_intercept[-1],24.,rtol=0,atol=1e-12)
    missing=obs.copy();missing.loc[missing.time==source.time.values[0],'qc_flag']='MISSING'
    fallback,receipt=update_state(prior,source,missing,'air_temperature',POLICY)
    assert float(fallback.affine_intercept[0])==float(prior.affine_intercept)
    assert receipt['prior_fallback_epochs']==1


@pytest.mark.parametrize('change,error',[
    ('target','Held target'),('alias','target-location alias'),('split','validation/test'),
    ('support','point support'),('unit','unit conflict'),('product','background product'),('duplicate','duplicate'),('crs','CRS conflict')])
def test_conditioning_access_rejects_incompatible_resources(change,error):
    source,prior,obs=fixture()
    if change=='support':source.air_temperature.attrs['spatial_semantics']='cell_mean'
    elif change=='target':obs.loc[0,'station_id']='TARGET'
    elif change=='alias':obs.loc[0,['longitude','latitude']]=[110.,30.]
    elif change=='split':obs.loc[0,'split']='test'
    elif change=='unit':obs.loc[0,'background_unit']='mm'
    elif change=='product':obs.loc[0,'background_product']='another_product'
    elif change=='duplicate':obs=pd.concat([obs,obs.iloc[[0]]],ignore_index=True)
    elif change=='crs':obs.loc[0,'crs']='EPSG:3857'
    with pytest.raises(ValueError,match=error):update_state(prior,source,obs,'air_temperature',POLICY)


def test_ratio_kernel_is_identical_to_existing_spatial_formula():
    x=np.array([[1.,2.,np.nan],[0.,0.,0.],[1.,np.nan,np.nan]])
    y=np.array([[2.,6.,9.],[0.,0.,0.],[2.,3.,4.]])
    valid=np.isfinite(x)&np.isfinite(y);den=np.where(valid,x,0).sum(axis=1,keepdims=True)
    num=np.where(valid,y,0).sum(axis=1,keepdims=True);okay=(valid.sum(axis=1,keepdims=True)>=2)&(den>0)
    expected=np.divide(num,den,out=np.ones_like(num),where=okay)
    actual,mask=fit_ratio_frames(x,y)
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(mask,okay)


def test_public_search_has_actual_observation_state_parent_and_truth_isolation(tmp_path):
    source,prior,obs=fixture()
    clock=pd.date_range('2000-01-01',periods=24)
    frame=pd.DataFrame(dict(station='TARGET',grid_time=clock,background=11.+np.arange(24),
        observed=15.+np.arange(24),longitude=110.,latitude=30.,unit='K'))
    request=dict(task='temporal',adapter='paired_station_calibration',variable='air_temperature',unit='K',
        observation_update=POLICY,background_product='declared_product')
    frame.loc[18:,'observed']=frame.loc[18:,'background']-4
    obs.loc[obs.time>=clock[18],'value']=obs.loc[obs.time>=clock[18],'background']-4
    inputs=[SearchInput('target.csv','paired_observations',frame.to_csv(index=False).encode()),
        SearchInput('conditioning.csv','observations',obs.to_csv(index=False).encode())]
    budget=dict(objective_limit=80,total_call_limit=120,proposal_limit=256,wall_seconds=30)
    result=submit_search(inputs,request,None,tmp_path/'run',budget=budget,seed=11)
    run=result['stations'][0];assert run['best']['metrics']['RMSE']<1e-12,run
    trace=run['best']['trace'];update=next(t for t in trace if t['operation']=='resource.observation_state_update')
    assert {e['from'] for e in update['edges']}=={'s0.field','__fitted_state.state','__conditioning.observations'}
    coefficients=update['outputs']['state']['metadata']['report']['coefficients']['affine_intercept']
    np.testing.assert_allclose(coefficients[:6],4.)
    np.testing.assert_allclose(coefficients[6:],-4.)
    assert any(e['from']=='__observation_state.state' for n in trace for e in n['edges'])
    assert (tmp_path/'run/station-1/conditioning_resource.json').exists()
    assert any('authorized exact-period' in n['outputs'].get('field',{}).get('metadata',{}).get('report',{}).get('fitted_on','') for n in trace)
    changed=frame.copy();changed.loc[18:,'observed']+=1000
    other=submit_search([SearchInput('target.csv','paired_observations',changed.to_csv(index=False).encode()),inputs[1]],
        request,None,tmp_path/'changed',budget=budget,seed=11)['stations'][0]
    assert other['best']['key']==run['best']['key'] and other['best']['metrics']==run['best']['metrics']
    assert other['independent_evaluation']['RMSE']!=run['independent_evaluation']['RMSE']


def test_actual_dag_cache_updates_changed_observations_but_reuses_background():
    from ecodata.algorithms.autonomous_field import build_context
    from ecodata.algorithms.autonomous_decision import Executor
    from ecodata.algorithms.platform_search import platform_registry
    source,prior,obs=fixture()
    inputs=[SearchInput('background.nc','primary',source.to_netcdf()),SearchInput('state.nc','state',prior.to_netcdf()),
        SearchInput('observations.csv','observations',obs.to_csv(index=False).encode())]
    request=dict(task='temporal',variable='air_temperature',unit='K',semantic_type='continuous',allow_training=True,
        parameters=dict(observation_state_update=POLICY))
    context=build_context(inputs,request)[0]
    executor=Executor(context,platform_registry())
    graph=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        output=dict(operation='bias.prior_series_affine',parents=['s0'],parameters={}))
    first,_=executor.execute(graph,'output');same,trace=executor.execute(graph,'output')
    xr.testing.assert_allclose(first,same);assert all(n['cache_hit'] for n in trace)
    changed=context.observations.copy()
    mask=pd.to_datetime(changed.time)==pd.Timestamp(source.time.values[-1]);changed.loc[mask,'value']+=2
    executor.context=replace(context,observations=changed)
    second,trace=executor.execute(graph,'output')
    assert next(n for n in trace if n['operation']=='source')['cache_hit']
    assert not next(n for n in trace if n['operation']=='resource.observation_state_update')['cache_hit']
    np.testing.assert_array_equal(first.air_temperature.values[:-1],second.air_temperature.values[:-1])
    np.testing.assert_allclose(second.air_temperature.values[-1]-first.air_temperature.values[-1],2.)


def test_interval_update_requires_identical_bounds_and_storage_units():
    source,prior,obs=fixture()
    source.air_temperature.attrs['temporal_semantics']='mean'
    clock=pd.DatetimeIndex(source.time.values)
    source['time_bounds']=(('time','bounds'),np.column_stack([clock.values-np.timedelta64(1,'D'),clock.values]))
    obs['temporal_semantics']='mean';obs['interval_end']=obs.time
    obs['interval_start']=obs.time-pd.Timedelta(days=1)
    a,_=update_state(prior,source,obs,'air_temperature',POLICY)
    obs['interval_start']=obs.interval_start.astype('datetime64[us]');obs['interval_end']=obs.interval_end.astype('datetime64[us]')
    b,_=update_state(prior,source,obs,'air_temperature',POLICY)
    xr.testing.assert_allclose(a,b)
    obs.loc[obs.time==clock[0],'interval_end']+=pd.Timedelta(hours=1)
    with pytest.raises(ValueError,match='intervals differ'):update_state(prior,source,obs,'air_temperature',POLICY)


def test_updated_coefficients_cannot_be_borrowed_without_real_producer():
    from ecodata.algorithms.contracts import AlgorithmContext
    from ecodata.core.schemas import EcoDataBundle
    from ecodata.algorithms.series_fusion import PriorSeriesCalibration
    source,prior,obs=fixture();state,_=update_state(prior,source,obs,'air_temperature',POLICY)
    context=AlgorithmContext((EcoDataBundle(source),),'air_temperature','continuous',parameters=dict(fitted_state=state))
    check=PriorSeriesCalibration('affine').compatibility(context)
    assert not check.eligible and any('producer parent' in reason for reason in check.reasons)
    prior['monthly_ratio']=('month',np.ones(12))
    context=replace(context,parameters=dict(fitted_state=prior,observation_state_update=POLICY))
    check=PriorSeriesCalibration('monthly_ratio').compatibility(context)
    assert not check.eligible and any('no concurrent' in reason for reason in check.reasons)
    with pytest.raises(ValueError,match='forecasts are not implied'):
        update_state(prior,source,obs,'air_temperature',dict(mode='forecast',evidence='Not supported'))
    wrong=prior.copy(deep=True);wrong.attrs['unit']='mm'
    with pytest.raises(ValueError,match='state unit conflict'):update_state(wrong,source,obs,'air_temperature',POLICY)


def test_batch_conditioning_cannot_borrow_another_evaluation_target(tmp_path):
    source,prior,obs=fixture()
    clock=pd.date_range('2000-01-01',periods=24)
    a=pd.DataFrame(dict(station='TARGET',grid_time=clock,background=11.+np.arange(24),observed=15.+np.arange(24),
        longitude=110.,latitude=30.,unit='K'))
    b=a.copy();b['station']='A';b['longitude']=109.
    request=dict(task='temporal',adapter='paired_station_calibration',variable='air_temperature',unit='K',
        observation_update=POLICY,background_product='declared_product')
    with pytest.raises(ValueError,match='evaluated target station'):
        submit_search([SearchInput('targets.csv','paired_observations',pd.concat([a,b]).to_csv(index=False).encode()),
            SearchInput('conditioning.csv','observations',obs.to_csv(index=False).encode())],request,None,tmp_path/'run')
