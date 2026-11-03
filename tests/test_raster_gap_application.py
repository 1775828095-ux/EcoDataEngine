from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms.raster_gap_prediction import GridGapRecovery,PriorGridForest,DeclaredRangeProjection
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms import attribute_registry as ar
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import measured


def sample():
    lat=np.linspace(29.7,29.73,5);lon=np.linspace(118.,118.03,5);times=pd.to_datetime([f'{y}-07-01' for y in range(2013,2018)])
    x=np.arange(125).reshape(5,5,5)/125;truth=.2+.5*x
    aux=xr.Dataset(dict(lai=(('time','lat','lon'),x),rain=(('time','lat','lon'),x*100),
        training_target=(('time','lat','lon'),np.where(np.arange(5)[:,None,None]<2,truth,np.nan))),
        coords=dict(time=times,lat=lat,lon=lon),attrs=dict(crs='EPSG:4326',training_end='2014-12-31'))
    for name in ['lai','rain']:aux[name].attrs=dict(units='1' if name=='lai' else 'mm',feature_role='covariate')
    aux.training_target.attrs=dict(units='1',split='train')
    ds=xr.Dataset(dict(fvc=(('time','lat','lon'),truth[2:].copy())),coords=dict(time=times[2:],lat=lat,lon=lon),attrs=dict(crs='EPSG:4326',source_id='observed'))
    ds.fvc.attrs=dict(units='1',physical_variable='fvc',spatial_semantics='cell_mean',temporal_semantics='mean',physical_definition='Declared FVC proxy',data_kind='continuous',valid_min=0.,valid_max=1.)
    ds.fvc.values[:,2,2]=np.nan
    context=AlgorithmContext((EcoDataBundle(ds),),'fvc','continuous',tuple(lat),tuple(lon),auxiliaries=(aux,),
        parameters=dict(grid_gap_recovery=True,grid_predictor_features=['lai'],target_valid_range=[0.,1.]),model_constraints=dict(nonnegative=True,unit='1'))
    return ds,aux,context,truth[2:]


def registry(monkeypatch,tmp_path):
    monkeypatch.setattr(ar,'ROOT',tmp_path/'registries')
    attributes=deepcopy(ar.load('builtin-v1')['attributes'])
    for metric in ['missing_fraction','negative_fraction','range_excess']:
        attributes.append(dict(id='grid_'+metric,definition='Measured '+metric,tables=['data','algorithm','model'],unit='1',
            role='scoring',scoring_enabled=True,hard_allowed=False,provider='state_metric',property=metric,
            activation='constant_one',evidence='Explicit reconstruction task diagnostic',history=['Initial tested contract']))
    return ar.publish('grid-test-v1',attributes,evidence='Synthetic unit test contract')


def contract():
    return dict(id='fvc-reconstruction',version='1',field='fvc',unit='1',semantic_type='continuous',valid_range=[0.,1.],
        evidence='Bounded fraction reconstruction, not native simulator sensitivity',issues=[dict(attribute='grid_'+metric,metric=metric,
        acceptable=0.,bad=1.,importance=1.,evidence='Explicit finite bounded fraction requirement') for metric in ['missing_fraction','negative_fraction','range_excess']])


@pytest.mark.parametrize('method',['nearest','linear','cubic'])
def test_grid_gap_executes_real_kernel_and_preserves_visible(method):
    source,_,context,_=sample();result=GridGapRecovery(method).run(context).data.dataset
    assert np.isfinite(result.fvc).all()
    known=np.isfinite(source.fvc);np.testing.assert_array_equal(result.fvc.values[known],source.fvc.values[known])
    complete=replace(context,sources=(EcoDataBundle(result),))
    assert GridGapRecovery(method).compatibility(complete).eligible
    np.testing.assert_array_equal(GridGapRecovery(method).run(complete).data.dataset.fvc,result.fvc)


def test_forest_automatic_feature_subsets_and_future_target_rejection():
    source,aux,context,_=sample();plugin=PriorGridForest()
    assert plugin.available_features(context)==['lai','rain']
    proposals=[p['grid_predictor_features'] for seed in range(20) for p in plugin.parameter_proposals(context,np.random.default_rng(seed))]
    assert ['lai'] in proposals and ['rain'] in proposals and ['lai','rain'] in proposals
    output=plugin.run(context).data.dataset;assert np.isfinite(output.fvc).all()
    np.testing.assert_array_equal(output.fvc.values[np.isfinite(source.fvc)],source.fvc.values[np.isfinite(source.fvc)])
    poisoned=aux.copy(deep=True);poisoned.training_target.values[4]=.9
    assert not plugin.compatibility(replace(context,auxiliaries=(poisoned,))).eligible
    forbidden=replace(context,parameters=dict(context.parameters,grid_predictor_features=['training_target']))
    assert not plugin.compatibility(forbidden).eligible


def test_explicit_range_is_terminal_gate_and_real_projection():
    source,_,context,truth=sample();source.fvc.values[:]=1.2
    ref=source.fvc.isel(time=[0],lat=[0],lon=[0]).copy();ref.values[:]=.8
    with pytest.raises(ValueError,match='physical range'):measured(source,ref,context)
    corrected=DeclaredRangeProjection().run(context).data.dataset
    assert corrected.fvc.max()==1.


def test_application_contract_public_random_and_no_mutation(monkeypatch,tmp_path):
    registry(monkeypatch,tmp_path)
    ds,aux,_,truth=sample();prior=ds.copy(deep=True);prior.fvc.values[:]=.4;prior.attrs['source_id']='prior'
    request=dict(task='multisource',variable='fvc',unit='1',semantic_type='continuous',nonnegative=True,
        target_lat=ds.lat.values.tolist(),target_lon=ds.lon.values.tolist(),target_times=[str(t) for t in ds.time.values],allow_training=True,
        parameters=dict(grid_gap_recovery=True,require_complete_output=True),application_contract=contract(),
        scoring=dict(registry_version='grid-test-v1'),response_scoring=dict(enabled=False),protect_observed_source='observed')
    inputs=[api.SearchInput('a.nc','primary',ds.to_netcdf()),api.SearchInput('b.nc','background',prior.to_netcdf()),api.SearchInput('aux.nc','auxiliary',aux.to_netcdf())]
    ref=ds.fvc.isel(time=[0],lat=[2],lon=[2]).copy();ref.values[:]=truth[0,2,2];ref.attrs['split']='search_validation'
    results=[]
    for control in ['random','none']:
        q=dict(request,mutation_control=control)
        r=api.submit_search(inputs,q,None,tmp_path/control,mode='PortraitMutation',validation=api.SearchValidation(ref,'fvc'),
             budget=dict(objective_limit=80,proposal_limit=4000,wall_seconds=30),seed=11)
        assert r['status']=='PASS' and r['total_cost']['used']<=80
        with xr.open_dataset(r['product']) as output:assert np.isfinite(output.fvc).all() and output.fvc.max()<=1
        results.append(r)
    assert results[0]['evolution_used'] and not results[1]['evolution_used']
    bad=deepcopy(request);bad['application_contract']['unit']='K'
    from ecodata.algorithms.profile_matching import prepare_model_request
    with pytest.raises(ValueError,match='identity'):prepare_model_request(bad)
