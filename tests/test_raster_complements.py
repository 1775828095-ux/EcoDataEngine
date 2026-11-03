from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.raster_complements import EdgeDenoise,WaveletRaster
from ecodata.algorithms.open_source_bias import XsdbaPlugin
from ecodata.algorithms import platform_search as api


IDS=['spatial.denoise_tv','spatial.denoise_bilateral','spatial.wavelet_lowpass','fusion.wavelet_detail_transfer','bias.xsdba_scaling']


def data(n=24,dates=None,source='fine',seed=11):
    dates=pd.to_datetime(dates or ['2015-01-01','2015-01-09','2015-01-17'])
    y,x=np.mgrid[:n,:n];base=.25+.25*(x>=n//2)+.1*y/n
    values=np.stack([base+.02*t+np.random.default_rng(seed+t).normal(0,.025,(n,n)) for t in range(len(dates))])
    ds=xr.Dataset(dict(ndvi=(('time','lat','lon'),values)),coords=dict(time=dates,lat=np.linspace(29.7,29.73,n),lon=np.linspace(118.,118.03,n)),
        attrs=dict(crs='EPSG:4326',source_id=source))
    ds.ndvi.attrs=dict(units='1',data_kind='continuous',physical_variable='ndvi',physical_definition='Normalized difference vegetation index',
        spatial_semantics='cell_mean',temporal_semantics='representative_observation')
    return ds


def context(*sources,**parameters):
    return AlgorithmContext(tuple(EcoDataBundle(s) for s in sources),'ndvi','continuous',
        tuple(sources[0].lat.values),tuple(sources[0].lon.values),parameters=parameters,model_constraints=dict(unit='1'))


@pytest.mark.parametrize('method',['tv','bilateral'])
def test_real_denoisers_preserve_geometry_units_and_separate_frames(method):
    ds=data();original=ds.copy(deep=True);plugin=EdgeDenoise(method)
    c=context(ds,noise_multiplier=1.,sigma_spatial_pixels=2.)
    result=plugin.run(c);out=result.data.dataset
    assert plugin.spec.status=='available' and result.report['reference_used'] is False
    xr.testing.assert_equal(ds,original)
    assert out.ndvi.attrs['units']=='1' and np.isfinite(out.ndvi).all()
    assert np.mean(abs(np.diff(out.ndvi.values,axis=2)))<np.mean(abs(np.diff(ds.ndvi.values,axis=2)))
    for i in range(len(ds.time)):
        local=plugin.run(context(ds.isel(time=i,drop=True),noise_multiplier=1.,sigma_spatial_pixels=2.)).data.dataset
        np.testing.assert_allclose(out.ndvi.values[i],local.ndvi.values)
    assert out.ndvi.attrs['distribution_transform']


@pytest.mark.parametrize('method',['tv','bilateral'])
def test_filters_are_idempotent_on_constant_field(method):
    ds=data();ds.ndvi.values[:]=.4
    out=EdgeDenoise(method).run(context(ds,noise_multiplier=1.,sigma_spatial_pixels=2.)).data.dataset
    np.testing.assert_array_equal(out.ndvi,ds.ndvi)


@pytest.mark.parametrize('method',['tv','bilateral'])
def test_scalar_units_are_not_silently_normalized_or_mixed(method):
    ds=data();plugin=EdgeDenoise(method)
    a=plugin.run(context(ds,noise_multiplier=1.,sigma_spatial_pixels=2.)).data.dataset.ndvi
    changed=ds.copy(deep=True);changed.ndvi.values=changed.ndvi.values*100+273.15;changed.ndvi.attrs['units']='K'
    b=plugin.run(context(changed,noise_multiplier=1.,sigma_spatial_pixels=2.)).data.dataset.ndvi
    np.testing.assert_allclose(b,(a*100+273.15),atol=1e-6)
    assert b.attrs['units']=='K'


def test_wavelet_odd_shape_reconstructs_same_field_and_diagnostic_details():
    ds=data(n=17);c=context(ds,ds,wavelet_level=2,detail_index=0,detail_strength=1.)
    out=WaveletRaster(True).run(c).data.dataset
    np.testing.assert_allclose(out.ndvi,ds.ndvi,atol=1e-12)
    np.testing.assert_allclose(out._wavelet_background+out._wavelet_detail,ds.ndvi,atol=1e-12)
    assert out.ndvi.shape==ds.ndvi.shape and out.attrs['fusion_method']=='stationary_wavelet_detail_transfer'
    low=WaveletRaster().run(context(ds,wavelet_level=2)).data.dataset
    np.testing.assert_allclose(low.ndvi,out._wavelet_background,atol=1e-12)


def test_detail_branch_changes_fusion_and_requires_actual_alignment():
    fine=data();base=data(source='coarse',seed=21)
    c=context(fine,base,wavelet_level=2,detail_index=0,detail_strength=1.)
    a=WaveletRaster(True).run(c).data.dataset
    altered=fine.copy(deep=True);altered.ndvi.values[:,:,::2]+=.05
    b=WaveletRaster(True).run(replace(c,sources=(EcoDataBundle(altered),c.sources[1]))).data.dataset
    assert float(abs(a.ndvi-b.ndvi).max())>.001
    assert not WaveletRaster(True).compatibility(context(fine,base.assign_coords(lon=base.lon+.001),wavelet_level=2,detail_index=0,detail_strength=1.)).eligible
    wrong=base.copy(deep=True);wrong.ndvi.attrs['units']='K'
    assert not WaveletRaster(True).compatibility(context(fine,wrong,wavelet_level=2,detail_index=0,detail_strength=1.)).eligible


def test_missing_input_is_rejected_not_silently_filled_and_all_levels_are_discoverable():
    ds=data();ds.ndvi.values[0,0,0]=np.nan
    assert not EdgeDenoise('tv').compatibility(context(ds,noise_multiplier=1.)).eligible
    assert not WaveletRaster().compatibility(context(ds,wavelet_level=1)).eligible
    good=data(n=48);p=WaveletRaster().parameter_proposals(context(good),np.random.default_rng(4))
    assert {q['wavelet_level'] for q in p}=={1,2,3,4}


def calibration():
    prior=data(n=12,dates=[str(t.date()) for t in pd.date_range('2013-01-01',periods=12,freq='MS')],source='coarse')
    reference=prior.ndvi.copy();reference.attrs=dict(reference.attrs,split='train')
    history=prior.ndvi+.125;history.attrs=dict(reference.attrs)
    app=data(n=12,source='coarse');app.ndvi.values+=.125
    return app,reference,history


def test_official_xsdba_scaling_real_grid_fit_and_prior_boundary():
    app,ref,hist=calibration();plugin=XsdbaPlugin('scaling')
    c=context(app,training_reference=ref,historical_simulation=hist,calibration_source_id='coarse')
    assert plugin.compatibility(c).eligible
    result=plugin.run(c)
    np.testing.assert_allclose(result.data.dataset.ndvi,app.ndvi-.125,atol=1e-12)
    assert result.report['test_used_for_fitting'] is False and result.report['training_samples']==12
    bad=ref.assign_coords(time=app.time.values[0]+np.arange(12)*np.timedelta64(1,'D'))
    bad_hist=hist.assign_coords(time=bad.time)
    assert not plugin.compatibility(replace(c,parameters=dict(c.parameters,training_reference=bad,historical_simulation=bad_hist))).eligible


def test_calibration_state_must_match_preprocessing_not_fixed_order():
    app,ref,hist=calibration();plugin=XsdbaPlugin('scaling')
    filtered=EdgeDenoise('tv').run(context(app,noise_multiplier=1.)).data.dataset
    c=context(filtered,training_reference=ref,historical_simulation=hist,calibration_source_id='coarse')
    assert not plugin.compatibility(c).eligible
    training=hist.to_dataset();training.attrs=dict(crs='EPSG:4326',source_id='coarse')
    filtered_hist=EdgeDenoise('tv').run(context(training,noise_multiplier=1.)).data.dataset.ndvi
    filtered_hist.attrs['split']='train'
    assert plugin.compatibility(replace(c,parameters=dict(c.parameters,historical_simulation=filtered_hist))).eligible


def test_public_frozen_graph_executes_calibration_denoise_and_wavelet_without_labels(monkeypatch,tmp_path):
    from test_raster_gap_application import registry,contract
    registry(monkeypatch,tmp_path);app,ref,hist=calibration();fine=data(n=12)
    declaration=contract();declaration.update(field='ndvi',valid_range=[-1.,1.]);declaration['issues']=declaration['issues'][:1]
    sources=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('coarse.nc','background',app.to_netcdf()),
        api.SearchInput('train_ref.nc','training_reference',ref.to_dataset().to_netcdf()),api.SearchInput('train_hist.nc','historical_simulation',hist.to_dataset().to_netcdf())]
    request=dict(task='multisource',variable='ndvi',unit='1',semantic_type='continuous',allow_training=True,
        target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),target_times=[str(t) for t in fine.time.values],
        parameters=dict(calibration_source_id='coarse'),application_contract=declaration,scoring=dict(registry_version='grid-test-v1'))
    records={f's{i}':dict(operation='source',parents=[],parameters=dict(source='source'+str(i))) for i in [0,1]}
    records.update(cal=dict(operation='bias.xsdba_scaling',parents=['s1'],parameters={}),
        den=dict(operation='spatial.denoise_tv',parents=['cal'],parameters=dict(noise_multiplier=1.)),
        wav=dict(operation='fusion.wavelet_detail_transfer',parents=['den','s0'],parameters=dict(wavelet_level=1,detail_index=1,detail_strength=.7)))
    result=api.execute_frozen_graph(sources,request,dict(records=records,key='wav'),tmp_path/'deploy')
    assert not result['reference_used'] and result['cost']['used']==3
    with xr.open_dataset(result['product']) as out:assert np.isfinite(out.ndvi).all() and '_wavelet_detail' in out
    assert set(IDS)<=set(s.id for s in api.platform_registry().specs())


def test_platform_generates_new_multistep_links_and_runs_public_random_search(monkeypatch,tmp_path):
    from test_raster_gap_application import registry,contract
    from ecodata.algorithms.profile_matching import prepare_model_request
    from ecodata.algorithms.autonomous_field import build_context,field_registry
    from ecodata.algorithms.field_mutation import GraphMutation
    registry(monkeypatch,tmp_path);coarse,ref,hist=calibration();full=data(n=12)
    fine=full.isel(time=[0,2]);truth=full.ndvi.isel(time=[1]).copy();truth.attrs['split']='search_validation'
    sources=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('coarse.nc','background',coarse.to_netcdf()),
        api.SearchInput('ref.nc','training_reference',ref.to_dataset().to_netcdf()),api.SearchInput('hist.nc','historical_simulation',hist.to_dataset().to_netcdf())]
    declaration=contract();declaration.update(field='ndvi',valid_range=[-1.,1.]);declaration['issues']=declaration['issues'][:1]
    request=dict(task='multisource',variable='ndvi',unit='1',semantic_type='continuous',allow_training=True,
        target_lat=full.lat.values.tolist(),target_lon=full.lon.values.tolist(),target_times=[str(t) for t in full.time.values],
        parameters=dict(calibration_source_id='coarse',raster_target_times=[str(t) for t in full.time.values],grid_gap_recovery=True),
        application_contract=declaration,scoring=dict(registry_version='grid-test-v1'),response_scoring=dict(enabled=False),mutation_control='random')
    prepared,model=prepare_model_request(request);ctx,_,_=build_context(sources,prepared)
    folder=tmp_path/'generated';folder.mkdir()
    engine=GraphMutation(ctx,field_registry(api.platform_registry(),ctx),prepared,model,truth,folder,
        dict(objective_limit=160,proposal_limit=30000,wall_seconds=120),11)
    try:
        initial=engine.initialize();assert initial
        chosen=None
        for edit in engine.neighbourhood(initial['key'],'structure'):
            if not edit:continue
            names={engine.records[k]['operation'] for k in engine.nodes(edit['child'])}
            if 'fusion.wavelet_detail_transfer' in names and names&{'spatial.denoise_tv','spatial.denoise_bilateral'}:
                chosen=edit;break
        assert chosen is not None
        row=engine.execute(chosen['child'],'mechanism_reachability')
        assert row['status']=='ELIGIBLE_OUTPUT'
        observed={engine.records[r['key']]['operation'] for r in engine.evaluated.values()}
        assert set(IDS)<=observed
        graph=dict(records={k:engine.records[k] for k in engine.nodes(chosen['child'])},key=chosen['child'])
        (tmp_path/'generated_graph.json').write_text(json.dumps(graph,indent=2),encoding='utf-8')
        receipt=api.execute_frozen_graph(sources,request,graph,tmp_path/'generated_public_deploy')
        assert not receipt['reference_used']
    finally:engine.close()
    result=api.submit_search(sources,request,None,tmp_path/'public_random',mode='PortraitMutation',
        validation=api.SearchValidation(truth,'ndvi'),budget=dict(objective_limit=100,proposal_limit=8000,wall_seconds=45),seed=11)
    assert result['status']=='PASS' and result['evolution_used']
    (tmp_path/'mechanism_acceptance.json').write_text(json.dumps(dict(status='PASS',engine_generated_graph=True,
        all_new_methods_executed=sorted(set(IDS)&observed),public_search_status=result['status'],
        role='Synthetic interface/reachability check, not scientific superiority or optimizer preference'),indent=2),encoding='utf-8')
