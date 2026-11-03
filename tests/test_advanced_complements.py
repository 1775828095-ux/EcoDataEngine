from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter,shift
from ecodata.algorithms.advanced_complements import WhittakerRaster,DineofRaster,GuidedRaster,ArosicsRaster,StarfmRaster,plugins
from ecodata.algorithms.raster_complements import EdgeDenoise
from ecodata.algorithms.raster_gap_prediction import GridGapRecovery
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms import platform_search as api


def field(n=16,t=8,kind='ndvi',source='fine'):
    dates=pd.date_range('2015-01-01',periods=t,freq='8D');y,x=np.mgrid[:n,:n]
    base=.3+.1*np.sin(x/2)+.05*np.cos(y/3)
    values=np.stack([base+.05*np.sin(k/2) for k in range(t)])
    ds=xr.Dataset({kind:(('time','lat','lon'),values)},coords=dict(time=dates,lat=29.7+np.arange(n)*.0003,lon=118.+np.arange(n)*.0003),attrs=dict(crs='EPSG:4326',source_id=source))
    ds[kind].attrs=dict(units='1',data_kind='continuous',physical_variable=kind,physical_definition=kind,
        spatial_semantics='cell_mean',temporal_semantics='representative_observation',measurement_uncertainty=.02)
    return ds


def context(*data,**p):
    variable=next(iter(data[0].data_vars))
    return AlgorithmContext(tuple(EcoDataBundle(d) for d in data),variable,'continuous',tuple(data[0].lat.values),tuple(data[0].lon.values),
        parameters=dict(raster_target_times=[str(t) for t in data[0].time.values],**p),model_constraints=dict(unit='1'))


def test_all_five_public_nodes_are_available():
    registry=api.platform_registry()
    for plugin in plugins():assert registry.get(plugin.spec.id).spec.status=='available'


def test_whittaker_real_kernel_matches_independent_penalty_equation():
    ds=field(n=8);ds.ndvi.values[3:5]=np.nan;original=ds.copy(deep=True)
    c=context(ds,smoothing_lambda=3.);result=WhittakerRaster().run(c)
    y=ds.ndvi.values[:,2,2];w=np.isfinite(y).astype(float);d=np.diff(np.eye(len(y)),n=2,axis=0)
    expected=np.linalg.solve(np.diag(w)+3*d.T@d,w*np.nan_to_num(y))
    np.testing.assert_allclose(result.data.dataset.ndvi.values[:,2,2],expected,atol=1e-11)
    assert result.report['kernel']=='vam.whittaker.ws2d'
    xr.testing.assert_equal(ds,original)


def test_whittaker_rejects_irregular_time_and_all_missing_pixel():
    ds=field();ds=ds.assign_coords(time=ds.time.values+np.arange(8)**2*np.timedelta64(1,'D'))
    assert not WhittakerRaster().compatibility(context(ds,smoothing_lambda=2.)).eligible
    ds=field();ds.ndvi.values[:,1,1]=np.nan
    assert not WhittakerRaster().compatibility(context(ds,smoothing_lambda=2.)).eligible


def test_dineof_actual_R_kernel_restores_mask_and_preserves_observations():
    ds=field(n=8);rng=np.random.default_rng(7);mask=rng.uniform(size=ds.ndvi.shape)<.15
    original=ds.ndvi.values.copy();ds.ndvi.values[mask]=np.nan
    c=context(ds,max_modes=4,internal_seed=9);result=DineofRaster().run(c)
    values=result.data.dataset.ndvi.values
    assert np.isfinite(values).all() and result.report['external_labels_used'] is False
    np.testing.assert_array_equal(values[~mask],original[~mask])
    np.testing.assert_array_equal(DineofRaster().run(c).data.dataset.ndvi.values,values)
    assert np.sqrt(np.mean((values[mask]-original[mask])**2))<.15


def test_dineof_does_not_invent_entirely_unobserved_date():
    ds=field();ds.ndvi.values[3]=np.nan
    assert not DineofRaster().compatibility(context(ds,max_modes=3,internal_seed=0)).eligible


def test_guided_filter_real_two_parent_kernel_and_unit_equivariance():
    guide=field();target=field(source='coarse');target.ndvi.values=gaussian_filter(target.ndvi.values,(0,1.5,1.5))
    p=dict(guide_index=0,radius_pixels=3,relative_eps=1e-3)
    a=GuidedRaster().run(context(guide,target,**p)).data.dataset
    changed=guide.copy(deep=True);changed.ndvi.values=np.roll(changed.ndvi.values,4,axis=2)
    b=GuidedRaster().run(context(changed,target,**p)).data.dataset
    assert not np.allclose(a.ndvi,b.ndvi)
    scaled=[]
    for ds in [guide,target]:
        ds=ds.copy(deep=True);ds.ndvi.values=ds.ndvi.values*100+273.15;ds.ndvi.attrs['units']='K';scaled.append(ds)
    c=GuidedRaster().run(context(*scaled,**p)).data.dataset
    np.testing.assert_allclose(c.ndvi,a.ndvi*100+273.15,atol=2e-4)


def test_missing_then_denoise_order_is_enforced_not_fixed_globally():
    ds=field(n=8);ds.ndvi.values[2:4]=np.nan
    assert not EdgeDenoise('tv').compatibility(context(ds,noise_multiplier=1.)).eligible
    filled=WhittakerRaster().run(context(ds,smoothing_lambda=2.)).data.dataset
    denoised=EdgeDenoise('tv').run(context(filled,noise_multiplier=1.)).data.dataset
    reverse=WhittakerRaster().run(context(denoised,smoothing_lambda=2.)).data.dataset
    assert np.isfinite(reverse.ndvi).all()


def registration_fields():
    ds=field(n=128,t=4);rng=np.random.default_rng(21)
    ds=ds.assign_coords(lat=np.arange(128)*30.,lon=np.arange(128)*30.);ds.attrs['crs']='EPSG:32650'
    image=gaussian_filter(rng.normal(size=(128,128)),1.5)
    image=.3+.2*(image-image.min())/np.ptp(image)
    ds.ndvi.values[:]=image
    moved=ds.copy(deep=True);moved.attrs['source_id']='misregistered'
    moved.ndvi.values[:]=shift(image,(2.,-3.),order=1,mode='reflect')
    return ds,moved


def test_arosics_real_shift_and_later_gap_recovery():
    ref,moved=registration_fields();c=context(ref,moved,reference_index=0,max_shift_pixels=5,allow_image_coregistration=True)
    result=ArosicsRaster().run(c);out=result.data.dataset
    before=np.mean((moved.ndvi.values[:,10:-10,10:-10]-ref.ndvi.values[:,10:-10,10:-10])**2)
    after=np.mean((out.ndvi.values[:,10:-10,10:-10]-ref.ndvi.values[:,10:-10,10:-10])**2)
    assert after<before*.1
    assert np.isnan(out.ndvi.values).any()
    fill=GridGapRecovery('nearest').run(context(out,grid_gap_recovery=True)).data.dataset
    assert np.isfinite(fill.ndvi).all()
    assert abs(result.report['frames'][0]['x_pixels']-3)<.3
    assert abs(result.report['frames'][0]['y_pixels']+2)<.3


def test_arosics_requires_explicit_permission_and_rejects_model_veto():
    ref,moved=registration_fields();c=context(ref,moved,reference_index=0,max_shift_pixels=5)
    assert not ArosicsRaster().compatibility(c).eligible
    c=replace(c,parameters=dict(c.parameters,allow_image_coregistration=True),model_constraints=dict(forbid_spatial_shift=True))
    assert not ArosicsRaster().compatibility(c).eligible


def test_public_request_carries_spatial_shift_veto_to_real_plugin():
    from ecodata.algorithms.autonomous_field import build_context
    ref=field(n=64,t=4);moved=ref.copy(deep=True);moved.attrs['source_id']='moving'
    sources=[api.SearchInput('ref.nc','primary',ref.to_netcdf()),api.SearchInput('moving.nc','background',moved.to_netcdf())]
    request=dict(task='multisource',variable='ndvi',unit='1',semantic_type='continuous',constraints=dict(forbid_spatial_shift=True),
        parameters=dict(allow_image_coregistration=True,reference_index=0,max_shift_pixels=5))
    ctx,_,_=build_context(sources,request)
    assert ctx.model_constraints['forbid_spatial_shift'] is True
    assert not ArosicsRaster().compatibility(ctx).eligible


def starfm_fields():
    fine=field(n=8,t=4,kind='surface_reflectance');coarse=fine.copy(deep=True);coarse.attrs['source_id']='coarse'
    fine=fine.isel(time=[0,3]);coarse.surface_reflectance.values+=.01
    return fine,coarse


def test_starfm_actual_upstream_call_and_target_coarse_response():
    fine,coarse=starfm_fields();p=dict(fine_index=0,window_pixels=3,spatial_impact_m=150.)
    c=context(coarse,fine,**dict(p,fine_index=1));plugin=StarfmRaster();a=plugin.run(c)
    assert a.data.dataset.sizes['time']==4 and np.isfinite(a.data.dataset.surface_reflectance).all()
    changed=coarse.copy(deep=True);changed.surface_reflectance.values[1]+=.03
    b=plugin.run(context(changed,fine,**dict(p,fine_index=1)))
    assert not np.allclose(a.data.dataset.surface_reflectance.values[1],b.data.dataset.surface_reflectance.values[1])
    assert a.report['anchors'][0].startswith('2015-01-01')


def test_starfm_rejects_unknown_uncertainty_misalignment_and_wrong_variable():
    fine,coarse=starfm_fields();p=dict(fine_index=0,window_pixels=3,spatial_impact_m=150.)
    fine.surface_reflectance.attrs.pop('measurement_uncertainty')
    assert not StarfmRaster().compatibility(context(fine,coarse,**p)).eligible
    fine,coarse=starfm_fields();coarse=coarse.assign_coords(lon=coarse.lon+1)
    assert not StarfmRaster().compatibility(context(fine,coarse,**p)).eligible
    assert not StarfmRaster().compatibility(context(field(),field(source='coarse'),**p)).eligible


def test_public_api_deploys_whittaker_then_guided_filter(monkeypatch,tmp_path):
    from test_raster_gap_application import registry,contract
    registry(monkeypatch,tmp_path);fine=field(n=8);coarse=field(n=8,source='coarse');fine.ndvi.values[2:4]=np.nan
    declaration=contract();declaration.update(field='ndvi',valid_range=[-1.,1.]);declaration['issues']=declaration['issues'][:1]
    request=dict(task='multisource',variable='ndvi',unit='1',semantic_type='continuous',target_lat=fine.lat.values.tolist(),
        target_lon=fine.lon.values.tolist(),target_times=[str(t) for t in fine.time.values],application_contract=declaration,scoring=dict(registry_version='grid-test-v1'))
    sources=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('coarse.nc','background',coarse.to_netcdf())]
    records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),s1=dict(operation='source',parents=[],parameters=dict(source='source1')),
        w=dict(operation='temporal.whittaker',parents=['s0'],parameters=dict(smoothing_lambda=2.)),
        g=dict(operation='fusion.guided_filter',parents=['w','s1'],parameters=dict(guide_index=0,radius_pixels=2,relative_eps=.001)))
    result=api.execute_frozen_graph(sources,request,dict(records=records,key='g'),tmp_path/'deploy')
    assert not result['reference_used'] and result['cost']['used']==2
    assert all(e['status']=='PASS' for e in result['cost']['events'])
    (tmp_path/'receipt.json').write_text(json.dumps(result,default=str,indent=2),encoding='utf-8')


def test_starfm_and_temporal_filter_both_orders_through_public_api(monkeypatch,tmp_path):
    from test_raster_gap_application import registry,contract
    registry(monkeypatch,tmp_path);fine,coarse=starfm_fields()
    declaration=contract();declaration.update(field='surface_reflectance',valid_range=[0.,1.]);declaration['issues']=declaration['issues'][:1]
    request=dict(task='multisource',variable='surface_reflectance',unit='1',semantic_type='continuous',
        target_lat=coarse.lat.values.tolist(),target_lon=coarse.lon.values.tolist(),target_times=[str(t) for t in coarse.time.values],
        parameters=dict(raster_target_times=[str(t) for t in coarse.time.values]),application_contract=declaration,scoring=dict(registry_version='grid-test-v1'))
    sources=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('coarse.nc','background',coarse.to_netcdf())]
    roots={f's{i}':dict(operation='source',parents=[],parameters=dict(source=f'source{i}')) for i in [0,1]}
    sp=dict(fine_index=0,window_pixels=3,spatial_impact_m=150.)
    before=dict(roots,w=dict(operation='temporal.whittaker',parents=['s1'],parameters=dict(smoothing_lambda=2.)),
        f=dict(operation='fusion.starfm',parents=['s0','w'],parameters=sp))
    after=dict(roots,f=dict(operation='fusion.starfm',parents=['s0','s1'],parameters=sp),
        w=dict(operation='temporal.whittaker',parents=['f'],parameters=dict(smoothing_lambda=2.)))
    a=api.execute_frozen_graph(sources,request,dict(records=before,key='f'),tmp_path/'before')
    b=api.execute_frozen_graph(sources,request,dict(records=after,key='w'),tmp_path/'after')
    assert a['cost']['used']==b['cost']['used']==2
    with xr.open_dataset(a['product']) as x,xr.open_dataset(b['product']) as y:
        assert not np.allclose(x.surface_reflectance,y.surface_reflectance)


def test_public_registry_automatic_mutation_can_generate_all_five_nodes(monkeypatch,tmp_path):
    from test_raster_gap_application import registry,contract
    from ecodata.algorithms.profile_matching import prepare_model_request
    from ecodata.algorithms.autonomous_field import build_context,field_registry
    from ecodata.algorithms.field_mutation import GraphMutation
    registry(monkeypatch,tmp_path)
    coarse=field(n=64,t=4,kind='surface_reflectance',source='coarse');truth=coarse.surface_reflectance.copy(deep=True)
    rng=np.random.default_rng(11);texture=gaussian_filter(rng.normal(size=(64,64)),1.)*.02
    coarse.surface_reflectance.values+=texture;fine=coarse.copy(deep=True);fine.attrs['source_id']='fine'
    mask=rng.uniform(size=(64,64))<.15;fine.surface_reflectance.values[1,mask]=np.nan
    declaration=contract();declaration.update(field='surface_reflectance',valid_range=[0.,1.]);declaration['issues']=declaration['issues'][:1]
    request=dict(task='multisource',variable='surface_reflectance',unit='1',semantic_type='continuous',target_lat=coarse.lat.values.tolist(),
        target_lon=coarse.lon.values.tolist(),target_times=[str(t) for t in coarse.time.values],
        parameters=dict(raster_target_times=[str(t) for t in coarse.time.values],allow_image_coregistration=True,grid_gap_recovery=True),
        application_contract=declaration,scoring=dict(registry_version='grid-test-v1'),response_scoring=dict(enabled=False),mutation_control='random')
    sources=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('coarse.nc','background',coarse.to_netcdf())]
    prepared,model=prepare_model_request(request);ctx,_,_=build_context(sources,prepared)
    folder=tmp_path/'generated';folder.mkdir()
    engine=GraphMutation(ctx,field_registry(api.platform_registry(),ctx),prepared,model,truth,folder,
        dict(objective_limit=100,proposal_limit=50000,wall_seconds=180),11)
    needed={p.spec.id for p in plugins()};found={};current='s0'
    try:
        for _ in range(5):
            for edit in engine.neighbourhood(current,'structure'):
                if not edit or edit.get('algorithm') not in needed-set(found):continue
                row=engine.execute(edit['child'],'mechanism_reachability')
                if row['status'] in ('ELIGIBLE_OUTPUT','INTERMEDIATE_ONLY'):
                    found[edit['algorithm']]=dict(edit=edit,result=row['status']);current=edit['child'];break
            if set(found)==needed:break
        assert set(found)==needed
        assert any(engine.node_count(x['edit']['child'])>1 for x in found.values())
        if engine.evaluated[current]['status']!='ELIGIBLE_OUTPUT':
            repaired=None
            for edit in engine.neighbourhood(current,'structure'):
                if not edit or edit.get('algorithm')!='reconstruction.grid_gap_nearest' or edit.get('target')!=current:continue
                row=engine.execute(edit['child'],'mechanism_boundary_completion')
                if row['status']=='ELIGIBLE_OUTPUT':repaired=edit['child'];break
            assert repaired is not None
            current=repaired
        graph=dict(records={k:engine.records[k] for k in engine.nodes(current)},key=current)
        assert needed<={r['operation'] for r in graph['records'].values()}
        deployed=api.execute_frozen_graph(sources,request,graph,tmp_path/'automatic_graph_deploy')
        assert not deployed['reference_used']
        assert needed<={e['algorithm'] for e in deployed['cost']['events'] if e['status']=='PASS'}
        (tmp_path/'generated_graph.json').write_text(json.dumps(graph,indent=2),encoding='utf-8')
        input_folder=tmp_path/'inputs';input_folder.mkdir()
        saved=[]
        for item in sources:
            path=input_folder/item.filename;path.write_bytes(item.content)
            saved.append(dict(path=str(path.resolve()),role=item.role))
        (tmp_path/'generated_config.json').write_text(json.dumps(dict(inputs=saved,request=request,graph=graph),indent=2),encoding='utf-8')
        (tmp_path/'automatic_reachability.json').write_text(json.dumps(dict(status='PASS',found=found,
            full_public_registry=True,complete_public_deployment=True,registered_calls=deployed['cost']['used'],
            purpose='Reachability only; reviewer selects generated edits to exercise methods, not an optimization superiority test'),default=str,indent=2),encoding='utf-8')
    finally:engine.close()
