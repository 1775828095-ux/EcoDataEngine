from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.raster_temporal import RasterTimeInterpolation,VisibleAnchorResidual
from ecodata.algorithms.autonomous_field import measured
from ecodata.core.schemas import EcoDataBundle


def source(times,size,values):
    a=xr.DataArray(np.broadcast_to(np.asarray(values)[:,None,None],(len(times),size,size)).copy(),
        dims=('time','lat','lon'),coords=dict(time=pd.to_datetime(times),lat=np.linspace(0,1,size),lon=np.linspace(0,1,size)),name='v',
        attrs=dict(units='1',physical_variable='v',physical_definition='ndvi',spatial_semantics='cell_mean',temporal_semantics='representative_observation'))
    return a.to_dataset()


def context():
    fine=source(['2016-01-01','2016-01-05'],3,[.2,.6])
    coarse=source(['2016-01-01','2016-01-03','2016-01-05'],2,[.1,.7,.5])
    return AlgorithmContext((EcoDataBundle(fine),EcoDataBundle(coarse)),'v','continuous',tuple(fine.lat.values),tuple(fine.lon.values),
        parameters=dict(raster_target_times=['2016-01-01','2016-01-03','2016-01-05']))


@pytest.mark.parametrize('method',['linear','nearest'])
def test_temporal_visible_only(method):
    c=context();c=replace(c,sources=(c.sources[0],));plugin=RasterTimeInterpolation(method)
    assert plugin.compatibility(c).eligible
    a=plugin.run(c).data.dataset
    assert np.isfinite(a.v).all()
    assert float(a.v.isel(time=0).mean())==pytest.approx(.2)
    assert float(a.v.isel(time=-1).mean())==pytest.approx(.6)
    assert c.validation_reference is None


def test_coarse_changes_reconstruction_without_truth():
    c=context();p=VisibleAnchorResidual();c=replace(c,parameters=dict(c.parameters,fine_index=0,residual_strength=1.))
    a=p.run(c).data.dataset
    np.testing.assert_allclose(a.v.isel(time=1),.8)
    coarse=c.sources[1].dataset.copy(deep=True);coarse.v.values[1]=.3
    b=p.run(replace(c,sources=(c.sources[0],EcoDataBundle(coarse)))).data.dataset
    np.testing.assert_allclose(b.v.isel(time=1),.4)
    np.testing.assert_allclose(a.v.isel(time=[0,2]),c.sources[0].dataset.v)


def test_unobserved_fine_detail_uses_coarse_background_only():
    c=context();fine=c.sources[0].dataset.copy(deep=True);fine.v.values[:,1,1]=np.nan
    c=replace(c,sources=(EcoDataBundle(fine),c.sources[1]),parameters=dict(c.parameters,fine_index=0,residual_strength=1.))
    result=VisibleAnchorResidual().run(c)
    np.testing.assert_allclose(result.data.dataset.v.values[:,1,1],[.1,.7,.5])
    assert result.report['coarse_background_fallback_cells']==3


def test_partial_reference_fixed_support_rejects_missing_prediction():
    c=context();out=source(c.parameters['raster_target_times'],3,[.2,.4,.6]);ref=out.v.isel(time=[1]).copy()
    ref.values[0,0,0]=np.nan
    with pytest.raises(ValueError,match='partial-reference'):measured(out,ref,c)
    c=replace(c,parameters=dict(c.parameters,reference_support='fixed_finite_reference'))
    assert measured(out,ref,c)['denominator']==8
    out.v.values[1,1,1]=np.nan
    with pytest.raises(ValueError,match='fixed reference support'):measured(out,ref,c)


def test_same_values_different_dtype_have_identical_accuracy():
    c=context();out=source(c.parameters['raster_target_times'],3,[.2234567,.4123456,.6234567]).astype('float32')
    ref=source(['2016-01-03'],3,[.8345678]).v.astype('float32')
    assert measured(out,ref,c)==measured(out.astype('float64'),ref.astype('float64'),c)


def test_roundoff_does_not_reward_redundant_node_growth(tmp_path):
    from test_field_mutation import engine
    e=engine(tmp_path)
    try:
        parent=e.intern('constraint.series_nonnegative',['s0'],{})
        child=e.intern('constraint.series_nonnegative',[parent],{})
        scale=e.objective_resolution
        a=dict(key=parent,metrics=dict(objective=scale*1000.1),algorithm_nodes=1,status='ELIGIBLE_OUTPUT')
        b=dict(key=child,metrics=dict(objective=scale*1000.09),algorithm_nodes=2,status='ELIGIBLE_OUTPUT')
        e.evaluated.update({parent:a,child:b})
        assert not e.accept(parent,child)
        b['metrics']['objective']=scale*990
        assert e.accept(parent,child)
    finally:e.close()


def test_complete_grid_centre_resampling_rejects_identity():
    from ecodata.algorithms.platform_search import platform_registry
    from ecodata.algorithms.raster_reconstruction_task import GridCentreStationAdapter
    c=context();ds=c.sources[0].dataset.isel(time=0,drop=True);ds.attrs['crs']='EPSG:4326'
    local=replace(c,sources=(EcoDataBundle(ds),),model_constraints=dict(unit='1'))
    upstream=platform_registry().get('station.pykrige_ordinary')
    adapter=GridCentreStationAdapter(upstream)
    assert not adapter.compatibility(local).eligible
    assert adapter.compatibility(replace(local,target_lat=(.1,.5,.9),target_lon=(.1,.5,.9))).eligible


def test_declared_range_deploys_when_new_input_needs_no_clipping():
    from ecodata.algorithms.raster_gap_prediction import DeclaredRangeProjection
    c=context();c=replace(c,sources=(c.sources[0],),parameters=dict(c.parameters,target_valid_range=[-1.,1.]))
    plugin=DeclaredRangeProjection();assert plugin.compatibility(c).eligible
    xr.testing.assert_equal(plugin.run(c).data.dataset.v,c.sources[0].dataset.v)


@pytest.mark.parametrize('method',['linear','nearest'])
def test_complete_time_grid_is_legal_at_deployment(method):
    c=context();fine=c.sources[0]
    c=replace(c,sources=(fine,),parameters=dict(raster_target_times=[str(t) for t in fine.dataset.time.values]))
    plugin=RasterTimeInterpolation(method);assert plugin.compatibility(c).eligible
    xr.testing.assert_equal(plugin.run(c).data.dataset.v,fine.dataset.v)


def test_public_coarse_only_control_executes_without_truth(monkeypatch,tmp_path):
    from test_raster_gap_application import registry,contract
    from ecodata.algorithms import platform_search as api
    registry(monkeypatch,tmp_path);c=context();declaration=contract()
    declaration.update(field='v',valid_range=[-1.,1.]);declaration['issues']=declaration['issues'][:1]
    coarse=source(['2016-01-01','2016-01-02','2016-01-04','2016-01-05'],2,[.1,.3,.4,.5])
    sources=[]
    for i,ds in enumerate([c.sources[0].dataset,coarse]):
        ds=ds.copy();ds.attrs.update(crs='EPSG:4326',source_id='sensor'+str(i))
        sources.append(api.SearchInput(str(i)+'.nc','primary' if i==0 else 'background',ds.to_netcdf()))
    req=dict(task='multisource',variable='v',unit='1',semantic_type='continuous',target_lat=list(c.target_lat),target_lon=list(c.target_lon),
        target_times=c.parameters['raster_target_times'],parameters=c.parameters,application_contract=declaration,scoring=dict(registry_version='grid-test-v1'))
    graph=dict(records={'s1':dict(operation='source',parents=[],parameters=dict(source='source1')),
        't':dict(operation='temporal.raster_linear',parents=['s1'],parameters={}),
        'u':dict(operation='spatial.bilinear',parents=['t'],parameters={})},key='u')
    receipt=api.execute_frozen_graph(sources,req,graph,tmp_path/'coarse_deploy')
    assert not receipt['reference_used']
    with xr.open_dataset(receipt['product']) as ds:
        assert np.isfinite(ds.v).all();np.testing.assert_allclose(ds.v.isel(time=1),.35)
        assert ds.v.attrs['data_kind']=='continuous'
    # The same frozen graph must remain executable when its gap-fill step
    # encounters a newly complete parent in another year.
    gap_graph=dict(records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0')),
        't':dict(operation='temporal.raster_linear',parents=['s0'],parameters={}),
        'g':dict(operation='reconstruction.grid_gap_linear',parents=['t'],parameters={})},key='g')
    gap_req=dict(req,parameters=dict(req['parameters'],grid_gap_recovery=True))
    deployed=api.execute_frozen_graph(sources,gap_req,gap_graph,tmp_path/'complete_gap_deploy')
    with xr.open_dataset(deployed['product']) as ds:np.testing.assert_allclose(ds.v.isel(time=1),.4)
    conflict=c.sources[0].dataset.copy(deep=True);conflict.attrs.update(crs='EPSG:4326');conflict.v.attrs['data_kind']='categorical'
    bad=[api.SearchInput('bad.nc','primary',conflict.to_netcdf()),sources[1]]
    with pytest.raises(ValueError,match='data kind conflicts'):
        api.execute_frozen_graph(bad,req,graph,tmp_path/'conflict')


def test_visible_anchor_protection_on_expanded_time_axis():
    from ecodata.algorithms.autonomous_decision import protect_observed
    c=context();out=source(c.parameters['raster_target_times'],3,[.1,.3,.5])
    protected=protect_observed(out,c.sources[0].dataset,'v')
    np.testing.assert_allclose(protected.v.isel(time=[0,2]),c.sources[0].dataset.v)
    np.testing.assert_allclose(protected.v.isel(time=1),.3)


def test_empty_parameter_stream_yields_to_other_algorithms():
    from ecodata.algorithms.field_mutation import GraphMutation
    from ecodata.algorithms.contracts import AlgorithmSpec
    owner=GraphMutation.__new__(GraphMutation)
    owner.rng=np.random.default_rng(1);owner.roots=['s0']
    owner.records={'s0':dict(operation='source',parents=[],parameters={})}
    owner.states={f's{i}':None for i in range(12)}
    owner.available=[AlgorithmSpec('fake','fake','1','multi_source_fusion','available',(),())]
    owner.nodes=lambda sink:['s0'];owner.active_states=lambda:list(owner.states)
    owner.options=lambda spec,parents:[];owner.exhausted=lambda:False
    assert next(owner.neighbourhood('s0','structure')) is None


def test_public_search_and_new_year_deployment_without_reference(monkeypatch,tmp_path):
    from test_raster_gap_application import registry,contract
    from ecodata.algorithms import platform_search as api
    registry(monkeypatch,tmp_path);c=context();declaration=contract()
    declaration.update(field='v',valid_range=[-1.,1.]);declaration['issues']=declaration['issues'][:1]
    sources=[]
    for i,s in enumerate(c.sources):
        ds=s.dataset.copy();ds.attrs.update(crs='EPSG:4326',source_id='sensor'+str(i))
        sources.append(api.SearchInput(str(i)+'.nc','primary' if i==0 else 'background',ds.to_netcdf()))
    request=dict(task='multisource',variable='v',unit='1',semantic_type='continuous',
        target_lat=list(c.target_lat),target_lon=list(c.target_lon),target_times=c.parameters['raster_target_times'],
        parameters=c.parameters,application_contract=declaration,scoring=dict(registry_version='grid-test-v1'),
        mutation_control='random',response_scoring=dict(enabled=False))
    ref=source(['2016-01-03'],3,[.8]).v;ref.attrs['split']='search_validation'
    result=api.submit_search(sources,request,None,tmp_path/'search',mode='PortraitMutation',validation=api.SearchValidation(ref,'v'),
        budget=dict(objective_limit=90,proposal_limit=4000,wall_seconds=40),seed=11)
    assert result['status']=='PASS' and result['evolution_used']
    import json
    records=json.loads((tmp_path/'search/graphs.json').read_text())
    graph=dict(records=records,key=result['best']['key'])
    deployed=[]
    for item in sources:
        import io
        with xr.open_dataset(io.BytesIO(item.content)) as ds:new=ds.load().assign_coords(time=ds.time+pd.Timedelta(days=366))
        deployed.append(api.SearchInput(item.filename,item.role,new.to_netcdf()))
    targets=[str(pd.Timestamp(t)+pd.Timedelta(days=366)) for t in request['target_times']]
    req=dict(request,target_times=targets,parameters=dict(c.parameters,raster_target_times=targets))
    receipt=api.execute_frozen_graph(deployed,req,graph,tmp_path/'deploy')
    assert not receipt['reference_used'] and not receipt['selection_performed']
    with xr.open_dataset(receipt['product']) as ds:assert int(ds.time.dt.year.min())==2017
