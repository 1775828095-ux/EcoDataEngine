"""Public geometry/lineage/measured-selection acceptance, not accuracy claims."""
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.spatial_fusion_task import SpatialTask,SpatialSearch,positions
from ecodata.fusion.point_weights import distance_weights,weighted_values


def fixture(kind='station_grid',aux=False):
    times=pd.date_range('2000-01-01',periods=10)
    xy=np.array([[0,0],[1,0],[0,1],[1,1],[.5,.5]],float)
    xy+=np.array([110.,30.])
    values=3+xy[:,0]-110+2*(xy[:,1]-30)+np.arange(10)[:,None]*.1
    obs=pd.DataFrame(dict(time=np.repeat(times,5),station_id=np.tile(list('abcde'),10),
        longitude=np.tile(xy[:,0],10),latitude=np.tile(xy[:,1],10),value=values.ravel(),
        unit='K',variable='temperature',split='train',temporal_semantics='instantaneous',crs='EPSG:4326'))
    q=dict(task='multisource',variable='temperature',unit='K',physical_variable='temperature',
        semantic_type='continuous',temporal_semantics='instantaneous',allow_training=True,processing_goal='spatial_fusion',
        search_stopping=dict(enabled=False))
    items=[api.SearchInput('obs.csv','observations',obs.to_csv(index=False).encode())]
    for index,n in enumerate([3,4]):
        lat=np.linspace(30,31,n);lon=np.linspace(110,111,n);xx,yy=np.meshgrid(lon,lat)
        field=np.array([3+(xx-110)+2*(yy-30)+t*.1+index+1 for t in range(10)])
        ds=xr.Dataset(dict(temperature=(('time','lat','lon'),field)),coords=dict(time=times,lat=lat,lon=lon),attrs=dict(crs='EPSG:4326',source_id='grid'+str(index)))
        ds.temperature.attrs.update(units='K',physical_variable='temperature',physical_definition='temperature estimate',spatial_semantics='grid_estimate',temporal_semantics='instantaneous',data_kind='continuous')
        if kind!='stations':items.append(api.SearchInput(f'grid{index}.nc','primary' if index==0 else 'background',ds.to_netcdf()))
    target=xr.Dataset(coords=dict(point=np.arange(3),lat=('point',[30.2,30.4,30.8]),lon=('point',[110.2,110.7,110.8])),attrs=dict(crs='EPSG:4326'))
    items.append(api.SearchInput('target.nc','target',target.to_netcdf()))
    if kind=='grids':items=[i for i in items if i.role!='observations']
    if aux:
        ds=xr.Dataset(dict(elevation=(('lat','lon'),xx+2*yy)),coords=dict(lat=lat,lon=lon),attrs=dict(crs='EPSG:4326',source_id='dem'))
        ds.elevation.attrs.update(units='m',spatial_semantics='grid_estimate')
        items.append(api.SearchInput('dem.nc','auxiliary',ds.to_netcdf()))
    return items,q,obs


BUDGET=dict(objective_limit=130,total_call_limit=180,proposal_limit=100,wall_seconds=45)


@pytest.mark.parametrize('kind,aux',[('stations',False),('station_grid',False),('grids',False),('station_grid',True)])
def test_all_spatial_relationships_use_public_entry_and_real_dag(tmp_path,kind,aux):
    items,q,_=fixture(kind,aux)
    result=api.process_current_input(items,q,tmp_path/kind,seed=29,budget=BUDGET)
    assert result['status']=='PASS' and result['optimizer']['backend']=='random'
    assert result['optimizer']['shared_driver'].endswith('random_graph_search')
    assert result['total_cost']['phases']['deploy']>0
    assert result['self_validation_protocol']['self_gridded_reference']==(kind=='grids')
    for fold in result['self_validation_protocol']['folds']:
        assert not set(fold['held_station_ids'])&set(fold['training_station_ids'])
    with xr.open_dataset(result['product']) as ds:
        assert ds.temperature.dims==('time','point') and ds.sizes['point']==3
        assert np.isfinite(ds.temperature).all() and ds.temperature.attrs['units']=='K'
    rows=json.loads((tmp_path/kind/'candidate_results.json').read_text())
    assert all('traces' in row for row in rows if row['status']=='PASS')
    assert result['confirmation']['used_for_selection'] is False


def test_raw_point_dataset_is_automatically_interpreted_as_multiple_stations(tmp_path):
    items,q,obs=fixture('stations')
    xy=obs.drop_duplicates('station_id')
    ds=xr.Dataset(dict(temperature=(('time','station'),obs.value.to_numpy().reshape(10,5))),
        coords=dict(time=sorted(obs.time.unique()),station=list('abcde'),lat=('station',xy.latitude),lon=('station',xy.longitude)),
        attrs=dict(crs='EPSG:4326',source_id='observed'))
    ds.temperature.attrs.update(units='K',physical_variable='temperature',physical_definition='temperature estimate',spatial_semantics='point',temporal_semantics='instantaneous')
    items=[i for i in items if i.role!='observations']+[api.SearchInput('stations.nc','primary',ds.to_netcdf())]
    result=api.process_current_input(items,q,tmp_path/'run',seed=11,budget=BUDGET)
    assert result['status']=='PASS'
    with xr.open_dataset(result['product']) as out:assert out.sizes['point']==3


def test_held_station_values_never_enter_its_fitting_cohort(tmp_path):
    items,q,obs=fixture();parsed,audit=api.parse_inputs(items)
    first=SpatialSearch(SpatialTask.build(parsed,q,audit),tmp_path,BUDGET,11)
    held=first.folds[0]['held'];obs.loc[obs.station_id.isin(held),'value']+=1000
    items[0]=api.SearchInput('obs.csv','observations',obs.to_csv(index=False).encode())
    parsed,audit=api.parse_inputs(items);second=SpatialSearch(SpatialTask.build(parsed,q,audit),tmp_path,BUDGET,11)
    np.testing.assert_array_equal(first.folds[0]['cohort']['y'],second.folds[0]['cohort']['y'])
    np.testing.assert_array_equal(first.folds[0]['cohort']['sources'],second.folds[0]['cohort']['sources'])
    assert not np.array_equal(first.folds[0]['reference'],second.folds[0]['reference'])


def test_live_station_and_background_really_enter_spatial_residual(tmp_path):
    items,q,obs=fixture();parsed,audit=api.parse_inputs(items);task=SpatialTask.build(parsed,q,audit)
    engine=SpatialSearch(task,tmp_path,BUDGET,11)
    key=engine.intern('spatial.residual_idw',[engine.leaves[0]],dict(power=2.))
    nodes=engine.nodes(key)
    assert any(n.operation=='spatial.observations' for n in nodes)
    assert 'observations' in next(n for n in nodes if n.id==key).inputs
    base,_=engine.product(key)
    task.observations.loc[task.observations.station_id=='a','value']+=3
    changed,_=engine.product(key)
    assert float(abs(changed.temperature-base.temperature).max())>.1
    task.fields[0].temperature.values[:,1,1]+=5
    changed2,_=engine.product(key)
    assert float(abs(changed2.temperature-changed.temperature).max())>.1


@pytest.mark.parametrize('bad',['unit','test','time','unknown_support','accumulation','physical_variable','hard_model'])
def test_invalid_semantics_and_test_labels_block_before_selection(tmp_path,bad):
    items,q,obs=fixture()
    if bad in {'unit','test','time','physical_variable','accumulation'}:
        if bad=='unit':obs['unit']='mm/day'
        if bad=='test':obs['split']='test'
        if bad=='time':obs['temporal_semantics']='accumulation'
        if bad=='physical_variable':obs['variable']='precipitation_amount'
        if bad=='accumulation':q['temporal_semantics']='accumulation'
        items[0]=api.SearchInput('obs.csv','observations',obs.to_csv(index=False).encode())
    elif bad=='unknown_support':
        ds=xr.open_dataset(__import__('io').BytesIO(items[1].content)).load();ds.temperature.attrs['spatial_semantics']='UNKNOWN'
        items[1]=api.SearchInput('bad.nc','primary',ds.to_netcdf())
    else:q['model_profile']=dict(id='swat_plus',role='weather',field='precipitation')
    with pytest.raises(ValueError):api.process_current_input(items,q,tmp_path/bad,budget=BUDGET)


def test_reproducible_page_and_file_clients(tmp_path):
    from ecodata.interface.search_page import process_search_upload
    items,q,_=fixture();q['decision_policy']='current_input'
    a=process_search_upload(items,q,None,tmp_path/'page',seed=11,budget=BUDGET)
    files=[]
    for i,item in enumerate(items):
        p=tmp_path/f'{i}_{item.filename}';p.write_bytes(item.content);files.append(dict(path=str(p),role=item.role))
    b=api.submit_file_request(dict(request=q,inputs=files,budget=BUDGET,seed=11),tmp_path/'file')
    assert a['selected_graph']==b['selected_graph']
    with xr.open_dataset(a['product']) as x,xr.open_dataset(b['product']) as y:xr.testing.assert_equal(x,y)


def test_idw_missingness_and_paired_geometry():
    xy=np.array([[110,30],[111,30],[110,31]])
    w=distance_weights(xy,xy)
    y=np.array([[1,2,3],[np.nan,4,5]])
    actual=weighted_values(y,w)
    np.testing.assert_allclose(actual[0],[1,2,3])
    assert np.isfinite(actual).all() and actual.shape==(2,3)


def test_batched_affine_matches_existing_least_squares():
    from ecodata.algorithms.quality_aware import fit_affine_frames,fit_affine_source
    rng=np.random.default_rng(41);x=rng.normal(size=(7,6));y=2*x+1+rng.normal(size=x.shape)*.01
    x[0,0]=np.nan
    slope,intercept,failed=fit_affine_frames(x,y)
    assert failed==0
    for index in range(7):np.testing.assert_allclose([slope[index,0],intercept[index,0]],fit_affine_source(x[index],y[index]),rtol=1e-12)


def test_fixed_control_is_separate_from_automatic_search(tmp_path):
    items,q,_=fixture()
    graph=dict(key='idw',records=dict(idw=dict(operation='spatial.idw',parents=[],parameters=dict(power=2.))))
    result=api.execute_frozen_graph(items,q,graph,tmp_path/'fixed')
    assert result['status']=='PASS' and not result['selection_performed']
    with pytest.raises(ValueError,match='target|prescribed'):
        api.process_current_input(items,dict(q,fixed_graph=graph),tmp_path/'auto',budget=BUDGET)


def test_auxiliary_is_fitted_response_not_added_as_target(tmp_path):
    items,q,obs=fixture('stations',aux=True);parsed,audit=api.parse_inputs(items)
    task=SpatialTask.build(parsed,q,audit);engine=SpatialSearch(task,tmp_path,BUDGET,11)
    key=next(k for k in engine.leaves if engine.records[k]['operation']=='spatial.aux_affine')
    result,_=engine.product(key)
    expected=3+(result.lon.values-110)+2*(result.lat.values-30)+np.arange(10)[:,None]*.1
    np.testing.assert_allclose(result.temperature,expected,atol=1e-10)
    assert result.temperature.attrs['units']=='K'


def test_grid_reference_alias_cannot_bypass_spatial_holdout(tmp_path):
    items,q,_=fixture('grids')
    ds=xr.open_dataset(__import__('io').BytesIO(items[0].content)).load()
    ds.attrs.update(source_id='derived',derived_from=json.dumps(['grid0']))
    items.append(api.SearchInput('derived.nc','background',ds.to_netcdf()))
    with pytest.raises(ValueError,match='alias'):
        api.process_current_input(items,q,tmp_path/'blocked',budget=BUDGET)


def test_cell_mean_not_silently_treated_as_point_resolution_change(tmp_path):
    items,q,_=fixture()
    ds=xr.open_dataset(__import__('io').BytesIO(items[1].content)).load()
    ds.temperature.attrs['spatial_semantics']='cell_mean'
    items[1]=api.SearchInput('cells.nc','primary',ds.to_netcdf())
    with pytest.raises(ValueError,match='conservative'):
        api.process_current_input(items,q,tmp_path/'blocked',budget=BUDGET)


def test_spatial_search_overrun_still_executes_selected_graph(tmp_path,monkeypatch):
    items,q,_=fixture();original=SpatialSearch.propose
    def expire(self,*args,**kwargs):
        before=len(self.account.events);result=original(self,*args,**kwargs)
        if len(self.account.events)>before:self.account.started-=1000
        return result
    monkeypatch.setattr(SpatialSearch,'propose',expire)
    result=api.process_current_input(items,q,tmp_path/'run',budget=BUDGET)
    assert result['status']=='PASS' and result['total_cost']['phases']['deploy']>0
    assert result['total_cost']['service_seconds']>1000
    with xr.open_dataset(result['product']) as out:assert np.isfinite(out.temperature).all()


def test_instantaneous_covariate_time_sampling_has_no_extrapolation():
    from ecodata.algorithms.spatial_fusion_task import sample
    items,q,_=fixture();parsed,_=api.parse_inputs(items);ds=next(ds for role,ds in parsed if role=='primary').isel(time=[0,2,4,6,8])
    dates=pd.DatetimeIndex(['2000-01-02','2000-01-04','2000-01-10'])
    result=sample(ds,'temperature',dates,np.array([[110.5,30.5]]))
    np.testing.assert_allclose(result[:2,0],[5.6,5.8]);assert np.isnan(result[2,0])
    ds.temperature.attrs['temporal_semantics']='accumulation'
    with pytest.raises(ValueError,match='interval'):
        sample(ds,'temperature',dates,np.array([[110.5,30.5]]))


def test_fixed_bank_scores_real_controls_without_selecting_or_seeding(tmp_path):
    items,q,_=fixture()
    graphs=dict(idw=dict(key='x',records=dict(x=dict(operation='spatial.idw',parents=[],parameters=dict(power=2.)))),
        background=dict(key='x',records=dict(x=dict(operation='spatial.source',parents=[],parameters=dict(index=0)))))
    result=api.evaluate_graph_bank(items,q,graphs,tmp_path/'bank',budget=BUDGET)
    assert result['product'] is None and not result['selection_performed']
    assert result['control_keys']['idw']!=result['control_keys']['background']
    assert all(r['status']=='PASS' and r['metrics']['count']==35 for r in result['controls'].values())


@pytest.mark.parametrize('conflict',[False,True])
def test_same_product_pieces_mosaic_with_conflict_rejection(conflict):
    from ecodata.algorithms.spatial_fusion_task import mosaic_fields
    items,q,_=fixture();parsed,_=api.parse_inputs(items);ds=next(ds for role,ds in parsed if role=='primary')
    # Two overlapping subsets of the exact same supplied product.
    a=ds.isel(lon=slice(0,2)).copy(deep=True);b=ds.isel(lon=slice(1,3)).copy(deep=True)
    for index,d in enumerate([a,b]):d.attrs.update(source_id=str(index),mosaic_group='known_product',source_product='test-v1',processing_level='raw',mosaic_evidence='fixture parts from the same known field')
    if conflict:
        b.temperature.values[:,:,0]+=1
        with pytest.raises(ValueError,match='overlap conflicts'):mosaic_fields([a,b],'temperature',pd.DatetimeIndex(ds.time.values))
    else:
        fields,relations=mosaic_fields([a,b],'temperature',pd.DatetimeIndex(ds.time.values))
        assert len(fields)==1 and fields[0].sizes['point']==9 and relations[0]['identical_overlap_locations']==3
        from ecodata.algorithms.spatial_fusion_task import sample
        xy,_,_=positions(ds)
        np.testing.assert_array_equal(sample(fields[0],'temperature',pd.DatetimeIndex(ds.time.values),xy),ds.temperature.values.reshape(10,9))


def test_unknown_product_pieces_are_not_mosaicked():
    from ecodata.algorithms.spatial_fusion_task import mosaic_fields
    items,q,_=fixture();parsed,_=api.parse_inputs(items);fields=[ds for role,ds in parsed if role in {'primary','background'}]
    result,relations=mosaic_fields(fields,'temperature',pd.DatetimeIndex(fields[0].time.values))
    assert len(result)==2 and not relations
