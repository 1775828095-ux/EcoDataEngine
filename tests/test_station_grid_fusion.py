from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.station_grid_fusion import StationGridFusion,sample_grid
from ecodata.algorithms.series_fusion import SeriesMissingFallback


def fixture(method='affine',rectilinear=False):
    lat,lon=25.34,-107.22
    def make(time,identity,semantics,values):
        ds=xr.Dataset(dict(precipitation=(('time','lat','lon'),np.asarray(values,float)[:,None,None])),
            coords=dict(time=time,lat=[lat],lon=[lon]),attrs=dict(crs='EPSG:4326',source_id=identity))
        ds.precipitation.attrs=dict(units='mm/day',physical_variable='precipitation_amount',physical_definition='Daily precipitation water-equivalent depth',
            spatial_semantics=semantics,temporal_semantics='accumulation',data_kind='continuous')
        ds['time_bounds']=(('time','bounds'),np.stack([time.values,(time+pd.Timedelta(days=1)).values],axis=1))
        return ds
    def pair(time):
        x=(np.arange(len(time))*3%11)+1.;offset=1. if method=='affine' else 0.
        if rectilinear:
            # Known plane; independent analytic value at the query below.
            base=make(time,'grid','cell_mean',x)
            base=base.reindex(lat=[25.,26.],lon=[-108.,-107.])
            base.precipitation.values=x[:,None,None]+np.array([0.,1.])[None,:,None]+np.array([0.,2.])[None,None,:]
            station_x=x+(lat-25.)+2*(lon+108.)
        else:
            base=make(time,'grid','cell_mean',x);station_x=x
            base.attrs.update(geometry_kind='extracted_cell_series',pixel_selection='nearest_native_cell',
                extraction_latitude=lat,extraction_longitude=lon,extraction_evidence='Synthetic documented query fixture')
        point=make(time,'station','point',2*station_x+offset)
        return point,base
    prior_point,prior_grid=pair(pd.date_range('1999-01-01','1999-12-31'))
    for ds in [prior_point,prior_grid]:ds.attrs.update(training_end='1999-12-31',split='training')
    point,grid=pair(pd.date_range('2000-01-01','2001-12-31'));truth=point.copy(deep=True)
    point.precipitation.values[11:14]=np.nan;point.precipitation.values[190:204]=np.nan
    params=dict(allow_training=True,sampling='linear' if rectilinear else 'extracted',
        prior_resources=dict(training_reference=prior_point,historical_simulation=prior_grid))
    context=AlgorithmContext((EcoDataBundle(point),EcoDataBundle(grid)),'precipitation','continuous',
        target_lat=(lat,),target_lon=(lon,),parameters=params,model_constraints=dict(nonnegative=True,unit='mm/day'))
    return context,truth


@pytest.mark.parametrize('method',['affine','ratio','monthly_ratio'])
@pytest.mark.parametrize('rectilinear',[False,True])
def test_real_sampling_prior_mapping_and_fallback_preserve_raw_support(method,rectilinear):
    ctx,truth=fixture(method,rectilinear);plugin=StationGridFusion(method)
    before=ctx.sources[1].dataset.copy(deep=True)
    assert plugin.compatibility(ctx).eligible
    result=plugin.run(ctx)
    xr.testing.assert_identical(ctx.sources[1].dataset,before)
    np.testing.assert_allclose(result.data.dataset.precipitation,truth.precipitation,rtol=1e-12,atol=1e-12)
    assert result.data.dataset.precipitation.attrs['spatial_semantics']=='point'
    assert before.precipitation.attrs['spatial_semantics']=='cell_mean'
    assert result.report['training_end']=='1999-12-31' and result.report['training_pairs']==365
    assert result.report['filled']==17 and result.report['original_grid_semantics_preserved']
    assert not SeriesMissingFallback().compatibility(replace(ctx,parameters=dict(primary_index=0))).eligible


def test_current_values_cannot_refit_prior_model_and_both_live_sources_affect_output():
    ctx,truth=fixture();plugin=StationGridFusion('affine');base=plugin.run(ctx)
    point=ctx.sources[0].dataset.copy(deep=True);grid=ctx.sources[1].dataset.copy(deep=True)
    missing=point.precipitation.isnull().values
    grid.precipitation.values[missing]+=1
    changed=plugin.run(replace(ctx,sources=(ctx.sources[0],EcoDataBundle(grid))))
    np.testing.assert_allclose((changed.data.dataset.precipitation-base.data.dataset.precipitation).values[missing],2.)
    point.precipitation.values[~missing]+=7
    changed=plugin.run(replace(ctx,sources=(EcoDataBundle(point),ctx.sources[1]),validation_reference=truth.precipitation*1e8))
    np.testing.assert_array_equal(changed.data.dataset.precipitation.values[~missing],point.precipitation.values[~missing])
    np.testing.assert_allclose(changed.data.dataset.precipitation.values[missing],base.data.dataset.precipitation.values[missing])
    assert changed.report['coefficients']==base.report['coefficients']


@pytest.mark.parametrize('problem',['query','identity','overlap','false_end','test_labels','bounds','unit','unknown_support','no_training','geometry','no_pairs'])
def test_unknown_or_inconsistent_mapping_is_not_silently_fused(problem):
    ctx,_=fixture();point=ctx.sources[0].dataset.copy(deep=True);grid=ctx.sources[1].dataset.copy(deep=True)
    p=deepcopy(ctx.parameters);history=p['prior_resources']['historical_simulation'];target=p['prior_resources']['training_reference']
    if problem=='query':grid.attrs['extraction_latitude']+=.1
    elif problem=='identity':history.attrs['source_id']='wrong-grid'
    elif problem=='overlap':target.attrs['training_end']='2000-01-01'
    elif problem=='false_end':target.attrs['training_end']='1999-01-01'
    elif problem=='test_labels':history.precipitation.attrs['split']='test'
    elif problem=='bounds':grid.time_bounds.values[:,1]+=np.timedelta64(1,'D')
    elif problem=='unit':history.precipitation.attrs['units']='mm/hour'
    elif problem=='unknown_support':grid.precipitation.attrs['spatial_semantics']='UNKNOWN'
    elif problem=='no_training':p['allow_training']=False
    elif problem=='geometry':history=history.assign_coords(lon=[-108.]);p['prior_resources']['historical_simulation']=history
    elif problem=='no_pairs':p['prior_resources'].pop('training_reference')
    c=replace(ctx,sources=(EcoDataBundle(point),EcoDataBundle(grid)),parameters=p)
    assert not StationGridFusion('affine').compatibility(c).eligible


@pytest.mark.parametrize('mode',['random','none','page','background_first','csv','multiple_grids'])
def test_public_entry_selects_real_fusion_without_hand_graph_and_never_refits_visible_targets(mode,tmp_path):
    ctx,truth=fixture();point,grid=[s.dataset for s in ctx.sources]
    items=[api.SearchInput('gauge.nc','primary',point.to_netcdf()),api.SearchInput('grid.nc','background',grid.to_netcdf())]
    if mode=='csv':
        time=pd.DatetimeIndex(grid.time.values);native=time.dayofyear<=365
        frame=pd.DataFrame(dict(year=time[native].year,yday=time[native].dayofyear))
        frame['prcp (mm/day)']=grid.precipitation.values[native,0,0]
        content=('Latitude: 25.34  Longitude: -107.22\nDaymet Software Version4\nHow to cite: Synthetic Daymet test product\n'+frame.to_csv(index=False)).encode()
        parsed,_=api.parse_inputs([api.SearchInput('native.csv','background',content)])
        ctx.parameters['prior_resources']['historical_simulation'].attrs['source_id']=parsed[0][1].attrs['source_id']
        ctx.parameters['prior_resources']['historical_simulation'].precipitation.attrs['spatial_semantics']='grid_estimate'
        items[1]=api.SearchInput('native.csv','background',content)
    for role,ds in ctx.parameters['prior_resources'].items():items.append(api.SearchInput(role+'.nc',role,ds.to_netcdf()))
    q=dict(task='multisource',variable='precipitation',unit='mm/day',semantic_type='continuous',nonnegative=True,
        allow_training=True,target_lat=[float(point.lat[0])],target_lon=[float(point.lon[0])],
        target_times=[str(t) for t in point.time.values],parameters={},
        model_profile=dict(id='swat_plus',role='weather',field='precipitation'),
        portrait_policy=dict(version='robit-era5-multisource-swatplus-v1',scenario='daily-recovery'),
        scoring=dict(registry_version='robit-daily-state-v1'))
    if mode=='background_first':items[0],items[1]=items[1],items[0]
    if mode=='multiple_grids':
        wrong=grid.copy(deep=True).assign_coords(lat=[26.34])
        wrong.attrs.update(source_id='wrong-grid',extraction_latitude=26.34)
        items.insert(1,api.SearchInput('unmatched_grid.nc','background',wrong.to_netcdf()))
    budget=dict(objective_limit=300,total_call_limit=300,probe_call_limit=30,proposal_limit=4000,wall_seconds=60)
    if mode=='none':result=api.submit_search(items,dict(q,mutation_control='none'),None,tmp_path/'run',mode='PortraitMutation',budget=budget,seed=11)
    elif mode=='page':
        from ecodata.interface.search_page import process_search_upload
        result=process_search_upload(items,dict(q,decision_policy='current_input'),None,tmp_path/'run',budget=budget,seed=11)
    else:result=api.process_current_input(items,q,tmp_path/'run',budget=budget,seed=11)
    assert result['status']=='PASS' and result['test_used_for_selection'] is False
    assert any(t['operation']=='fusion.station_grid_affine' for t in result['best']['trace'])
    if mode=='multiple_grids':
        assert any(t.get('parameters',{}).get('source')=='source2' for t in result['best']['trace'])
        assert not any(t.get('parameters',{}).get('source')=='source1' for t in result['best']['trace'])
    assert result['total_cost']['training_capable_calls']>0
    with xr.open_dataset(result['product']) as ds:np.testing.assert_allclose(ds.precipitation,truth.precipitation,atol=1e-10)
    assert result['best']['metrics']['RMSE']<1e-10


def test_regular_grid_sampler_rejects_outside_query_and_unproven_single_pixel():
    ctx,_=fixture(rectilinear=True)
    with pytest.raises(ValueError,match='outside'):sample_grid(ctx.sources[1].dataset,'precipitation',30.,-107.5,'linear')
    ctx,_=fixture();grid=ctx.sources[1].dataset.copy();grid.attrs.pop('geometry_kind')
    with pytest.raises(ValueError):sample_grid(grid,'precipitation',25.34,-107.22,'nearest')


def test_raw_daymet_reader_keeps_leap_day_but_not_invent_dec31_or_cell_centre():
    from ecodata.io.daymet import read_daymet_precipitation
    body='year,yday,prcp (mm/day)\n'+''.join(f'2000,{d},{d/100}\n' for d in range(1,366))
    raw=('Latitude: 25.34  Longitude: -107.22\nDaymet Software Version4\nHow to cite: Synthetic Daymet test product\n'+body).encode()
    ds,descriptor=api.recognise_file('official.csv',raw)
    assert descriptor.structure=='grid' and ds.sizes['time']==366
    assert ds.precipitation.sel(time='2000-02-29').item()==.6
    assert np.isnan(ds.precipitation.sel(time='2000-12-31')).all()
    assert ds.precipitation.attrs['spatial_semantics']=='grid_estimate'
    assert 'not inferred' in ds.attrs['coordinate_role']
    with pytest.raises(ValueError,match='ordered'):read_daymet_precipitation(raw.replace(b'2000,60,0.6\n',b''))


@pytest.mark.parametrize('method',['ratio','monthly_ratio'])
def test_unidentifiable_historical_ratio_is_not_replaced_with_invented_coefficient(method):
    ctx,_=fixture(method);ctx.parameters['prior_resources']['historical_simulation'].precipitation.values[:]=0
    with pytest.raises(ValueError,match='unidentifiable|identifiable'):StationGridFusion(method).run(ctx)


def test_same_support_bias_adjustment_cannot_replace_explicit_point_grid_mapping():
    from ecodata.algorithms.open_source_bias import XsdbaPlugin
    ctx,_=fixture();prior=ctx.parameters['prior_resources']
    params=dict(ctx.parameters,training_reference=prior['training_reference'].precipitation.assign_attrs(split='train'),
        historical_simulation=prior['historical_simulation'].precipitation.assign_attrs(split='train'))
    result=XsdbaPlugin('eqm').compatibility(replace(ctx,sources=(ctx.sources[1],),parameters=params))
    assert not result.eligible and any('supports require' in r for r in result.reasons)
