from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.series_fusion import SeriesMissingFallback,PriorSeriesCalibration
from ecodata.algorithms.autonomous_field import measured
from ecodata.algorithms.daily_reconstruction import DailyRecovery
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms import platform_search as api


def data(values,source='station',times=None):
    ds=xr.Dataset(dict(precipitation=(('time','lat','lon'),np.array(values,float)[:,None,None])),
        coords=dict(time=pd.date_range('2000-01-01',periods=len(values)) if times is None else times,lat=[11.6],lon=[37.36]),
        attrs=dict(crs='EPSG:4326',source_id=source))
    ds.precipitation.attrs=dict(units='mm/day',physical_variable='precipitation_amount',
        physical_definition='Daily site precipitation estimate',spatial_semantics='point',temporal_semantics='accumulation',data_kind='continuous')
    return ds


def context(*datasets,**params):
    return AlgorithmContext(tuple(EcoDataBundle(d) for d in datasets),'precipitation','continuous',
        target_lat=(11.6,),target_lon=(37.36,),parameters=params,model_constraints=dict(unit='mm/day',nonnegative=True))


def test_fallback_retains_observations_and_uses_only_support_at_missing():
    plugin=SeriesMissingFallback();c=context(data([1,np.nan,3]),data([7,8,9]),primary_index=0)
    output=plugin.run(c).data.dataset.precipitation.values[:,0,0]
    np.testing.assert_array_equal(output,[1,8,3])
    bad=c.sources[1].dataset.copy(deep=True);bad.precipitation.attrs['spatial_semantics']='cell_mean'
    assert not plugin.compatibility(replace(c,sources=(c.sources[0],EcoDataBundle(bad)))).eligible


def test_prior_calibration_binds_source_time_and_distribution():
    s=xr.Dataset(dict(affine_slope=2.,affine_intercept=-1.),attrs=dict(training_end='1999-12-31',calibration_source_id='era5'))
    plugin=PriorSeriesCalibration('affine');c=context(data([0,2,4],'era5'),fitted_state=s)
    np.testing.assert_array_equal(plugin.run(c).data.dataset.precipitation.values[:,0,0],[0,3,7])
    assert not plugin.compatibility(context(data([1,2,3],'station'),fitted_state=s)).eligible
    corrected=plugin.run(c).data
    assert not plugin.compatibility(replace(c,sources=(corrected,))).eligible
    mixed=data([1,2,3],'era5');mixed.attrs['fusion_method']='declared_primary_weighted_mean'
    assert not plugin.compatibility(context(mixed,fitted_state=s)).eligible
    overlap=s.copy();overlap.attrs['training_end']='2000-01-01'
    assert not plugin.compatibility(replace(c,parameters=dict(fitted_state=overlap))).eligible


def test_singleton_metric_is_year_rmse_not_average_daily_absolute_error():
    dates=pd.to_datetime(['2000-01-01','2000-01-02','2001-01-01','2001-01-02'])
    ref=data([0,0,0,0],times=dates).precipitation
    output=data([0,4,0,0],times=dates)
    m=measured(output,ref,context(output,objective_scope='calendar_year_RMSE'))
    assert m['objective']==pytest.approx(np.sqrt(8)/2)
    assert m['RMSE']==2 and m['MAE']==1 and m['gradient_RMSE'] is None


def test_public_entry_preserves_visible_values_and_cannot_restore_hidden_truth(tmp_path,monkeypatch):
    reg=AlgorithmRegistry();reg.register(DailyRecovery('linear'));reg.register(SeriesMissingFallback())
    monkeypatch.setattr(api,'platform_registry',lambda:reg)
    station=data([1,np.nan,3,np.nan,5]);support=data([9]*5,'era5')
    labels=data([2,4],times=station.time.values[[1,3]]).precipitation;labels.attrs['split']='search_validation'
    req=dict(task='multisource',variable='precipitation',unit='mm/day',semantic_type='continuous',nonnegative=True,
        target_lat=[11.6],target_lon=[37.36],target_times=[str(t) for t in station.time.values],
        parameters=dict(objective_scope='calendar_year_RMSE'),protect_observed_source='station')
    result=api.submit_search([api.SearchInput('station.nc','primary',station.to_netcdf()),
        api.SearchInput('support.nc','background',support.to_netcdf())],req,None,tmp_path,mode='AutonomousField',
        validation=api.SearchValidation(labels,'precipitation'),budget=dict(objective_limit=20,proposal_limit=100,wall_seconds=10))
    with xr.open_dataset(result['product']) as f:np.testing.assert_array_equal(f.precipitation.values[:,0,0],[1,2,3,4,5])
    assert result['best']['metrics']['objective']==0


def test_named_axis_normalization_does_not_reject_valid_correction_output():
    from ecodata.algorithms.autonomous_decision import Executor
    from ecodata.algorithms.contracts import AlgorithmResult,AlgorithmSpec,Compatibility
    class Reordered:
        spec=AlgorithmSpec('reordered','reordered','1','test','available',('series',),('continuous',))
        def compatibility(self,c):return Compatibility(True,())
        def run(self,c):
            ds=c.sources[0].dataset.copy();ds['precipitation']=ds.precipitation.transpose('lat','lon','time')
            return AlgorithmResult(EcoDataBundle(ds),self.spec.id,{})
    reg=AlgorithmRegistry();reg.register(Reordered());source=data([1,2,3])
    engine=Executor(context(source),reg)
    records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        g0=dict(operation='reordered',parents=['s0'],parameters={}))
    output,_=engine.execute(records,'g0')
    assert output.precipitation.dims==('time','lat','lon')
    np.testing.assert_array_equal(output.precipitation,source.precipitation)


def test_point_series_does_not_create_fake_spatial_frame_adapters():
    from ecodata.algorithms.autonomous_field import field_registry
    reg=AlgorithmRegistry();reg.register(DailyRecovery('linear'))
    field_registry(reg,context(data([1,2,3])))
    assert [s.id for s in reg.specs()]==['reconstruction.daily_linear']


def test_xsdba_training_pair_cannot_be_applied_to_wrong_source():
    from ecodata.algorithms.open_source_bias import XsdbaPlugin
    training=data(np.arange(50),times=pd.date_range('1990-01-01',periods=50)).precipitation
    training.attrs['split']='train'
    c=context(data([1,2,3],'station'),training_reference=training,historical_simulation=training,
        calibration_source_id='era5')
    assert any('source/distribution' in reason for reason in XsdbaPlugin('eqm').compatibility(c).reasons)


def bounded(values):
    ds=data(values);dates=pd.DatetimeIndex(ds.time.values)
    ds['time_bounds']=(('time','bounds'),np.column_stack([dates.values,(dates+pd.Timedelta(days=1)).values]))
    return ds


def test_identity_interval_resampling_preserves_nan_and_avoids_dense_overlap(monkeypatch):
    from ecodata.temporal import semantic
    source=bounded([1,np.nan,3,4])
    monkeypatch.setattr(semantic,'_overlap_seconds',lambda *args:(_ for _ in ()).throw(AssertionError('Dense identity overlap')))
    out=semantic.SemanticTimeResampler(semantic.SemanticTimeConfig('1D','precipitation')).apply(EcoDataBundle(source)).data.dataset
    np.testing.assert_array_equal(out.precipitation.values,source.precipitation.values)
    np.testing.assert_array_equal(out.time.values,source.time_bounds.values[:,1])


def test_nonoverlapping_nan_does_not_contaminate_other_accumulation_bins():
    from ecodata.temporal.semantic import SemanticTimeResampler,SemanticTimeConfig
    out=SemanticTimeResampler(SemanticTimeConfig('2D','precipitation')).apply(EcoDataBundle(bounded([1,np.nan,3,4]))).data.dataset
    np.testing.assert_array_equal(out.precipitation.values[:,0,0],[np.nan,7])


def test_graph_executor_aligns_accumulated_values_by_interval_not_end_label():
    from ecodata.algorithms.autonomous_decision import Executor
    from ecodata.algorithms.plugins import TemporalSemanticPlugin
    source=bounded([1,2,3,4]);reg=AlgorithmRegistry();reg.register(TemporalSemanticPlugin())
    e=Executor(context(source,target_time_frequency='1D'),reg)
    records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        g0=dict(operation='temporal.semantic_resample',parents=['s0'],parameters={}))
    out,_=e.execute(records,'g0')
    np.testing.assert_array_equal(out.time.values,source.time.values)
    np.testing.assert_array_equal(out.precipitation,source.precipitation)
