from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_decision import Executor
from ecodata.algorithms.field_mutation import GraphMutation
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.daily_reconstruction import DailyRecovery
from ecodata.algorithms.series_fusion import SeriesMissingFallback,SeriesNonnegative,PriorSeriesCalibration
from ecodata.algorithms.composition_decision import PrimaryWeightedMean
from ecodata.algorithms.scientific_state import signature
from ecodata.algorithms.search_accounting import SearchAccount
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility
from test_series_fusion import data,context
from test_field_mutation import model,engine


def response_engine(folder,weights=None):
    folder.mkdir(exist_ok=True)
    times=pd.date_range('2001-01-01','2002-12-31')
    a=data(np.full(len(times),10.),times=times)
    a.precipitation.values[30:33]=np.nan
    b=data(11+np.where(np.arange(len(times))%2, .5,-.5),source='other',times=times)
    state=xr.Dataset(dict(scale=4.,affine_slope=4.,affine_intercept=-34.,month_mean=('month',np.full(12,10.))),
                     attrs=dict(training_end='1999-12-31',calibration_source_id='other'))
    c=context(a,b,fitted_state=state,require_complete_output=True)
    registry=AlgorithmRegistry()
    for plugin in [SeriesMissingFallback(),SeriesNonnegative(),PriorSeriesCalibration('affine'),DailyRecovery('linear'),DailyRecovery('month_mean'),PrimaryWeightedMean()]:registry.register(plugin)
    reference=data([10.],times=[times[31]]).precipitation
    request=dict(response_scoring=dict(period=['2001-01-01','2002-12-31'],weights=weights or dict(data=.5,model=.5,collaboration=0.),evidence='Synthetic mechanism test priorities'))
    return GraphMutation(c,registry,request,model('expected_error'),reference,folder,
                         dict(objective_limit=100,probe_call_limit=60,proposal_limit=4000,wall_seconds=30),11)


def test_equal_error_growth_rejected_and_validation_not_removed(tmp_path):
    e=engine(tmp_path)
    try:
        e.initialize()
        child=e.intern('fusion.series_missing_fallback',['s0','s1'],dict(primary_index=0))
        e.execute(child,'mutation','s1')
        assert e.evaluated[child]['metrics']['objective']==e.evaluated['s1']['metrics']['objective']
        assert not e.accept('s1',child)
        assert e.evaluated[child]['quality']['status']!='BLOCK'
    finally:e.close()


def test_cached_unknown_response_keeps_valid_unknown_interval_without_retry(tmp_path):
    e=response_engine(tmp_path)
    try:
        first=e.matching('s0');calls=len(e.account.events)
        assert e.response.responses['s0']['status']=='UNKNOWN'
        second=e.matching('s0')
        assert first['score_interval']==second['score_interval']
        assert len(e.account.events)==calls
        assert e.response.score('s0') is None
    finally:e.close()


def test_full_scientific_state_identity_preserves_physical_and_fit_contracts():
    a=data([1,np.nan,3]);b=a.copy(deep=True);b.attrs['processing_history']='extra logging'
    assert signature(a)==signature(b)
    for change in ['calibration_signature','source_id']:
        c=a.copy(deep=True);c.attrs[change]='changed'
        assert signature(a)!=signature(c)
    c=a.copy(deep=True);c.precipitation.attrs['units']='kg'
    assert signature(a)!=signature(c)
    c=a.copy(deep=True);c.precipitation.values[0]=2
    assert signature(a)!=signature(c)
    c=a.copy(deep=True);c.precipitation.values[1]=0
    assert signature(a)!=signature(c)


def test_equivalent_different_graphs_share_expansion_state(tmp_path):
    e=engine(tmp_path)
    try:
        e.initialize()
        # Projection of a nonnegative source is identical; direct executor check
        # verifies post-execution dedup independently of the preflight shortcut.
        child=e.intern('constraint.series_nonnegative',['s1'],{})
        r=e.execute(child,'mutation','s1')
        assert r['dominated'] and r['equivalent_to']=='s1'
        assert child not in e.active_states()
    finally:e.close()


def test_multisource_response_has_real_distinction_and_no_root_truth_leak(tmp_path):
    e=response_engine(tmp_path)
    try:
        assert e.response.executor is not None
        ref=e.response.reference
        assert np.isnan(e.response.context.sources[0].dataset.precipitation.sel(time=ref.time)).all()
        assert np.isfinite(e.context.sources[0].dataset.precipitation.sel(time=ref.time)).all()
        # A production-state cache containing visible probe truth must not be
        # consumed by the separate masked-root response executor.
        e.states['s0'].precipitation.loc[dict(time=ref.time)]=999.
        fallback=e.intern('fusion.series_missing_fallback',['s0','s1'],dict(primary_index=0))
        affine=e.intern('bias.prior_series_affine',['s1'],{})
        a=e.matching(fallback);b=e.matching(affine)
        assert a['status']==b['status']=='MEASURED_VISIBLE_MASK'
        assert a['score_interval']!=b['score_interval']
        assert max(a['predicted'])<20
        assert e.account.counts['probe']>0
        assert all(r['phase']=='probe' for r in e.account.events)
    finally:e.close()


def test_explicit_model_preference_changes_measured_algorithm_order(tmp_path):
    results=[]
    for name,weights in [('accuracy',dict(data=1.,model=0.,collaboration=0.)),('volume',dict(data=0.,model=1.,collaboration=0.))]:
        e=response_engine(tmp_path/name,weights)
        try:
            affine=e.intern('bias.prior_series_affine',['s1'],{})
            raw=e.matching('s1');cal=e.matching(affine)
            results.append(raw['score_interval'][0]>cal['score_interval'][0])
        finally:e.close()
    assert results==[True,False]


def test_joint_destroy_removes_subgraph_and_repairs_actual_sources(tmp_path):
    e=response_engine(tmp_path)
    try:
        e.initialize()
        one=e.intern('reconstruction.daily_linear',['s0'],{})
        two=e.intern('reconstruction.daily_month_mean',[one],{})
        e.execute(two,'mechanism')
        plan=e.destroy(two,np.random.default_rng(11))
        assert len(plan['removed_nodes'])==2
        assert '__repair_hole__' in plan['partial_graph']
        candidates=[]
        for row in e.neighbourhood(two,'joint',plan):
            if row:
                candidates.append(row)
                if len(candidates)>=2:break
        assert candidates
        child=candidates[0]['child']
        assert e.origin(child)=={'source0','source1'}
        assert all(e.records[n]['operation']!='REPAIR_REQUIRED' for n in e.nodes(child))
        result=e.execute(child,'mutation',two)
        assert result['status']=='ELIGIBLE_OUTPUT'
    finally:e.close()


def test_shared_call_budget_includes_probe_failures_and_cache(tmp_path):
    registry=AlgorithmRegistry();registry.register(SeriesNonnegative())
    c=context(data([1,2,3]));account=SearchAccount(2,30,probe_calls=1)
    records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0')),
             'g':dict(operation='constraint.series_nonnegative',parents=['s0'],parameters={})}
    probe=Executor(c,registry,account=account,phase='probe');search=Executor(c,registry,account=account)
    probe.execute(records,'g');probe.execute(records,'g')
    assert len(account.events)==1
    search.execute(records,'g');assert len(account.events)==2
    third=Executor(c,registry,account=account,phase='deploy')
    with pytest.raises(Exception,match='budget exhausted'):third.execute(records,'g')
    assert len(account.events)==2 and account.counts=={'probe':1,'search':1}


def test_failed_kernel_is_charged_and_trace_retained():
    class Failing:
        spec=AlgorithmSpec('test.failure','Deliberate unit-test failure','1','test','available',(),('continuous',))
        def compatibility(self,c):return Compatibility(True,())
        def run(self,c):raise ValueError('intentional kernel failure')
    registry=AlgorithmRegistry();registry.register(Failing());account=SearchAccount(1,30,probe_calls=1)
    records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0')),
             'g':dict(operation='test.failure',parents=['s0'],parameters={})}
    e=Executor(context(data([1,2,3])),registry,account=account,phase='probe')
    with pytest.raises(Exception,match='intentional kernel failure') as exc:e.execute(records,'g')
    assert account.events[0]['status']=='FAIL' and account.counts['probe']==1
    assert getattr(exc.value,'trace',None)


def test_downstream_prior_source_failure_rejected_before_new_kernel(tmp_path):
    e=response_engine(tmp_path)
    try:
        one=e.intern('bias.prior_series_affine',['s1'],{})
        two=e.intern('bias.prior_series_affine',[one],{})
        assert e.downstream_contract(two)=='Downstream calibration has no unmodified training distribution'
        wrong=e.intern('bias.prior_series_affine',['s0'],{})
        assert e.downstream_contract(wrong)=='Downstream calibration source/state mismatch'
        assert len(e.account.events)==0
    finally:e.close()


def test_fit_namespace_separates_otherwise_identical_scientific_states(tmp_path):
    a=response_engine(tmp_path/'a');b=response_engine(tmp_path/'b')
    try:
        c=replace(b.context,parameters=dict(b.context.parameters,fitted_state=b.context.parameters['fitted_state'].assign(scale=8.)))
        registry=b.registry
        b.close()
        folder=tmp_path/'c';folder.mkdir()
        c_engine=GraphMutation(c,registry,b.request,b.field['model'],b.reference,folder,b.budget,11)
        try:
            assert a.state_keys['s0']!=c_engine.state_keys['s0']
        finally:c_engine.close()
    finally:
        a.close()


def test_fixed_control_uses_public_executor_quality_and_total_budget(tmp_path):
    a=data([1,np.nan,3,5,2]);b=data([2,2,2,2,2],source='other')
    ref=a.precipitation.isel(time=[1]).copy();ref.values[:]=4;ref.attrs['split']='search_validation'
    records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0')),
             's1':dict(operation='source',parents=[],parameters=dict(source='source1')),
             'g':dict(operation='fusion.series_missing_fallback',parents=['s0','s1'],parameters=dict(primary_index=0))}
    request=dict(task='multisource',variable='precipitation',unit='mm/day',semantic_type='continuous',nonnegative=True,
       target_lat=[11.6],target_lon=[37.36],target_times=[str(t) for t in pd.DatetimeIndex(a.time.values)],
       parameters=dict(require_complete_output=True),protect_observed_source='station',
       model_profile=dict(id='swat_plus',role='weather',field='precipitation'),
       portrait_policy=dict(version='robit-era5-multisource-swatplus-v1',scenario='daily-recovery'),
       scoring=dict(registry_version='robit-daily-state-v1'),response_scoring=dict(enabled=False),
       mutation_control='fixed',fixed_graph=dict(records=records,key='g'))
    inputs=[api.SearchInput('a.nc','primary',a.to_netcdf()),api.SearchInput('b.nc','background',b.to_netcdf())]
    r=api.submit_search(inputs,request,None,tmp_path,mode='PortraitMutation',
       validation=api.SearchValidation(ref,'precipitation'),budget=dict(objective_limit=1,proposal_limit=5,wall_seconds=30))
    assert r['status']=='PASS' and r['stop_reason']=='FIXED_GRAPH_EXECUTED'
    assert r['total_cost']['used']==1 and r['best']['quality']['status']!='BLOCK'
    with xr.open_dataset(r['product']) as output:np.testing.assert_allclose(output.precipitation.values.ravel(),[1,2,3,5,2])


def test_unscored_shorter_representative_promoted_without_extra_main_kernel(tmp_path):
    e=engine(tmp_path)
    try:
        e.execute('s0','initial');e.execute('s1','initial')
        inner=e.intern('fusion.series_missing_fallback',['s0','s1'],dict(primary_index=0))
        outer=e.intern('constraint.series_nonnegative',[inner],{})
        row=e.execute(outer,'mutation','s1')
        assert row['promoted_representative']==inner
        assert e.evaluated[inner]['origin']=='scientific_state_alias'
        assert e.evaluated[inner]['metrics']==row['metrics']
        assert e.evaluated[inner]['total_new_calls']==0
        assert len(e.account.events)==2
    finally:e.close()
