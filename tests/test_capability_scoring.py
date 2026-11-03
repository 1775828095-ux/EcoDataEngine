"""Frozen functional cases. Synthetic label fixtures are NOT empirical evidence."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pytest,xarray as xr
from copy import deepcopy
from dataclasses import replace
from ecodata.algorithms import capability_scoring as cs
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor,RequirementMapper
from ecodata.algorithms.platform_search import make_context,platform_registry,bind_registry
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.golem_search import semantic_proposal_weights
from test_platform_integration import grid,request,prepare
from test_local_problem import prepare as branches,sources,canonical_candidates


def evidence(group='r0',i=0):
    context=dict(task_family='spatial_raster',variable='aod',unit='1',metric='RMSE',metric_kind='reference_error',
        capability='coarse_scale_consistency',implementation='raster_task.coarse_consistency',base_algorithm='raster_task.spatial.bilinear',support='cell_mean')
    a=dict(input_fingerprint=f'{group}_{i}',queue_fingerprint='queue',reference_fingerprint=f'{group}_{i}',information_fingerprint='no targets',
        parameters_fingerprint='same',metric='RMSE',metric_kind='reference_error',loss_unit='1',status='PASS',actual_execution=True,quality_pass=True,loss=1.,steps=['base','none'])
    b=dict(a,loss=.8,steps=['base','projection'])
    return dict(id=f'{group}_{i}',split='development',group=group,comparison_id=f'{group}_{i}',before=a,after=b,
        attribution='single_capability',changed_stages=[dict(before='none',after='projection')],features={k:float(i) for k in cs.FEATURES},context=context)


@pytest.mark.parametrize('fault',['queue','metric','negative','failed','unknown','composite','multi','near_zero','upstream','test'])
def test_label_gate_rejects_noncomparable_evidence(fault):
    r=evidence()
    if fault=='queue':r['after']['queue_fingerprint']='other'
    if fault=='metric':r['after']['metric_kind']='engineering'
    if fault=='negative':r['before']['loss']=-1
    if fault=='failed':r['after']['quality_pass']=False
    if fault=='unknown':r['after']['status']='UNKNOWN'
    if fault=='composite':r['attribution']='composite'
    if fault=='multi':r['changed_stages'].append({})
    if fault=='near_zero':r['before']['loss']=1e-10
    if fault=='upstream':r['after']['steps'][0]='different'
    if fault=='test':r['split']='sealed_test'
    good,bad=cs.paired_evidence([r],epsilon=1e-8,loss_unit='1')
    assert not good and len(bad)==1


def test_pair_dedup_reward_and_unit():
    r=evidence();good,bad=cs.paired_evidence([r,deepcopy(r)],epsilon=1e-8,loss_unit='1')
    assert len(good)==len(bad)==1 and good[0]['reward']==pytest.approx(.2)
    assert not cs.paired_evidence([r],epsilon=1e-8,loss_unit='K')[0]
    renamed=deepcopy(r);renamed['group']='different_client'
    assert len(cs.paired_evidence([r,renamed],epsilon=1e-8,loss_unit='1')[0])==1


def test_model_group_isolation_unknown_ood_and_version(tmp_path,monkeypatch):
    monkeypatch.setattr(cs,'ROOT',tmp_path)
    rows=[evidence(g,i) for g in ['r0','r1','r2'] for i in range(3)]
    report=cs.CapabilityScoringService.train(rows,dict(epsilon=1e-8,loss_unit='1',metric_kind='reference_error',metric='RMSE',holdout_groups=['r2'],seed=17),'unit_fixture')
    assert set(report['training_groups']).isdisjoint(report['holdout_groups'])
    s=cs.CapabilityScoringService('unit_fixture');assert s.score(rows[0]['features'],rows[0]['context'])['estimated_relative_benefit']==pytest.approx(.2)
    x=s.score(dict(rows[0]['features'],mean=99),rows[0]['context'])
    assert x['status']=='INSUFFICIENT_EVIDENCE' and x['estimated_relative_benefit'] is None
    assert s.score(rows[0]['features'],dict(rows[0]['context'],metric_kind='engineering'))['estimated_relative_benefit'] is None
    with pytest.raises(ValueError,match='Immutable'):cs.CapabilityScoringService.train(rows,report['protocol'],'unit_fixture')
    with (tmp_path/'unit_fixture/model.joblib').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='fingerprint'):cs.CapabilityScoringService('unit_fixture')


def test_required_vs_optional_and_unknown_score_not_zero():
    *_,d,f,external,excluded=prepare()
    sem=d.semantics
    optional=RequirementMapper().map(sem,dict(operation='spatial_processing',coarse_observations_available=True))
    required=RequirementMapper().map(sem,dict(operation='spatial_processing',coarse_observations_available=True,require_coarse_consistency=True))
    assert 'coarse_scale_consistency' in optional.optional_capabilities and 'coarse_scale_consistency' not in optional.required
    assert 'coarse_scale_consistency' in required.required and 'coarse_scale_consistency' not in required.optional_capabilities
    d.requirements=optional
    assert cs.optional_capability_matches(d)['optional'][0]['estimated_relative_benefit'] is None
    explicit=RequirementMapper().map(sem,dict(required_capabilities=['coarse_scale_consistency'],optional_capabilities=['coarse_scale_consistency']))
    assert 'coarse_scale_consistency' in explicit.required and 'coarse_scale_consistency' not in explicit.optional_capabilities


def declaration(name,value,source='role:primary:v',status='CONFIRMED'):
    return dict(feature=name,value=value,unit='1',source_id=source,status=status,confirmed=True,evidence='independent functional fixture inspection')


@pytest.mark.parametrize('category',[False,True])
def test_two_layers_same_mapper_and_actual_candidates(category):
    ds=grid(category);req=request(category);ctx,_,_=make_context([('primary',ds)],req);reg=platform_registry()
    s=SemanticFeatureExtractor().extract(ds,'v',confirmed=dict(variable_class=req['semantic_type'],unit='1'))
    target=dict(operation='spatial_processing',grid_change=True,coarsening=True)
    declared,t,audit=SemanticFeatureExtractor.apply_declarations(s,target,[declaration('variable_class',req['semantic_type']),declaration('grid_change',True)],source_id='role:primary:v')
    d1=AlgorithmSelector(reg).decide(ctx,t,semantics=declared);d2=AlgorithmSelector(reg).decide(ctx,target,semantics=s)
    f1,_,_=bind_registry(reg,ctx,d1,req);f2,_,_=bind_registry(reg,ctx,d2,req)
    assert d1.requirements.required==d2.requirements.required and set(f1.adapters)==set(f2.adapters)
    assert all(x['verification']=='CONFIRMED' for x in audit)
    assert ('spatial.bilinear','run') in f1.adapters if not category else ('spatial.bilinear','run') not in f1.adapters


def test_declaration_conflict_unknown_no_winner_and_units():
    s=SemanticFeatureExtractor().extract(grid(),'v')
    for item in [declaration('missing_count',2),dict(declaration('missing_count',0),unit='hours'),declaration('algorithm','nearest')]:
        with pytest.raises(ValueError):SemanticFeatureExtractor.apply_declarations(s,{},[item],source_id='role:primary:v')
    unknown=declaration('temporal_semantics','instantaneous')
    same,_,audit=SemanticFeatureExtractor.apply_declarations(s,{},[unknown],source_id='role:primary:v')
    assert audit[0]['verification']=='UNKNOWN' and same.temporal.value['semantics']=='UNKNOWN'


@pytest.mark.parametrize('semantics,expected',[('instantaneous','temporal_interpolation'),('interval_mean','time_weighted_aggregation'),('accumulation','accumulation_to_rate')])
def test_time_semantics_layers_are_not_record_count_guesses(semantics,expected):
    s=SemanticFeatureExtractor().extract(grid(),'v')
    s=replace(s,temporal=replace(s.temporal,value=dict(resolution='1h',semantics=semantics)))
    target=dict(operation='temporal_processing',temporal_resolution='6h')
    declared,t,_=SemanticFeatureExtractor.apply_declarations(s,target,[declaration('temporal_semantics',semantics)],source_id='role:primary:v')
    a=RequirementMapper().map(s,target);b=RequirementMapper().map(declared,t)
    assert a.required==b.required and a.required
    # Concrete IDs are declared by the existing rule file, not local routing.
    assert a.trace==b.trace


def test_local_grid_change_and_unrelated_branch_are_consumed():
    *_,a,fa,_=branches(sources(False));*_,b,fb,_=branches(sources(True))
    options,_=canonical_candidates(b,fb)
    assert options and any('prepare.grid.role:primary:aod.' in n.name for g,_ in options for n in g.nodes)
    assert fa.problem_resources['source1']==fb.problem_resources['source1']
    assert not any(p['rule']=='source_grid_mismatch' for p in a.requirements.problems)


@pytest.mark.parametrize('one_sided',[False,True])
def test_declared_vs_recognized_time_support_common_requirements(one_sided):
    from types import SimpleNamespace
    import pandas as pd
    values=np.arange(6,dtype=float);values[0 if one_sided else 2]=np.nan
    s=SemanticFeatureExtractor().extract_series([SimpleNamespace(time=pd.date_range('2015-01-01',periods=6,freq='h'),values=values)],variable='pm25',unit='ug m-3')
    item=declaration('support_conditions',deepcopy(s.observed.value['support_conditions']),source='role:batch:pm25');item['unit']='records'
    target=dict(operation='temporal_missing_recovery',require_complete_coverage=True)
    declared,t,audit=SemanticFeatureExtractor.apply_declarations(s,target,[item],source_id='role:batch:pm25')
    a=RequirementMapper().map(s,target);b=RequirementMapper().map(declared,t)
    assert a.required==b.required
    assert ('boundary_temporal_recovery' in a.required)==one_sided
    assert audit[0]['verification']=='CONFIRMED'


def test_legal_training_bias_vs_insufficient_reference():
    import pandas as pd
    t=pd.date_range('2015-01-01',periods=4,freq='h')
    train=xr.DataArray([1.,2.,3.,4.],dims='time',coords={'time':t},attrs={'split':'train','units':'1'})
    hist=train.copy();hist.values=hist.values+.2
    application=train.assign_coords(time=t+pd.Timedelta(days=7))
    known=SemanticFeatureExtractor.training_bias(train,hist,application)
    unknown=SemanticFeatureExtractor.training_bias(train,hist,train)
    sem=SemanticFeatureExtractor().extract(grid(),'v')
    a=RequirementMapper().map(sem,dict(training_bias=known));b=RequirementMapper().map(sem,dict(training_bias=unknown))
    assert 'bias_correction' in a.optional_capabilities and 'bias_correction' not in a.required
    assert 'bias_correction' not in b.optional_capabilities and unknown['status']=='UNKNOWN'


def test_hourly_adapter_consumes_same_declarations(tmp_path):
    from test_platform_numerical_search import evaluator
    e=evaluator(tmp_path);s=e.decision().semantics
    old=e.decision().requirements.required;e._decision=None
    e.request=dict(e.request,feature_declarations=[declaration('missing_count',s.observed.value['missing_count'],source='role:batch:pm25')])
    assert e.decision().requirements.required==old
    assert e.decision().feature_declarations[0]['verification']=='CONFIRMED'


def test_interval_mean_alias_executes_original_kernel_without_input_mutation():
    from ecodata.core.schemas import EcoDataBundle
    from ecodata.algorithms.contracts import AlgorithmContext
    from ecodata.algorithms.plugins import TemporalSemanticPlugin
    import pandas as pd
    times=pd.date_range('2020-01-01 01:00',periods=6,freq='h').values
    ds=xr.Dataset({'v':('time',np.arange(6.)), 'time_bounds':(('time','bounds'),np.column_stack((times-np.timedelta64(1,'h'),times)))},coords={'time':times})
    ds.v.attrs.update(units='K',data_kind='continuous',temporal_semantics='interval_mean')
    original=ds.copy(deep=True);ctx=AlgorithmContext((EcoDataBundle(ds),),'v','continuous',parameters=dict(target_time_frequency='3h'))
    p=TemporalSemanticPlugin();assert p.compatibility(ctx).eligible
    result=p.run(ctx).data.dataset
    np.testing.assert_allclose(result.v,[1.,4.]);xr.testing.assert_identical(ds,original)
    assert result.v.attrs['temporal_semantics']=='interval_mean'
    bad=ds.copy(deep=True);bad.v.attrs['temporal_semantics']='UNKNOWN'
    assert not p.compatibility(replace(ctx,sources=(EcoDataBundle(bad),))).eligible


def test_scoring_actual_probability_consumer_and_off(tmp_path,monkeypatch):
    from golem.core.optimisers.graph import OptGraph,OptNode
    from types import SimpleNamespace
    monkeypatch.setattr(cs,'ROOT',tmp_path)
    rows=[evidence(g,i) for g in ['r0','r1','r2'] for i in range(3)]
    cs.CapabilityScoringService.train(rows,dict(epsilon=1e-8,loss_unit='1',metric_kind='reference_error',metric='RMSE',holdout_groups=['r2'],seed=17),'probability_fixture')
    d=SimpleNamespace(benefit_service=cs.CapabilityScoringService('probability_fixture'),benefit_context={k:v for k,v in rows[0]['context'].items() if k!='base_algorithm'},
        benefit_features=rows[0]['features'],benefit_base_algorithms=['raster_task.spatial.bilinear'],requirements=SimpleNamespace(optional_capabilities=['coarse_scale_consistency'],required=[]))
    base=OptNode('raster_task.spatial.bilinear');plain=OptGraph(base);composite=OptGraph(OptNode('raster_task.coarse_consistency',nodes_from=[base]))
    weights,trace=cs.empirical_proposal_weights(d,None,[plain,composite],[1,1],{})
    assert weights[1]>weights[0]>0 and sum(weights)==pytest.approx(1)
    d.requirements.required=['coarse_scale_consistency']
    unchanged,_=cs.empirical_proposal_weights(d,None,[plain,composite],[1,1],{})
    assert unchanged==[.5,.5] # Benefit never waives required acceptance.
    del d.benefit_service
    unchanged,_=cs.empirical_proposal_weights(d,None,[plain,composite],[1,1],{})
    assert unchanged==[.5,.5]
