from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.series_fusion import SeriesNonnegative, SeriesMissingFallback
from ecodata.algorithms.composition_decision import PrimaryWeightedMean, score_state
from ecodata.algorithms.field_mutation import GraphMutation, visible_validation
from ecodata.algorithms.composition_optimizers import search_graph
from ecodata.algorithms.portrait_state import measure
from test_series_fusion import data, context


def model(metric='negative_fraction'):
    return dict(attribute_registry=dict(attributes=[dict(id='state',scoring_enabled=True)]),
      state_policy=dict(scenario=dict(issues=[dict(attribute='state',metric=metric,
                        acceptable=0.,bad=1.,importance=1.,evidence='Explicit test input requirement')])))


def engine(tmp_path,metric='negative_fraction',**budget):
    registry=AlgorithmRegistry()
    for p in [SeriesNonnegative(),SeriesMissingFallback(),PrimaryWeightedMean()]:registry.register(p)
    a=data([1,np.nan,3,5,2]);b=data([2,2,2,2,2],source='other')
    c=context(a,b);ref=a.precipitation.isel(time=[1]).copy();ref.values[:]=4
    return GraphMutation(c,registry,dict(response_scoring=dict(enabled=False)),model(metric),ref,tmp_path,
        dict(objective_limit=20,proposal_limit=4000,wall_seconds=30,**budget),11)


def test_portrait_formula_shared_and_need_changes_order(tmp_path):
    e=engine(tmp_path)
    try:
        e.states['s0'].precipitation.values[0]=-2
        e.baseline=measure(e.states['s0'],e.context)['metrics']
        key=e.intern('constraint.series_nonnegative',['s0'],{})
        negative=e.matching(key)['score_interval']
        missing=deepcopy(e.field);missing['model']=model('missing_fraction')
        expected=score_state(missing,e.baseline,dict(missing_fraction=e.baseline['missing_fraction']))
        e.field=missing
        assert e.matching(key)['score_interval']==expected['score_interval']
        assert negative!=expected['score_interval']
        # Data changes alter the same explicit negative-risk reduction.
        e.field['model']=model();e.states['s0'].precipitation.values[0]=2
        e.baseline=measure(e.states['s0'],e.context)['metrics']
        assert e.matching(key)['score_interval'][0]==0.
    finally:e.close()


def test_alns_executes_actual_parent_child_changes_and_keeps_best(tmp_path):
    e=engine(tmp_path)
    try:
        best=e.initialize();before=best['metrics']['objective']
        receipt=search_graph(e,best['key'],seed=11)
        assert receipt['backend']=='alns'
        assert sum(sum(r) for r in receipt['operator_counts'])>0
        assert e.transitions and any(t['actual_execution'] for t in e.transitions)
        assert len(e.transitions)==sum(r['origin']=='mutation' for r in e.evaluated.values())
        assert len({t['child'] for t in e.transitions})==len(e.transitions)
        assert all(t['parent']!=t['child'] for t in e.transitions)
        assert e.best()['metrics']['objective']<=before
        assert any(r['origin']=='mutation' for r in e.evaluated.values())
        assert all(not t['accepted'] or t['after_objective']<=t['before_objective'] for t in e.transitions)
        assert all(t['parent_is_best_at_proposal'] for t in e.transitions)
    finally:e.close()


def test_rewire_rebuilds_descendants_instead_of_changing_old_graph(tmp_path):
    e=engine(tmp_path)
    try:
        child=e.intern('constraint.series_nonnegative',['s0'],{})
        sink=e.intern('fusion.series_missing_fallback',[child,'s1'],dict(primary_index=0))
        original=deepcopy(e.records)
        changed=e.replace_node(sink,child,'s1')
        assert changed!=sink and e.records[changed]['parents']==['s1','s1']
        assert all(e.records[k]==v for k,v in original.items())
        # Immutable construction remains acyclic even when reconnecting to old branches.
        assert len(e.nodes(changed))==len(set(e.nodes(changed)))
    finally:e.close()


def test_missing_validation_is_derived_only_from_visible_values():
    import pandas as pd
    dates=pd.date_range('2000-01-01','2001-12-31')
    a=data(np.ones(len(dates)),times=dates);a.precipitation.values[10:13]=np.nan
    b=data(np.full(len(dates),2.),source='other',times=dates)
    req=dict(task='multisource',variable='precipitation',unit='mm/day',semantic_type='continuous',
             target_lat=[11.6],target_lon=[37.36],parameters={})
    inputs=[api.SearchInput('a.nc','primary',a.to_netcdf()),api.SearchInput('b.nc','background',b.to_netcdf())]
    changed,validation,protocol=visible_validation(inputs,req)
    parsed,_=api.parse_inputs(changed)
    assert np.isfinite(validation.reference).all()
    assert not pd.DatetimeIndex(a.time.values[10:13]).isin(validation.reference.time.values).any()
    assert np.isnan(parsed[0][1].precipitation.sel(time=validation.reference.time)).all()
    assert not np.isnan(parsed[1][1].precipitation).any()


def test_public_entry_does_not_silently_fallback_without_registered_portrait(tmp_path):
    with pytest.raises(ValueError,match='model_profile'):
        api.submit_search([],dict(task='multisource'),None,tmp_path,mode='PortraitMutation')
    failure=json.loads((tmp_path/'failure.json').read_text())
    assert failure['status']=='BLOCKED'


def test_empty_mutation_is_rejected_not_rewarded_as_equal_parent():
    class Empty:
        evaluated={'parent':dict(status='ELIGIBLE_OUTPUT',metrics=dict(objective=2.))}
        attempts=0
        def exhausted(self):return self.attempts>=12
        def propose(self,*args):self.attempts+=1;return None
    e=Empty();receipt=search_graph(e,'parent',seed=11)
    assert e.attempts==12
    assert sum(sum(row) for row in receipt['operator_counts'])==12
    assert all(value==0 for row in receipt['operator_average_rewards'] for value in row)


def test_proven_identity_projection_not_used_as_mutation(tmp_path):
    e=engine(tmp_path)
    try:
        spec=e.registry.get('constraint.series_nonnegative').spec
        assert e.candidate('s1',None,spec,['s1'],{},'insert_node') is None
        e.screen_log.flush()
        row=json.loads((tmp_path/'portrait_screening.jsonl').read_text().splitlines()[-1])
        assert row['status']=='ANALYTIC_IDENTITY' and e.executions==0
    finally:e.close()


def test_page_client_forwards_same_mutation_entry(monkeypatch,tmp_path):
    from ecodata.interface import search_page
    calls=[]
    def public(*args,**options):calls.append(options);return dict(service='same')
    monkeypatch.setattr(search_page,'submit_search',public)
    assert search_page.process_search_upload([],{},None,tmp_path,mode='PortraitMutation')['service']=='same'
    assert calls==[dict(mode='PortraitMutation')]


def test_model_issue_changes_real_visible_branch_priority(tmp_path):
    e=engine(tmp_path,metric='missing_fraction')
    try:
        def priority():return min(e.roots,key=lambda key:(-e.matching(key)['score_interval'][0],key))
        assert priority()=='s1'  # complete independent estimate
        e.field['model']=model('peak_loss')
        assert priority()=='s0'  # retains the visible station peak
        assert e.matching('s1')['terms'][0]['algorithm_evidence']['after']>0
    finally:e.close()


def test_soft_disabled_never_calls_state_score_but_executes_candidates(tmp_path,monkeypatch):
    e=engine(tmp_path)
    e.soft_guidance=False;e.response.enabled=False
    def forbidden(*args,**kwargs):raise AssertionError('Disabled soft scorer was called')
    monkeypatch.setattr('ecodata.algorithms.field_mutation.score_state',forbidden)
    try:
        assert e.matching('s0')['status']=='DISABLED'
        best=e.initialize()
        assert best is not None and best['metrics']['objective']>=0
        assert e.executions>0
        assert all(r['matching']['status']=='DISABLED' for r in e.edits.values() if 'matching' in r)
    finally:e.close()
