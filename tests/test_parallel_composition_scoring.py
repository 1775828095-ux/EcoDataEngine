"""Actual visible-mask responses, independent of hidden target labels."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scheduled_fusion_data import case
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.composition_decision import Composition
from ecodata.models import portrait_policy as policy
from test_full_screen_spatiotemporal import publish,stores


def setup(length=33):
    publish();inputs,req,truth=case(0,synthetic=True,length=length)
    req['composition']['parallel_scoring']=dict(weights=dict(data=1,model=1,collaboration=1),
        evidence='Equal engineering route priority for synthetic mechanism check, not WRF sensitivity')
    req['composition']['search'].update(backend='none',generations=0)
    return inputs,req,truth


def engine_for(inputs,req):
    parsed,_=api.parse_inputs(inputs)
    return Composition(parsed,req,api.platform_registry())


def test_real_responses_distinguish_candidates_and_keep_unknown(tmp_path):
    inputs,req,_=setup();engine=engine_for(inputs,req)
    scores=[engine.screen(engine.decode(i)) for i in range(engine.count)]
    assert len({tuple(s['score_interval']) for s in scores})>1
    scorer=engine.parallel_scorer
    assert scorer.executions==72
    assert len(scorer.responses)==72
    assert all(r['status']=='MEASURED_VISIBLE_MASK' for r in scorer.responses.values())
    for probe in scorer.probes:
        assert not {np.datetime64(t) for t in probe['withheld_times']} & {np.datetime64(t) for t in probe['retained_times']}
    assert any(t.get('before')!=t.get('after') for r in scorer.responses.values() for t in r['state_transitions'] if 'before' in t)
    # Neither target labels nor a response invented for absent probe support.
    missing_inputs,missing_req,_=case(0,synthetic=True,length=17)
    missing_req['composition']['parallel_scoring']=deepcopy(req['composition']['parallel_scoring'])
    missing=engine_for(missing_inputs,missing_req);row=missing.screen(missing.decode(0))
    assert row['score_interval']==[0.,1.]
    assert all(s['status']=='UNKNOWN' for s in row['state_responses'])


def test_public_entry_without_labels_and_independent_labels_do_not_change_selection(tmp_path):
    inputs,req,truth=setup()
    results=[]
    for name,reference in [('none',None),('changed_test',api.DatasetReference(truth+1000,req['variable']))]:
        result=api.submit_search(inputs,req,reference,tmp_path/name,mode='Profile',budget=dict(objective_limit=2,wall_seconds=120))
        assert result['parallel_scoring']['unique_score_intervals']>1
        assert result['parallel_scoring']['known_score_fraction']==1
        receipt=json.loads((tmp_path/name/'parallel_scoring.json').read_text())
        assert receipt['probe_graph_executions']==72
        results.append(result)
    assert results[0]['recommended_key']==results[1]['recommended_key']
    assert results[0]['best']['numeric_fingerprint']==results[1]['best']['numeric_fingerprint']


def test_weight_changes_change_corresponding_ranking_and_unknown_rejected():
    inputs,req,_=setup()
    parsed,_=api.parse_inputs(inputs)
    biased=[]
    for (role,ds),item in zip(parsed,inputs):
        if role=='background':ds[req['variable']].values+=5
        biased.append(api.SearchInput(item.filename,role,ds.to_netcdf()))
    engine=engine_for(biased,req)
    engine.prepare_parallel_scoring();scorer=engine.parallel_scorer
    rankings=[]
    for route in ('data','model','collaboration'):
        scorer.weights={r:float(r==route) for r in scorer.weights}
        ranked=sorted(range(engine.count),key=lambda i:(-scorer.score(engine.plan(engine.decode(i)))['score_interval'][0],engine.steps(engine.decode(i)),i))
        rankings.append(ranked[0])
    assert len(set(rankings))>1
    bad=deepcopy(req);bad['composition']['parallel_scoring']['weights']['model']=float('nan')
    with pytest.raises(ValueError,match='Finite'):engine_for(inputs,bad).screen(engine.decode(0))


def test_model_preference_and_visible_data_change_public_decisions(tmp_path):
    inputs,req,_=setup();parsed,_=api.parse_inputs(inputs)
    biased=[]
    for (role,ds),item in zip(parsed,inputs):
        if role=='background':ds[req['variable']].values+=5
        biased.append(api.SearchInput(item.filename,role,ds.to_netcdf()))
    choices=[]
    for metric in ('coarse_error','detail_error'):
        scenario=deepcopy(policy.load('complementary-wrf-v1')['scenarios'][0])
        for issue in scenario['issues']:issue['importance']=float(issue['metric']==metric)
        version='probe-model-'+metric
        policy.publish(version,'wrf',[scenario],evidence='Synthetic explicit preference tradeoff; not simulator response')
        request=deepcopy(req);request['portrait_policy']['version']=version
        request['composition']['parallel_scoring']['weights']=dict(data=0,model=1,collaboration=0)
        result=api.submit_search(biased,request,None,tmp_path/metric,mode='Profile',budget=dict(objective_limit=2,wall_seconds=120))
        choices.append(result['recommended_key'])
    assert choices[0]!=choices[1]
    # Same data-fidelity preference, changed only the visible coarse field.
    data_req=deepcopy(req);data_req['composition']['parallel_scoring']['weights']=dict(data=1,model=0,collaboration=0)
    selected=[]
    for name,items in [('clean',inputs),('biased',biased)]:
        result=api.submit_search(items,data_req,None,tmp_path/name,mode='Profile',budget=dict(objective_limit=2,wall_seconds=120))
        selected.append(result['recommended_key'])
    assert selected[0]!=selected[1]
