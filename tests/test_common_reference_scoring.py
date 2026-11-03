from copy import deepcopy
import hashlib,json

import numpy as np
import pytest

from ecodata.algorithms.composition_scoring import reference_scope,gain_priority
from test_parallel_composition_scoring import setup,engine_for,stores


def common_setup(tmp_path):
    inputs,req,_=setup();engine=engine_for(inputs,req)
    fixed=engine.plan(engine.decode(0));single=deepcopy(fixed)
    for p in single:
        if p['branch']=='fusion':p['choice']=dict(algorithm='prepare.select_source',source='coarse_fast',evidence='Synthetic frozen control')
        if p['branch']=='post':p['choice']=[]
    profile=dict(training_partition='development',test_used=False,training_groups=['synthetic_fixture'],
        scope=reference_scope(engine),plans=dict(fixed=fixed,single=single))
    path=tmp_path/'reference.json';path.write_text(json.dumps(profile),encoding='utf-8')
    req['composition']['parallel_scoring']=dict(weights=dict(data=.5,model=.5,collaboration=0),
        common_reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),evidence='Synthetic common-reference gate acceptance')
    return inputs,req


def test_common_reference_is_shared_and_not_constituent_dependent(tmp_path):
    inputs,req=common_setup(tmp_path);engine=engine_for(inputs,req)
    scores=[engine.screen(engine.decode(i)) for i in range(engine.count)]
    assert engine.parallel_scorer.executions==76
    refs={r['common_gain']['reference_rmse'] for r in scores}
    assert len(refs)==1
    for row in scores:
        c=row['common_gain'];assert c['delta_rmse_K']==pytest.approx(c['reference_rmse']-c['candidate_rmse'])
        assert row['score_interval'][0]==pytest.approx(.5*row['routes']['data'][0]+.5*row['routes']['model'][0])
    before=engine.screen(engine.decode(0))
    for r in engine.parallel_scorer.responses.values():r['constituent_rmse']={k:1e9 for k in r['constituent_rmse']}
    assert engine.screen(engine.decode(0))==before
    # A secondary gain cannot invert two different primary scores.
    rows=[dict(score_interval=[.7,.7],common_gain=dict(stable_gain=True,delta_rmse_K=100)),
        dict(score_interval=[.8,.8],common_gain=dict(stable_gain=False,delta_rmse_K=-1))]
    assert sorted(rows,key=lambda r:(-r['score_interval'][0],*[-x for x in gain_priority(r)]))[0]['score_interval'][0]==.8


def test_reference_provenance_and_unknown_support(tmp_path):
    inputs,req=common_setup(tmp_path)
    bad=deepcopy(req);bad['composition']['parallel_scoring']['common_reference']['sha256']='bad'
    with pytest.raises(ValueError,match='digest'):engine_for(inputs,bad).screen((0,)*8)
    bad=deepcopy(req);bad['composition']['parallel_scoring']['weights']['collaboration']=1
    with pytest.raises(ValueError,match='secondary'):engine_for(inputs,bad).screen((0,)*8)
    engine=engine_for(inputs,req);engine.prepare_parallel_scoring()
    engine.parallel_scorer.reference_probes=[]
    row=engine.screen(engine.decode(0))
    assert row['common_gain']['status']=='UNKNOWN' and row['routes']['collaboration']==[0.,1.]
