from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scheduled_fusion_data import case
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.composition_decision import Composition
from test_full_screen_spatiotemporal import publish,stores


def setup():
    publish();inputs,req,truth=case(0,synthetic=True)
    return inputs,req,truth


def test_full_relation_count_and_two_window_choices():
    # Called through the same parsed-input and registered portrait mechanism.
    inputs,req,_=setup();parsed,_=api.parse_inputs(inputs)
    engine=Composition(parsed,req,api.platform_registry())
    assert engine.count==1296
    legal=sum(engine.preflight(engine.decode(i))['status']!='REJECT' for i in range(engine.count))
    assert legal==1296
    assert {tuple(s[1:] ) for s in engine.slots}=={(w,b) for w in ('first','second') for b in ('fine_slow','coarse_fast','fusion','post')}
    chains=next(a for s,a in zip(engine.slots,engine.axes) if s[1:] == ('first','coarse_fast'))
    assert len(chains)==6
    assert any(c[0]['algorithm']=='spatial.median_filter' for c in chains)
    assert any(c[-1]['algorithm']=='spatial.median_filter' for c in chains)


def test_real_validation_feedback_selects_actual_error_minimum(tmp_path):
    inputs,req,truth=setup()
    req['composition']['windows']=[dict(id='all',start=req['composition']['windows'][0]['start'],end=req['composition']['windows'][1]['end'])]
    req['composition']['search'].update(exact_check=True,backend='alns',generations=2)
    parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
    # Known numeric product creates an exact-zero-error oracle; not a fake scalar scorer.
    target,*_=engine.execute(engine.decode(engine.count-1));labels=target[req['variable']].copy();labels.attrs['split']='search_validation'
    result=api.submit_search(inputs,req,None,tmp_path/'validated',mode='Profile',validation=api.SearchValidation(labels,req['variable']),budget=dict(objective_limit=36,proposal_limit=8,wall_seconds=120))
    assert result['final_validation_rmse']==pytest.approx(0.,abs=1e-9)
    assert result['certification']=='EXACT_VALIDATION_RMSE_OPTIMUM_IN_DECLARED_DOMAIN'
    assert result['search_validation_used_for_selection'] and not result['independent_test_used_for_selection']
    assert result['native_optimizer']['operator_counts']
    assert result['final_validation_rmse']==min(r['search_validation']['RMSE'] for r in result['rows'] if r['status']=='PASS')


def test_label_roles_and_overlap_blocked(tmp_path):
    inputs,req,truth=setup()
    labels=truth.copy();labels.attrs['split']='test'
    with pytest.raises(ValueError,match='search_validation'):api.SearchValidation(labels,req['variable'])
    labels.attrs['split']='search_validation';validation=api.SearchValidation(labels,req['variable'])
    with pytest.raises(ValueError,match='overlaps'):
        api.submit_search(inputs,req,api.DatasetReference(truth,req['variable']),tmp_path/'overlap',mode='Profile',validation=validation)
