from copy import deepcopy
from itertools import permutations
import json

import numpy as np
import pytest

from ecodata.algorithms import reliable_selection as selection


@pytest.fixture(autouse=True)
def no_ambient_calibration(monkeypatch,tmp_path):
    monkeypatch.setattr(selection,'POLICY',tmp_path/'not_published.json')


def policy(**declaration):
    return selection.make_policy(dict(reliable_selection=declaration),[0.,2.])


def fold(identifier,rmse,mae=None,bias=0.,n=10,**changes):
    return dict(fold_id=identifier,status='PASS',metrics=dict(RMSE=rmse,MAE=rmse if mae is None else mae,
        Bias=bias,denominator=n,**changes))


def row(key,loss,nodes,**extra):
    return dict(key=key,status='ELIGIBLE_OUTPUT',metrics=dict(objective=loss),algorithm_nodes=nodes,**extra)


def test_default_policy_uses_visible_scale_and_no_invented_calibration():
    p=policy()
    assert p['scales']=={'RMSE':1.} and p['weights']=={'RMSE':1.}
    assert p['tolerance']['status']=='UNCALIBRATED_EXACT_LOSS_ONLY'
    assert p['delta']==0 and p['mode']=='full'
    assert p['data_portrait']['retained_finite_count']==2
    assert p['sha256']==policy()['sha256']
    with pytest.raises(ValueError,match='retained'):
        selection.make_policy({},[np.nan])


def test_fold_aggregation_keeps_physical_metrics_and_unknowns_honest():
    p=policy();result=selection.aggregate([fold('b',3,2,bias=-2,n=20),fold('a',1,.5,bias=1,n=5)],p,['a','b'])
    assert result['fold_ids']==['a','b'] and result['fold_losses']==[1.,3.]
    assert result['objective']==result['RMSE']==2.
    assert result['MAE']==1.25 and result['Bias']==-.5 and result['valid_n']==25
    assert result['loss_stats']==dict(mean=2.,std=1.,minimum=1.,maximum=3.,
        std_interpretation='Dispersion across these folds; NOT standard error or confidence interval')
    assert not result['folds_are_independent']
    response=result['portrait_response'];assert response['status']=='MEASURED_MULTI_FOLD'
    term=response['terms'][0]['terms'][0]
    assert term['algorithm_response']==1. and term['model_weight']==1.
    assert term['response_status']=='MEASURED_COMPLETE_GRAPH'
    assert response['predicted_gain'].startswith('UNKNOWN')


@pytest.mark.parametrize('changes',[dict(status='EXECUTION_FAILED'),dict(status='INTERMEDIATE_ONLY')])
def test_failed_required_fold_is_never_silently_removed(changes):
    a=fold('a',1);b=fold('b',2);b.update(changes)
    with pytest.raises(ValueError,match='did not pass'):
        selection.aggregate([a,b],policy(),['a','b'])


def test_mismatched_duplicate_or_missing_folds_fail_closed():
    for rows,expected in [([fold('a',1)],['a','b']),([fold('a',1),fold('a',2)],['a']),([fold('a',1)],['a','a'])]:
        with pytest.raises(ValueError):selection.aggregate(rows,policy(),expected)


def test_explicit_model_priority_changes_real_response_selection():
    a=[fold('a',.1,.4,bias=0)];b=[fold('a',.2,.1,bias=0)]
    rmse=selection.make_policy({},[0.,2.],model_weights={'RMSE':1.})
    mae=selection.make_policy({},[0.,2.],model_weights={'MAE':1.})
    rank=lambda p:selection.choose([row('A',selection.aggregate(a,p,['a'])['objective'],1),
                                  row('B',selection.aggregate(b,p,['a'])['objective'],1)],p)['key']
    assert rank(rmse)=='A' and rank(mae)=='B'
    assert mae['weight_basis']=='Explicit application/model metric priorities'


def test_abs_bias_uses_mean_absolute_fold_bias_not_cancelled_signed_mean():
    p=selection.make_policy({},[0.,2.],model_weights={'abs_bias':1.})
    result=selection.aggregate([fold('a',1,bias=-2),fold('b',1,bias=2)],p,['a','b'])
    assert result['objective']==2 and result['Bias']==0
    missing=fold('a',1);missing['metrics']['Bias']=None
    with pytest.raises(ValueError,match='UNKNOWN'):selection.aggregate([missing],p,['a'])


def test_generic_objective_is_not_silently_mixed_with_explicit_metrics():
    p=selection.make_policy({},[1.,2.],model_weights={'objective':1.})
    assert p['weights']=={'RMSE':1.} and 'generic legacy objective' in p['weight_basis']
    with pytest.raises(ValueError,match='mixed'):
        selection.make_policy({},[0.,2.],model_weights={'objective':1.,'RMSE':1.})
    with pytest.raises(ValueError,match='override'):
        selection.make_policy({'reliable_selection':{'weights':{'RMSE':1.}}},[0.,2.],model_weights={'MAE':1.})


@pytest.mark.parametrize('value',[None,float('nan'),True,-.1])
def test_unknown_or_invalid_required_response_is_not_a_middle_score(value):
    r=fold('a',1);r['metrics']['RMSE']=value
    with pytest.raises(ValueError):selection.aggregate([r],policy(),['a'])


def test_development_publication_is_hashed_scoped_and_test_evidence_rejected(monkeypatch,tmp_path):
    published=dict(version='dev1',delta=.1,evidence_role='development',evidence='Frozen grouped development cases',
        scope=dict(variable='ndvi',unit='1'))
    path=tmp_path/'policy.json';path.write_text(json.dumps(published),encoding='utf-8')
    monkeypatch.setattr(selection,'POLICY',path)
    p=selection.make_policy(dict(variable='ndvi',unit='1'),[0.,2.])
    assert p['delta']==.1 and p['tolerance']['status']=='DEVELOPMENT_CALIBRATED'
    assert p['tolerance']['source']['sha256']
    other=selection.make_policy(dict(variable='temperature',unit='K'),[0.,2.])
    assert other['delta']==0 and other['tolerance']['status']=='OUT_OF_CALIBRATION_SCOPE_EXACT_LOSS_ONLY'
    published['evidence_role']='test'
    with pytest.raises(ValueError,match='development'):
        selection.make_policy({},[0.,2.],calibration=published)


def test_positive_explicit_tolerance_requires_an_auditable_basis():
    with pytest.raises(ValueError,match='evidence'):policy(delta=.1)
    assert policy(delta=.1,evidence='Predeclared development choice')['delta']==.1


def test_equivalence_set_is_global_and_not_pairwise_nontransitive():
    p=policy(delta=.1,evidence='Development tolerance')
    candidates=[row('a',0.,10),row('b',.09,5),row('c',.18,1)]
    for order in permutations(candidates):
        assert [r['key'] for r in selection.ordered(order,p)]==['b','a','c']
        assert selection.choose(order,p)['key']=='b'
    # c is close to b but outside the global optimum's tolerance, so it cannot
    # leapfrog a and b solely through a chain of pairwise equivalences.


def test_real_extra_gain_keeps_complex_graph_and_no_parsimony_is_explicit():
    candidates=[row('simple',.1,2),row('complex',.085,10)]
    p=policy(delta=.002,evidence='Development tolerance')
    assert selection.choose(candidates,p)['key']=='complex'
    near=[row('simple',.1,2),row('complex',.0998,10)]
    assert selection.choose(near,p)['key']=='simple'
    assert selection.choose(near,policy(mode='no_parsimony',delta=.002,evidence='Development tolerance'))['key']=='complex'


def test_cost_tie_uses_declared_units_never_dynamic_walltime():
    p=policy(delta=.1,evidence='Development tolerance')
    a=row('a',1.,2,estimated_cost=100,execution_seconds=100)
    b=row('b',1.,2,estimated_cost=.01,execution_seconds=.01)
    assert selection.choose([a,b],p)['key']=='a'
    a['declared_cost']=3;b['cost_units']=2
    assert selection.choose([a,b],p)['key']=='b'
    assert selection.choose([dict(a,status='EXECUTION_FAILED'),dict(b,dominated=True)],p) is None


def test_common_gain_is_paired_diagnostic_not_a_second_reward():
    p=policy();a=selection.aggregate([fold('a',1),fold('b',3)],p,['a','b'])
    b=selection.aggregate([fold('a',2),fold('b',2)],p,['a','b'])
    original=deepcopy(a);gain=selection.common_gain(a,b,p)
    assert gain['gain']==0 and gain['fold_gains']==[1.,-1.] and gain['fold_wins']==1
    assert not gain['used_as_additional_accuracy_reward'] and a==original
    c=dict(b,fold_ids=['b','a'])
    with pytest.raises(ValueError,match='identical folds'):selection.common_gain(a,c,p)
    c=dict(b,loss_signature='another definition')
    with pytest.raises(ValueError,match='semantics'):selection.common_gain(a,c,p)


@pytest.mark.parametrize('mode',['fold_only','no_portraits','legacy_single'])
def test_no_portrait_modes_do_not_invent_soft_model_priorities(mode):
    p=selection.make_policy({'reliable_selection':{'mode':mode}},[0.,2.],model_weights={'MAE':1.})
    assert p['weights']=={'RMSE':1.} and not p['portraits_enabled']
    assert p['parsimony_enabled']==(mode=='no_portraits')


def test_explicit_metric_scale_is_fixed_and_changes_objective_not_physical_metrics():
    p=selection.make_policy({'preferences':{'weights':{'RMSE':1.},'scales':{'RMSE':2.}}},[0.,100.])
    result=selection.aggregate([fold('a',4)],p,['a'])
    assert result['objective']==2 and result['RMSE']==4
    assert result['portrait_response']['terms'][0]['terms'][0]['scale_evidence']=='Explicit application/task scale'


def test_primary_history_folds_do_not_mix_with_local_diagnostics():
    p=dict(policy(),objective_fold_ids=['history'])
    result=selection.aggregate([fold('local',100,70,bias=-30,n=20),fold('history',2,1,bias=.5,n=10)],p,['local','history'])
    assert result['fold_losses']==[100.,2.] and result['objective']==result['RMSE']==2.
    assert result['MAE']==1. and result['Bias']==.5
    assert result['loss_stats']['mean']==2 and result['loss_stats']['std']==0
    assert result['diagnostic_loss_stats']['mean']==100
    assert result['objective_fold_ids']==['history'] and result['diagnostic_fold_ids']==['local']
    assert result['valid_n']==10 and result['all_folds_valid_n']==30 and result['diagnostic_valid_n']==20
    assert result['portrait_response']['terms'][0]['role']=='diagnostic'
    assert result['portrait_response']['terms'][1]['role']=='objective'
    assert result['loss_definition']['objective_fold_ids']==['history']


def test_failed_diagnostic_is_still_required_for_primary_history_selection():
    p=dict(policy(),objective_fold_ids=['history'])
    local=fold('local',1);local['status']='EXECUTION_FAILED'
    with pytest.raises(ValueError,match='did not pass'):
        selection.aggregate([local,fold('history',2)],p,['local','history'])
    with pytest.raises(ValueError,match='Incomplete'):
        selection.aggregate([fold('history',2)],p,['local','history'])


def test_common_gain_separates_primary_and_diagnostic_improvement():
    p=dict(policy(),objective_fold_ids=['history'])
    a=selection.aggregate([fold('local',100),fold('history',1)],p,['local','history'])
    b=selection.aggregate([fold('local',2),fold('history',3)],p,['local','history'])
    gain=selection.common_gain(a,b,p)
    assert gain['gain']==2 and gain['fold_gains']==[-98.,2.]
    assert gain['objective_fold_gains']==[2.] and gain['diagnostic_fold_gains']==[-98.]
    assert gain['fold_wins']==1 and gain['diagnostic_fold_wins']==0
    different=selection.aggregate([fold('local',2),fold('history',3)],dict(policy(),objective_fold_ids=['local']),['local','history'])
    with pytest.raises(ValueError,match='semantics'):selection.common_gain(a,different,p)


@pytest.mark.parametrize('ids',[[],['history','history'],['missing'],'history'])
def test_invalid_primary_fold_declaration_fails_closed(ids):
    with pytest.raises(ValueError,match='objective-fold subset'):
        selection.aggregate([fold('history',1)],dict(policy(),objective_fold_ids=ids),['history'])
