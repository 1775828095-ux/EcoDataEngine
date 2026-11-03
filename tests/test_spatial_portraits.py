"""Soft-score formula reuse, label isolation, and actual candidate decisions."""
from test_spatial_relationship_fusion import fixture,BUDGET
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.spatial_fusion_task import SpatialTask,SpatialSearch


def make(tmp_path,items=None):
    data,q,_=fixture()
    q.update(spatial_soft_guidance=True,spatial_candidate_pool=4)
    parsed,audit=api.parse_inputs(data if items is None else items)
    issues=[dict(attribute='missing',metric='missing_fraction',importance=1.,acceptable=0.,bad=1.,evidence='test spatial completeness preference'),
            dict(attribute='error',metric='expected_error',importance=1.,acceptable=0.,bad=1.,evidence='unknown accuracy')]
    model=dict(state_policy=dict(scenario=dict(issues=issues)),scoring_parameters={})
    return SpatialSearch(SpatialTask.build(parsed,q,audit),tmp_path,BUDGET,11,model=model)


def test_spatial_scores_separate_actions_and_keep_unknown(tmp_path):
    e=make(tmp_path)
    source=e.leaves[0];idw=e.leaves[2]
    a=e.portraits.score(source);b=e.portraits.score(idw)
    assert b['score_interval'][0]>a['score_interval'][0]
    assert b['accuracy_prediction']=='UNKNOWN' and b['validation_loss_used'] is False
    assert b['folds'][0]['model_terms'][1]['status']=='UNKNOWN'
    e.portraits.model['scoring_parameters']={'missing':dict(multiplier=0.)}
    e.portraits.cache={}
    assert e.portraits.score(idw)['score_interval'][0]==0.


def test_held_labels_and_confirmation_do_not_change_fold_profile(tmp_path):
    e=make(tmp_path);items,_,obs=fixture()
    held=e.folds[0]['held']
    obs.loc[obs.station_id.isin(held),'value']+=10000
    obs.loc[obs.time>=e.confirm_times.min(),'value']+=10000
    items[0]=api.SearchInput('obs.csv','observations',obs.to_csv(index=False).encode())
    changed=make(tmp_path,items)
    assert e.portraits.profiles[0]==changed.portraits.profiles[0]


def test_portrait_tournament_changes_executed_candidate(tmp_path):
    e=make(tmp_path);source=e.leaves[0];idw=e.leaves[2]
    choices=iter([source,idw,source,source])
    e.candidate=lambda *a,**k:next(choices)
    key=e.propose(source,'structure','portrait',e.rng)
    assert key==idw and e.transitions[-1]['guidance_changed_choice']
    assert e.rows[-1]['key']==idw and e.rows[-1]['status']=='PASS'
    assert e.rows[-1]['candidate_prior']['score_interval'][0]>0


def test_soft_guidance_requires_explicit_model_policy(tmp_path):
    import pytest
    items,q,_=fixture();q['spatial_soft_guidance']=True
    parsed,audit=api.parse_inputs(items)
    with pytest.raises(ValueError,match='explicit registered'):
        SpatialSearch(SpatialTask.build(parsed,q,audit),tmp_path,BUDGET,11)
