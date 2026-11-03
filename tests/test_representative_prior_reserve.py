from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.task_validation import _ring_profile,plan
from ecodata.algorithms.candidate_portraits import CandidatePortraits
from ecodata.algorithms.search_accounting import SearchAccount,SearchBudgetExceeded
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.autonomous_field import build_context
from ecodata.algorithms.raster_gap_prediction import GridGapRecovery
from ecodata.algorithms.raster_temporal import RasterTimeInterpolation
from ecodata.algorithms.raster_reconstruction_task import NonnegativeRaster
from ecodata.algorithms.registry import AlgorithmRegistry
from test_current_input import setup
from ecodata.algorithms.search_stopping import SearchStopper


def test_ring_excludes_held_values():
    a=np.arange(4*8*8,dtype=float).reshape(4,8,8);template=np.ones((1,2,2),bool)
    before=_ring_profile(a,1,3,3,template,10.)
    a[1,3:5,3:5]=100000
    np.testing.assert_equal(before,_ring_profile(a,1,3,3,template,10.))


@pytest.mark.parametrize('edge',[False,True])
def test_blocks_match_boundary_instead_of_first_top_row(monkeypatch,tmp_path,edge):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8)
    y=0 if edge else 3;fine.ndvi.values[4,y:y+2,3:5]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf());request,_=prepare_model_request(q)
    folds,p=plan(inputs,request,build_context(inputs,request)[0])
    assert p['support_level']=='TASK_MATCHED_INTERNAL'
    for f in folds:
        assert f['protocol']['boundary_contacts']==[edge,False,False,False]
        assert np.isfinite(f['protocol']['visible_ring_distance'])


def test_shared_cap_really_preserves_final_execution():
    account=SearchAccount(10,30,total_calls=10);account.configure_deployment(10,30)
    account.protect_deployment(3,1)
    for i in range(7):account.after(account.before('op','search',{}))
    with pytest.raises(SearchBudgetExceeded):account.before('op','search',{})
    for i in range(3):account.after(account.before('op','deploy',{}))
    assert account.receipt()['used']==10 and account.receipt()['deployment_call_reserve']==3


def test_empty_proposals_cannot_stop_before_family_coverage():
    stopper=SearchStopper(dict(enabled=True,min_evaluations=10,patience_evaluations=10,no_novel_iterations=2,min_relative_gain=0.),0.)
    for _ in range(5):stopper.observe(0,None,empty_coverage_complete=False)
    assert stopper.reason is None
    stopper.observe(0,None,empty_coverage_complete=True)
    assert stopper.reason=='NO_NEW_NUMERICAL_EVALUATIONS'


def owner_for(context,model):
    registry=AlgorithmRegistry()
    for plugin in [GridGapRecovery('linear'),RasterTimeInterpolation('linear'),NonnegativeRaster()]:registry.register(plugin)
    records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0'))}
    for key,operation in [('space','reconstruction.grid_gap_linear'),('time','temporal.raster_linear'),('clip','raster_task.nonnegative')]:
        records[key]=dict(operation=operation,parents=['s0'],parameters={})
    owner=SimpleNamespace(field=dict(model=model),context=context,registry=registry,records=records,nodes=lambda key:['s0',key],node_count=lambda key:1)
    return owner


def test_profile_priority_changes_with_data_without_error_labels(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8);request,model=prepare_model_request(q)
    for axis in ['spatial','temporal']:
        values=np.zeros(fine.ndvi.shape)+.3
        if axis=='spatial':values+=np.arange(8)[None,None,:]*.02+np.arange(8)[:,None,None]*.0001
        else:values+=np.arange(8)[:,None,None]*.02+np.arange(8)[None,None,:]*.0001
        values[4,3:5,3:5]=np.nan;fine.ndvi.values[:]=values
        local=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),inputs[1]];context=build_context(local,request)[0]
        prior=CandidatePortraits(owner_for(context,model),context)
        preferred='time' if axis=='spatial' else 'space'
        other='space' if preferred=='time' else 'time'
        assert prior.order_key(preferred)<prior.order_key(other)
        assert prior.score(preferred)['validation_loss_used'] is False


def test_model_importance_changes_profile_choice(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8);fine.ndvi.values[0,:4]=-0.2;fine.ndvi.values[3,3:5,3:5]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf());request,model=prepare_model_request(q);context=build_context(inputs,request)[0]
    missing=model['state_policy']['scenario']['issues'][0]
    negative=dict(missing,attribute='grid_negative_fraction',metric='negative_fraction')
    for prefer in ['space','clip']:
        local=deepcopy(model);m=dict(missing,importance=100 if prefer=='space' else 1);n=dict(negative,importance=1 if prefer=='space' else 100)
        local['state_policy']['scenario']['issues']=[m,n]
        prior=CandidatePortraits(owner_for(context,local),context)
        assert prior.order_key(prefer)<prior.order_key('clip' if prefer=='space' else 'space')


@pytest.mark.parametrize('seed,calls',[(11,140),(47,320)])
def test_public_search_delivers_last_validation_choice_within_shared_budget(monkeypatch,tmp_path,seed,calls):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8);fine.ndvi.values[4,3:5,3:5]=np.nan
    q['soft_guidance']=True
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    result=api.process_current_input(inputs,q,tmp_path/'public',seed=seed,budget=dict(objective_limit=calls,total_call_limit=calls,probe_call_limit=30,proposal_limit=5000,wall_seconds=60))
    assert result['status'] in {'PASS','BUDGET_EXHAUSTED_BEFORE_MUTATION'} and result['final_validation_best_delivered']
    assert not (result['current_input_deployment'] or {}).get('checkpoint_fallback',False)
    assert result['total_cost']['used']<=calls
    assert result['total_cost']['reservation_history']
    assert result['candidate_prior']['validation_loss_used'] is False
    transitions=json.loads((tmp_path/'public/mutation_trace.json').read_text(encoding='utf-8'))
    assert result['stop_reason']!='INSUFFICIENT_MATCHED_VALIDATION_FOR_PROMOTION'
    if calls>140:assert any(row['strategy']=='portrait' for row in transitions)
    for row in transitions:
        if row['strategy']=='portrait':assert row['edit']['candidate_prior']['validation_loss_used'] is False


def test_public_initial_screening_actually_uses_changed_model_importance(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8)
    q['soft_guidance']=True
    fine.ndvi.values[0,:4]=-.2;fine.ndvi.values[3,3:5,3:5]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    issue=q['application_contract']['issues'][0]
    negative=dict(issue,attribute='grid_negative_fraction',metric='negative_fraction')
    rankings=[]
    for prefer in ['missing','negative']:
        local=deepcopy(q)
        local['application_contract']['issues']=[dict(issue,importance=100 if prefer=='missing' else 1),dict(negative,importance=1 if prefer=='missing' else 100)]
        folder=tmp_path/prefer
        api.process_current_input(inputs,local,folder,seed=11,budget=dict(objective_limit=140,total_call_limit=140,probe_call_limit=30,proposal_limit=5000,wall_seconds=60))
        rows=json.loads((folder/'initial_screening.json').read_text())['rows']
        graphs=json.loads((folder/'graphs.json').read_text())
        choices={}
        for position,row in enumerate(rows):
            operation=graphs[row['child']]['operation']
            if operation in {'reconstruction.grid_gap_linear','framewise.raster_task.nonnegative'}:
                choices.setdefault(operation,dict(position=position,score=row['candidate_prior']['score_interval'][0]))
        rankings.append(choices)
    fill='reconstruction.grid_gap_linear';clip='framewise.raster_task.nonnegative'
    assert rankings[0][fill]['position']<rankings[0][clip]['position']
    assert rankings[1][clip]['position']<rankings[1][fill]['position']
