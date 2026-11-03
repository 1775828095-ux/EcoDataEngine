from copy import deepcopy
import numpy as np
import pytest
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.task_validation import plan,_geometry,_calendar_distance,_source_presence
from ecodata.algorithms.autonomous_field import build_context
from ecodata.algorithms.profile_matching import prepare_model_request
from ecodata.algorithms.reliable_selection import make_policy,protected_order
from test_current_input import setup


def protocol(inputs,q):
    q,_=prepare_model_request(q)
    return plan(inputs,q,build_context(inputs,q)[0])


def test_boundary_distance_cannot_alias_half_image_with_edge():
    assert _geometry(6,0,(12,24),(24,24))!=_geometry(1,0,(12,24),(24,24))


def test_calendar_wrap_is_not_a_year_discontinuity():
    assert _calendar_distance('2024-12-30','2025-01-02')==3


def test_point_task_really_uses_point_masks(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=6);fine.ndvi.values[3,2,2]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    folds,p=protocol(inputs,q)
    assert p['tasks']==['point_missing'] and p['design']=='point_missing'
    assert folds[0]['confirmation'] is not None
    assert p['isolation_applied'] and p['confirmation_points']>0
    assert p['search_points']==p['confirmation_points']==1
    assert all(np.isfinite(folds[0]['validation'].reference.values).sum((1,2))<64)
    result=api.process_current_input(inputs,q,tmp_path/'point_public',seed=11,
        budget=dict(objective_limit=200,total_call_limit=200,probe_call_limit=50,proposal_limit=5000,wall_seconds=60))
    assert result['status']=='PASS' and result['final_validation_best_delivered']
    assert result['stop_reason']!='INSUFFICIENT_MATCHED_VALIDATION_FOR_PROMOTION'
    assert len(__import__('json').loads((tmp_path/'point_public/mutation_trace.json').read_text(encoding='utf-8')))>1


def test_smaller_unmatched_component_is_not_silently_certified(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=12)
    fine.ndvi.values[5,2:5,2:5]=np.nan
    fine.ndvi.values[:,0,0:2]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    _,p=protocol(inputs,q)
    assert p['support_limited']
    assert any(s['status']=='INSUFFICIENT' for s in p['task_support'])


def test_spatial_masks_stay_at_actual_position_and_local_calendar(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=12);fine.ndvi.values[6,2:4,2:5]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    folds,p=protocol(inputs,q)
    assert p['support_level']=='TASK_MATCHED_INTERNAL' and len(folds)==2
    for fold in folds:
        detail=fold['protocol']
        assert detail['placement'][1:]==[2,2]
        assert detail['calendar_distance_days']<=8
        assert detail['independent_source_presence_matched']


def test_independent_source_availability_is_part_of_context(monkeypatch,tmp_path):
    fine,coarse,_,_=setup(monkeypatch,tmp_path,t=6)
    coarse.ndvi.values[:3]=np.nan
    sources=[(0,'primary',fine),(1,'background',coarse)]
    assert _source_presence(sources,0,fine,fine.time.values[1],fine.time.values[1],'ndvi')==[False]
    assert _source_presence(sources,0,fine,fine.time.values[4],fine.time.values[4],'ndvi')==[True]


def test_scale_recovery_and_alias_isolation_are_composed_and_executed(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8);fine.ndvi.values[4,2:4,2:4]=np.nan
    inputs=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('alias.nc','background',fine.to_netcdf())]
    q['target_lat']=np.linspace(float(fine.lat[0])-.000075,float(fine.lat[-1])+.000075,16).tolist()
    q['target_lon']=np.linspace(float(fine.lon[0])-.000075,float(fine.lon[-1])+.000075,16).tolist()
    folds,p=protocol(inputs,q)
    assert p['tasks']==['scale_refinement','spatial_block']
    assert p['recovery_support']=='TASK_MATCHED_INTERNAL' and p['support_limited']
    assert len(folds)==3 and p['isolation_applied']
    assert folds[0]['protocol']['design']=='reduced_resolution_scale'
    assert all(f['protocol']['masked_input_indices']==[0,1] for f in folds[1:])
    graph=dict(records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        a=dict(operation='temporal.raster_linear',parents=['s0'],parameters={}),
        b=dict(operation='framewise.raster_task.spatial.nearest',parents=['a'],parameters={})),key='b')
    result=api.evaluate_graph_bank(inputs,q,dict(combo=graph),tmp_path/'mixed',budget=dict(objective_limit=200,wall_seconds=60))
    assert result['product'] and result['best']['metrics']['complete_required_folds']
    assert result['task_acceptance']['status']=='EVIDENCE_INSUFFICIENT'
    assert result['promotion_guard']['support_sufficient'] is False


@pytest.mark.parametrize('support,losses,expected',[(False,[.8,.8],'trial'),(True,[.2,1.1],'trial'),(True,[.8,.8],'trial'),(False,[1.2,1.2],'base'),(False,[1.,1.],'base')])
def test_complete_measured_loss_selects_best_without_support_veto(support,losses,expected):
    from ecodata.algorithms.reliable_selection import aggregate
    policy=make_policy({},[0.,1.],calibration=dict(evidence_role='development',evidence='Test exact tolerance',scope={}))
    def row(key,values,nodes):return dict(key=key,status='ELIGIBLE_OUTPUT',algorithm_nodes=nodes,
        metrics=aggregate([dict(fold_id=k,status='PASS',metrics=dict(RMSE=v)) for k,v in zip(['a','b'],values)],policy,['a','b']))
    baseline=row('base',[1.,1.],1);trial=row('trial',losses,3)
    assert protected_order([trial,baseline],policy,baseline,support_sufficient=support)[0]['key']==expected


def test_one_valid_fold_can_rank_without_claiming_representativeness():
    from ecodata.algorithms.reliable_selection import aggregate
    policy=make_policy({},[0.,1.],calibration=dict(evidence_role='development',evidence='Test exact tolerance',scope={}))
    def metric(v):return aggregate([dict(fold_id='a',status='PASS',metrics=dict(RMSE=v))],policy,['a'])
    baseline=dict(key='base',status='ELIGIBLE_OUTPUT',algorithm_nodes=1,metrics=metric(1.))
    trial=deepcopy(baseline);trial.update(key='trial',algorithm_nodes=4,metrics=metric(.1))
    assert protected_order([trial,baseline],policy,baseline,support_sufficient=False)[0]['key']=='trial'


def test_real_fold_execution_can_promote_consistent_improvement(monkeypatch,tmp_path):
    from ecodata.algorithms.field_mutation import GraphMutation
    from ecodata.algorithms.autonomous_field import field_registry
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=24)
    fine.ndvi.values[:]=.2+np.arange(24)[:,None,None]*.01+np.arange(8)[None,None,:]*.001
    fine.ndvi.values[14:17]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    request,model=prepare_model_request(q);native=build_context(inputs,request)[0]
    specs,p=plan(inputs,request,native);assert not p['support_limited'] and len(specs)>=2
    folds=[]
    for f in specs:
        ctx=build_context(f['inputs'],f['request'])[0]
        folds.append(dict(id=f['id'],context=ctx,registry=field_registry(api.platform_registry(),ctx),reference=f['validation'].reference,request=f['request']))
    first=folds[0];folder=tmp_path/'real';folder.mkdir()
    search=GraphMutation(first['context'],first['registry'],request,model,first['reference'],folder,
        dict(objective_limit=100,wall_seconds=60,proposal_limit=1000),11,folds=folds,
        selection_policy=make_policy(request,fine.ndvi.values),profile_context=native)
    try:
        nearest=search.intern('temporal.raster_nearest',['s0'],{})
        linear=search.intern('temporal.raster_linear',['s0'],{})
        base=search.execute(nearest,'regression');assert base['status']=='ELIGIBLE_OUTPUT'
        search.promotion_baseline=base;search.promotion_support=True
        trial=search.execute(linear,'regression');assert trial['status']=='ELIGIBLE_OUTPUT'
        assert search.best()['key']==linear
        assert all(x<y for x,y in zip(trial['metrics']['fold_losses'],base['metrics']['fold_losses']))
        search.promotion_support=False
        assert search.best()['key']==linear
    finally:search.close()
