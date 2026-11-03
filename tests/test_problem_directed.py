"""Behavioral checks over public typed grammar; synthetic boundary inputs only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from test_platform_integration import prepare,grid,request
from ecodata.algorithms.platform_search import EngineeringMetric,SearchInput,submit_search
from ecodata.algorithms.numerical_search import record

def test_missing_output_is_connected_by_real_port_dependencies():
    _,_,decision,factory,_,_=prepare()
    options,attempts=factory.legacy_problem_candidates(decision,factory.initial(),limit=4,attempt_limit=10)
    assert options and len(attempts)<=10
    assert any(action['type']=='connect' for _,action in options)
    for graph,action in options:
        assert factory.inspect(graph)['gap']==0
        assert action['graph_after_gap']<action['graph_before_gap']
        assert action['connections']
        assert not decision.assess_graph(factory,graph)['hard_reject']

def test_optional_stages_insert_replace_and_simplify_actual_ancestors():
    _,_,decision,factory,_,_=prepare()
    factory.reverse_limit=64
    complete,_=factory.legacy_problem_candidates(decision,factory.initial(),limit=8,attempt_limit=20)
    observed=set()
    for graph,_ in complete:
        options,_=factory.legacy_problem_candidates(decision,graph,limit=8,attempt_limit=20)
        for child,action in options:
            observed.add(action['type'])
            assert factory.inspect(child)['gap']==0
            assert action['parent_graph']!=action['child_graph']
    assert {'insert','replace','simplify'}<=observed

def test_problem_rules_change_with_actual_semantics_not_file_names():
    _,_,a,fa,_,_=prepare()
    _,_,b,fb,_,_=prepare(grid(True),request(True))
    assert a.requirements.problems and b.requirements.problems
    assert a.requirements.required!=b.requirements.required
    assert ('spatial.bilinear','run') in fa.adapters
    assert ('spatial.bilinear','run') not in fb.adapters
    assert all(p['status'] in {'CONFIRMED','UNKNOWN','WORKING_ASSUMPTION'} for p in a.requirements.problems)

def test_small_space_is_executed_not_lost_while_filling_population(tmp_path):
    import numpy as np,xarray as xr
    ds=xr.Dataset({'v':('time',np.arange(24.))},coords={'time':np.arange('2020-01-01','2020-01-02',dtype='datetime64[h]')})
    ds.v.attrs.update(units='K',data_kind='continuous',temporal_semantics='instantaneous')
    req=dict(task='temporal',variable='v',unit='K',semantic_type='continuous',temporal_semantics='instantaneous',parameters=dict(target_time_frequency='6h'))
    result=submit_search([SearchInput('unknown.nc','primary',ds.to_netcdf(engine='h5netcdf'))],req,EngineeringMetric('v'),tmp_path/'run',mode='C-directed',seed=11,
        budget=dict(objective_limit=4,proposal_limit=32,callback_limit=80,wall_seconds=10))
    assert result['best'] and result['counts']['real_executions']>=1
    transitions=json.loads((tmp_path/'run/operation_transitions.json').read_text(encoding='utf-8'))
    assert transitions['q_model_trained'] is False
    for row in transitions['records']:
        if row['reward'] is not None:assert row['reward_status']=='REAL_PAIRED_GRAPH_DIFFERENCE'
        else:assert not row['single_action_learning_label']

def test_feature_cache_tracks_values_and_training_diagnosis_requires_legal_pairs():
    import numpy as np,xarray as xr
    from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
    ds=grid();a=SemanticFeatureExtractor();first=a.extract(ds,'v');a.extract(ds,'v')
    assert a.cache_info['hit']
    ds.v.values[0,0]=np.nan;changed=a.extract(ds,'v')
    assert not a.cache_info['hit'] and changed.observed.value['missing_count']==1
    train=xr.DataArray([1.,2.,3.],dims='time',coords={'time':np.arange('2020-01-01','2020-01-04',dtype='datetime64[D]')},attrs=dict(units='K',split='train'))
    history=train.copy(deep=True);history.values+=2
    application=train.assign_coords(time=train.time+np.timedelta64(10,'D'))
    assert a.training_bias(train,history,application)['bias']==2
    assert a.training_bias(train,history,train)['status']=='UNKNOWN'
    history.attrs['split']='validation'
    assert a.training_bias(train,history,application)['status']=='UNKNOWN'
