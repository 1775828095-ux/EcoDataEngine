"""Frozen functional boundary cases, not environmental accuracy evidence.

Only source A's grid/support changes; source B, target, variable, pool and seed
are fixed. Expected properties are declared before any numerical ranking.
"""
import sys,json,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,xarray as xr,pytest
from ecodata.algorithms.platform_search import make_context,platform_registry,bind_registry,RegistryEvaluator,EngineeringMetric
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from golem.core.optimisers.graph import OptGraph

def sources(mismatch=True,support='cell_mean',description=None):
    result=[]
    for role in ['primary','background']:
        axis=np.array([30.,32.,34.]) if role=='primary' and mismatch else np.array([31.,32.,33.])
        yy,xx=np.meshgrid(axis,axis+80,indexing='ij')
        ds=xr.Dataset({'aod':(('lat','lon'),.1+(yy-30)**2*.03+(xx-110)*.02),'uncertainty':(('lat','lon'),np.ones((3,3)))},coords={'lat':axis,'lon':axis+80},attrs={'crs':'EPSG:4326'})
        ds.aod.attrs.update(units='1',data_kind='continuous',spatial_semantics=support if role=='primary' else 'cell_mean',physical_variable='aod',physical_definition='AOD')
        ds.uncertainty.attrs['units']='1'
        if description:ds.attrs['description']=description
        result.append((role,ds))
    return result

def prepare(parsed=None,seed=11,bound=32):
    parsed=parsed if parsed is not None else sources()
    req=dict(task='multisource',variable='aod',unit='1',semantic_type='continuous',nonnegative=True,
        target_lat=[31.,32.,33.],target_lon=[111.,112.,113.],_local_problem_mode=True)
    # Same ordering rule as the public dispatcher, no filename-dependent ID.
    parsed=sorted(parsed,key=lambda item:({'primary':0,'background':1}[item[0]]))
    ctx,_,_=make_context(parsed,req);registry=platform_registry()
    decision=AlgorithmSelector(registry).decide(ctx,dict(operation='fusion',source_count=2))
    f,external,_=bind_registry(registry,ctx,decision,req)
    f.local_recipe_seed=seed;f.reverse_limit=bound;decision.bind_problem_ports(f,external)
    return req,registry,ctx,decision,f,external

def canonical_candidates(d,f,g=None,limit=8,attempts=24):
    return f.problem_candidates(d,g or f.initial(),limit=limit,attempt_limit=attempts,recipe_limit=32)

def test_one_source_grid_changes_only_its_local_problem_and_producers():
    *_,d0,f0,e0=prepare(sources(False));*_,d1,f1,e1=prepare(sources(True))
    local=[p for p in d1.requirements.problems if p['rule']=='source_grid_mismatch']
    assert len(local)==1 and local[0]['source_ids']==['role:primary:aod']
    assert local[0]['binding']['kind']!='validated_field'
    assert not [p for p in d0.requirements.problems if p['rule']=='source_grid_mismatch']
    assert f0.problem_resources['source1']==f1.problem_resources['source1']
    global0=next(p for p in d0.requirements.problems if p['rule']=='multi_source')
    global1=next(p for p in d1.requirements.problems if p['rule']=='multi_source')
    assert global0['id']==global1['id'] and global0['capabilities']==global1['capabilities']
    opts,attempts=canonical_candidates(d1,f1)
    relevant=[a for _,a in opts if a['problem_ids']==[local[0]['id']]]
    assert relevant and len(attempts)<=24
    assert any('prepare.grid.role:primary:aod.' in n.name for graph,_ in opts for n in graph.nodes)
    assert not any(n.name.startswith('prepare.grid.') for graph,_ in canonical_candidates(d0,f0)[0] for n in graph.nodes)

def test_single_actual_support_condition_changes_compatibility_not_whole_task():
    from test_platform_integration import prepare as spatial_prepare,grid
    original=grid();changed=original.copy(deep=True);changed.v.attrs['spatial_semantics']='point'
    _,_,d,f,_,_=spatial_prepare(original)
    _,_,d2,f2,_,_=spatial_prepare(changed)
    before={key[0] for key in f.adapters};after={key[0] for key in f2.adapters}
    assert 'spatial.conservative' in before and 'spatial.conservative' not in after
    assert {'spatial.nearest','spatial.bilinear'}<=after
    with pytest.raises(ValueError,match='spatial support'):prepare(sources(True,'point'))

def test_unrelated_metadata_and_input_order_do_not_change_candidates():
    *_,d,f,_=prepare()
    *_,d2,f2,_=prepare(list(reversed(sources(description='renamed description'))))
    assert d.requirements.problems==d2.requirements.problems
    a,_=canonical_candidates(d,f);b,_=canonical_candidates(d2,f2)
    assert {f.inspect(g)['key'] for g,_ in a}=={f2.inspect(g)['key'] for g,_ in b}

def test_local_edits_preserve_other_branch_and_name_exact_problem():
    *_,d,f,_=prepare()
    graph=canonical_candidates(d,f)[0][0][0]
    local=next(p for p in d.requirements.problems if p['rule']=='source_grid_mismatch')
    options,_=canonical_candidates(d,f,graph)
    actions=[a for _,a in options if a['problem_ids']==[local['id']]]
    assert actions
    for action in actions:
        assert action['source_ids']==['role:primary:aod']
        assert action['target_operation']=='fusion.uncertainty_weighted'
        assert action['target_port']=='source0'
        assert action['protected_ancestry']
        assert all(n['operation']!='source.source1' for n in action['added_nodes']+action['removed_nodes'])

def test_registration_order_is_irrelevant_and_truncated_producers_are_reachable():
    *_,d,f,_=prepare(bound=64)
    first={f.inspect(g)['key'] for g,_ in canonical_candidates(d,f)[0]}
    for choices in f.producers.values():choices.reverse()
    f._memo.clear()
    assert first=={f.inspect(g)['key'] for g,_ in canonical_candidates(d,f)[0]}
    kind=next(p['binding']['kind'] for p in d.requirements.problems if p['rule']=='source_grid_mismatch')
    reached=set()
    for seed in range(10):
        f.local_recipe_seed=seed;f.reverse_limit=1;f._memo.clear()
        reached.update(r[0] for _,r in f.recipes(kind))
    assert reached=={key[0] for key,_ in f.producers[kind]}

def test_real_upstream_output_is_consumed_and_other_branch_is_reused(tmp_path):
    req,registry,ctx,d,f,external=prepare()
    graph=canonical_candidates(d,f)[0][0][0]
    evaluator=RegistryEvaluator(registry,f,external,d,EngineeringMetric('aod'),tmp_path/'a')
    first=evaluator.execute(graph,'first')
    assert first['status']=='PASS'
    with xr.open_dataset(first['product'],engine='h5netcdf') as ds:a=ds.aod.values.copy()
    from copy import deepcopy
    changed=deepcopy(external);changed['source0'].value.dataset.aod.values+=.2
    # Numerical executor input cache keys must change without changing source B.
    assert external['source1'].fingerprint()==changed['source1'].fingerprint()
    evaluator.external=changed
    second=evaluator.execute(graph,'second')
    assert second['status']=='PASS'
    with xr.open_dataset(second['product'],engine='h5netcdf') as ds:b=ds.aod.values.copy()
    np.testing.assert_allclose(b-a,.1,atol=1e-10)
    assert any(t.get('operation')=='source.source1' and t.get('cache_hit') for t in second['node_trace'])
    assert any(t.get('operation')=='source.source0' and not t.get('cache_hit') for t in second['node_trace'])

def test_downstream_consumes_modified_intermediate_not_raw_input(tmp_path):
    req,registry,ctx,d,f,external=prepare()
    graph=canonical_candidates(d,f)[0][0][0]
    evaluator=RegistryEvaluator(registry,f,external,d,EngineeringMetric('aod'),tmp_path/'intervention')
    first=evaluator.execute(graph,'baseline');assert first['status']=='PASS'
    node=next(n for n in graph.nodes if n.name.startswith('prepare.grid.'))
    adapter=f.adapter(node);original=adapter.execute
    def intervention(values,p,e):
        result=original(values,p,e)
        result['value'].value.dataset.aod.values+=.2
        return result
    adapter.execute=intervention;adapter.version+='-functional-intervention'
    second=evaluator.execute(graph,'intermediate_changed');assert second['status']=='PASS'
    with xr.open_dataset(first['product'],engine='h5netcdf') as a,xr.open_dataset(second['product'],engine='h5netcdf') as b:np.testing.assert_allclose(b.aod-a.aod,.1,atol=1e-10)
    assert any(t.get('operation')=='source.source1' and t.get('cache_hit') for t in second['node_trace'])

def test_attempts_and_duplicates_are_bounded():
    *_,d,f,_=prepare()
    options,attempts=canonical_candidates(d,f,limit=2,attempts=3)
    assert len(options)<=2 and len(attempts)<=3

def test_two_branch_problems_have_distinct_ids_and_precise_action_attribution():
    data=sources();data[1]=('background',data[1][1].assign_coords(lat=[30.,32.,34.],lon=[110.,112.,114.]))
    *_,d,f,_=prepare(data)
    local=[p for p in d.requirements.problems if p['rule']=='source_grid_mismatch']
    assert len(local)==2 and len({p['id'] for p in local})==2
    graph=canonical_candidates(d,f)[0][0][0]
    options,_=canonical_candidates(d,f,graph)
    by_id={p['id']:p for p in local};seen=set()
    for _,action in options:
        if action['problem_ids'][0] not in by_id:continue
        p=by_id[action['problem_ids'][0]];seen.add(p['id'])
        assert len(action['problem_ids'])==1 and action['source_ids']==p['source_ids']
        expected='source0' if 'role:primary:aod' in p['source_ids'] else 'source1'
        assert action['target_port']==expected and action['target_operation']=='fusion.uncertainty_weighted'
        assert action['dependency_paths'] and action['added_edges']
    assert seen==set(by_id)

def test_source_content_invalidation_does_not_reassign_other_source():
    *_,d,f,e=prepare()
    changed=sources();changed[0][1].aod.values[0,0]=np.nan
    *_,d2,f2,e2=prepare(changed)
    assert f.problem_resources['source0']['source_id']==f2.problem_resources['source0']['source_id']
    assert f.problem_resources['source0']['fingerprint']!=f2.problem_resources['source0']['fingerprint']
    assert f.problem_resources['source1']==f2.problem_resources['source1']
    p=next(p for p in d2.requirements.problems if p['rule']=='source_grid_mismatch')
    assert p['measure']['input_statistics']['missing_count']==1

def test_old_strategies_do_not_inherit_local_enumeration():
    from test_platform_integration import prepare as old_prepare
    *_,f,external,excluded=old_prepare()
    assert not hasattr(f,'local_recipe_seed')
    assert callable(f.legacy_problem_candidates)

def test_optional_constraint_insert_and_simplify_keep_real_upstream_program():
    from test_platform_integration import prepare as old_prepare
    _,_,d,f,external,_=old_prepare()
    d.bind_problem_ports(f,external);f.local_recipe_seed=11;f.reverse_limit=64
    options,_=canonical_candidates(d,f)
    kinds=set()
    for graph,_ in options:
        for _,action in canonical_candidates(d,f,graph)[0]:
            kinds.add(action['type'])
            if action['type']=='simplify':
                assert not action['problem_ids'] and action['goal']['type']=='remove_optional_processing'
            if action['type']=='insert':
                assert action['added_edges'] and action['added_nodes']
    assert {'insert','simplify'}<=kinds

@pytest.mark.parametrize('declaration,expected',[
    (dict(operation='compute',registered=False,capabilities=[]),'REJECT'),
    (dict(operation='unrecognised_contract'),'UNKNOWN')])
def test_local_required_capability_cannot_be_borrowed_from_final_fusion(declaration,expected):
    *_,d,f,_=prepare()
    graph=canonical_candidates(d,f)[0][0][0]
    node=next(n for n in graph.nodes if n.name.startswith('prepare.grid.'))
    f.adapter(node).decision_contract=declaration
    assessment=d.assess_graph(f,graph)
    assert assessment['acceptance_status']==expected
    assert assessment['local_acceptance'][0]['status'] in {'MISSING','UNKNOWN'}
    with pytest.raises(Exception,match='Task requirements'):d.require_graph(f,graph)
