"""Readback, controlled intervention and evidence capture; no search loop."""
from pathlib import Path
import json,sys,hashlib,shutil,zipfile,io
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,xarray as xr
from ecodata.algorithms.platform_search import parse_inputs,make_context,platform_registry,bind_registry,RegistryEvaluator,EngineeringMetric
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.algorithms.numerical_search import save,restore
from accept_local_problem import OUT,inputs_for

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def scrub(x):
    if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in {'seconds','elapsed'}}
    if isinstance(x,list):return [scrub(v) for v in x]
    return x

def main():
    summaries={};pairs={}
    for case,clients in [('aligned',['toolkit','page']),('mismatch',['toolkit','page']),('spatial',['toolkit']),('temporal',['toolkit']),('multisource',['toolkit','page'])]:
        results=[]
        for client in clients:
            folder=OUT/'runs'/f'{case}_{client}_v5';result=load(folder/'summary.json')
            assert result['best'] and result['best']['file_readback']=='PASS'
            actual_inputs,_,_=inputs_for(case)
            expected={x['role']:x['sha256'] for x in load(folder/'protocol.json')['inputs']}
            assert expected=={x.role:hashlib.sha256(x.content).hexdigest() for x in actual_inputs}
            for row in result['rows']:
                if row['status']!='PASS':continue
                with xr.open_dataset(row['product'],engine='h5netcdf') as ds:
                    variable='pm25' if case=='multisource' else 'tp' if case=='temporal' else 'aod'
                    assert np.isfinite(ds[variable]).all()
            summaries[case+'_'+client]=dict(counts=result['counts'],objective=result['best']['objective'],seconds=result['seconds'],
                candidate_generation_seconds=result['candidate_generation_seconds'],preparation_costs=result['preparation_costs'],
                selected_graph=result['best']['graph'],product=result['best']['product'])
            results.append((folder,result))
        if len(results)==2:
            (left,a),(right,b)=results
            assert scrub(load(left/'decision.json'))==scrub(load(right/'decision.json'))
            assert a['candidate_domain']==b['candidate_domain']
            assert scrub(a['proposals'])==scrub(b['proposals'])
            assert a['best']['graph']==b['best']['graph']
            with xr.open_dataset(a['best']['product'],engine='h5netcdf') as x,xr.open_dataset(b['best']['product'],engine='h5netcdf') as y:xr.testing.assert_equal(x,y)
            pairs[case]='PASS: cosmetic filenames and upload order changed; same decisions, proposals, graph and product'
    aligned=load(OUT/'runs/aligned_toolkit_v5/local_port_contracts.json')
    mismatch=load(OUT/'runs/mismatch_toolkit_v5/local_port_contracts.json')
    assert aligned['resources']['source1']==mismatch['resources']['source1']
    assert aligned['resources']['source0']['source_id']==mismatch['resources']['source0']['source_id']
    changed=[a for a in mismatch['adapters'] if a['operation'].startswith('prepare.grid.')]
    assert changed and not any(a['operation'].startswith('prepare.grid.') for a in aligned['adapters'])
    # Use a graph actually selected by official GOLEM, not a hand-authored route.
    inputs,request,_=inputs_for('mismatch');request=dict(request,_local_problem_mode=True)
    parsed,_=parse_inputs(inputs);ctx,_,_=make_context(parsed,request);registry=platform_registry()
    semantics=SemanticFeatureExtractor().extract(ctx.sources[0].dataset,'aod',confirmed=dict(variable_class='continuous',unit='1',nonnegative=True))
    decision=AlgorithmSelector(registry).decide(ctx,dict(operation='fusion',source_count=2,working_assumptions=request['working_assumptions']),semantics=semantics)
    factory,external,_=bind_registry(registry,ctx,decision,request);factory.local_recipe_seed=11;factory.reverse_limit=64
    decision.bind_problem_ports(factory,external)
    graph=restore(summaries['mismatch_toolkit']['selected_graph'])
    options,attempts=factory.problem_candidates(decision,graph,limit=8,attempt_limit=24)
    local=next(p for p in decision.requirements.problems if p['rule']=='source_grid_mismatch')
    actions=[a for _,a in options if a['problem_ids']==[local['id']]]
    assert actions and all(a['target_operation']=='fusion.uncertainty_weighted' and a['target_port']=='source0' for a in actions)
    assert all(a['protected_ancestry'] and not any(n['operation']=='source.source1' for n in a['added_nodes']+a['removed_nodes']) for a in actions)
    save(OUT/'real_local_edit_candidates.json',dict(problem=local,attempts=attempts,actions=actions))
    evaluator=RegistryEvaluator(registry,factory,external,decision,EngineeringMetric('aod'),OUT/'intervention')
    input_files=[OUT/'inputs/mismatch_primary.nc',OUT/'inputs/mismatch_background.nc']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    baseline=evaluator.execute(graph,'baseline');assert baseline['status']=='PASS'
    stage=next(n for n in graph.nodes if n.name.startswith('prepare.grid.'))
    adapter=factory.adapter(stage);original=adapter.execute
    def intervention(values,p,e):
        out=original(values,p,e);out['value'].value.dataset.aod.values+=.2
        return out
    adapter.execute=intervention;adapter.version+='-explicit-functional-intervention'
    altered=evaluator.execute(graph,'upstream_output_plus_0_2');assert altered['status']=='PASS'
    with xr.open_dataset(baseline['product'],engine='h5netcdf') as a,xr.open_dataset(altered['product'],engine='h5netcdf') as b:
        difference=(b.aod-a.aod).values;np.testing.assert_allclose(difference,.1,atol=1e-10)
    assert any(t.get('operation')=='source.source1' and t.get('cache_hit') for t in altered['node_trace'])
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    save(OUT/'real_intermediate_intervention.json',dict(status='PASS',scope='Functional intervention on real-input graph, NOT an environmental accuracy experiment',
        changed_operation=stage.name,input_files_unchanged=True,upstream_output_delta=.2,expected_downstream_delta=.1,
        actual_min_delta=float(difference.min()),actual_max_delta=float(difference.max()),unrelated_branch_cache_reused=True,
        baseline=baseline,altered=altered))
    browser_folder=ROOT/'output/platform_search_requests/ad239d60a59640109ad3207eb2d8e72d'
    browser=load(browser_folder/'summary.json');assert browser['mode']=='C-local' and browser['best']['file_readback']=='PASS'
    downloaded=Path('C:/Users/LX/Downloads/platform_search (2).zip')
    with zipfile.ZipFile(downloaded) as archive:
        assert archive.testzip() is None
        products=[n for n in archive.namelist() if n.endswith('product.nc')];assert products
        with xr.open_dataset(io.BytesIO(archive.read(products[0])),engine='h5netcdf') as ds:assert np.isfinite(ds.aod).all()
    browser_evidence=dict(status='PASS',request_directory=str(browser_folder),download=str(downloaded),download_sha256=hashlib.sha256(downloaded.read_bytes()).hexdigest(),
        downloaded_product_readback='PASS',counts=browser['counts'],screenshot=str(OUT/'browser_current.png'),mode=browser['mode'])
    sources={}
    for path in (OUT/'source_before').rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(OUT/'source_before')
        if relative.parts[0] not in {'src','docs'}:continue
        sources[str(relative)]=dict(before_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),after_sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest())
        if relative.parts[0]=='src':
            destination=OUT/'source_executed'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,destination)
    for name in ['tools/accept_local_problem.py','tools/verify_local_problem.py','tests/test_local_problem.py','tests/test_problem_directed.py']:
        path=ROOT/name;destination=OUT/'source_executed'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        sources[name]=dict(after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(OUT/'verification.json',dict(runs=summaries,client_pairs=pairs,controlled_grid_change=dict(status='PASS',source='role:primary:aod',unrelated_source='role:background:aod',
        candidate_algorithms=sorted(a['decision']['algorithm_id'] for a in changed),scope='Two resolutions of same real product, not independent data-source benefit'),
        real_local_edit='PASS',real_intermediate_intervention='PASS',source_fingerprints=sources,browser=browser_evidence,input_hashes_rechecked=True,q_model_trained=False))
    print('VERIFIED',pairs)

if __name__=='__main__':main()
