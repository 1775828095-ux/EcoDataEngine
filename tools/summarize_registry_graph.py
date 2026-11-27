"""Summarize bounded integration evidence, never infer unexecuted successes."""
from collections import Counter
from pathlib import Path
import csv,hashlib,json,shutil
import xml.etree.ElementTree as ET
from ecodata.algorithms.platform_search import platform_registry
from ecodata.algorithms.spatial_registry_binding import operations

ROOT=Path('output/registry_graph_20261127')
read=lambda path:json.loads(path.read_text(encoding='utf8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')
registry=platform_registry();station=operations(registry)
cases={};scans={}
for name in ['spatial','temporal','multisource']:
    directory=ROOT/'real-v9'/name
    result=read(directory/'summary.json');cost=read(directory/'total_cost.json')
    scans[name]={row['id']:row for row in read(directory/'registry_discovery.json')['algorithms']}
    metrics=result['best']['metrics'] if result.get('best') else {}
    cases[name]=dict(status=result['status'],numerical_product=bool(result.get('product')),
        RMSE=metrics.get('RMSE'),MAE=metrics.get('MAE'),objective=metrics.get('objective'),
        objective_basis=metrics.get('objective_basis'),kernel_calls=result.get('kernel_calls'),
        actual_new_mutations=result.get('mutation_actual_new_executions'),attempts=result.get('attempts'),
        recommendations=result.get('recommendation_validation'),
        product=result.get('product'),product_sha256=hashlib.sha256(Path(result['product']).read_bytes()).hexdigest() if result.get('product') else None,
        total_cost=cost,recommendation_cost=read(directory/'recommendation_cost.json'))
rows=[]
for spec in registry.specs():
    row=dict(id=spec.id,status=spec.status,category=spec.category,backend=spec.backend,
        capabilities=';'.join(spec.capabilities),input_kinds=';'.join(spec.input_kinds),
        input_contract=json.dumps(spec.input_contract,ensure_ascii=False),
        requires_observations=spec.requires_observations,requires_auxiliary=spec.requires_auxiliary,
        requires_training=spec.requires_training,requires_weights=spec.requires_weights,
        declared_verification=spec.verification_status,
        discovery='ALL_REGISTRY_ENTRIES',DS_catalogue=True,
        field_graph_dispatch='CONDITIONAL_ACTUAL_COMPATIBILITY' if spec.status=='available' and spec.category!='validation' else 'NOT_PROCESSING_CANDIDATE',
        spatial_station_binding=spec.id in station,
        limitation=spec.notes or (spec.status if spec.status!='available' else 'Not every resource/parameter configuration was executed in bounded acceptance'))
    for name,scan in scans.items():
        evidence=scan.get(spec.id)
        row[name+'_root_status']=evidence['candidate_status'] if evidence else 'NOT_REGISTERED_IN_CONTEXT'
        row[name+'_reasons']=json.dumps(evidence['trials'] if evidence else [],ensure_ascii=False)
    rows.append(row)
with (ROOT/'algorithm_integration_matrix.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
pending=dict(
    routes=[dict(route='windowed_grid_recipe',status='LEGACY_RECOMMENDATION_PROTOCOL',
        reason='Native window/alignment preparation still consumes ordered steps; arbitrary graph recommendation is not advertised'),
      dict(route='pair_processing / observation_gap_search / original_engineering / native_reference_search',
        status='LEGACY_RECOMMENDATION_PROTOCOL',reason='Existing execution retained; arbitrary Registry graph transport not migrated in this acceptance')],
    interpretation='Every registration is discovered; conditional public field dispatch does not prove every plugin ran, nor every specialized frontend supports arbitrary graphs',
    spatial_condition='Station plugins bind actual retained observations; auxiliary regression/3D still require lawful covariates/z resources; missing resources fail explicitly',
    model_condition='No new checkpoint, auxiliary dataset, solver or training run supplied')
save(ROOT/'pending_bindings.json',pending)
tests=ET.parse(ROOT/'final3-targeted.xml').getroot().find('testsuite')
spatial=read(ROOT/'spatial_registry_real/summary.json')
spatial_events=spatial['total_cost']['events']
acceptance=dict(version='registry-graph-20261127-v1',registry_count=len(rows),
    registry_status=dict(Counter(s.status for s in registry.specs())),
    complete_discovery=True,all_plugins_numerically_verified=False,
    tests={key:tests.get(key) for key in ['tests','failures','errors','skipped','time']},
    cases=cases,spatial_registry_real=dict(status=spatial['status'],
       attempted_algorithms=sorted({e['algorithm'] for e in spatial_events}),
       failed_algorithms=sorted({e['algorithm'] for e in spatial_events if e['status']!='PASS'}),
       product=spatial.get('product')),
    desktop_browser_acceptance='NOT_RUN_CURRENT_VERSION',
    temporal_new_mutation_acceptance='NOT_ACHIEVED_IN_FROZEN_BUDGET',
    algorithm_superiority='NOT_TESTED',all_specialized_routes_migrated=False,
    historical_outputs_overwritten=False,new_algorithms_added=0,GOLEM='RETIRED_BACKEND_NOT_REINTRODUCED')
save(ROOT/'acceptance.json',acceptance)
files=['src/ecodata/algorithms/'+name for name in ['registered_graph.py','spatial_registry_binding.py','platform_search.py',
    'field_mutation.py','autonomous_decision.py','semantic_requirements.py','paired_station_search.py','spatial_fusion_task.py']]
files+=['src/ecodata/desktop/'+name for name in ['recommendations.py','service.py','__main__.py','station_grid.py',
    'independent_alignment.py','indexed_table_recipe.py','windowed_grid_recipe.py','release_gate.py']]
files+=['src/ecodata/interface/search_page.py','tests/test_registered_graph.py',
    'tools/verify_registry_graph.py','tools/summarize_registry_graph.py']
snapshot=ROOT/'source_final_v9';hashes={}
for name in files:
    source=Path(name);target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
save(ROOT/'source_fingerprints.json',hashes)
save(ROOT/'dependency_source_fingerprints.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(Path('src/ecodata').rglob('*.py'))})
print(json.dumps(dict(registry_count=len(rows),registry_status=acceptance['registry_status'],
    tests=acceptance['tests'],cases={k:{n:v[n] for n in ['status','RMSE','actual_new_mutations','kernel_calls']} for k,v in cases.items()},
    station=acceptance['spatial_registry_real']),ensure_ascii=False))
