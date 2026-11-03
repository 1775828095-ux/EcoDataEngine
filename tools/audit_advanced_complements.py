"""Read back acceptance artifacts and independently replay the generated graph."""
from pathlib import Path
import json,hashlib,shutil
import xml.etree.ElementTree as ET
import importlib.metadata as metadata
import xarray as xr
from ecodata.algorithms import platform_search as api

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/complementary_five_20261029'


def run():
    folder=OUT/'pytest_final_boundary/test_public_registry_automatic0'
    cfg=json.loads((folder/'generated_config.json').read_text(encoding='utf-8'))
    inputs=[api.SearchInput(Path(p['path']).name,p['role'],Path(p['path']).read_bytes()) for p in cfg['inputs']]
    # This is the saved synthetic test policy, never a new production policy.
    from ecodata.algorithms import attribute_registry
    original_registry=attribute_registry.ROOT
    attribute_registry.ROOT=folder/'registries'
    try:result=api.execute_frozen_graph(inputs,cfg['request'],cfg['graph'],OUT/'generated_independent_replay')
    finally:attribute_registry.ROOT=original_registry
    with xr.open_dataset(folder/'automatic_graph_deploy/product.nc') as a,xr.open_dataset(result['product']) as b:xr.testing.assert_allclose(a,b)
    methods={'temporal.whittaker','reconstruction.dineof','fusion.guided_filter','spatial.arosics_coreg','fusion.starfm'}
    assert methods<={e['algorithm'] for e in result['cost']['events'] if e['status']=='PASS'}
    tests={}
    for filename in ['test_results_release.xml','test_results_final_boundary.xml']:
        for test in ET.parse(OUT/filename).getroot().iter('testcase'):
            assert not list(test.iter('failure')) and not list(test.iter('error')) and not list(test.iter('skipped'))
            tests[(test.attrib.get('classname'),test.attrib['name'])]=True
    assert len(tests)==75
    real=json.loads((OUT/'real_development_acceptance.json').read_text(encoding='utf-8'))
    assert real['status']=='PASS' and real['replay_matches'] and not real['formal_test_used']
    files=['src/ecodata/algorithms/advanced_complements.py','src/ecodata/algorithms/plugins.py',
        'src/ecodata/algorithms/platform_search.py','src/ecodata/algorithms/raster_gap_prediction.py',
        'tools/run_dineof.R','tools/run_geo_complements.py','tools/run_starfm.py',
        'tools/verify_advanced_complements.py','tools/audit_advanced_complements.py',
        'tests/test_advanced_complements.py','requirements-advanced-complements.txt','pyproject.toml',
        'external/sinkr/R/dineof.R','external/starfm4py/src/starfm4py.py','external/geo-runtime/verified.json']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    for p in files:
        dest=OUT/'source_final'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
        assert dest.read_bytes()==(ROOT/p).read_bytes()
    registry=api.platform_registry()
    assert all(registry.get(k).spec.status=='available' for k in methods)
    report=dict(status='PASS',unique_tests_passed=len(tests),generated_graph_replay_matches=True,
        five_methods_executed=sorted(methods),replay_registered_calls=result['cost']['used'],
        runtime=json.loads((ROOT/'external/geo-runtime/verified.json').read_text(encoding='utf-8')),
        main_packages={k:metadata.version(k) for k in ['numpy','opencv-contrib-python-headless','dask','zarr']},
        source_sha256=hashes,scientific_advantage='NOT_EVALUATED',formal_test_used=False,
        real_development_replay_matches=True)
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('INDEPENDENT AUDIT PASS',len(tests),'UNIQUE TESTS; FIVE GENERATED METHODS REPLAYED')


if __name__=='__main__':run()
