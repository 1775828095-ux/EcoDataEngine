"""Independently replay saved interface acceptance; no effect experiment."""
import hashlib
import json
from pathlib import Path
import importlib.metadata as metadata
import xml.etree.ElementTree as ET
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.capabilities import describe_algorithm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/complementary_library_20261029'
IDS = ['bias.xsdba_scaling', 'spatial.denoise_tv', 'spatial.denoise_bilateral',
       'spatial.wavelet_lowpass', 'fusion.wavelet_detail_transfer']


def run():
    cfg = json.loads((OUT / 'real_2015/mechanism_config.json').read_text(encoding='utf-8'))
    sources = [api.SearchInput(Path(p['path']).name, p['role'], Path(p['path']).read_bytes()) for p in cfg['inputs']]
    for spec in cfg['inputs']:
        with xr.open_dataset(spec['path']) as ds:
            years = set(ds.time.dt.year.values.tolist())
            assert years <= ({2013, 2014} if spec['role'] in ('training_reference', 'historical_simulation') else {2015})
    result = api.execute_frozen_graph(sources, cfg['request'], cfg['graph'], OUT / 'independent_replay_final')
    receipt = json.loads((OUT / 'real_data_acceptance.json').read_text(encoding='utf-8'))
    original = Path(receipt['output'])
    with xr.open_dataset(original) as a, xr.open_dataset(result['product']) as b:
        xr.testing.assert_allclose(a, b)
    events = result['cost']['events']
    assert set(IDS) <= {e['algorithm'] for e in events if e['status'] == 'PASS'}
    assert all(e['status'] == 'PASS' for e in events)
    assert not result['reference_used'] and not result['selection_performed']
    registry = api.platform_registry()
    profiles = [dict(spec=registry.get(k).spec.as_dict(), capability=describe_algorithm(registry.get(k).spec).as_dict(),
                     empirical_gain='UNKNOWN', ranking_bonus=None) for k in IDS]
    (OUT / 'algorithm_profiles.json').write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
    files = ['src/ecodata/algorithms/raster_complements.py', 'src/ecodata/algorithms/plugins.py',
             'src/ecodata/algorithms/open_source_bias.py', 'src/ecodata/algorithms/platform_search.py',
             'tools/run_xsdba_adjustment.py', 'tests/test_raster_complements.py',
             'tools/verify_raster_complements.py', 'tools/audit_raster_complements.py',
             'requirements-raster-complements.txt', 'pyproject.toml']
    report = dict(status='PASS', replay_matches_saved_product=True, new_algorithms=IDS,
                  application_year=2015, training_years=[2013, 2014], final_test_read=False,
                  selection_reference_used=False, historical_training_reference_used=True,
                  scientific_advantage='NOT_EVALUATED', registered_calls=result['cost']['used'],
                  packages={k: metadata.version(k) for k in ['scikit-image', 'PyWavelets', 'xsdba']},
                  source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files})
    xml = OUT / 'test_results_final.xml'
    if xml.exists():
        report['test_suites'] = [dict(s.attrib) for s in ET.parse(xml).getroot().iter('testsuite')]
    (OUT / 'independent_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('INDEPENDENT REPLAY PASS', result['cost']['used'])


if __name__ == '__main__':
    run()
