"""Bounded real-file public-client acceptance; never a scientific benefit study."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import argparse

import numpy as np
import xarray as xr

from ecodata.algorithms.platform_search import submit_file_request, SearchInput, EngineeringMetric
from ecodata.interface.search_page import process_search_upload


def verify(config_path, output):
    config_path=Path(config_path);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    base=json.loads(config_path.read_text(encoding='utf-8'))
    paths=[Path(i['path']) for i in base['inputs']]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    results={}
    for name,priorities,hard in [('continuous',{'continuity':1},[]),('samples',{'no_mixing':1},[]),
                                 ('hard',{},['range_preservation','continuity'])]:
        config=deepcopy(base)
        config['request']['profile_options']['candidate_limit']=64
        config['request']['model_requirements']=dict(priorities=priorities,hard=hard,
            evidence='Acceptance task requirements on unchanged ERA5 field; not WRF-Chem sensitivity evidence')
        config['budget']['objective_limit']=20 if hard else 1
        (output/(name+'_request.json')).write_text(json.dumps(config,indent=2),encoding='utf-8')
        results[name]=submit_file_request(config,output/name)
        assert results[name]['best'] is not None
        if name=='continuous':
            inputs=[SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes(),i.get('metadata')) for i in config['inputs']]
            results['page']=process_search_upload(inputs,config['request'],EngineeringMetric(config['request']['variable']),
                output/'page',mode='Profile',budget=config['budget'])
            assert results['page']['best']['graph']==results[name]['best']['graph']
            assert results['page']['best']['matching']==results[name]['best']['matching']
    assert results['continuous']['best']['steps'][0].endswith('bilinear')
    assert results['samples']['best']['steps'][0].endswith('nearest')
    variable=base['request']['variable']
    with xr.open_dataset(results['continuous']['best']['product']) as a, xr.open_dataset(results['samples']['best']['product']) as b:
        difference=float(np.max(np.abs(a[variable].values-b[variable].values)))
        assert difference>0
    checks=[]
    for name,result in results.items():
        for row in result['rows']:
            if row['status']!='PASS':continue
            with xr.open_dataset(row['product']) as ds:
                assert np.isfinite(ds[variable]).all()
            checks.append(dict(case=name,key=row['key'],product_sha256=hashlib.sha256(Path(row['product']).read_bytes()).hexdigest(),
                               matches_record=hashlib.sha256(Path(row['product']).read_bytes()).hexdigest()==row['product_sha256']))
    assert all(c['matches_record'] for c in checks)
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=dict(status='PASS',input_sha256=before,source_config=str(config_path),
        cases={name:dict(steps=r['best']['steps'],counts=r['counts'],matching=r['best']['matching'],
                        graph=r['best']['graph'],accuracy_assessed=r['best'].get('accuracy_assessed')) for name,r in results.items()},
        actual_product_max_difference=difference,product_checks=checks,
        evidence_scope='Same actual field, changed task requirements, different numerical processing; coverage is NOT accuracy; page handler not external browser click')
    (output/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config');parser.add_argument('output')
    args=parser.parse_args()
    result=verify(args.config,args.output)
    print(json.dumps(dict(status=result['status'],cases={k:v['steps'] for k,v in result['cases'].items()},
                          actual_product_max_difference=result['actual_product_max_difference'])))
