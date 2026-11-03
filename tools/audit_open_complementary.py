"""Independent graph replay and saved-product audit, not an experiment winner."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,hashlib,collections
import numpy as np
import xarray as xr
from evaluate_open_complementary import OUT,PROTOCOL,materialize,recipe,replay,load
from ecodata.algorithms.autonomous_field import measured
from ecodata.algorithms.numerical_search import save


def main():
    manifest=load(OUT/'source_manifest.json')
    for name,sha in manifest.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    rows=[]
    folders=list((OUT/'runs_v6/test').iterdir())
    if (OUT/'cost_matched').exists():folders+=list((OUT/'cost_matched').iterdir())
    for folder in sorted(folders):
        if not (folder/'summary.json').exists():continue
        summary=load(folder/'summary.json');index=int(folder.name.split('_')[0])
        candidate_rows=[json.loads(x) for x in (folder/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(candidate_rows)==summary['attempts']
        assert sum('key' in r for r in candidate_rows)==summary['executed_candidates']
        assert sum(r['status']=='EXECUTION_FAILED' for r in candidate_rows)==summary['failed_candidates']
        item=recipe(folder,summary['best']['key'])
        search,request,reference,full,_=materialize(PROTOCOL['test'][index])
        output,context=replay(item,search,request)
        metrics=measured(output,reference,context)
        np.testing.assert_allclose(metrics['objective'],summary['best']['metrics']['objective'],rtol=1e-10,atol=1e-10)
        deployed,_=replay(item,full,request)
        with xr.open_dataset(summary['product']) as f:saved=f.load()
        np.testing.assert_allclose(saved.air_temperature,deployed.air_temperature,rtol=1e-12,atol=1e-12)
        assert np.isfinite(saved.air_temperature).all()
        origins={r['parameters']['source'] for r in item['records'].values() if r['operation']=='source'}
        rows.append(dict(run=str(folder),replayed_search_objective=metrics['objective'],
            product_replay='PASS',sources=sorted(origins),actual_nodes=len(item['records'])-len(origins),
            true_kernel_calls=summary['kernel_calls'],complete_space=summary['complete_space']))
        print('AUDITED',folder.name,flush=True)
    for entry in load(OUT/'predictions_frozen.json'):
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    result=dict(status='PASS',replayed_runs=len(rows),all_frozen_product_hashes_match=True,
        source_hashes_match=True,selection_reference_only_visible_anchor=True,rows=rows)
    save(OUT/'independent_audit.json',result)


if __name__=='__main__':main()
