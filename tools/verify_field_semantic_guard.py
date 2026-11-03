"""Verify post-score contract hardening preserves every frozen selected graph."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,difflib
import numpy as np
import xarray as xr
from evaluate_open_complementary import OUT,PROTOCOL,materialize,recipe,replay,load
from ecodata.algorithms.autonomous_field import measured
from ecodata.algorithms.numerical_search import save


def main():
    path=Path('src/ecodata/algorithms/autonomous_field.py')
    before=(OUT/'source_frozen'/path).read_text(encoding='utf-8');after=path.read_text(encoding='utf-8')
    block="""    for name in ('physical_variable','spatial_semantics','temporal_semantics'):
        expected=reference.attrs.get(name)
        if expected in (None,'UNKNOWN') or field.attrs.get(name)!=expected:
            raise ValueError('Output physical contract unknown or changed: '+name)
"""
    assert after.replace(block,'',1)==before,'Only the declared terminal guard may change'
    (OUT/'terminal_guard.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True))),encoding='utf-8')
    folders=[(stage,p) for stage in ('development','validation','test') for p in (OUT/'runs_v6'/stage).iterdir()]
    folders += [('test',p) for p in (OUT/'cost_matched').iterdir()]
    count=0
    for stage,folder in folders:
        if not (folder/'summary.json').exists():continue
        s=load(folder/'summary.json');index=int(folder.name.split('_')[0]);item=recipe(folder,s['best']['key'])
        inputs,request,reference,_,_=materialize(PROTOCOL[stage][index])
        output,context=replay(item,inputs,request);metrics=measured(output,reference,context)
        np.testing.assert_allclose(metrics['objective'],s['best']['metrics']['objective'],rtol=1e-10,atol=1e-10)
        count+=1
    save(OUT/'semantic_guard_audit.json',dict(status='PASS',selected_search_graphs_verified=count,
        sole_scientific_module_change='Additional terminal physical-semantics rejection',
        frozen_source_sha256=hashlib.sha256((OUT/'source_frozen'/path).read_bytes()).hexdigest(),
        current_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reasoning='Guard only narrows scored outputs; generation uses coordinates and node count, never score or terminal label. Every old minimum remains eligible with identical loss, so the selected minimum is unchanged.',
        full_search_rerun=False))


if __name__=='__main__':main()
