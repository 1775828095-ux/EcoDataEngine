"""Replay saved decisions independently; never use external experiment labels."""
from pathlib import Path
from copy import deepcopy
import hashlib,json
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import build_context,field_registry,measured
from ecodata.algorithms.autonomous_decision import Executor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/current_input_repair_20261029'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    results=[]
    for folder in sorted((OUT/'real_inputs').iterdir()):
        platform=folder/'platform';summary=read(platform/'summary.json');saved=read(platform/'request.json')
        acceptance=read(folder/'acceptance.json');phase=acceptance['phase']
        original=read(ROOT/f'output/halfdate_reconstruction_20261028/cases/patch2_contiguous/config_{phase}.json')
        inputs=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in original['inputs']]
        assert [hashlib.sha256(i.content).hexdigest() for i in inputs]==acceptance['input_sha256']
        records=read(platform/'graphs.json');key=summary['best']['key'];pending=[key];subset={}
        while pending:
            node=pending.pop()
            if node in subset:continue
            subset[node]=records[node];pending+=records[node]['parents']
        request=deepcopy(saved['request']);request.pop('decision_policy',None)
        replay=api.execute_frozen_graph(inputs,request,dict(records=subset,key=key),folder/'independent_deployment')
        with xr.open_dataset(replay['product']) as f,xr.open_dataset(summary['product']) as g:
            xr.testing.assert_equal(f.load(),g.load())
        masked=[api.SearchInput(i.filename,i.role,(platform/f'self_validation_roots/source{n}.nc').read_bytes()) for n,i in enumerate(inputs)]
        context,_,_=build_context(masked,request)
        output,_=Executor(context,field_registry(api.platform_registry(),context)).execute(subset,key)
        with xr.open_dataset(platform/'self_confirmation_labels.nc') as f:reference=f.ndvi.load()
        aligned=output.ndvi.sel(time=reference.time);valid=np.isfinite(reference.values)
        rmse=float(np.sqrt(np.mean((aligned.values[valid].astype(float)-reference.values[valid].astype(float))**2)))
        np.testing.assert_allclose(rmse,summary['confirmation']['metrics']['RMSE'],rtol=1e-12,atol=1e-12)
        scores=measured(output,reference,context)
        np.testing.assert_allclose(scores['objective'],summary['confirmation']['metrics']['objective'],rtol=1e-12)
        with xr.open_dataset(original['inputs'][0]['path']) as f:visible=f.ndvi.load()
        with xr.open_dataset(platform/'self_validation_labels.nc') as f:search=f.ndvi.load()
        masks=[np.isfinite(a.reindex(time=visible.time).values) for a in [search,reference]]
        assert not (masks[0]&masks[1]).any()
        assert not ((masks[0]|masks[1])&~np.isfinite(visible.values)).any()
        for n in saved['self_validation']['masked_input_indices']:
            with xr.open_dataset(platform/f'self_validation_roots/source{n}.nc') as f:
                for ref in [search,reference]:
                    values=f.ndvi.sel(time=ref.time).values
                    assert np.isnan(values[np.isfinite(ref.values)]).all()
        results.append(dict(case=folder.name,status='PASS',exact_input_hashes=True,
            deployment_replay_equal=True,confirmation_RMSE=rmse,confirmation_recomputed_before_protection=True,
            search_confirmation_disjoint=True,natural_missing_not_truth=True,alias_mask_checked=True,
            algorithm_nodes=len([r for r in subset.values() if r['operation']!='source']),
            scientific_advantage='NOT_EVALUATED'))
    files=['src/ecodata/algorithms/current_input.py','src/ecodata/algorithms/field_mutation.py',
        'src/ecodata/algorithms/platform_search.py','src/ecodata/algorithms/search_accounting.py',
        'src/ecodata/interface/search_page.py','tests/test_current_input.py',
        'tools/verify_current_input_repair.py','tools/audit_current_input_repair.py']
    frozen=OUT/'source_frozen';frozen.mkdir(exist_ok=True)
    hashes={}
    for name in files:
        data=(ROOT/name).read_bytes();target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    report=dict(status='PASS',cases=results,source_sha256=hashes,external_test_error='NOT_EVALUATED')
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
