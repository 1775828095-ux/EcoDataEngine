"""Mechanism acceptance on the exact previously failing inputs, no truth files."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json,hashlib
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/halfdate_reconstruction_20261028/cases/patch2_contiguous'
OUT=ROOT/'output/current_input_repair_20261029'


def run_one(phase,seed):
    config=json.loads((OLD/f'config_{phase}.json').read_text(encoding='utf-8'))
    inputs=[api.SearchInput(Path(p['path']).name,p['role'],Path(p['path']).read_bytes()) for p in config['inputs']]
    request=deepcopy(config['request']);request.pop('mutation_control',None)
    folder=OUT/'real_inputs'/f'{phase}_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'acceptance.json').exists():return json.loads((folder/'acceptance.json').read_text(encoding='utf-8'))
    original_open=Path.open;blocked=[]
    def guarded(path,*args,**kwargs):
        if path.name in {'test_labels.nc','validation_labels.nc','development_labels.nc'}:
            blocked.append(str(path));raise AssertionError('External truth must not be read by current-input decisions')
        return original_open(path,*args,**kwargs)
    Path.open=guarded
    try:
        result=api.process_current_input(inputs,request,folder/'platform',
            budget=dict(objective_limit=240,proposal_limit=50000,wall_seconds=240,probe_call_limit=60),seed=seed)
    finally:Path.open=original_open
    assert result['status']=='PASS' and result['optimizer']['backend']=='random' and result['evolution_used']
    assert not blocked and result['confirmation']['used_for_selection'] is False
    assert result['current_input_deployment']['status']=='PASS'
    with xr.open_dataset(result['product']) as opened:
        product=opened.load();assert np.isfinite(product.ndvi).all()
    with xr.open_dataset(config['inputs'][0]['path']) as opened:
        visible=opened.ndvi.load();known=np.isfinite(visible.values)
        np.testing.assert_array_equal(product.ndvi.sel(time=visible.time).values[known],visible.values[known])
    with xr.open_dataset(folder/'platform/self_validation_input.nc') as opened:masked=opened.ndvi.load()
    with xr.open_dataset(folder/'platform/self_validation_labels.nc') as opened:search=opened.ndvi.load()
    with xr.open_dataset(folder/'platform/self_confirmation_labels.nc') as opened:confirm=opened.ndvi.load()
    for ref in [search,confirm]:
        aligned=masked.sel(time=ref.time);mask=np.isfinite(ref.values);assert np.isnan(aligned.values[mask]).all()
    search_mask=np.isfinite(search.reindex(time=visible.time).values)
    confirm_mask=np.isfinite(confirm.reindex(time=visible.time).values)
    assert not (search_mask&confirm_mask).any()
    report=dict(status='PASS',phase=phase,seed=seed,input_dates=int(len(visible.time)),
        input_sha256=[hashlib.sha256(i.content).hexdigest() for i in inputs],product=result['product'],
        product_sha256=hashlib.sha256(Path(result['product']).read_bytes()).hexdigest(),
        shape=list(product.ndvi.shape),observations_preserved=True,all_output_finite=True,
        external_truth_reads=blocked,search_confirmation_disjoint=True,backend='random',
        selected_key=result['best']['key'],confirmation=result['confirmation'],
        deployment=result['current_input_deployment'],cost=result['total_cost'],
        scientific_advantage='NOT_EVALUATED',old_graph_reused=False)
    (folder/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return {k:report[k] for k in ['status','phase','seed','input_dates','shape','selected_key']}


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);jobs=[('validation',11),('test',11),('test',29)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_one,*job) for job in jobs]
        result=[]
        for future in as_completed(futures):
            row=future.result();result.append(row);print(json.dumps(row),flush=True)
    (OUT/'real_input_acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
