"""Independent selected-graph replay and masked numerical evidence checks."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import xarray as xr
from run_robit_era5_mutation import ROOT,OLD,OUT,load,freeze
from evaluate_robit_era5_fusion import bench,labels,protect,VAR
from ecodata.algorithms.numerical_search import save


def main():
    freeze()
    for path,digest in load(OUT/'input_hashes.json').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    replay=[];edits={}
    for folder in sorted((OUT/'runs').iterdir()):
        kind=folder.name.split('_')[0];context,engine,_=bench(kind)
        primary=context.sources[0].dataset[VAR]
        for phase in ['development','validation','test']:
            ref=labels(kind,phase);assert np.isnan(primary.sel(time=ref.time)).all()
        summary=load(folder/'summary.json');assert summary['test_used_for_selection'] is False
        if folder.name.endswith('_alns'):
            assert summary['optimizer']['backend']=='alns' and summary['evolution_used']
            assert summary['mutation_new_executions']>0
        records=load(folder/'graphs.json');best=summary['best'];output,_=engine.execute(records,best['key'])
        reference=labels(kind,'validation');pred=output[VAR].sel(time=reference.time).values.ravel()
        error=pred-reference.values.ravel();years=pd.DatetimeIndex(reference.time.values).year
        independent=float(np.mean([np.sqrt(np.mean(error[years==y]**2)) for y in np.unique(years)]))
        np.testing.assert_allclose(independent,best['metrics']['objective'],rtol=1e-10,atol=1e-10)
        assert independent<=summary['initial_best']['metrics']['objective']+1e-10
        with xr.open_dataset(summary['product']) as f:actual=f.load()
        np.testing.assert_allclose(protect(output,context)[VAR],actual[VAR],rtol=1e-10,atol=1e-10)
        visible=np.isfinite(primary.values);np.testing.assert_array_equal(actual[VAR].values[visible],primary.values[visible])
        trace=load(folder/'mutation_trace.json')
        assert len(trace)==summary['mutation_new_executions']
        assert len({t['child'] for t in trace})==len(trace)
        for item in trace:
            assert item['parent']!=item['child']
            assert item['actual_execution']
            assert item['parent'] in records and item['child'] in records
            assert item['accepted']==(item['after_objective'] is not None and item['after_objective']<=item['before_objective'])
            operator=item['edit']['operator'];edits[operator]=edits.get(operator,0)+1
        replay.append(dict(run=folder.name,independent_validation_RMSE=independent,mutations=len(trace)))
    manifest=load(OUT/'predictions_frozen.json')
    for row in manifest:
        path=Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with xr.open_dataset(path) as f:assert np.isfinite(f[VAR]).all() and (f[VAR]>=0).all()
    self_folder=OUT/'diagnostic_v6/short_11_alns_self'
    with xr.open_dataset(self_folder/'self_validation_input.nc') as f:masked=f.load()
    with xr.open_dataset(self_folder/'self_validation_labels.nc') as f:self_labels=f.load()
    assert np.isfinite(self_labels[VAR]).all()
    assert np.isnan(masked[VAR].sel(time=self_labels.time)).all()
    assert load(self_folder/'summary.json')['self_validation_from_visible']
    resolved=load(self_folder/'request.json')['request']
    assert resolved['policy_resolution'].startswith('Unique registered')
    assert resolved['portrait_policy']['version']=='robit-era5-multisource-swatplus-v1'
    assert resolved['scoring']['registry_version']=='robit-daily-state-v1'
    result=dict(status='PASS',replayed_runs=len(replay),frozen_predictions=len(manifest),input_hashes_unchanged=True,
          hidden_labels_absent_from_algorithm_inputs=True,parent_child_edits_checked=edits,
          best_never_worse_than_initial_validation=True,original_visible_values_preserved=True,rows=replay)
    result['automatic_visible_self_validation_checked']=True
    save(OUT/'independent_audit.json',result);print('AUDIT PASS',len(replay),len(manifest),flush=True)


if __name__=='__main__':main()
