"""Independent timestamps, leakage, numerical replay and product checks."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
from evaluate_robit_era5_fusion import OUT,PROTOCOL,VAR,load,bench,labels,recipe
from ecodata.algorithms.numerical_search import save


def main():
    assert (OUT/'predictions_frozen.json').exists()
    for name,digest in load(OUT/'source_manifest_v6.json').items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            assert Path(name).as_posix()=='tools/evaluate_robit_era5_fusion.py',name
            frozen=(OUT/'source_frozen'/name).read_text(encoding='utf-8')
            assert frozen.replace('sel(time=base.time)','sel(time=pd.DatetimeIndex(base.time))',1)==Path(name).read_text(encoding='utf-8')
    checks=[];contexts={kind:bench(kind) for kind in PROTOCOL['masks']}
    for kind,(context,engine,request) in contexts.items():
        primary=context.sources[0].dataset[VAR]
        train=context.parameters['training_reference'];hist=context.parameters['historical_simulation']
        assert pd.Timestamp(train.time.max().values).year==1995
        assert len(np.intersect1d(train.time.values,primary.time.values))==0
        np.testing.assert_array_equal(train.time,hist.time)
        state=context.parameters['fitted_state'];assert state.attrs['training_end']=='1995-12-31'
        validation_dates=labels(kind,'validation').time.values
        test_dates=labels(kind,'test').time.values
        assert len(np.intersect1d(validation_dates,test_dates))==0
        for phase in ['development','validation','test']:
            reference=labels(kind,phase)
            assert np.isnan(primary.sel(time=reference.time).values).all()
            assert np.isfinite(context.sources[1].dataset[VAR].sel(time=reference.time)).all()
        for stage in ['development','validation']:
            for folder in (OUT/'runs_v6'/stage).glob(kind+'_*'):
                summary=load(folder/'summary.json');item=recipe(folder,summary['best']['key'])
                output,_=engine.execute(item['records'],item['key']);reference=labels(kind,stage)
                prediction=output[VAR].sel(time=reference.time).values.ravel();truth=reference.values.ravel()
                years=pd.DatetimeIndex(reference.time.values).year
                yearly={str(y):float(np.sqrt(np.mean((prediction[years==y]-truth[years==y])**2))) for y in np.unique(years)}
                actual=float(np.mean(list(yearly.values())))
                np.testing.assert_allclose(actual,summary['best']['metrics']['objective'],rtol=1e-10,atol=1e-10)
                records=[json.loads(x) for x in (folder/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(records)==summary['attempts']
                assert sum('key'in r for r in records)==summary['executed_candidates']
                checks.append(dict(case=kind,stage=stage,run=folder.name,objective=actual,metric_replay='PASS'))
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        primary=contexts[entry['case']][0].sources[0].dataset[VAR]
        with xr.open_dataset(path) as f:output=f.load()
        assert np.isfinite(output[VAR]).all() and (output[VAR]>=0).all()
        visible=np.isfinite(primary.values)
        np.testing.assert_array_equal(output[VAR].values[visible],primary.values[visible])
        np.testing.assert_array_equal(output.time,primary.time)
        np.testing.assert_array_equal(output.time_bounds.values[:,0],output.time.values)
        assert np.all(np.diff(output.time_bounds.values,axis=1)==np.timedelta64(1,'D'))
    result=dict(status='PASS',replayed_selected_runs=len(checks),prediction_files_checked=len(load(OUT/'predictions_frozen.json')),
        hidden_station_values_absent_from_every_algorithm_input=True,fit_application_times_disjoint=True,
        validation_test_times_disjoint=True,visible_station_values_preserved=True,
        interval_alignment_verified=True,source_hashes_verified=True,rows=checks)
    save(OUT/'independent_audit.json',result);print('INDEPENDENT AUDIT PASS',len(checks),flush=True)


if __name__=='__main__':main()
