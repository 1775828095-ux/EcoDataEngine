"""Independent saved-artifact replay and accounting audit; no parameter tuning."""
from pathlib import Path
import hashlib,json
from time import perf_counter
import numpy as np
import pandas as pd
import xarray as xr
from run_unseen_portrait_response import ROOT,OUT,VAR,cases,load,freeze
from evaluate_unseen_portrait_response import bench,labels
from ecodata.algorithms.numerical_search import save


def main():
    started=perf_counter();audit_calls=0
    freeze()
    for name,digest in load(OUT/'input_hashes.json').items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    metrics={(r['case'],r['method']):r for r in load(OUT/'test_metrics.json')}
    for case in cases():
        config=load(OUT/'cases'/case/'config.json')
        with xr.open_dataset(config['inputs'][0]['path']) as f:primary=f.load()
        for phase,lower,upper in [('development',2018,2019),('validation',2020,2021),('test',2022,2024)]:
            reference=labels(case,phase);years=pd.DatetimeIndex(reference.time.values).year
            assert years.min()>=lower and years.max()<=upper
            assert np.isnan(primary[VAR].sel(time=reference.time)).all()
        with xr.open_dataset(config['inputs'][2]['path']) as f:
            assert pd.Timestamp(f.attrs['training_end']).year==2017
    checks=[]
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        context,engine,config=bench(entry['case']);graph=load(path.with_suffix('.graph.json'))
        replay,_=engine.execute(graph['records'],graph['key'])
        audit_calls+=engine.calls
        with xr.open_dataset(path) as f:saved=f.load()
        with xr.open_dataset(config['inputs'][0]['path']) as f:primary=f.load()
        # The public entry explicitly protects retained station observations at deployment.
        replay[VAR]=xr.where(np.isfinite(primary[VAR]),primary[VAR],replay[VAR])
        np.testing.assert_allclose(replay[VAR],saved[VAR],equal_nan=True,rtol=1e-10,atol=1e-10)
        observed=np.isfinite(primary[VAR]);np.testing.assert_allclose(saved[VAR].values[observed],primary[VAR].values[observed],rtol=0,atol=1e-10)
        assert np.isfinite(saved[VAR]).all() and (saved[VAR]>=0).all()
        reference=labels(entry['case'],'test');pred=saved[VAR].sel(time=reference.time)
        error=(pred-reference).values.ravel();years=pd.DatetimeIndex(reference.time.values).year
        independent=float(np.mean([np.sqrt(np.mean(error[years==year]**2)) for year in np.unique(years)]))
        np.testing.assert_allclose(independent,metrics[(entry['case'],entry['method'])]['objective'],rtol=1e-10)
        bounds=primary.time_bounds.values
        assert ((bounds[:,1]-bounds[:,0])==np.timedelta64(1,'D')).all()
        np.testing.assert_array_equal(primary.time,bounds[:,1])
        checks.append(dict(case=entry['case'],method=entry['method'],replay='PASS',independent_test_RMSE=independent))
    runs=[]
    for folder in sorted((OUT/'runs/validation').iterdir()):
        summary=load(folder/'summary.json');cost=load(folder/'total_cost.json')
        assert cost['used']<=cost['limit']==180
        assert len(cost['events'])==cost['used']
        assert all(e['status']!='STARTED' for e in cost['events'])
        trace=load(folder/'mutation_trace.json')
        assert all(t['parent_is_best_at_proposal'] for t in trace)
        graphs=load(folder/'graphs.json')
        def count(key):
            seen=set()
            def visit(k):
                if k in seen:return
                seen.add(k)
                for p in graphs[k]['parents']:visit(p)
            visit(key)
            return sum(graphs[k]['operation']!='source' for k in seen)
        for t in trace:
            if t['accepted'] and t['before_objective']==t['after_objective']:
                assert count(t['child'])<=count(t['parent'])
            if t['family']=='joint':
                plan=t['edit']['destruction']
                # A legitimate raw-source incumbent has zero processing nodes.
                assert len(plan['removed_nodes'])==count(t['parent'])
                assert '__repair_hole__' in plan['partial_graph']
        response=load(folder/'response_scoring.json')
        if response['enabled']:
            protocol=response['protocol'];times=pd.DatetimeIndex(protocol['mask_times'])
            assert times.year.min()>=2018 and times.year.max()<=2019
            config=load(folder/'client_config.json')
            with xr.open_dataset(config['inputs'][0]['path']) as f:
                np.testing.assert_allclose(f[VAR].sel(time=times).values.ravel(),protocol['mask_values'])
            assert protocol['all_derived_branches_reexecuted_from_masked_roots']
            assert not protocol['final_test_labels_used']
        runs.append(dict(run=folder.name,calls=cost['used'],trace_entries=len(trace),status='PASS'))
    assert len(checks)==108 and len(runs)==72
    handoff=load(OUT/'precipitation_handoff/index.json');assert len(handoff)==30
    assert all(x['readback']=='PASS' and x['days']==1096 for x in handoff)
    save(OUT/'independent_audit.json',dict(status='PASS',predictions=checks,runs=runs,
         mechanism_gate=load(ROOT/'output/portrait_response_20261028/mechanism_acceptance.json'),
         source_and_inputs_unchanged=True,precipitation_handoffs=30,global_optimum_proven=False,native_SWAT_run=False))
    save(OUT/'audit_cost.json',dict(adapter_calls=audit_calls,seconds=perf_counter()-started,
         role='Post-freeze independent verification; does not affect selection'))
    print('INDEPENDENT AUDIT PASS',len(checks),len(runs),flush=True)


if __name__=='__main__':main()
