"""Independent replay, split, raw-unit and decision-trajectory checks."""
from pathlib import Path
from time import perf_counter
import hashlib,json
import numpy as np
import pandas as pd
import xarray as xr
from run_basin_ecology import ROOT,OUT,DATA,SEEDS,load,freeze
from evaluate_basin_ecology import bench,labels,cases
from ecodata.algorithms.numerical_search import save


def main():
    started=perf_counter();freeze()
    for name,digest in load(OUT/'input_hashes.json').items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    for name,digest in load(DATA/'source_manifest.json').items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    with xr.open_dataset(DATA/'standardized.nc') as f:standard=f.load()
    raw=ROOT/'data/basin_ecology_2013_2017/raw_subsets/climate'
    for prefix,variable in [('pre','precipitation'),('tmp','air_temperature')]:
        for start,end in [(2012,2014),(2015,2017)]:
            with xr.open_dataset(raw/f'{prefix}_{start}_{end}_basin.nc') as f:
                field=f[prefix].sortby('lat').sortby('lon')
                for year in range(max(2013,start),end+1):
                    array=field.isel(time=np.arange((year-start)*12+4,(year-start)*12+10)).values.astype(float)/10
                    expected=array.sum(0) if prefix=='pre' else np.average(array,axis=0,weights=[31,30,31,31,30,31])
                    np.testing.assert_allclose(standard[variable].sel(time=f'{year}-07-01').values,expected,rtol=2e-6,atol=1e-5)
    fractions=sum(standard['land_fraction_'+str(c)] for c in range(1,10))
    np.testing.assert_allclose(fractions.values[np.isfinite(fractions)],1.,atol=2e-6)
    for case in cases():
        config=load(OUT/'cases'/case/'config.json')
        assert 'composition' not in config['request'] and 'fixed_graph' not in config['request']
        with xr.open_dataset(config['inputs'][0]['path']) as f:primary=f.load()
        with xr.open_dataset(config['inputs'][2]['path']) as f:aux=f.load()
        assert 'fvc' not in aux
        assert not np.isfinite(aux.training_target.sel(time=aux.time.dt.year>2014)).any()
        assert aux.training_target.attrs['split']=='train'
        for phase,year in [('development',2015),('validation',2016),('test',2017)]:
            reference=labels(case,phase);assert pd.DatetimeIndex(reference.time.values).year.tolist()==[year]
            assert np.isnan(primary.fvc.sel(time=reference.time,lat=reference.lat,lon=reference.lon)).all()
    metrics={(r['case'],r['method']):r for r in load(OUT/'test_metrics.json')};replay_calls=0;checks=[]
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        graph=load(path.with_suffix('.graph.json'));context,executor,_=bench(entry['case'])
        actual,_=executor.execute(graph['records'],graph['key']);replay_calls+=executor.calls
        source=context.sources[0].dataset;actual.fvc.values[:]=np.where(np.isfinite(source.fvc),source.fvc,actual.fvc)
        with xr.open_dataset(path) as f:saved=f.load()
        np.testing.assert_allclose(actual.fvc,saved.fvc,rtol=1e-10,atol=1e-10)
        known=np.isfinite(source.fvc);np.testing.assert_array_equal(saved.fvc.values[known],source.fvc.values[known])
        assert np.isfinite(saved.fvc).all() and bool(((saved.fvc>=0)&(saved.fvc<=1)).all())
        np.testing.assert_array_equal(saved.lat,source.lat);np.testing.assert_array_equal(saved.lon,source.lon);np.testing.assert_array_equal(saved.time,source.time)
        reference=labels(entry['case'],'test');values=saved.fvc.sel(time=reference.time,lat=reference.lat,lon=reference.lon).values
        independent=float(np.sqrt(np.mean((values-reference.values)**2)))
        np.testing.assert_allclose(independent,metrics[(entry['case'],entry['method'])]['objective'],rtol=1e-10,atol=1e-10)
        checks.append(dict(case=entry['case'],method=entry['method'],independent_RMSE=independent,status='PASS'))
    trajectories=[]
    for case in cases():
        for seed in SEEDS:
            pair={control:load(OUT/'runs/validation'/(case+'_'+control+'_'+str(seed))/'summary.json') for control in ['random','none']}
            np.testing.assert_allclose(pair['random']['initial_best']['metrics']['objective'],pair['none']['initial_best']['metrics']['objective'],rtol=1e-10)
    for folder in (OUT/'runs/validation').iterdir():
        s=load(folder/'summary.json');cost=load(folder/'total_cost.json');trace=load(folder/'mutation_trace.json');records=load(folder/'graphs.json')
        assert cost['used']<=180 and len(cost['events'])==cost['used']
        assert all(e['status']!='STARTED' for e in cost['events'])
        assert cost['training_capable_calls']==sum(e['requires_training'] for e in cost['events'])
        assert all(t['parent_is_best_at_proposal'] for t in trace)
        def count(key):
            seen=set()
            def visit(k):
                if k in seen:return
                seen.add(k)
                for p in records[k]['parents']:visit(p)
            visit(key);return sum(records[k]['operation']!='source' for k in seen)
        for t in trace:
            if t['accepted'] and t['before_objective']==t['after_objective']:assert count(t['child'])<=count(t['parent'])
            if t['family']=='joint':assert len(t['edit']['destruction']['removed_nodes'])==count(t['parent'])
        response=load(folder/'response_scoring.json');assert response['unsupported'] and not response['responses']
        meta=load(folder/'metadata.json');assert bool(trace)==(meta['control']=='random')
        trajectories.append(dict(run=folder.name,calls=cost['used'],trace_entries=len(trace),training_calls=cost['training_capable_calls'],status='PASS'))
    assert len(checks)==33 and len(trajectories)==18
    save(OUT/'independent_audit.json',dict(status='PASS',predictions=checks,trajectories=trajectories,
        raw_climate_units_and_months='PASS',categorical_fraction_sum='PASS',prior_target_isolation='PASS',
        same_initial_objectives='PASS',no_handwritten_pool_in_search=True,source_and_input_hashes_unchanged=True,
        independent_replay_adapter_calls=replay_calls,audit_seconds=perf_counter()-started,
        ground_true_vegetation_accuracy_claim=False,global_optimum_claim=False))
    print('INDEPENDENT AUDIT PASS',len(checks),len(trajectories),replay_calls,flush=True)


if __name__=='__main__':main()
