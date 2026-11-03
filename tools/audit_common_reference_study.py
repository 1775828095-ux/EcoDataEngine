"""Independent saved-file arithmetic for common baselines and mutation-only hits."""
from pathlib import Path
import hashlib,json

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.attribute_registry import digest

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/common_reference_20261026'
OLD=ROOT/'output/parallel_portrait_20261025'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def a(p):
    with h5py.File(p) as d:return d['air_temperature'][:]
def close(x,y):np.testing.assert_allclose(x,y,rtol=1e-9,atol=1e-9)


def main():
    frozen=load(OUT/'frozen_selections.json');assert not frozen['test_labels_read'] and not frozen['oracle_used_to_choose']
    protocol=load(OUT/'protocol.json');profile=load(OUT/'common_reference.json');results=load(OUT/'results.json')
    assert profile['training_partition']=='development' and not profile['test_used']
    assert sha(Path(profile['fixed_provenance']['path']))==profile['fixed_provenance']['sha256']
    for x in load(OUT/'source_manifest.json'):assert sha(Path(x['path']))==x['sha256']
    counts=dict(numeric_products=0,reference_probes=0,common_scores=0,test_products=0)
    checked_product_paths=set();oracle_receipts=[]
    for p in (OUT/'single_development').glob('*/*/result.json'):
        d=load(p);assert sha(p.parent/'product.nc')==d['sha256']
        truth=a(OLD/'cases'/p.parents[1].name/'labels.nc')
        close(np.sqrt(np.mean((a(p.parent/'product.nc')-truth)**2)),d['metrics']['RMSE'])
    means={name:np.mean(v) for name,v in profile['single_development_losses'].items()}
    assert profile['selected_single']==min(means,key=lambda k:(means[k],k))
    optimum={}
    for item in protocol['cases']:
        identifier=item['id'];truth=a(OUT/'cases'/identifier/'labels.nc')
        assert item['start']+32<item['test_start']
        d=load(OUT/'oracles'/identifier/'summary.json');assert len(d['rows'])==1296
        if 'reuse_receipt' in d:
            receipt=d['reuse_receipt'];assert receipt['reused_real_products']+receipt['new_oracle_only_executions']==1296
            assert not receipt['mutation_trajectories_extended'];oracle_receipts.append(receipt)
        errors=[]
        for row in d['rows']:
            assert row['status']=='PASS';path=Path(row['product']);assert sha(path)==row['product_sha256']
            error=float(np.sqrt(np.mean((a(path)-truth)**2)));close(error,row['search_validation']['RMSE']);errors.append(error);counts['numeric_products']+=1
            checked_product_paths.add(str(path))
        optimum[identifier]=min(errors)
        for method in protocol['methods']:
            for seed in protocol['seeds']:
                folder=OUT/'search'/identifier/method/str(seed);s=load(folder/'summary.json')
                assert not s['exact_audit_additional_executions'] and s['certification']=='NOT_CERTIFIED'
                assert all(r['origin']!='exact_audit' for r in s['rows'])
                assert not s['independent_test_used_for_selection']
                for row in s['rows']:
                    assert row['status']=='PASS';path=Path(row['product']);assert sha(path)==row['product_sha256']
                    close(np.sqrt(np.mean((a(path)-truth)**2)),row['search_validation']['RMSE']);counts['numeric_products']+=1
                    checked_product_paths.add(str(path))
                if method=='common':
                    receipt=load(folder/'parallel_scoring.json')
                    assert receipt['probe_graph_executions']==76
                    targets={p['window']:np.asarray(p['truth']) for p in receipt['probes']}
                    config=load(OUT/'cases'/identifier/'config.json')
                    fine_path=next(Path(x['path']) for x in config['inputs'] if x['role']=='primary')
                    with xr.open_dataset(fine_path,engine='h5netcdf') as raw:fine=raw.load()
                    for p in receipt['probes']:
                        held=pd.DatetimeIndex(p['withheld_times']);retained=pd.DatetimeIndex(p['retained_times'])
                        assert not held.isin(retained).any()
                        close(fine.air_temperature.sel(time=held).values,p['truth'])
                    local_errors={};local_counts={}
                    for response in receipt['responses']:
                        assert response['status']=='MEASURED_VISIBLE_MASK'
                        error=float(np.sqrt(np.mean((np.asarray(response['product']['values'])-targets[response['window']])**2)))
                        close(error,response['anchor_rmse'])
                        key=(response['window'],response['key']);local_errors[key]=error
                        local_counts[key]=np.asarray(response['product']['values']).size
                    reference={}
                    for r in receipt['reference_probes']:
                        assert r['status']=='PASS';error=float(np.sqrt(np.mean((np.asarray(r['values'])-targets[r['window']])**2)))
                        close(error,r['rmse']);reference.setdefault(r['name'],[]).append((error,r['samples']));counts['reference_probes']+=1
                    referr={k:float(np.sqrt(np.average([x[0]**2 for x in v],weights=[x[1] for x in v]))) for k,v in reference.items()}
                    chosen=min(referr,key=lambda k:(referr[k],k));ref=referr[chosen]
                    screen=load(folder/'initial_screening.json');shared=set()
                    axes=load(folder/'request.json')['axes']
                    for row in screen['rows']:
                        g=row['common_gain'];assert g['reference']==chosen
                        value=row['index'];digits=[0]*len(axes)
                        for j in range(len(axes)-1,-1,-1):digits[j]=value%len(axes[j]['choices']);value//=len(axes[j]['choices'])
                        plan=[dict(field=x['field'],window=x['window'],branch=x['branch'],choice=x['choices'][v]) for x,v in zip(axes,digits)]
                        keys=[(w,digest([dict(field=p['field'],branch=p['branch'],choice=p['choice']) for p in plan if p['window']==w])) for w in sorted(targets)]
                        actual_error=float(np.sqrt(np.average([local_errors[k]**2 for k in keys],weights=[local_counts[k] for k in keys])))
                        close(actual_error,g['candidate_rmse'])
                        close(g['reference_rmse'],ref);close(g['delta_rmse_K'],ref-g['candidate_rmse'])
                        gain=0 if ref+g['candidate_rmse']==0 else (ref-g['candidate_rmse'])/(ref+g['candidate_rmse'])
                        close(row['routes']['collaboration'],[(1+gain)/2]*2)
                        close(row['score_interval'],[.5*row['routes']['data'][i]+.5*row['routes']['model'][i] for i in (0,1)])
                        assert g['stable_gain']==all(v>0 for v in g['paired_window_delta']);shared.add(g['reference_rmse']);counts['common_scores']+=1
                    assert len(shared)==1
                for budget in protocol['budget_prefixes']:
                    row=next(x for x in results['rows'] if (x['case'],x['method'],x['seed'],x['budget'])==(identifier,method,seed,budget))
                    choice=next(x for x in frozen['rows'] if (x['case'],x['method'],x['seed'],x['budget'])==(identifier,method,seed,budget))
                    assert choice['key']==row['key']
                    evaluated=s['rows'][:budget]
                    close(row['validation_rmse'],min(r['search_validation']['RMSE'] for r in evaluated if r['status']=='PASS'))
                    hits=[i+1 for i,r in enumerate(evaluated) if abs(r['search_validation']['RMSE']-optimum[identifier])<=1e-9]
                    assert row['hit']==bool(hits) and row['first_hit']==(hits[0] if hits else None)
                    assert row['initial_hit']==bool(hits and hits[0]<=2) and row['mutation_hit']==bool(hits and hits[0]>2)
    for p in (OUT/'test').glob('*/*/result.json'):
        d=load(p);assert sha(p.parent/'product.nc')==d['sha256']
        truth=a(OUT/'cases'/p.parents[1].name/'test_labels.nc');close(np.sqrt(np.mean((a(p.parent/'product.nc')-truth)**2)),d['metrics']['RMSE']);counts['test_products']+=1
    for row in results['rows']:
        saved=load(OUT/'test'/row['case']/row['key']/'result.json')
        assert saved['plan']==row['plan'];close(saved['metrics']['RMSE'],row['test_rmse'])
    for method,table in results['summary'].items():
        for budget,s in table.items():
            group=[r for r in results['rows'] if r['method']==method and r['budget']==int(budget)]
            assert s['hits']==sum(r['hit'] for r in group)
            assert s['initial_hits']+s['mutation_hits']==s['hits']
            close(s['mean_test_rmse'],np.mean([r['test_rmse'] for r in group]))
    counts['numeric_products']=len(checked_product_paths)
    report=dict(status='PASS',**counts,oracle_receipts=oracle_receipts,all_search_runs_excluded_exact_completion=True,
        common_baseline_shared_and_development_only=True,initial_and_mutation_hits_separated=True,
        scope='Independent product, score and prefix-hit arithmetic; four controlled cases, no unbounded guarantee')
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
