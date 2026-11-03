"""Independent saved-artifact arithmetic; never calls screening or optimisation."""
from pathlib import Path
import hashlib,json
import sys

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/parallel_portrait_20261025'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-9)
def values(path):
    with h5py.File(path,'r') as raw:return raw['air_temperature'][:]


def main():
    development_only='--development-only' in sys.argv
    if not development_only:assert (OUT/'frozen_selections.json').exists(),'Never audit test labels before choice freeze'
    protocol=read(OUT/'protocol.json');cal=read(OUT/'calibration.json')
    result=dict(runs=[],summary={}) if development_only else read(OUT/'results.json')
    assert cal['training_partition']=='development' and not cal['test_used']
    assert set(cal['training_groups'])=={x['id'] for x in protocol['development']}
    for item in read(OUT/'source_manifest.json'):assert sha(Path(item['path']))==item['sha256']
    # Same-region raw time intervals are disjoint across all partitions.
    all_cases=protocol['development']+protocol['validation']+protocol['test']
    for i,a in enumerate(all_cases):
        for b in all_cases[i+1:]:
            if a['region']['id']==b['region']['id']:assert a['start']+32<b['start'] or b['start']+32<a['start']
    assert not {x['region']['id'] for x in protocol['development']} & {'south_east','north_east'}
    products=0;probes=0;repeated_probes=0;discrimination=[];correlations=[];local_order_witness=False
    probe_reference={}
    for folder in sorted((OUT/'oracles').iterdir()):
        if development_only and not folder.name.startswith('dev_'):continue
        report=read(folder/'summary.json');screen=read(folder/'initial_screening.json')
        assert len(report['rows'])==1296 and report['counts']['evaluated']==1296
        assert screen['scored_count']==1296 and screen['complete'] and not screen['sampling']
        config=read(OUT/'cases'/folder.name/'config.json')
        truth=values(OUT/'cases'/folder.name/'labels.nc')
        errors={}
        for row in report['rows']:
            assert row['status']=='PASS',row.get('error')
            path=Path(row['product']);assert sha(path)==row['product_sha256']
            a=values(path);assert a.shape==truth.shape and np.isfinite(a).all()
            rmse=float(np.sqrt(np.mean((a-truth)**2)));close(rmse,row['search_validation']['RMSE'])
            errors[row['index']]=rmse;products+=1
        close(min(errors.values()),report['final_validation_rmse'])
        receipt=read(folder/'parallel_scoring.json');assert receipt['probe_graph_executions']==72
        probe_reference[folder.name]={(r['window'],r['key']):r for r in receipt['responses']}
        assert receipt['unsupported'] is None and len(receipt['responses'])==72
        model=read(folder/'request.json')['portraits'][0]['model']
        issues={i['attribute']:i for i in model['state_policy']['scenario']['issues']}
        probe_truth={p['window']:np.array(p['truth']) for p in receipt['probes']}
        fine_path=next(Path(x['path']) for x in config['inputs'] if x['role']=='primary')
        coarse_path=next(Path(x['path']) for x in config['inputs'] if x['role']=='background')
        with xr.open_dataset(fine_path,engine='h5netcdf') as raw:fine=raw.load()
        with xr.open_dataset(coarse_path,engine='h5netcdf') as raw:coarse=raw.load()
        # Independent nested-cell spherical area matrix, not the writer's operator.
        lat=fine.lat.values.astype(float);lat_edges=np.r_[lat[0]-(lat[1]-lat[0])/2,(lat[:-1]+lat[1:])/2,lat[-1]+(lat[-1]-lat[-2])/2]
        area=np.diff(np.sin(np.deg2rad(lat_edges)));ay=np.zeros((4,8));ax=np.zeros((4,8))
        for i in range(4):ay[i,2*i:2*i+2]=area[2*i:2*i+2]/area[2*i:2*i+2].sum();ax[i,2*i:2*i+2]=.5
        operator=np.kron(ay,ax);inverse=operator.T@np.linalg.inv(operator@operator.T)
        metrics={}
        for p in receipt['probes']:
            held={pd.Timestamp(t) for t in p['withheld_times']};assert not held & {pd.Timestamp(t) for t in p['retained_times']}
            assert len(held)==1
            held_times=pd.DatetimeIndex(p['withheld_times']);retained=pd.DatetimeIndex(p['retained_times'])
            close(fine.air_temperature.sel(time=held_times).values,p['truth'])
            proxy=fine.air_temperature.sel(time=retained).interp(time=held_times).values.reshape(-1,64)
            observed=coarse.air_temperature.sel(time=held_times).values.reshape(-1,16)
            detail=proxy-(proxy@operator.T)@inverse.T
            metrics[p['window']]=dict(proxy=proxy,coarse=observed,detail=detail,
                coarse_error=float(np.sqrt(np.mean((proxy@operator.T-observed)**2))),detail_error=float(np.sqrt(np.mean(detail**2))))
        for response in receipt['responses']:
            assert response['status']=='MEASURED_VISIBLE_MASK'
            target=probe_truth[response['window']];a=np.array(response['product']['values'])
            e=float(np.sqrt(np.mean((a-target)**2)));scale=float(target.std())
            close(e,response['anchor_rmse']);close(scale,response['normalization_scale'])
            close([1/(1+e/scale)]*2,response['routes']['data'])
            baseline=min(response['constituent_rmse'].values());gain=0 if baseline+e==0 else (baseline-e)/(baseline+e)
            close([(1+gain)/2]*2,response['routes']['collaboration'])
            utility=0.
            m=metrics[response['window']];flat=a.reshape(-1,64)
            after_metrics=dict(coarse_error=float(np.sqrt(np.mean((flat@operator.T-m['coarse'])**2))),
                detail_error=float(np.sqrt(np.mean((flat-(flat@operator.T)@inverse.T-m['detail'])**2))))
            for term in response['model_terms']:
                issue=issues[term['dimension']]
                before=term['data_evidence']['before'];after=term['algorithm_evidence']['after']
                close(before,m[issue['metric']]);close(after,after_metrics[issue['metric']])
                risk=lambda v:float(np.clip((v-issue['acceptable'])/(issue['bad']-issue['acceptable']),0,1))
                u=risk(before)-risk(after);close(u,term['utility'])
                utility+=term['weight']*u
            close([(1+utility)/2]*2,response['routes']['model']);probes+=1
        weights=receipt['weights'];joint=[];actual=[]
        for row in screen['rows']:
            computed=[sum(weights[k]*row['routes'][k][j] for k in weights) for j in (0,1)]
            close(computed,row['score_interval']);joint.append(computed[0]);actual.append(errors[row['index']])
        unique=len(set(joint));assert unique>1
        discrimination.append(dict(case=folder.name,unique_scores=unique,total=1296))
        calibrated=[sum(cal['weights']['full'][k]*r['routes'][k][0] for k in weights) for r in screen['rows']]
        route_correlations={k:float(spearmanr([r['routes'][k][0] for r in screen['rows']],-np.array(actual)).statistic) for k in weights}
        correlations.append(dict(case=folder.name,equal_weight_score_vs_negative_error_spearman=float(spearmanr(joint,-np.array(actual)).statistic),
            calibrated_joint_vs_negative_error_spearman=float(spearmanr(calibrated,-np.array(actual)).statistic),routes=route_correlations))
        # Witness that changing order is measured, rather than a static tag sum.
        route_values=[r['routes']['data'][0] for r in receipt['responses']]
        local_order_witness|=len(set(route_values))>1
    assert local_order_witness
    fixed=cal['fixed'];losses={}
    for item in protocol['development']:
        d=read(OUT/'oracles'/item['id']/'summary.json')
        for row in d['rows']:losses.setdefault(row['key'],[]).append(row['search_validation']['RMSE'])
    assert fixed==min(losses,key=lambda k:(np.mean(losses[k]),k))
    frozen=dict(test_labels_read=False) if development_only else read(OUT/'frozen_selections.json')
    assert not frozen['test_labels_read']
    for item in result['runs']:
        original=next(x for x in frozen['rows'] if (x['case'],x['method'],x['seed'])==(item['case'],item['method'],item['seed']))
        assert item['key']==original['key'] and item['plan']==original['plan']
        folder=OUT/'search'/item['case']/item['method']/str(item['seed']);d=read(folder/'summary.json')
        assert d['recommended_key']==item['key'] and not d['independent_test_used_for_selection']
        close(item['validation_rmse'],min(r['search_validation']['RMSE'] for r in d['rows'] if r['status']=='PASS'))
        if d.get('parallel_scoring'):
            receipt=read(folder/'parallel_scoring.json')
            expected=cal['weights']['full' if item['method']=='score_only' else item['method']]
            assert receipt['weights']==expected
            for response in receipt['responses']:
                original=probe_reference[item['case']][response['window'],response['key']]
                close(response['product']['values'],original['product']['values'])
                for route in expected:close(response['routes'][route],original['routes'][route])
                repeated_probes+=1
            screening=read(folder/'initial_screening.json')
            for row in screening['rows']:
                close(row['score_interval'],[sum(expected[k]*row['routes'][k][j] for k in expected) for j in (0,1)])
        # Check every search product, not just its winner.
        truth=values(OUT/'cases'/item['case']/'labels.nc')
        for row in d['rows']:
            assert row['status']=='PASS',row.get('error')
            path=Path(row['product']);assert sha(path)==row['product_sha256']
            close(np.sqrt(np.mean((values(path)-truth)**2)),row['search_validation']['RMSE']);products+=1
    test_products=0
    for path in ([] if development_only else (OUT/'test').glob('*/*/result.json')):
        d=read(path);product=Path(d['product']);assert sha(product)==d['sha256']
        truth=values(OUT/'cases'/path.parents[1].name/'labels.nc')
        close(np.sqrt(np.mean((values(product)-truth)**2)),d['metrics']['RMSE']);test_products+=1
    for run in result['runs']:
        saved=read(OUT/'test'/run['test_id']/run['key']/'result.json')
        close(run['test_rmse'],saved['metrics']['RMSE'])
    # Recreate calibration winners independently from frozen development regret tables.
    for method,options in cal['candidate_results'].items():
        for option in options:
            for identifier,recorded in option['relative_regret_by_group'].items():
                d=read(OUT/'oracles'/identifier/'summary.json');by_index={r['index']:r for r in d['rows']}
                candidates=read(OUT/'oracles'/identifier/'initial_screening.json')['rows'];weights=option['weights']
                ranked=sorted(candidates,key=lambda r:(-sum(weights[k]*r['routes'][k][0] for k in weights),r['scientific_steps'],r['index']))
                chosen=min(by_index[r['index']]['search_validation']['RMSE'] for r in ranked[:2])
                close(chosen/min(r['search_validation']['RMSE'] for r in d['rows'])-1,recorded)
        winner=min(range(len(options)),key=lambda i:(np.mean(list(options[i]['relative_regret_by_group'].values())),i))
        assert cal['weights'][method]==options[winner]['weights']
    for method,summary in result['summary'].items():
        selected=[r for r in result['runs'] if r['method']==method]
        cases={r['case'] for r in selected}
        mean=np.mean([np.mean([r['test_rmse'] for r in selected if r['case']==c]) for c in cases])
        close(mean,summary['mean_test_rmse']);assert sum(r['hit'] for r in selected)==summary['validation_hits']
    report=dict(status='PASS',real_numeric_products_checked=products,test_products_checked=test_products,
        real_probe_responses_recomputed=probes,discrimination=discrimination,score_error_correlations=correlations,
        repeated_probe_responses_consistency_checked=repeated_probes,
        calibrated_weights=cal['weights'],label_isolation='Choice freeze, phase separation and disjoint time blocks checked; source-level masking covered by directed tests',
        scope='Independent saved-file arithmetic, not rerunning writer/scorer/search; descriptive finite controlled data')
    (OUT/('development_audit.json' if development_only else 'independent_audit.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
