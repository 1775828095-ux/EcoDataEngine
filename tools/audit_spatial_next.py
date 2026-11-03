"""Independent numerical/product readback for the expanded experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/spatial_next_20261023'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bounds(v):
    v=np.asarray(v,dtype=float)
    return np.r_[v[0]-(v[1]-v[0])/2,(v[1:]+v[:-1])/2,v[-1]+(v[-1]-v[-2])/2]


def run():
    protocol=read(OUT/'protocol.json');assert sha(protocol['source'])==protocol['data_sha256']
    selected=read(OUT/'final_selection.json')
    assert all(sha(ROOT/name)==digest for name,digest in selected['code_sha256'].items())
    assert sha(OUT/selected['manifest'])==selected['manifest_sha256']
    tasks=read(OUT/selected['manifest']);events=[json.loads(l) for l in (OUT/'events.jsonl').read_text().splitlines()]
    cut=next(i for i,e in enumerate(events) if e['kind']=='ALL_AUDIT_CHOICES_FROZEN')
    assert events[cut]['sha256']==sha(OUT/'audit_choices_frozen.json')
    assert not any(e['kind']=='REFERENCE_OPEN' and e['stage']=='audit' for e in events[:cut])
    assert sum(e['kind']=='REFERENCE_OPEN' and e['stage']=='audit' for e in events[cut+1:])==24
    checked=0;recomputed=0;max_delta=0.;failures=[];graphs=[];pairs=[]
    frozen=read(OUT/'audit_choices_frozen.json')
    for t in tasks:
        assert sha(t['coarse'])==t['coarse_sha256'] and sha(t['reference'])==t['reference_sha256']
        summary=read(OUT/'runs'/t['id']/'summary.json')
        assert summary['counts']['real_executions']==14 and len(summary['rows'])==14
        assert len({tuple(r['steps']) for r in summary['rows']})==14
        if t['stage']=='audit':
            with xr.open_dataset(t['reference']) as ds:y=ds.temperature.load()
            with xr.open_dataset(t['coarse']) as ds:c=ds.load()
            w=np.outer(np.diff(np.sin(np.deg2rad(bounds(y.lat)))),np.diff(np.deg2rad(bounds(y.lon))))
            mask=np.outer((y.lat>=c.lat.min())&(y.lat<=c.lat.max()),(y.lon>=c.lon.min())&(y.lon<=c.lon.max()))
        mapping={r['key']:r for r in summary['rows']}
        for row in summary['rows']:
            if row['status']!='PASS':failures.append(dict(task=t['id'],error=row.get('error')));continue
            assert row['hidden_truth_passed_to_nodes'] is False and row['numerical_execution_attempted'] and row['file_readback']=='PASS'
            assert sha(row['product'])==row['product_sha256'];checked+=1
            graphs.append(dict(task=t['id'],stage=t['stage'],date=t['date'],region=t['region'],scale=t['scale'],
                method=' + '.join(row['steps']),RMSE=row['RMSE'],MAE=row['MAE'],Bias=row['Bias'],full_RMSE=row['full_RMSE'],
                coarse_RMSE=row['coarse_RMSE'],seconds=row['seconds']))
            if t['stage']=='audit':
                with xr.open_dataset(row['product']) as ds:
                    assert np.array_equal(ds.lat,y.lat) and np.array_equal(ds.lon,y.lon)
                    assert ds.temperature.attrs['units']=='K' and np.isfinite(ds.temperature).all()
                    d=ds.temperature.values-y.values
                error=float(np.sqrt((d*d*w)[mask].sum()/w[mask].sum()))
                max_delta=max(max_delta,abs(error-row['RMSE']));assert abs(error-row['RMSE'])<1e-10;recomputed+=1
        for choice in [c for c in frozen if c['task']==t['id']]:
            assert mapping[choice['key']]['steps']==choice['steps']
            if choice['scenario']=='M2':assert choice['steps'][-1]=='raster_task.coarse_consistency'
        if t['stage']=='audit':
            for single in [r for r in summary['rows'] if len(r['steps'])==1]:
                combination=next(r for r in summary['rows'] if r['steps']==single['steps']+['raster_task.coarse_consistency'])
                pairs.append(dict(task=t['id'],date=t['date'],base=single['steps'][0],single_RMSE=single['RMSE'],
                    combination_RMSE=combination['RMSE'],gain=single['RMSE']-combination['RMSE']))
    assert checked+len(failures)==14*len(tasks)
    model=ROOT/'output/plan_ranking_models'/selected['model'];report=read(model/'report.json');evidence=read(model/'portrait_evidence.json')
    assert report['training_rows']==14*sum(t['stage']=='fit' for t in tasks) and report['holdout_rows']==112
    assert all(g.startswith('2013-') for g in report['training_groups'])
    assert report['holdout_groups']==['2014-01-01','2014-02-01']
    assert {c['source']['group'] for c in evidence['data_cases']}==set(report['training_groups'])
    assert sha(model/'model.joblib')==report['model_sha256'] and sha(model/'portrait_evidence.json')==report['portrait_evidence_sha256']
    assert sha(report['policy_validation']['path'])==report['policy_validation']['sha256']
    assert report['default_policy']==dict(learning_order=selected['learning_order'],fallback_policy=selected['fallback_policy'])
    for tag in ['final_audit','final_metric_ablation','final_rank_ablation']:
        for path in (OUT/'recommendations'/tag).glob('*/*/*/summary.json'):
            result=read(path);assert result['best'] is None and result['counts']['real_executions']==0
            if tag=='final_audit' and path.parent.name=='learning':
                options=read(path.parent/'request.json')['request']['profile_options']
                assert 'learning_order' not in options and 'fallback_policy' not in options
                assert result['learning_status']['learning_order']==report['default_policy']['learning_order']
                assert result['learning_status']['fallback_policy']==report['default_policy']['fallback_policy']
    df=pd.DataFrame(graphs);df.to_csv(OUT/'all_graph_results.csv',index=False)
    combo=pd.DataFrame(pairs)
    combo.to_csv(OUT/'audit_combination_pairs.csv',index=False)
    combo.groupby('base').agg(single=('single_RMSE','mean'),combination=('combination_RMSE','mean'),gain=('gain','mean'),
        wins=('gain',lambda x:int((x>1e-10).sum())),losses=('gain',lambda x:int((x< -1e-10).sum()))).to_csv(OUT/'audit_combinations.csv')
    results=pd.read_csv(OUT/'final_audit_results.csv')
    daily=results.groupby(['date','scenario','strategy']).RMSE.mean().unstack('strategy');daily.to_csv(OUT/'audit_daily.csv')
    comparisons=[]
    for scenario in ['M1','M2']:
        table=daily.xs(scenario,level='scenario')
        for baseline in ['rules','random','development_fixed']:
            gains=(table[baseline]-table.learning).to_numpy();rng=np.random.default_rng(17)
            bootstrap=rng.choice(gains,(2000,len(gains)),replace=True).mean(axis=1)
            comparisons.append(dict(scenario=scenario,baseline=baseline,date_groups=len(gains),mean_gain_K=float(gains.mean()),
                descriptive_bootstrap_95=[float(x) for x in np.quantile(bootstrap,[.025,.975])],
                winning_dates=int((gains>1e-10).sum()),losing_dates=int((gains< -1e-10).sum())))
    result=dict(status='PASS',products_verified=checked,failures=failures,audit_metrics_recomputed=recomputed,max_RMSE_delta=max_delta,
        audit_sealed_before_reference=True,development_and_audit_disjoint=True,source_and_final_code_hashes_unchanged=True,
        comparisons=comparisons,uncertainty_scope='six monthly audit date groups; descriptive bootstrap, not broad significance proof')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
