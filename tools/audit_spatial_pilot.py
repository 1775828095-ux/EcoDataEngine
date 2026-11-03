"""Read back pilot products and frozen choices independently of the runner."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/spatial_pilot_20261023_v2'
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def boundaries(values):
    values=np.asarray(values)
    return np.r_[values[0]-(values[1]-values[0])/2,(values[:-1]+values[1:])/2,values[-1]+(values[-1]-values[-2])/2]


def run():
    protocol=read(OUT/'protocol.json');tasks=read(OUT/'dataset_manifest.json')
    assert sha(protocol['source'])==protocol['data_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protocol['source_sha256'].items())
    events=[json.loads(line) for line in (OUT/'events.jsonl').read_text(encoding='utf8').splitlines()]
    frozen_index=next(i for i,e in enumerate(events) if e['kind']=='ALL_AUDIT_CHOICES_FROZEN')
    frozen=read(OUT/'audit_choices_frozen.json')
    assert sha(OUT/'audit_choices_frozen.json')==events[frozen_index]['sha256']
    assert len([e for e in events[frozen_index+1:] if e['kind']=='REFERENCE_EVALUATION_OPEN' and e['stage']=='audit'])==16
    assert not any(e['kind']=='REFERENCE_EVALUATION_OPEN' and e['stage']=='audit' for e in events[:frozen_index])
    checked=0;max_metric_delta=0.;failures=[]
    for task in tasks:
        assert sha(task['coarse'])==task['coarse_sha256'] and sha(task['reference'])==task['reference_sha256']
        summary=read(OUT/'runs'/task['id']/'summary.json')
        assert summary['counts']['real_executions']==8 and len(summary['rows'])==8
        assert len({tuple(row['steps']) for row in summary['rows']})==8
        with xr.open_dataset(task['reference']) as ds:reference=ds.t2m.load()
        with xr.open_dataset(task['coarse']) as ds:coarse=ds.load()
        weights=np.outer(np.diff(np.sin(np.deg2rad(boundaries(reference.lat)))),np.diff(np.deg2rad(boundaries(reference.lon))))
        mask=np.outer((reference.lat>=coarse.lat.min())&(reference.lat<=coarse.lat.max()),(reference.lon>=coarse.lon.min())&(reference.lon<=coarse.lon.max()))
        for row in summary['rows']:
            if row['status']!='PASS':failures.append(dict(task=task['id'],key=row['key'],error=row.get('error')));continue
            assert sha(row['product'])==row['product_sha256']
            with xr.open_dataset(row['product']) as ds:
                assert np.array_equal(ds.lat,reference.lat) and np.array_equal(ds.lon,reference.lon)
                assert ds.t2m.attrs['units']=='K' and np.isfinite(ds.t2m).all()
                difference=ds.t2m.values-reference.values
            independent=float(np.sqrt(np.sum((difference**2*weights)[mask])/np.sum(weights[mask])))
            delta=abs(independent-row['RMSE']);max_metric_delta=max(max_metric_delta,delta)
            assert delta<1e-10
            checked+=1
        if task['split']=='audit':
            for choice in [c for c in frozen if c['task']==task['id']]:
                row=next(r for r in summary['rows'] if r['key']==choice['key'])
                assert row['steps']==choice['steps']
                if choice['scenario']=='M2':assert row['steps'][-1]=='raster_task.coarse_consistency'
    training=read(OUT/'training_report.json')
    assert training['training_groups']==[f'2019-08-0{i}' for i in range(1,5)]
    assert training['holdout_groups']==['2019-08-05'] and training['training_rows']==128 and training['holdout_rows']==32
    model=ROOT/'output/plan_ranking_models'/protocol['model_id']
    assert sha(model/'model.joblib')==training['model_sha256']
    evidence=read(model/'portrait_evidence.json')
    assert len(evidence['data_cases'])==128
    assert {c['source']['group'] for c in evidence['data_cases']}==set(training['training_groups'])
    recommendation_count=0
    for path in (OUT/'recommendations').glob('*/*/*/summary.json'):
        report=read(path)
        assert report['counts']['real_executions']==0 and report['best'] is None
        recommendation_count+=1
    assert recommendation_count==288 and checked+len(failures)==384
    strategies=pd.read_csv(OUT/'strategy_results.csv');audit=strategies[strategies.split=='audit']
    aggregates=audit.groupby(['scenario','strategy']).agg(tasks=('task','nunique'),records=('task','size'),RMSE=('RMSE','mean'),MAE=('MAE','mean'),Bias=('Bias','mean'),coarse_RMSE=('coarse_RMSE','mean'),regret=('regret','mean')).reset_index()
    aggregates.to_csv(OUT/'audit_strategy_aggregates.csv',index=False)
    combos=pd.read_csv(OUT/'combination_results.csv');combos=combos[combos.split=='audit']
    combo_stats=combos.groupby(['base','scale']).agg(n=('task','size'),mean_gain_K=('RMSE_gain','mean'),wins=('RMSE_gain',lambda x:int((x>1e-10).sum())),losses=('RMSE_gain',lambda x:int((x< -1e-10).sum())),base_RMSE=('base_RMSE','mean'),combination_RMSE=('combination_RMSE','mean')).reset_index()
    combo_stats.to_csv(OUT/'audit_combination_aggregates.csv',index=False)
    daily=audit.groupby(['date','scenario','strategy']).RMSE.mean().unstack('strategy')
    daily['learned_gain_over_fixed']=daily.fixed-daily.learned
    daily['learned_gain_over_random']=daily.random-daily.learned
    daily.to_csv(OUT/'audit_daily_paired_effects.csv')
    graphs=pd.read_csv(OUT/'all_graph_results.csv');g=graphs[graphs.split=='audit']
    best_single=g[~g.method.str.contains('coarse_consistency')].groupby('task').RMSE.min()
    best_combo=g[g.method.str.contains('coarse_consistency')].groupby('task').RMSE.min()
    pool_gap=best_single-best_combo
    diagnostics=dict(best_combo_vs_best_single_mean_gain_K=float(pool_gap.mean()),wins=int((pool_gap>1e-10).sum()),losses=int((pool_gap< -1e-10).sum()),ties=int((abs(pool_gap)<=1e-10).sum()),scope='post-hoc oracle only, not recommendation')
    support=[]
    for task in tasks:
        if task['split'] not in {'validation','audit'}:continue
        for scenario in ['M1','M2']:
            report=read(OUT/'recommendations'/task['id']/scenario/'learned'/'summary.json')
            support.append(dict(task=task['id'],split=task['split'],scenario=scenario,matching_status=report['matching_status'],**report['learning_status']))
    support_counts=pd.DataFrame(support).groupby(['split','scenario','matching_status']).size().reset_index(name='tasks')
    support_counts.to_csv(OUT/'learning_support_counts.csv',index=False)
    result=dict(status='PASS',products_verified=checked,failed_graphs=failures,max_RMSE_recalculation_delta=max_metric_delta,
        reference_free_recommendations_checked=recommendation_count,audit_choices_before_evaluation=True,
        train_holdout_disjoint=True,training_evidence_cases=128,source_code_and_data_unchanged=True,oracle_combination=diagnostics)
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(aggregates.to_string(index=False));print(combo_stats.to_string(index=False));print(support_counts.to_string(index=False));print(json.dumps(result,indent=2))


if __name__=='__main__':run()
