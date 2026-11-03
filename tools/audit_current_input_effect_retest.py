"""Independent saved-artifact checks and descriptive comparisons, no reranking."""
from collections import Counter
from pathlib import Path
import json,hashlib
import numpy as np
import xarray as xr
from run_current_input_effect_retest import ROOT,OLD,BEFORE,OUT,BUDGET,SEEDS,cases,config,load,digest,inputs
from ecodata.algorithms import platform_search as api


def save(path,row):Path(path).write_text(json.dumps(row,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def run():
    for name,value in load(OUT/'source_manifest.json').items():assert digest(ROOT/name)==value
    for name,value in load(OUT/'input_manifest.json').items():assert digest(OLD/name)==value
    for name,value in load(OUT/'historical_manifest.json').items():assert digest(ROOT/name)==value
    expected={(c,m,s) for c in cases() for m in ['random','none'] for s in SEEDS}
    assert expected=={tuple(r) for r in load(OUT/'planned_jobs.json')}
    metrics=load(OUT/'test_metrics.json');lookup={(r['case'],r['method']):r for r in metrics}
    supports={};products=[];runs=[];masks={};replays=[]
    for c,m,s in sorted(expected):
        folder=OUT/'runs'/f'{c}_{m}_{s}';client=load(folder/'client_result.json');platform=folder/'platform'
        assert client['external_reference_reads']==[]
        if not (platform/'summary.json').exists():
            runs.append(dict(case=c,mode=m,seed=s,status=client['status'],failure=client.get('error')));continue
        summary=load(platform/'summary.json');request=load(platform/'request.json')
        continuation=load(OUT/'deployment_continuation'/folder.name/'receipt.json')
        assert continuation['external_reference_reads']==[] and continuation['new_search_calls']==0
        extra=continuation['cost'];assert extra['used']==len(extra['events'])<=10000
        assert all(request['budget'][k]==v for k,v in BUDGET.items())
        assert summary['optimizer']['backend']==m
        if m=='random':
            assert summary['evolution_used']
            assert request['request']['decision_policy']=='current_input'
            assert not {'fixed_graph','composition'}&set(request['request'])
        else:assert not summary['evolution_used']
        assert summary['self_validation_from_current_input']
        assert not summary['confirmation']['used_for_selection'] if summary.get('confirmation') else True
        protocol=request['self_validation'];assert not protocol['external_test_used']
        assert protocol['current_input_sha256']==[digest(i['path']) for i in config(c)['inputs']]
        with xr.open_dataset(platform/'self_validation_labels.nc') as f:search=f.ndvi.load()
        with xr.open_dataset(platform/'self_confirmation_labels.nc') as f:confirm=f.ndvi.load()
        with xr.open_dataset(config(c)['inputs'][0]['path']) as f:visible=f.ndvi.load()
        left=np.isfinite(search.reindex(time=visible.time).values);right=np.isfinite(confirm.reindex(time=visible.time).values)
        assert not (left&right).any() and not ((left|right)&~np.isfinite(visible.values)).any()
        identity=(hashlib.sha256(left.tobytes()).hexdigest(),hashlib.sha256(right.tobytes()).hexdigest())
        if c in masks:assert masks[c]==identity
        masks[c]=identity
        for n in protocol['masked_input_indices']:
            with xr.open_dataset(platform/f'self_validation_roots/source{n}.nc') as f:
                for ref in [search,confirm]:assert np.isnan(f.ndvi.sel(time=ref.time).values[np.isfinite(ref.values)]).all()
        cost=load(platform/'total_cost.json');assert cost['used']==len(cost['events'])<=BUDGET['objective_limit']
        attempts=[json.loads(line) for line in (platform/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in attempts:
            if row['status']=='ELIGIBLE_OUTPUT':
                assert 'metrics' in row
                assert row.get('trace') or row['algorithm_nodes']==0 or row['origin']=='scientific_state_alias'
            if row['status'] in ['EXECUTION_FAILED','BUDGET_STOP']:assert 'metrics' not in row
        graphs=load(platform/'graphs.json');ops=Counter();seen=set();pending=[continuation['key']] if continuation.get('key') else []
        while pending:
            node=pending.pop()
            if node in seen:continue
            seen.add(node);record=graphs[node];pending+=record['parents']
            if record['operation']!='source':ops[record['operation']]+=1
        response=load(platform/'response_scoring.json')
        runs.append(dict(case=c,mode=m,seed=s,status=continuation['status'],original_status=summary['status'],
            calls=cost['used']+extra['used'],seconds=cost['service_seconds']+extra['service_seconds'],
            original_calls=cost['used'],additional_deployment_calls=extra['used'],
            deploy_calls=cost['phases'].get('deploy',0)+extra['phases'].get('deploy',0),failed_calls=cost['failed_calls']+extra['failed_calls'],evaluations=len(attempts),
            eligible=sum(r['status']=='ELIGIBLE_OUTPUT' for r in attempts),selected_operations=dict(ops),
            response_unsupported=response.get('unsupported'),confirmation=summary.get('confirmation'),
            deployment=continuation['deployment'],external_reference_reads=0))
    for entry in load(OUT/'predictions_frozen.json'):
        assert digest(entry['path'])==entry['sha256']
        c=entry['case'];row=lookup[c,entry['method']]
        with xr.open_dataset(OLD/'cases'/c/'test_labels.nc') as f:truth=f.ndvi.load()
        with xr.open_dataset(entry['path']) as f:out=f.ndvi.load()
        predicted=out.sel(time=truth.time,lat=truth.lat,lon=truth.lon).values.astype(float)
        reference=truth.values.astype(float);finite=np.isfinite(reference)
        supports.setdefault(c,int(finite.sum()));assert supports[c]==int(finite.sum())
        assert np.isfinite(predicted[finite]).all()
        squared=np.where(finite,(predicted-reference)**2,np.nan)
        objective=float(np.sqrt(np.nanmean(squared,axis=(1,2))).mean())
        np.testing.assert_allclose(objective,row['objective'],rtol=1e-12,atol=1e-12)
        assert row['denominator']==supports[c]
        assert out.attrs['units']==truth.attrs['units']
        with xr.open_dataset(config(c)['inputs'][0]['path']) as f:visible=f.ndvi.load()
        known=np.isfinite(visible.values);np.testing.assert_array_equal(out.sel(time=visible.time).values[known],visible.values[known])
        products.append(dict(case=c,method=entry['method'],objective=objective,denominator=supports[c],observations_preserved=True))
        if entry['method']=='random_11':
            folder=OUT/'independent_replay'/c;cfg=config(c)
            receipt=load(folder/'deployment.json') if (folder/'deployment.json').exists() else api.execute_frozen_graph(inputs(cfg),cfg['request'],entry['graph'],folder)
            with xr.open_dataset(receipt['product']) as f:xr.testing.assert_equal(out,f.ndvi.load())
            replays.append(dict(case=c,method=entry['method'],status='PASS',saved_graph_replay_equal=True))
    effect=load(OUT/'effect_summary.json');before=load(BEFORE/'effect_summary.json');comparisons=[]
    for key in ['none','development_fixed']:
        a,b=effect['random'],effect[key]
        if a['status']=='PASS' and b['status']=='PASS':
            comparisons.append(dict(candidate='random',reference=key,status='PASS',improvement_percent=100*(b['RMSE']-a['RMSE'])/b['RMSE'],
                case_wins=sum(a['by_case'][c]<b['by_case'][c] for c in cases()),cases=len(cases())))
        else:comparisons.append(dict(candidate='random',reference=key,status='INCOMPLETE'))
    for key in ['legacy_random','expanded_random','expanded_none']:
        a,b=effect['random'],before[key]
        row=dict(candidate='random',reference='before_'+key,status='INCOMPLETE',protocol_difference=True)
        if a['status']=='PASS' and b['status']=='PASS':
            row.update(status='PASS',improvement_percent=100*(b['RMSE']-a['RMSE'])/b['RMSE'])
        comparisons.append(row)
    old_metrics=load(BEFORE/'test_metrics.json');paired=[]
    for c in cases():
        old=[r for r in old_metrics if r['case']==c and r['method'].startswith('expanded_random_')]
        new=[r for r in metrics if r['case']==c and r['method'].startswith('random_')]
        complete=all(len(rows)==3 and all(r['status']=='PASS' for r in rows) for rows in [old,new])
        pair=dict(case=c,status='PASS' if complete else 'INCOMPLETE',old_count=len(old),new_count=len(new))
        if complete:
            x,y=[float(np.mean([r['objective'] for r in rows])) for rows in [old,new]]
            pair.update(before=x,after=y,improvement_percent=100*(x-y)/x)
        paired.append(pair)
    costs={m:dict(calls=sum(r.get('calls',0) for r in runs if r['mode']==m),
        service_seconds=sum(r.get('seconds',0) for r in runs if r['mode']==m)) for m in ['random','none']}
    report=dict(status='PASS',jobs_checked=len(runs),expected_jobs=len(expected),products_checked=len(products),
        input_source_historical_hashes_unchanged=True,identical_holdouts_across_controls_and_seeds=True,
        external_reference_reads_during_decisions=0,all_reported_metrics_real_execution=True,shared_support=supports,
        comparisons=comparisons,paired_before_after=paired,costs=costs,
        independent_graph_replays=replays,
        failure_count=len(load(OUT/'failures.json')),historical_test_already_seen=True,
        original_budget_failures=len(load(OUT/'original_budget_failures.json')),
        deployment_continuation='Uniform additional deployment allowance, no extra search or changed rank; original failures retained',
        scope='Current-input operational mode retrospective check; no isolated code-only effect or universal superiority claim')
    save(OUT/'independent_audit.json',report);save(OUT/'independent_metrics.json',products);save(OUT/'runs_audit.json',runs)
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
