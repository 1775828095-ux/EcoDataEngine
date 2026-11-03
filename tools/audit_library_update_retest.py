"""Independent artifact, cohort, library-membership and numerical checks."""
from pathlib import Path
import json,hashlib
from collections import Counter,defaultdict
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from run_library_update_retest import ROOT,OLD,OUT,BUDGET,SEEDS,cases,config,load,BASE_FACTORY


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def run():
    for name,digest in load(OUT/'source_manifest.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in load(OUT/'input_manifest.json').items():assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==digest
    for name,digest in load(OLD/'input_manifest.json').items():assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==digest
    legacy=set(load(OUT/'legacy_algorithm_ids.json'));added=set(load(OUT/'protocol.json')['added_algorithms'])
    expected={(lib,c,mode,s) for lib in ['legacy','expanded'] for c in cases() for mode in ['random','none'] for s in SEEDS}
    assert {tuple(x) for x in load(OUT/'planned_jobs.json')}==expected
    jobs=[];usage=Counter();winners=Counter();failures=[];shared_budget=None
    for lib,c,mode,seed in sorted(expected):
        folder=OUT/'runs'/lib/f'{c}_{mode}_{seed}'
        if not (folder/'summary.json').exists():
            failures.append(dict(library=lib,case=c,mode=mode,seed=seed,stage='search'));continue
        request=load(folder/'request.json');assert all(request['budget'][k]==v for k,v in BUDGET.items())
        if shared_budget is None:shared_budget=request['budget']
        assert request['budget']==shared_budget
        if mode=='random':
            assert request['request']['mutation_control']=='random'
            assert not {'fixed_graph','composition'}&set(request['request'])
        catalog=load(folder/'catalog.json');ids={r['id'].removeprefix('framewise.') for r in catalog}
        if lib=='legacy':assert not ids&added
        else:assert added<=ids
        summary=load(folder/'summary.json');cost=load(folder/'total_cost.json')
        assert summary['optimizer']['backend']==mode
        if mode=='random':assert summary['evolution_used'] is True
        assert len(cost['events'])==cost['used'] and cost['used']<=cost['limit']==BUDGET['objective_limit']
        for event in cost['events']:
            if event['algorithm'].removeprefix('framewise.') in added:usage[event['algorithm'],event['status']]+=1
        graph=load(folder/'graphs.json');selected=Counter()
        if summary.get('best'):
            visited=set()
            def visit(k):
                if k in visited:return
                visited.add(k)
                row=graph[k]
                if row['operation']!='source':
                    winners[lib,row['operation']]+=1;selected[row['operation']]+=1
                for parent in row['parents']:visit(parent)
            visit(summary['best']['key'])
        jobs.append(dict(library=lib,case=c,mode=mode,seed=seed,status=summary['status'],used=cost['used'],seconds=summary.get('total_service_seconds'),
            selected_nodes=sum(selected.values()),distinct_operations=len(selected),repeated_operations={k:n for k,n in selected.items() if n>1}))
    metrics=load(OUT/'test_metrics.json');lookup={(r['case'],r['method']):r for r in metrics}
    support={};recomputed=[];protection=[]
    for entry in load(OUT/'predictions_frozen.json'):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        case=entry['case'];record=lookup[case,entry['method']]
        with xr.open_dataset(OLD/'cases'/case/'test_labels.nc') as d:ref=d.ndvi.load()
        with xr.open_dataset(path) as d:out=d.ndvi.load()
        assert out.attrs['units']==ref.attrs['units']
        predicted=out.sel(time=ref.time,lat=ref.lat,lon=ref.lon).values.astype(float);truth=ref.values.astype(float);mask=np.isfinite(truth)
        support.setdefault(case,int(mask.sum()));assert support[case]==int(mask.sum())
        if record['status']=='PASS':
            assert np.isfinite(predicted[mask]).all()
            e=np.where(mask,predicted-truth,np.nan);objective=float(np.sqrt(np.nanmean(e*e,axis=(1,2))).mean())
            np.testing.assert_allclose(objective,record['objective'],rtol=1e-12,atol=1e-12)
            assert record['denominator']==support[case]
            recomputed.append(dict(case=case,method=entry['method'],objective=objective))
        cfg=config(case,'test');source_path=Path(cfg['inputs'][0]['path'])
        with xr.open_dataset(source_path) as d:visible=d.ndvi.load()
        selected=out.sel(time=visible.time).values;known=np.isfinite(visible.values)
        np.testing.assert_array_equal(selected[known],visible.values[known])
        protection.append(dict(case=case,method=entry['method'],known_values=int(known.sum()),status='PASS'))
    summary=load(OUT/'effect_summary.json');comparisons=[]
    case_effects=[]
    for case in cases():
        values={lib:[r for r in metrics if r['case']==case and r['method'].startswith(lib+'_random_')] for lib in ['legacy','expanded']}
        entry=dict(case=case,expected_per_library=3,observed={k:len(v) for k,v in values.items()})
        if all(len(v)==3 and all(r['status']=='PASS' for r in v) for v in values.values()):
            means={k:float(np.mean([r['objective'] for r in v])) for k,v in values.items()}
            entry.update(status='PASS',RMSE=means,improvement_percent=100*(means['legacy']-means['expanded'])/means['legacy'])
        else:entry['status']='INCOMPLETE'
        case_effects.append(entry)
    for a,b in [('expanded_random','legacy_random'),('expanded_none','legacy_none'),('expanded_random','expanded_none'),('expanded_random','development_fixed')]:
        if summary[a]['status']!='PASS' or summary[b]['status']!='PASS':
            comparisons.append(dict(candidate=a,reference=b,status='INCOMPLETE'));continue
        comparisons.append(dict(candidate=a,reference=b,status='PASS',improvement_percent=100*(summary[b]['RMSE']-summary[a]['RMSE'])/summary[b]['RMSE'],
            case_wins=sum(summary[a]['by_case'][c]<summary[b]['by_case'][c] for c in cases()),cases=len(cases())))
    report=dict(status='PASS',jobs_checked=len(jobs),expected_jobs=72,search_failures=failures,input_hashes_unchanged=True,
        source_hashes_unchanged=True,library_membership_verified=True,same_budget_verified=True,
        independent_primary_metric_checks=len(recomputed),support=support,visible_observation_protection_checks=len(protection),
        newly_added_calls=[dict(algorithm=k[0],status=k[1],count=n) for k,n in sorted(usage.items())],
        selected_operations=[dict(library=k[0],algorithm=k[1],count=n) for k,n in sorted(winners.items())],comparisons=comparisons,
        per_case_comparison=case_effects,historical_test_already_seen=True,scientific_scope='Retrospective same-data library comparison; three dependent patches, no general superiority claim')
    save(OUT/'independent_audit.json',report);save(OUT/'independent_metrics.json',recomputed);save(OUT/'visible_protection.json',protection)
    save(OUT/'selected_graph_complexity.json',jobs)
    print(json.dumps(comparisons,indent=2))
    print('AUDIT PASS',len(jobs),'JOBS',len(recomputed),'PRIMARY METRIC CHECKS')


if __name__=='__main__':run()
