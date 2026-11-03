"""Independent arithmetic/readback audit; never calls platform scoring/search."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/scheduled_fusion_20261025'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(ds,truth,var):
    y=ds[var].sel(time=truth.time)
    for a in truth.coords:assert np.array_equal(y[a],truth[a]),a
    assert y.attrs['units']==truth.attrs['units']=='K'
    e=(y-truth).values;assert np.isfinite(e).all()
    return float(np.sqrt(np.mean(e*e)))
def stationary(plan):
    return all(next(p['choice'] for p in plan if p['window']=='first' and p['branch']==b)==next(p['choice'] for p in plan if p['window']=='second' and p['branch']==b) for b in ('fine_slow','coarse_fast','fusion','post'))

def area_matrix(fine,coarse):
    def edges(a):
        a=np.asarray(a,float);return np.r_[a[0]-(a[1]-a[0])/2,(a[:-1]+a[1:])/2,a[-1]+(a[-1]-a[-2])/2]
    def overlap(f,c,latitude=False):
        f,c=edges(f),edges(c)
        if latitude:f,c=np.sin(np.deg2rad(f)),np.sin(np.deg2rad(c))
        width=np.maximum(0,np.minimum(c[1:,None],f[None,1:])-np.maximum(c[:-1,None],f[None,:-1]))
        return width/np.diff(c)[:,None]
    return np.kron(overlap(fine.lat,coarse.lat,True),overlap(fine.lon,coarse.lon))

def main():
    protocol=load(OUT/'frozen_protocol.json');result=load(OUT/'results.json');frozen=load(OUT/'frozen_selections.json')
    assert not protocol['test_seen_at_freeze'] and not frozen['test_labels_read']
    assert sha(Path(protocol['parent_data']['path']))==protocol['parent_data']['sha256']
    for item in load(OUT/'source_manifest.json'):
        source=Path(item['path']);frozen_source=OUT/'source_frozen'/source.relative_to(ROOT)
        assert sha(frozen_source)==item['sha256']
        if sha(source)!=item['sha256']:
            if source.name=='platform_search.py':
                receipt=load(OUT/'cli_type_repair.json')
                assert sha(source)==receipt['released_sha256'] and item['sha256']==receipt['frozen_sha256']
                continue
            if source.name=='run_scheduled_fusion_study.py':
                receipt=load(OUT/'orchestration_repair.json')
                assert sha(source)==receipt['released_sha256'] and item['sha256']==receipt['frozen_sha256']
                continue
            receipt=load(OUT/'boundary_diagnostic_fix.json')
            assert source.name=='composition_decision.py' and receipt['status']=='PASS'
            assert sha(source)==receipt['released_sha256'] and item['sha256']==receipt['frozen_sha256']
    for domain in protocol['domains']:assert domain['grammar_count']==domain['structural_legal']==1296 and domain['stationary_count']==36
    expected_cases={s for pair in protocol['validation_test_pairs'] for s in pair}|set(protocol['development'])
    assert len(expected_cases)==12
    labels={};times={}
    for start in expected_cases:
        config=load(OUT/'cases'/str(start)/'config.json');path=OUT/'cases'/str(start)/'labels.nc'
        assert sha(path)==config['labels_sha256']
        with xr.open_dataset(path,engine='h5netcdf') as ds:labels[start]=ds[config['request']['variable']].load()
        times[start]=pd.DatetimeIndex(labels[start].time)
        assert len(times[start])==14
        for i in config['inputs']:assert sha(Path(i['path']))==i['sha256']
        source_data={}
        for i in config['inputs']:
            with xr.open_dataset(i['path'],engine='h5netcdf') as ds:
                if i['role']=='primary':assert ds.sizes['time']==3 and not times[start].isin(ds.time.values).any()
                else:assert ds.sizes['time']==17 and ds.sizes['lat']==ds.sizes['lon']==6
                source_data[i['role']]=ds.load()
        high=xr.concat([source_data['primary'].air_temperature,labels[start]],dim='time').sortby('time')
        assert high.sizes['time']==17 and len(np.unique(high.time))==17
        low=source_data['background'];matrix=area_matrix(source_data['primary'],low)
        np.testing.assert_allclose(high.values.reshape(17,-1)@matrix.T,low.air_temperature.values.reshape(17,-1),atol=1e-9,rtol=0)
    for a in expected_cases:
        for b in expected_cases:
            if a!=b:assert not times[a].isin(times[b]).any(),(a,b)
    products=0;runs=0;oracles={};search={};changes=defaultdict(int);numeric_counts={}
    for path in sorted(OUT.rglob('summary.json')):
        if not any(part in {'oracles','search'} for part in path.parts):continue
        report=load(path);folder=path.parent
        if report.get('service')!='PlatformSearch/Composition':continue
        request=load(folder/'request.json');screen=load(folder/'initial_screening.json')
        assert screen['complete'] and not screen['sampling'] and screen['scored_count']==1296 and not screen['rejected']
        assert report['selection_objective']=='search_validation_RMSE' and report['search_validation_used_for_selection']
        assert not report['independent_test_used_for_selection'] and report['independent_evaluation'] is None
        vpath=Path(request['search_validation']['path']);assert sha(vpath)==request['search_validation']['sha256']
        with xr.open_dataset(vpath,engine='h5netcdf') as ds:truth=ds[request['request']['variable']].load()
        assert truth.attrs['split']=='search_validation'
        case=next(start for start in expected_cases if times[start].equals(pd.DatetimeIndex(truth.time)))
        assert load(OUT/'cases'/str(case)/'config.json')['phase']!='test'
        valid=[]
        for row in report['rows']:
            if row['status']!='PASS':assert row.get('error');continue
            p=Path(row['product']);assert sha(p)==row['product_sha256']
            with xr.open_dataset(p,engine='h5netcdf') as ds:value=rmse(ds,truth,request['request']['variable'])
            assert abs(value-row['search_validation']['RMSE'])<1e-9
            assert row['search_validation']['denominator']==2016
            assert row['execution_seconds']>=0 and row['completed_seconds']>=row['execution_seconds']
            valid.append(row);products+=1
        winner=min(valid,key=lambda r:(round(r['search_validation']['RMSE'],9),-round(r['matching']['score_interval'][0],9),r['scientific_steps'],r['index']))
        assert winner['key']==report['recommended_key'] and winner['search_validation']['RMSE']==report['final_validation_rmse']
        assert len(report['rows'])<=1296 and len({r['index'] for r in report['rows']})==len(report['rows'])
        if 'oracles' in path.parts:
            assert len(report['rows'])==1296 and report['counts']['exact_count_complete']
            assert report['certification']=='EXACT_VALIDATION_RMSE_OPTIMUM_IN_DECLARED_DOMAIN'
            oracles[case]={r['key']:r for r in valid}
            numeric_counts[case]=report['counts']['distinct_numeric_products_observed']
        else:
            method=path.parts[-3];seed=int(path.parts[-2]);search[case,method,seed]=report
            assert len(report['rows'])<=(2 if method=='none' else 64)
            assert screen.get('initialization','portrait')==('uniform_control' if method=='random' else 'portrait')
            for proposal in report['proposals']:
                for edit in proposal.get('edits',[]):changes[edit['axis'][2]]+=1
        runs+=1
    assert len(oracles)==8 and len(search)==80
    assert all(changes[k]>0 for k in ('coarse_fast','fusion','post'))
    fixed=load(OUT/'fixed_from_development.json')
    common=set.intersection(*(set(oracles[s]) for s in protocol['development']))
    def mean(k):return np.mean([oracles[s][k]['search_validation']['RMSE'] for s in protocol['development']])
    assert min(common,key=lambda k:(mean(k),k))==fixed['key']
    assert min([k for k in common if stationary(oracles[protocol['development'][0]][k]['plan'])],key=lambda k:(mean(k),k))==fixed['stationary_key']
    per_case=[]
    for a,b in protocol['validation_test_pairs']:
        optimum=min(r['search_validation']['RMSE'] for r in oracles[a].values())
        for method in protocol['methods']:
            rows=[r for r in result['runs'] if r['validation_case']==a and r['backend']==method];assert len(rows)==5
            for r in rows:
                report=search[a,method,r['seed']]
                assert r['key']==report['recommended_key']
                hits=[(i+1,x) for i,x in enumerate(report['rows']) if x['status']=='PASS' and abs(x['search_validation']['RMSE']-optimum)<=1e-9]
                assert bool(hits)==r['hit']
                assert r['first_hit_execution']==(hits[0][0] if hits else None)
                assert abs(r['validation_rmse']-report['final_validation_rmse'])<1e-9
                test_record=load(OUT/'test'/str(b)/r['key']/'evaluation.json')
                assert test_record['graph_frozen'] and test_record['plan']==r['plan']==report['best']['plan']
                assert (OUT/'frozen_selections.json').stat().st_mtime_ns<= (OUT/'test'/str(b)/r['key']/'evaluation.json').stat().st_mtime_ns
                p=Path(test_record['product']);assert sha(p)==test_record['product_sha256']
                with xr.open_dataset(p,engine='h5netcdf') as ds:value=rmse(ds,labels[b],'air_temperature')
                assert abs(value-r['test_rmse'])<1e-9
            hits=[r for r in rows if r['hit']]
            per_case.append(dict(case=a,method=method,successes=len(hits),runs=5,
                mean_test_rmse=float(np.mean([r['test_rmse'] for r in rows])),
                expected_evaluations_to_success=float(sum(r['first_hit_execution'] if r['hit'] else r['executions'] for r in rows)/len(hits)) if hits else None))
    # Baselines must also have independent products; never choose from test error.
    for r in result['baselines']:
        test_record=load(OUT/'test'/str(r['case'])/r['key']/'evaluation.json');p=Path(test_record['product'])
        assert sha(p)==test_record['product_sha256']
        with xr.open_dataset(p,engine='h5netcdf') as ds:assert abs(rmse(ds,labels[r['case']],'air_temperature')-r['rmse'])<1e-9
    # Real numeric witnesses, not just the availability of graph syntax.
    first=oracles[protocol['development'][0]];buckets=defaultdict(list)
    for row in first.values():
        plan=row['plan'];chain=next(p['choice'] for p in plan if p['window']=='first' and p['branch']=='coarse_fast')
        if len(chain)!=2:continue
        rest=[p for p in plan if not(p['window']=='first' and p['branch']=='coarse_fast')]
        methods=sorted(s['algorithm'] for s in chain)
        signature=json.dumps([rest,methods],sort_keys=True)
        buckets[signature].append(row)
    witness=next((rows[0],rows[1]) for rows in buckets.values() if len(rows)==2 and rows[0]['numeric_fingerprint']!=rows[1]['numeric_fingerprint'])
    p,q=witness
    with xr.open_dataset(p['product'],engine='h5netcdf') as ds:a=ds.air_temperature.load()
    with xr.open_dataset(q['product'],engine='h5netcdf') as ds:b=ds.air_temperature.load()
    np.testing.assert_allclose(a.isel(time=slice(7,None)),b.isel(time=slice(7,None)),atol=1e-12)
    assert not np.allclose(a.isel(time=slice(0,7)),b.isel(time=slice(0,7)),atol=1e-9,rtol=0)
    audit=dict(status='PASS',decisions=runs,real_numeric_products_checked=products,exact_domains=8,structural_legal_per_domain=1296,
        numeric_distinct_per_oracle=numeric_counts,actual_gene_edits=dict(changes),per_case=per_case,
        order_and_temporal_locality_witness=dict(keys=[p['key'],q['key']],first_window_numeric_changed=True,second_window_unchanged=True),
        source='Independent arithmetic over saved products/labels, hashes and frozen selections; no search or scorer called')
    (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
