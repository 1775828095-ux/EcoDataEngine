"""Old-data public-entry acceptance; never imports new-data test labels."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/portrait_response_20261028'


def run(kind,self_validation=False):
    config=json.loads((ROOT/'output/robit_era5_mutation_20261028/cases'/kind/'config.json').read_text())
    config.update(mode='PortraitMutation',seed=11,budget=dict(objective_limit=180,proposal_limit=40000,wall_seconds=120))
    config['request']['response_scoring']=dict(period=['1996-01-01','2001-12-31'],weights=dict(data=.5,model=.5,collaboration=0.),
       evidence='Reuse original frozen daily visible-response experimental priorities; not measured SWAT sensitivity')
    if not self_validation:config['search_validation']=dict(path=str(ROOT/'output/robit_era5_fusion_20261027/cases'/kind/'development_labels.nc'))
    folder=OUT/'mechanism_v3'/(kind+('_self' if self_validation else ''))
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    save(folder/'client_config.json',config)
    return api.submit_file_request(config,folder)


if __name__=='__main__':
    paths=list((ROOT/'src/ecodata').rglob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(OUT/'mechanism_source_hashes.json',hashes)
    checks=[]
    for kind,self_validation in [('short',False),('long',False),('short',True)]:
        r=run(kind,self_validation);assert r['status']=='PASS',r['status']
        assert r['total_cost']['used']<=180
        assert r['response_score_intervals']>1
        assert r['parent_policy']=='best'
        folder=OUT/'mechanism_v3'/(kind+('_self' if self_validation else ''))
        trace=json.loads((folder/'mutation_trace.json').read_text())
        assert trace and all(t['parent_is_best_at_proposal'] for t in trace)
        records=json.loads((folder/'graphs.json').read_text())
        def count(key):
            seen=set()
            def visit(k):
                if k in seen:return
                seen.add(k)
                for p in records[k]['parents']:visit(p)
            visit(key)
            return sum(records[k]['operation']!='source' for k in seen)
        for t in trace:
            if t['accepted'] and t['after_objective']==t['before_objective']:
                assert count(t['child'])<=count(t['parent'])
        joints=[t for t in trace if t['family']=='joint']
        assert joints
        assert any(t['edit']['destruction']['removed_nodes'] for t in joints)
        checks.append(dict(case=kind,self_validation=self_validation,status='PASS',total_new_calls=r['total_cost']['used'],
          response_score_intervals=r['response_score_intervals'],scientific_duplicates=r['scientific_state_deduplications'],
          joint_trials=len(joints),selected_nodes=r['best']['algorithm_nodes']))
        print(checks[-1],flush=True)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    save(OUT/'mechanism_acceptance.json',dict(status='PASS',old_data_only=True,source_hashes_checked=True,checks=checks))
