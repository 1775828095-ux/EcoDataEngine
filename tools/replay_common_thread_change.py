"""Replay pre-cap completed runs; verify the runtime change did not select wins."""
from pathlib import Path
import json
import numpy as np
import h5py
from threadpoolctl import threadpool_info
from run_common_reference_study import OUT,bundle,feedback
from run_scheduled_fusion_study import load,save
from ecodata.algorithms import platform_search as api


def main():
    boundary=(OUT/'runtime_thread_change.json').stat().st_mtime
    originals=[p for p in (OUT/'search').glob('*/*/*/summary.json') if p.stat().st_mtime<boundary]
    reports=[]
    for path in originals:
        identifier,method,seed=path.parents[2].name,path.parents[1].name,int(path.parent.name)
        old=load(path);req=load(path.parent/'request.json')['request'];inputs,_=bundle(identifier)
        folder=OUT/'thread_replay'/identifier/method/str(seed)
        new=load(folder/'summary.json') if (folder/'summary.json').exists() else api.submit_search(inputs,req,None,folder,mode='Profile',validation=feedback(identifier),seed=seed,budget=load(OUT/'protocol.json')['maximum_budget'])
        assert [r['index'] for r in old['rows']]==[r['index'] for r in new['rows']]
        assert [(p['child'],p['operator']) for p in old['proposals']]==[(p['child'],p['operator']) for p in new['proposals']]
        assert old['recommended_key']==new['recommended_key']
        maximum=0.
        for a,b in zip(old['rows'],new['rows']):
            with h5py.File(a['product']) as x,h5py.File(b['product']) as y:
                delta=float(np.max(np.abs(x['air_temperature'][:]-y['air_temperature'][:])))
            maximum=max(maximum,delta);assert delta<1e-10
        reports.append(dict(case=identifier,method=method,seed=seed,executions=len(old['rows']),max_absolute_difference_K=maximum,
            sequence_and_selection_identical=True))
        print(json.dumps(reports[-1]),flush=True)
    assert len(reports)==6
    save(OUT/'thread_replay_acceptance.json',dict(status='PASS',runs=reports,threadpools=threadpool_info(),
        scope='Runtime resource change only, no scientific rule changes; these supplementary replays are not additional efficiency repetitions'))


if __name__=='__main__':main()
