"""Score the separately frozen cost-matched arm; never alters choices."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np
import xarray as xr
from evaluate_open_complementary import OUT,PROTOCOL,materialize,load
from ecodata.algorithms.autonomous_field import measured,build_context
from ecodata.algorithms.numerical_search import save


def main():
    entries=load(OUT/'cost_matched_predictions_frozen.json');rows=[]
    assert (OUT/'predictions_frozen.json').exists()
    for entry in entries:
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        _,request,_,inputs,truth=materialize(PROTOCOL['test'][entry['case']]);context,_,_=build_context(inputs,request)
        with xr.open_dataset(path) as f:output=f.load()
        rows.append(dict(case=entry['case'],seed=entry['seed'],**measured(output,truth,context)))
    guided=load(OUT/'test_metrics.json')
    a=np.array([np.mean([r['objective'] for r in guided if r['case']==i and r['method'].startswith('contract_guided')]) for i in range(len(PROTOCOL['test']))])
    b=np.array([np.mean([r['objective'] for r in rows if r['case']==i]) for i in range(len(PROTOCOL['test']))])
    summaries=[]
    for stage in ['development','validation','test']:
        for p in (OUT/'runs_v6'/stage).glob('*/summary.json'):
            s=load(p);summaries.append(dict(stage=stage,run=p.parent.name,**{k:s[k] for k in ['executed_candidates','kernel_calls','adapter_invocations','elapsed_seconds','failed_candidates']}))
    for p in (OUT/'cost_matched').glob('*/summary.json'):
        s=load(p);summaries.append(dict(stage='cost_matched',run=p.parent.name,**{k:s[k] for k in ['executed_candidates','kernel_calls','adapter_invocations','elapsed_seconds','failed_candidates']}))
    result=dict(cost_matched_case_objectives=b.tolist(),guided_case_objectives=a.tolist(),
        guided_mean=float(a.mean()),cost_matched_random_mean=float(b.mean()),
        guided_relative_improvement_percent=float(100*(1-a.mean()/b.mean())),guided_case_wins=int((a<b).sum()),
        rows=rows,all_predictions_frozen_before_outcome_scoring=True,
        timing_limitations='Three worker processes; other local workload can affect wall time. Kernel costs also reported; no speed-optimality claim')
    save(OUT/'cost_matched_effect.json',result);save(OUT/'all_execution_costs.json',summaries)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
