"""Read-only support and decision diagnostics for development iterations."""
import collections
import json
from pathlib import Path
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/spatial_next_20261023'
def diagnose(label,tasks,model_id):
    payload=joblib.load(ROOT/'output/plan_ranking_models'/model_id/'model.joblib')
    reasons=collections.Counter();states=collections.Counter();rows=[]
    for t in tasks:
        if t['stage']!='validation':continue
        p=OUT/'recommendations'/label/t['id']/'M1/learning/summary.json'
        report=json.loads(p.read_text(encoding='utf8'));states[report['matching_status']]+=1
        task_bad=set()
        for candidate in report['recommendations']:
            f=candidate['public_features'];categories={k:v for k,v in f.items() if isinstance(v,str)}
            train=[s for s in payload['support'] if {k:v for k,v in s.items() if isinstance(v,str)}==categories and set(s)==set(f)]
            if not train:task_bad.add('category_or_schema');continue
            for k,v in f.items():
                if not isinstance(v,str) and (not np.isfinite(v) or not min(x[k] for x in train)<=v<=max(x[k] for x in train)):task_bad.add(k)
        for k in task_bad:reasons[k]+=1
        rows.append(dict(task=t['id'],supported=report['learning_status']['supported_candidates'],
            candidates=len(report['recommendations']),reasons=sorted(task_bad)))
    data=dict(states=dict(states),out_of_range_task_counts=dict(reasons),tasks=rows,
        diagnosis='unseen context' if reasons.get('category_or_schema') else 'numerical support coverage' if reasons else 'supported; inspect selection quality',
        gate_changed=False,scope='validation only; not audit')
    (OUT/(label+'_diagnosis.json')).write_text(json.dumps(data,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in data.items() if k!='tasks'},indent=2));return data
