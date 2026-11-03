"""Independent-case execution only; delegates to the frozen public runner."""
import sys,json
from pathlib import Path
import run_scheduled_fusion_study as study
case=int(sys.argv[1]);study.stores()
protocol=study.load(study.OUT/'frozen_protocol.json')
assert case in [a for a,b in protocol['validation_test_pairs']]
for item in study.load(study.OUT/'source_manifest.json'):assert study.sha(Path(item['path']))==item['sha256']
for seed in study.SEEDS:
    for backend in study.METHODS:
        study.run(case,backend,seed)
    print(json.dumps(dict(case=case,seed_completed=seed)),flush=True)
study.run(case,'none',11,exact=True)
print(json.dumps(dict(case=case,complete=True)),flush=True)
