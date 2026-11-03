"""Verify the frozen platform without installing scientific dependencies."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'handoff-protected-files.json').read_text(encoding='utf-8'))
failures = []
for item in manifest['files']:
    path = root / item['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        failures.append(item['path'])
if failures:
    print('FAIL: frozen platform files changed/missing:')
    print('\n'.join(failures))
    raise SystemExit(1)
print('PASS:', len(manifest['files']), 'frozen source/UI/configuration files unchanged')
