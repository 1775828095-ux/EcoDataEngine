"""Execute unchanged official examples and save process evidence."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / "output/algorithm_library_expansion/upstream/GeoStat-Framework__PyKrige"
results = []
for example in ("00_ordinary.py", "01_universal.py"):
    process = subprocess.run([sys.executable, str(folder / "examples" / example)], cwd=folder,
        env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True, timeout=60)
    results.append({"example": example, "returncode": process.returncode,
                    "stdout": process.stdout, "stderr": process.stderr})
    if process.returncode:
        raise RuntimeError(process.stderr)
(folder.parent.parent / "pykrige_official_examples.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Official PyKrige examples:", [r["example"] for r in results])
