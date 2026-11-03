"""Run the first two code cells of the pinned upstream basic-usage notebook."""
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import numpy as np

root = Path(__file__).resolve().parents[1] / "output/algorithm_library_expansion"
path = root / "upstream/Ouranosinc__xsdba/docs/notebooks/example.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
namespace = {}
np.random.seed(11)
for index in (1, 2):
    source = "".join(notebook["cells"][index]["source"])
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    exec(compile(source, str(path) + f":cell{index}", "exec"), namespace)
assert np.isfinite(namespace["scen"]).all()
result = {"source": str(path), "cells": [1, 2], "only_adaptation": "omit IPython matplotlib magic; use Agg renderer",
          "sample_count": namespace["scen"].size, "finite": True, "seed": 11,
          "status": "PASS", "scope": "official EQM basic example; DQM/QDM tested by EcoData adapters"}
(root / "xsdba_official_example.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(result)
