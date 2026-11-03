from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ENGINE = Path(r"D:\EcoDataHandoff_20261102\EcoDataEngine")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE / "src"))
from ecodata.interface.engineering_upload import LocalUpload

module_path = REPO / "integrations/existing_air/existing_air_converters/wrf_bundle_pipeline.py"
spec = importlib.util.spec_from_file_location("wrf_bundle_pipeline_workspace", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module._sst_update_enabled(b"&physics\n sst_update = 1,\n/\n") is True
assert module._sst_update_enabled(b"&physics\n sst_update = 0,\n/\n") is False
assert module._sst_update_enabled(b"&physics\n ! sst_update = 1,\n/\n") is False
assert module._sst_update_enabled(b"&physics\n sf_surface_physics = 1,\n/\n") is False

source = Path(r"C:\Users\ASUS\Desktop\WRF_20080205_完整工程与多源观测补缺数据_20261103.zip")
out = REPO.parent / "wrf_sst_gate_verification_20261103"
result = module.prepare([LocalUpload(source.name, source)], out)
inventory = json.loads((out / "bundle_inventory.json").read_text(encoding="utf-8"))
decision = inventory["route_decision"]
assert result is None
assert decision["status"] == "SST_SUPPLEMENT_NOT_REQUESTED"
assert decision["sst_update_enabled"] is False
assert decision["existing_initial_sst_present"] is True
print(json.dumps({"status": "PASS", "route_decision": decision}, ensure_ascii=False, indent=2))
