"""Final artifact integrity check; no algorithm reruns or test-set fitting."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def verify(root):
    records=json.loads((root/"sealed_reference/manifest.json").read_text(encoding="utf-8"))
    assert len(records)==180 and len({r["field_id"] for r in records})==90
    for r in records:assert hashlib.sha256(Path(r["input"]).read_bytes()).hexdigest()==r["input_sha256"]
    outputs=0
    for f in (root/"runs").glob("*/metrics.json"):
        record=json.loads(f.read_text(encoding="utf-8"))
        assert len(record["metrics"])==8
        for m in record["metrics"]:
            assert hashlib.sha256(Path(m["output"]).read_bytes()).hexdigest()==m["output_sha256"]
            assert m["coverage"]==1 and m["negative_count"]==0 and m["quality"]["status"]=="PASS"
            outputs+=1
    assert outputs==1440
    freeze=json.loads((root/"pretest_freeze.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((root/"router.joblib").read_bytes()).hexdigest()==freeze["router_sha256"]
    assert hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()==freeze["protocol_sha256"]
    tests=0
    for name in ["targeted_tests.xml","repro_tests.xml"]:
        suite=ET.parse(root/name).getroot()[0]
        assert all(suite.attrib[key]=="0" for key in ["failures","errors","skipped"])
        tests+=int(suite.attrib["tests"])
    value=dict(input_hashes_verified=len(records),output_hashes_verified=outputs,model_and_protocol_unchanged_since_pretest=True,
        related_distinct_tests_passed=tests,failures=0,skipped=0,synthetic_scope_only=True,
        main_input_only_routing="EXECUTED",simple_task_avoids_qp="SUPPORTED_IN_THIS_TASK",
        smooth_field_composition_benefit="SUPPORTED_IN_THIS_TASK",peak_routing_beats_strongest_fixed="NOT_PROVEN",
        real_environmental_accuracy="NOT_VALIDATED",files_and_history_deleted=False,
        source_snapshot_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"source_executed").glob("*.py")})
    (root/"final_acceptance.json").write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(value,ensure_ascii=True,indent=2))


if __name__=="__main__":verify(Path("output/three_task_routing_20261020"))
