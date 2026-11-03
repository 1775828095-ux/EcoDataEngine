"""Re-execute existing reliable SWAT+ writers; no replacement or model-run claim."""
import hashlib
import json
from pathlib import Path

from ecodata.round1 import convert

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"examples/ai_round1/swat_plus"
OUT=ROOT/"output/algorithm_library_expansion/v5/swat_existing"


def main():
    cases=[("soil","A_official_valid_subset.csv","config.json"),
           ("landuse","A_official_vegetated_bundle.zip","config_official.json"),
           ("management","A_synthetic_standard.zip","config_standard.json")]
    rows=[]
    for kind,name,config in cases:
        cfg=json.loads((BASE/kind/config).read_text(encoding="utf-8"))
        source=BASE/kind/name
        files,report=convert("swat_plus",source.read_bytes(),cfg)
        folder=OUT/kind;folder.mkdir(parents=True,exist_ok=True)
        outputs=[]
        for filename,content in files.items():
            path=folder/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            assert content
            outputs.append({"file":str(path.relative_to(ROOT)),"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()})
        rows.append({"case":kind,"input":str(source.relative_to(ROOT)),"input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "source":cfg.get("source"),"outputs":outputs,"report":report,
            "official_software_read_verified":False,"official_model_run_verified":False})
    (OUT.parent/"swat_adapter_audit.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps([{r['case']:len(r['outputs'])} for r in rows]))


if __name__=="__main__":main()
