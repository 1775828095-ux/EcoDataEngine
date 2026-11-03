"""Fetch bounded official S2S example inputs; keep originals and hashes."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import time
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/capability_final_acceptance"
COMMIT="ba5005aab36832248832012ae388b7ae897c2564"


def fetch(url):
    if "raw.githubusercontent.com" in url:
        with tempfile.TemporaryDirectory(prefix="official_download_") as directory:
            path=Path(directory)/"data"
            subprocess.run(["curl.exe","--http1.1","--fail","--location","--retry","2","--retry-all-errors","--max-time","120",url,"--output",str(path)],check=True,capture_output=True)
            return path.read_bytes()
    for i in range(3):
        try:
            with urlopen(Request(url,headers={"User-Agent":"EcoDataEngine-final-audit"}),timeout=40) as r:return r.read()
        except Exception:
            if i==2:raise
            time.sleep(1)


def main():
    folder=OUT/"official_s2s"
    tree=json.loads(fetch(f"https://api.github.com/repos/USEPA/S2S-Tool/git/trees/{COMMIT}?recursive=1"))["tree"]
    selected=[x for x in tree if x["type"]=="blob" and (
        x["path"].startswith("input/") or (x["path"].startswith("modules/") and x["path"].endswith(".py")) or
        x["path"] in {"S2S.main.py","readme.md","license.md","output/gspro.CB6R5_AE7_CRITERIA_VOC.CMAQ.2026-02-19.txt","output/gscnv.CB6R5_AE7_CRITERIA_VOC.CMAQ.2026-02-19.txt"})]
    assert sum(x.get("size",0) for x in selected)<40_000_000
    records=[]
    for entry in selected:
        target=folder/entry["path"]
        url=f"https://raw.githubusercontent.com/USEPA/S2S-Tool/{COMMIT}/{entry['path']}"
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(fetch(url))
        assert target.stat().st_size==entry["size"]
        records.append({"file":entry["path"],"url":url,"bytes":target.stat().st_size,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    (OUT/"s2s_source_manifest.json").write_text(json.dumps({"commit":COMMIT,"license":"MIT","files":records},indent=2),encoding="utf-8")
    print(json.dumps({"files":len(records),"bytes":sum(r['bytes'] for r in records)}))


if __name__=="__main__":main()
