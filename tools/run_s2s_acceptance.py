"""Run EPA's unchanged S2S code, changing only its documented mechanism setting."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/capability_final_acceptance"


def main():
    source=OUT/"official_s2s"
    run=OUT/"s2s_run"
    run.mkdir(exist_ok=True)
    for folder in ("input","modules"):
        shutil.copytree(source/folder,run/folder,dirs_exist_ok=True)
    (run/"output").mkdir(exist_ok=True)
    text=(source/"S2S.main.py").read_text(encoding="utf-8")
    assert text.count("MECH_BASIS = 'CB6R3_AE7'")==1
    configured=text.replace("MECH_BASIS = 'CB6R3_AE7'","MECH_BASIS = 'CB6R5_AE7'")
    os.chdir(run)
    start=time.perf_counter()
    exec(compile(configured,str(source/"S2S.main.py"),"exec"),{"__name__":"__main__"})
    files=[]
    for path in (run/"output").glob("*.txt"):
        files.append({"path":str(path.relative_to(OUT)),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/"s2s_execution.json").write_text(json.dumps({"status":"EXECUTED","commit":"ba5005aab36832248832012ae388b7ae897c2564",
        "mechanism":"CB6R5_AE7","configuration_change_only":True,"official_core_modified":False,"seconds":time.perf_counter()-start,"files":files},indent=2),encoding="utf-8")


if __name__=="__main__":main()
