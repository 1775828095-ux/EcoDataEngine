"""Bounded official-data downloads; preserve upstream files without editing them."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion/v3"
UP = ROOT / "output/algorithm_library_expansion/upstream"


def download(url, target, max_bytes):
    if not target.exists():
        with urlopen(url, timeout=60) as response:
            expected = int(response.headers.get("Content-Length", 0))
            if expected > max_bytes:
                raise RuntimeError("Download exceeds explicit byte budget")
            content = response.read()
            if expected and len(content) != expected:
                raise RuntimeError("Incomplete download: HTTP Content-Length mismatch")
        if len(content) > max_bytes:
            raise RuntimeError("Download exceeds explicit byte budget")
        target.write_bytes(content)
    return {"url": url, "path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "checked_utc": datetime.now(timezone.utc).isoformat()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = {}
    source = UP / "WisdomTogether__DualTimesField"
    target = ROOT / "external/dualtimesfield"
    (target / "interpolation").mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "requirements.txt", "source_evidence.json", "interpolation/models.py", "interpolation/datasets.py", "interpolation/train.py"):
        shutil.copy2(source / name, target / name)
    record["license"] = {"declaration": "MIT License in upstream README", "standalone_LICENSE": (source / "LICENSE").exists(),
                         "note": "Do not fabricate absent LICENSE; retain declaration and source provenance"}
    record["ushcn"] = download("https://raw.githubusercontent.com/usail-hkust/t-PatchGNN/main/data/ushcn/raw/small_chunked_sporadic.csv",
                                OUT / "small_chunked_sporadic.csv", 32_000_000)
    record["phydl_archive"] = download("https://zenodo.org/records/15482043/files/yingtaoluo/PhyDL-NWP-publish.zip?download=1",
                                       OUT / "PhyDL-NWP-publish.zip", 12_000_000)
    with zipfile.ZipFile(OUT / "PhyDL-NWP-publish.zip") as archive:
        record["phydl_archive"]["members"] = archive.namelist()
        record["phydl_archive"]["data_members"] = [n for n in archive.namelist() if n.endswith((".npy", ".npz", ".mat", ".nc", ".csv"))]
    record["phydl_latest_archive"] = download("https://zenodo.org/records/15540502/files/yingtaoluo/PhyDL-NWP-publish.zip?download=1",
        OUT / "PhyDL-NWP-publish-latest.zip", 12_000_000)
    with zipfile.ZipFile(OUT / "PhyDL-NWP-publish-latest.zip") as archive:
        record["phydl_latest_archive"]["members"] = archive.namelist()
        record["phydl_latest_archive"]["data_members"] = [n for n in archive.namelist() if n.endswith((".npy", ".npz", ".mat", ".nc", ".csv"))]
    target = ROOT / "external/phydl_nwp"
    target.mkdir(exist_ok=True)
    source = UP / "yingtaoluo__PhyDL-NWP"
    for name in ("README.md", "LICENSE", "requirements.txt", "source_evidence.json", "main.py", "PDE_Discovery.py", "train_forecasting_models.py"):
        shutil.copy2(source / name, target / name)
    (OUT / "source_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "phydl_archive"}, ensure_ascii=False))
    print("PhyDL official archive data members:", record["phydl_archive"]["data_members"])


if __name__ == "__main__":
    main()
