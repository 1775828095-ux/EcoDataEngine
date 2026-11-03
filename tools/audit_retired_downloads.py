"""List only provenance-verified, retired NOAA raw downloads; never delete here."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/three_experiments_20261019"
OUT = ROOT / "output/taiwan_2017_preparation_20261019"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for file in (BASE / "anonymous").glob("satellite_manifest_*.json"):
        records.extend(json.loads(file.read_text(encoding="utf-8")))
    sources = {Path(r["file"]).name: r for r in records if r.get("sha256") and r.get("url") and r.get("file")}
    rows = []
    for path in sorted((BASE / "anonymous/original").glob("OR_ABI-L2-AODC*.nc")):
        source = sources.get(path.name)
        if source is None:
            continue
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != source["sha256"]:
            continue
        refs = []
        for folder in ["src", "tools", "tests", "docs"]:
            for file in (ROOT / folder).rglob("*"):
                if file.suffix not in {".py", ".json", ".md", ".yaml", ".yml"}:
                    continue
                if path.name in file.read_text(encoding="utf-8", errors="replace"):
                    refs.append(str(file.relative_to(ROOT)))
        rows.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, source=source["url"], sha256=checksum, direct_references=";".join(refs), eligible=not refs, reason="Retired California NOAA raw download; derived historical outputs, manifests and scripts retained; new route Taiwan AHI only"))
    pd.DataFrame(rows).to_csv(OUT / "cleanup_candidates.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(dict(files=len(rows), eligible=sum(r["eligible"] for r in rows), bytes=sum(r["bytes"] for r in rows if r["eligible"]))))


if __name__ == "__main__":
    main()
