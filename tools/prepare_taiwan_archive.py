"""Fetch the fixed official research archive with bounded retries and checksums."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/taiwan_2017_zenodo4420335"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    metadata_path = DEST / "official_record.json"
    if metadata_path.exists():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        response = requests.get("https://zenodo.org/api/records/4420335", timeout=60)
        response.raise_for_status()
        record = response.json()
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if (DEST / "range_progress.json").exists() and not (DEST / record["files"][0]["key"]).exists():
        from resume_taiwan_ranges import main as resume
        resume()
        return
    entry = record["files"][0]
    final = DEST / entry["key"]
    target = final if final.exists() else DEST / (entry["key"] + ".part")
    url = entry["links"]["self"]
    if not target.exists() or target.stat().st_size != entry["size"]:
        subprocess.run(["curl.exe", "--location", "--fail", "--retry", "1", "--connect-timeout", "45", "--continue-at", "-", "--output", str(target), url], check=True)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    evidence = dict(url=url, bytes=target.stat().st_size, expected_bytes=entry["size"], md5=md5.hexdigest(), expected_checksum=entry["checksum"], sha256=sha.hexdigest(), checked_at=datetime.now(timezone.utc).isoformat())
    evidence["verified"] = evidence["bytes"] == entry["size"] and "md5:" + evidence["md5"] == entry["checksum"]
    (DEST / "download_verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence), flush=True)
    if not evidence["verified"]:
        raise ValueError("Official archive checksum mismatch")
    if target != final:
        target.rename(final)


if __name__ == "__main__":
    main()
