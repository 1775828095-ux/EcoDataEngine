"""Extract only Taiwan inputs and small documentation after official MD5 success."""
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from prepare_taiwan_archive import DEST

SIZE = 13435034939
MD5 = "d7329012a175b3ad52012d1f3d6dd50f"


def main():
    archive = DEST / "aod_pm25_data.tar.gz"
    evidence = json.loads((DEST / "download_verification.json").read_text(encoding="utf-8"))
    if not (archive.is_file() and archive.stat().st_size == SIZE and evidence.get("verified") is True
            and evidence.get("md5") == MD5 and evidence.get("bytes") == SIZE):
        raise ValueError("Official complete archive and matching MD5 are mandatory")
    inventory = DEST / "full_archive_member_inventory.csv"
    if inventory.exists():
        raise ValueError("Full inventory already exists; preserve it instead of overwriting")
    root = DEST / "provisional_members"
    documentation = DEST / "archive_documentation"
    rows, texts = [], []
    with inventory.open("w", encoding="utf-8-sig", newline="") as output, tarfile.open(archive, "r|gz") as stream:
        writer = csv.DictWriter(output, fieldnames=["member", "bytes", "kind", "extracted"])
        writer.writeheader()
        for member in stream:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe member path")
            taiwan = len(relative.parts) >= 2 and relative.parts[0] == "zhang_data" and relative.parts[1] in {"pm25_taiwan", "ahi_aod_taiwan"}
            small_text = member.isfile() and member.size < 2*1024*1024 and (
                relative.name.lower().startswith(("readme", "license", "metadata"))
                or relative.suffix.lower() in {".py", ".pro", ".r", ".m", ".ipynb", ".ncl", ".yaml", ".yml"})
            take = member.isfile() and (taiwan or small_text)
            writer.writerow(dict(member=member.name, bytes=member.size, kind="file" if member.isfile() else "other", extracted=take))
            if not take:
                continue
            if shutil.disk_usage(DEST).free < member.size + 2*1024**3:
                raise OSError("Insufficient free extraction space; originals retained")
            data = stream.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError("Incomplete member")
            target = (root if taiwan else documentation) / relative
            checksum = hashlib.sha256(data).hexdigest()
            if target.exists():
                if target.stat().st_size != member.size or hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                    raise ValueError(f"Existing source member differs: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o444)
            item = dict(member=member.name, file=str(target.relative_to(DEST)), bytes=member.size, sha256=checksum,
                official_archive_md5_verified=True, formal_time_and_datum_verified=False)
            (rows if taiwan else texts).append(item)
            if len(rows) and len(rows) % 500 == 0:
                output.flush()
                print(f"Taiwan members extracted/reused: {len(rows)}", flush=True)
    manifest = dict(source="https://zenodo.org/records/4420335", license="CC-BY-4.0", official_bytes=SIZE,
        official_md5=MD5, official_sha256=evidence.get("sha256"), archive_verified=True,
        original_prefix_manifests_preserved=True, taiwan_members=rows, documentation_members=texts,
        formal_data_contract_verified=False)
    (DEST / "verified_taiwan_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLETE: Taiwan files={len(rows)}; small source/documentation files={len(texts)}", flush=True)


if __name__ == "__main__":
    main()
