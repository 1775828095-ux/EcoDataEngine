"""Extract complete Taiwan members from the downloading official archive prefix.

Members remain provisional until the whole archive checksum is verified.
"""
import json
import tarfile
import hashlib
from pathlib import Path

from prepare_taiwan_archive import DEST


def main():
    output = DEST / "provisional_members"
    output.mkdir(exist_ok=True)
    rows = []
    progress_path = DEST / "range_progress.json"
    archive_path = DEST / "aod_pm25_data.tar.gz"
    if not archive_path.exists():
        archive_path = DEST / "aod_pm25_data.tar.gz.part"
    limit = archive_path.stat().st_size
    if progress_path.exists() and archive_path.suffix == ".part":
        progress = json.loads(progress_path.read_text())
        limit = progress["prefix_bytes"]
        completed = set(progress["completed"])
        while limit in completed:
            limit = min(progress["total"], limit + 16 * 1024 * 1024)

    class PrefixReader:
        def __init__(self, source):
            self.source = source
            self.remaining = limit

        def read(self, size):
            data = self.source.read(min(size, self.remaining))
            self.remaining -= len(data)
            return data

    try:
        with archive_path.open("rb") as source, tarfile.open(fileobj=PrefixReader(source), mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive member")
                if len(path.parts) < 2 or path.parts[1] not in {"pm25_taiwan", "ahi_aod_taiwan"}:
                    continue
                stream = archive.extractfile(member)
                data = stream.read()
                if len(data) != member.size:
                    raise ValueError("Incomplete member")
                target = output / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(data)
                checksum = hashlib.sha256(data).hexdigest()
                if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                    raise ValueError("Extracted member changed")
                rows.append(dict(member=member.name, bytes=member.size, sha256=checksum))
    except (tarfile.ReadError, EOFError) as error:
        print(f"Reached partial download boundary: {error}")
    (DEST / "provisional_member_inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    verification = DEST / "download_verification.json"
    verified = json.loads(verification.read_text()).get("verified", False) if verification.exists() else False
    print(f"Complete extracted members: {len(rows)}; full official archive checksum verified={verified}")


if __name__ == "__main__":
    main()
