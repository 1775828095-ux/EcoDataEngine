"""Bounded official HTTP-range download; resume the existing prefix, no duplicate archive."""
import hashlib
import json
import threading
import argparse
from time import monotonic
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from prepare_taiwan_archive import DEST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument("--endpoint", choices=["api", "direct"], default="direct")
    parser.add_argument("--backend", choices=["requests", "curl"], default="requests")
    parser.add_argument("--resolve-ip", help="Optional current official DNS address; HTTPS host/certificate stay zenodo.org")
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--max-failures", type=int, default=3)
    args = parser.parse_args()
    if args.max_minutes <= 0 or args.max_failures < 1:
        parser.error("Time budget and failure bound must be positive")
    if args.resolve_ip and args.backend != "curl":
        parser.error("--resolve-ip requires curl")
    deadline = monotonic() + args.max_minutes * 60
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    record = json.loads((DEST / "official_record.json").read_text(encoding="utf-8"))
    entry = record["files"][0]
    path = DEST / (entry["key"] + ".part")
    state_path = DEST / "range_progress.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = dict(prefix_bytes=path.stat().st_size, completed=[], errors=[], total=entry["size"])
        state_path.write_text(json.dumps(state, indent=2))
    lock = threading.Lock()
    chunk_size = 16 * 1024 * 1024
    segments = [(start, min(entry["size"]-1, start+chunk_size-1)) for start in range(state["prefix_bytes"], entry["size"], chunk_size)]

    def transfer(segment):
        start, end = segment
        if start in state["completed"]:
            return
        if monotonic() >= deadline:
            raise TimeoutError("Bounded download wall-clock budget exhausted")
        url = entry["links"]["self"] if args.endpoint == "api" else f"https://zenodo.org/records/4420335/files/{entry['key']}?download=1"
        for attempt in range(2):
            try:
                expected = f"bytes {start}-{end}/{entry['size']}"
                if args.backend == "curl":
                    remaining = deadline-monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Bounded download wall-clock budget exhausted")
                    parts = DEST / ("range_parts_" + run_id)
                    parts.mkdir(exist_ok=True)
                    part = parts / f"{start}_{end}_attempt{attempt}.part"
                    seconds = min(120, remaining)
                    route = ["-4", "--resolve", "zenodo.org:443:" + args.resolve_ip] if args.resolve_ip else []
                    process = subprocess.run(["curl.exe", *route, "--http1.1", "--location", "--fail", "--silent", "--show-error",
                        "--retry", "0", "--connect-timeout", "20", "--max-time", str(seconds), "--max-filesize", str(end-start+1),
                        "--range", f"{start}-{end}", "--dump-header", "-", "--output", str(part), url],
                        capture_output=True, text=True, timeout=seconds+10)
                    ranges = re.findall(r"^content-range:\s*(.+)$", process.stdout, flags=re.I | re.M)
                    if process.returncode or not ranges or ranges[-1].strip() != expected:
                        raise ValueError(f"curl range failed: exit={process.returncode}; content_range={ranges[-1].strip() if ranges else None}; error={process.stderr[-500:]}")
                    data = part.read_bytes()
                    if len(data) != end-start+1:
                        raise ValueError("Incomplete curl range")
                    with path.open("r+b") as output:
                        output.seek(start)
                        output.write(data)
                else:
                  with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 45), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Range contract failed: {response.status_code} {response.headers.get('Content-Range')}")
                    written = 0
                    with path.open("r+b") as output:
                        output.seek(start)
                        for data in response.iter_content(256 * 1024):
                            if written+len(data) > end-start+1:
                                raise ValueError("Excess range bytes")
                            output.write(data)
                            written += len(data)
                    if written != end-start+1:
                        raise ValueError("Incomplete range")
                with lock:
                    state["completed"].append(start)
                    state.setdefault("attempt_history", []).append(dict(start=start, end=end, endpoint=args.endpoint, backend=args.backend, workers=args.workers, resolve_ip=args.resolve_ip, status="COMPLETE", utc=datetime.now(timezone.utc).isoformat()))
                    state_path.write_text(json.dumps(state, indent=2))
                return
            except Exception as error:
                if attempt == 1:
                    with lock:
                        state["errors"].append(dict(start=start, end=end, endpoint=args.endpoint, backend=args.backend, workers=args.workers, resolve_ip=args.resolve_ip, error=repr(error), utc=datetime.now(timezone.utc).isoformat()))
                        state_path.write_text(json.dumps(state, indent=2))
                    raise

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transfer, segment): segment for segment in segments}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(repr(error))
                if len(failures) >= args.max_failures:
                    for pending in futures:
                        pending.cancel()
                    break
            if len(state["completed"]) % 16 == 0:
                print(f"Complete ranges {len(state['completed'])}/{len(segments)}; failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} ranges failed; details preserved in range_progress.json")
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
            md5.update(data); sha.update(data)
    result = dict(url=entry["links"]["self"], bytes=path.stat().st_size, md5=md5.hexdigest(), sha256=sha.hexdigest(), expected_checksum=entry["checksum"], verified="md5:"+md5.hexdigest() == entry["checksum"], checked_at=datetime.now(timezone.utc).isoformat())
    (DEST / "download_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["verified"]:
        raise ValueError("Official checksum mismatch")
    path.rename(DEST / entry["key"])


if __name__ == "__main__":
    main()
