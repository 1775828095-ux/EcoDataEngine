"""Fetch small official source evidence, never model weights or full datasets."""
import argparse
import hashlib
import json
from http.client import IncompleteRead
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "EcoDataEngine-source-audit"}), timeout=40) as response:
                return response.read()
        except (IncompleteRead, URLError):
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--ref")
    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "output/algorithm_library_expansion/upstream"
    folder = root / args.repo.replace("/", "__")
    folder.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/" + args.repo
    metadata = json.loads(fetch(api))
    commit = json.loads(fetch(api + "/commits?per_page=1&sha=" + (args.ref or metadata["default_branch"])))[0]
    tree = json.loads(fetch(api + "/git/trees/" + commit["sha"] + "?recursive=1"))
    paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"]
    wanted = list(args.files)
    wanted += [p for p in paths if p.lower() in {"license", "license.md", "license.txt", "readme.md", "readme.rst", "pyproject.toml", "requirements.txt", "environment.yml", "setup.py"}]
    files = []
    for path in dict.fromkeys(wanted):
        if path not in paths:
            continue
        url = f"https://raw.githubusercontent.com/{args.repo}/{commit['sha']}/{path}"
        content = fetch(url)
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files.append({"path": path, "url": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    record = {"repository": metadata["html_url"], "ref": args.ref, "commit": commit["sha"],
              "license": metadata.get("license"), "checked_utc": datetime.now(timezone.utc).isoformat(),
              "files": files, "available_paths": paths}
    (folder / "source_evidence.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"repo": args.repo, "commit": commit["sha"], "license": metadata.get("license"),
                      "test_paths": [p for p in paths if "test" in p.lower()][:25]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
