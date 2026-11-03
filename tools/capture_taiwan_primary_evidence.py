"""Save bounded primary-source checks without credentials or invented metadata."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/taiwan_final_20261019"
SOURCES = {
    "zenodo_record": "https://zenodo.org/api/records/4420335",
    "publisher_paper": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EA001599",
    "noaa_paper_record": "https://repository.library.noaa.gov/view/noaa/41813",
    "official_station_schema": "https://data.moenv.gov.tw/dataset/detail/AQX_P_07",
    "official_station_catalog": "https://data.gov.tw/dataset/6075",
    "archive_preview": "https://zenodo.org/records/4420335/preview/aod_pm25_data.tar.gz?include_deleted=0",
}


def main():
    folder = OUT / "primary_evidence"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        row = dict(name=name, url=url, accessed_utc=datetime.now(timezone.utc).isoformat())
        try:
            response = requests.get(url, timeout=(15, 25))
            row.update(http_status=response.status_code, final_url=response.url)
            response.raise_for_status()
            if "json" in response.headers.get("Content-Type", ""):
                text = json.dumps(response.json(), ensure_ascii=False, indent=2)
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                for tag in soup(["script", "style", "input", "form"]):
                    tag.decompose()
                text = soup.get_text("\n", strip=True)
            path = folder / (name + ".txt")
            path.write_text(text, encoding="utf-8")
            row.update(status="READ", file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as error:
            row.update(status="FAILED", error=repr(error))
        records.append(row)
        (OUT / "primary_source_access.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, row["status"], flush=True)


if __name__ == "__main__":
    main()
