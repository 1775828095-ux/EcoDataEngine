"""Official AQS county/month client. Explicit fetch; never logs API credentials."""
import argparse
import calendar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--county", choices=["037", "059", "065", "071"], default="037")
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    if args.year not in range(2018, 2023) or args.month not in range(1, 13):
        raise ValueError("Only the requested 2018-2022 scope is allowed")
    args.root.mkdir(parents=True, exist_ok=True)
    credentials = {}
    config = Path.home() / ".aqs.json"
    if config.exists():
        credentials = json.loads(config.read_text(encoding="utf-8"))
    email = os.environ.get("AQS_EMAIL") or os.environ.get("EPA_AQS_EMAIL") or credentials.get("email")
    key = os.environ.get("AQS_KEY") or os.environ.get("AQS_API_KEY") or os.environ.get("EPA_AQS_KEY") or credentials.get("key")
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "configured": bool(email and key),
        "api_requests": 0, "secret_values_logged": False, "state": "06", "county": args.county,
        "year": args.year, "month": args.month, "parameter": "88101", "duration": "1",
        "official_docs": "https://aqs.epa.gov/aqsweb/documents/data_api.html",
        "registration": "https://aqs.epa.gov/data/api/signup", "registration_performed": False}
    if not email or not key or key == "test":
        report["status"] = "BLOCKED_NEEDS_PERSONAL_AQS_EMAIL_AND_KEY"
    elif not args.fetch:
        report["status"] = "CONFIGURED_NOT_FETCHED"
    else:
        endpoint = "https://aqs.epa.gov/data/api/sampleData/byCounty"
        name = f"aqs_88101_06_{args.county}_{args.year}{args.month:02d}.json"
        path = args.root / name
        if path.exists():
            report.update(status="REUSED_EXISTING_FILE", file=name)
        else:
            query = {"email": email, "key": key, "param": "88101", "state": "06", "county": args.county,
                "bdate": f"{args.year}{args.month:02d}01",
                "edate": f"{args.year}{args.month:02d}{calendar.monthrange(args.year, args.month)[1]}", "duration": "1"}
            report["api_requests"] = 1
            try:
                response = requests.get(endpoint, params=query, timeout=(10, 60))
                report["http_status"] = response.status_code
                response.raise_for_status()
                payload = response.json()
                headers = payload.get("Header", [])
                if not headers or headers[0].get("status") not in ("Success", "No data found"):
                    raise ValueError("AQS did not return a successful result")
                # Observation fields remain unchanged. API URLs/messages in the
                # response header can echo credentials; retain only safe fields.
                clean = {"Header": [{k: h[k] for k in ("status", "rows") if k in h} for h in headers],
                         "Data": payload.get("Data", [])}
                path.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
                path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
                report.update(status="DOWNLOADED_OBSERVATIONS", file=name, rows=len(clean["Data"]),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_endpoint=endpoint,
                    original_response_sha256=hashlib.sha256(response.content).hexdigest(),
                    header_credentials_removed=True, raw_observation_fields_modified=False)
            except Exception as exc:
                # requests exception text may include the full authenticated URL.
                report.update(status="BLOCKED_REQUEST_FAILED", error_type=type(exc).__name__)
    (args.root / "access_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
