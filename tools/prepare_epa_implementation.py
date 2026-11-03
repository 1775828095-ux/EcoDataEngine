"""Bounded official-source acquisition; no private credentials in artefacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile
import requests
import pandas as pd

BASE = Path("D:/codex/数据/公开数据三层预审_20261018")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def acquire(root, name, url, rows):
    path = root / "original" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"file": str(path.relative_to(root)), "url": url,
           "checked_utc": datetime.now(timezone.utc).isoformat(), "attempt_limit": 1}
    try:
        if not path.exists():
            response = requests.get(url, timeout=(10, 45))
            response.raise_for_status()
            if name.endswith(".zip"):
                import io
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    if archive.testzip():
                        raise ValueError("ZIP CRC failed")
            else:
                obj = response.json()
                if "error" in obj:
                    raise ValueError("Official service returned an error")
            path.write_bytes(response.content)
            row["last_modified"] = response.headers.get("Last-Modified")
        row.update(status="VALID_FILE", bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
        row["license"] = "Official public EPA/CARB/Census data; provider terms apply, no invented SPDX licence"
    except Exception as exc:
        # No authenticated request is made here, so URLs contain no secrets.
        row.update(status="BLOCKED_DOWNLOAD", error=type(exc).__name__ + ": " + str(exc))
    rows.append(row)
    save(root / "source_manifest.json", rows)
    return path if row["status"] == "VALID_FILE" else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=BASE / "05_实验落地_20261019")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    credential_keys = ("AQS_EMAIL", "AQS_KEY", "AQS_API_KEY", "EPA_AQS_EMAIL", "EPA_AQS_KEY",
                       "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN", "CDSAPI_KEY")
    save(root / "access_status.json", {"checked_utc": datetime.now(timezone.utc).isoformat(),
        "environment_presence_only": {k: bool(os.environ.get(k)) for k in credential_keys},
        "file_presence_only": {name: (Path.home() / name).is_file()
                               for name in (".aqs.json", ".aqsapi", ".netrc", "_netrc", ".cdsapirc")},
        "protected_requests_made": 0, "secret_values_logged": False,
        "aqs_registration": "https://aqs.epa.gov/aqsweb/documents/data_api.html#signup",
        "stop_rule": "No configured credentials: do not call protected data endpoints or public test accounts"})
    rows = []
    service = "https://gis.carb.arb.ca.gov/hosting/rest/services/California_Air_Resources_Board_Air_Basin_MIL1/MapServer/0"
    acquire(root, "carb_layer.json", service + "?f=pjson", rows)
    basin = acquire(root, "carb_basins.geojson", service + "/query?where=1%3D1&outFields=*&outSR=4326&returnGeometry=true&f=geojson", rows)
    counties = acquire(root, "four_counties.geojson",
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query?"
        "where=STATE%3D%2706%27%20AND%20COUNTY%20IN%20%28%27037%27%2C%27059%27%2C%27065%27%2C%27071%27%29"
        "&outFields=STATE%2CCOUNTY%2CNAME%2CGEOID&outSR=4326&returnGeometry=true&f=geojson", rows)
    # Existing five-year official summaries: all agencies, no PQAO restriction.
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    basin_geometry = None
    if basin:
        features = json.loads(basin.read_text())["features"]
        selected = [f for f in features if any("south coast" in str(v).lower() for v in f["properties"].values())]
        if len(selected) == 1:
            basin_geometry = shape(selected[0]["geometry"])
    county_geometry = None
    if counties:
        features = json.loads(counties.read_text())["features"]
        if len(features) != 4:
            raise ValueError("Official county query did not return the four requested polygons")
        county_geometry = unary_union([shape(f["geometry"]) for f in features])
    summaries, all_stations = [], []
    for year in range(2018, 2023):
        table = pd.read_csv(BASE / f"02_EPA_PM25/statistics/annual_conc_by_monitor_{year}_california_pm25.csv", low_memory=False)
        table["site_id"] = table["State Code"].astype(str).str.zfill(2)+"-"+table["County Code"].astype(str).str.zfill(3)+"-"+table["Site Num"].astype(str).str.zfill(4)
        # Datum must be explicitly converted before official polygon membership.
        from pyproj import Transformer
        converted = {}
        for datum, group in table.groupby("Datum"):
            if datum not in ("WGS84", "NAD83"):
                continue
            transformer = Transformer.from_crs("EPSG:4326" if datum == "WGS84" else "EPSG:4269", "EPSG:4326", always_xy=True)
            xx, yy = transformer.transform(group.Longitude.to_numpy(), group.Latitude.to_numpy())
            converted.update({idx: (x, y) for idx, x, y in zip(group.index, xx, yy)})
        for name, geometry in (("South_Coast_Air_Basin", basin_geometry), ("Four_Counties", county_geometry)):
            if geometry is None:
                summaries.append({"year": year, "region": name, "status": "BLOCKED_BOUNDARY", "hourly_sites": None})
                continue
            mask = [idx in converted and geometry.covers(Point(*converted[idx])) for idx in table.index]
            subset = table.loc[mask].copy()
            subset.to_csv(root / f"{name}_{year}_annual.csv", index=False)
            hourly = subset[subset["Sample Duration"].eq("1 HOUR")]
            summaries.append({"year": year, "region": name, "status": "ANNUAL_METADATA_ONLY",
                "sites": subset.site_id.nunique(), "hourly_sites": hourly.site_id.nunique(),
                "manual_sites": subset[subset["Sample Duration"].eq("24 HOUR")].site_id.nunique(),
                "qualified_daily_sites": None})
            all_stations.append(hourly.assign(region=name, audit_year=year))
    pd.DataFrame(summaries).to_csv(root / "all_agency_region_coverage.csv", index=False)
    if all_stations:
        pd.concat(all_stations).to_csv(root / "all_agency_hourly_metadata.csv", index=False)
    save(root / "region_status.json", {"basin_boundary_available": basin_geometry is not None,
         "county_boundary_available": county_geometry is not None, "coverage": summaries,
         "note": "Annual metadata counts are an upper bound, not >=18-hour daily qualification"})
    # A small official daily summary is a fallback smoke-test source, NOT raw hours.
    acquire(root, "daily_88101_2022.zip", "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2022.zip", rows)
    print(json.dumps({"sources": [{"file": r["file"], "status": r["status"]} for r in rows],
                      "coverage": summaries}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
