"""Bounded EPA AirData acquisition and read-only preaudit (no algorithm changes)."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from collections import Counter

import pandas as pd


BASE = "https://aqs.epa.gov/aqsweb/airdata/"
FILES = ["aqs_sites.zip", "aqs_monitors.zip"] + [
    f"annual_conc_by_monitor_{year}.zip" for year in range(2018, 2023)
] + ["hourly_88101_2022.zip"]


def utc():
    return pd.Timestamp.now(tz="UTC").isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def acquire(root, name):
    raw = root / "original"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / name
    url = BASE + name
    started = utc()
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        result = subprocess.run([
            "curl.exe", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "15", "--max-time", "90", "--retry", "0",
            "--dump-header", str(raw / (name + ".headers.txt")),
            "--output", str(part), url,
        ], capture_output=True, text=True)
        if result.returncode:
            return {"file": name, "url": url, "status": "DOWNLOAD_FAILED", "error": result.stderr,
                    "attempt_started_utc": started}
        part.replace(path)
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError("ZIP CRC failed")
            members = z.namelist()
    except Exception as exc:
        return {"file": name, "url": url, "status": "INVALID_ZIP", "error": str(exc)}
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    os.chmod(path, 0o444)
    return {"file": name, "url": url, "status": "DOWNLOADED_VALID_ZIP", "bytes": path.stat().st_size,
            "sha256": digest, "members": members, "download_checked_utc": utc(),
            "attempt_started_utc": started, "version": name + "; live EPA AirData snapshot; Last-Modified in headers",
            "license": "US EPA public data; no explicit SPDX dataset license asserted",
            "license_source": "https://www.epa.gov/web-policies-and-procedures/epa-disclaimers",
            "raw_read_only": True}


def readzip(path, **kwargs):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            return pd.read_csv(f, low_memory=False, **kwargs)


def inspect_metadata(root):
    raw = root / "original"
    derived = root / "statistics"
    derived.mkdir(exist_ok=True)
    fields = {}
    for name in FILES[:-1]:
        path = raw / name
        if not path.exists():
            continue
        df = readzip(path)
        fields[name] = {"rows": len(df), "columns": list(df.columns)}
        if name in {"aqs_sites.zip", "aqs_monitors.zip"}:
            state = "State Code"
            ca = df[pd.to_numeric(df[state], errors="coerce").eq(6)]
            if name == "aqs_monitors.zip":
                ca = ca[pd.to_numeric(ca["Parameter Code"], errors="coerce").isin([88101, 88502])]
            ca.to_csv(derived / name.replace(".zip", "_california.csv"), index=False)
            fields[name]["california_rows"] = len(ca)
            for c in df.columns:
                if any(s in c.lower() for s in ["agency", "region", "district", "datum", "method", "duration"]):
                    fields[name][c] = ca[c].value_counts(dropna=False).head(100).to_dict()
        else:
            ca = df[pd.to_numeric(df["State Code"], errors="coerce").eq(6)
                    & pd.to_numeric(df["Parameter Code"], errors="coerce").eq(88101)]
            ca.to_csv(derived / name.replace(".zip", "_california_pm25.csv"), index=False)
        del df
    dump(derived / "metadata_schema.json", fields)


def site_id(df):
    col = "Site Number" if "Site Number" in df else "Site Num"
    return (pd.to_numeric(df["State Code"], errors="raise").astype(int).astype(str).str.zfill(2) + "-"
            + df["County Code"].astype(int).astype(str).str.zfill(3) + "-"
            + df[col].astype(int).astype(str).str.zfill(4))


def analyze(root):
    raw, out = root / "original", root / "statistics"
    out.mkdir(exist_ok=True)
    monitors = readzip(raw / "aqs_monitors.zip")
    monitors = monitors[(pd.to_numeric(monitors["State Code"], errors="coerce").eq(6))
                        & monitors["Parameter Code"].eq(88101)].copy()
    monitors["site_id"] = site_id(monitors)
    district = monitors[monitors["PQAO"].eq("South Coast Air Quality Management District")
                        & monitors["Reporting Agency"].eq("South Coast Air Quality Management District")].copy()
    district.to_csv(out / "district_monitor_identity.csv", index=False)
    ids = set(district.site_id)
    active = district[(district["First Year of Data"] <= 2022)
                      & (district["Last Sample Date"] >= "2022-01-01")]
    active.to_csv(out / "district_active_2022_monitor_candidates.csv", index=False)
    annuals, yearly, coordinate = [], [], []
    for year in range(2018, 2023):
        df = readzip(raw / f"annual_conc_by_monitor_{year}.zip")
        df = df[(pd.to_numeric(df["State Code"], errors="coerce").eq(6)) & df["Parameter Code"].eq(88101)].copy()
        df["site_id"] = site_id(df)
        df = df[df.site_id.isin(ids)].copy()
        df.to_csv(out / f"district_annual_{year}.csv", index=False)
        # Summary rows repeat durations, standards and exceptional-event variants.
        # Do not add Observation Count across these rows.
        yearly.append({"year": year, "summary_rows": len(df), "all_sites": df.site_id.nunique(),
                       "one_hour_sites": df.loc[df["Sample Duration"].eq("1 HOUR"), "site_id"].nunique(),
                       "manual_24hour_sites": df.loc[df["Sample Duration"].eq("24 HOUR"), "site_id"].nunique(),
                       "calculated_daily_block_sites": df.loc[df["Sample Duration"].eq("24-HR BLK AVG"), "site_id"].nunique(),
                       "distinct_monitors": len(df[["site_id", "POC"]].drop_duplicates()),
                       "units": sorted(df["Units of Measure"].dropna().unique()),
                       "datum": sorted(df["Datum"].dropna().unique())})
        coordinate.append(df[["site_id", "Year", "Latitude", "Longitude", "Datum"]].drop_duplicates())
        annuals.append(df)
    annual = pd.concat(annuals, ignore_index=True)
    pd.DataFrame(yearly).to_csv(out / "annual_site_coverage_2018_2022.csv", index=False)
    coordinates = pd.concat(coordinate, ignore_index=True)
    coordinates.to_csv(out / "annual_site_coordinates_2018_2022.csv", index=False)
    continuous = (annual[annual["Sample Duration"].eq("1 HOUR")].groupby("site_id")
                  .agg(years=("Year", lambda s: ";".join(map(str, sorted(set(s))))),
                       year_count=("Year", "nunique")))
    continuous.to_csv(out / "hourly_site_crossyear_presence.csv")
    summary = {"sources": {"catalog": BASE + "download_files.html", "format": BASE + "FileFormats.html",
                           "advisory": BASE + "Data_Advisory_PM25_88101_Pregen_Files.pdf"},
               "metadata_snapshot_utc": utc(), "district_membership": "Current official Monitor Listing: both PQAO and Reporting Agency exactly South Coast Air Quality Management District; not four-county approximation",
               "history_limitation": "Monitor Listing is current, not a historical agency-membership register. Annual station presence and coordinates are measured separately.",
               "active_2022_metadata_sites": active.site_id.nunique(), "active_2022_metadata_pocs": len(active),
               "active_2022_bbox": [active.Longitude.min(), active.Latitude.min(), active.Longitude.max(), active.Latitude.max()],
               "annual": yearly, "five_year_hourly_presence_sites": int(continuous.year_count.eq(5).sum()),
               "annual_coordinate_change_sites": int(coordinates.groupby("site_id").apply(lambda d: len(d[["Latitude", "Longitude", "Datum"]].drop_duplicates()) > 1, include_groups=False).sum()),
               "daily_rule": {"minimum_distinct_valid_hours": 18, "minimum_valid_days_for_qualified_annual_station": 270,
                              "unit": "Micrograms/cubic meter (LC)", "duration": "1 HOUR",
                              "exclude_original_methods": [236, 238], "negative_values": "retain raw; exclude derived physical-concentration daily means and count",
                              "qualifiers": "raw retained; strict derived sample accepts only blank qualifier; no interpretation as AQI",
                              "poc": "Select maximum valid-day coverage per site then most valid hours then lowest POC, using coverage only (not predictive performance); do not average POCs"},
               "hourly_downloaded": (raw / "hourly_88101_2022.zip").exists(),
               "qualified_continuous_sites": None, "date_intersection": None, "season_coverage": None,
               "conclusion": "BLOCKED until the hourly download and actual daily-coverage checks finish; annual rows alone are not sufficient evidence"}
    if not summary["hourly_downloaded"]:
        summary["conclusion"] = "NOT_PASSED: 2022 annual summaries show only 10 88101 1-HOUR sites in this identified network (<20 target); exact 18-hour daily coverage and seasonal date intersections remain BLOCKED by incomplete hourly download"
        dump(out / "epa_preaudit_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
        return
    chunks, global_rows, counter = [], 0, Counter()
    path = raw / "hourly_88101_2022.zip"
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            for df in pd.read_csv(f, chunksize=250000, low_memory=False):
                global_rows += len(df)
                df = df[pd.to_numeric(df["State Code"], errors="coerce").eq(6)].copy()
                df["site_id"] = site_id(df)
                df = df[df.site_id.isin(ids)]
                if len(df):
                    chunks.append(df)
    hourly = pd.concat(chunks, ignore_index=True)
    hourly.to_csv(out / "district_hourly_2022_raw_subset.csv", index=False)
    summary["hourly"] = {"national_rows_streamed": global_rows, "district_rows": len(hourly),
                         "columns": list(hourly.columns), "duration_counts": hourly["Sample Duration"].value_counts(dropna=False).to_dict(),
                         "units": hourly["Units of Measure"].value_counts(dropna=False).to_dict(),
                         "datum": hourly["Datum"].value_counts(dropna=False).to_dict(),
                         "methods": hourly["Method Code"].value_counts(dropna=False).to_dict(),
                         "qualifiers": hourly["Qualifier"].value_counts(dropna=False).to_dict(),
                         "missing_measurement": int(hourly["Sample Measurement"].isna().sum()),
                         "negative_measurement": int(hourly["Sample Measurement"].lt(0).sum()),
                         "date_range_local": [hourly["Date Local"].min(), hourly["Date Local"].max()]}
    k = ["site_id", "POC", "Date GMT", "Time GMT"]
    summary["hourly"]["duplicate_site_poc_utc_rows"] = int(hourly.duplicated(k, keep=False).sum())
    valid = hourly[(hourly["Sample Duration"].eq("1 HOUR"))
                   & hourly["Units of Measure"].eq("Micrograms/cubic meter (LC)")
                   & ~hourly["Method Code"].isin([236, 238])
                   & hourly["Sample Measurement"].notna()
                   & hourly["Sample Measurement"].ge(0)
                   & hourly["Qualifier"].fillna("").str.strip().eq("")].copy()
    # Any conflicting duplicate is rejected rather than averaged; identical duplicates collapse.
    bad = valid.groupby(k, dropna=False)["Sample Measurement"].transform("nunique").gt(1)
    summary["hourly"]["conflicting_duplicate_rows_rejected"] = int(bad.sum())
    valid = valid[~bad].drop_duplicates(k)
    offsets = (pd.to_datetime(valid["Date GMT"] + " " + valid["Time GMT"])
               - pd.to_datetime(valid["Date Local"] + " " + valid["Time Local"])).dt.total_seconds() / 3600
    summary["hourly"]["utc_minus_local_hours"] = offsets.value_counts(dropna=False).to_dict()
    daily = valid.groupby(["site_id", "POC", "Date Local"]).agg(
        valid_hours=("Time GMT", "nunique"), pm25=("Sample Measurement", "mean")).reset_index()
    daily["daily_valid"] = daily.valid_hours.ge(18)
    daily.to_csv(out / "daily_by_poc_18hour_rule.csv", index=False)
    coverage = daily.groupby(["site_id", "POC"]).agg(valid_days=("daily_valid", "sum"),
                  valid_hours=("valid_hours", "sum")).reset_index()
    coverage = coverage.sort_values(["site_id", "valid_days", "valid_hours", "POC"], ascending=[True, False, False, True])
    chosen = coverage.drop_duplicates("site_id").copy()
    chosen["qualified_270days"] = chosen.valid_days.ge(270)
    chosen.to_csv(out / "primary_poc_station_coverage.csv", index=False)
    selected = daily.merge(chosen[["site_id", "POC", "qualified_270days"]], on=["site_id", "POC"])
    selected = selected[selected.daily_valid]
    selected.to_csv(out / "selected_continuous_daily.csv", index=False)
    days = pd.DataFrame({"Date Local": pd.date_range("2022-01-01", "2022-12-31").strftime("%Y-%m-%d")})
    allcounts = selected.groupby("Date Local").site_id.nunique().rename("all_valid_sites")
    qcounts = selected[selected.qualified_270days].groupby("Date Local").site_id.nunique().rename("qualified_valid_sites")
    days = days.join(allcounts, on="Date Local").join(qcounts, on="Date Local").fillna(0)
    days.to_csv(out / "daily_common_station_counts.csv", index=False)
    seasons = []
    for month in [1, 4, 7, 10]:
        dates = pd.date_range(f"2022-{month:02d}-15", periods=7).strftime("%Y-%m-%d")
        s = selected[selected["Date Local"].isin(dates)]
        freq = s.groupby("site_id")["Date Local"].nunique()
        seasons.append({"start": dates[0], "end": dates[-1], "station_days": len(s),
                        "stations_any_day": s.site_id.nunique(), "stations_all_7_days": int(freq.eq(7).sum()),
                        "qualified_station_days": int(s.qualified_270days.sum()),
                        "qualified_stations_all_7_days": int(s[s.qualified_270days].groupby("site_id")["Date Local"].nunique().eq(7).sum())})
    summary["season_coverage"] = seasons
    summary["qualified_continuous_sites"] = int(chosen.qualified_270days.sum())
    summary["any_valid_daily_sites"] = chosen.loc[chosen.valid_days.gt(0), "site_id"].nunique()
    summary["date_intersection"] = {"all_days": 365, "minimum_qualified_sites": int(days.qualified_valid_sites.min()),
                                    "median_qualified_sites": float(days.qualified_valid_sites.median()),
                                    "maximum_qualified_sites": int(days.qualified_valid_sites.max()),
                                    "days_with_at_least20": int(days.qualified_valid_sites.ge(20).sum())}
    summary["conclusion"] = ("PASS data-size threshold only; geometry/splits require separate protocol" if summary["qualified_continuous_sites"] >= 20
                              else "NOT_PASSED >=20 qualified continuous stations; do not merge manual 24h samples merely to meet threshold")
    dump(out / "epa_preaudit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)


def finalize(root):
    manifest_path = root / "source_manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in rows:
        if row["file"] in {"aqs_sites.zip", "hourly_88101_2022.zip"} and row["status"] == "DOWNLOAD_FAILED":
            row["error"] = "First GET transferred only a truncated archive after prolonged stall and was stopped; one HTTP/1.1 official-URL cache-query retry reached its 90-second bound. No valid ZIP available. No further attempts in this task."
            retry = root / "original" / row["file"].replace(".zip", "_retry.zip.part")
            original = root / "original" / (row["file"] + ".part")
            row["retry_url"] = row["url"] + "?preaudit=20261018"
            row["retry_status"] = "TIMEOUT_90_SECONDS"
            row["incomplete_artifacts"] = [{"file": p.name, "bytes": p.stat().st_size,
                                           "status": "INCOMPLETE_NOT_DATASET", "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()}
                                          for p in [original, retry] if p.exists()]
            row["download_checked_utc"] = utc()
            row["raw_read_only"] = True
        elif row["status"] == "DOWNLOADED_VALID_ZIP":
            header_path = root / "original" / (row["file"] + ".headers.txt")
            if header_path.exists():
                row["http_headers"] = header_path.read_text(encoding="utf-8")
    dump(manifest_path, rows)
    for p in (root / "original").iterdir():
        if p.is_file():
            os.chmod(p, 0o444)
    evidence = {
        "format_url": BASE + "FileFormats.html",
        "agency_field_evidence": "Monitor Description fields Reporting Agency and PQAO identify responsible reporting and primary quality assurance organizations; not a polygon boundary nor a historical agency-membership register.",
        "coordinate_evidence": "Site/Monitor/Annual Datum columns explicitly report the coordinate datum; this audit preserves NAD83 and WGS84 without silently treating them as identical.",
        "time_evidence": "AirData local times and daily summaries use local STANDARD time, not daylight saving; hourly CSV also carries Date GMT and Time GMT. No actual local-to-UTC check can be completed without the downloaded hourly data.",
        "poc_evidence": "POC distinguishes monitors for the same parameter at the same site. Repeated annual rows additionally reflect standards, durations and exceptional events; not independent sites.",
        "advisory_url": BASE + "Data_Advisory_PM25_88101_Pregen_Files.pdf",
        "advisory_posted": "2024-08-26",
        "advisory_measured_scope": "T640/T640X measurements from 2017 onward may include original 236/238 and corrected 736/738 or updated FEM 636/638. Exclude original 236/238; preserve corrected/updated and record methods. Multiple Methods Used annual labels are not enough to audit per-record methods.",
        "license_url": "https://www.epa.gov/web-policies-and-procedures/epa-disclaimers",
        "license_limit": "US EPA openly distributes these files. No SPDX license or blanket copyright waiver for every third-party-submitted datum is invented.",
        "manual_sampling_limitation": "24 HOUR records are physical 24-hour samples and are separate from 24-HR BLK AVG computed from continuous measurements; actual scheduled cadence and dates cannot be derived uniquely from annual counts.",
        "remaining_actions": ["Obtain the same official hourly ZIP in a later authorized acquisition attempt; no new download performed here", "Decide whether to expand the official geographical network or conduct a smaller spatial-method pilot; do not mix 24h manual and 1h continuous merely to reach 20", "If including parameter 88502 to increase continuous sites, first confirm measurement definition and QA; not downloaded/merged in this task"]}
    dump(root / "statistics" / "official_field_evidence.json", evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["download", "metadata", "analyze", "finalize"], default="download")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == "download":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(lambda name: acquire(args.root, name), FILES))
        dump(args.root / "source_manifest.json", rows)
        print(json.dumps(rows, ensure_ascii=False, default=str), flush=True)
    elif args.stage == "metadata":
        inspect_metadata(args.root)
    elif args.stage == "analyze":
        analyze(args.root)
    else:
        finalize(args.root)


if __name__ == "__main__":
    main()
