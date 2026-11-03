"""Versioned user conventions, explicitly not proof of author metadata."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/taiwan_2017_working_v1.json"
CONTRACT = json.loads(CONFIG.read_text(encoding="utf-8"))
DATA = ROOT / "data/taiwan_2017_zenodo4420335"
OUT = ROOT / "output/taiwan_annual_working_v1"
PROJECTION = Transformer.from_crs("EPSG:4326", CONTRACT["space"]["metric_crs"], always_xy=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def timestamps(values):
    text = pd.Series(values, dtype="string")
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    result = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
    naive = pd.to_datetime(text.loc[~aware], errors="coerce", format="mixed")
    result.loc[~aware] = naive.dt.tz_localize("Asia/Taipei").dt.tz_convert("UTC")
    result.loc[aware] = pd.to_datetime(text.loc[aware], errors="coerce", format="mixed", utc=True)
    return result


def scan_hour(filename, attrs):
    start = attrs.get("time_coverage_start") or attrs.get("observation_start_time")
    end = attrs.get("time_coverage_end") or attrs.get("observation_end_time")
    if start:
        start = start.decode() if isinstance(start, bytes) else str(start)
        time = timestamps([start]).iloc[0]
        origin = "OBSERVATION_METADATA"
        if end:
            end = end.decode() if isinstance(end, bytes) else str(end)
            finish = timestamps([end]).iloc[0]
            if pd.isna(finish) or finish <= time:
                raise ValueError("Invalid declared observation interval")
            if (finish-pd.Timedelta(nanoseconds=1)).floor("h") != time.floor("h"):
                raise ValueError("CROSS_HOUR_COMPOSITE_REQUIRES_INTERVAL_MODEL")
    else:
        label = pd.to_datetime(Path(filename).stem.replace("results_Taiwan_", ""), format="%Y%m%d_%H%M")
        time = label.tz_localize("Asia/Taipei").tz_convert("UTC")
        origin = "CUSTOM_FILENAME_WORKING_ASSUMPTION"
    if pd.isna(time):
        raise ValueError("Unparseable observation timestamp")
    return time, time.floor("h"), origin


def aod_values(raw, qa, attrs):
    raw = np.asarray(raw)
    low, high = attrs.get("valid_range", CONTRACT["satellite"]["valid_range"])
    valid = np.isfinite(raw) & (raw >= low) & (raw <= high) & np.isin(qa, CONTRACT["satellite"]["qa_accept"])
    for key in ["_FillValue", "missing_value"]:
        if key in attrs:
            valid &= ~np.isin(raw, np.atleast_1d(attrs[key]))
    values = raw.astype(float)*attrs.get("scale_factor", 1)+attrs.get("add_offset", 0)
    return values, valid & np.isfinite(values)


def sites():
    source = ROOT / CONTRACT["split"]["source"]
    assert sha(source) == CONTRACT["split"]["sha256"]
    frame = pd.read_csv(source)
    original = pd.read_csv(DATA / "provisional_members/zhang_data/pm25_taiwan/StationInfo.csv")
    frame = frame.merge(original, left_on="station_id", right_on="SiteEngName", validate="one_to_one")
    assert np.isfinite(frame[["Lon", "Lat"]]).all().all()
    if not ((frame.Lon.abs() <= 180) & (frame.Lat.abs() <= 90)).all():
        raise ValueError("Coordinates not geographic: explicit input CRS required")
    assert not frame.duplicated(["Lon", "Lat"]).any()
    frame["easting"], frame["northing"] = PROJECTION.transform(frame.Lon.to_numpy(), frame.Lat.to_numpy())
    return frame


def validate_ground_intervals(frame):
    if frame.loc[frame.time.notna()].duplicated(["station_id", "time"]).any():
        raise ValueError("Duplicate station/time interval")
    if not (frame.time.dropna() == frame.time.dropna().dt.floor("h")).all():
        raise ValueError("Ground hourly label is not aligned to hour")


def frozen_selection(folder=OUT):
    receipt = json.loads((folder/"selection_freeze_receipt.json").read_text())
    file = folder/"selection_frozen.json"
    if receipt["sha256"] != sha(file):
        raise ValueError("Frozen selection was modified; do not unseal test labels")
    selected = json.loads(file.read_text())
    if selected["contract_sha256"] != sha(CONFIG) or selected["final_test_targets_used"] is not False:
        raise ValueError("Selection contract or leakage gate failed")
    return selected


def ground(include_test=False):
    frame = pd.read_csv(DATA / "provisional_members/zhang_data/pm25_taiwan/Taiwan_PM_2017.csv", dtype=str)
    frame["source_row"] = np.arange(len(frame))+2
    frame = frame.merge(sites(), left_on="stn_eng", right_on="station_id", validate="many_to_one")
    frame = frame.loc[frame.split.eq("test") if include_test else ~frame.split.eq("test")].copy()
    frame["raw_DT"] = frame.DT
    frame["raw_value"] = frame["PM2.5"]
    frame["time"] = timestamps(frame.DT)
    frame["value"] = pd.to_numeric(frame["PM2.5"], errors="coerce")
    frame["reason"] = np.select([frame.time.isna(), ~np.isfinite(frame.value), frame.value.lt(0)],
        ["INVALID_TIME", "MISSING_OR_NONFINITE_TARGET", "NEGATIVE_TARGET"], default="VALID")
    validate_ground_intervals(frame)
    frame["local_hour"] = frame.time.dt.tz_convert("Asia/Taipei")
    frame["month"] = frame.local_hour.dt.month
    frame["unit"] = CONTRACT["ground"]["unit"]
    frame["instrument_qa"] = "ABSENT_IN_AUTHOR_TABLE"
    frame["source_file"] = "zhang_data/pm25_taiwan/Taiwan_PM_2017.csv"
    return frame
