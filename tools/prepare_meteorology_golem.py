"""Stream the six specified workbooks without changing raw files or inferring units."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from lxml import etree

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def finite(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def extract(job):
    path, output = map(Path, job)
    year = int(path.name[:4])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    z = zipfile.ZipFile(path)
    strings = ["".join(item.itertext()) for item in etree.fromstring(z.read("xl/sharedStrings.xml"))]
    header, records = None, []
    count, years, per_station, times = 0, Counter(), defaultdict(lambda: [0, set()]), set()
    valid, missing = Counter(), Counter()
    limits = {"rh": [float("inf"), float("-inf")], "pressure": [float("inf"), float("-inf")]}
    for _, row in etree.iterparse(z.open("xl/worksheets/sheet1.xml"), events=("end",), tag=NS + "row"):
        values = {}
        for cell in row:
            key = "".join(c for c in cell.attrib["r"] if c.isalpha())
            node = cell.find(NS + "v")
            value = node.text if node is not None else None
            if cell.attrib.get("t") == "s" and value is not None:
                value = strings[int(value)]
            elif cell.attrib.get("t") == "inlineStr":
                value = "".join(cell.itertext())
            values[key] = value
        if header is None:
            header = values
            cols = {value: key for key, value in values.items()}
            required = ["站点编号", "经度", "纬度", "相对湿度24小时均值", "海平面气压24小时均值", "创建时间"]
            if any(name not in cols for name in required):
                raise ValueError(f"Unsupported workbook schema: {path}, {list(cols)}")
        else:
            count += 1
            get = lambda name: values.get(cols[name])
            serial = finite(get("创建时间"))
            if serial is not None:
                date = datetime(1899, 12, 30) + timedelta(days=serial)
            else:
                date = datetime.fromisoformat(get("创建时间"))
            years[date.year] += 1
            times.add(date.isoformat())
            station = str(get("站点编号"))
            x, y = finite(get("经度")), finite(get("纬度"))
            per_station[station][0] += 1
            per_station[station][1].add((x, y))
            rh, pressure = finite(get("相对湿度24小时均值")), finite(get("海平面气压24小时均值"))
            for var, value in (("rh", rh), ("pressure", pressure)):
                if value is None:
                    missing[var] += 1
                else:
                    valid[var] += 1
                    limits[var][0] = min(limits[var][0], value)
                    limits[var][1] = max(limits[var][1], value)
            # Fixed numerical rectangle, not a claim about the source datum.
            if x is not None and y is not None and 114 <= x <= 119 and 36 <= y <= 41:
                records.append((station, date, x, y, rh, pressure))
        row.clear()
        while row.getprevious() is not None:
            del row.getparent()[0]
    frame = pd.DataFrame(records, columns=["station", "time", "x_as_stored", "y_as_stored", "rh", "pressure"])
    frame.to_parquet(output / f"region_{year}.parquet", index=False)
    limits = {name: [value if math.isfinite(value) else None for value in pair] for name, pair in limits.items()}
    audit = {"source": str(path), "sha256": digest, "bytes": path.stat().st_size,
             "rows": count, "years_present": dict(years), "header": list(header.values()),
             "time_start": min(times), "time_end": max(times), "unique_times": len(times),
             "valid": dict(valid), "missing": dict(missing), "numeric_range": limits,
             "stations": {sid: {"rows": value[0], "coordinate_pairs": list(value[1])} for sid, value in per_station.items()},
             "unit": "UNDECLARED", "crs": "UNDECLARED", "selected_rectangle_rows": len(frame)}
    (output / f"audit_{year}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("source", "rows", "years_present", "unique_times", "selected_rectangle_rows", "numeric_range")}, ensure_ascii=False), flush=True)
    return year


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.data / f"{year}年.xlsx"), str(args.output)) for year in range(2017, 2023)
            if not (args.output / f"audit_{year}.json").exists()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(extract, jobs))
    # Preserve standards-compliant null for a completely missing source column.
    for year in range(2017, 2023):
        path = args.output / f"audit_{year}.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["numeric_range"] = {name: [value if value is not None and math.isfinite(value) else None for value in pair]
                                  for name, pair in audit["numeric_range"].items()}
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    frames = [pd.read_parquet(args.output / f"region_{year}.parquet") for year in range(2017, 2023)]
    table = pd.concat(frames, ignore_index=True)
    table["time"] = pd.to_datetime(table.time)
    table.to_parquet(args.output / "regional_six_years.parquet", index=False)
    print(table.groupby(table.time.dt.year).agg(rows=("station", "size"), stations=("station", "nunique"),
              rh_count=("rh", "count"), pressure_count=("pressure", "count")).to_string(), flush=True)


if __name__ == "__main__":
    main()
