"""Allocate a bounded numeric evidence round across approved sources and variables."""
from collections import deque


def strata(scan, originals):
    result = {}
    for fid, (index, _, field) in originals.items():
        row = scan["rows"][index]
        if row["format"] != "NetCDF" or field.get("axis"):
            continue
        meaning = field.get("standard_name") or field.get("physical_variable") or field["name"]
        result[fid] = dict(source=f"D{index + 1}", variable=meaning, unit=field.get("unit"),
                           declared_frequency=row.get("declared_time_resolution"))
    return result


def choose(requested, descriptions, limit):
    """Round-robin meaning/cadence and files, without introducing unrequested fields."""
    grouped = {}
    for fid in requested:
        info = descriptions.get(fid)
        if info:
            key = tuple(str(info[name] or "") for name in ("variable", "unit", "declared_frequency"))
            grouped.setdefault(key, {}).setdefault(info["source"], deque()).append(fid)
        else:
            grouped[(fid,)] = {fid: deque([fid])}
    queues = deque(deque(sources.values()) for sources in grouped.values())
    chosen = []
    while queues and len(chosen) < limit:
        files = queues.popleft()
        fields = files.popleft()
        chosen.append(fields.popleft())
        if fields:
            files.append(fields)
        if files:
            queues.append(files)
    return chosen
