"""Prepare a shared, local-only holdout for inspected daily precipitation products."""
from contextlib import ExitStack
from pathlib import Path
import re
from ecodata.core.temporary import TemporaryDirectory
from .scan import assert_current, local_member, scientific

CMORPH_DAY = re.compile(r"(CMORPH.*[_-])\d{8}\.nc\Z", re.IGNORECASE)


def shard_family(name):
    path = Path(name)
    match = CMORPH_DAY.fullmatch(path.name)
    return (str(path.parent), match[1].casefold()) if match else None


def prepare(scan):
    import pandas as pd
    import xarray as xr
    from ecodata.models.engineering.precipitation_products import inspect_product

    rows = scan["rows"]
    groups = {}
    for index, row in enumerate(rows):
        if row["format"] == "NetCDF" and (key := shard_family(row["name"])):
            groups.setdefault(key, []).append(index)
    power = [i for i, row in enumerate(rows) if row["format"] == "NetCDF"
             and any(field["name"] == "PRECTOTCORR" for field in row["fields"])]
    if len(groups) != 1 or len(power) != 1:
        return
    siblings, = groups.values()
    if sum(rows[i]["bytes"] for i in siblings) > 24 * 1024**2:
        return
    assert_current(scan)
    with ExitStack() as stack:
        pieces, audits = [], []
        for index in siblings:
            path = stack.enter_context(local_member(rows[index]))
            dataset = stack.enter_context(xr.open_dataset(path, cache=False))
            product = inspect_product(dataset)
            if not product or product["product"] != "NOAA_CMORPH" or len(product["fields"]) != 1:
                return
            audit, = product["fields"]
            if audit["blockers"] or audit["time_basis"] != "UTC":
                return
            pieces.append(dataset)
            audits.append((product["source_id"], audit["field"], audit["unit"]))
        if len(set(audits)) != 1:
            return
        merged = xr.concat(pieces, dim="time", data_vars="minimal", coords="minimal", compat="equals",
                           join="exact").sortby("time")
        dates = pd.DatetimeIndex(merged.time.values)
        if dates.has_duplicates:
            raise ValueError("CMORPH 同产品分片存在重复 UTC 日期")
        other_path = stack.enter_context(local_member(rows[power[0]]))
        with xr.open_dataset(other_path, cache=False) as other:
            other_dates = pd.DatetimeIndex(other.time.values)
        common = dates.intersection(other_dates).sort_values()
        if len(common) < 2:
            return
        held = common[max(1, len(common) - max(1, len(common) // 5)):]
        cutoff = held[0].date().isoformat()
        selected_days = [date.date().isoformat() for date in held]
        folder = TemporaryDirectory(prefix="ecodata-precipitation-")
        try:
            path = Path(folder.name) / "CMORPH_daily.nc"
            bound = merged.time.attrs["bounds"]
            merged.attrs.update(time_coverage_start=dates[0].strftime("%Y-%m-%dT00:00:00Z"),
                                time_coverage_end=dates[-1].strftime("%Y-%m-%dT23:59:59Z"))
            for clock in (merged.time, merged[bound]):
                clock.encoding.update(units="seconds since 1970-01-01", dtype="float64")
            merged.to_netcdf(path)
            first = rows[siblings[0]]
            combined = dict(first, ref=(path, "", None, None), containers=[], bytes=path.stat().st_size, fields=[])
            scientific(combined, path)
            if not combined["fields"] or combined["bytes"] + rows[power[0]]["bytes"] > 32 * 1024**2:
                folder.cleanup()
                return
            assert_current(scan)
        except Exception:
            folder.cleanup()
            raise
    # Keep the original member name for references, and record the members used.
    combined.update(recommendation_cutoff=cutoff, precipitation_holdout_days=selected_days,
                    precipitation_members=[rows[i]["name"] for i in siblings])
    rows[siblings[0]] = combined
    rows[power[0]].update(recommendation_cutoff=cutoff, precipitation_holdout_days=selected_days)
    scan["rows"] = [row for i, row in enumerate(rows) if i not in siblings[1:]]
    scan["precipitation_workspace"] = folder
