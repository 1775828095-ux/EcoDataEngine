"""Local-only sampling and exposure records for a desktop run."""


def field_key(index, field):
    return index, field["name"], field.get("sheet"), field.get("band")


def source_record(scan, index, field):
    from .sampling_plan import sample

    records = scan.setdefault("source_sampling_records", {})
    key = field_key(index, field)
    if key not in records:
        records[key] = sample(scan["rows"][index], field)
    return records[key]


def export_field(scan, index, field, *, phase, limit=None):
    from .sampling_plan import export_field as export

    row = scan["rows"][index]
    if not scan.get("on_demand"):
        return export(row, field)
    record = source_record(scan, index, field)
    payload = export(row, field, require_partition=True, record=record)
    if limit is not None:
        payload["perturbed_samples"] = payload["perturbed_samples"][:limit]
    if payload["perturbed_samples"]:
        record_exposure(scan, index, field, phase, record)
    return payload


def record_exposure(scan, index, field, phase, record):
    """Conservatively record every window or group contributing to this pool."""
    events = scan.setdefault("numeric_exposure", {})
    events.setdefault(field_key(index, field), {})[phase] = {
        "basis": record["basis"], "partition": record.get("partition"),
        "groups": tuple(record.get("sampled_group_keys", ())),
        "windows": tuple({key: window[key] for key in ("axis", "start_index", "stop_index",
                                                      "row", "column", "width", "height") if key in window}
                         for window in record.get("windows", ())),
    }


def exposures(scan, index, field):
    return scan.get("numeric_exposure", {}).get(field_key(index, field), {})


def snapshot(scan):
    return [dict(index=index, field=name, sheet=sheet, band=band, exposures=events)
            for (index, name, sheet, band), events in scan.get("numeric_exposure", {}).items()]


def table_groups(source, exposures):
    """Remove groups touching any time slice or group exported for this field."""
    held = set(source.held)
    for exposure in exposures.values():
        basis = exposure["basis"]
        if basis == "time_windows":
            boundary = time_boundary({"exposure": exposure}, exposure["partition"]["axis"])
            held = {key for key in held if isinstance(source.groups[key], slice)
                    and source.groups[key].start >= boundary}
        elif (source.category, basis) in {("station", "station_groups"), ("month", "calendar_months")}:
            held.difference_update(exposure["groups"])
        else:
            return set()
    return held


def time_boundary(events, axis):
    boundary = 0
    for exposure in events.values():
        partition = exposure.get("partition") or {}
        if exposure["basis"] != "time_windows" or partition.get("axis") != axis:
            raise ValueError("外发范围与时间回检轴不一致，无法证明两者独立")
        boundary = max(boundary, partition["holdout_start"],
                       max((window["stop_index"] for window in exposure["windows"]), default=0))
    return boundary


def freeze(scan, gated, tasks):
    from ecodata.core.exceptions import EcoDataError
    from . import independent_alignment, indexed_table_recipe, station_grid, windowed_grid_recipe

    adapters = {"independent_grid_search": independent_alignment.freeze_holdout,
                "indexed_table_recipe": indexed_table_recipe.freeze_holdout,
                "station_grid": station_grid.freeze_holdout,
                "windowed_grid_recipe": windowed_grid_recipe.freeze_holdout}
    for candidate in tasks:
        key = candidate["candidate"], candidate["operation"]
        task = gated.get(key)
        resolver = adapters.get((task or {}).get("route"))
        if resolver is None or "locked_holdout" in task or task.get("holdout_lock_error"):
            continue
        try:
            task["locked_holdout"] = resolver(scan, task)
        except (EcoDataError, OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, ImportError) as error:
            task["holdout_lock_error"] = str(error)
