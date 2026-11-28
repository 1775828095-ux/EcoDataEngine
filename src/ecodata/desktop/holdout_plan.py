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
        return export(row, field, limit=limit)
    key = field_key(index, field)
    if key in scan.get("station_reserved_fields", ()) or (phase == "field_probe" and station_probe_deferred(scan, index, field)):
        return dict(status="unavailable", basis="station_review_reserved", perturbed_samples=[])
    record = source_record(scan, index, field)
    payload = export(row, field, require_partition=True, record=record, limit=limit)
    if payload["perturbed_samples"]:
        record_exposure(scan, index, field, phase, record)
    return payload


def station_probe_deferred(scan, index, field):
    """Delay numeric evidence for possible station/grid pairs until the split."""
    from .discovery import field_identity, station_wind_components
    from .station_grid import eligible

    rows = scan["rows"]
    current = rows[index]
    quantity, _ = field_identity(current, field)

    def matches(other, variable):
        other_quantity, _ = field_identity(other, variable)
        return (not quantity or not other_quantity or quantity == other_quantity
                or station_wind_components(current, field, other, variable)
                or station_wind_components(other, variable, current, field))

    if current["format"] == "NetCDF":
        return any(eligible(current, field, row) and any(not variable.get("axis") and matches(row, variable)
                                                        for variable in row["fields"])
                   for row in rows if row is not current)
    return any(eligible(row, variable, current) and matches(row, variable)
               for row in rows if row["format"] == "NetCDF" for variable in row.get("fields", ()))


def reserve_station_fields(scan, gated, tasks):
    from .station_grid import background_fields

    reserved = scan.setdefault("station_reserved_fields", set())
    for candidate in tasks:
        task = gated.get((candidate["candidate"], candidate["operation"]), {})
        if task.get("route") != "station_grid":
            continue
        reserved.update(field_key(task["grid_index"], field) for field in background_fields(scan, task))
        index = task["table_index"]
        fields = scan["rows"][index]["fields"]
        field = next(field for field in fields if field["name"] == task["table_name"])
        reserved.add(field_key(index, field))
        if task.get("wind_components") and field.get("source_column"):
            reserved.update(field_key(index, item) for item in fields if item["name"] == field["source_column"])


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
            reserve_mutation_confirmation(scan,task)
        except (EcoDataError, OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, ImportError) as error:
            task["holdout_lock_error"] = str(error)


def reserve_mutation_confirmation(scan,task):
    """Split the already unexposed reserve before any recommendation is scored.

    Keep its original identity for the existing exposure checks. No values are
    read here. One-unit and non-temporal reserves cannot prove final isolation.
    """
    import pandas as pd
    plan=task['locked_holdout']
    if 'days' in plan:
        units=plan['days']
        times=list(units)
        key='days'
    elif 'times' in plan:
        from contextlib import ExitStack
        from .windowed_grid_recipe import pair
        units=plan['times']
        with ExitStack() as stack:
            primary,_,_,_=pair(stack,scan,task)
            if primary.times is None:
                task['mutation_reservation']=dict(status='INSUFFICIENT',reason='No temporal holdout for a separate mutation review')
                return
            times=[str(pd.Timestamp(primary.times[item[0]])) for item in units]
        key='times'
    else:
        task['mutation_reservation']=dict(status='INSUFFICIENT',reason='This route requires its native independent reference protocol')
        return
    if len(units)<2:
        task['mutation_reservation']=dict(status='INSUFFICIENT',reason='Fewer than two unexposed units; no repeated acceptance on the same unit')
        return
    cut=max(1,len(units)//2)
    task['recipe_review_units']={key:units[:cut]}
    task['mutation_confirmation_times']=times[cut:]
    task['mutation_reservation']=dict(status='RESERVED',version='recommendation_then_mutation_v1',
        recommendation_review_times=times[:cut],confirmation_times=times[cut:],
        source_exposure=snapshot(scan),frozen_before_recommendation_review=True,
        scope='Temporal units of the current unexposed reserve, not a universal independence guarantee')
