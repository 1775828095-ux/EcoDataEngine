"""Fixed multi-source table recipes over local station or calendar indexes."""
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.engines.units import UnitEngine

from . import csv_index, recipe_trials
from .sampling_plan import IDENTIFIERS, PROBE_LIMIT, TIME, normalized, records, station_reserved
from .scan import local_member

FORMATS = {"CSV", "TSV", "Excel", "Parquet"}
LINEAR = "temporal.wide_table_interpolation"
TIME_NAMES = TIME | {"time_utc"}
STATION_NAMES = IDENTIFIERS - {"id"}


class IndexRequired(ValueError):
    pass


@dataclass
class Table:
    header: list
    groups: dict
    category: str
    held: set
    read: object
    value: int
    time: int
    station: int | None
    scale: float
    offset: float


def supports(row, field):
    if row["format"] in FORMATS:
        return True
    dims = field.get("dimensions", ())
    return (row["format"] in {"NetCDF", "HDF5"} and len(dims) == 1
            and normalized(dims[0]) in TIME and not row.get("original_reader"))


def cf_series(stack, row, field, unit, *, short_windows=False):
    """Index complete holdout groups using coordinates, then slice requested values."""
    from .time_metadata import cell_methods
    path = stack.enter_context(local_member(row))
    ds = stack.enter_context(xr.open_dataset(path, cache=False))
    array = ds[field["name"]]
    axis, = array.dims
    clock = ds.coords[axis]
    operations = cell_methods(array.attrs.get("cell_methods", ""), {axis, "time"})
    semantics = array.attrs.get("temporal_semantics") or ("instantaneous" if operations == ["point"] else None)
    if (not np.issubdtype(clock.dtype, np.datetime64)
            or clock.attrs.get("bounds") or semantics != "instantaneous"
            or array.attrs.get("flag_values") is not None or array.attrs.get("flag_meanings")
            or array.attrs.get("data_kind") == "categorical"
            or any(operation != "point" for operation in operations)):
        raise ValueError("一维时间序列插值需要明确的瞬时连续观测；区间统计量或分类变量需要对应的评分规则")
    stamps = np.asarray(clock)
    if not len(stamps) or np.isnat(stamps).any() or (stamps[1:] <= stamps[:-1]).any():
        raise ValueError("时间坐标为空、无效、重复或乱序")
    by_day = short_windows and len(stamps) > 1 and np.median(np.diff(stamps)) <= np.timedelta64(6, "h")
    periods = stamps.astype("datetime64[D]" if by_day else "datetime64[M]")
    starts = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1]
    groups = {str(periods[i]): slice(int(i), int(j)) for i, j in zip(starts, [*starts[1:], len(stamps)])}
    boundary = len(stamps) - max(1, len(stamps) // 5)
    held = {key for key, span in groups.items() if span.start >= boundary}

    def read(key):
        span = groups[key]
        return zip(stamps[span], array.isel({axis: span}).values, strict=True)

    scale, offset = UnitEngine().affine(array.attrs["units"], unit)
    return Table([axis, field["name"]], groups, "day" if by_day else "month", held, read, 1, 0, None, scale, offset)


def columns(header, field):
    names = [normalized(name) for name in header]
    target = field.get("source_column") or field.get("original_header") or field["name"]
    time = next((i for i, name in enumerate(names) if name in TIME_NAMES), None)
    station = next((i for i, name in enumerate(names) if name in STATION_NAMES), None)
    if target not in header or time is None:
        raise ValueError("联合表格方案需能从表头核对数值列和时间键")
    return header.index(target), time, station


def workbook_frame(stack, row, field):
    from openpyxl import load_workbook

    path = stack.enter_context(local_member(row))
    stream = stack.enter_context(path.open("rb"))
    book = load_workbook(stream, read_only=True, data_only=True)
    stack.callback(book.close)
    sheet = book[field.get("sheet") or next(s.title for s in book if s.sheet_state == "visible")]
    header = list(next(sheet.values, ()))
    _, time, station = columns(header, field)
    key_column = time if station is None else station
    groups, initial_months = {}, set()
    for line, (value,) in enumerate(sheet.iter_rows(min_row=2, min_col=key_column+1,
                                                    max_col=key_column+1, values_only=True), 2):
        key = csv_index.month(value) if station is None else str(value).strip() if value is not None else ""
        if not key:
            continue
        groups.setdefault(key, []).append(line)
        if station is None and line <= PROBE_LIMIT + 1:
            initial_months.add(key)

    def read(group):
        rows = groups[group]
        positions = set(rows)
        for index, values in enumerate(sheet.iter_rows(min_row=rows[0], max_row=rows[-1], values_only=True), rows[0]):
            if index in positions:
                yield values

    return header, groups, sorted(initial_months), read


def indexed_csv(stack, row, field):
    cached_path = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with records(row, field, encoding) as (rows, _, __):
                header = list(next(rows, ()))
            _, _, station = columns(header, field)
        except UnicodeDecodeError:
            continue
        index = csv_index.load(row, encoding, header)
        local = row
        if index is None:
            if cached_path is None:
                cached_path = stack.enter_context(local_member(row))
            try:
                document = csv_index.scan_index(cached_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
            if document["headers"] != header:
                raise IndexRequired("生成分组索引时发现表头已变化")
            index = document["groups"]
            months = document["prefix_months"]
            local = dict(row, ref=(cached_path, "", None, None), containers=[])
        else:
            months = sorted(index.get("month", {}))
        category = "station" if station is not None else "month"
        groups = index.get(category, {})
        return header, groups, months, lambda key: csv_index.indexed_records(local, encoding, groups[key], None)
    raise IndexRequired("无法读取 CSV/TSV 分组键，需可识别的编码、表头或本地索引")


def parquet_frame(stack, row, field):
    from pyarrow.parquet import ParquetFile
    from .sample_values import PROBE_ROWS

    path = stack.enter_context(local_member(row))
    parquet = ParquetFile(stack.enter_context(path.open("rb")))
    header = parquet.schema_arrow.names
    _, time, station = columns(header, field)
    key_name = header[time if station is None else station]

    def key(value):
        return csv_index.month(value) if station is None else str(value).strip() if value is not None else ""

    groups, outbound = {}, set()
    offset = 0
    for index in range(parquet.num_row_groups):
        keys = parquet.read_row_group(index, columns=[key_name]).column(0).to_pylist()
        for local, value in enumerate(keys):
            if (group := key(value)):
                groups.setdefault(group, set()).add(index)
                if offset + local < PROBE_ROWS:
                    outbound.add(group)
        offset += len(keys)

    def read(group):
        for index in sorted(groups[group]):
            for batch in parquet.iter_batches(row_groups=[index], batch_size=4096):
                for row in batch.to_pylist():
                    if key(row[key_name]) == group:
                        yield [row[name] for name in header]

    return header, groups, set(groups) - outbound, read


def table(stack, row, field, unit):
    if row["format"] in {"NetCDF", "HDF5"}:
        return cf_series(stack, row, field, unit)
    scale, offset = UnitEngine().affine(field.get("unit"), unit)
    if row["format"] == "Excel":
        header, groups, months, read = workbook_frame(stack, row, field)
    elif row["format"] == "Parquet":
        header, groups, withheld, read = parquet_frame(stack, row, field)
    else:
        header, groups, months, read = indexed_csv(stack, row, field)
    value, time, station = columns(header, field)
    if row["format"] == "Parquet":
        held = withheld
    elif station is None:
        # With an index, months lists the whole file; without one, only the
        # bounded outbound prefix was available. Later unseen months are local.
        held = set(groups).difference(months)
        if len(months) >= 2 and not row.get("original_reader"):
            held.add(months[-1])
    elif normalized(header[station]) in IDENTIFIERS - {"id"}:
        held = {key for key in groups if station_reserved(key)}
    else:
        held = set()
    return Table(header, groups, "station" if station is not None else "month", held, read,
                 value, time, station, scale, offset)


def rows_for(source, group):
    records = []
    for row in source.read(group):
        if len(row) <= max(source.value, source.time, source.station or 0):
            raise ValueError("索引指向的观测记录缺少键或数值列")
        station = str(row[source.station]).strip() if source.station is not None else ""
        raw = row[source.time]
        if source.station is not None:
            key = station
        elif source.category == "day":
            key = str(np.datetime64(raw, "D"))
        else:
            key = csv_index.month(raw)
        if key != group:
            raise ValueError("索引分组与原始记录不一致")
        records.append((station, raw, row[source.value]))
    frame = pd.DataFrame(records, columns=["station_id", "time", "value"])
    frame["time"] = pd.to_datetime(frame.time.astype(str).str.strip(), format="mixed", utc=True, errors="raise").dt.tz_localize(None)
    frame["value"] = pd.to_numeric(frame.value, errors="coerce") * source.scale + source.offset
    if frame.time.isna().any() or frame.duplicated(["station_id", "time"]).any():
        raise ValueError("表格组有无效或重复的站点时刻")
    return frame.set_index(["station_id", "time"]).value.sort_index()


def pair(stack, scan, task):
    rows = [scan["rows"][index] for index, _ in task["fields"]]
    if rows[0]["ref"] == rows[1]["ref"] and rows[0].get("containers") == rows[1].get("containers"):
        raise ValueError("同一输入不能组成两个独立来源")
    if any(field.get("temporal_semantics") not in {None, "", "instantaneous"} for field in task["field_metadata"]):
        raise ValueError("累计或区间平均的表格需与可核对的区间支持配对")
    timed_arrays = all(row["format"] in {"NetCDF", "HDF5"} for row in rows)
    sources = []
    for row, field in zip(rows, task["field_metadata"], strict=True):
        if timed_arrays:
            sources.append(cf_series(stack, row, field, task["unit"], short_windows=True))
        else:
            sources.append(table(stack, row, field, task["unit"]))
    if timed_arrays and sources[0].category != sources[1].category:
        for i, source in enumerate(sources):
            if source.category == "day":
                sources[i] = cf_series(stack, rows[i], task["field_metadata"][i], task["unit"])
    if sources[0].category != sources[1].category:
        raise ValueError("两张表需使用同一种可对应的站点或月份分组键")
    return sources


def combine(series, task, steps):
    aligned = pd.concat(series, axis=1).sort_index()
    if LINEAR in steps:
        for _, group in aligned.groupby(level="station_id", sort=False):
            interp = group.droplevel("station_id").interpolate(method="time", limit_area="inside")
            aligned.loc[group.index] = interp.to_numpy()
    variable = task["quantity"]
    attrs = dict(physical_variable=variable, units=task["unit"], spatial_semantics="point",
                 temporal_semantics="instantaneous", data_kind="continuous")
    arrays = [xr.Dataset({variable: ("record", aligned.iloc[:, column].to_numpy(), attrs)}) for column in range(2)]
    result, _ = recipe_trials.combine(arrays, variable, steps)
    return pd.Series(result[variable].values, index=aligned.index), aligned


def held_groups(scan, task, primary, backup):
    from .holdout_plan import exposures, table_groups

    eligible = [table_groups(source, exposures(scan, index, field))
                for source, (index, _), field in zip((primary, backup), task["fields"], task["field_metadata"], strict=True)]
    return sorted(eligible[0] & eligible[1])


def freeze_holdout(scan, task):
    with ExitStack() as stack:
        return {"groups": held_groups(scan, task, *pair(stack, scan, task))}


def review(scan, task, steps, *, independent=False):
    with ExitStack() as stack:
        primary, backup = pair(stack, scan, task)
        common = set(primary.groups) & set(backup.groups)
        groups = sorted(common) if independent else held_groups(scan, task, primary, backup)
        if not independent and (plan := task.get("locked_holdout")) is not None:
            if not set(plan["groups"]) <= set(groups):
                raise ValueError("锁定的表格回检组与外发记录重叠，或原分组已变化")
            groups = plan["groups"]
        scope = task.get("bayesian_review_scope")
        phase = scope["mask_phase"] if scope else None if independent else task.get("bayesian_mask_phase")

        def trial(index):
            group = groups[index]
            left, right = rows_for(primary, group), rows_for(backup, group)
            paired = left[np.isfinite(left)].index.intersection(right[np.isfinite(right)].index)
            actual, aligned = combine([left, right], task, steps)
            if (np.isfinite(aligned).any(axis=1) & ~np.isfinite(actual)).any():
                return group, None, "recipe_output_missing_on_reference"

            scores = []
            failures = []
            dropout = paired[1::4] if phase is None else paired[
                recipe_trials.reference_positions(len(paired), 2*phase, index, partitions=8)]
            withheld = left.copy()
            withheld.loc[dropout] = np.nan
            fill_steps = [step for step in steps if step != recipe_trials.BLEND]
            filled, _ = combine([withheld, right], task, fill_steps)
            metrics, reason = recipe_trials.score(
                filled.reindex(dropout).to_numpy(), right.reindex(dropout).to_numpy(),
                left.reindex(dropout).to_numpy(), "masked_local_table_observations",
                retained=np.r_[withheld.to_numpy(), right.to_numpy()], request=task.get("scoring_request"))
            if metrics:
                scores.append(dict(metrics, assessment="gap_fill"))
            if reason:
                failures.append("insufficient_gap_score_support" if reason == "insufficient_score_support" else reason)

            if recipe_trials.BLEND in steps:
                # The original primary measurement is the reference. Reconstruct
                # it using the recipe's temporal alignment before applying fusion,
                # and score only points with both inputs actually available.
                fusion = paired[2::4] if phase is None else paired[
                    recipe_trials.reference_positions(len(paired), 2*phase+1, index, partitions=8)]
                withheld = left.copy()
                withheld.loc[fusion] = np.nan
                blended, inputs = combine([withheld, right], task, steps)
                inputs = inputs.reindex(fusion)
                supported = np.isfinite(inputs).all(axis=1)
                truth = left.reindex(fusion).where(supported).to_numpy()
                baselines = inputs.to_numpy().T
                best = min(baselines, key=lambda prediction: np.nansum((prediction - truth)**2))
                metrics, reason = recipe_trials.score(
                    blended.reindex(fusion).to_numpy(), best, truth, "reconstructed_primary_table_proxy",
                    retained=np.r_[withheld.to_numpy(), right.to_numpy()], request=task.get("scoring_request"))
                if metrics:
                    scores.append(dict(metrics, assessment="fusion"))
                if reason:
                    failures.append("insufficient_fusion_score_support" if reason == "insufficient_score_support" else reason)
            failure = next((reason for reason in failures if reason not in recipe_trials.UNSCORED), None)
            return group, scores, failure or (failures[0] if failures else None)

        return recipe_trials.audit(len(groups), f"withheld_{primary.category}", trial, scope=scope)


def run(scan, task_id, task, folder, model_id, progress):
    from ecodata.io.structures import export_structure, table_dataset

    if task.get("search_after_rejection") or task.get('recommended_graphs'):
        return run_search(scan, task_id, task, folder, progress)
    steps, = task["recipes"]
    with ExitStack() as stack:
        left, right = pair(stack, scan, task)
        groups = sorted(set(left.groups) & set(right.groups))
        values = [combine([rows_for(left, group), rows_for(right, group)], task, steps)[0] for group in groups]
        if not values:
            raise ValueError("表格没有共同分组")
        data = pd.concat(values).rename(task["quantity"]).reset_index()
    ds = table_dataset(data)
    ds[task["quantity"]].attrs["units"] = task["unit"]
    raw, filename, _ = export_structure(ds, "Parquet")
    artifact = dict(name=filename, dimensions=dict(ds.sizes), fields=list(ds.data_vars))
    return dict(files=[(filename, raw)], source_formats=[scan["rows"][index]["format"] for index, _ in task["fields"]],
                target_format="Parquet", original_report=[dict(task=task_id, methods=steps)],
                local=dict(artifacts=[artifact], reason="已输出按真实键对齐、补缺和融合的观测表。"))


def run_search(scan, task_id, task, folder, progress):
    from .local_search import run_paired

    with ExitStack() as stack:
        left, right = pair(stack, scan, task)
        # Recommendation review groups are excluded, not recycled as test truth.
        held = set(task.get("locked_holdout", {}).get("groups", ()))
        common = set(left.groups) & set(right.groups)
        groups = sorted(common - held)
        reused_development = not groups
        if reused_development:
            groups = sorted(common)
        frames = []
        for group in groups:
            joined = pd.concat([rows_for(left, group).rename("observed"),
                                rows_for(right, group).rename("background")], axis=1).dropna()
            frames.append(joined.reset_index())
        if not frames:
            raise ValueError("没有与推荐抽检分离的真实双源配对记录")
        frame = pd.concat(frames, ignore_index=True).rename(columns={"station_id":"station", "time":"grid_time"})
        frame["station"] = frame.station.replace("", "unlocated-series")
        frame["unit"] = task["unit"]
    request = dict(task="temporal", adapter="paired_station_calibration", variable=task["quantity"],
                   unit=task["unit"], temporal_semantics="instantaneous", unlocated_series=True,
                   recommended_recipes=[], reference_basis="RECONSTRUCTED_PRIMARY_SERIES_PROXY",
                   recommendation_isolation=task.get("mutation_reservation", dict(status="INSUFFICIENT",
                       reason="No prospective separate recommendation/mutation reserve")),
                   evaluation_protocol="blocked_confirmation_v1")
    if reused_development:
        request["recommendation_isolation"] = dict(status="INSUFFICIENT",
            reason="Recommendation groups reused as development only; no independent confirmation claim")
    if task.get('recommended_graphs'):request['recommended_graphs']=task['recommended_graphs']
    return run_paired(scan, task_id, task, frame, request, folder, progress,
                      dict(excluded_recommendation_groups=[] if reused_development else sorted(held),
                           recommendation_groups_reused_for_development=reused_development, paired_records=len(frame),
                           reference="primary-series proxy, not independent measurement truth"),
                      source_indices=[index for index, _ in task["fields"]])
