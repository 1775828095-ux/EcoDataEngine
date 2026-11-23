"""Disjoint recommendation and local review partitions for numeric evidence."""
import csv
import hashlib
import re
from contextlib import contextmanager
from io import TextIOWrapper
from itertools import chain, islice, zip_longest

from .scan import AXES, local_member, open_tabular_member

SUPPORTED = {"NetCDF", "HDF5", "GeoTIFF", "CSV", "TSV", "Excel"}
IDENTIFIERS = {"station_id", "station", "site_id", "site", "sensor_id", "object_id", "id", "站点", "站点编号"}
TIME = {"time", "tstep", "date", "datetime", "timestamp", "time_utc", "时间", "日期"}
PROBE_LIMIT = 4096
VALUE_LIMIT = 64
TIME_COORD_LIMIT = 32768
LOCAL_SUMMARIES = {"original_probe_payload", "time_coverage", "time_coverage_evidence",
                   "declared_coverage_duration_seconds", "declared_output_print_period",
                   "sampling_interval_seconds", "time_cell_duration_seconds", "time_seconds",
                   "time_resolution_class", "time_spacing_evidence"}


def recommendation_metadata(value):
    """Whole-series observed coverage stays in the local catalogue."""
    if isinstance(value, dict):
        return {key: recommendation_metadata(item) for key, item in value.items() if key not in LOCAL_SUMMARIES}
    if isinstance(value, list):
        return [recommendation_metadata(item) for item in value]
    return value


def numerics(values):
    from .sample_values import numeric
    return list(numeric(values))


def member_available(row):
    from .sample_values import MAX_SAMPLE_MATERIALIZATION
    leaf = row.get("attribute_member", row)
    if any(size > MAX_SAMPLE_MATERIALIZATION for _, size in leaf.get("containers", ())):
        return False
    archive = leaf["ref"][1]
    streamable = row["format"] in {"CSV", "TSV"} and archive in {"zip", "tar"}
    return not ((row.get("original_reader") or (archive and not streamable))
                and leaf["bytes"] > MAX_SAMPLE_MATERIALIZATION)


def sample(row, field, *, holdout=False, spatial=False, future=False):
    if row["format"] not in SUPPORTED or row.get("original_reader") or not member_available(row):
        return dict(status="pending", basis="bounded_reader_unavailable", values=[])
    name = normalized(field.get("source_column") or field["name"])
    if field.get("axis") or name in AXES or name in IDENTIFIERS or name in TIME:
        return dict(status="pending", basis="non_observation_field", values=[])
    if row["format"] in {"NetCDF", "HDF5"}:
        record = netcdf(row, field, holdout, spatial)
    elif row["format"] == "GeoTIFF":
        record = raster(row, field, holdout)
    else:
        record = table(row, field, holdout, future=future)
    record["status"] = "checked" if any(v is not None for v in record["values"]) else "pending"
    return record


def time_dimension(array, dataset):
    return next((dim for dim in array.dims
                 if dim.casefold() in TIME or
                 (dim in dataset.coords and dataset[dim].attrs.get("axis") == "T")), None)


def open_netcdf_for_sampling(path):
    from inspect import signature
    import xarray as xr

    # Older supported xarray releases don't accept this option.
    supports_lazy_indexes = "create_default_indexes" in signature(xr.open_dataset).parameters
    options = {"create_default_indexes": False} if supports_lazy_indexes else {}
    return xr.open_dataset(path, decode_times=False, cache=False, **options)


def netcdf(row, field, holdout, spatial=False):
    import numpy as np

    with local_member(row) as path, open_netcdf_for_sampling(path) as dataset:
        array = dataset[field["name"]]
        if not array.ndim or not all(array.shape):
            return dict(basis="no_disjoint_partition", values=[])
        dims = tuple(array.dims)
        temporal = time_dimension(array, dataset)
        spatial_axis = next((dim for dim in dims if dim != temporal and array.sizes[dim] > 1), None)
        time_partition = temporal if temporal and array.sizes[temporal] > 1 else None
        partition = (spatial_axis if spatial else time_partition) or spatial_axis or time_partition
        if partition is None or array.sizes[partition] < 2:
            return dict(basis="no_disjoint_partition", values=[])
        size = array.sizes[partition]
        reserved = max(1, size // 5)
        boundary = size - reserved
        spacing = row.get("declared_time_resolution")
        tile_axes = [dim for dim in dims if dim not in {partition, temporal} and array.sizes[dim] >= 4][:2]
        width = min(16 // 2**len(tile_axes), reserved if holdout else max(1, boundary // 4))
        start, stop = (boundary, size) if holdout else (0, boundary)
        coordinate = dataset.coords.get(temporal) if partition == temporal else None
        windows, frequency_scope = frequency_windows(coordinate, partition, start, stop, width)
        regions, groups = [], []
        for index, (lower, upper, interval) in enumerate(windows):
            portion = index / max(1, len(windows) - 1)
            selection = {}
            for dim in dims:
                if dim == partition:
                    continue
                size_dim = array.sizes[dim]
                limit = size_dim - max(1, size_dim // 5) if size_dim > 1 and dim == spatial_axis else size_dim
                tile = 2 if dim in tile_axes else 1
                position = int((limit - tile) * (0.5 if holdout else portion))
                selection[dim] = slice(position, position + tile) if tile > 1 else position
            selection[partition] = slice(lower, upper)
            groups.append(array.isel(selection).values.reshape(-1))
            region = {"axis": partition, "start_index": lower, "stop_index": upper, "interval": interval}
            if holdout:
                region["local_values"] = numerics(groups[-1])
            if coordinate is not None:
                coordinate_values = coordinate.isel({temporal: slice(lower, upper)}).values
                bounds = coordinate.attrs.get("bounds")
                numeric_times = np.issubdtype(coordinate_values.dtype, np.number)
                region["time"] = dict(first=float(coordinate_values[0]) if numeric_times else str(coordinate_values[0]),
                    last=float(coordinate_values[-1]) if numeric_times else str(coordinate_values[-1]),
                    units=coordinate.attrs.get("units"), calendar=coordinate.attrs.get("calendar"))
                if numeric_times:
                    intervals = np.diff(coordinate_values)
                    region["time"]["ordered"] = bool(np.isfinite(coordinate_values).all() and (intervals > 0).all())
                    if len(intervals):
                        region["time"]["interval_range"] = [float(intervals.min()), float(intervals.max())]
                if bounds in dataset and temporal in dataset[bounds].dims:
                    edges = dataset[bounds].isel({temporal: slice(lower, upper)}).transpose(temporal, ...).values
                    if np.isfinite(edges).all():
                        region["time"]["bounds"] = edges.tolist()
                        if edges.ndim == 2 and edges.shape[1] == 2:
                            region["time"]["interval_widths"] = (edges[:, 1] - edges[:, 0]).tolist()
                            region["time"]["intervals_valid"] = bool(
                                (edges[:, 1] > edges[:, 0]).all() and (edges[1:, 0] >= edges[:-1, 1]).all())
            regions.append(region)
        values = numerics(v for group in zip_longest(*groups) for v in group)
        return dict(basis="time_windows" if partition == temporal else "spatial_blocks",
                    partition=dict(axis=partition, holdout_start=boundary, size=size),
                    unit=array.attrs.get("units") or field.get("unit"), frequency=spacing,
                    frequency_scope=frequency_scope,
                    variable=field["name"], windows=regions, values=values)


def frequency_windows(coordinate, axis, start, stop, width):
    """Group regular runs on the time coordinate without loading the data variable."""
    import numpy as np

    def split(lower, upper):
        if coordinate is None:
            return [(lower, upper, None)]
        values = coordinate.isel({axis: slice(lower, upper)}).values
        if not np.issubdtype(values.dtype, np.number) or len(values) < 2:
            return [(lower, upper, None)]
        points = np.diff(values.astype(float))
        runs = []
        segment_start = 0
        cadence = points[0]
        discontinuities = np.flatnonzero(np.r_[not np.isfinite(cadence) or cadence <= 0, (
            ~np.isclose(points[1:], points[:-1], rtol=1e-5, atol=1e-9)
            | ~np.isfinite(points[1:]) | (points[1:] <= 0))])
        for index in discontinuities:
            if index >= segment_start:
                runs.append((lower + segment_start, lower + index + 1,
                             float(cadence) if np.isfinite(cadence) and cadence > 0 else None))
                segment_start = index + 1
                cadence = points[index + 1] if index + 1 < len(points) else np.nan
        runs.append((lower + segment_start, upper, float(cadence) if np.isfinite(cadence) and cadence > 0 else None))
        return runs

    count = min(4, max(1, (stop - start) // width))
    starts = sorted({start + i * (stop - start - width) // max(1, count - 1) for i in range(count)})
    regular = [(index, index + width, None) for index in starts]
    if coordinate is None:
        return regular, "not_time_axis"
    if stop - start > TIME_COORD_LIMIT:
        runs = [run for lo, hi, _ in regular for run in split(lo, hi)]
        scope = "sampled_time_coordinate_windows"
    else:
        runs = split(start, stop)
        if len(runs) == 1:
            return [(lo, hi, runs[0][2]) for lo, hi, _ in regular], "partition_time_coordinates"
        scope = "partition_time_coordinates"
    # Cover distinct cadences first, then use remaining slots to spread over runs.
    selected = {}
    for run in sorted(runs, key=lambda item: item[1] - item[0], reverse=True):
        selected.setdefault(run[2], run)
    ordered = list(selected.values())[:4]
    ordered.extend([run for run in runs if run not in ordered][:4 - len(ordered)])
    return [(lo, min(hi, lo + width), interval) for lo, hi, interval in sorted(ordered)], scope


def netcdf_time_review(row, field, *, allow_materialization=False):
    """Audit the complete time coordinate locally, with bounded memory and no grid loads."""
    import numpy as np

    if not allow_materialization and not member_available(row):
        return None
    with local_member(row) as path, open_netcdf_for_sampling(path) as dataset:
        dim = time_dimension(dataset[field["name"]], dataset)
        if dim not in dataset.coords or not np.issubdtype(dataset[dim].dtype, np.number):
            return None
        clock = dataset[dim]
        bound_name = clock.attrs.get("bounds")
        bounds = dataset[bound_name].transpose(dim, ...) if bound_name in dataset and dim in dataset[bound_name].dims else None
        count = clock.size
        previous = None
        previous_step = np.nan
        previous_end = None
        previous_width = np.nan
        bad_coordinates = overlaps = bad_bounds = nonincreasing = changes = 0
        examples = []
        width_changes = 0
        width_examples = []
        for start in range(0, count, TIME_COORD_LIMIT):
            window = {dim: slice(start, start + TIME_COORD_LIMIT)}
            values = np.asarray(clock.isel(window).values, dtype=float)
            bad_coordinates += int((~np.isfinite(values)).sum())
            steps = np.diff(values, prepend=previous) if previous is not None else np.diff(values)
            if len(steps):
                earlier = np.r_[previous_step, steps[:-1]]
                changed = (np.isfinite(earlier) & (earlier > 0) & np.isfinite(steps) & (steps > 0)
                           & ~np.isclose(earlier, steps, rtol=1e-5, atol=1e-9))
                changes += int(changed.sum())
                indices = np.flatnonzero(changed)[:max(0, 16 - len(examples))]
                examples.extend(int(start + idx + (previous is None)) for idx in indices)
                nonincreasing += int((steps <= 0).sum())
                previous_step = steps[-1]
            if len(values):
                previous = values[-1]
            if bounds is not None:
                edge = np.asarray(bounds.isel(window).values, dtype=float)
                if edge.ndim != 2 or edge.shape[1] != 2:
                    bad_bounds += len(values)
                    continue
                bad_bounds += int((~np.isfinite(edge).all(axis=1) | (edge[:, 1] <= edge[:, 0])).sum())
                overlaps += int((edge[1:, 0] < edge[:-1, 1]).sum())
                if len(edge):
                    overlaps += int(previous_end is not None and edge[0, 0] < previous_end)
                    previous_end = edge[-1, 1]
                    width = edge[:, 1] - edge[:, 0]
                    earlier_width = np.r_[previous_width, width[:-1]]
                    different = np.isfinite(earlier_width) & np.isfinite(width) & ~np.isclose(earlier_width, width)
                    width_changes += int(different.sum())
                    width_examples.extend(int(start + index) for index in
                                          np.flatnonzero(different)[:max(0, 16 - len(width_examples))])
                    previous_width = width[-1]
        return dict(scope="all_time_coordinates", coordinate_count=count, interval_changes=changes,
                    change_examples=examples, nonincreasing=nonincreasing, invalid_coordinates=bad_coordinates,
                    invalid_bounds=bad_bounds, overlapping_bounds=overlaps,
                    interval_width_changes=width_changes, interval_width_examples=width_examples)


def raster(row, field, holdout):
    import rasterio
    from rasterio.windows import Window

    with local_member(row) as path, rasterio.open(path) as dataset:
        along_y = dataset.height >= 2
        size = dataset.height if along_y else dataset.width
        if size < 2:
            return dict(basis="no_disjoint_partition", values=[])
        boundary = size - max(1, size // 5)
        choice = [(1, 1)] if holdout else [(0, 0), (1, 0), (0, 1), (1, 1)]
        windows, groups = [], []
        for across_x, across_y in choice:
            start, stop = (boundary, size) if holdout else (0, boundary)
            width = min(4, dataset.width if along_y else stop - start)
            height = min(4, stop - start if along_y else dataset.height)
            x = int((dataset.width - width) * across_x) if along_y else start + int((stop - start - width) * across_x)
            y = start + int((stop - start - height) * across_y) if along_y else int((dataset.height - height) * across_y)
            window = Window(x, y, width, height)
            groups.append(dataset.read(field["band"], window=window, masked=True).ravel().tolist())
            windows.append(dict(row=y, column=x, width=width, height=height))
        return dict(basis="spatial_blocks", partition=dict(axis="row" if along_y else "column",
                    holdout_start=boundary, size=size), windows=windows, crs=str(dataset.crs) if dataset.crs else None,
                    resolution=list(dataset.res), nodata=dataset.nodata, unit=field.get("unit"),
                    values=numerics(v for group in zip_longest(*groups) for v in group))


def raster_windows(dataset, band):
    from rasterio.windows import Window

    for _, block in dataset.block_windows(band):
        for row in range(0, int(block.height), 512):
            for column in range(0, int(block.width), 512):
                yield Window(int(block.col_off) + column, int(block.row_off) + row,
                             min(512, int(block.width) - column), min(512, int(block.height) - row))


def normalized(value):
    return re.sub(r"[\s-]+", "_", str(value or "").strip().casefold())


@contextmanager
def records(row, field, encoding, sheet_name=None):
    if row["format"] == "Excel":
        from openpyxl import load_workbook
        with local_member(row) as path:
            with path.open("rb") as stream:
                workbook = load_workbook(stream, read_only=True, data_only=True)
                try:
                    visible = [sheet.title for sheet in workbook if sheet.sheet_state == "visible"
                               and sheet.title != "_ecodata_metadata"]
                    if not visible:
                        raise ValueError("没有可用于抽样的可见工作表")
                    sheet = workbook[sheet_name or field.get("sheet") or visible[0]]
                    yield iter(sheet.values), sheet.title, visible
                finally:
                    workbook.close()
    else:
        with open_tabular_member(row) as raw, TextIOWrapper(raw, encoding=encoding, newline="") as stream:
            yield csv.reader(stream, delimiter="\t" if row["format"] == "TSV" else ","), None, []


def table(row, field, holdout, *, future=False):
    from .csv_index import indexed_records, load, month

    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with records(row, field, encoding) as (rows, sheet, sheets):
                header = list(next(rows, ()))
                names = [normalized(name) for name in header]
                original = field.get("source_column") or field.get("original_header") or field["name"]
                if original not in header:
                    return dict(basis="column_changed", values=[])
                col = header.index(original)
                station = next((i for i, name in enumerate(names) if name in IDENTIFIERS - {"id"}), None)
                station_col = station
                time = next((i for i, name in enumerate(names) if name in TIME), None)
                indexed = load(row, encoding, header) if row["format"] in {"CSV", "TSV"} else None
                prefix = [] if indexed else list(islice(rows, PROBE_LIMIT))
                months = (sorted(indexed.get("month", {})) if indexed else
                          sorted({candidate for record in prefix if time is not None and len(record) > time
                                  if (candidate := month(record[time]))}))
                held_month = months[-1] if len(months) >= 2 else None
                if future and holdout and time is not None and not indexed and row["format"] in {"CSV", "TSV"}:
                    review = latest_month_review(prefix, rows, time, station, col, held_month)
                    if review["values"] or station is None:
                        return dict(basis="calendar_months", unit=field.get("unit"), sheet=sheet, **review)
                    # A single observed month cannot support a future-month test.
                    # Fall back to independently reserved stations within the prefix.
                    future = False
                    with records(row, field, encoding) as (again, _, __):
                        next(again, None)
                        review = stratified_csv_review(again, station, time, col, held_month)
                    return dict(basis="station_groups", unit=field.get("unit"), sheet=sheet, **review)
                if future and held_month:
                    station = None
                basis = "station_groups" if station is not None else "calendar_months" if time is not None else "row_blocks"
                if basis == "row_blocks" and len(sheets) > 1:
                    reserved = sheets[-1]
                    if sheet == reserved and not holdout:
                        return dict(basis="worksheet_groups", partition=dict(held_sheet=reserved), values=[])
                    if holdout and sheet != reserved:
                        with records(row, field, encoding, reserved) as (other, _, __):
                            alternate = list(next(other, ()))
                            if original not in alternate:
                                return dict(basis="heldout_schema_mismatch", values=[])
                            index = alternate.index(original)
                            peer = next((item for item in row.get("fields", ())
                                         if item.get("sheet") == reserved and
                                         (item.get("original_header") or item.get("name")) == original), {})
                            if field.get("unit") and peer.get("unit") and field["unit"] != peer["unit"]:
                                from ecodata.engines.units import UnitEngine
                                if not UnitEngine().equivalent(field["unit"], peer["unit"]):
                                    return dict(basis="heldout_sheet_unit_conflict", values=[])
                            verified_semantics = False
                            for attribute in ("standard_name", "physical_variable", "temporal_semantics"):
                                if field.get(attribute) and peer.get(attribute) and field[attribute] != peer[attribute]:
                                    return dict(basis="heldout_sheet_semantics_conflict", values=[])
                                if field.get(attribute) and peer.get(attribute) and attribute != "temporal_semantics":
                                    verified_semantics = True
                            preview = list(islice(other, VALUE_LIMIT))
                            clock = next((i for i, name in enumerate(alternate) if normalized(name) in TIME), None)
                            date_check = valid_dates([record[clock] for record in preview if len(record) > clock]) if clock is not None else None
                            cells = (record[index] for record in preview if len(record) > index)
                            return dict(basis="worksheet_groups", partition=dict(held_sheet=reserved),
                                        unit=peer.get("unit"), dates_valid=date_check, values=numerics(cells),
                                        worksheet_semantics="confirmed" if verified_semantics else "needs_declaration")
                    if sheet == reserved and holdout:
                        cells = (record[col] for record in islice(rows, VALUE_LIMIT) if len(record) > col)
                        return dict(basis="worksheet_groups", partition=dict(held_sheet=reserved),
                                    unit=field.get("unit"), values=numerics(cells))
                groups = {}
                stamps = {}
                category = "station" if station is not None else "month" if time is not None else None
                if holdout and category and row["format"] in {"CSV", "TSV"}:
                    if indexed:
                        selected = [key for key in sorted(indexed.get(category, {}))
                                    if (station_reserved(key) if category == "station" else month_reserved(key, held_month))]

                        def indexed_holdout():
                            for key in distributed(selected, 16):
                                for item in indexed_records(row, encoding, indexed[category][key], None):
                                    column = station if station is not None else time
                                    if len(item) > column:
                                        actual = item[column].strip() if station is not None else month(item[column])
                                        if actual == key:
                                            yield item

                        review = stratified_csv_review(indexed_holdout(), station, time, col, held_month, station_col)
                        review["partition"]["coverage"] = "indexed_file"
                    else:
                        review = stratified_csv_review(chain(prefix, rows), station, time, col, held_month, station_col)
                    return dict(basis=basis, unit=field.get("unit"), sheet=sheet, **review)
                if indexed and category:
                    eligible = sorted(indexed.get(category, {}))
                    if category == "station":
                        eligible = [group for group in eligible if station_reserved(group) == holdout]
                    else:
                        eligible = [group for group in eligible if month_reserved(group, held_month) == holdout]
                    staged = []
                    for key in distributed(eligible, 12):
                        limit = VALUE_LIMIT if len(eligible) == 1 else 16
                        for record in indexed_records(row, encoding, indexed[category][key], limit):
                            column = station if category == "station" else time
                            if len(record) <= column:
                                continue
                            observed = str(record[column]).strip() if category == "station" else month(record[column])
                            if observed == key:
                                staged.append(record)
                    prefix = staged
                elif indexed:
                    prefix = list(islice(rows, PROBE_LIMIT))
                if holdout and basis == "row_blocks" and row["format"] in {"CSV", "TSV"}:
                    from random import Random
                    rng = Random(0)
                    pool = []
                    seen = 0
                    for index, record in enumerate(chain(prefix, rows)):
                        if len(record) <= col or (index // 8) % 4 != 3:
                            continue
                        seen += 1
                        if len(pool) < VALUE_LIMIT:
                            pool.append(record[col])
                        elif (slot := rng.randrange(seen)) < VALUE_LIMIT:
                            pool[slot] = record[col]
                    return dict(basis="uniform_heldout_row_blocks", unit=field.get("unit"),
                                partition=dict(coverage="streamed_file", eligible_records=seen),
                                values=numerics(pool))
                for index, record in enumerate(prefix):
                    if len(record) <= col:
                        continue
                    if basis == "row_blocks":
                        # Use real consecutive ranges, without scanning to the end of a large table.
                        block = index // 8
                        group = str(block)
                        reserved = block % 4 == 3
                    else:
                        key = station if station is not None else time
                        if len(record) <= key or record[key] is None:
                            continue
                        group = str(record[key]).strip()
                        if time == key and station is None:
                            group = month(group)
                            if group is None:
                                continue
                        if not group:
                            continue
                        if station is not None and not holdout and held_month and len(record) > time and month(record[time]) == held_month:
                            continue
                        reserved = station_reserved(group) if station is not None else month_reserved(group, held_month)
                    if reserved == holdout:
                        if basis == "calendar_months" and station_col is not None and station_col < len(record):
                            group = (group, str(record[station_col]).strip())
                        bucket = groups.setdefault(group, [])
                        if len(bucket) < VALUE_LIMIT:
                            bucket.append(record[col])
                            if time is not None and time < len(record):
                                stamps.setdefault(group, []).append((record[time], record[col]))
                values = numerics(islice((value for batch in zip_longest(*groups.values())
                                          for value in batch if value is not None), VALUE_LIMIT))
                return dict(basis=basis, unit=field.get("unit"), sheet=sheet,
                            partition=dict(held_month=held_month, coverage="indexed_file" if indexed else "bounded_prefix"),
                            dates_valid=valid_dates([clock for batch in stamps.values() for clock, _ in batch]) if holdout and time is not None else None,
                            groups=len(groups), values=values, observation_groups=list(stamps.values()) if holdout else [])
        except UnicodeDecodeError:
            if encoding == "gb18030":
                raise


def station_reserved(group):
    return hashlib.sha256(group.encode("utf-8")).digest()[0] % 4 == 0


def month_reserved(group, held_month):
    return bool(held_month and group == held_month)


def distributed(values, count):
    if len(values) <= count:
        return values
    return [values[index * (len(values) - 1) // (count - 1)] for index in range(count)]


def valid_dates(values):
    if not values:
        return None
    import pandas as pd
    dates = pd.to_datetime([str(v).replace("_", "T", 1) for v in values], format="mixed", errors="coerce", utc=True)
    return bool(dates.notna().all())


class StratifiedReservoir:
    """Bound memory while giving each retained stratum uniform within-stratum selection."""

    def __init__(self):
        self.groups = {}
        self.seen = 0

    def add(self, group, item):
        from random import Random

        self.seen += 1
        key = int.from_bytes(hashlib.sha256(repr(group).encode()).digest()[:8], "big")
        if key not in self.groups:
            if len(self.groups) >= 16:
                largest = max(self.groups)
                if key >= largest:
                    return
                del self.groups[largest]
            self.groups[key] = ([], 0, Random(key))
        pool, count, rng = self.groups[key]
        count += 1
        self.groups[key] = pool, count, rng
        if len(pool) < 16:
            pool.append(item)
        elif (slot := rng.randrange(count)) < len(pool):
            pool[slot] = item

    def result(self, partition):
        pools = [pool for pool, _, _ in self.groups.values()]
        values = numerics(islice((pair[1] for batch in zip_longest(*pools) for pair in batch if pair), VALUE_LIMIT))
        clocks = [stamp for pool in pools for stamp, _ in pool if stamp is not None]
        return dict(partition=dict(partition, eligible_records=self.seen, sampled_strata=len(pools),
                                   selection="stratified_uniform_reservoir"),
                    values=values, observation_groups=[pool for pool in pools if all(stamp is not None for stamp, _ in pool)],
                    dates_valid=valid_dates(clocks))


def stratified_csv_review(records, station, time, column, held_month, station_col=None):
    from .csv_index import month

    reservoir = StratifiedReservoir()
    group_col = station if station is not None else time
    for record in records:
        if len(record) <= max(group_col, column):
            continue
        stamp = record[time] if time is not None and len(record) > time else None
        if station is not None:
            key = record[station].strip()
            if not key or not station_reserved(key):
                continue
        elif not month_reserved(month(stamp), held_month):
            continue
        else:
            key = (held_month, record[station_col].strip() if station_col is not None and len(record) > station_col else "")
        reservoir.add(key, (stamp, record[column]))
    return reservoir.result(dict(held_month=held_month, coverage="streamed_file"))


def latest_month_review(prefix, remainder, time, station, column, prefix_holdout):
    """Find the actual last month only after recommendation, without sending new rows."""
    from .csv_index import month

    latest = None
    reservoir = StratifiedReservoir()
    # Exclude every month that may have contributed to first-round evidence.
    sampled = {month(record[time]) for record in prefix
               if len(record) > max(time, column)
               and month(record[time]) != prefix_holdout
               and (station is None or (len(record) > station and not station_reserved(str(record[station]).strip())))
               and (station is not None or not month_reserved(month(record[time]) or "", prefix_holdout))}
    for record in chain(prefix, remainder):
        if len(record) <= max(time, column):
            continue
        key = month(record[time])
        if not key or latest is not None and key < latest:
            continue
        if key != latest:
            latest, reservoir = key, StratifiedReservoir()
        group = str(record[station]).strip() if station is not None and len(record) > station else ""
        reservoir.add(group, (record[time], record[column]))
    partition = dict(held_month=latest, coverage="streamed_file", eligible_records=reservoir.seen)
    if latest in sampled or not latest:
        partition["reason"] = "latest_month_not_independent_or_absent"
        return dict(partition=partition, values=[], observation_groups=[])
    return reservoir.result(partition)


def export_field(row, field):
    from .sample_values import export_sample, field_sample

    if row["format"] not in SUPPORTED or row.get("original_reader") or field.get("noaa_component"):
        return export_sample(field_sample(row, field, outbound=True))
    record = sample(row, field)
    payload = export_sample(record["values"])
    windows = [dict(axis=window.get("axis"), points=window["stop_index"] - window["start_index"],
                    interval=window.get("interval"),
                    time={key: value for key, value in window.get("time", {}).items()
                          if key in {"ordered", "intervals_valid", "interval_range", "interval_widths", "calendar"}})
               for window in record.get("windows", []) if "stop_index" in window]
    payload["sampling"] = dict(basis=record["basis"], groups=record.get("groups"),
                               windows=windows, unit=record.get("unit"), frequency=record.get("frequency"),
                               frequency_scope=record.get("frequency_scope"))
    return payload


def recommendation_review(scan, recommendation, gated, fields, cache=None):
    """Run only local holdout checks backed by the selected operation."""
    import numpy as np
    import xarray as xr
    from ecodata.core.exceptions import EcoDataError
    from ecodata.engines.units import UnitEngine

    steps = recommendation.get("steps", ())
    spatial = any(step.startswith(("spatial.", "raster_task.spatial")) for step in steps)
    future = any(step.startswith("temporal.") for step in steps)
    cache = {} if cache is None else cache
    reports = []
    for index, fid, field in fields:
        report = dict(field=fid, status="pending", checks=[])
        reports.append(report)
        try:
            key = (index, fid, spatial, future)
            if key not in cache:
                cache[key] = sample(scan["rows"][index], field, holdout=True, spatial=spatial, future=future)
            held = cache[key]
            values = [v for v in held["values"] if v is not None]
            report.update(partition=held.get("partition"), holdout_basis=held["basis"],
                          basis=held["basis"], count=len(values))
            row = scan["rows"][index]
            if (row["format"] == "NetCDF" and held["basis"] == "time_windows"
                    and held["partition"]["size"] > TIME_COORD_LIMIT):
                axis = held.get("partition", {}).get("axis")
                time_key = (index, "time_coordinate_audit", axis)
                if time_key not in cache:
                    cache[time_key] = netcdf_time_review(row, field)
                clock = cache[time_key]
                if clock:
                    report["time_coordinate_audit"] = clock
                    if any(clock[name] for name in ("nonincreasing", "invalid_coordinates", "invalid_bounds", "overlapping_bounds")):
                        report.update(status="failed", basis="invalid_netcdf_time_coordinates")
                        continue
            if held["basis"] in {"heldout_sheet_unit_conflict", "heldout_schema_mismatch", "heldout_sheet_semantics_conflict"}:
                report["status"] = "failed"
            if held.get("dates_valid") is False:
                report.update(status="failed", basis="invalid_heldout_table_dates")
                continue
            if held.get("dates_valid"):
                report["checks"].append("heldout_dates_parsed")
            if not values:
                continue
            times = [window["time"] for window in held.get("windows", ()) if "time" in window]
            if any(t.get("ordered") is False or t.get("intervals_valid") is False for t in times):
                report.update(status="failed", basis="invalid_heldout_time_axis")
                continue
            if "engineering.precipitation_adaptation" in steps:
                from .release_gate import precipitation_trial
                trial = precipitation_trial(held, field)
                if trial:
                    report["precipitation_trial"] = trial
                    report["checks"].append("precipitation_integrated_with_explicit_bounds")
            if held["basis"] == "spatial_blocks" and "resolution" in held:
                if not held["crs"] or any(not np.isfinite(v) or v <= 0 for v in held["resolution"]):
                    report.update(status="failed", basis="invalid_heldout_grid")
                    continue
                report["checks"].append("raster_crs_resolution_and_valid_values")
                from .raster_reference import METHODS
                method = next((METHODS[step] for step in steps if step in METHODS), None)
                source = scan["rows"][index]
                reference = scan.get("local_raster_references", {}).get(source["name"])
                if method and reference:
                    from .raster_reference import compare
                    metrics = compare(source, field, held, reference, method)
                    if metrics:
                        report["metrics"] = metrics
                        report["checks"].append("aligned_candidate_and_baseline_scored_on_independent_raster")
                        if len(steps) == 1:
                            report["status"] = "checked"
                specification = (gated or {}).get("target_grid") or {}
                resolution = (gated or {}).get("target_resolution") or specification.get("resolution")
                if "longitude_step" in specification and "latitude_step" in specification:
                    resolution = (specification["longitude_step"], specification["latitude_step"])
                target_crs = specification.get("crs") or held["crs"]
                if method and not resolution:
                    pixel = [scan["rows"][source].get("resolution", {}).get("pixel", {}) for source, _, __ in fields]
                    resolutions = [entry["step"] for entry in pixel if entry.get("step") and entry.get("crs") == target_crs]
                    resolution = tuple(map(max, zip(*resolutions))) if resolutions else None
                if method and resolution:
                    report["grid_trial"] = raster_trial(scan["rows"][index], field, held, resolution, method,
                                                        target_crs=target_crs)
                    if report["grid_trial"]:
                        report["checks"].append("raster_reproject_executed_on_holdout")
            if "format.normalize" in steps or any(step.startswith("format.") for step in steps):
                report["checks"].append("finite_values_in_heldout_source")
            if "units.harmonize" in steps:
                target_unit = (gated or {}).get("target_unit")
                if target_unit and held.get("unit"):
                    segmented = None
                    if row["format"] in {"CSV", "TSV", "Excel"} and not row.get("original_reader"):
                        from .table_unit_segments import convert_holdout
                        segmented = convert_holdout(row, field, held, target_unit)
                    if segmented is not None:
                        converted, units = segmented
                        if not converted.size:
                            report.update(status="pending", basis="heldout_segmented_units_need_observations")
                            continue
                        report["observed_unit_segments"] = units
                    else:
                        array = xr.DataArray(values, dims=["sample"], attrs={"units": held["unit"]})
                        converted = UnitEngine().convert(array, target_unit)
                    if not np.isfinite(converted.values).all():
                        report.update(status="failed", basis="nonfinite_heldout_conversion")
                        continue
                    report["checks"].append("unit_conversion_executed_on_holdout")
                    if list(steps) == ["units.harmonize"]:
                        report["status"] = "checked"
            formats = {"format.netcdf": "NetCDF", "format.hdf5_netcdf": "HDF5-NetCDF",
                       "format.parquet": "Parquet", "format.excel": "Excel"}
            if steps and steps[-1] in formats and all(
                    step in {"units.harmonize", "format.normalize", steps[-1]} for step in steps):
                if "units.harmonize" not in steps or "unit_conversion_executed_on_holdout" in report["checks"]:
                    array = (converted if "units.harmonize" in steps else
                             xr.DataArray(values, dims=["sample"], attrs={"units": held.get("unit") or "1"}))
                    from ecodata.io.structures import export_structure, read_structure
                    encoded, filename, _ = export_structure(array.to_dataset(name="value"), formats[steps[-1]])
                    with read_structure(filename, encoded, record_read_source=False) as restored:
                        if not np.allclose(array.values, restored["value"].values, equal_nan=True):
                            report.update(status="failed", basis="heldout_format_roundtrip_mismatch")
                            continue
                    report["checks"].append("format_roundtrip_executed_on_holdout")
                    report["status"] = "checked"
            if "temporal.wide_table_interpolation" in steps and held.get("observation_groups"):
                metrics = interpolation_trial(held["observation_groups"])
                if metrics:
                    report["checks"].append("time_interpolation_compared_to_withheld_observations")
                    report["metrics"] = metrics
                    if len(steps) == 1:
                        report["status"] = "checked"
            if report["status"] == "pending":
                report["basis"] = "algorithm_or_independent_reference_not_verified"
            if held.get("worksheet_semantics") == "needs_declaration" and report["status"] != "failed":
                report.update(status="pending", basis="heldout_worksheet_field_semantics_need_declaration")
        except (EcoDataError, OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError, ImportError):
            report["basis"] = "local_holdout_reader_or_operator_unavailable"
    return reports


def interpolation_trial(observation_groups):
    """Hold out measured interior points and use the same reference for both methods."""
    from dataclasses import asdict
    import pandas as pd
    import xarray as xr
    from ecodata.evaluation.metrics import fidelity_metrics
    from .pair_processing import table_clock

    truth, linear, baseline = [], [], []
    for group in observation_groups:
        frame = pd.DataFrame(group, columns=["time", "value"])
        frame["time"] = table_clock(frame["time"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        observed = frame.dropna().set_index("time")["value"].sort_index()
        if len(observed) < 3 or observed.index.has_duplicates:
            continue
        selected = observed.index[1:-1:3]
        training = observed.copy()
        training.loc[selected] = float("nan")
        truth.extend(observed.loc[selected].tolist())
        linear.extend(training.interpolate(method="time", limit_area="inside").loc[selected].tolist())
        baseline.extend(training.ffill().loc[selected].tolist())
    if truth:
        expected = xr.DataArray(truth, dims=["heldout"])
        return dict(n=len(truth), reference="withheld_observations", baseline_method="previous_observation",
                    candidate=asdict(fidelity_metrics(xr.DataArray(linear, dims=["heldout"]), expected)),
                    baseline=asdict(fidelity_metrics(xr.DataArray(baseline, dims=["heldout"]), expected)))
    return None


def raster_trial(row, field, held, resolution, method, *, target_crs=None):
    from math import ceil
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.windows import Window

    ref = held["windows"][0]
    window = Window(ref["column"], ref["row"], ref["width"], ref["height"])
    dx, dy = (resolution, resolution) if isinstance(resolution, (float, int)) else resolution
    with local_member(row) as path, rasterio.open(path) as dataset:
        target_crs = target_crs or dataset.crs
        left, bottom, right, top = transform_bounds(dataset.crs, target_crs, *dataset.window_bounds(window))
        width, height = ceil((right - left) / dx - 1e-10), ceil((top - bottom) / dy - 1e-10)
        if not (0 < width * height <= PROBE_LIMIT and width > 0 and height > 0):
            raise ValueError("留出区域转换尺寸超出回检预算")
        values = dataset.read(field["band"], window=window, masked=True).astype(float).filled(np.nan)
        transform = from_origin(left, top, dx, dy)
        if method == "conservative":
            from rasterio.crs import CRS
            from .raster_reference import conservative

            semantics = field.get("spatial_semantics") or dataset.tags(field["band"]).get("spatial_semantics")
            if CRS.from_user_input(target_crs) != dataset.crs or semantics not in {"cell_mean", "flux"}:
                return None
            target = conservative(values, dataset.window_transform(window), (height, width), transform, dataset.crs)
            if target is None:
                return None
        else:
            target = np.full((height, width), np.nan)
            reproject(values, target, src_transform=dataset.window_transform(window), src_crs=dataset.crs,
                      dst_transform=transform, dst_crs=target_crs,
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=getattr(Resampling, method))
        if not np.isfinite(target).any():
            raise ValueError("留出空间块回检没有有效目标像元")
        return dict(shape=[height, width], resolution=[dx, dy], crs=str(target_crs),
                    valid_cells=int(np.isfinite(target).sum()), nodata_cells=int(np.isnan(target).sum()))
