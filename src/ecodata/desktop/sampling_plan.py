"""Disjoint recommendation and local review partitions for numeric evidence."""
import csv
import hashlib
import re
from contextlib import contextmanager
from io import TextIOWrapper
from itertools import islice, zip_longest

from .scan import AXES, local_member, open_tabular_member

SUPPORTED = {"NetCDF", "GeoTIFF", "CSV", "TSV", "Excel"}
IDENTIFIERS = {"station_id", "station", "site_id", "site", "sensor_id", "object_id", "id", "站点", "站点编号"}
TIME = {"time", "tstep", "date", "datetime", "timestamp", "time_utc", "时间", "日期"}
PROBE_LIMIT = 4096
VALUE_LIMIT = 64


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


def sample(row, field, *, holdout=False, spatial=False):
    if row["format"] not in SUPPORTED or row.get("original_reader") or not member_available(row):
        return dict(status="pending", basis="bounded_reader_unavailable", values=[])
    name = normalized(field.get("source_column") or field["name"])
    if field.get("axis") or name in AXES or name in IDENTIFIERS or name in TIME:
        return dict(status="pending", basis="non_observation_field", values=[])
    if row["format"] == "NetCDF":
        record = netcdf(row, field, holdout, spatial)
    elif row["format"] == "GeoTIFF":
        record = raster(row, field, holdout)
    else:
        record = table(row, field, holdout)
    record["status"] = "checked" if any(v is not None for v in record["values"]) else "pending"
    return record


def netcdf(row, field, holdout, spatial=False):
    import numpy as np
    import xarray as xr

    with local_member(row) as path, xr.open_dataset(path, decode_times=False, cache=False) as dataset:
        array = dataset[field["name"]]
        if not array.ndim or not all(array.shape):
            return dict(basis="no_disjoint_partition", values=[])
        dims = tuple(array.dims)
        temporal = next((dim for dim in dims
                         if dim.casefold() in TIME or
                         (dim in dataset.coords and dataset[dim].attrs.get("axis") == "T")), None)
        spatial_axis = next((dim for dim in dims if dim != temporal and array.sizes[dim] > 1), None)
        time_partition = temporal if temporal and array.sizes[temporal] > 1 else None
        partition = (spatial_axis if spatial else time_partition) or spatial_axis or time_partition
        if partition is None or array.sizes[partition] < 2:
            return dict(basis="no_disjoint_partition", values=[])
        size = array.sizes[partition]
        reserved = max(1, size // 5)
        boundary = size - reserved
        spacing = row.get("declared_time_resolution") or row.get("resolution", {}).get("time_seconds")
        tile_axes = [dim for dim in dims if dim not in {partition, temporal} and array.sizes[dim] >= 4][:2]
        width = min(16 // 2**len(tile_axes), reserved if holdout else max(1, boundary // 4))
        start = boundary if holdout else 0
        last = size - width if holdout else boundary - width
        windows = 1 if holdout else min(4, last + 1)
        regions, groups = [], []
        for index in range(windows):
            portion = index / max(1, windows - 1)
            lower = start + int((last - start) * portion)
            selection = {}
            for dim in dims:
                if dim == partition:
                    continue
                size_dim = array.sizes[dim]
                limit = size_dim - max(1, size_dim // 5) if size_dim > 1 and dim == spatial_axis else size_dim
                tile = 2 if dim in tile_axes else 1
                position = int((limit - tile) * (0.5 if holdout else portion))
                selection[dim] = slice(position, position + tile) if tile > 1 else position
            selection[partition] = slice(lower, lower + width)
            groups.append(array.isel(selection).values.reshape(-1))
            region = {"axis": partition, "start_index": lower, "stop_index": lower + width}
            if temporal == partition and temporal in dataset.coords:
                coordinate = dataset[temporal]
                coordinate_values = coordinate.isel({temporal: slice(lower, lower + width)}).values
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
                    edges = dataset[bounds].isel({temporal: slice(lower, lower + width)}).transpose(temporal, ...).values
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
                    variable=field["name"], windows=regions, values=values)


def raster(row, field, holdout):
    import rasterio
    from rasterio.windows import Window

    with local_member(row) as path, rasterio.open(path) as dataset:
        along_y = dataset.height >= 2
        size = dataset.height if along_y else dataset.width
        if size < 2:
            return dict(basis="no_disjoint_partition", values=[])
        boundary = size - max(1, size // 5)
        choice = [1] if holdout else [0, 0.5, 1]
        windows, groups = [], []
        for portion in choice:
            start, stop = (boundary, size) if holdout else (0, boundary)
            width = min(4, dataset.width if along_y else stop - start)
            height = min(4, stop - start if along_y else dataset.height)
            x = int((dataset.width - width) * portion) if along_y else start + int((stop - start - width) * portion)
            y = start + int((stop - start - height) * portion) if along_y else int((dataset.height - height) * portion)
            window = Window(x, y, width, height)
            groups.append(dataset.read(field["band"], window=window, masked=True).ravel().tolist())
            windows.append(dict(row=y, column=x, width=width, height=height))
        return dict(basis="spatial_blocks", partition=dict(axis="row" if along_y else "column",
                    holdout_start=boundary, size=size), windows=windows, crs=str(dataset.crs) if dataset.crs else None,
                    resolution=list(dataset.res), nodata=dataset.nodata, unit=field.get("unit"),
                    values=numerics(v for group in zip_longest(*groups) for v in group))


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


def table(row, field, holdout):
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
                time = next((i for i, name in enumerate(names) if name in TIME), None)
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
                                return dict(basis="heldout_sheet_unit_conflict", values=[])
                            cells = (record[index] for record in islice(other, VALUE_LIMIT) if len(record) > index)
                            return dict(basis="worksheet_groups", partition=dict(held_sheet=reserved),
                                        unit=peer.get("unit"), values=numerics(cells))
                    if sheet == reserved and holdout:
                        cells = (record[col] for record in islice(rows, VALUE_LIMIT) if len(record) > col)
                        return dict(basis="worksheet_groups", partition=dict(held_sheet=reserved),
                                    unit=field.get("unit"), values=numerics(cells))
                groups = {}
                stamps = {}
                for index, record in enumerate(islice(rows, PROBE_LIMIT)):
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
                            match = re.match(r"^\d{4}-\d{2}", group)
                            if match is None:
                                continue
                            group = match[0]
                        if not group:
                            continue
                        reserved = hashlib.sha256(group.encode("utf-8")).digest()[0] % 4 == 0
                    if reserved == holdout:
                        bucket = groups.setdefault(group, [])
                        if len(bucket) < VALUE_LIMIT:
                            bucket.append(record[col])
                            if time is not None and time < len(record):
                                stamps.setdefault(group, []).append((record[time], record[col]))
                values = numerics(islice((value for batch in zip_longest(*groups.values())
                                          for value in batch if value is not None), VALUE_LIMIT))
                return dict(basis=basis, unit=field.get("unit"), sheet=sheet,
                            groups=len(groups), values=values, observation_groups=list(stamps.values()) if holdout else [])
        except UnicodeDecodeError:
            if encoding == "gb18030":
                raise


def export_field(row, field):
    from .sample_values import export_sample, field_sample

    if row["format"] not in SUPPORTED or row.get("original_reader") or field.get("noaa_component"):
        return export_sample(field_sample(row, field, outbound=True))
    record = sample(row, field)
    payload = export_sample(record["values"])
    windows = [dict(axis=window.get("axis"), points=window["stop_index"] - window["start_index"],
                    time={key: value for key, value in window.get("time", {}).items()
                          if key in {"ordered", "intervals_valid", "interval_range", "interval_widths", "calendar"}})
               for window in record.get("windows", []) if "stop_index" in window]
    payload["sampling"] = dict(basis=record["basis"], groups=record.get("groups"),
                               windows=windows, unit=record.get("unit"), frequency=record.get("frequency"))
    return payload


def recommendation_review(scan, recommendation, gated, fields, cache=None):
    """Run only local holdout checks backed by the selected operation."""
    import numpy as np
    import xarray as xr
    from ecodata.core.exceptions import EcoDataError
    from ecodata.engines.units import UnitEngine

    steps = recommendation.get("steps", ())
    spatial = any(step.startswith(("spatial.", "raster_task.spatial")) for step in steps)
    cache = {} if cache is None else cache
    reports = []
    for index, fid, field in fields:
        report = dict(field=fid, status="pending", checks=[])
        reports.append(report)
        try:
            key = (index, fid, spatial)
            if key not in cache:
                cache[key] = sample(scan["rows"][index], field, holdout=True, spatial=spatial)
            held = cache[key]
            values = [v for v in held["values"] if v is not None]
            report.update(partition=held.get("partition"), basis=held["basis"], count=len(values))
            if held["basis"] in {"heldout_sheet_unit_conflict", "heldout_schema_mismatch"}:
                report["status"] = "failed"
            if not values:
                continue
            times = [window["time"] for window in held.get("windows", ()) if "time" in window]
            if any(t.get("ordered") is False or t.get("intervals_valid") is False for t in times):
                report.update(status="failed", basis="invalid_heldout_time_axis")
                continue
            if held["basis"] == "spatial_blocks" and "resolution" in held:
                if not held["crs"] or any(not np.isfinite(v) or v <= 0 for v in held["resolution"]):
                    report.update(status="failed", basis="invalid_heldout_grid")
                    continue
                report["checks"].append("raster_crs_resolution_and_valid_values")
                method = next((step.rsplit(".", 1)[-1] for step in steps
                               if step in {"raster_task.spatial.bilinear", "raster_task.spatial.nearest"}), None)
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
                    report["checks"].append("raster_reproject_executed_on_holdout")
            if "format.normalize" in steps or any(step.startswith("format.") for step in steps):
                report["checks"].append("finite_values_in_heldout_source")
            if "units.harmonize" in steps:
                target_unit = (gated or {}).get("target_unit")
                if target_unit and held.get("unit"):
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
        target = np.full((height, width), np.nan)
        reproject(values, target, src_transform=dataset.window_transform(window), src_crs=dataset.crs,
                  dst_transform=from_origin(left, top, dx, dy), dst_crs=target_crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=getattr(Resampling, method))
        if not np.isfinite(target).any():
            raise ValueError("留出空间块回检没有有效目标像元")
        return dict(shape=[height, width], resolution=[dx, dy], crs=str(target_crs),
                    valid_cells=int(np.isfinite(target).sum()), nodata_cells=int(np.isnan(target).sum()))
